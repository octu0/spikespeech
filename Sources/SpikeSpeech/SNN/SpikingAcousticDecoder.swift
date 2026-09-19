import Foundation

/// 多層 SNN 音響デコーダー
///
/// レート符号化スパイク列に伴う時間分解能の低下を防ぎ、言語特徴量を直流電流として直接膜電位へ伝達する。
/// 4 または 6 のブロック構造により、層 0 の再帰結合ダイナミクスと上位ブロックのフレーム間時間畳み込み
/// （Temporal 1D Depthwise Convolution）および RMSNorm 残差加算によって、
/// Mel スペクトログラムの横縞・平坦化を解消し、自然な音響遷移（フォルマント軌跡）を生成する。
public final class SpikingAcousticDecoder: @unchecked Sendable {
    public let weights: SpikingNetworkWeights
    public let config: LIFConfig

    /// 転置再帰重み
    public let wRecT: [Float]

    /// 上位層の転置結合重み
    public let wLayersT: [[Float]]

    /// 転置出力射影重み
    public let wOutT: [Float]

    public init(weights: SpikingNetworkWeights) {
        self.weights = weights
        self.config = weights.lifConfig
        self.wRecT = weights.makeWRecT()
        self.wLayersT = weights.makeWLayersT()
        self.wOutT = weights.makeWOutT()
    }

    /// 1 フレームの潜在特徴量ベクトルから音響特徴量を生成する。
    /// 単一フレーム時は時間畳み込みの境界条件（t-1 = t = t+1）として同一の数理ダイナミクスで計算する。
    @inline(__always)
    public func decodeFrame(
        features: UnsafePointer<Float>,
        workspace: AcousticWorkspace,
        outputFeatures: UnsafeMutablePointer<Float>
    ) {
        let inDim = weights.inputDim
        var singleFeat = [Float](repeating: 0.0, count: inDim)
        singleFeat.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: features, count: inDim)
        }

        let seqResult = decodeSequence(featuresSeq: [singleFeat], workspace: workspace)
        if seqResult.isEmpty != true {
            let outDim = weights.outputDim
            seqResult[0].withUnsafeBufferPointer { src in
                outputFeatures.update(from: src.baseAddress!, count: outDim)
            }
        }
    }

    /// フレーム系列全体をデコードして音響特徴量系列を生成する。
    /// 各ブロック（層 0 再帰 LIF、層 1 以降のフレーム間時間畳み込み＋RMSNorm＋LIF）を順次実行し、
    /// Mel スペクトログラムの平坦化・横縞を解消した自然な音響特徴量を再構成する。
    @discardableResult
    public func decodeSequence(
        featuresSeq: [[Float]],
        workspace: AcousticWorkspace
    ) -> [[Float]] {
        let totalFrames = featuresSeq.count
        if totalFrames <= 0 {
            return []
        }

        let inDim = weights.inputDim
        let hMax = weights.maxHiddenDim
        let tSteps = weights.timeSteps
        let numLayers = weights.numLayers
        let outDim = weights.outputDim
        let hLimit = hMax - (hMax % 8)

        workspace.reset()

        // ------------------------------------------------------------
        // ブロック 0: 層 0 再帰 LIF（直流入力電流＋再帰スパイク結合）
        // ------------------------------------------------------------
        var layer0Out = [[Float]](repeating: [Float](repeating: 0.0, count: hMax), count: totalFrames)
        let layer0 = workspace.layerStates[0]
        let safeTimeSteps = max(1, tSteps)
        let invT = 1.0 / Float(safeTimeSteps)

        weights.wIn.withUnsafeBufferPointer { wInBuf in
            weights.bH.withUnsafeBufferPointer { bhBuf in
                self.wRecT.withUnsafeBufferPointer { recBuf in
                    let wIn = wInBuf.baseAddress!
                    let bh = bhBuf.baseAddress!
                    let recT = recBuf.baseAddress!

                    var t = 0
                    while t < totalFrames {
                        // 1. 直流入力電流の事前計算
                        featuresSeq[t].withUnsafeBufferPointer { fBuf in
                            workspace.inputCurrents.withUnsafeMutableBufferPointer { curBuf in
                                let fPtr = fBuf.baseAddress!
                                let cur = curBuf.baseAddress!

                                var n = 0
                                while n < hLimit {
                                    var acc = SIMD8<Float>(
                                        bh[n + 0], bh[n + 1], bh[n + 2], bh[n + 3],
                                        bh[n + 4], bh[n + 5], bh[n + 6], bh[n + 7]
                                    )
                                    let r0 = (n + 0) * inDim
                                    let r1 = (n + 1) * inDim
                                    let r2 = (n + 2) * inDim
                                    let r3 = (n + 3) * inDim
                                    let r4 = (n + 4) * inDim
                                    let r5 = (n + 5) * inDim
                                    let r6 = (n + 6) * inDim
                                    let r7 = (n + 7) * inDim

                                    var d = 0
                                    while d < inDim {
                                        let featVal = SIMD8<Float>(repeating: fPtr[d])
                                        let w = SIMD8<Float>(
                                            wIn[r0 + d], wIn[r1 + d], wIn[r2 + d], wIn[r3 + d],
                                            wIn[r4 + d], wIn[r5 + d], wIn[r6 + d], wIn[r7 + d]
                                        )
                                        acc = acc + (w * featVal)
                                        d += 1
                                    }

                                    cur[n + 0] = acc[0]
                                    cur[n + 1] = acc[1]
                                    cur[n + 2] = acc[2]
                                    cur[n + 3] = acc[3]
                                    cur[n + 4] = acc[4]
                                    cur[n + 5] = acc[5]
                                    cur[n + 6] = acc[6]
                                    cur[n + 7] = acc[7]
                                    n += 8
                                }

                                while n < hMax {
                                    var curr = bh[n]
                                    let inOffset = n * inDim
                                    var d = 0
                                    while d < inDim {
                                        curr += wIn[inOffset + d] * fPtr[d]
                                        d += 1
                                    }
                                    cur[n] = curr
                                    n += 1
                                }
                            }
                        }

                        workspace.clearReadoutSums()

                        // 2. 内部時間ステップループ（再帰 LIF）
                        var step = 0
                        while step < tSteps {
                            var activeCount0 = 0
                            layer0.s.withUnsafeBufferPointer { pS in
                                var j = 0
                                while j < hMax {
                                    if 0.0 < pS[j] {
                                        workspace.activeSpikes[activeCount0] = j
                                        activeCount0 += 1
                                    }
                                    j += 1
                                }
                            }

                            workspace.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                                let stepCur = stepBuf.baseAddress!
                                workspace.inputCurrents.withUnsafeBufferPointer { inBuf in
                                    stepCur.update(from: inBuf.baseAddress!, count: hMax)
                                }

                                var a = 0
                                while a < activeCount0 {
                                    let rowOffset = workspace.activeSpikes[a] * hMax
                                    var n = 0
                                    while n < hLimit {
                                        let acc = SIMD8<Float>(
                                            stepCur[n + 0], stepCur[n + 1], stepCur[n + 2], stepCur[n + 3],
                                            stepCur[n + 4], stepCur[n + 5], stepCur[n + 6], stepCur[n + 7]
                                        )
                                        let w = SIMD8<Float>(
                                            recT[rowOffset + n + 0], recT[rowOffset + n + 1],
                                            recT[rowOffset + n + 2], recT[rowOffset + n + 3],
                                            recT[rowOffset + n + 4], recT[rowOffset + n + 5],
                                            recT[rowOffset + n + 6], recT[rowOffset + n + 7]
                                        )
                                        let sum = acc + w
                                        stepCur[n + 0] = sum[0]
                                        stepCur[n + 1] = sum[1]
                                        stepCur[n + 2] = sum[2]
                                        stepCur[n + 3] = sum[3]
                                        stepCur[n + 4] = sum[4]
                                        stepCur[n + 5] = sum[5]
                                        stepCur[n + 6] = sum[6]
                                        stepCur[n + 7] = sum[7]
                                        n += 8
                                    }
                                    while n < hMax {
                                        stepCur[n] += recT[rowOffset + n]
                                        n += 1
                                    }
                                    a += 1
                                }
                            }

                            layer0.v.withUnsafeMutableBufferPointer { vBuf in
                                layer0.s.withUnsafeMutableBufferPointer { sBuf in
                                    layer0.a.withUnsafeMutableBufferPointer { aBuf in
                                        workspace.stepCurrents.withUnsafeBufferPointer { curBuf in
                                            workspace.readoutSums.withUnsafeMutableBufferPointer { sumBuf in
                                                LIFNeuronEngine.stepHardResetReadoutAdaptiveSIMD8(
                                                    config: self.config,
                                                    vPtr: vBuf.baseAddress!,
                                                    sPtr: sBuf.baseAddress!,
                                                    aPtr: aBuf.baseAddress!,
                                                    curPtr: curBuf.baseAddress!,
                                                    readoutSumPtr: sumBuf.baseAddress!,
                                                    count: hMax
                                                )
                                            }
                                        }
                                    }
                                }
                            }

                            step += 1
                        }

                        // 各フレームの膜電位積算（Readout / tSteps）をブロック 0 出力とする
                        layer0Out[t].withUnsafeMutableBufferPointer { dstBuf in
                            workspace.readoutSums.withUnsafeBufferPointer { sumBuf in
                                let dst = dstBuf.baseAddress!
                                let sums = sumBuf.baseAddress!
                                var k = 0
                                while k < hLimit {
                                    let sVec = SIMD8<Float>(
                                        sums[k + 0], sums[k + 1], sums[k + 2], sums[k + 3],
                                        sums[k + 4], sums[k + 5], sums[k + 6], sums[k + 7]
                                    )
                                    let rVec = sVec * invT
                                    dst[k + 0] = rVec[0]
                                    dst[k + 1] = rVec[1]
                                    dst[k + 2] = rVec[2]
                                    dst[k + 3] = rVec[3]
                                    dst[k + 4] = rVec[4]
                                    dst[k + 5] = rVec[5]
                                    dst[k + 6] = rVec[6]
                                    dst[k + 7] = rVec[7]
                                    k += 8
                                }
                                while k < hMax {
                                    dst[k] = sums[k] * invT
                                    k += 1
                                }
                            }
                        }

                        t += 1
                    }
                }
            }
        }

        // ------------------------------------------------------------
        // 上位ブロック 1 ..< numLayers
        // フレーム間時間畳み込み（1D Depthwise Conv）＋ チャネル結合 ＋ RMSNorm ＋ 残差加算 ＋ LIF
        // ------------------------------------------------------------
        var prevBlockOut = layer0Out
        var layerIdx = 1
        while layerIdx < numLayers {
            let upperIdx = layerIdx - 1
            let curLayer = workspace.layerStates[layerIdx]
            var curBlockOut = [[Float]](repeating: [Float](repeating: 0.0, count: hMax), count: totalFrames)

            let wConvData: [Float]
            if upperIdx < weights.wConv.count {
                wConvData = weights.wConv[upperIdx]
            } else {
                wConvData = [Float](repeating: 0.0, count: 5 * hMax)
            }
            let bData = weights.bHLayers[upperIdx]
            let wLayerTData = wLayersT[upperIdx]
            let gammaData = weights.gammaRMS[upperIdx]
            let d = 1 << upperIdx

            wConvData.withUnsafeBufferPointer { convBuf in
                bData.withUnsafeBufferPointer { bBuf in
                    wLayerTData.withUnsafeBufferPointer { wBuf in
                        gammaData.withUnsafeBufferPointer { gammaBuf in
                            let pConv = convBuf.baseAddress!
                            let pB = bBuf.baseAddress!
                            let pWT = wBuf.baseAddress!
                            let pGamma = gammaBuf.baseAddress!

                            var t = 0
                            while t < totalFrames {
                                let t0 = max(0, t - (2 * d))
                                let t1 = max(0, t - d)
                                let t2 = t
                                let t3 = min(totalFrames - 1, t + d)
                                let t4 = min(totalFrames - 1, t + (2 * d))

                                // 1. 時間畳み込み (Temporal 1D Depthwise Conv, K=5, Dilated) ＋ 残差加算
                                prevBlockOut[t0].withUnsafeBufferPointer { p0Buf in
                                    prevBlockOut[t1].withUnsafeBufferPointer { p1Buf in
                                        prevBlockOut[t2].withUnsafeBufferPointer { p2Buf in
                                            prevBlockOut[t3].withUnsafeBufferPointer { p3Buf in
                                                prevBlockOut[t4].withUnsafeBufferPointer { p4Buf in
                                                    workspace.convCurrents.withUnsafeMutableBufferPointer { mBuf in
                                                        let p0 = p0Buf.baseAddress!
                                                        let p1 = p1Buf.baseAddress!
                                                        let p2 = p2Buf.baseAddress!
                                                        let p3 = p3Buf.baseAddress!
                                                        let p4 = p4Buf.baseAddress!
                                                        let pM = mBuf.baseAddress!

                                                        var c = 0
                                                        while c < hLimit {
                                                            let w0 = SIMD8<Float>(
                                                                pConv[(0 * hMax) + c + 0], pConv[(0 * hMax) + c + 1],
                                                                pConv[(0 * hMax) + c + 2], pConv[(0 * hMax) + c + 3],
                                                                pConv[(0 * hMax) + c + 4], pConv[(0 * hMax) + c + 5],
                                                                pConv[(0 * hMax) + c + 6], pConv[(0 * hMax) + c + 7]
                                                            )
                                                            let w1 = SIMD8<Float>(
                                                                pConv[(1 * hMax) + c + 0], pConv[(1 * hMax) + c + 1],
                                                                pConv[(1 * hMax) + c + 2], pConv[(1 * hMax) + c + 3],
                                                                pConv[(1 * hMax) + c + 4], pConv[(1 * hMax) + c + 5],
                                                                pConv[(1 * hMax) + c + 6], pConv[(1 * hMax) + c + 7]
                                                            )
                                                            let w2 = SIMD8<Float>(
                                                                pConv[(2 * hMax) + c + 0], pConv[(2 * hMax) + c + 1],
                                                                pConv[(2 * hMax) + c + 2], pConv[(2 * hMax) + c + 3],
                                                                pConv[(2 * hMax) + c + 4], pConv[(2 * hMax) + c + 5],
                                                                pConv[(2 * hMax) + c + 6], pConv[(2 * hMax) + c + 7]
                                                            )
                                                            let w3 = SIMD8<Float>(
                                                                pConv[(3 * hMax) + c + 0], pConv[(3 * hMax) + c + 1],
                                                                pConv[(3 * hMax) + c + 2], pConv[(3 * hMax) + c + 3],
                                                                pConv[(3 * hMax) + c + 4], pConv[(3 * hMax) + c + 5],
                                                                pConv[(3 * hMax) + c + 6], pConv[(3 * hMax) + c + 7]
                                                            )
                                                            let w4 = SIMD8<Float>(
                                                                pConv[(4 * hMax) + c + 0], pConv[(4 * hMax) + c + 1],
                                                                pConv[(4 * hMax) + c + 2], pConv[(4 * hMax) + c + 3],
                                                                pConv[(4 * hMax) + c + 4], pConv[(4 * hMax) + c + 5],
                                                                pConv[(4 * hMax) + c + 6], pConv[(4 * hMax) + c + 7]
                                                            )

                                                            let v0 = SIMD8<Float>(
                                                                p0[c + 0], p0[c + 1], p0[c + 2], p0[c + 3],
                                                                p0[c + 4], p0[c + 5], p0[c + 6], p0[c + 7]
                                                            )
                                                            let v1 = SIMD8<Float>(
                                                                p1[c + 0], p1[c + 1], p1[c + 2], p1[c + 3],
                                                                p1[c + 4], p1[c + 5], p1[c + 6], p1[c + 7]
                                                            )
                                                            let v2 = SIMD8<Float>(
                                                                p2[c + 0], p2[c + 1], p2[c + 2], p2[c + 3],
                                                                p2[c + 4], p2[c + 5], p2[c + 6], p2[c + 7]
                                                            )
                                                            let v3 = SIMD8<Float>(
                                                                p3[c + 0], p3[c + 1], p3[c + 2], p3[c + 3],
                                                                p3[c + 4], p3[c + 5], p3[c + 6], p3[c + 7]
                                                            )
                                                            let v4 = SIMD8<Float>(
                                                                p4[c + 0], p4[c + 1], p4[c + 2], p4[c + 3],
                                                                p4[c + 4], p4[c + 5], p4[c + 6], p4[c + 7]
                                                            )

                                                            let z = (((v0 * w0) + (v1 * w1)) + (v2 * w2)) + ((v3 * w3) + (v4 * w4))
                                                            let m = z + v2
                                                            pM[c + 0] = m[0]
                                                            pM[c + 1] = m[1]
                                                            pM[c + 2] = m[2]
                                                            pM[c + 3] = m[3]
                                                            pM[c + 4] = m[4]
                                                            pM[c + 5] = m[5]
                                                            pM[c + 6] = m[6]
                                                            pM[c + 7] = m[7]
                                                            c += 8
                                                        }

                                                        while c < hMax {
                                                            let w0 = pConv[(0 * hMax) + c]
                                                            let w1 = pConv[(1 * hMax) + c]
                                                            let w2 = pConv[(2 * hMax) + c]
                                                            let w3 = pConv[(3 * hMax) + c]
                                                            let w4 = pConv[(4 * hMax) + c]
                                                            let z = (((p0[c] * w0) + (p1[c] * w1)) + (p2[c] * w2)) + ((p3[c] * w3) + (p4[c] * w4))
                                                            pM[c] = z + p2[c]
                                                            c += 1
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // 2. チャネル間結合 (Pointwise Dense) ＋ RMSNorm ＋ 残差加算
                                workspace.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                                    workspace.convCurrents.withUnsafeBufferPointer { mBuf in
                                        let stepCur = stepBuf.baseAddress!
                                        let pM = mBuf.baseAddress!

                                        stepCur.update(from: pB, count: hMax)

                                        var j = 0
                                        while j < hMax {
                                            let act = pM[j]
                                            if act != 0.0 {
                                                let rowOffset = j * hMax
                                                let actVec = SIMD8<Float>(repeating: act)
                                                var n = 0
                                                while n < hLimit {
                                                    let acc = SIMD8<Float>(
                                                        stepCur[n + 0], stepCur[n + 1], stepCur[n + 2], stepCur[n + 3],
                                                        stepCur[n + 4], stepCur[n + 5], stepCur[n + 6], stepCur[n + 7]
                                                    )
                                                    let w = SIMD8<Float>(
                                                        pWT[rowOffset + n + 0], pWT[rowOffset + n + 1],
                                                        pWT[rowOffset + n + 2], pWT[rowOffset + n + 3],
                                                        pWT[rowOffset + n + 4], pWT[rowOffset + n + 5],
                                                        pWT[rowOffset + n + 6], pWT[rowOffset + n + 7]
                                                    )
                                                    let sum = acc + (w * actVec)
                                                    stepCur[n + 0] = sum[0]
                                                    stepCur[n + 1] = sum[1]
                                                    stepCur[n + 2] = sum[2]
                                                    stepCur[n + 3] = sum[3]
                                                    stepCur[n + 4] = sum[4]
                                                    stepCur[n + 5] = sum[5]
                                                    stepCur[n + 6] = sum[6]
                                                    stepCur[n + 7] = sum[7]
                                                    n += 8
                                                }
                                                while n < hMax {
                                                    stepCur[n] += pWT[rowOffset + n] * act
                                                    n += 1
                                                }
                                            }
                                            j += 1
                                        }

                                        // RMSNorm
                                        var sumSqVec = SIMD8<Float>(repeating: 0.0)
                                        var n = 0
                                        while n < hLimit {
                                            let v = SIMD8<Float>(
                                                stepCur[n + 0], stepCur[n + 1], stepCur[n + 2], stepCur[n + 3],
                                                stepCur[n + 4], stepCur[n + 5], stepCur[n + 6], stepCur[n + 7]
                                            )
                                            sumSqVec = sumSqVec + (v * v)
                                            n += 8
                                        }
                                        var totalSq = (sumSqVec[0] + sumSqVec[1] + sumSqVec[2] + sumSqVec[3]) +
                                                      (sumSqVec[4] + sumSqVec[5] + sumSqVec[6] + sumSqVec[7])
                                        while n < hMax {
                                            totalSq += stepCur[n] * stepCur[n]
                                            n += 1
                                        }

                                        let meanSq = totalSq / Float(hMax)
                                        let rms = sqrt(meanSq + 1e-5)
                                        let invRms = 1.0 / rms
                                        let invRmsVec = SIMD8<Float>(repeating: invRms)

                                        n = 0
                                        while n < hLimit {
                                            let raw = SIMD8<Float>(
                                                stepCur[n + 0], stepCur[n + 1], stepCur[n + 2], stepCur[n + 3],
                                                stepCur[n + 4], stepCur[n + 5], stepCur[n + 6], stepCur[n + 7]
                                            )
                                            let g = SIMD8<Float>(
                                                pGamma[n + 0], pGamma[n + 1], pGamma[n + 2], pGamma[n + 3],
                                                pGamma[n + 4], pGamma[n + 5], pGamma[n + 6], pGamma[n + 7]
                                            )
                                            let p = SIMD8<Float>(
                                                pM[n + 0], pM[n + 1], pM[n + 2], pM[n + 3],
                                                pM[n + 4], pM[n + 5], pM[n + 6], pM[n + 7]
                                            )
                                            let norm = (raw * invRmsVec) * g
                                            let totalCur = norm + p
                                            stepCur[n + 0] = totalCur[0]
                                            stepCur[n + 1] = totalCur[1]
                                            stepCur[n + 2] = totalCur[2]
                                            stepCur[n + 3] = totalCur[3]
                                            stepCur[n + 4] = totalCur[4]
                                            stepCur[n + 5] = totalCur[5]
                                            stepCur[n + 6] = totalCur[6]
                                            stepCur[n + 7] = totalCur[7]
                                            n += 8
                                        }
                                        while n < hMax {
                                            let norm = (stepCur[n] * invRms) * pGamma[n]
                                            stepCur[n] = norm + pM[n]
                                            n += 1
                                        }
                                    }
                                }

                                workspace.clearReadoutSums()
                                curLayer.reset()

                                // 3. 内部時間ステップループ（LIF）
                                var step = 0
                                while step < tSteps {
                                    curLayer.v.withUnsafeMutableBufferPointer { vBuf in
                                        curLayer.s.withUnsafeMutableBufferPointer { sBuf in
                                            curLayer.a.withUnsafeMutableBufferPointer { aBuf in
                                                workspace.stepCurrents.withUnsafeBufferPointer { curBuf in
                                                    workspace.readoutSums.withUnsafeMutableBufferPointer { sumBuf in
                                                    let isLast = (layerIdx + 1) == numLayers
                                                    switch isLast {
                                                    case true:
                                                        LIFNeuronEngine.stepReadoutAdaptiveSIMD8(
                                                            config: self.config,
                                                            vPtr: vBuf.baseAddress!,
                                                            sPtr: sBuf.baseAddress!,
                                                            aPtr: aBuf.baseAddress!,
                                                            curPtr: curBuf.baseAddress!,
                                                            readoutSumPtr: sumBuf.baseAddress!,
                                                            count: hMax
                                                        )
                                                    case false:
                                                        LIFNeuronEngine.stepHardResetReadoutAdaptiveSIMD8(
                                                            config: self.config,
                                                            vPtr: vBuf.baseAddress!,
                                                            sPtr: sBuf.baseAddress!,
                                                            aPtr: aBuf.baseAddress!,
                                                            curPtr: curBuf.baseAddress!,
                                                            readoutSumPtr: sumBuf.baseAddress!,
                                                            count: hMax
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                step += 1
                            }

                                curBlockOut[t].withUnsafeMutableBufferPointer { dstBuf in
                                    workspace.readoutSums.withUnsafeBufferPointer { sumBuf in
                                        let dst = dstBuf.baseAddress!
                                        let sums = sumBuf.baseAddress!
                                        var k = 0
                                        while k < hLimit {
                                            let sVec = SIMD8<Float>(
                                                sums[k + 0], sums[k + 1], sums[k + 2], sums[k + 3],
                                                sums[k + 4], sums[k + 5], sums[k + 6], sums[k + 7]
                                            )
                                            let rVec = sVec * invT
                                            dst[k + 0] = rVec[0]
                                            dst[k + 1] = rVec[1]
                                            dst[k + 2] = rVec[2]
                                            dst[k + 3] = rVec[3]
                                            dst[k + 4] = rVec[4]
                                            dst[k + 5] = rVec[5]
                                            dst[k + 6] = rVec[6]
                                            dst[k + 7] = rVec[7]
                                            k += 8
                                        }
                                        while k < hMax {
                                            dst[k] = sums[k] * invT
                                            k += 1
                                        }
                                    }
                                }

                                t += 1
                            }
                        }
                    }
                }
            }

            prevBlockOut = curBlockOut
            layerIdx += 1
        }

        // ------------------------------------------------------------
        // 最終線形射影: 最上位ブロック出力系列 -> Mel スペクトル
        // ------------------------------------------------------------
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: outDim), count: totalFrames)
        let outLimit = outDim - (outDim % 8)

        weights.bOut.withUnsafeBufferPointer { bBuf in
            self.wOutT.withUnsafeBufferPointer { outBuf in
                let pBOut = bBuf.baseAddress!
                let outT = outBuf.baseAddress!

                var t = 0
                while t < totalFrames {
                    result[t].withUnsafeMutableBufferPointer { outFeaturesBuf in
                        let outputFeatures = outFeaturesBuf.baseAddress!
                        outputFeatures.update(from: pBOut, count: outDim)

                        prevBlockOut[t].withUnsafeBufferPointer { rBuf in
                            let rPtr = rBuf.baseAddress!
                            var k = 0
                            while k < hMax {
                                let rate = rPtr[k]
                                if rate != 0.0 {
                                    let rowOffset = k * outDim
                                    let rateVec = SIMD8<Float>(repeating: rate)
                                    var c = 0
                                    while c < outLimit {
                                        let acc = SIMD8<Float>(
                                            outputFeatures[c + 0], outputFeatures[c + 1], outputFeatures[c + 2], outputFeatures[c + 3],
                                            outputFeatures[c + 4], outputFeatures[c + 5], outputFeatures[c + 6], outputFeatures[c + 7]
                                        )
                                        let w = SIMD8<Float>(
                                            outT[rowOffset + c + 0], outT[rowOffset + c + 1],
                                            outT[rowOffset + c + 2], outT[rowOffset + c + 3],
                                            outT[rowOffset + c + 4], outT[rowOffset + c + 5],
                                            outT[rowOffset + c + 6], outT[rowOffset + c + 7]
                                        )
                                        let sum = acc + (w * rateVec)
                                        outputFeatures[c + 0] = sum[0]
                                        outputFeatures[c + 1] = sum[1]
                                        outputFeatures[c + 2] = sum[2]
                                        outputFeatures[c + 3] = sum[3]
                                        outputFeatures[c + 4] = sum[4]
                                        outputFeatures[c + 5] = sum[5]
                                        outputFeatures[c + 6] = sum[6]
                                        outputFeatures[c + 7] = sum[7]
                                        c += 8
                                    }
                                    while c < outDim {
                                        outputFeatures[c] += outT[rowOffset + c] * rate
                                        c += 1
                                    }
                                }
                                k += 1
                            }
                        }
                    }
                    t += 1
                }
            }
        }

        return result
    }
}
