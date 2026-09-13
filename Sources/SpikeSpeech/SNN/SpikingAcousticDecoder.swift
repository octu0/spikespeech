import Foundation

/// 多層 SNN 音響デコーダー
///
/// レート符号化スパイク列に伴う時間分解能の低下を防ぎ、言語特徴量を直流電流として直接膜電位へ伝達する。
/// また、発火ニューロンの転置結合重みのみを SIMD8 で並列加算することで推論レイテンシを極小化する。
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
    /// 直流入力電流は内部時間ステップの間一定であるため、時間ループの外側で事前計算してキャッシュする。
    @inline(__always)
    public func decodeFrame(
        features: UnsafePointer<Float>,
        workspace: AcousticWorkspace,
        outputFeatures: UnsafeMutablePointer<Float>
    ) {
        let inDim = weights.inputDim
        let hMax = weights.maxHiddenDim
        let tSteps = weights.timeSteps
        let numLayers = weights.numLayers
        let outDim = weights.outputDim

        // 1. 直流入力電流の事前計算
        let hLimit = hMax - (hMax % 8)
        weights.wIn.withUnsafeBufferPointer { wInBuf in
            weights.bH.withUnsafeBufferPointer { bhBuf in
                workspace.inputCurrents.withUnsafeMutableBufferPointer { curBuf in
                    let wIn = wInBuf.baseAddress!
                    let bh = bhBuf.baseAddress!
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
                            let featVal = SIMD8<Float>(repeating: features[d])
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
                            curr += wIn[inOffset + d] * features[d]
                            d += 1
                        }
                        cur[n] = curr
                        n += 1
                    }
                }
            }
        }

        // 2. リードアウト積算バッファのクリア
        workspace.clearReadoutSums()

        let layer0 = workspace.layerStates[0]

        // 3. 内部時間ステップループ
        var step = 0
        while step < tSteps {
            // 3.1 層 0: 直前ステップ発火ニューロンからの再帰結合電流を加算
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

                self.wRecT.withUnsafeBufferPointer { recBuf in
                    let recT = recBuf.baseAddress!
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
            }

            if 1 < numLayers {
                workspace.stepCurrentsPrev.withUnsafeMutableBufferPointer { prevBuf in
                    workspace.stepCurrents.withUnsafeBufferPointer { curBuf in
                        prevBuf.baseAddress!.update(from: curBuf.baseAddress!, count: hMax)
                    }
                }
            }

            layer0.v.withUnsafeMutableBufferPointer { vBuf in
                layer0.s.withUnsafeMutableBufferPointer { sBuf in
                    layer0.a.withUnsafeMutableBufferPointer { aBuf in
                        workspace.stepCurrents.withUnsafeBufferPointer { curBuf in
                            workspace.readoutSums.withUnsafeMutableBufferPointer { sumBuf in
                                if numLayers <= 1 {
                                    LIFNeuronEngine.stepReadoutAdaptiveSIMD8(
                                        config: self.config,
                                        vPtr: vBuf.baseAddress!,
                                        sPtr: sBuf.baseAddress!,
                                        aPtr: aBuf.baseAddress!,
                                        curPtr: curBuf.baseAddress!,
                                        readoutSumPtr: sumBuf.baseAddress!,
                                        count: hMax
                                    )
                                } else {
                                    LIFNeuronEngine.stepAdaptiveSIMD8(
                                        config: self.config,
                                        vPtr: vBuf.baseAddress!,
                                        sPtr: sBuf.baseAddress!,
                                        aPtr: aBuf.baseAddress!,
                                        curPtr: curBuf.baseAddress!,
                                        count: hMax
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // 3.2 層 1 以降: 前層スパイク結合電流の RMSNorm と前層電流残差加算
            var layerIdx = 1
            while layerIdx < numLayers {
                let upperIdx = layerIdx - 1
                let curLayer = workspace.layerStates[layerIdx]
                let prevLayer = workspace.layerStates[layerIdx - 1]

                var activeLayerCount = 0
                prevLayer.s.withUnsafeBufferPointer { pPrevS in
                    var kj = 0
                    while kj < hMax {
                        if 0.0 < pPrevS[kj] {
                            workspace.activeLayerSpikes[activeLayerCount] = kj
                            activeLayerCount += 1
                        }
                        kj += 1
                    }
                }

                let bData = weights.bHLayers[upperIdx]
                let wLayerTData = wLayersT[upperIdx]
                workspace.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                    let stepCur = stepBuf.baseAddress!
                    bData.withUnsafeBufferPointer { bBuf in
                        stepCur.update(from: bBuf.baseAddress!, count: hMax)
                    }
                    wLayerTData.withUnsafeBufferPointer { wBuf in
                        let wT = wBuf.baseAddress!
                        var a = 0
                        while a < activeLayerCount {
                            let rowOffset = workspace.activeLayerSpikes[a] * hMax
                            var n = 0
                            while n < hLimit {
                                let acc = SIMD8<Float>(
                                    stepCur[n + 0], stepCur[n + 1], stepCur[n + 2], stepCur[n + 3],
                                    stepCur[n + 4], stepCur[n + 5], stepCur[n + 6], stepCur[n + 7]
                                )
                                let w = SIMD8<Float>(
                                    wT[rowOffset + n + 0], wT[rowOffset + n + 1],
                                    wT[rowOffset + n + 2], wT[rowOffset + n + 3],
                                    wT[rowOffset + n + 4], wT[rowOffset + n + 5],
                                    wT[rowOffset + n + 6], wT[rowOffset + n + 7]
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
                                stepCur[n] += wT[rowOffset + n]
                                n += 1
                            }
                            a += 1
                        }
                    }
                }

                let gammaData = weights.gammaRMS[upperIdx]
                workspace.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                    workspace.stepCurrentsPrev.withUnsafeMutableBufferPointer { prevBuf in
                        gammaData.withUnsafeBufferPointer { gammaBuf in
                            let stepCur = stepBuf.baseAddress!
                            let prevCur = prevBuf.baseAddress!
                            let gamma = gammaBuf.baseAddress!

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
                                    gamma[n + 0], gamma[n + 1], gamma[n + 2], gamma[n + 3],
                                    gamma[n + 4], gamma[n + 5], gamma[n + 6], gamma[n + 7]
                                )
                                let p = SIMD8<Float>(
                                    prevCur[n + 0], prevCur[n + 1], prevCur[n + 2], prevCur[n + 3],
                                    prevCur[n + 4], prevCur[n + 5], prevCur[n + 6], prevCur[n + 7]
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
                                let norm = (stepCur[n] * invRms) * gamma[n]
                                stepCur[n] = norm + prevCur[n]
                                n += 1
                            }

                            if (layerIdx + 1) < numLayers {
                                prevCur.update(from: stepCur, count: hMax)
                            }
                        }
                    }
                }

                curLayer.v.withUnsafeMutableBufferPointer { vBuf in
                    curLayer.s.withUnsafeMutableBufferPointer { sBuf in
                        curLayer.a.withUnsafeMutableBufferPointer { aBuf in
                            workspace.stepCurrents.withUnsafeBufferPointer { curBuf in
                                workspace.readoutSums.withUnsafeMutableBufferPointer { sumBuf in
                                    let isLast = (layerIdx + 1) == numLayers
                                    if isLast {
                                        LIFNeuronEngine.stepReadoutAdaptiveSIMD8(
                                            config: self.config,
                                            vPtr: vBuf.baseAddress!,
                                            sPtr: sBuf.baseAddress!,
                                            aPtr: aBuf.baseAddress!,
                                            curPtr: curBuf.baseAddress!,
                                            readoutSumPtr: sumBuf.baseAddress!,
                                            count: hMax
                                        )
                                    } else {
                                        LIFNeuronEngine.stepAdaptiveSIMD8(
                                            config: self.config,
                                            vPtr: vBuf.baseAddress!,
                                            sPtr: sBuf.baseAddress!,
                                            aPtr: aBuf.baseAddress!,
                                            curPtr: curBuf.baseAddress!,
                                            count: hMax
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                layerIdx += 1
            }

            step += 1
        }

        // 4. 最終層アナログ膜電位積算値からの線形スペクトル射影
        let safeTimeSteps = max(1, tSteps)
        let invT = 1.0 / Float(safeTimeSteps)
        var activeOutCount = 0
        workspace.readoutSums.withUnsafeBufferPointer { sumBuf in
            let sums = sumBuf.baseAddress!
            var k = 0
            while k < hMax {
                let rate = sums[k] * invT
                if rate != 0.0 {
                    workspace.activeReadoutIndices[activeOutCount] = k
                    workspace.activeRates[activeOutCount] = rate
                    activeOutCount += 1
                }
                k += 1
            }
        }

        let outLimit = outDim - (outDim % 8)
        weights.bOut.withUnsafeBufferPointer { bBuf in
            outputFeatures.update(from: bBuf.baseAddress!, count: outDim)
        }

        self.wOutT.withUnsafeBufferPointer { outBuf in
            let outT = outBuf.baseAddress!
            var a = 0
            while a < activeOutCount {
                let rowOffset = workspace.activeReadoutIndices[a] * outDim
                let rate = SIMD8<Float>(repeating: workspace.activeRates[a])
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
                    let sum = acc + (w * rate)
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
                    outputFeatures[c] += outT[rowOffset + c] * workspace.activeRates[a]
                    c += 1
                }
                a += 1
            }
        }
    }

    /// フレーム系列全体をデコードして音響特徴量系列を生成する。
    @discardableResult
    public func decodeSequence(
        featuresSeq: [[Float]],
        workspace: AcousticWorkspace
    ) -> [[Float]] {
        let totalFrames = featuresSeq.count
        if totalFrames <= 0 {
            return []
        }

        let outDim = weights.outputDim
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: outDim), count: totalFrames)

        workspace.reset()

        var t = 0
        while t < totalFrames {
            if 0 < t {
                // 直前フレームの直流入力で蓄積した膜電位を減衰させ、音素遷移に対する感度を確保する
                workspace.leakMembranes(decay: 0.2)
            }
            featuresSeq[t].withUnsafeBufferPointer { pIn in
                result[t].withUnsafeMutableBufferPointer { pOut in
                    decodeFrame(
                        features: pIn.baseAddress!,
                        workspace: workspace,
                        outputFeatures: pOut.baseAddress!
                    )
                }
            }
            t += 1
        }

        return result
    }
}
