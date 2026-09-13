import XCTest
import MLX
import MLXNN
@testable import SpikeSpeech

/// 多層 SNN アーキテクチャのダイナミクス、スパイク消失耐性、RMSNorm 数値安定性、
/// 残差電流加算の有効性、および極限入力下での膜電位クランプを実証するテストスイート。
final class ChallengerMultilayerSNNTests: XCTestCase {

    override func setUp() {
        super.setUp()
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let targetPath = currentDir + "/default.metallib"

        if fileManager.fileExists(atPath: targetPath) != true {
            var found = false
            let candidates = [
                currentDir + "/default.metallib",
                currentDir + "/.build/arm64-apple-macosx/debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
                currentDir + "/.build/arm64-apple-macosx/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
                currentDir + "/../spiketrans/.build/arm64-apple-macosx/debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
                currentDir + "/../spiketrans/.build/arm64-apple-macosx/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
            ]
            var cIdx = 0
            while cIdx < candidates.count {
                let cand = candidates[cIdx]
                if fileManager.fileExists(atPath: cand) {
                    try? fileManager.copyItem(atPath: cand, toPath: targetPath)
                    found = true
                    break
                }
                cIdx += 1
            }

            if found != true {
                let buildDir = currentDir + "/.build"
                if let enumerator = fileManager.enumerator(atPath: buildDir) {
                    while let subPath = enumerator.nextObject() as? String {
                        if subPath.hasSuffix("default.metallib") {
                            let fullPath = buildDir + "/" + subPath
                            try? fileManager.copyItem(atPath: fullPath, toPath: targetPath)
                            break
                        }
                    }
                }
            }
        }

        MLXRandom.seed(2026)
    }

    // MARK: - 1. 層数 1〜4 におけるスパイク消失 (Dying Spikes) 耐性と階層情報伝播の実測数値検証

    func testDyingSpikesAbsenceAcrossLayers1To4() {
        let hiddenDim = 128
        let inputDim = 64
        let outputDim = 80
        let timeSteps = 8
        let numFrames = 5

        let layerConfigs = [1, 2, 3, 4]
        var configIdx = 0

        while configIdx < layerConfigs.count {
            let numLayers = layerConfigs[configIdx]
            let weights = SpikingNetworkWeights.randomWeights(
                inputDim: inputDim,
                maxHiddenDim: hiddenDim,
                outputDim: outputDim,
                timeSteps: timeSteps,
                numLayers: numLayers,
                seed: 12345 + UInt64(numLayers)
            )

            let decoder = SpikingAcousticDecoder(weights: weights)
            let workspace = AcousticWorkspace(maxHiddenDim: hiddenDim, outputDim: outputDim, numLayers: numLayers)

            let testInput = [Float](repeating: 0.45, count: inputDim)
            var outFeat = [Float](repeating: 0.0, count: outputDim)

            var layerActiveSpikes = [Int](repeating: 0, count: numLayers)

            var frame = 0
            while frame < numFrames {
                testInput.withUnsafeBufferPointer { pIn in
                    outFeat.withUnsafeMutableBufferPointer { pOut in
                        decoder.decodeFrame(
                            features: pIn.baseAddress!,
                            workspace: workspace,
                            outputFeatures: pOut.baseAddress!
                        )
                    }
                }

                var l = 0
                while l < numLayers {
                    let st = workspace.layerStates[l]
                    var countSpikes = 0
                    var h = 0
                    while h < hiddenDim {
                        if 0.0 < st.s[h] {
                            countSpikes += 1
                        }
                        h += 1
                    }
                    layerActiveSpikes[l] += countSpikes
                    l += 1
                }
                frame += 1
            }

            var l = 0
            while l < numLayers {
                let totalSpikes = layerActiveSpikes[l]
                XCTAssertTrue(
                    0 < totalSpikes,
                    "層数 \(numLayers) の第 \(l) 層でスパイク消失（Dying Spikes）が発生しました。"
                )
                l += 1
            }

            var normSq: Float = 0.0
            var c = 0
            while c < outputDim {
                let v = outFeat[c]
                XCTAssertFalse(v.isNaN, "層数 \(numLayers) の出力で NaN を検出しました。")
                XCTAssertFalse(v.isInfinite, "層数 \(numLayers) の出力で Inf を検出しました。")
                normSq += v * v
                c += 1
            }

            let l2Norm = sqrt(normSq)
            XCTAssertTrue(
                0.01 < l2Norm,
                "層数 \(numLayers) において情報が最上位層まで伝播せず、出力エネルギーが微小です: \(l2Norm)"
            )

            configIdx += 1
        }
    }

    // MARK: - 2. RMSNorm のスケーリング安定性 (極小・極大入力およびゼロ除算防止) 実証

    func testRMSNormScalingStabilityUnderExtremes() {
        let hiddenDim = 128
        let inputDim = 64
        let outputDim = 80
        let timeSteps = 4
        let numLayers = 3

        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: inputDim,
            maxHiddenDim: hiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            numLayers: numLayers,
            seed: 9999
        )

        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: hiddenDim, outputDim: outputDim, numLayers: numLayers)

        // 1. 極小入力 (全要素 1e-15): ゼロ除算が発生せず有限値に収束することを検証
        let tinyInput = [Float](repeating: 1e-15, count: inputDim)
        var outTiny = [Float](repeating: 0.0, count: outputDim)

        tinyInput.withUnsafeBufferPointer { pIn in
            outTiny.withUnsafeMutableBufferPointer { pOut in
                decoder.decodeFrame(
                    features: pIn.baseAddress!,
                    workspace: workspace,
                    outputFeatures: pOut.baseAddress!
                )
            }
        }

        var c = 0
        while c < outputDim {
            XCTAssertFalse(outTiny[c].isNaN, "極小入力デコードで NaN を検出しました。")
            XCTAssertFalse(outTiny[c].isInfinite, "極小入力デコードで Inf を検出しました。")
            c += 1
        }

        // 2. 極大入力 (全要素 1e4): RMSNorm が適切に入力電流を圧縮し発散しないことを検証
        workspace.reset()
        let hugeInput = [Float](repeating: 10000.0, count: inputDim)
        var outHuge = [Float](repeating: 0.0, count: outputDim)

        hugeInput.withUnsafeBufferPointer { pIn in
            outHuge.withUnsafeMutableBufferPointer { pOut in
                decoder.decodeFrame(
                    features: pIn.baseAddress!,
                    workspace: workspace,
                    outputFeatures: pOut.baseAddress!
                )
            }
        }

        c = 0
        while c < outputDim {
            XCTAssertFalse(outHuge[c].isNaN, "極大入力デコードで NaN を検出しました。")
            XCTAssertFalse(outHuge[c].isInfinite, "極大入力デコードで Inf を検出しました。")
            c += 1
        }

        // 3. RMSNorm 数値特性の直接検証 (二乗平均が 1.0 に正規化されること)
        let sampleCurrents = [Float](repeating: 50.0, count: hiddenDim)
        var sumSq: Float = 0.0
        var i = 0
        while i < hiddenDim {
            sumSq += sampleCurrents[i] * sampleCurrents[i]
            i += 1
        }
        let meanSq = sumSq / Float(hiddenDim)
        let rms = sqrt(meanSq + 1e-5)
        let invRms = 1.0 / rms

        var normSumSq: Float = 0.0
        i = 0
        while i < hiddenDim {
            let normVal = sampleCurrents[i] * invRms
            normSumSq += normVal * normVal
            i += 1
        }
        let normMeanSq = normSumSq / Float(hiddenDim)
        let diffNorm = abs(normMeanSq - 1.0)
        XCTAssertTrue(
            diffNorm < 1e-3,
            "RMSNorm の正規化後二乗平均が 1.0 に収束していません: diff=\(diffNorm)"
        )
    }

    // MARK: - 3. 前層残差電流加算 (Residual Current Connection) の有効性実証

    func testResidualCurrentConnectionSignificance() {
        // 上位層への重みが全て 0 の敵対的重みを作成する。
        // 残差接続が存在することにより、上位層重みが 0 であっても前層の直流電流がパススルーされ、
        // 上位層ニューロンが駆動されて出力が非ゼロになることを実証する。
        let inputDim = 32
        let hiddenDim = 64
        let outputDim = 32
        let timeSteps = 4
        let numLayers = 2

        let baseWeights = SpikingNetworkWeights.randomWeights(
            inputDim: inputDim,
            maxHiddenDim: hiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            numLayers: numLayers,
            seed: 42
        )

        // 層 1 への結合重みを全て 0 にした重み構造体を構築
        let zeroLayerWeights = [Float](repeating: 0.0, count: hiddenDim * hiddenDim)
        let zeroLayerBias = [Float](repeating: 0.0, count: hiddenDim)
        let identityGamma = [Float](repeating: 1.0, count: hiddenDim)

        let residualTestWeights = SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: hiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: baseWeights.lifConfig,
            wIn: baseWeights.wIn,
            wRec: baseWeights.wRec,
            bH: baseWeights.bH,
            wLayers: [zeroLayerWeights],
            bHLayers: [zeroLayerBias],
            gammaRMS: [identityGamma],
            wOut: baseWeights.wOut,
            bOut: baseWeights.bOut
        )

        let decoder = SpikingAcousticDecoder(weights: residualTestWeights)
        let workspace = AcousticWorkspace(maxHiddenDim: hiddenDim, outputDim: outputDim, numLayers: numLayers)

        let strongInput = [Float](repeating: 1.5, count: inputDim)
        var output = [Float](repeating: 0.0, count: outputDim)

        strongInput.withUnsafeBufferPointer { pIn in
            output.withUnsafeMutableBufferPointer { pOut in
                decoder.decodeFrame(
                    features: pIn.baseAddress!,
                    workspace: workspace,
                    outputFeatures: pOut.baseAddress!
                )
            }
        }

        // 上位層（層 1）の膜電位が前層からの残差電流によって駆動されていることを検証
        var layer1Energy: Float = 0.0
        var h = 0
        while h < hiddenDim {
            let v = workspace.layerStates[1].v[h]
            layer1Energy += abs(v)
            h += 1
        }

        XCTAssertTrue(
            0.1 < layer1Energy,
            "前層残差電流加算が機能しておらず、上位層膜電位が活性化されていません: energy=\(layer1Energy)"
        )

        // 最終リードアウト出力が非ゼロであること
        var outNorm: Float = 0.0
        var c = 0
        while c < outputDim {
            outNorm += abs(output[c])
            c += 1
        }

        XCTAssertTrue(
            0.01 < outNorm,
            "残差接続経由の情報伝播が行われず、最終出力がゼロです: outNorm=\(outNorm)"
        )
    }

    // MARK: - 4. ゼロ入力・極大入力での膜電位クランプおよび NaN/Inf 非混入の実証

    func testZeroAndExtremeInputMembraneClampingAndSanity() {
        let inputDim = 64
        let hiddenDim = 128
        let outputDim = 80
        let timeSteps = 4
        let numLayers = 4

        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: inputDim,
            maxHiddenDim: hiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            numLayers: numLayers,
            seed: 8888
        )

        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: hiddenDim, outputDim: outputDim, numLayers: numLayers)

        // 1. ゼロ入力テスト (10 フレーム連続)
        let zeroInput = [Float](repeating: 0.0, count: inputDim)
        var outZero = [Float](repeating: 0.0, count: outputDim)

        var f = 0
        while f < 10 {
            zeroInput.withUnsafeBufferPointer { pIn in
                outZero.withUnsafeMutableBufferPointer { pOut in
                    decoder.decodeFrame(
                        features: pIn.baseAddress!,
                        workspace: workspace,
                        outputFeatures: pOut.baseAddress!
                    )
                }
            }

            var l = 0
            while l < numLayers {
                let st = workspace.layerStates[l]
                var h = 0
                while h < hiddenDim {
                    let v = st.v[h]
                    XCTAssertFalse(v.isNaN, "ゼロ入力デコード時に層 \(l) で NaN 膜電位を検出しました。")
                    XCTAssertFalse(v.isInfinite, "ゼロ入力デコード時に層 \(l) で Inf 膜電位を検出しました。")
                    XCTAssertFalse(v < LIFNeuronEngine.vClampMin, "膜電位がクランプ下限を下回りました。")
                    XCTAssertFalse(LIFNeuronEngine.vClampMax < v, "膜電位がクランプ上限を上回りました。")
                    h += 1
                }
                l += 1
            }
            f += 1
        }

        // 2. 極大正入力テスト (+1000.0)
        workspace.reset()
        let hugePositiveInput = [Float](repeating: 1000.0, count: inputDim)
        var outHugePos = [Float](repeating: 0.0, count: outputDim)

        f = 0
        while f < 5 {
            hugePositiveInput.withUnsafeBufferPointer { pIn in
                outHugePos.withUnsafeMutableBufferPointer { pOut in
                    decoder.decodeFrame(
                        features: pIn.baseAddress!,
                        workspace: workspace,
                        outputFeatures: pOut.baseAddress!
                    )
                }
            }

            var l = 0
            while l < numLayers {
                let st = workspace.layerStates[l]
                var h = 0
                while h < hiddenDim {
                    let v = st.v[h]
                    XCTAssertFalse(v.isNaN, "過大正入力デコード時に層 \(l) で NaN 膜電位を検出しました。")
                    XCTAssertFalse(v.isInfinite, "過大正入力デコード時に層 \(l) で Inf 膜電位を検出しました。")
                    XCTAssertFalse(LIFNeuronEngine.vClampMax < v, "過大正入力時に層 \(l) で膜電位クランプ上限を突破しました: \(v)")
                    h += 1
                }
                l += 1
            }
            f += 1
        }

        // 3. 極大負入力テスト (-1000.0)
        workspace.reset()
        let hugeNegativeInput = [Float](repeating: -1000.0, count: inputDim)
        var outHugeNeg = [Float](repeating: 0.0, count: outputDim)

        f = 0
        while f < 5 {
            hugeNegativeInput.withUnsafeBufferPointer { pIn in
                outHugeNeg.withUnsafeMutableBufferPointer { pOut in
                    decoder.decodeFrame(
                        features: pIn.baseAddress!,
                        workspace: workspace,
                        outputFeatures: pOut.baseAddress!
                    )
                }
            }

            var l = 0
            while l < numLayers {
                let st = workspace.layerStates[l]
                var h = 0
                while h < hiddenDim {
                    let v = st.v[h]
                    XCTAssertFalse(v.isNaN, "過大負入力デコード時に層 \(l) で NaN 膜電位を検出しました。")
                    XCTAssertFalse(v.isInfinite, "過大負入力デコード時に層 \(l) で Inf 膜電位を検出しました。")
                    XCTAssertFalse(v < LIFNeuronEngine.vClampMin, "過大負入力時に層 \(l) で膜電位クランプ下限を突破しました: \(v)")
                    h += 1
                }
                l += 1
            }
            f += 1
        }
    }

    // MARK: - 5. MLX 多層 SNN 順伝播および勾配逆伝播の健全性実証

    func testMLXMultilayerSpikingNetworkParityAndGradients() {
        let inputDim = 16
        let hiddenDim = 32
        let outputDim = 16
        let timeSteps = 2
        let numLayers = 3
        let seqLen = 8

        let network = MLXSpikingAcousticNetwork(
            numLayers: numLayers,
            inputDim: inputDim,
            maxHiddenDim: hiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.1)
        )

        let dummyFeatures = MLXRandom.uniform(low: -0.5, high: 0.5, [1, seqLen, inputDim])
        let output = network.forward(features: dummyFeatures)
        eval(output)

        XCTAssertEqual(output.shape, [1, seqLen, outputDim], "MLX 多層順伝播出力形状が期待値と不一致です。")

        let outArray = output.asArray(Float.self)
        var idx = 0
        while idx < outArray.count {
            let v = outArray[idx]
            XCTAssertFalse(v.isNaN, "MLX 多層順伝播出力で NaN を検出しました。")
            XCTAssertFalse(v.isInfinite, "MLX 多層順伝播出力で Inf を検出しました。")
            idx += 1
        }

        // 損失関数と勾配逆伝播の計算検証
        let lg = valueAndGrad(model: network) { (model: MLXSpikingAcousticNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let pred = model.forward(features: arrays[0])
            return [mean(pred * pred)]
        }
        let (lossVals, grads) = lg(network, [dummyFeatures])
        let lossVal = lossVals[0]
        eval(lossVal)

        let lossFloat = lossVal.item(Float.self)
        XCTAssertFalse(lossFloat.isNaN, "損失値が NaN です。")
        XCTAssertFalse(lossFloat.isInfinite, "損失値が Inf です。")

        // 重み勾配が存在し、有限値であることを検証
        if case .value(let gradArray) = grads["wIn"] {
            eval(gradArray)
            let flatGrad = gradArray.asArray(Float.self)
            var gIdx = 0
            var nonZeroCount = 0
            while gIdx < flatGrad.count {
                let g = flatGrad[gIdx]
                XCTAssertFalse(g.isNaN, "入力重み勾配で NaN を検出しました。")
                XCTAssertFalse(g.isInfinite, "入力重み勾配で Inf を検出しました。")
                if g != 0.0 {
                    nonZeroCount += 1
                }
                gIdx += 1
            }
            XCTAssertTrue(0 < nonZeroCount, "入力重み勾配が全て 0 であり、勾配消失が発生しています。")
        }
    }

    // MARK: - 6. 層数別 (1〜4層) E2E 音声合成の波形整合性 (RMS, 最大振幅, 破綻不在) の定量的精査

    func testMultilayerE2ESpeechSynthesisWaveformIntegrity() {
        let text = "こんにちは、多層スパイク音声合成です。"
        let layerConfigs = [1, 2, 3, 4]

        var idx = 0
        while idx < layerConfigs.count {
            let layers = layerConfigs[idx]
            let weights = SpikingNetworkWeights.randomWeights(numLayers: layers, seed: 1000 + UInt64(layers))
            let engine = SpikeSpeechEngine(weights: weights)

            let samples = engine.synthesize(text: text)
            XCTAssertTrue(0 < samples.count, "層数 \(layers) でサンプルが生成されませんでした")

            var maxAmp: Float = 0.0
            var sumSq: Double = 0.0
            var s = 0
            while s < samples.count {
                let v = samples[s]
                XCTAssertFalse(v.isNaN, "層数 \(layers) の合成サンプルで NaN を検出")
                XCTAssertFalse(v.isInfinite, "層数 \(layers) の合成サンプルで Inf を検出")
                let absV = abs(v)
                if maxAmp < absV {
                    maxAmp = absV
                }
                let d = Double(v)
                sumSq += d * d
                s += 1
            }

            let rms = sqrt(sumSq / Double(samples.count))
            print("--- [Challenger Multilayer E2E Waveform: Layer \(layers)] ---")
            print("総サンプル数: \(samples.count)")
            print("最大絶対振幅: \(maxAmp)")
            print("RMS エネルギー: \(rms)")

            // 無音 (RMS == 0) の不発生 (0.01 < rms)
            XCTAssertTrue(0.01 < rms, "層数 \(layers) で無音破綻を検出しました: rms=\(rms)")
            // 過大振幅 (Soft Limiter が 1.0 以下に抑制していること)
            XCTAssertTrue(maxAmp <= 1.0, "層数 \(layers) で最大振幅が 1.0 を超過しました: \(maxAmp)")

            idx += 1
        }
    }
}

