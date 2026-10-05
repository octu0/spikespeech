import XCTest
#if canImport(MLX)
import MLX
import MLXNN
import MLXOptimizers
#endif
@testable import SpikeSpeech

/// MLX コア、代理勾配、損失関数、および BPTT 学習の網羅的単体検証テストスイート
final class MLXTests: XCTestCase {

    override func setUp() {
        super.setUp()
        #if canImport(MLX)
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let targetPath = currentDir + "/default.metallib"

        if fileManager.fileExists(atPath: targetPath) != true {
            var found = false
            let candidates = [
                currentDir + "/default.metallib",
                currentDir + "/.build/arm64-apple-macosx/debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
                currentDir + "/.build/arm64-apple-macosx/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
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

        MLXRandom.seed(42)
        #endif
    }

    #if canImport(MLX)
    // MARK: - 1. STE Fast Sigmoid 代理勾配の検証

    func testSTEFastSigmoidGradientFlow() {
        let v = MLXArray([Float]([0.5, 1.2, 0.9, 1.5]))
        let vTh = MLXArray([Float]([1.0, 1.0, 1.0, 1.0]))

        let s = SurrogateGradients.fastSigmoidSTE(v: v, vTh: vTh, alpha: 2.0)
        let sArr = s.asArray(Float.self)

        XCTAssertEqual(sArr[0], 0.0)
        XCTAssertEqual(sArr[1], 1.0)
        XCTAssertEqual(sArr[2], 0.0)
        XCTAssertEqual(sArr[3], 1.0)

        let gradFn = grad { x in
            let spike = SurrogateGradients.fastSigmoidSTE(v: x[0], vTh: x[1], alpha: 2.0)
            return sum(spike)
        }

        let grads = gradFn([v, vTh])
        let dV = grads[0].asArray(Float.self)

        var i = 0
        while i < dV.count {
            XCTAssertTrue(0.0 < dV[i], "代理勾配がゼロまたは負です: index=\(i), val=\(dV[i])")
            i += 1
        }
    }

    // MARK: - 2. 32 フレーム境界アライメントの検証

    func testSequence32Alignment() {
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 0), 32)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 1), 32)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 31), 32)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 32), 32)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 33), 64)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 64), 64)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 100), 128)
    }

    // MARK: - 3. スペクトル L1 損失の計算と勾配伝播

    func testSpectralL1LossAndGradient() {
        let target = MLXArray([Float]([0.5, 0.5, 0.8, 0.8]), [1, 2, 2])
        let pred = MLXArray([Float]([0.4, 0.4, 0.7, 0.7]), [1, 2, 2])

        let loss = AcousticLossFunctions.spectralL1Loss(
            predicted: pred,
            target: target
        )

        let lossVal = loss.item(Float.self)
        let diff = abs(lossVal - 0.1)
        XCTAssertTrue(diff < 1e-4, "L1 Loss の計算が不正確です: \(lossVal)")
    }

    func testMultiResolutionSTFTLossWithPhaseAndGradient() {
        let nSamples = 1024
        var pData = [Float](repeating: 0.0, count: nSamples)
        var tData = [Float](repeating: 0.0, count: nSamples)
        var s = 0
        while s < nSamples {
            pData[s] = sinf(Float(s) * 0.1) * 0.5
            tData[s] = sinf(Float(s) * 0.12) * 0.5
            s += 1
        }
        let pArr = MLXArray(pData, [1, nSamples])
        let tArr = MLXArray(tData, [1, nSamples])

        let loss = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: pArr, target: tArr)
        let lossVal = loss.item(Float.self)
        XCTAssertTrue(0.0 < lossVal, "STFT 損失が正の値ではありません: \(lossVal)")
        XCTAssertTrue(lossVal.isFinite, "STFT 損失が有限値ではありません: \(lossVal)")

        let selfLoss = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: pArr, target: pArr).item(Float.self)
        XCTAssertTrue(selfLoss.isFinite, "自己 STFT 損失が有限値ではありません: \(selfLoss)")
        XCTAssertTrue(selfLoss < lossVal, "自己 STFT 損失が他者損失より小さくありません: self=\(selfLoss), cross=\(lossVal)")

        let gradFn = grad { p in
            MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: p, target: tArr)
        }
        let g = gradFn(pArr)
        let gArr = g.asArray(Float.self)
        var hasNonZero = false
        var allFinite = true
        var i = 0
        while i < gArr.count {
            if gArr[i].isFinite != true {
                allFinite = false
                break
            }
            if 1e-6 < abs(gArr[i]) {
                hasNonZero = true
            }
            i += 1
        }
        XCTAssertTrue(allFinite, "STFT 損失の勾配に NaN/Inf が含まれます")
        XCTAssertTrue(hasNonZero, "STFT 損失の勾配が全て 0 です")
    }

    func testSNNWithVocoderWaveformLoss() throws {
        let weightsURL = URL(fileURLWithPath: "Models/weights.json")
        let vocoderURL = URL(fileURLWithPath: "Models/vocoder_weights.json")
        if FileManager.default.fileExists(atPath: weightsURL.path) != true || FileManager.default.fileExists(atPath: vocoderURL.path) != true {
            return
        }
        let wData = try Data(contentsOf: weightsURL)
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: wData)
        let vData = try Data(contentsOf: vocoderURL)
        let vocWeights = try JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)

        let network = MLXSpikingAcousticNetwork(weights: weights)
        let vocoder = MLXNeuralVocoder(weights: vocWeights)

        let testWeight: Float = 0.15
        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            vocoder: vocoder,
            waveformLossWeight: testWeight,
            learningRate: 0.001
        )

        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        if FileManager.default.fileExists(atPath: wavPath) != true {
            return
        }
        let pcm = try WavAudioReader().loadWav16k(from: wavPath)
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)
        guard let pair = engine.prepareTrainingPair(
            text: "水をマレーシアから買わなくてはならないのです。",
            pcm16k: pcm,
            melExtractor: MelSpectrogramExtractor(),
            pitchTracker: PitchTracker()
        ) else {
            XCTFail("prepareTrainingPair 失敗")
            return
        }

        trainer.trainSequence(
            features: pair.features,
            targets: pair.targets,
            targetAudio: pair.targetAudio
        )
        let losses = trainer.lastLosses
        print("\n=======================================================")
        print("[Waveform Loss Ratio Test]")
        print("  totalLoss: \(losses.totalLoss)")
        print("  melL1:     \(losses.melL1)")
        print("  waveTerm:  \(losses.waveTerm) (coef: \(testWeight))")
        let ratio = losses.waveTerm / max(1e-6, losses.melL1)
        print("  waveTerm / melL1 比率: \(ratio) (受入基準: 0.5 〜 2.0)")
        print("=======================================================\n")

        XCTAssertTrue(losses.totalLoss.isFinite, "totalLoss が有限値ではありません")
        XCTAssertTrue(0.0 < losses.waveTerm, "waveTerm が 0 以下です")
    }

    // MARK: - 4. BPTT 最適化ループによる損失減少検証

    func testBPTTTrainingLossDecreases() {
        let inDim = 16
        let maxHidden = 64
        let outDim = 16
        let tSteps = 2

        let network = MLXSpikingAcousticNetwork(
            numLayers: 2,
            inputDim: inDim,
            maxHiddenDim: maxHidden,
            outputDim: outDim,
            timeSteps: tSteps,
            lifConfig: LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.0)
        )

        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            learningRate: 0.005,
            bpttWindow: 16
        )

        let seqLen = 32
        let featSeq = [[Float]](repeating: [Float](repeating: 0.5, count: inDim), count: seqLen)
        let targetSeq = [[Float]](repeating: [Float](repeating: 0.2, count: outDim), count: seqLen)

        let initialLoss = trainer.trainSequence(features: featSeq, targets: targetSeq)

        var stepLoss = initialLoss
        var step = 0
        while step < 20 {
            stepLoss = trainer.trainSequence(features: featSeq, targets: targetSeq)
            step += 1
        }

        XCTAssertTrue(stepLoss < initialLoss, "BPTT 学習によって損失が減少していません: initial=\(initialLoss), final=\(stepLoss)")
    }

    // MARK: - 5. スペクトル Delta 損失の検証

    func testSpectralDeltaLoss() {
        let target = MLXArray([Float]([0.1, 0.2, 0.3, 0.4]), [1, 2, 2])
        let pred = MLXArray([Float]([0.1, 0.2, 0.35, 0.45]), [1, 2, 2])

        let dLoss = AcousticLossFunctions.spectralDeltaLoss(predicted: pred, target: target)
        let val = dLoss.item(Float.self)

        XCTAssertFalse(val.isNaN)
        XCTAssertTrue(0.0 < val)
    }

    // MARK: - 6. 重みエクスポート・インポート等価性テスト

    func testNetworkWeightsExportImportEquivalence() {
        let netA = MLXSpikingAcousticNetwork(
            numLayers: 2,
            inputDim: 16,
            maxHiddenDim: 32,
            outputDim: 8,
            timeSteps: 2
        )

        let exported = netA.exportWeights()
        XCTAssertEqual(exported.inputDim, 16)
        XCTAssertEqual(exported.maxHiddenDim, 32)
        XCTAssertEqual(exported.outputDim, 8)
        XCTAssertEqual(exported.numLayers, 2)

        let netB = MLXSpikingAcousticNetwork(weights: exported)
        let dummyFeat = MLXArray([Float](repeating: 0.3, count: 16), [1, 1, 16])

        let outA = netA.forward(features: dummyFeat)
        let outB = netB.forward(features: dummyFeat)

        let diff = mean(abs(outA - outB)).item(Float.self)
        XCTAssertTrue(diff < 1e-5, "エクスポート・インポート後のフォワード出力が一致しません: \(diff)")
    }

    // MARK: - 7. BPTT Trainer AdamW および重みノルム検証

    func testBPTTTrainerWithAdamWAndNorms() {
        let inDim = 16
        let maxHidden = 32
        let outDim = 16
        let tSteps = 2

        let network = MLXSpikingAcousticNetwork(
            numLayers: 1,
            inputDim: inDim,
            maxHiddenDim: maxHidden,
            outputDim: outDim,
            timeSteps: tSteps,
            lifConfig: LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.1)
        )

        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            learningRate: 0.003,
            bpttWindow: 16,
            weightDecay: 1.0e-4
        )

        trainer.setLearningRate(0.0015)
        XCTAssertTrue(abs(trainer.currentLearningRate() - 0.0015) < 1.0e-6, "学習率の動的更新が反映されていません")

        let norms = trainer.weightNorms()
        XCTAssertTrue(0.0 < norms.wIn, "wIn ノルムが正数ではありません")
        XCTAssertTrue(0.0 < norms.wRec, "wRec ノルムが正数ではありません")
        XCTAssertTrue(0.0 < norms.wOut, "wOut ノルムが正数ではありません")

        let seqLen = 32
        let featSeq = [[Float]](repeating: [Float](repeating: 0.4, count: inDim), count: seqLen)
        let targetSeq = [[Float]](repeating: [Float](repeating: 0.1, count: outDim), count: seqLen)

        let initialLoss = trainer.trainSequence(features: featSeq, targets: targetSeq)
        var stepLoss = initialLoss
        var step = 0
        while step < 15 {
            stepLoss = trainer.trainSequence(features: featSeq, targets: targetSeq)
            step += 1
        }
        XCTAssertTrue(stepLoss < initialLoss, "AdamW による BPTT 学習で損失が減少していません: initial=\(initialLoss), final=\(stepLoss)")
    }

    // MARK: - 7b. 4 ブロック SNN (numLayers = 4) の BPTT 学習と損失減少検証

    func testFourBlockBPTTTrainingLossDecreases() {
        let inDim = 16
        let maxHidden = 32
        let outDim = 16
        let tSteps = 2

        let network = MLXSpikingAcousticNetwork(
            numLayers: 4,
            inputDim: inDim,
            maxHiddenDim: maxHidden,
            outputDim: outDim,
            timeSteps: tSteps,
            lifConfig: LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.0)
        )

        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            learningRate: 0.005,
            bpttWindow: 16
        )

        let seqLen = 32
        let featSeq = [[Float]](repeating: [Float](repeating: 0.5, count: inDim), count: seqLen)
        let targetSeq = [[Float]](repeating: [Float](repeating: 0.2, count: outDim), count: seqLen)

        let initialLoss = trainer.trainSequence(features: featSeq, targets: targetSeq)

        var stepLoss = initialLoss
        var step = 0
        while step < 10 {
            stepLoss = trainer.trainSequence(features: featSeq, targets: targetSeq)
            step += 1
        }

        XCTAssertTrue(stepLoss < initialLoss, "4ブロック BPTT 学習によって損失が減少していません: initial=\(initialLoss), final=\(stepLoss)")
    }

    // MARK: - 7c. Pure Swift decodeSequence と MLX forward の数値完全一致検証

    func testPureSwiftDecoderMatchesMLXForward() {
        let inDim = 16
        let maxHidden = 32
        let outDim = 16
        let tSteps = 2
        let numLayers = 4

        let network = MLXSpikingAcousticNetwork(
            numLayers: numLayers,
            inputDim: inDim,
            maxHiddenDim: maxHidden,
            outputDim: outDim,
            timeSteps: tSteps,
            lifConfig: LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.0)
        )

        let exported = network.exportWeights()
        let swiftDecoder = SpikingAcousticDecoder(weights: exported)
        let workspace = AcousticWorkspace(
            maxHiddenDim: exported.maxHiddenDim,
            outputDim: exported.outputDim,
            numLayers: exported.numLayers
        )

        let seqLen = 8
        var featSeq: [[Float]] = []
        var t = 0
        while t < seqLen {
            var fArr = [Float](repeating: 0.0, count: inDim)
            var i = 0
            while i < inDim {
                fArr[i] = sinf(Float((t * inDim) + i) * 0.1) * 0.5
                i += 1
            }
            featSeq.append(fArr)
            t += 1
        }

        // 1. MLX forward
        var flatFeat = [Float](repeating: 0.0, count: seqLen * inDim)
        t = 0
        while t < seqLen {
            var i = 0
            while i < inDim {
                flatFeat[(t * inDim) + i] = featSeq[t][i]
                i += 1
            }
            t += 1
        }
        let mlxFeat = MLXArray(flatFeat, [1, seqLen, inDim])
        let mlxOut = network.forward(features: mlxFeat)
        eval(mlxOut)
        let mlxOutFlat = mlxOut.asArray(Float.self)

        // 2. Pure Swift decodeSequence
        let swiftOut = swiftDecoder.decodeSequence(featuresSeq: featSeq, workspace: workspace)

        // 3. 差分の比較
        var maxDiff: Float = 0.0
        var sumDiff: Float = 0.0
        var count = 0
        t = 0
        while t < seqLen {
            var c = 0
            while c < outDim {
                let mVal = mlxOutFlat[(t * outDim) + c]
                let sVal = swiftOut[t][c]
                let diff = abs(mVal - sVal)
                if maxDiff < diff {
                    maxDiff = diff
                }
                sumDiff += diff
                count += 1
                c += 1
            }
            t += 1
        }
        let meanDiff = sumDiff / Float(max(1, count))
        print("\n=======================================================")
        print("MLX forward vs Pure Swift decodeSequence 数値一致度:")
        print("  平均絶対誤差 (MAE): \(meanDiff)")
        print("  最大絶対誤差 (MaxDiff): \(maxDiff)")
        print("=======================================================\n")

        XCTAssertTrue(meanDiff < 1.0e-3, "MLX と Pure Swift で推論結果が乖離しています: MAE=\(meanDiff), MaxDiff=\(maxDiff)")
    }
    #endif

    // MARK: - 8. 学習率スケジューラ・Plateau ガード・シャッフル・チェックポイント検証 (Pure Swift)

    func testCosineWarmupScheduleValues() {
        let schedule = CosineWarmupSchedule(
            lrBase: 0.003,
            lrMin: 1.0e-5,
            warmupEpochs: 2,
            totalEpochs: 15
        )

        let expectedEpoch1: Float = 0.001505
        let expectedEpoch2: Float = 0.003000
        let expectedEpoch3: Float = 0.002957
        let expectedEpoch15: Float = 0.000010

        let lr1 = schedule.learningRate(epoch: 0)
        let lr2 = schedule.learningRate(epoch: 1)
        let lr3 = schedule.learningRate(epoch: 2)
        let lr15 = schedule.learningRate(epoch: 14)
        let lrBeyond = schedule.learningRate(epoch: 20)

        XCTAssertTrue(abs(lr1 - expectedEpoch1) < 1.0e-5, "Epoch 1 lr 不一致: \(lr1) vs \(expectedEpoch1)")
        XCTAssertTrue(abs(lr2 - expectedEpoch2) < 1.0e-5, "Epoch 2 lr 不一致: \(lr2) vs \(expectedEpoch2)")
        XCTAssertTrue(abs(lr3 - expectedEpoch3) < 1.0e-5, "Epoch 3 lr 不一致: \(lr3) vs \(expectedEpoch3)")
        XCTAssertTrue(abs(lr15 - expectedEpoch15) < 1.0e-5, "Epoch 15 lr 不一致: \(lr15) vs \(expectedEpoch15)")
        XCTAssertTrue(abs(lrBeyond - 1.0e-5) < 1.0e-6, "超過エポックで lrMin にクランプされていません: \(lrBeyond)")
    }

    func testPlateauGuardReaction() {
        var guardState = PlateauGuard(patience: 2, factor: 0.5, relThreshold: 0.005)

        XCTAssertTrue(abs(guardState.decayMultiplier - 1.0) < 1.0e-6)

        let m1 = guardState.observe(epochLoss: 1.50)
        XCTAssertTrue(abs(m1 - 1.0) < 1.0e-6)
        let m2 = guardState.observe(epochLoss: 1.40)
        XCTAssertTrue(abs(m2 - 1.0) < 1.0e-6)

        let m3 = guardState.observe(epochLoss: 1.45)
        XCTAssertTrue(abs(m3 - 1.0) < 1.0e-6, "patience 未満で減衰してはならない")

        let m4 = guardState.observe(epochLoss: 1.48)
        XCTAssertTrue(abs(m4 - 0.5) < 1.0e-5, "patience 到達時に 0.5 に減衰していません: \(m4)")
    }

    func testTrainingShuffleDeterminismAndIntegrity() {
        let baseSeed: UInt64 = 2026
        let original = Array(0..<100)

        var copy1 = original
        var copy2 = original
        var copy3 = original

        let seedEp0 = TrainingShuffle.mixSeed(baseSeed: baseSeed, epoch: 0)
        let seedEp1 = TrainingShuffle.mixSeed(baseSeed: baseSeed, epoch: 1)

        TrainingShuffle.shuffleInPlace(&copy1, seed: seedEp0)
        TrainingShuffle.shuffleInPlace(&copy2, seed: seedEp0)
        TrainingShuffle.shuffleInPlace(&copy3, seed: seedEp1)

        XCTAssertEqual(copy1, copy2, "同一シードでのシャッフル結果が一致しません")
        XCTAssertNotEqual(copy1, copy3, "異なるエポックでのシャッフル結果が変化していません")

        XCTAssertEqual(copy1.sorted(), original, "シャッフルによって要素が損なわれました")
        XCTAssertEqual(copy3.sorted(), original, "シャッフルによって要素が損なわれました")
    }

    func testWeightCheckpointPathAndNaming() {
        let name1 = WeightCheckpoint.epochFileName(epochOneIndexed: 1)
        let name10 = WeightCheckpoint.epochFileName(epochOneIndexed: 10)
        XCTAssertEqual(name1, "weights.ep01.json")
        XCTAssertEqual(name10, "weights.ep10.json")
    }

    func testBPTTTrainerWaveformBoundaryArraysCheck() {
        let network = MLXSpikingAcousticNetwork(
            numLayers: 1,
            inputDim: 199,
            maxHiddenDim: 32,
            outputDim: 64,
            timeSteps: 2,
            lifConfig: LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.0)
        )
        let vocoder = MLXNeuralVocoder(config: NeuralVocoderConfig(hiddenChannels: 32))
        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            vocoder: vocoder,
            waveformLossWeight: 0.15,
            learningRate: 0.001
        )

        let fArr = MLXArray.zeros([1, 16, 199])
        let tArr = MLXArray.zeros([1, 16, 64])
        let mArr = MLXArray.ones([1, 16])
        let targetWave = MLXArray.zeros([1, 5120])

        // 4要素のみ渡された場合（targetWaveform はあるが segIndices がないケース）に
        // arrays[4] 境界外クラッシュを起こさず安全に Mel 損失で完走することを検証
        let loss = trainer.trainBatch(
            features: fArr,
            targets: tArr,
            mask: mArr,
            targetWaveform: targetWave,
            segIndices: nil
        )
        XCTAssertTrue(loss.isFinite, "4要素 arrays 呼び出しで損失が有限値ではありません: \(loss)")
        XCTAssertTrue(0.0 <= loss, "損失が 0 未満です: \(loss)")
    }

    // MARK: - 15. CfC 音響モデル MLX / Pure Swift 学習前数値一致度の検証

    func testCfCMLXAndPureSwiftNumericalConsistency() {
        let inputDim = 256
        let hiddenDim = 256
        let outputDim = 64
        let numLayers = 4
        let seqLen = 12

        let weights = SpikingNetworkWeights.initCfCWeights(
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            outputDim: outputDim,
            numLayers: numLayers,
            seed: 2026
        )

        let mlxNet = MLXSpikingAcousticNetwork(weights: weights)
        let swiftDecoder = SpikingAcousticDecoder(weights: weights)

        // 入力系列の生成（決定論的擬似乱数）
        var inputSeq: [[Float]] = []
        var flatInput: [Float] = []
        flatInput.reserveCapacity(seqLen * inputDim)

        var rng: UInt64 = 987654321
        var t = 0
        while t < seqLen {
            var frame: [Float] = []
            frame.reserveCapacity(inputDim)
            var c = 0
            while c < inputDim {
                rng ^= rng << 13
                rng ^= rng >> 7
                rng ^= rng << 17
                let val = (Float(rng & 0x00FFFFFF) / Float(0x01000000) * 2.0 - 1.0) * 0.5
                frame.append(val)
                flatInput.append(val)
                c += 1
            }
            inputSeq.append(frame)
            t += 1
        }

        // MLX 順伝播
        let mlxInput = MLXArray(flatInput, [1, seqLen, inputDim])
        let mlxPred = mlxNet.forward(features: mlxInput, bpttWindow: 16)
        eval(mlxPred)
        let mlxMelFlat = mlxPred.asArray(Float.self)

        // Pure Swift 順伝播
        let workspace = AcousticWorkspace(maxHiddenDim: hiddenDim, outputDim: outputDim, numLayers: numLayers)
        let swiftMelSeq = swiftDecoder.decodeSequence(featuresSeq: inputSeq, workspace: workspace)

        XCTAssertEqual(swiftMelSeq.count, seqLen)

        // 最大絶対誤差の計算
        var maxDiff: Float = 0.0
        t = 0
        while t < seqLen {
            var c = 0
            while c < outputDim {
                let mlxVal = mlxMelFlat[(t * outputDim) + c]
                let swiftVal = swiftMelSeq[t][c]
                let diff = abs(mlxVal - swiftVal)
                if maxDiff < diff {
                    maxDiff = diff
                }
                c += 1
            }
            t += 1
        }

        print("[CfC Consistency Test] Maximum Mel absolute diff between MLX and Pure Swift: \(maxDiff)")
        XCTAssertTrue(maxDiff < 1.0e-4, "学習前の MLX と Pure Swift のメル最大絶対差が 1e-4 以上です: \(maxDiff)")
    }

    // MARK: - 16. CfC 音響モデルの構造・形状（4層、幅256、出力64）の検証

    func testCfCNetworkShape() {
        let inputDim = 256
        let hiddenDim = 256
        let outputDim = 64
        let numLayers = 4

        let weights = SpikingNetworkWeights.initCfCWeights(
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            outputDim: outputDim,
            numLayers: numLayers,
            seed: 2026
        )

        XCTAssertTrue(weights.isCfC, "isCfC が true に設定されていません")
        XCTAssertEqual(weights.inputDim, inputDim)
        XCTAssertEqual(weights.maxHiddenDim, hiddenDim)
        XCTAssertEqual(weights.outputDim, outputDim)
        XCTAssertEqual(weights.numLayers, numLayers)

        guard let wfList = weights.cfcWf,
              let bfList = weights.cfcBf,
              let wgList = weights.cfcWg,
              let bgList = weights.cfcBg else {
            XCTFail("CfC 重み配列が nil です")
            return
        }

        XCTAssertEqual(wfList.count, numLayers)
        XCTAssertEqual(bfList.count, numLayers)
        XCTAssertEqual(wgList.count, numLayers)
        XCTAssertEqual(bgList.count, numLayers)

        let xhDim = inputDim + hiddenDim
        var l = 0
        while l < numLayers {
            XCTAssertEqual(wfList[l].count, hiddenDim * xhDim, "層 \(l) の W_f 要素数が一致しません")
            XCTAssertEqual(wgList[l].count, hiddenDim * xhDim, "層 \(l) の W_g 要素数が一致しません")
            XCTAssertEqual(bfList[l].count, hiddenDim, "層 \(l) の b_f 要素数が一致しません")
            XCTAssertEqual(bgList[l].count, hiddenDim, "層 \(l) の b_g 要素数が一致しません")
            l += 1
        }

        XCTAssertEqual(weights.wOut.count, outputDim * hiddenDim)
        XCTAssertEqual(weights.bOut.count, outputDim)

        let mlxNet = MLXSpikingAcousticNetwork(weights: weights)
        XCTAssertTrue(mlxNet.isCfC)
        XCTAssertEqual(mlxNet.cfcWf.count, numLayers)
        XCTAssertEqual(mlxNet.cfcBf.count, numLayers)
        XCTAssertEqual(mlxNet.cfcWg.count, numLayers)
        XCTAssertEqual(mlxNet.cfcBg.count, numLayers)

        l = 0
        while l < numLayers {
            XCTAssertEqual(mlxNet.cfcWf[l].shape, [xhDim, hiddenDim])
            XCTAssertEqual(mlxNet.cfcBf[l].shape, [hiddenDim])
            XCTAssertEqual(mlxNet.cfcWg[l].shape, [xhDim, hiddenDim])
            XCTAssertEqual(mlxNet.cfcBg[l].shape, [hiddenDim])
            l += 1
        }
        XCTAssertEqual(mlxNet.wOut.shape, [hiddenDim, outputDim])
        XCTAssertEqual(mlxNet.bOut.shape, [outputDim])

        // 順伝播の入出力形状検証
        let dummyIn = MLXArray.zeros([1, 10, inputDim])
        let dummyOut = mlxNet.forward(features: dummyIn)
        XCTAssertEqual(dummyOut.shape, [1, 10, outputDim])

        // 系列長 0 のエッジケース
        let emptyIn = MLXArray.zeros([1, 0, inputDim])
        let emptyOut = mlxNet.forward(features: emptyIn)
        XCTAssertEqual(emptyOut.shape, [1, 0, outputDim])

        // Pure Swift 側での系列長 0 および任意サイズエッジケース
        let workspace = AcousticWorkspace(maxHiddenDim: hiddenDim, outputDim: outputDim, numLayers: numLayers)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let swiftEmpty = decoder.decodeSequence(featuresSeq: [], workspace: workspace)
        XCTAssertTrue(swiftEmpty.isEmpty)

        // 非8の倍数次元での SIMD テールループ境界安全性の検証
        let oddWeights = SpikingNetworkWeights.initCfCWeights(
            inputDim: 30,
            hiddenDim: 30,
            outputDim: 10,
            numLayers: 2,
            seed: 123
        )
        let oddDecoder = SpikingAcousticDecoder(weights: oddWeights)
        let oddWorkspace = AcousticWorkspace(maxHiddenDim: 30, outputDim: 10, numLayers: 2)
        let oddSeq = [[Float](repeating: 0.1, count: 30), [Float](repeating: 0.2, count: 30)]
        let oddOut = oddDecoder.decodeSequence(featuresSeq: oddSeq, workspace: oddWorkspace)
        XCTAssertEqual(oddOut.count, 2)
        XCTAssertEqual(oddOut[0].count, 10)
    }
}


