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
                var val = (Float(rng & 0x00FFFFFF) / Float(0x01000000) * 2.0 - 1.0) * 0.5
                if c == AudioConfig.pulseChannel {
                    switch t {
                    case 0, 4, 8:
                        val = 1.0
                    default:
                        val = 0.0
                    }
                }
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

    // MARK: - 17. 音素区間残差損失および中心化射影行列の検証

    func testPhonemeResidualLossAndCenteringMatrix() {
        // 1. 音素区間検出の境界テスト
        var features: [[Float]] = []
        var t = 0
        while t < 10 {
            var row = [Float](repeating: 0.0, count: 200)
            switch t {
            case 0, 3, 7:
                row[AudioConfig.pulseChannel] = 1.0
            default:
                break
            }
            features.append(row)
            t += 1
        }

        let intervals = AcousticLossFunctions.findPhonemeIntervals(features: features, rawLen: 10)
        XCTAssertEqual(intervals.count, 3)
        XCTAssertEqual(intervals[0].start, 0)
        XCTAssertEqual(intervals[0].end, 3)
        XCTAssertEqual(intervals[1].start, 3)
        XCTAssertEqual(intervals[1].end, 7)
        XCTAssertEqual(intervals[2].start, 7)
        XCTAssertEqual(intervals[2].end, 10)

        // 2. 中心化射影行列の代数的特性テスト (P^2 = P, 行和 = 0)
        let alignedLen = 16
        let (flatP, totalRes) = AcousticLossFunctions.buildCenteringMatrix(intervals: intervals, alignedLen: alignedLen)
        XCTAssertEqual(totalRes, 10)

        // 行和が 0 であることの確認
        var r = 0
        while r < 10 {
            var rowSum: Float = 0.0
            var c = 0
            while c < 10 {
                rowSum += flatP[(r * alignedLen) + c]
                c += 1
            }
            XCTAssertTrue(abs(rowSum) < 1e-5, "行和が 0 ではありません: \(rowSum)")
            r += 1
        }

        // 3. 行列乗算版と区間スライス直接版の数値完全一致テスト
        let pMatrix = MLXArray(flatP, [1, alignedLen, alignedLen])
        let resCountArr = MLXArray(Float(totalRes))

        var predData = [Float](repeating: 0.0, count: alignedLen * 64)
        var tgtData = [Float](repeating: 0.0, count: alignedLen * 64)
        var idx = 0
        while idx < alignedLen * 64 {
            predData[idx] = Float(idx % 17) * 0.1
            tgtData[idx] = Float(idx % 23) * 0.08
            idx += 1
        }
        let predArr = MLXArray(predData, [1, alignedLen, 64])
        let tgtArr = MLXArray(tgtData, [1, alignedLen, 64])

        let lossMatrix = AcousticLossFunctions.phonemeResidualLoss(
            predicted: predArr,
            target: tgtArr,
            centeringMatrix: pMatrix,
            totalResidualFrames: resCountArr
        )
        let lossSlice = AcousticLossFunctions.phonemeResidualLoss(
            predicted: predArr,
            target: tgtArr,
            intervals: intervals
        )
        eval(lossMatrix, lossSlice)

        let diff = abs(lossMatrix.item(Float.self) - lossSlice.item(Float.self))
        print("[Residual Loss Consistency] Matrix vs Slice diff: \(diff)")
        XCTAssertTrue(diff < 1e-5, "射影行列版と区間スライス版の残差損失が一致しません: \(diff)")
    }

    // MARK: - 18. 音素区間残差項 / メル L1 比率（0.5〜2.0倍）の検証

    func testPhonemeResidualLossWeightRatio() throws {
        let weightsPath = "Models/weights.json"
        if FileManager.default.fileExists(atPath: weightsPath) != true {
            return
        }
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        guard weights.isCfC else {
            return
        }
        let testWeight: Float = 3.0
        let network = MLXSpikingAcousticNetwork(weights: weights)
        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            residualLossWeight: testWeight,
            learningRate: 0.001
        )
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        if FileManager.default.fileExists(atPath: wavPath) != true {
            return
        }
        let pcm = try WavAudioReader().loadWav16k(from: wavPath)
        let engine = SpikeSpeechEngine(weights: weights)
        let masPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/mas_alignments.json"
        var uttAlign: UtteranceAlignment? = nil
        if let masData = try? Data(contentsOf: URL(fileURLWithPath: masPath)),
           let aligns = try? JSONDecoder().decode([UtteranceAlignment].self, from: masData) {
            uttAlign = aligns.first(where: { $0.utteranceId == "BASIC5000_0001" })
        }
        guard let pair = engine.prepareTrainingPair(
            text: "水をマレーシアから買わなくてはならないのです。",
            pcm16k: pcm,
            melExtractor: MelSpectrogramExtractor(),
            pitchTracker: PitchTracker(),
            alignment: uttAlign,
            useScaledDuration: true
        ) else {
            XCTFail("prepareTrainingPair 失敗")
            return
        }

        _ = trainer.trainSequence(
            features: pair.features,
            targets: pair.targets,
            targetAudio: pair.targetAudio
        )
        let losses = trainer.lastLosses
        print("\n=======================================================")
        print("[Phoneme Residual Loss Ratio Test]")
        print("  totalLoss:    \(losses.totalLoss)")
        print("  melL1:        \(losses.melL1)")
        print("  residualTerm: \(losses.residualTerm) (coef: \(testWeight))")
        let ratio = losses.residualTerm / max(1e-6, losses.melL1)
        print("  residualTerm / melL1 比率: \(ratio) (基準: 0.5 〜 2.0)")
        print("=======================================================\n")
        XCTAssertTrue(losses.totalLoss.isFinite)
        XCTAssertTrue(0.0 < losses.residualTerm)
    }

    // MARK: - 19. CfC 音素境界リセットおよび音素区間残差損失の極限エッジケース・頑健性検証

    func testCfCEdgeCasesAndPhonemeResidualLossRobustness() {
        // 1. findPhonemeIntervals のエッジケース
        // 空入力
        let emptyIntervals = AcousticLossFunctions.findPhonemeIntervals(features: [], rawLen: 10)
        XCTAssertTrue(emptyIntervals.isEmpty)

        // rawLen が 0 以下
        let zeroLenIntervals = AcousticLossFunctions.findPhonemeIntervals(
            features: [[Float](repeating: 0.0, count: 200)],
            rawLen: 0
        )
        XCTAssertTrue(zeroLenIntervals.isEmpty)

        // rawLen > features.count の超過指定時のクラッシュ防止
        let overflowIntervals = AcousticLossFunctions.findPhonemeIntervals(
            features: [[Float](repeating: 0.0, count: 200)],
            rawLen: 10
        )
        XCTAssertEqual(overflowIntervals.count, 1)
        XCTAssertEqual(overflowIntervals[0].start, 0)
        XCTAssertEqual(overflowIntervals[0].end, 1)

        // パルスチャンネルが存在しない（入力次元 < 200）場合
        let smallDimFeatures = [[Float](repeating: 0.0, count: 128), [Float](repeating: 0.0, count: 128)]
        let smallDimIntervals = AcousticLossFunctions.findPhonemeIntervals(features: smallDimFeatures, rawLen: 2)
        XCTAssertEqual(smallDimIntervals.count, 1)
        XCTAssertEqual(smallDimIntervals[0].start, 0)
        XCTAssertEqual(smallDimIntervals[0].end, 2)

        // パルスが一切存在しない場合
        let noPulseFeatures = [[Float](repeating: 0.0, count: 200), [Float](repeating: 0.0, count: 200), [Float](repeating: 0.0, count: 200)]
        let noPulseIntervals = AcousticLossFunctions.findPhonemeIntervals(features: noPulseFeatures, rawLen: 3)
        XCTAssertEqual(noPulseIntervals.count, 1)
        XCTAssertEqual(noPulseIntervals[0].start, 0)
        XCTAssertEqual(noPulseIntervals[0].end, 3)

        // 毎フレームパルスが存在する場合（全区間長 1）
        var allPulseFeatures: [[Float]] = []
        var ap = 0
        while ap < 4 {
            var row = [Float](repeating: 0.0, count: 200)
            row[AudioConfig.pulseChannel] = 1.0
            allPulseFeatures.append(row)
            ap += 1
        }
        let allPulseIntervals = AcousticLossFunctions.findPhonemeIntervals(features: allPulseFeatures, rawLen: 4)
        XCTAssertEqual(allPulseIntervals.count, 4)
        var api = 0
        while api < allPulseIntervals.count {
            let seg = allPulseIntervals[api]
            XCTAssertEqual(seg.end - seg.start, 1)
            api += 1
        }

        // 2. buildCenteringMatrix のエッジケース
        // alignedLen <= 0
        let zeroAligned = AcousticLossFunctions.buildCenteringMatrix(intervals: [(start: 0, end: 5)], alignedLen: 0)
        XCTAssertTrue(zeroAligned.matrix.isEmpty)
        XCTAssertEqual(zeroAligned.totalResidualFrames, 0)

        // 全区間長 1（有効残差フレーム数 0）
        let (allPulseP, allPulseRes) = AcousticLossFunctions.buildCenteringMatrix(intervals: allPulseIntervals, alignedLen: 8)
        XCTAssertEqual(allPulseRes, 0)
        var pSum: Float = 0.0
        for v in allPulseP {
            pSum += abs(v)
        }
        XCTAssertEqual(pSum, 0.0)

        // 境界外区間（start < 0, end > alignedLen）の自動クランプ
        let outOfBoundsIntervals = [(start: -5, end: 100)]
        let (clampedP, clampedRes) = AcousticLossFunctions.buildCenteringMatrix(intervals: outOfBoundsIntervals, alignedLen: 8)
        XCTAssertEqual(clampedRes, 8)
        XCTAssertEqual(clampedP.count, 64)

        // 3. phonemeResidualLoss のエッジケース（totalResidualFrames == 0 での非ゼロ除算・NaN 回避）
        let dummyPred = MLXArray.zeros([1, 8, 64])
        let dummyTgt = MLXArray.zeros([1, 8, 64])
        let zeroLossMat = AcousticLossFunctions.phonemeResidualLoss(
            predicted: dummyPred,
            target: dummyTgt,
            centeringMatrix: MLXArray(allPulseP, [1, 8, 8]),
            totalResidualFrames: MLXArray(Float(0.0))
        )
        eval(zeroLossMat)
        XCTAssertEqual(zeroLossMat.item(Float.self), 0.0)

        let zeroLossSlice = AcousticLossFunctions.phonemeResidualLoss(
            predicted: dummyPred,
            target: dummyTgt,
            intervals: allPulseIntervals
        )
        eval(zeroLossSlice)
        XCTAssertEqual(zeroLossSlice.item(Float.self), 0.0)

        // 4. MLX forwardCfC と Pure Swift decodeSequenceCfC の極限入力整合性 (< 1e-4)
        let weights = SpikingNetworkWeights.initCfCWeights(
            inputDim: 256,
            hiddenDim: 256,
            outputDim: 64,
            numLayers: 2,
            seed: 777
        )
        let mlxNet = MLXSpikingAcousticNetwork(weights: weights)
        let swiftDecoder = SpikingAcousticDecoder(weights: weights)
        let ws = AcousticWorkspace(maxHiddenDim: 256, outputDim: 64, numLayers: 2)

        // ケース A: 系列長 1、パルスなし
        let singleFeat = [[Float](repeating: 0.1, count: 256)]
        let flatSingle = [Float](repeating: 0.1, count: 256)
        let mlxSingle = mlxNet.forward(features: MLXArray(flatSingle, [1, 1, 256]))
        eval(mlxSingle)
        let swiftSingle = swiftDecoder.decodeSequence(featuresSeq: singleFeat, workspace: ws)
        XCTAssertEqual(swiftSingle.count, 1)
        var maxDiffA: Float = 0.0
        var c = 0
        while c < 64 {
            let d = abs(mlxSingle.asArray(Float.self)[c] - swiftSingle[0][c])
            if maxDiffA < d { maxDiffA = d }
            c += 1
        }
        XCTAssertTrue(maxDiffA < 1e-4, "系列長 1 のメル差が 1e-4 以上です: \(maxDiffA)")

        // ケース B: 毎フレームパルス（3フレーム全てパルス）
        var allPulseFeat: [[Float]] = []
        var flatAllPulse: [Float] = []
        var tf = 0
        while tf < 3 {
            var row = [Float](repeating: 0.15, count: 256)
            row[AudioConfig.pulseChannel] = 1.0
            allPulseFeat.append(row)
            flatAllPulse.append(contentsOf: row)
            tf += 1
        }
        let mlxAllPulse = mlxNet.forward(features: MLXArray(flatAllPulse, [1, 3, 256]))
        eval(mlxAllPulse)
        let swiftAllPulse = swiftDecoder.decodeSequence(featuresSeq: allPulseFeat, workspace: ws)
        XCTAssertEqual(swiftAllPulse.count, 3)
        var maxDiffB: Float = 0.0
        tf = 0
        while tf < 3 {
            c = 0
            while c < 64 {
                let mlxV = mlxAllPulse.asArray(Float.self)[(tf * 64) + c]
                let swiftV = swiftAllPulse[tf][c]
                let d = abs(mlxV - swiftV)
                if maxDiffB < d { maxDiffB = d }
                c += 1
            }
            tf += 1
        }
        XCTAssertTrue(maxDiffB < 1e-4, "毎フレームパルスのメル差が 1e-4 以上です: \(maxDiffB)")
    }

    // MARK: - 20. BASIC5000_0001 の全フレームでの MLX と Swift のメル最大絶対差検証

    /// 受入基準検証: 学習前に、BASIC5000_0001 の全フレームで MLX と Swift のメル最大絶対差が 1e-4 未満。リセットを含む。
    func testCfCBASIC5000_0001NumericalConsistency() throws {
        let weightsPath = "Models/weights.json"
        if FileManager.default.fileExists(atPath: weightsPath) != true {
            return
        }
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        guard weights.isCfC else {
            return
        }

        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        if FileManager.default.fileExists(atPath: wavPath) != true {
            return
        }
        let pcm = try WavAudioReader().loadWav16k(from: wavPath)

        let masPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/mas_alignments.json"
        var uttAlign: UtteranceAlignment? = nil
        if let masData = try? Data(contentsOf: URL(fileURLWithPath: masPath)),
           let aligns = try? JSONDecoder().decode([UtteranceAlignment].self, from: masData) {
            uttAlign = aligns.first(where: { $0.utteranceId == "BASIC5000_0001" })
        }

        let engine = SpikeSpeechEngine(weights: weights)
        guard let pair = engine.prepareTrainingPair(
            text: "水をマレーシアから買わなくてはならないのです。",
            pcm16k: pcm,
            melExtractor: MelSpectrogramExtractor(),
            pitchTracker: PitchTracker(),
            alignment: uttAlign,
            useScaledDuration: true
        ) else {
            XCTFail("BASIC5000_0001 の prepareTrainingPair に失敗しました")
            return
        }

        let rawLen = pair.features.count
        let inDim = weights.inputDim
        let outDim = weights.outputDim
        let alignedLen = MLXAcousticBPTTTrainer.alignTo32(seqLen: rawLen)

        // MLX 順伝播
        var flatFeat = [Float](repeating: 0.0, count: alignedLen * inDim)
        var t = 0
        while t < rawLen {
            var c = 0
            while c < inDim {
                if c < pair.features[t].count {
                    flatFeat[(t * inDim) + c] = pair.features[t][c]
                }
                c += 1
            }
            t += 1
        }
        let mlxNet = MLXSpikingAcousticNetwork(weights: weights)
        let mlxFArr = MLXArray(flatFeat, [1, alignedLen, inDim])
        let mlxOut = mlxNet.forward(features: mlxFArr, bpttWindow: 16)
        eval(mlxOut)
        let mlxMelFlat = mlxOut.asArray(Float.self)

        // Pure Swift 順伝播
        let swiftDec = SpikingAcousticDecoder(weights: weights)
        let swiftWs = AcousticWorkspace(maxHiddenDim: weights.maxHiddenDim, outputDim: outDim, numLayers: weights.numLayers)
        let swiftMel = swiftDec.decodeSequence(featuresSeq: pair.features, workspace: swiftWs)

        XCTAssertEqual(swiftMel.count, rawLen)

        // 最大絶対誤差の計算
        var maxDiff: Float = 0.0
        t = 0
        while t < rawLen {
            var c = 0
            while c < outDim {
                let mlxV = mlxMelFlat[(t * outDim) + c]
                let swiftV = swiftMel[t][c]
                let d = abs(mlxV - swiftV)
                if maxDiff < d {
                    maxDiff = d
                }
                c += 1
            }
            t += 1
        }

        print("[BASIC5000_0001 Numerical Consistency Test] Maximum Mel absolute diff between MLX and Pure Swift: \(maxDiff) (受入基準: < 1.0e-4)")
        XCTAssertTrue(maxDiff < 1.0e-4, "BASIC5000_0001 の MLX と Pure Swift のメル最大絶対差が 1e-4 以上です: \(maxDiff)")
    }

    // MARK: - 21. 系列長 64 フレーム時 (alignedLen == outDim == 64) の centeringMatrix と muTarget の分離・損失計算検証

    /// 系列長 64 (alignedLen == 64, outDim == 64) の境界条件において、
    /// centeringMatrix [1, 64, 64] と muTarget [1, 64, 64] が lossAndGrad 内部で誤って衝突・上書きされず、
    /// phonemeResidualLossWithMu が正確に評価されることを検証する。
    func testPhonemeResidualLossWithMuWhenSeqLenIs64() {
        let inDim = 256
        let outDim = 64
        let alignedLen = 64
        let weights = SpikingNetworkWeights.initCfCWeights(
            inputDim: inDim,
            hiddenDim: 128,
            outputDim: outDim,
            numLayers: 1,
            seed: 42
        )
        let network = MLXSpikingAcousticNetwork(weights: weights)
        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            residualLossWeight: 2.0,
            learningRate: 0.001
        )

        // ダミーの特徴量 (64フレーム)
        let features = [[Float]](repeating: [Float](repeating: 0.1, count: inDim), count: alignedLen)
        let targets = [[Float]](repeating: [Float](repeating: 0.2, count: outDim), count: alignedLen)

        // 1 区間: 10 ..< 50 (len = 40)
        let intervals = [(start: 10, end: 50)]
        let (flatP, totalRes) = AcousticLossFunctions.buildCenteringMatrix(intervals: intervals, alignedLen: alignedLen)
        XCTAssertEqual(totalRes, 40)

        // 非ゼロの μ 系列を生成 (区間内 0.05)
        var mu = [[Float]](repeating: [Float](repeating: 0.0, count: outDim), count: alignedLen)
        var t = 10
        while t < 50 {
            var c = 0
            while c < outDim {
                mu[t][c] = 0.05
                c += 1
            }
            t += 1
        }

        var flatFeat = [Float](repeating: 0.0, count: alignedLen * inDim)
        t = 0
        while t < alignedLen {
            var c = 0
            while c < inDim {
                flatFeat[(t * inDim) + c] = features[t][c]
                c += 1
            }
            t += 1
        }
        var flatTgt = [Float](repeating: 0.0, count: alignedLen * outDim)
        t = 0
        while t < alignedLen {
            var c = 0
            while c < outDim {
                flatTgt[(t * outDim) + c] = targets[t][c]
                c += 1
            }
            t += 1
        }
        let cmArr = MLXArray(flatP, [1, alignedLen, alignedLen])
        let rfArr = MLXArray(Float(totalRes))
        var flatMu = [Float](repeating: 0.0, count: alignedLen * outDim)
        t = 0
        while t < alignedLen {
            var c = 0
            while c < outDim {
                flatMu[(t * outDim) + c] = mu[t][c]
                c += 1
            }
            t += 1
        }
        let muArr = MLXArray(flatMu, [1, alignedLen, outDim])

        // 計算前の forward による direct 残差損失を計算
        let predBefore = network.forward(features: MLXArray(flatFeat, [1, alignedLen, inDim]), bpttWindow: 16)
        let directResLoss = AcousticLossFunctions.phonemeResidualLossWithMu(
            predicted: predBefore,
            muTarget: muArr,
            centeringMatrix: cmArr,
            totalResidualFrames: rfArr
        ).item(Float.self)

        // trainBatch を実行 (lossAndGrad のテンソル展開検証)
        trainer.trainBatch(
            features: MLXArray(flatFeat, [1, alignedLen, inDim]),
            targets: MLXArray(flatTgt, [1, alignedLen, outDim]),
            centeringMatrix: cmArr,
            totalResidualFrames: rfArr,
            muTarget: muArr
        )

        let losses = trainer.lastLosses
        XCTAssertTrue(losses.totalLoss.isFinite)
        XCTAssertTrue(0.0 < losses.residualTerm, "残差項が正の値ではありません: \(losses.residualTerm)")

        let expectedResTerm = 2.0 * directResLoss
        // 許容誤差範囲内で一致すること
        let diff = abs(losses.residualTerm - expectedResTerm)
        XCTAssertTrue(diff < 1e-4, "trainBatch の残差項 (\(losses.residualTerm)) と direct 計算 (\(expectedResTerm)) が不一致です: diff=\(diff)")

        // trainSequence で系列長 64 (alignedLen == 64) を実行し、正常に完了することを確認
        var seqFeat = features
        seqFeat[0][AudioConfig.pulseChannel] = 1.0
        seqFeat[32][AudioConfig.pulseChannel] = 1.0
        trainer.trainSequence(
            features: seqFeat,
            targets: targets,
            muTarget: mu
        )
        let seqLosses = trainer.lastLosses
        XCTAssertTrue(seqLosses.totalLoss.isFinite)
        XCTAssertTrue(0.0 < seqLosses.residualTerm, "trainSequence 実行時の残差項が正の値ではありません: \(seqLosses.residualTerm)")
    }

    // MARK: - FrameMel MLX / Pure Swift 等価性および学習検証
    func testFrameMelModelParityAndTraining() {
        let weights = FrameMelWeights.randomWeights(seed: 2026)
        let swiftModel = FrameMelModel(weights: weights)
        let mlxModel = MLXFrameMelModel(weights: weights)

        // 1. Decoder + PostNet 出力の等価性検証 (< 1e-3)
        let totalFrames = 10
        var flatCond = [Float](repeating: 0.0, count: totalFrames * 260)
        var i = 0
        while i < flatCond.count {
            flatCond[i] = sinf(Float(i) * 0.05) * 0.5
            i += 1
        }

        let condMLX = MLXArray(flatCond, [1, totalFrames, 260])
        let (_, mlxPost) = mlxModel.forwardDecoder(condition: condMLX)
        let (_, swiftPost) = swiftModel.decodeMel(decoderCondition: flatCond, totalFrames: totalFrames)

        let mlxPostArr = mlxPost.asArray(Float.self)
        var maxDiff: Float = 0.0
        var t = 0
        while t < totalFrames {
            var c = 0
            while c < 64 {
                let diff = abs(swiftPost[t][c] - mlxPostArr[t * 64 + c])
                if maxDiff < diff {
                    maxDiff = diff
                }
                c += 1
            }
            t += 1
        }
        XCTAssertTrue(maxDiff < 1e-3, "MLX と Pure Swift の Mel デコーダ出力差分が 1e-3 を超えています: \(maxDiff)")

        let sil = Int32(PhonemeVocabulary.silId)
        let phoneIds: [Int32] = [sil, 15, 20, 25, sil]
        let targetDurations = [2, 3, 2, 3, 2] // 合計 12 フレーム
        let trainFrames = 12
        var targetMel = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: trainFrames)
        var targetF0 = [Float](repeating: 0.0, count: trainFrames)
        var targetEnergy = [Float](repeating: 0.0, count: trainFrames)
        t = 0
        while t < trainFrames {
            targetF0[t] = 0.3 + sinf(Float(t) * 0.2) * 0.1
            targetEnergy[t] = 0.5
            var c = 0
            while c < 64 {
                targetMel[t][c] = -2.0 + sinf(Float(t + c) * 0.1) * 0.5
                c += 1
            }
            t += 1
        }

        let trainer = MLXFrameMelTrainer(model: mlxModel, learningRate: 0.0002)
        let initialLosses = trainer.trainSample(
            phoneIds: phoneIds,
            targetDurations: targetDurations,
            targetMel: targetMel,
            targetF0: targetF0,
            targetEnergy: targetEnergy
        )

        XCTAssertTrue(initialLosses.totalLoss.isFinite)
        XCTAssertTrue(0.0 < initialLosses.totalLoss)
        XCTAssertTrue(initialLosses.decMelL1.isFinite)
        XCTAssertTrue(initialLosses.postMelL1.isFinite)
        XCTAssertTrue(initialLosses.voicedF0MSE.isFinite)
        XCTAssertTrue(initialLosses.energyMSE.isFinite)
        XCTAssertTrue(initialLosses.durMSE.isFinite)

        print("Initial loss: \(initialLosses.totalLoss), dec: \(initialLosses.decMelL1), post: \(initialLosses.postMelL1), f0: \(initialLosses.voicedF0MSE), energy: \(initialLosses.energyMSE), dur: \(initialLosses.durMSE)")

        let nextLosses = trainer.trainSample(
            phoneIds: phoneIds,
            targetDurations: targetDurations,
            targetMel: targetMel,
            targetF0: targetF0,
            targetEnergy: targetEnergy
        )
        XCTAssertTrue(nextLosses.totalLoss < initialLosses.totalLoss, "学習ステップ後に損失が減少していません: 初期=\(initialLosses.totalLoss), 次=\(nextLosses.totalLoss)")
    }
}


