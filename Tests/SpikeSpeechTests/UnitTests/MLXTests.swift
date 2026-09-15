import XCTest
import MLX
import MLXNN
@testable import SpikeSpeech

/// MLX コアおよび BPTT 学習単体検証テストスイート
final class MLXTests: XCTestCase {

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

        MLXRandom.seed(42)
    }

    override func tearDown() {
        super.tearDown()
    }

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

        print("--- [MLX BPTT Test] Initial Loss: \(initialLoss), Final Loss: \(stepLoss) ---")
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

    // MARK: - 7. 静的コード規約機械検査

    func testMLXStaticRuleCheck() {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let mlxPath = currentDir + "/Sources/SpikeSpeech/MLX"

        guard let enumerator = fileManager.enumerator(atPath: mlxPath) else {
            XCTFail("Sources/SpikeSpeech/MLX が走査できませんでした")
            return
        }

        var checkedFiles = 0
        while let relativePath = enumerator.nextObject() as? String {
            if relativePath.hasSuffix(".swift") != true {
                continue
            }

            let fullPath = mlxPath + "/" + relativePath
            guard let content = try? String(contentsOfFile: fullPath, encoding: .utf8) else {
                continue
            }

            let lines = content.components(separatedBy: .newlines)
            var lineIdx = 0
            while lineIdx < lines.count {
                let line = lines[lineIdx]
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                if trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") {
                    lineIdx += 1
                    continue
                }

                XCTAssertFalse(
                    trimmed.contains(" > ") && trimmed.contains("->") != true,
                    "比較演算子 > が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(" >= "),
                    "比較演算子 >= が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains("else if"),
                    "else if が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(" ? ") && trimmed.contains("??") != true,
                    "三項演算子が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                lineIdx += 1
            }
            checkedFiles += 1
        }

        XCTAssertTrue(0 < checkedFiles, "チェック対象の Swift ファイルがありません")
        print("--- [MLX Static Rule Check] ---")
        print("検証完了 MLX ファイル数: \(checkedFiles) 件 (全ファイル規約適合)")
        print("-------------------------------")
    }

    // MARK: - 8. 学習率スケジューラ・Plateau ガード・シャッフル・チェックポイント検証

    func testCosineWarmupScheduleValues() {
        let schedule = CosineWarmupSchedule(
            lrBase: 0.003,
            lrMin: 1.0e-5,
            warmupEpochs: 2,
            totalEpochs: 15
        )

        // なぜ各エポックの計算値を検証するか:
        // 仕様書で定めた数理テーブルと厳密に一致し、実効学習率の暴走や不連続変化がないことを保証するため
        let expectedEpoch1: Float = 0.001505 // warmup step 1
        let expectedEpoch2: Float = 0.003000 // warmup step 2 (peak)
        let expectedEpoch3: Float = 0.002957 // cosine step
        let expectedEpoch15: Float = 0.000010 // final step (lrMin)

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

        // なぜ初期状態で 1.0 であることを確認するか: 最初は減衰がかかっていないことを保証するため
        XCTAssertTrue(abs(guardState.decayMultiplier - 1.0) < 1.0e-6)

        // 改善時は 1.0 を維持
        let m1 = guardState.observe(epochLoss: 1.50)
        XCTAssertTrue(abs(m1 - 1.0) < 1.0e-6)
        let m2 = guardState.observe(epochLoss: 1.40)
        XCTAssertTrue(abs(m2 - 1.0) < 1.0e-6)

        // 1回目の悪化 (1.40 -> 1.45)
        let m3 = guardState.observe(epochLoss: 1.45)
        XCTAssertTrue(abs(m3 - 1.0) < 1.0e-6, "patience 未満で減衰してはならない")

        // 2回目の悪化 (1.45 -> 1.48): patience=2 到達で 0.5 に減衰
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

        // なぜ決定論性を検証するか: 同じシード・エポックであれば全く同一の並び替えが再現されることを保証するため
        XCTAssertEqual(copy1, copy2, "同一シードでのシャッフル結果が一致しません")
        XCTAssertNotEqual(copy1, copy3, "異なるエポックでのシャッフル結果が変化していません")

        // なぜ要素集合を検証するか: シャッフルによって要素が欠損・重複せず完全に保存されていることを保証するため
        XCTAssertEqual(copy1.sorted(), original, "シャッフルによって要素が損なわれました")
        XCTAssertEqual(copy3.sorted(), original, "シャッフルによって要素が損なわれました")
    }

    func testWeightCheckpointPathAndNaming() {
        let name1 = WeightCheckpoint.epochFileName(epochOneIndexed: 1)
        let name10 = WeightCheckpoint.epochFileName(epochOneIndexed: 10)
        XCTAssertEqual(name1, "weights.ep01.json")
        XCTAssertEqual(name10, "weights.ep10.json")

        let path = WeightCheckpoint.resolvePath(directory: "Models", fileName: name1)
        XCTAssertTrue(path.path.hasSuffix("Models/weights.ep01.json"))
    }

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

        // 学習率の更新と取得の検証
        trainer.setLearningRate(0.0015)
        XCTAssertTrue(abs(trainer.currentLearningRate() - 0.0015) < 1.0e-6, "学習率の動的更新が反映されていません")

        // 重みノルムの取得検証
        let norms = trainer.weightNorms()
        XCTAssertTrue(0.0 < norms.wIn, "wIn ノルムが正数ではありません")
        XCTAssertTrue(0.0 < norms.wRec, "wRec ノルムが正数ではありません")
        XCTAssertTrue(0.0 < norms.wOut, "wOut ノルムが正数ではありません")

        // AdamW による損失減少検証
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
}
