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
}
