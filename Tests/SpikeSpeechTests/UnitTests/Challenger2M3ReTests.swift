import XCTest
import MLX
import MLXNN
@testable import SpikeSpeech

/// 多層 SNN における MLX BPTT 最適化収束性および連続推論時 O(1) アロケーション実証テストスイート
final class Challenger2M3ReTests: XCTestCase {

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

        MLXRandom.seed(1234)
    }

    // MARK: - 1. 多層 SNN に対する MLX BPTT 学習ループの損失減少およびパラメータ更新実証

    func testMLXBPTTLossMonotonicDecreaseAndParamUpdate() {
        let inDim = 16
        let maxHidden = 32
        let outDim = 16
        let tSteps = 2

        let network = MLXSpikingAcousticNetwork(
            numLayers: 2,
            inputDim: inDim,
            maxHiddenDim: maxHidden,
            outputDim: outDim,
            timeSteps: tSteps,
            lifConfig: LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.05)
        )

        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            learningRate: 0.01,
            bpttWindow: 16
        )

        let initialWIn = network.wIn.asArray(Float.self)
        let initialWRec = network.wRec.asArray(Float.self)
        let initialWLayer0 = network.wLayers[0].asArray(Float.self)
        let initialWOut = network.wOut.asArray(Float.self)

        let seqLen = 32
        let featSeq = [[Float]](repeating: [Float](repeating: 0.4, count: inDim), count: seqLen)
        let targetSeq = [[Float]](repeating: [Float](repeating: 0.25, count: outDim), count: seqLen)

        var lossHistory: [Float] = []
        var step = 0
        while step < 15 {
            let stepLoss = trainer.trainSequence(features: featSeq, targets: targetSeq)
            lossHistory.append(stepLoss)
            step += 1
        }

        let firstLoss = lossHistory[0]
        let lastLoss = lossHistory[lossHistory.count - 1]

        XCTAssertFalse(firstLoss.isNaN, "初期損失が NaN です")
        XCTAssertFalse(lastLoss.isNaN, "最終損失が NaN です")
        XCTAssertTrue(lastLoss < firstLoss, "BPTT 学習によって損失が減少していません: 初期=\(firstLoss), 最終=\(lastLoss)")

        // パラメータが実際に更新されていることを検証する
        let updatedWIn = network.wIn.asArray(Float.self)
        let updatedWRec = network.wRec.asArray(Float.self)
        let updatedWLayer0 = network.wLayers[0].asArray(Float.self)
        let updatedWOut = network.wOut.asArray(Float.self)

        var diffWIn: Float = 0.0
        var i = 0
        while i < updatedWIn.count {
            diffWIn += abs(updatedWIn[i] - initialWIn[i])
            i += 1
        }

        var diffWRec: Float = 0.0
        i = 0
        while i < updatedWRec.count {
            diffWRec += abs(updatedWRec[i] - initialWRec[i])
            i += 1
        }

        var diffWLayer0: Float = 0.0
        i = 0
        while i < updatedWLayer0.count {
            diffWLayer0 += abs(updatedWLayer0[i] - initialWLayer0[i])
            i += 1
        }

        var diffWOut: Float = 0.0
        i = 0
        while i < updatedWOut.count {
            diffWOut += abs(updatedWOut[i] - initialWOut[i])
            i += 1
        }

        XCTAssertTrue(0.0 < diffWIn, "wIn パラメータが更新されていません")
        XCTAssertTrue(0.0 < diffWRec, "wRec パラメータが更新されていません")
        XCTAssertTrue(0.0 < diffWLayer0, "wLayers[0] パラメータが更新されていません")
        XCTAssertTrue(0.0 < diffWOut, "wOut パラメータが更新されていません")
    }

    // MARK: - 2. AcousticWorkspace を用いた 1,000 フレーム連続推論時 O(1) アロケーション実証

    func testAcousticWorkspace1000FramesO1Allocation() {
        let inDim = 16
        let maxHidden = 64
        let outDim = 16
        let numLayers = 3
        let tSteps = 4

        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: inDim,
            maxHiddenDim: maxHidden,
            outputDim: outDim,
            timeSteps: tSteps,
            numLayers: numLayers,
            seed: 777
        )

        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: maxHidden, outputDim: outDim, numLayers: numLayers)

        // 事前確保された各内部配列のキャパシティを記録する
        let capInput = workspace.inputCurrents.capacity
        let capStep = workspace.stepCurrents.capacity
        let capPrev = workspace.stepCurrentsPrev.capacity
        let capSpikes = workspace.activeSpikes.capacity
        let capLayerSpikes = workspace.activeLayerSpikes.capacity
        let capReadoutIndices = workspace.activeReadoutIndices.capacity
        let capRates = workspace.activeRates.capacity
        let capReadoutSums = workspace.readoutSums.capacity
        let capOutput = workspace.outputFeatures.capacity

        var layerVCapacities: [Int] = []
        var layerSCapacities: [Int] = []
        var layerACapacities: [Int] = []
        var l = 0
        while l < numLayers {
            layerVCapacities.append(workspace.layerStates[l].v.capacity)
            layerSCapacities.append(workspace.layerStates[l].s.capacity)
            layerACapacities.append(workspace.layerStates[l].a.capacity)
            l += 1
        }

        let totalFrames = 1000
        var dummyFeatures = [Float](repeating: 0.3, count: inDim)
        var frameOutput = [Float](repeating: 0.0, count: outDim)

        workspace.reset()

        var f = 0
        while f < totalFrames {
            dummyFeatures[0] = sin(Float(f) * 0.05)

            dummyFeatures.withUnsafeBufferPointer { pIn in
                frameOutput.withUnsafeMutableBufferPointer { pOut in
                    decoder.decodeFrame(
                        features: pIn.baseAddress!,
                        workspace: workspace,
                        outputFeatures: pOut.baseAddress!
                    )
                }
            }

            // 出力に NaN または Inf が混入していないことを検証する
            var o = 0
            while o < outDim {
                let val = frameOutput[o]
                XCTAssertFalse(val.isNaN, "フレーム \(f) の出力に NaN が検出されました: index=\(o)")
                XCTAssertFalse(val.isInfinite, "フレーム \(f) の出力に Inf が検出されました: index=\(o)")
                o += 1
            }

            f += 1
        }

        // 1,000 フレーム連続実行後も配列のキャパシティが変動せず再確保ゼロを維持していることを検証する
        XCTAssertEqual(workspace.inputCurrents.capacity, capInput, "inputCurrents が再確保されました")
        XCTAssertEqual(workspace.stepCurrents.capacity, capStep, "stepCurrents が再確保されました")
        XCTAssertEqual(workspace.stepCurrentsPrev.capacity, capPrev, "stepCurrentsPrev が再確保されました")
        XCTAssertEqual(workspace.activeSpikes.capacity, capSpikes, "activeSpikes が再確保されました")
        XCTAssertEqual(workspace.activeLayerSpikes.capacity, capLayerSpikes, "activeLayerSpikes が再確保されました")
        XCTAssertEqual(workspace.activeReadoutIndices.capacity, capReadoutIndices, "activeReadoutIndices が再確保されました")
        XCTAssertEqual(workspace.activeRates.capacity, capRates, "activeRates が再確保されました")
        XCTAssertEqual(workspace.readoutSums.capacity, capReadoutSums, "readoutSums が再確保されました")
        XCTAssertEqual(workspace.outputFeatures.capacity, capOutput, "outputFeatures が再確保されました")

        l = 0
        while l < numLayers {
            XCTAssertEqual(workspace.layerStates[l].v.capacity, layerVCapacities[l], "層 \(l) の v が再確保されました")
            XCTAssertEqual(workspace.layerStates[l].s.capacity, layerSCapacities[l], "層 \(l) の s が再確保されました")
            XCTAssertEqual(workspace.layerStates[l].a.capacity, layerACapacities[l], "層 \(l) の a が再確保されました")
            l += 1
        }
    }
}
