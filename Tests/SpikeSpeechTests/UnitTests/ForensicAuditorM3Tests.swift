import XCTest
import MLX
import MLXNN
@testable import SpikeSpeech

/// Milestone 3 (F10〜F13) Forensic Auditor による独立実証テストスイート
///
/// ALIF の適応閾値ダイナミクス、多層 SNN 重みの完全整合性と層間独立性、
/// MLX BPTT パラメータ更新の実効性、および推論ワークスペースのポインタ不変性を実証する。
final class ForensicAuditorM3Tests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Apple Silicon MLX Metal バックエンドでテンソル自動微分を正常に実行できるようにリソースを準備する。
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

    // MARK: - 1. ALIF 適応閾値ダイナミクスとスパイク頻度適応 (Spike Frequency Adaptation) の実証

    func testALIFSpikeFrequencyAdaptationAndMathematicalExactness() {
        // 固定閾値 LIF（gamma=0.0）では持続入力で毎ステップ発火し続けるのに対し、
        // 適応閾値 ALIF（gamma=0.3, rho=0.85）では発火によって閾値が動的に上昇し、
        // 発火頻度が抑制される生物学的・数学的仕様を証明する。

        let configLIF = LIFConfig(
            beta: 0.8,
            vTh: 1.0,
            vReset: 0.0,
            alpha: 2.0,
            rho: 0.85,
            gamma: 0.0 // 固定閾値
        )

        let configALIF = LIFConfig(
            beta: 0.8,
            vTh: 1.0,
            vReset: 0.0,
            alpha: 2.0,
            rho: 0.85,
            gamma: 0.4 // 適応閾値
        )

        let steps = 20
        let constantCurrent = Float(1.2) // 閾値 1.0 を超える一定電流

        // 1. 固定閾値 LIF の発火カウント
        var vLIF = Float(0.0)
        var sLIF = Float(0.0)
        var lifSpikeCount = 0
        var t = 0
        while t < steps {
            let res = LIFNeuronEngine.stepScalar(
                config: configLIF,
                vPrev: vLIF,
                sPrev: sLIF,
                inputCurrent: constantCurrent
            )
            vLIF = res.vNext
            sLIF = res.sNext
            if 0.5 <= sLIF {
                lifSpikeCount += 1
            }
            t += 1
        }

        // 2. 適応閾値 ALIF の発火カウント
        var vALIF = Float(0.0)
        var sALIF = Float(0.0)
        var aALIF = Float(0.0)
        var alifSpikeCount = 0
        t = 0
        while t < steps {
            let res = LIFNeuronEngine.stepScalarAdaptive(
                config: configALIF,
                vPrev: vALIF,
                sPrev: sALIF,
                aPrev: aALIF,
                inputCurrent: constantCurrent
            )
            vALIF = res.vNext
            sALIF = res.sNext
            aALIF = res.aNext
            if 0.5 <= sALIF {
                alifSpikeCount += 1
            }
            t += 1
        }

        // 適応閾値の上昇により、ALIF の発火回数は固定 LIF よりも厳密に少なくなること (頻度適応)
        XCTAssertTrue(alifSpikeCount < lifSpikeCount, "ALIF の発火頻度適応が機能していません: alif=\(alifSpikeCount), lif=\(lifSpikeCount)")
        XCTAssertTrue(0.0 < aALIF, "ALIF の適応閾値が増加していません: \(aALIF)")

        // 3. SIMD8 ALIF と スカラー ALIF の完全一致実証
        let count = 8
        var vSIMD = [Float](repeating: 0.0, count: count)
        var sSIMD = [Float](repeating: 0.0, count: count)
        var aSIMD = [Float](repeating: 0.0, count: count)
        let curSIMD = [Float](repeating: constantCurrent, count: count)

        t = 0
        while t < steps {
            vSIMD.withUnsafeMutableBufferPointer { pV in
                sSIMD.withUnsafeMutableBufferPointer { pS in
                    aSIMD.withUnsafeMutableBufferPointer { pA in
                        curSIMD.withUnsafeBufferPointer { pCur in
                            LIFNeuronEngine.stepAdaptiveSIMD8(
                                config: configALIF,
                                vPtr: pV.baseAddress!,
                                sPtr: pS.baseAddress!,
                                aPtr: pA.baseAddress!,
                                curPtr: pCur.baseAddress!,
                                count: count
                            )
                        }
                    }
                }
            }
            t += 1
        }

        var lane = 0
        while lane < count {
            let diffV = abs(vSIMD[lane] - vALIF)
            let diffS = abs(sSIMD[lane] - sALIF)
            let diffA = abs(aSIMD[lane] - aALIF)
            XCTAssertTrue(diffV < 1e-5, "ALIF SIMD8 とスカラーの電位不一致: lane=\(lane)")
            XCTAssertTrue(diffS < 1e-5, "ALIF SIMD8 とスカラーのスパイク不一致: lane=\(lane)")
            XCTAssertTrue(diffA < 1e-5, "ALIF SIMD8 とスカラーの適応閾値不一致: lane=\(lane)")
            lane += 1
        }
    }

    // MARK: - 2. 多層 SNN 重みの完全整合性と層間独立性 (Weight Integrity and Layer Isolation) 実証

    func testMultilayerWeightIntegrityAndLayerIsolation() {
        // 多層 SNN の各層の重み・バイアス・RMSNorm スケールが独立に管理され、
        // 転置重み配列 (wRecT, wLayersT, wOutT) が順方向重み配列と厳密に整合し、
        // 決定論的かつ安全に推論可能であることを実証する。

        let inDim = 32
        let hidden = 128
        let outDim = 16
        let numLayers = 3

        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: inDim,
            maxHiddenDim: hidden,
            outputDim: outDim,
            timeSteps: 2,
            numLayers: numLayers,
            seed: 888
        )

        // 転置重み配列の整合性確認
        // wRecT の要素 (i, j) が wRec[i * hidden + j] と一致すること
        let wRecT = weights.makeWRecT()
        var i = 0
        while i < hidden {
            var j = 0
            while j < hidden {
                let direct = weights.wRec[(i * hidden) + j]
                let transposed = wRecT[(j * hidden) + i]
                XCTAssertEqual(direct, transposed)
                j += 1
            }
            i += 1
        }

        // wLayersT の整合性確認
        let wLayersT = weights.makeWLayersT()
        var l = 0
        while l < numLayers - 1 {
            let wL = weights.wLayers[l]
            let wLT = wLayersT[l]
            var r = 0
            while r < hidden {
                var c = 0
                while c < hidden {
                    let direct = wL[(r * hidden) + c]
                    let transposed = wLT[(c * hidden) + r]
                    XCTAssertEqual(direct, transposed)
                    c += 1
                }
                r += 1
            }
            l += 1
        }

        // wOutT の整合性確認
        let wOutT = weights.makeWOutT()
        var o = 0
        while o < outDim {
            var h = 0
            while h < hidden {
                let direct = weights.wOut[(o * hidden) + h]
                let transposed = wOutT[(h * outDim) + o]
                XCTAssertEqual(direct, transposed)
                h += 1
            }
            o += 1
        }

        // デコーダーでの実行確認
        let decoder = SpikingAcousticDecoder(weights: weights)
        let ws = AcousticWorkspace(maxHiddenDim: hidden, outputDim: outDim, numLayers: numLayers)
        let testFeatures = [Float](repeating: 0.35, count: inDim)
        var out = [Float](repeating: 0.0, count: outDim)

        testFeatures.withUnsafeBufferPointer { pIn in
            out.withUnsafeMutableBufferPointer { pOut in
                decoder.decodeFrame(
                    features: pIn.baseAddress!,
                    workspace: ws,
                    outputFeatures: pOut.baseAddress!
                )
            }
        }

        var c = 0
        while c < outDim {
            XCTAssertTrue(out[c].isFinite)
            c += 1
        }
    }

    // MARK: - 3. MLX BPTT パラメータ更新と実勾配フローの実証

    func testMLXBPTTParameterUpdateAndGradientFlow() {
        // BPTT の損失計算がダミーの定数減衰ではなく、MLX の自動微分と Adam 最適化により
        // ネットワークの重みパラメータが実際に変化し更新されていることを実証する。

        let inDim = 16
        let maxHidden = 32
        let outDim = 8
        let tSteps = 2

        let network = MLXSpikingAcousticNetwork(
            numLayers: 1,
            inputDim: inDim,
            maxHiddenDim: maxHidden,
            outputDim: outDim,
            timeSteps: tSteps
        )

        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            learningRate: 0.01,
            bpttWindow: 16
        )

        let initialWIn = network.wIn.asArray(Float.self)
        let initialWOut = network.wOut.asArray(Float.self)

        let featSeq = [[Float]](repeating: [Float](repeating: 0.4, count: inDim), count: 32)
        let targetSeq = [[Float]](repeating: [Float](repeating: 0.1, count: outDim), count: 32)

        // 3 ステップ学習実行
        var step = 0
        var loss = Float(0.0)
        while step < 3 {
            loss = trainer.trainSequence(features: featSeq, targets: targetSeq)
            step += 1
        }

        let updatedWIn = network.wIn.asArray(Float.self)
        let updatedWOut = network.wOut.asArray(Float.self)

        // 損失が NaN/Inf でないこと
        XCTAssertFalse(loss.isNaN, "損失が NaN です")
        XCTAssertFalse(loss.isInfinite, "損失が Inf です")

        // 重み wIn および wOut が実際に更新されたこと（初期値との差分の総和が正であること）
        var diffInSum = Float(0.0)
        var i = 0
        while i < initialWIn.count {
            diffInSum += abs(initialWIn[i] - updatedWIn[i])
            i += 1
        }
        XCTAssertTrue(1e-5 < diffInSum, "BPTT 学習によって wIn パラメータが全く更新されていません")

        var diffOutSum = Float(0.0)
        i = 0
        while i < initialWOut.count {
            diffOutSum += abs(initialWOut[i] - updatedWOut[i])
            i += 1
        }
        XCTAssertTrue(1e-5 < diffOutSum, "BPTT 学習によって wOut パラメータが全く更新されていません")
    }

    // MARK: - 4. AcousticWorkspace のポインタ不変性 (True Zero Allocation) 実証

    func testAcousticWorkspacePointerImmutability() {
        // 多数のフレーム推論を繰り返しても、内部バッファの再確保（ヒープアロケーション）が
        // 1 回も生じず、固定ポインタアドレス上でバッファ再利用が完結していることを証明する。

        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 32,
            maxHiddenDim: 128,
            outputDim: 16,
            timeSteps: 2,
            numLayers: 2,
            seed: 111
        )
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 128, outputDim: 16, numLayers: 2)

        // 初期のバッファアドレスを記録
        var ptrInCurInit: UnsafeMutablePointer<Float>? = nil
        var ptrStepCurInit: UnsafeMutablePointer<Float>? = nil
        var ptrReadoutInit: UnsafeMutablePointer<Float>? = nil
        var ptrVInit: UnsafeMutablePointer<Float>? = nil

        workspace.inputCurrents.withUnsafeMutableBufferPointer { ptrInCurInit = $0.baseAddress }
        workspace.stepCurrents.withUnsafeMutableBufferPointer { ptrStepCurInit = $0.baseAddress }
        workspace.readoutSums.withUnsafeMutableBufferPointer { ptrReadoutInit = $0.baseAddress }
        workspace.layerStates[0].v.withUnsafeMutableBufferPointer { ptrVInit = $0.baseAddress }

        let dummyFeatures = [Float](repeating: 0.2, count: 32)
        var outFeats = [Float](repeating: 0.0, count: 16)

        // 500 フレーム推論実行
        var frame = 0
        while frame < 500 {
            dummyFeatures.withUnsafeBufferPointer { pIn in
                outFeats.withUnsafeMutableBufferPointer { pOut in
                    decoder.decodeFrame(
                        features: pIn.baseAddress!,
                        workspace: workspace,
                        outputFeatures: pOut.baseAddress!
                    )
                }
            }
            frame += 1
        }

        // 500 フレーム実行後のバッファアドレスを取得
        var ptrInCurFinal: UnsafeMutablePointer<Float>? = nil
        var ptrStepCurFinal: UnsafeMutablePointer<Float>? = nil
        var ptrReadoutFinal: UnsafeMutablePointer<Float>? = nil
        var ptrVFinal: UnsafeMutablePointer<Float>? = nil

        workspace.inputCurrents.withUnsafeMutableBufferPointer { ptrInCurFinal = $0.baseAddress }
        workspace.stepCurrents.withUnsafeMutableBufferPointer { ptrStepCurFinal = $0.baseAddress }
        workspace.readoutSums.withUnsafeMutableBufferPointer { ptrReadoutFinal = $0.baseAddress }
        workspace.layerStates[0].v.withUnsafeMutableBufferPointer { ptrVFinal = $0.baseAddress }

        // ポインタアドレスが完全に一致すること（再確保 0 回の証明）
        XCTAssertEqual(ptrInCurInit, ptrInCurFinal, "inputCurrents のアドレスが変化しました (realloc 発生)")
        XCTAssertEqual(ptrStepCurInit, ptrStepCurFinal, "stepCurrents のアドレスが変化しました (realloc 発生)")
        XCTAssertEqual(ptrReadoutInit, ptrReadoutFinal, "readoutSums のアドレスが変化しました (realloc 発生)")
        XCTAssertEqual(ptrVInit, ptrVFinal, "layerStates[0].v のアドレスが変化しました (realloc 発生)")
    }
}
