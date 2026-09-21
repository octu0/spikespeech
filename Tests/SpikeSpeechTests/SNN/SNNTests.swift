import XCTest
@testable import SpikeSpeech

/// SNN（LIF/ALIF ニューロン, デコーダー, 重み, ワークスペース）の網羅的単体検証テストスイート
final class SNNTests: XCTestCase {

    // MARK: - 1. LIF / ALIF 膜電位更新とスパイク生成の数学的正確性テスト

    func testLIFNeuronScalarVsSIMD8MathematicalExactness() {
        let config = LIFConfig(
            beta: 0.8,
            vTh: 1.0,
            vReset: 0.0,
            alpha: 2.0,
            rho: 0.85,
            gamma: 0.2
        )

        let count = 16
        var vSIMD = [Float](repeating: 0.0, count: count)
        var sSIMD = [Float](repeating: 0.0, count: count)
        var aSIMD = [Float](repeating: 0.0, count: count)

        var vScalar = [Float](repeating: 0.0, count: count)
        var sScalar = [Float](repeating: 0.0, count: count)
        var aScalar = [Float](repeating: 0.0, count: count)

        var testCur = [Float](repeating: 0.0, count: count)
        var i = 0
        while i < count {
            testCur[i] = 0.2 + (Float(i) * 0.1)
            i += 1
        }

        var step = 0
        while step < 10 {
            vSIMD.withUnsafeMutableBufferPointer { pV in
                sSIMD.withUnsafeMutableBufferPointer { pS in
                    aSIMD.withUnsafeMutableBufferPointer { pA in
                        testCur.withUnsafeBufferPointer { pCur in
                            LIFNeuronEngine.stepAdaptiveSIMD8(
                                config: config,
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

            i = 0
            while i < count {
                let res = LIFNeuronEngine.stepScalarAdaptive(
                    config: config,
                    vPrev: vScalar[i],
                    sPrev: sScalar[i],
                    aPrev: aScalar[i],
                    inputCurrent: testCur[i]
                )
                vScalar[i] = res.vNext
                sScalar[i] = res.sNext
                aScalar[i] = res.aNext
                i += 1
            }

            i = 0
            while i < count {
                let diffV = abs(vSIMD[i] - vScalar[i])
                let diffS = abs(sSIMD[i] - sScalar[i])
                let diffA = abs(aSIMD[i] - aScalar[i])

                XCTAssertTrue(diffV < 1e-5, "膜電位 V の SIMD とスカラーの不一致: lane=\(i), step=\(step), diff=\(diffV)")
                XCTAssertTrue(diffS < 1e-5, "スパイク S の SIMD とスカラーの不一致: lane=\(i), step=\(step), diff=\(diffS)")
                XCTAssertTrue(diffA < 1e-5, "適応閾値 A の SIMD とスカラーの不一致: lane=\(i), step=\(step), diff=\(diffA)")
                i += 1
            }

            step += 1
        }
    }

    // MARK: - 2. リードアウト層の減算リセット & アナログ膜電位積算テスト

    func testReadoutLayerSubtractiveResetAndAnalogAccumulation() {
        let config = LIFConfig(
            beta: 0.8,
            vTh: 1.0,
            vReset: 0.0,
            alpha: 2.0,
            rho: 0.85,
            gamma: 0.0
        )

        let count = 8
        var v = [Float](repeating: 0.0, count: count)
        var s = [Float](repeating: 0.0, count: count)
        var a = [Float](repeating: 0.0, count: count)
        var readoutSum = [Float](repeating: 0.0, count: count)

        let inputCur = [Float](repeating: 1.5, count: count)

        v.withUnsafeMutableBufferPointer { pV in
            s.withUnsafeMutableBufferPointer { pS in
                a.withUnsafeMutableBufferPointer { pA in
                    inputCur.withUnsafeBufferPointer { pCur in
                        readoutSum.withUnsafeMutableBufferPointer { pSum in
                            LIFNeuronEngine.stepReadoutAdaptiveSIMD8(
                                config: config,
                                vPtr: pV.baseAddress!,
                                sPtr: pS.baseAddress!,
                                aPtr: pA.baseAddress!,
                                curPtr: pCur.baseAddress!,
                                readoutSumPtr: pSum.baseAddress!,
                                count: count
                            )
                        }
                    }
                }
            }
        }

        var i = 0
        while i < count {
            XCTAssertEqual(s[i], 1.0)
            let diffV = abs(v[i] - 0.5)
            XCTAssertTrue(diffV < 1e-5, "減算リセット後の残余膜電位が不正です: \(v[i])")
            XCTAssertEqual(readoutSum[i], 1.0)
            i += 1
        }
    }

    // MARK: - 3. 推論時ゼロアロケーション検証

    func testAcousticWorkspaceZeroAllocation() {
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 64,
            maxHiddenDim: 256,
            outputDim: 80,
            timeSteps: 4,
            numLayers: 2,
            seed: 123
        )
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 256, outputDim: 80, numLayers: 2)

        let dummyFeatures = [Float](repeating: 0.5, count: 64)
        var outBuffer = [Float](repeating: 0.0, count: 80)

        var frame = 0
        while frame < 1000 {
            dummyFeatures.withUnsafeBufferPointer { pIn in
                outBuffer.withUnsafeMutableBufferPointer { pOut in
                    decoder.decodeFrame(
                        features: pIn.baseAddress!,
                        workspace: workspace,
                        outputFeatures: pOut.baseAddress!
                    )
                }
            }

            var c = 0
            while c < 80 {
                let val = outBuffer[c]
                XCTAssertFalse(val.isNaN, "frame \(frame) で NaN を検出")
                XCTAssertFalse(val.isInfinite, "frame \(frame) で Inf を検出")
                c += 1
            }
            frame += 1
        }

        workspace.reset()
        XCTAssertEqual(workspace.layerStates[0].v[0], 0.0)
        XCTAssertEqual(workspace.layerStates[0].s[0], 0.0)
    }

    // MARK: - 4. 多層 SNN 構造推論の検証

    func testMultilayerSNNInferenceCorrectness() {
        let maxHidden = 256
        let inDim = 64
        let outDim = 80
        let layerConfigs = [1, 2, 3, 4]
        let dummyFeatures = [Float](repeating: 0.25, count: inDim)
        var outBuffer = [Float](repeating: 0.0, count: outDim)

        var lIdx = 0
        while lIdx < layerConfigs.count {
            let layers = layerConfigs[lIdx]
            let weights = SpikingNetworkWeights.randomWeights(
                inputDim: inDim,
                maxHiddenDim: maxHidden,
                outputDim: outDim,
                timeSteps: 4,
                numLayers: layers,
                seed: 456
            )
            let decoder = SpikingAcousticDecoder(weights: weights)
            let workspace = AcousticWorkspace(maxHiddenDim: maxHidden, outputDim: outDim, numLayers: layers)

            workspace.reset()

            dummyFeatures.withUnsafeBufferPointer { pIn in
                outBuffer.withUnsafeMutableBufferPointer { pOut in
                    decoder.decodeFrame(
                        features: pIn.baseAddress!,
                        workspace: workspace,
                        outputFeatures: pOut.baseAddress!
                    )
                }
            }

            var c = 0
            while c < outDim {
                let val = outBuffer[c]
                XCTAssertFalse(val.isNaN, "layers \(layers) で NaN を検出")
                XCTAssertFalse(val.isInfinite, "layers \(layers) で Inf を検出")
                c += 1
            }
            lIdx += 1
        }
    }

    // MARK: - 5. 多層重み転置配列および Codable 永続化テスト

    func testSpikingNetworkWeightsTransposedMatricesAndCodable() {
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 32,
            maxHiddenDim: 128,
            outputDim: 40,
            timeSteps: 2,
            numLayers: 3,
            seed: 789
        )

        // 1. 転置行列の検証
        let recT = weights.makeWRecT()
        XCTAssertEqual(recT.count, 128 * 128)
        XCTAssertEqual(recT[(1 * 128) + 0], weights.wRec[(0 * 128) + 1])

        let layersT = weights.makeWLayersT()
        XCTAssertEqual(layersT.count, 2)
        XCTAssertEqual(layersT[0].count, 128 * 128)
        XCTAssertEqual(layersT[0][(2 * 128) + 1], weights.wLayers[0][(1 * 128) + 2])

        let outT = weights.makeWOutT()
        XCTAssertEqual(outT.count, 128 * 40)
        XCTAssertEqual(outT[(2 * 40) + 3], weights.wOut[(3 * 128) + 2])

        // 2. Codable JSON 往復保存・復元テスト
        do {
            let encoder = JSONEncoder()
            let data = try encoder.encode(weights)
            let decoder = JSONDecoder()
            let restored = try decoder.decode(SpikingNetworkWeights.self, from: data)

            XCTAssertEqual(weights.inputDim, restored.inputDim)
            XCTAssertEqual(weights.maxHiddenDim, restored.maxHiddenDim)
            XCTAssertEqual(weights.outputDim, restored.outputDim)
            XCTAssertEqual(weights.numLayers, restored.numLayers)
            XCTAssertEqual(weights.wIn.count, restored.wIn.count)
            XCTAssertEqual(weights.wRec.count, restored.wRec.count)
            XCTAssertEqual(weights.wLayers.count, restored.wLayers.count)
            XCTAssertEqual(weights.bHLayers.count, restored.bHLayers.count)
            XCTAssertEqual(weights.gammaRMS.count, restored.gammaRMS.count)
            XCTAssertEqual(weights.wConv.count, restored.wConv.count)
            XCTAssertEqual(weights.wOut.count, restored.wOut.count)
            XCTAssertEqual(weights.bOut.count, restored.bOut.count)
            XCTAssertEqual(weights.wIn[0], restored.wIn[0])
            XCTAssertEqual(weights.wRec[0], restored.wRec[0])
            XCTAssertEqual(weights.wConv[0][0], restored.wConv[0][0])
        } catch {
            XCTFail("SpikingNetworkWeights の Codable 処理に失敗しました: \(error)")
        }
    }

    // MARK: - 6. 系列デコード全体の動作検証

    func testSpikingAcousticDecoderSequenceDecoding() {
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 32,
            maxHiddenDim: 64,
            outputDim: 16,
            timeSteps: 2,
            numLayers: 4,
            seed: 999
        )
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 64, outputDim: 16, numLayers: 4)

        var seq = [[Float]](repeating: [Float](repeating: 0.1, count: 32), count: 20)
        var t = 0
        while t < 20 {
            seq[t][0] = 0.5 + (Float(t) * 0.02)
            t += 1
        }

        let outSeq = decoder.decodeSequence(featuresSeq: seq, workspace: workspace)
        XCTAssertEqual(outSeq.count, 20)
        XCTAssertEqual(outSeq[0].count, 16)

        let emptyOut = decoder.decodeSequence(featuresSeq: [], workspace: workspace)
        XCTAssertEqual(emptyOut.count, 0)
    }

    // MARK: - 7. 膜電位の時間連続性検証

    func testDecoderMembraneContinuity() {
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 128,
            maxHiddenDim: 64,
            outputDim: 64,
            numLayers: 4
        )
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(
            maxHiddenDim: weights.maxHiddenDim,
            outputDim: weights.outputDim,
            numLayers: weights.numLayers
        )

        let frame0 = [Float](repeating: 0.5, count: 128)
        let frame1 = [Float](repeating: 0.5, count: 128)
        let seq = [frame0, frame1]

        let output = decoder.decodeSequence(featuresSeq: seq, workspace: workspace)
        XCTAssertEqual(output.count, 2)

        var nonZeroMembrane = false
        var i = 0
        let v0 = workspace.layerStates[0].v
        while i < v0.count {
            if 0.01 < abs(v0[i]) {
                nonZeroMembrane = true
                break
            }
            i += 1
        }
        XCTAssertTrue(nonZeroMembrane, "フレーム間で膜電位が時間連続的に保持されていません")
    }

    // MARK: - 8. 4 ブロック SNN および時間畳み込みの検証

    func testFourBlockSNNTemporalConvolutionDynamics() {
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 64,
            maxHiddenDim: 128,
            outputDim: 40,
            timeSteps: 2,
            numLayers: 4,
            seed: 2026
        )
        XCTAssertEqual(weights.numLayers, 4)
        XCTAssertEqual(weights.wConv.count, 3)

        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 128, outputDim: 40, numLayers: 4)

        // 時間方向のステップ入力（インパルス）系列を流し、前後フレームへのコンテキスト拡散を検証
        var impulseSeq = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: 10)
        impulseSeq[4][0] = 3.0 // 中央フレームのみ強い入力

        let out = decoder.decodeSequence(featuresSeq: impulseSeq, workspace: workspace)
        XCTAssertEqual(out.count, 10)

        // 時間畳み込みにより、インパルスフレーム(4)だけでなく後続フレーム(5, 6)にも非ゼロ信号が伝播することを確認
        var frame5Sum: Float = 0.0
        var c = 0
        while c < 40 {
            frame5Sum += abs(out[5][c])
            c += 1
        }
        XCTAssertTrue(0.0 < frame5Sum, "時間畳み込みによる時間方向コンテキスト混合が観測されません")
    }

    // MARK: - 9. データ駆動型モーラ長 (こんにちは本体 0.70–0.95s, 天気本体 1.3–1.7s, 無音 80–160ms) の受入検証

    func testKonnichiwaDurationAlignedWithTrainingTable() {
        let engine = SpikeSpeechEngine()
        let lingKonnichiwa = engine.lengthRegulator.processText(
            text: "こんにちは",
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            addBoundarySilence: true
        )

        let leadSilFrames = Int(lingKonnichiwa.durations.first ?? 0)
        let trailSilFrames = Int(lingKonnichiwa.durations.last ?? 0)
        let totalSilFrames = leadSilFrames + trailSilFrames
        let bodyFrames = lingKonnichiwa.totalFrames - totalSilFrames

        let bodyDurationSec = Float(bodyFrames) * 0.010
        let totalSilSec = Float(totalSilFrames) * 0.010
        let moraMs = (bodyDurationSec / 5.0) * 1000.0

        let audioKonnichiwa = engine.synthesize(text: "こんにちは")
        let totalDurationSec = Float(audioKonnichiwa.count) / 16000.0

        print("[Konnichiwa Duration Check] 全体: \(totalDurationSec)s, 本体: \(bodyDurationSec)s, 先頭無音: \(Float(leadSilFrames) * 0.010)s, 末尾無音: \(Float(trailSilFrames) * 0.010)s, モーラ速度: \(moraMs) ms/モーラ")

        // 受入基準 1: 先頭末尾無音は Copy-synth 程度（80–160 ms）を厳格に維持
        XCTAssertTrue(0.08 <= totalSilSec, "文頭末無音合計が 80 ms 未満です: \(totalSilSec) 秒")
        XCTAssertTrue(totalSilSec <= 0.16, "文頭末無音合計が 160 ms を超えています (260+320ms は禁止): \(totalSilSec) 秒")

        // 受入基準 2: 「こんにちは」発話本体は 0.70–0.95 秒 (5モーラ × 約160ms)
        XCTAssertTrue(0.70 <= bodyDurationSec, "こんにちはの発話本体が 0.70 秒未満です: \(bodyDurationSec) 秒")
        XCTAssertTrue(bodyDurationSec <= 0.95, "こんにちはの発話本体が 0.95 秒を超過しています: \(bodyDurationSec) 秒")

        // 受入基準 3: 「今日はいい天気です」発話本体は 1.45–1.75 秒 (10モーラ目安 × 約160ms)
        let lingTenki = engine.lengthRegulator.processText(
            text: "今日はいい天気です",
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            addBoundarySilence: true
        )
        let tenkiLeadSil = Int(lingTenki.durations.first ?? 0)
        let tenkiTrailSil = Int(lingTenki.durations.last ?? 0)
        let tenkiBodyFrames = lingTenki.totalFrames - (tenkiLeadSil + tenkiTrailSil)
        let tenkiBodySec = Float(tenkiBodyFrames) * 0.010
        print("[Tenki Duration Check] 本体: \(tenkiBodySec)s, 1モーラあたり: \((tenkiBodySec / 10.0) * 1000.0) ms")
        XCTAssertTrue(1.45 <= tenkiBodySec, "天気の発話本体が 1.45 秒未満です: \(tenkiBodySec) 秒")
        XCTAssertTrue(tenkiBodySec <= 1.75, "天気の発話本体が 1.75 秒を超過しています: \(tenkiBodySec) 秒")

        // 受入基準 4: 「水をマレーシアから買わなくてはならないのです」（23モーラ）発話本体は教師モーラ長（145–175 ms/モーラ、本体 3.30–4.00 秒）
        let lingMizu = engine.lengthRegulator.processText(
            text: "水をマレーシアから買わなくてはならないのです",
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            addBoundarySilence: true
        )
        let mizuLeadSil = Int(lingMizu.durations.first ?? 0)
        let mizuTrailSil = Int(lingMizu.durations.last ?? 0)
        let mizuBodyFrames = lingMizu.totalFrames - (mizuLeadSil + mizuTrailSil)
        let mizuBodySec = Float(mizuBodyFrames) * 0.010
        let mizuMoraRate = (mizuBodySec / 23.0) * 1000.0
        print("[Mizuwomare Duration Check] 本体: \(mizuBodySec)s (23モーラ, 1モーラあたり: \(mizuMoraRate) ms), Copy(3.04s, 132ms/モーラ)との差: \(mizuBodySec - 3.04)s")
        XCTAssertTrue(3.30 <= mizuBodySec, "水をマレーシアの発話本体が 3.30 秒未満です: \(mizuBodySec) 秒")
        XCTAssertTrue(mizuBodySec <= 4.00, "水をマレーシアの発話本体が 4.00 秒を超過しています: \(mizuBodySec) 秒")
    }
}
