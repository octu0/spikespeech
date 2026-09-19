import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import SpikeSpeech

final class ProsodyPredictorTests: XCTestCase {

    func testDurationPredictorWeightsCodableRoundtrip() throws {
        let weights = DurationPredictorWeights.randomWeights(seed: 12345)
        let encoder = JSONEncoder()
        let data = try encoder.encode(weights)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(DurationPredictorWeights.self, from: data)

        XCTAssertEqual(weights, decoded)
        XCTAssertEqual(weights.inputDim, decoded.inputDim)
        XCTAssertEqual(weights.hiddenDim, decoded.hiddenDim)
        XCTAssertEqual(weights.w1.count, decoded.w1.count)
        XCTAssertEqual(weights.b1.count, decoded.b1.count)
    }

    func testF0PredictorWeightsCodableRoundtrip() throws {
        let weights = F0PredictorWeights.randomWeights(seed: 54321)
        let encoder = JSONEncoder()
        let data = try encoder.encode(weights)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(F0PredictorWeights.self, from: data)
        XCTAssertEqual(weights, decoded)
        XCTAssertEqual(weights.wConv.count, decoded.wConv.count)
        XCTAssertEqual(decoded.wConv.count, 5 * decoded.hiddenDim)

        // 本番仕様 (hiddenDim: 128, K=5 -> 640 要素) の Codable ラウンドトリップ検証
        let weights128 = F0PredictorWeights.randomWeights(inputDim: 76, hiddenDim: 128, seed: 99999)
        let data128 = try encoder.encode(weights128)
        let decoded128 = try decoder.decode(F0PredictorWeights.self, from: data128)
        XCTAssertEqual(weights128, decoded128)
        XCTAssertEqual(decoded128.wConv.count, 640)
    }

    func testProsodyWeightsCodableRoundtrip() throws {
        let weights = ProsodyWeights.randomWeights()
        let encoder = JSONEncoder()
        let data = try encoder.encode(weights)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ProsodyWeights.self, from: data)

        XCTAssertEqual(weights, decoded)
    }

    func testProsodyPredictorFallbackWhenWeightsNil() {
        let predictor = ProsodyPredictor(weights: nil)
        let normalizer = TextNormalizer(morphology: ViterbiMorphology())
        let vocabulary = PhonemeVocabulary()
        let prosodyModel = ProsodyModel()
        let lengthRegulator = LengthRegulator()

        let morphemes = normalizer.normalize(text: "こんにちは")
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)

        let durations = predictor.predictDurations(
            phrases: phrases,
            vocabulary: vocabulary,
            lengthRegulator: lengthRegulator,
            speedFactor: 1.0,
            applyFluctuation: false
        )

        // フォールバック時は規則 duration と一致する
        XCTAssertFalse(durations.isEmpty)
        var i = 0
        while i < durations.count {
            XCTAssertTrue(1 <= durations[i])
            i += 1
        }

        let f0Result = predictor.predictF0Contour(
            phrases: phrases,
            vocabulary: vocabulary,
            baseF0: 220.0,
            prosodyModel: prosodyModel,
            durations: durations,
            applyFluctuation: false
        )

        XCTAssertEqual(f0Result.totalFrames, f0Result.f0Contour.count)
        XCTAssertEqual(f0Result.totalFrames, f0Result.voicedFlags.count)
    }

    func testProsodyPredictorWithWeightsGeneratesDynamicF0() {
        let pWeights = ProsodyWeights.randomWeights()
        let predictor = ProsodyPredictor(weights: pWeights)
        let normalizer = TextNormalizer(morphology: ViterbiMorphology())
        let vocabulary = PhonemeVocabulary()
        let prosodyModel = ProsodyModel()
        let lengthRegulator = LengthRegulator()

        let morphemes = normalizer.normalize(text: "今日はいい天気です")
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)

        let durations = predictor.predictDurations(
            phrases: phrases,
            vocabulary: vocabulary,
            lengthRegulator: lengthRegulator,
            speedFactor: 1.0,
            applyFluctuation: false
        )

        XCTAssertFalse(durations.isEmpty)
        var dSum = 0
        var i = 0
        while i < durations.count {
            XCTAssertTrue(1 <= durations[i])
            dSum += durations[i]
            i += 1
        }
        XCTAssertTrue(20 <= dSum)

        let f0Result = predictor.predictF0Contour(
            phrases: phrases,
            vocabulary: vocabulary,
            baseF0: 220.0,
            prosodyModel: prosodyModel,
            durations: durations,
            applyFluctuation: false
        )

        var voicedF0s: [Float] = []
        var f = 0
        while f < f0Result.totalFrames {
            if 0.5 <= f0Result.voicedFlags[f] {
                voicedF0s.append(f0Result.f0Contour[f])
            } else {
                XCTAssertEqual(f0Result.f0Contour[f], 0.0)
            }
            f += 1
        }

        XCTAssertFalse(voicedF0s.isEmpty)
        var minF0 = Float.greatestFiniteMagnitude
        var maxF0: Float = 0.0
        var v = 0
        while v < voicedF0s.count {
            let val = voicedF0s[v]
            if val < minF0 {
                minF0 = val
            }
            if maxF0 < val {
                maxF0 = val
            }
            v += 1
        }

        // F0 は一定トーン（ブザー音）ではなく、句内で変動すること
        XCTAssertTrue(minF0 < maxF0)
        XCTAssertTrue(10.0 <= (maxF0 - minF0))
    }

    func testMLXF0PredictorModelWConvIsTrainableAndUpdates() {
        let trainer = MLXProsodyTrainer(f0HiddenDim: 128, learningRate: 0.05)
        let initialConv = trainer.f0Model.wConv.asArray(Float.self)

        // trainableParameters に wConv が存在することを検証
        let params = trainer.f0Model.trainableParameters()
        let flattened = params.flattened()
        var hasWConv = false
        for (key, _) in flattened {
            if key.contains("wConv") {
                hasWConv = true
                break
            }
        }
        XCTAssertTrue(hasWConv, "MLXF0PredictorModel の trainableParameters に wConv が含まれていません")

        // ダミー系列データ（seqLen: 8）で勾配更新を実行
        let inDim = 76
        let seqLen = 8
        var dummyFeats = [[Float]](repeating: [Float](repeating: 0.1, count: inDim), count: seqLen)
        var s = 0
        while s < seqLen {
            dummyFeats[s][67] = 1.0 // high tone
            s += 1
        }
        let dummyFuji = [Float](repeating: 220.0, count: seqLen)
        let dummyTarget = [Float](repeating: 300.0, count: seqLen)
        let dummyVoiced = [Float](repeating: 1.0, count: seqLen)

        var step = 0
        while step < 5 {
            let loss = trainer.trainF0Step(
                features: dummyFeats,
                fujisakiF0: dummyFuji,
                targetF0: dummyTarget,
                voicedMask: dummyVoiced
            )
            XCTAssertTrue(0.0 < loss)
            step += 1
        }

        let updatedConv = trainer.f0Model.wConv.asArray(Float.self)
        var totalDiff: Float = 0.0
        var i = 0
        while i < initialConv.count {
            totalDiff += abs(initialConv[i] - updatedConv[i])
            i += 1
        }

        XCTAssertTrue(0.0001 <= totalDiff, "wConv のパラメータが勾配更新されていません: diff = \(totalDiff)")
    }

    func testSynthesizedSpeechHasPitchDynamics() {
        var pWeights: ProsodyWeights? = nil
        let weightsPath = "Models/weights.json"
        if let data = try? Data(contentsOf: URL(fileURLWithPath: weightsPath)) {
            if let decoded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) {
                pWeights = decoded.prosodyWeights
            }
        }

        if let fw = pWeights?.f0Weights {
            let w2Sum = fw.w2.reduce(0, +)
            let wConvSum = fw.wConv.reduce(0, +)
            print("[F0Weights Info] b2: \(fw.b2), w2 sum: \(w2Sum), wConv sum: \(wConvSum), inputDim: \(fw.inputDim), hiddenDim: \(fw.hiddenDim)")
        }

        let predictor = ProsodyPredictor(weights: pWeights)
        let normalizer = TextNormalizer(morphology: ViterbiMorphology())
        let vocabulary = PhonemeVocabulary()
        let prosodyModel = ProsodyModel()
        let lengthRegulator = LengthRegulator()

        let testSentences = ["こんにちは", "今日はいい天気です"]
        var sIdx = 0
        while sIdx < testSentences.count {
            let sentence = testSentences[sIdx]
            let morphemes = normalizer.normalize(text: sentence)
            let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
            let durations = predictor.predictDurations(
                phrases: phrases,
                vocabulary: vocabulary,
                lengthRegulator: lengthRegulator,
                speedFactor: 1.0,
                applyFluctuation: false
            )
            let f0Result = predictor.predictF0Contour(
                phrases: phrases,
                vocabulary: vocabulary,
                baseF0: 220.0,
                prosodyModel: prosodyModel,
                durations: durations,
                applyFluctuation: false
            )

            var minF0 = Float.greatestFiniteMagnitude
            var maxF0: Float = 0.0
            var sumF0: Float = 0.0
            var voicedCount = 0
            var f = 0
            while f < f0Result.totalFrames {
                if 0.5 <= f0Result.voicedFlags[f] {
                    let val = f0Result.f0Contour[f]
                    if val < minF0 {
                        minF0 = val
                    }
                    if maxF0 < val {
                        maxF0 = val
                    }
                    sumF0 += val
                    voicedCount += 1
                }
                f += 1
            }

            var meanF0: Float = 0.0
            if 0 < voicedCount {
                meanF0 = sumF0 / Float(voicedCount)
            }
            let deltaF0 = maxF0 - minF0
            print("[PitchDynamics] 「\(sentence)」 有声フレーム数: \(voicedCount), min: \(String(format: "%.1f", minF0)) Hz, max: \(String(format: "%.1f", maxF0)) Hz, mean: \(String(format: "%.1f", meanF0)) Hz, delta: \(String(format: "%.1f", deltaF0)) Hz")
            XCTAssertTrue(25.0 <= deltaF0, "「\(sentence)」のピッチ起伏が 25 Hz 未満です: \(deltaF0) Hz")
            XCTAssertTrue(120.0 <= minF0, "「\(sentence)」の最低 F0 が異常に低すぎます: \(minF0) Hz")
            XCTAssertTrue(maxF0 <= 400.0, "「\(sentence)」の最高 F0 が異常に高すぎます（金切り声）: \(maxF0) Hz")
            sIdx += 1
        }
    }
}

