import XCTest
@testable import SpikeSpeech

final class ForcedAlignmentTests: XCTestCase {

    /// UtteranceAlignment および PhonemeAlignment の JSON 入出力整合性を検証
    func testAlignmentRecordSerializationRoundtrip() throws {
        let original = [
            UtteranceAlignment(
                utteranceId: "TEST_0001",
                leadSilenceFrames: 10,
                trailSilenceFrames: 8,
                totalSpeechFrames: 75,
                phonemes: [
                    PhonemeAlignment(symbol: "k", phoneId: 10, durationFrames: 5),
                    PhonemeAlignment(symbol: "o", phoneId: 9, durationFrames: 12),
                    PhonemeAlignment(symbol: "N", phoneId: 24, durationFrames: 20),
                    PhonemeAlignment(symbol: "n", phoneId: 13, durationFrames: 6),
                    PhonemeAlignment(symbol: "i", phoneId: 6, durationFrames: 10),
                    PhonemeAlignment(symbol: "ch", phoneId: 28, durationFrames: 4),
                    PhonemeAlignment(symbol: "i", phoneId: 6, durationFrames: 10),
                    PhonemeAlignment(symbol: "w", phoneId: 18, durationFrames: 5),
                    PhonemeAlignment(symbol: "a", phoneId: 5, durationFrames: 13)
                ]
            )
        ]

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode([UtteranceAlignment].self, from: data)

        XCTAssertEqual(decoded.count, original.count)
        XCTAssertEqual(decoded[0].utteranceId, "TEST_0001")
        XCTAssertEqual(decoded[0].leadSilenceFrames, 10)
        XCTAssertEqual(decoded[0].trailSilenceFrames, 8)
        XCTAssertEqual(decoded[0].totalSpeechFrames, 75)
        XCTAssertEqual(decoded[0].phonemes.count, 9)
        XCTAssertEqual(decoded[0].phonemes[0].symbol, "k")
        XCTAssertEqual(decoded[0].phonemes[0].durationFrames, 5)
    }

    /// AlignmentStore による音素 ID 別平均継続時間集計の数学的正当性を検証
    func testAlignmentStoreAverageDurations() {
        let alignments = [
            UtteranceAlignment(
                utteranceId: "UTT_1",
                leadSilenceFrames: 5,
                trailSilenceFrames: 5,
                totalSpeechFrames: 30,
                phonemes: [
                    PhonemeAlignment(symbol: "a", phoneId: 5, durationFrames: 10),
                    PhonemeAlignment(symbol: "k", phoneId: 10, durationFrames: 4)
                ]
            ),
            UtteranceAlignment(
                utteranceId: "UTT_2",
                leadSilenceFrames: 5,
                trailSilenceFrames: 5,
                totalSpeechFrames: 30,
                phonemes: [
                    PhonemeAlignment(symbol: "a", phoneId: 5, durationFrames: 12),
                    PhonemeAlignment(symbol: "k", phoneId: 10, durationFrames: 6)
                ]
            )
        ]

        let averages = AlignmentStore.computeAverageDurations(from: alignments)
        XCTAssertEqual(averages[5], 11.0)
        XCTAssertEqual(averages[10], 5.0)
    }

    /// AcousticForcedAligner の DP による大局的アライメントで総フレーム数が厳密保存されることを検証
    func testAcousticForcedAlignerDPConvergence() {
        let aligner = AcousticForcedAligner()
        let frameCount = 60
        var features: [AcousticForcedAligner.FrameAcousticFeatures] = []

        // 前半 20F: 母音特徴 (有声高, パワー高), 中盤 15F: 子音特徴 (摩擦高), 後半 25F: 母音特徴
        var f = 0
        while f < frameCount {
            let feat: AcousticForcedAligner.FrameAcousticFeatures
            switch f {
            case 0..<20:
                feat = AcousticForcedAligner.FrameAcousticFeatures(power: 0.05, voiced: 0.95, spectralFlux: 0.1, highFreqRatio: 0.1, lowFreqRatio: 0.6)
            case 20..<35:
                feat = AcousticForcedAligner.FrameAcousticFeatures(power: 0.01, voiced: 0.1, spectralFlux: 0.3, highFreqRatio: 0.5, lowFreqRatio: 0.1)
            default:
                feat = AcousticForcedAligner.FrameAcousticFeatures(power: 0.04, voiced: 0.9, spectralFlux: 0.1, highFreqRatio: 0.1, lowFreqRatio: 0.6)
            }
            features.append(feat)
            f += 1
        }

        let tokens = [
            PhonemeToken(id: 5, symbol: "a", category: .vowel),
            PhonemeToken(id: 11, symbol: "s", category: .consonant),
            PhonemeToken(id: 6, symbol: "i", category: .vowel)
        ]

        guard let durations = aligner.align(features: features, phonemes: tokens) else {
            XCTFail("DP alignment failed")
            return
        }

        XCTAssertEqual(durations.count, tokens.count)
        var totalSum = 0
        var i = 0
        while i < durations.count {
            XCTAssertLessThanOrEqual(1, durations[i])
            totalSum += durations[i]
            i += 1
        }
        XCTAssertEqual(totalSum, frameCount)
    }

    /// LengthRegulator が 16.0 モーラ固定ではなく phonemeAverageDurations を正本として推論展開することを検証
    func testLengthRegulatorUsesAlignmentStatistics() {
        let customAverages: [Int32: Float] = [
            5: 14.0,  // a
            10: 3.0   // k
        ]
        let regulator = LengthRegulator(hiddenDimension: 64, phonemeAverageDurations: customAverages)

        let normalizer = TextNormalizer()
        let prosodyModel = ProsodyModel()
        let vocabulary = PhonemeVocabulary()

        let features = regulator.processText(
            text: "か",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 1.0,
            applyFluctuation: false,
            addBoundarySilence: false
        )

        // "か" は [k, a] の2音素。文末母音のため 1.15倍の句末伸長 (14 * 1.15 = 16.1 -> 16) が適用される
        XCTAssertEqual(features.phoneIds.count, 2)
        XCTAssertEqual(features.durations[0], 3)  // k: 3 frames
        XCTAssertEqual(features.durations[1], 16) // a: 14 * 1.15 = 16 frames
        XCTAssertEqual(features.totalFrames, 19)
    }

    /// SpikingNetworkWeights における音素平均フレームの JSON 永続化と復元を検証
    func testSpikingNetworkWeightsPhonemeAverageDurationsSerialization() throws {
        let originalTable: [Int32: Float] = [
            1: 8.0,
            5: 7.3,
            10: 14.5
        ]
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 64,
            maxHiddenDim: 64,
            outputDim: 80,
            timeSteps: 2,
            numLayers: 2,
            phonemeAverageDurations: originalTable
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(weights)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(SpikingNetworkWeights.self, from: data)

        XCTAssertNotNil(decoded.phonemeAverageDurations)
        switch decoded.phonemeAverageDurations {
        case .some(let table):
            XCTAssertEqual(table[1], 8.0)
            XCTAssertEqual(table[5], 7.3)
            XCTAssertEqual(table[10], 14.5)
        case .none:
            XCTFail("phonemeAverageDurations がデコードされませんでした")
        }
    }

    /// SpikeSpeechEngine が重みに含まれる音素平均テーブルを自動的に推論正本として引き継ぐことを検証
    func testSpikeSpeechEngineUsesWeightsPhonemeAverages() {
        let customTable: [Int32: Float] = [
            5: 12.0,
            10: 4.0
        ]
        let baseWeights = SpikingNetworkWeights.randomWeights(
            inputDim: 64,
            maxHiddenDim: 64,
            outputDim: 80,
            timeSteps: 2,
            numLayers: 2
        )
        let weightsWithTable = baseWeights.withPhonemeAverageDurations(customTable)
        let engine = SpikeSpeechEngine(weights: weightsWithTable)

        XCTAssertEqual(engine.lengthRegulator.phonemeAverageDurations[5], 12.0)
        XCTAssertEqual(engine.lengthRegulator.phonemeAverageDurations[10], 4.0)
    }
}
