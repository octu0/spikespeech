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

    /// 教師 Mel 実測プロトタイプ集計と MAS 単調動的計画法による境界推定で総フレーム数が厳密保存されることを検証
    func testMonotonicAlignmentSearchPrototypeConvergence() {
        let frameCount = 60
        var testMel = [[Float]](repeating: [Float](repeating: -7.0, count: AudioConfig.melChannels), count: frameCount)
        var testVoiced = [Float](repeating: 0.0, count: frameCount)

        // 前半 20F: 母音 /a/ (有声高, 低中域エネルギー高)
        var f = 0
        while f < 20 {
            var c = 4
            while c < 32 {
                testMel[f][c] = 2.0
                c += 1
            }
            testVoiced[f] = 1.0
            f += 1
        }
        // 中盤 15F: 子音 /s/ (無声, 高域エネルギー高)
        while f < 35 {
            var c = 48
            while c < AudioConfig.melChannels {
                testMel[f][c] = 1.5
                c += 1
            }
            testVoiced[f] = 0.0
            f += 1
        }
        // 後半 25F: 母音 /i/ (有声高, 高域フォルマントあり)
        while f < frameCount {
            var c = 8
            while c < 40 {
                testMel[f][c] = 1.8
                c += 1
            }
            testVoiced[f] = 0.95
            f += 1
        }

        let tokens = [
            PhonemeToken(id: 5, symbol: "a", category: .vowel),
            PhonemeToken(id: 11, symbol: "s", category: .consonant),
            PhonemeToken(id: 6, symbol: "i", category: .vowel)
        ]

        // 1. 初期仮割り
        let bootstrapDurs = MonotonicAlignmentSearch.initialBootstrapDurations(
            totalFrames: frameCount,
            phonemes: tokens
        )
        XCTAssertEqual(bootstrapDurs.count, tokens.count)
        var bootstrapSum = 0
        var bIdx = 0
        while bIdx < bootstrapDurs.count {
            XCTAssertTrue(1 <= bootstrapDurs[bIdx])
            bootstrapSum += bootstrapDurs[bIdx]
            bIdx += 1
        }
        XCTAssertEqual(bootstrapSum, frameCount)

        // 2. プロトタイプ集計
        let prototypes = MonotonicAlignmentSearch.accumulatePrototypes(
            utterances: [(mel: testMel, voiced: testVoiced, phonemes: tokens, durations: bootstrapDurs)]
        )
        XCTAssertEqual(prototypes.count, 3)

        // 3. MAS 単調アライメント
        let mas = MonotonicAlignmentSearch(prototypes: prototypes)
        guard let durations = mas.align(mel: testMel, voiced: testVoiced, phonemes: tokens, meanFramesPerMora: 16.0) else {
            XCTFail("MAS alignment failed")
            return
        }

        XCTAssertEqual(durations.count, tokens.count)
        var totalSum = 0
        var i = 0
        while i < durations.count {
            XCTAssertTrue(1 <= durations[i])
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

        // "か" は [k, a] の 1 モーラ。目標モーラ長 16.0 フレームに対し、重み比率 (k: 3.0, a: 14.0 * 1.15 = 16.1) で比例配分される
        XCTAssertEqual(features.phoneIds.count, 2)
        XCTAssertEqual(features.durations[0], 3)  // k: 3 frames
        XCTAssertEqual(features.durations[1], 13) // a: 13 frames
        XCTAssertEqual(features.totalFrames, 16)  // 1 モーラ = 正確に 16 frames
    }

    /// SpikingNetworkWeights における音素平均フレームおよびモーラ平均長の JSON 永続化と復元を検証
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
            phonemeAverageDurations: originalTable,
            meanFramesPerMora: 16.0
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
        XCTAssertEqual(decoded.meanFramesPerMora, 16.0)
    }

    /// SpikeSpeechEngine が重みに含まれる音素平均テーブルおよび meanFramesPerMora を自動的に推論正本として引き継ぐことを検証
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
        let weightsWithTable = baseWeights
            .withPhonemeAverageDurations(customTable)
            .withMeanFramesPerMora(16.5)
        let engine = SpikeSpeechEngine(weights: weightsWithTable)

        XCTAssertEqual(engine.lengthRegulator.phonemeAverageDurations[5], 12.0)
        XCTAssertEqual(engine.lengthRegulator.phonemeAverageDurations[10], 4.0)
        XCTAssertEqual(engine.lengthRegulator.meanFramesPerMora, 16.5)
    }

    /// 教師 Mel 単調動的計画法 (MAS) による音素アライメントで総フレーム数が厳密保存され、物理境界が正しく分離されることを検証
    func testMonotonicAlignmentSearchExactConvergence() {
        let tTotal = 40
        var testMel = [[Float]](repeating: [Float](repeating: -7.0, count: AudioConfig.melChannels), count: tTotal)
        var testVoiced = [Float](repeating: 0.0, count: tTotal)

        // 0..<15F: /s/ (無声、高周波帯域 ch 48-63 に強エネルギー)
        var t = 0
        while t < 15 {
            var c = 48
            while c < AudioConfig.melChannels {
                testMel[t][c] = 2.0
                c += 1
            }
            testVoiced[t] = 0.0
            t += 1
        }

        // 15..<40F: /a/ (有声、低中域 ch 4-32 にフォルマント)
        while t < tTotal {
            var c = 4
            while c < 32 {
                testMel[t][c] = 2.5
                c += 1
            }
            testVoiced[t] = 1.0
            t += 1
        }

        let tokens = [
            PhonemeToken(id: 11, symbol: "s", category: .consonant),
            PhonemeToken(id: 5, symbol: "a", category: .vowel)
        ]

        // 初期仮割りから音素プロトタイプを自律集計
        let bootstrapDurs = MonotonicAlignmentSearch.initialBootstrapDurations(totalFrames: tTotal, phonemes: tokens)
        let prototypes = MonotonicAlignmentSearch.accumulatePrototypes(
            utterances: [(mel: testMel, voiced: testVoiced, phonemes: tokens, durations: bootstrapDurs)]
        )
        let mas = MonotonicAlignmentSearch(prototypes: prototypes)

        guard let durs = mas.align(mel: testMel, voiced: testVoiced, phonemes: tokens, meanFramesPerMora: 16.0) else {
            XCTFail("MAS 単調アライメントに失敗しました")
            return
        }

        XCTAssertEqual(durs.count, 2)
        XCTAssertTrue(1 <= durs[0])
        XCTAssertTrue(1 <= durs[1])
        XCTAssertEqual(durs[0] + durs[1], tTotal)
        // /s/ が 15F 付近、/a/ が 25F 付近に境界決定されていること（許容差 ±3F）
        XCTAssertTrue(12 <= durs[0], "/s/ のフレーム長が短すぎます: \(durs[0])")
        XCTAssertTrue(durs[0] <= 18, "/s/ のフレーム長が長すぎます: \(durs[0])")
    }
}

