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

    /// 通常発話系列 (音素数 > 3) において 1音素20フレーム上限制約および最小継続時間制約が厳密に守られることを検証
    func testMASStrictlyEnforcesStandardMaxDuration() {
        let frameCount = 100
        let testMel = [[Float]](repeating: [Float](repeating: -6.0, count: AudioConfig.melChannels), count: frameCount)
        let testVoiced = [Float](repeating: 0.0, count: frameCount)

        let tokens = [
            PhonemeToken(id: 10, symbol: "k", category: .consonant),
            PhonemeToken(id: 9, symbol: "o", category: .vowel),
            PhonemeToken(id: 13, symbol: "n", category: .consonant),
            PhonemeToken(id: 6, symbol: "i", category: .vowel),
            PhonemeToken(id: 18, symbol: "w", category: .consonant),
            PhonemeToken(id: 5, symbol: "a", category: .vowel)
        ]

        // 6 音素で 100 フレーム (1音素平均 16.6F)
        let bootstrapDurs = MonotonicAlignmentSearch.initialBootstrapDurations(totalFrames: frameCount, phonemes: tokens)
        XCTAssertEqual(bootstrapDurs.count, tokens.count)
        var b = 0
        while b < bootstrapDurs.count {
            XCTAssertTrue(MonotonicAlignmentSearch.minDuration(for: tokens[b]) <= bootstrapDurs[b])
            XCTAssertTrue(bootstrapDurs[b] <= MonotonicAlignmentSearch.standardMaxDuration)
            b += 1
        }

        let prototypes = MonotonicAlignmentSearch.accumulatePrototypes(
            utterances: [(mel: testMel, voiced: testVoiced, phonemes: tokens, durations: bootstrapDurs)]
        )
        let mas = MonotonicAlignmentSearch(prototypes: prototypes)
        guard let durations = mas.align(mel: testMel, voiced: testVoiced, phonemes: tokens) else {
            XCTFail("MAS alignment failed")
            return
        }

        XCTAssertEqual(durations.count, tokens.count)
        var sum = 0
        var i = 0
        while i < durations.count {
            let d = durations[i]
            let reqMin = MonotonicAlignmentSearch.minDuration(for: tokens[i])
            XCTAssertTrue(reqMin <= d, "音素 \(tokens[i].symbol) のフレーム長 \(d) が最小要件 \(reqMin) 未満")
            XCTAssertTrue(d <= MonotonicAlignmentSearch.standardMaxDuration, "音素 \(tokens[i].symbol) のフレーム長 \(d) が上限 \(MonotonicAlignmentSearch.standardMaxDuration)F を超過")
            sum += d
            i += 1
        }
        XCTAssertEqual(sum, frameCount)

        var phList: [PhonemeAlignment] = []
        var pIdx = 0
        while pIdx < tokens.count {
            phList.append(PhonemeAlignment(symbol: tokens[pIdx].symbol, phoneId: Int32(tokens[pIdx].id), durationFrames: durations[pIdx]))
            pIdx += 1
        }
        let utt = UtteranceAlignment(
            utteranceId: "TEST_MAX20",
            leadSilenceFrames: 10,
            trailSilenceFrames: 10,
            totalSpeechFrames: frameCount,
            phonemes: phList
        )
        XCTAssertTrue(AlignmentStore.isUtteranceAlignmentValid(utt))
    }

    /// AlignmentStore.isUtteranceAlignmentValid が縮退音素 (1F, 40F超過) を厳格に拒絶することを検証
    func testAlignmentStoreValidationRejectsDegenerateAndExcessiveDurations() {
        // 正常発話
        let validUtt = UtteranceAlignment(
            utteranceId: "VALID",
            leadSilenceFrames: 10,
            trailSilenceFrames: 10,
            totalSpeechFrames: 15,
            phonemes: [
                PhonemeAlignment(symbol: "k", phoneId: 10, durationFrames: 5),
                PhonemeAlignment(symbol: "a", phoneId: 5, durationFrames: 10)
            ]
        )
        XCTAssertTrue(AlignmentStore.isUtteranceAlignmentValid(validUtt))

        // 子音 1F (縮退: < 2F)
        let degConsonant1F = UtteranceAlignment(
            utteranceId: "DEG_1F",
            leadSilenceFrames: 10,
            trailSilenceFrames: 10,
            totalSpeechFrames: 11,
            phonemes: [
                PhonemeAlignment(symbol: "k", phoneId: 10, durationFrames: 1),
                PhonemeAlignment(symbol: "a", phoneId: 5, durationFrames: 10)
            ]
        )
        XCTAssertTrue(AlignmentStore.isUtteranceAlignmentValid(degConsonant1F) != true)

        // 母音 1F (縮退: < 2F)
        let degVowel1F = UtteranceAlignment(
            utteranceId: "DEG_V1F",
            leadSilenceFrames: 10,
            trailSilenceFrames: 10,
            totalSpeechFrames: 6,
            phonemes: [
                PhonemeAlignment(symbol: "k", phoneId: 10, durationFrames: 5),
                PhonemeAlignment(symbol: "a", phoneId: 5, durationFrames: 1)
            ]
        )
        XCTAssertTrue(AlignmentStore.isUtteranceAlignmentValid(degVowel1F) != true)

        // 母音 41F (> 40F)
        let excessVowel41F = UtteranceAlignment(
            utteranceId: "EXCESS_41F",
            leadSilenceFrames: 10,
            trailSilenceFrames: 10,
            totalSpeechFrames: 46,
            phonemes: [
                PhonemeAlignment(symbol: "k", phoneId: 10, durationFrames: 5),
                PhonemeAlignment(symbol: "a", phoneId: 5, durationFrames: 41)
            ]
        )
        XCTAssertTrue(AlignmentStore.isUtteranceAlignmentValid(excessVowel41F) != true)
    }

    /// MAS が音素継続時間統計の二次ペナルティ（-1.0 * (d - mu)^2 / var）によって自然な長さに誘導されることを検証
    func testMonotonicAlignmentSearchWithQuadraticPenalty() {
        let tTotal = 30
        let testMel = [[Float]](repeating: [Float](repeating: -2.0, count: AudioConfig.melChannels), count: tTotal)
        let testVoiced = [Float](repeating: 1.0, count: tTotal)

        let tokens = [
            PhonemeToken(id: 5, symbol: "a", category: .vowel),
            PhonemeToken(id: 6, symbol: "i", category: .vowel)
        ]

        let protoA = MonotonicAlignmentSearch.PhonemeMelPrototype(
            meanMel: [Float](repeating: -2.0, count: AudioConfig.melChannels),
            voiced: 1.0,
            frameCount: 10.0
        )
        let protoI = MonotonicAlignmentSearch.PhonemeMelPrototype(
            meanMel: [Float](repeating: -2.0, count: AudioConfig.melChannels),
            voiced: 1.0,
            frameCount: 10.0
        )
        let protos = [5: protoA, 6: protoI]

        // /a/ の目標長 mu = 20.0, /i/ の目標長 mu = 10.0
        let statsA = MonotonicAlignmentSearch.PhonemeDurationStats(mean: 20.0, stdDev: 3.0, variance: 9.0, count: 10.0)
        let statsI = MonotonicAlignmentSearch.PhonemeDurationStats(mean: 10.0, stdDev: 3.0, variance: 9.0, count: 10.0)
        let durStats = [5: statsA, 6: statsI]

        let aligner = MonotonicAlignmentSearch(prototypes: protos, durationStats: durStats)
        guard let durs = aligner.align(mel: testMel, voiced: testVoiced, phonemes: tokens) else {
            XCTFail("MAS alignment failed")
            return
        }

        XCTAssertEqual(durs.count, 2)
        XCTAssertEqual(durs[0] + durs[1], tTotal)
        // Mel スペクトルが完全に一様でも、二次ペナルティにより /a/ は 20F 付近、/i/ は 10F 付近に決定される
        XCTAssertTrue(18 <= durs[0], "/a/ が目標平均長 20F 付近に引き寄せられていません: \(durs[0])")
        XCTAssertTrue(durs[0] <= 22, "/a/ が目標平均長 20F 付近に引き寄せられていません: \(durs[0])")
    }

    /// accumulateDurationStats および accumulatePrototypes が統計を正しく算出し、未観測音素を保持することを検証
    func testAccumulateDurationStatsAndPrototypePreservation() {
        let p1 = PhonemeToken(id: 5, symbol: "a", category: .vowel)
        let p2 = PhonemeToken(id: 10, symbol: "k", category: .consonant)

        let melA = [[Float]](repeating: [Float](repeating: -1.0, count: AudioConfig.melChannels), count: 10)
        let voicedA = [Float](repeating: 1.0, count: 10)

        let utterance1 = (
            mel: melA,
            voiced: voicedA,
            phonemes: [p1],
            durations: [10]
        )

        let initialStats = MonotonicAlignmentSearch.accumulateDurationStats(utterances: [utterance1])
        XCTAssertNotNil(initialStats[5])
        switch initialStats[5] {
        case .some(let s):
            XCTAssertEqual(s.mean, 10.0)
            XCTAssertTrue(2.0 <= s.variance, "分散下限 2.0 クランプが守られていません")
            XCTAssertTrue(s.variance <= 25.0, "分散上限 25.0 クランプが守られていません")
        case .none:
            break
        }

        // 次ラウンドで p1 が観測されず p2 のみ観測された場合、existing を渡せば p1 が保持されることを検証
        let melK = [[Float]](repeating: [Float](repeating: -3.0, count: AudioConfig.melChannels), count: 5)
        let voicedK = [Float](repeating: 0.0, count: 5)
        let utterance2 = (
            mel: melK,
            voiced: voicedK,
            phonemes: [p2],
            durations: [5]
        )

        let updatedStats = MonotonicAlignmentSearch.accumulateDurationStats(
            utterances: [utterance2],
            existing: initialStats
        )
        XCTAssertNotNil(updatedStats[5], "未観測音素 p1 の durationStats が消失しました")
        XCTAssertNotNil(updatedStats[10], "新規音素 p2 の durationStats が集計されていません")

        let proto1 = MonotonicAlignmentSearch.accumulatePrototypes(utterances: [utterance1])
        let updatedProtos = MonotonicAlignmentSearch.accumulatePrototypes(
            utterances: [utterance2],
            existing: proto1
        )
        XCTAssertNotNil(updatedProtos[5], "未観測音素 p1 のプロトタイプが消失しました")
        XCTAssertNotNil(updatedProtos[10], "新規音素 p2 のプロトタイプが集計されていません")
    }

    /// MelSpectrogramExtractor のスペクトル重心 (computeSpectralCentroid) および RMS (computeRMS) の音響物理計算を検証
    func testAudioFeatureExtractorSpectralCentroidAndRMS() {
        let extractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)

        // 1. 空配列の境界検証
        XCTAssertEqual(MelSpectrogramExtractor.computeRMS(pcm: []), 0.0)
        XCTAssertEqual(extractor.computeSpectralCentroid(pcm: []), 0.0)

        // 2. 無音信号の検証
        let silence = [Float](repeating: 0.0, count: 1600)
        XCTAssertEqual(MelSpectrogramExtractor.computeRMS(pcm: silence), 0.0)
        XCTAssertEqual(extractor.computeSpectralCentroid(pcm: silence), 0.0)

        // 3. 既知振幅サイン波の RMS 検証 (振幅 0.5 のサイン波 -> RMS は約 0.5 / sqrt(2) ≈ 0.3535)
        let sampleCount = 3200 // 200ms at 16kHz
        var sineLow = [Float](repeating: 0.0, count: sampleCount)
        var sineHigh = [Float](repeating: 0.0, count: sampleCount)
        let twoPi = Float.pi * 2.0

        var i = 0
        while i < sampleCount {
            let t = Float(i) / 16000.0
            sineLow[i] = 0.5 * sinf(twoPi * 400.0 * t)   // 400 Hz (低音)
            sineHigh[i] = 0.5 * sinf(twoPi * 4000.0 * t) // 4000 Hz (高音)
            i += 1
        }

        let rmsLow = MelSpectrogramExtractor.computeRMS(pcm: sineLow)
        XCTAssertTrue(0.34 <= rmsLow, "RMS が理論値 0.354 より過小: \(rmsLow)")
        XCTAssertTrue(rmsLow <= 0.37, "RMS が理論値 0.354 より過大: \(rmsLow)")

        // 4. スペクトル重心の周波数追従性検証 (400 Hz < 1000 Hz, 4000 Hz > 3000 Hz)
        let centroidLow = extractor.computeSpectralCentroid(pcm: sineLow)
        let centroidHigh = extractor.computeSpectralCentroid(pcm: sineHigh)

        XCTAssertTrue(centroidLow < 1000.0, "低周波サイン波のスペクトル重心が高すぎます: \(centroidLow) Hz")
        XCTAssertTrue(3000.0 < centroidHigh, "高周波サイン波のスペクトル重心が低すぎます: \(centroidHigh) Hz")
    }

    /// SpikingNetworkWeights.withResetAcousticInWeights が音響チャンネル列 (ch 192..198) のみを正しく再初期化し、他重みを厳密維持することを検証
    func testResetAcousticInWeights() {
        let baseWeights = SpikingNetworkWeights.randomWeights(
            inputDim: 256,
            maxHiddenDim: 256,
            outputDim: 64,
            timeSteps: 2,
            numLayers: 2,
            seed: 2026
        )

        let resetWeights = baseWeights.withResetAcousticInWeights(seed: 2026)

        // 1. 再帰重み・出力層重み・上位層重み・バイアスが完全一致
        XCTAssertEqual(resetWeights.wRec, baseWeights.wRec)
        XCTAssertEqual(resetWeights.wOut, baseWeights.wOut)
        XCTAssertEqual(resetWeights.bH, baseWeights.bH)
        XCTAssertEqual(resetWeights.bOut, baseWeights.bOut)
        XCTAssertEqual(resetWeights.wLayers, baseWeights.wLayers)

        // 2. 音素 one-hot 列 (ch 0..<192) が完全一致
        var h = 0
        while h < baseWeights.maxHiddenDim {
            let rowOffset = h * baseWeights.inputDim
            var ch = 0
            while ch < 192 {
                XCTAssertEqual(resetWeights.wIn[rowOffset + ch], baseWeights.wIn[rowOffset + ch])
                ch += 1
            }
            // 未使用列 (ch 200..<256) も完全一致
            ch = 200
            while ch < 256 {
                XCTAssertEqual(resetWeights.wIn[rowOffset + ch], baseWeights.wIn[rowOffset + ch])
                ch += 1
            }
            h += 1
        }

        // 3. 再初期化された音響列 (ch 192..198) の L2 ノルムが未使用チャンネルと同等の規模（約 0.5〜1.5）
        var ch = 192
        while ch <= 198 {
            var sumSq: Float = 0.0
            var row = 0
            while row < baseWeights.maxHiddenDim {
                let v = resetWeights.wIn[row * baseWeights.inputDim + ch]
                sumSq += v * v
                row += 1
            }
            let norm = sqrtf(sumSq)
            XCTAssertTrue(0.5 <= norm, "再初期化列 ch \(ch) の L2 ノルムが小さすぎます: \(norm)")
            XCTAssertTrue(norm <= 1.5, "再初期化列 ch \(ch) の L2 ノルムが大きすぎます: \(norm)")
            ch += 1
        }
    }

    /// BASIC5000_0001 音響特徴およびアライメント継続時間ゲートの受入基準（手順 3）を検証
    func testBasic0001AcousticGateAcceptance() throws {
        let corpusDir = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000"
        let wavPath0001 = "\(corpusDir)/wav/BASIC5000_0001.wav"
        let corpusAlignPath = "\(corpusDir)/mas_alignments.json"

        let fileManager = FileManager.default
        let hasCorpus = fileManager.fileExists(atPath: wavPath0001) && fileManager.fileExists(atPath: corpusAlignPath)
        if hasCorpus != true {
            throw XCTSkip("コーパスファイルが存在しないためスキップします")
        }

        guard let alignMap = try? AlignmentStore.load(from: corpusAlignPath),
              let align0001 = alignMap["BASIC5000_0001"] else {
            XCTFail("BASIC5000_0001 アライメントの読み込みに失敗しました")
            return
        }

        let wavReader = WavAudioReader()
        let pcm0001 = try wavReader.loadWav16k(from: wavPath0001)
        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)
        let hopSize = AudioConfig.hopSize

        var curOffset = align0001.leadSilenceFrames
        var b = 0
        while b < align0001.phonemes.count {
            let ph = align0001.phonemes[b]
            let d = ph.durationFrames
            let sStart = curOffset * hopSize
            let sEnd = min(pcm0001.count, (curOffset + d) * hopSize)
            var segPCM: [Float] = []
            if sStart < sEnd {
                segPCM = Array(pcm0001[sStart..<sEnd])
            }

            let isVowel: Bool
            switch ph.symbol {
            case "a", "i", "u", "e", "o", "_":
                isVowel = true
            default:
                isVowel = false
            }

            if isVowel {
                let rms = MelSpectrogramExtractor.computeRMS(pcm: segPCM)
                XCTAssertTrue(0.01 < rms, "母音 [\(ph.symbol)] の平均 RMS (\(rms)) が 0.01 以下です")
            }

            let isFricative = (ph.symbol == "s" || ph.symbol == "sh")
            if isFricative {
                let centroid = melExtractor.computeSpectralCentroid(pcm: segPCM)
                XCTAssertTrue(2000.0 < centroid, "摩擦音 [\(ph.symbol)] のスペクトル重心 (\(centroid) Hz) が 2000 Hz 以下です")
            }

            curOffset += d
            b += 1
        }
    }
}

