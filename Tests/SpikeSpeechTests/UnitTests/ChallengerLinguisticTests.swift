import XCTest
@testable import SpikeSpeech

/// Challenger 2 による Length Regulation & Prosody (F0) の数学的・定量的・敵対的検証テストスイート
///
/// 単体テストのパス確認にとどまらず、乱数オラクル生成、境界値ストレステスト、
/// F0 輪郭の不連続性計測、および累積和量子化の数式整合性を実験的に厳密検証する。
final class ChallengerLinguisticTests: XCTestCase {

    private var normalizer: TextNormalizer!
    private var morphology: ViterbiMorphology!
    private var vocabulary: PhonemeVocabulary!
    private var prosodyModel: ProsodyModel!
    private var lengthRegulator: LengthRegulator!

    override func setUp() {
        super.setUp()
        self.morphology = ViterbiMorphology()
        self.normalizer = TextNormalizer(morphology: morphology)
        self.vocabulary = PhonemeVocabulary()
        self.prosodyModel = ProsodyModel()
        self.lengthRegulator = LengthRegulator(hiddenDimension: 64)
    }

    // MARK: - 1. 累積和量子化 (quantizeDurations) の数学的オラクル検証

    func testCumulativeQuantizationMathematicalOracle() {
        // 累積和量子化が任意の正の実数 Duration 系列において
        // sum(F_i) == round(sum(d_i)) を厳密に満たすか数学的性質を検証する。
        var seed: UInt64 = 88172645463325252
        func lcgRand() -> UInt64 {
            seed = seed &* 6364136223846793005 &+ 1
            return seed
        }

        var trial = 0
        while trial < 200 {
            // 長さ 1〜50 のランダム系列
            let count = Int(lcgRand() % 50) + 1
            var durations = [Float](repeating: 0.0, count: count)
            var sumFloat: Float = 0.0

            var i = 0
            while i < count {
                // 1.0 〜 15.0 フレームの範囲 (一般的な音素長)
                let val = 1.0 + Float(lcgRand() % 1400) / 100.0
                durations[i] = val
                sumFloat += val
                i += 1
            }

            let quantized = lengthRegulator.quantizeDurations(durations: durations)
            let sumQuantized = quantized.reduce(0, +)
            let expectedSum = Int(roundf(sumFloat))

            XCTAssertEqual(
                sumQuantized,
                expectedSum,
                "累積和量子化の総フレーム数が round(sum(d_i)) と一致していません (trial: \(trial))"
            )

            // 各音素の Duration が最低 1 フレーム以上確保されていること
            var qIdx = 0
            while qIdx < quantized.count {
                XCTAssertTrue(1 <= quantized[qIdx], "音素フレーム数が 1 未満です")
                qIdx += 1
            }

            trial += 1
        }
    }

    // MARK: - 2. 累積和量子化の境界値・極端値ストレステスト

    func testCumulativeQuantizationExtremeDurations() {
        // frameCount < 1 ガードによる切り上げが発話全体の総和に与える影響を同定する。

        // ケース A: 0.0 のみで構成される配列
        let zeroDurations: [Float] = [0.0, 0.0, 0.0]
        let zeroQuantized = lengthRegulator.quantizeDurations(durations: zeroDurations)
        XCTAssertEqual(zeroQuantized.count, 3)
        // 最低1フレーム保証により、すべて1フレームになる
        XCTAssertEqual(zeroQuantized, [1, 1, 1])

        // ケース B: 微小小数 (0.05) の連続
        let tinyDurations: [Float] = [0.05, 0.05, 0.05, 0.05] // 合計 0.20 -> round = 0
        let tinyQuantized = lengthRegulator.quantizeDurations(durations: tinyDurations)
        XCTAssertEqual(tinyQuantized, [1, 1, 1, 1]) // 各要素が最低1フレームにクランプされる

        // ケース C: 空配列
        let emptyQuantized = lengthRegulator.quantizeDurations(durations: [])
        XCTAssertTrue(emptyQuantized.isEmpty)
    }

    // MARK: - 3. Length Regulation テンソル展開 (expand) の完全一致・メモリ連続性検証

    func testExpandTensorBitExactAndContiguity() {
        // ポインタコピーにおいて、各音素埋め込みベクトルが破損・混入なく
        // 正確なスライスに隙間なく展開されているか確認する。
        let dim = 64
        let phonemeCount = 5
        let durations = [3, 1, 4, 2, 5] // 合計 15 フレーム
        let totalFrames = 15

        var embeddings = [Float](repeating: 0.0, count: phonemeCount * dim)
        var p = 0
        while p < phonemeCount {
            var d = 0
            while d < dim {
                embeddings[p * dim + d] = Float(p * 1000 + d + 1)
                d += 1
            }
            p += 1
        }

        let expanded = lengthRegulator.expand(
            embeddings: embeddings,
            phonemeCount: phonemeCount,
            durations: durations
        )

        XCTAssertEqual(expanded.count, totalFrames * dim)

        var currentFrame = 0
        var ph = 0
        while ph < phonemeCount {
            let dur = durations[ph]
            var f = 0
            while f < dur {
                var d = 0
                while d < dim {
                    let expectedVal = Float(ph * 1000 + d + 1)
                    let actualVal = expanded[(currentFrame + f) * dim + d]
                    XCTAssertEqual(actualVal, expectedVal, "フレーム \(currentFrame + f) の dim \(d) が不一致")
                    d += 1
                }
                f += 1
            }
            currentFrame += dur
            ph += 1
        }
    }

    // MARK: - 4. expand の境界値・ゼロ Duration 耐性検証

    func testExpandBoundaryConditions() {
        // duration に 0 や負数が含まれる場合でもクラッシュせず安全にスキップされるか検証する。
        let dim = 64
        let phonemeCount = 3
        let durations = [2, 0, 3] // 合計 5 フレーム (第1音素は0フレーム)
        let embeddings = [Float](repeating: 7.0, count: phonemeCount * dim)

        let expanded = lengthRegulator.expand(
            embeddings: embeddings,
            phonemeCount: phonemeCount,
            durations: durations
        )

        XCTAssertEqual(expanded.count, 5 * dim)

        // phonemeCount <= 0 の場合
        let emptyExpanded = lengthRegulator.expand(
            embeddings: embeddings,
            phonemeCount: 0,
            durations: durations
        )
        XCTAssertTrue(emptyExpanded.isEmpty)
    }

    // MARK: - 5. ProsodyModel F0 輪郭の無声子音・ポーズ厳密 0.0 Hz 検証

    func testF0ContourUnvoicedAndPauseStrictZero() {
        // SNN 音響モデルで声帯振動が存在しない無声区間に微小な浮動小数点ゴミが混入すると
        // 不自然なクリック雑音や誤った有声判定が発生するため、厳密に 0.0 Hz であることを検証する。
        let testTexts = [
            "すしを食べたい。",
            "きっぷを買った。",
            "しっかりとしてください！",
            "東京、京都、大阪。"
        ]

        for text in testTexts {
            let features = lengthRegulator.processText(
                text: text,
                normalizer: normalizer,
                prosodyModel: prosodyModel,
                vocabulary: vocabulary
            )

            XCTAssertTrue(0 < features.totalFrames)
            XCTAssertEqual(features.f0Contour.count, features.totalFrames)
            XCTAssertEqual(features.voicedFlags.count, features.totalFrames)

            var i = 0
            while i < features.totalFrames {
                let f0 = features.f0Contour[i]
                let voiced = features.voicedFlags[i]

                if voiced == 0.0 {
                    // 無声区間では F0 が厳密に 0.0 (bit-exact 0.0f) であること
                    XCTAssertEqual(f0, 0.0, "無声区間なのに F0 が非ゼロです: \(f0) at frame \(i)")
                } else {
                    // 有声区間では 1.0 かつ F0 が正常な音声周波数帯域 (50Hz〜500Hz) に収まっていること
                    XCTAssertEqual(voiced, 1.0)
                    XCTAssertTrue(50.0 <= f0, "F0 が異常に低すぎます: \(f0)")
                    XCTAssertTrue(f0 <= 500.0, "F0 が異常に高すぎます: \(f0)")
                }
                i += 1
            }
        }
    }

    // MARK: - 6. 有声区間内での F0 連続性・平滑性（微分値・段差）の定量的検証

    func testF0SmoothnessInVoicedSegments() {
        // 有声音が連続する区間（母音連続やアクセント核の遷移）において、
        // 急激なステップ段差が生じていないか定量的に計測する。
        let text = "あおいあい" // 全て有声音 (母音連続: a-o-i-a-i)
        let features = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )

        var maxDeltaF0: Float = 0.0
        var i = 1
        while i < features.totalFrames {
            let vPrev = features.voicedFlags[i - 1]
            let vCurr = features.voicedFlags[i]

            // 連続して有声である区間のみを対象にフレーム間差分を計算
            if vPrev == 1.0 && vCurr == 1.0 {
                let f0Prev = features.f0Contour[i - 1]
                let f0Curr = features.f0Contour[i]
                let delta = abs(f0Curr - f0Prev)
                if maxDeltaF0 < delta {
                    maxDeltaF0 = delta
                }
            }
            i += 1
        }

        // 1フレーム (10ms) あたりの最大 F0 変動量を検証
        // 生理学的な発声において 10ms あたり 50Hz 以上の急激なジャンプは不自然
        print("連続有声区間における最大 F0 フレーム間差分: \(maxDeltaF0) Hz/10ms")
        XCTAssertTrue(maxDeltaF0 <= 50.0, "F0 に 50Hz/10ms を超える急激な不連続段差があります: \(maxDeltaF0)")
    }

    // MARK: - 7. 藤崎モデル風アクセント成分の立ち上がり・立ち下がり挙動の検証

    func testAccentComponentBehaviorAcrossPhrases() {
        // 頭高型（HL）および平板型（LHH）において、トーン変化時の F0 挙動が
        // 意図通りに動作しているか確認する。
        let tones = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 1) // 頭高型 [H, L, L]
        XCTAssertEqual(tones, [.high, .low, .low])

        let m1 = MoraToken(text: "あ", phonemes: [PhonemeToken(id: 5, symbol: "a", category: .vowel, durationFrames: 10)], tone: .high)
        let m2 = MoraToken(text: "め", phonemes: [PhonemeToken(id: 15, symbol: "m", category: .consonant, durationFrames: 4), PhonemeToken(id: 8, symbol: "e", category: .vowel, durationFrames: 8)], tone: .low)

        let phrase = AccentPhrase(moras: [m1, m2])
        let (f0List, voicedList, totalFrames) = prosodyModel.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocabulary
        )

        XCTAssertEqual(totalFrames, 22) // 10 + 4 + 8
        XCTAssertEqual(f0List.count, 22)
        XCTAssertEqual(voicedList.count, 22)

        // 1拍目 (High) の有声部での平均 F0 と 2拍目 (Low) の有声部での平均 F0 を比較
        var sumF0High: Float = 0.0
        var countHigh: Float = 0.0
        var hIdx = 0
        while hIdx < 10 {
            if voicedList[hIdx] == 1.0 {
                sumF0High += f0List[hIdx]
                countHigh += 1.0
            }
            hIdx += 1
        }
        let avgHigh = sumF0High / countHigh

        var sumF0Low: Float = 0.0
        var countLow: Float = 0.0
        var lIdx = 10
        while lIdx < 22 {
            if voicedList[lIdx] == 1.0 {
                sumF0Low += f0List[lIdx]
                countLow += 1.0
            }
            lIdx += 1
        }
        let avgLow = sumF0Low / countLow

        // High トーン区間の方が Low トーン区間よりも高いピッチになっていること
        XCTAssertTrue(avgLow < avgHigh, "頭高型アクセントにおいて High 区間の平均ピッチが Low 区間より高くなっていません")
    }

    // MARK: - 8. processText における speedFactor と総フレーム数の整合性検証

    func testProcessTextSpeedFactorTotalFrames() {
        // 発話速度を変更した際、全音素のフレーム数および F0 系列の長さが
        // totalFrames と厳密に完全一致しているか確認する。
        let text = "音声合成の実験です。"
        let speeds: [Float] = [0.8, 1.0, 1.2, 1.5]

        for spd in speeds {
            let features = lengthRegulator.processText(
                text: text,
                normalizer: normalizer,
                prosodyModel: prosodyModel,
                vocabulary: vocabulary,
                speedFactor: spd
            )

            let durationSum = features.durations.reduce(0, +)
            XCTAssertEqual(Int(durationSum), features.totalFrames, "durations の総和が totalFrames と不一致 (speed: \(spd))")
            XCTAssertEqual(features.f0Contour.count, features.totalFrames, "f0Contour の長さが totalFrames と不一致 (speed: \(spd))")
            XCTAssertEqual(features.voicedFlags.count, features.totalFrames, "voicedFlags の長さが totalFrames と不一致 (speed: \(spd))")
        }
    }

    // MARK: - 9. 敵対的調査: アクセント核直後の F0 不連続段差 (Cliff Jump) の精密測定

    func testAccentKernelF0DiscontinuityMeasurement() {
        // ProsodyModel の accentComp が High->Low 遷移時に時定数減衰を伴わず
        // 0.0 に即時リセットされるため、有声連続区間での F0 段差の大きさを定量化する。
        // 例: 「いのち」は中高型 [L, H, L] (すべて有声音: i - no - chi[ch(無声)+i(有声)])
        // 「たまご」は中高型 [L, H, L] (すべて有声音: t(無声)+a(有声) - m(有声)+a(有声) - g(有声)+o(有声))
        // 「まご」の部分は完全に連続有声音 (m-a-g-o)
        let text = "たまご"
        let features = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )

        var maxCliffDelta: Float = 0.0
        var cliffFrameIndex = -1
        var f0BeforeCliff: Float = 0.0
        var f0AfterCliff: Float = 0.0

        var i = 1
        while i < features.totalFrames {
            if features.voicedFlags[i - 1] == 1.0 && features.voicedFlags[i] == 1.0 {
                let prev = features.f0Contour[i - 1]
                let curr = features.f0Contour[i]
                let delta = abs(curr - prev)
                if maxCliffDelta < delta {
                    maxCliffDelta = delta
                    cliffFrameIndex = i
                    f0BeforeCliff = prev
                    f0AfterCliff = curr
                }
            }
            i += 1
        }

        print("--- [Challenger F0 Measurement] ---")
        print("テキスト: \(text)")
        print("有声連続部での最大段差: \(maxCliffDelta) Hz at frame \(cliffFrameIndex)")
        print("段差直前 F0: \(f0BeforeCliff) Hz, 段差直後 F0: \(f0AfterCliff) Hz")
        print("落差比率: \((abs(f0BeforeCliff - f0AfterCliff) / f0BeforeCliff) * 100.0)%")
        print("-----------------------------------")
    }

    // MARK: - 10. 敵対的調査: processText での累積誤差 (Round Drift) の実測

    func testProcessTextCumulativeRoundingDriftMeasurement() {
        // defaultDurationFrames が個別 roundf を行い、quantizeDurations が
        // パイプライン内でバイパスされていることによるドリフト量を定量化する。
        let text = "東京都千代田区永田町一丁目の一番地における音声合成エンジンの連続耐久テストを実施します。"
        let speed: Float = 1.35 // 非整数スピード

        let features = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: speed
        )

        // もし quantizeDurations を通した場合の期待フレーム数
        let morphemes = normalizer.normalize(text: text)
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)

        var rawFloatDurations: [Float] = []
        for phrase in phrases {
            for mora in phrase.moras {
                for token in mora.phonemes {
                    var base: Float = 8.0
                    switch token.category {
                    case .vowel: base = 8.0
                    case .consonant: base = 4.0
                    case .contracted: base = 5.0
                    case .geminate: base = 10.0
                    case .nasalSyllable: base = 8.0
                    case .prolonged: base = 9.0
                    case .pause:
                        if token.symbol == "<sil>" {
                            base = 30.0
                        } else {
                            base = 15.0
                        }
                    }
                    var scaled = base / speed
                    if scaled < 1.0 {
                        scaled = 1.0
                    }
                    rawFloatDurations.append(scaled)
                }
            }
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                rawFloatDurations.append(Float(phrase.pauseDurationFrames))
            }
        }

        let sumRawFloats = rawFloatDurations.reduce(0.0, +)
        let idealCumulativeTotal = Int(roundf(sumRawFloats))
        let actualProcessTextTotal = features.totalFrames

        let drift = abs(actualProcessTextTotal - idealCumulativeTotal)
        print("--- [Challenger Rounding Drift Measurement] ---")
        print("音素総数: \(rawFloatDurations.count)")
        print("浮動小数点 Duration の理想総和: \(sumRawFloats)")
        print("理想累積和量子化フレーム数: \(idealCumulativeTotal)")
        print("processText の実際フレーム数: \(actualProcessTextTotal)")
        print("累積丸めドリフトフレーム数: \(drift) フレーム (\(Float(drift) * 10.0) ms)")
        print("-----------------------------------------------")
    }
}
