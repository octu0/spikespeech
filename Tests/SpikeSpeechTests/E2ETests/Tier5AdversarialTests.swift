import XCTest
@testable import SpikeSpeech

/// 日本語言語処理フロントエンド (M1: F1〜F5) に対する敵対的・極限境界値・ストレステストスイート
///
/// 単体テストでは捕捉できない制御文字、絵文字、長文挙動、助詞誤爆、
/// 極大数値オーバーフロー、異常パラメータ等の入力に対するエンジンの安全性を検証する。
final class Tier5AdversarialTests: XCTestCase {

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

    // MARK: - 1. 極限・敵対的文字列入力テスト (Crash / Hang / NaN 防御)

    func testExtremeAdversarialInputs() {
        // 実環境での予期せぬ入力（制御コード、絵文字、未知漢字）によって
        // メモリアクセス違反やクラッシュが発生しないことを検証する。

        let adversarialCases: [(name: String, text: String)] = [
            ("EmptyString", ""),
            ("SingleSpace", " "),
            ("MultipleSpaces", "    "),
            ("FullWidthSpaces", "　　　"),
            ("TabsAndNewlines", "\t\r\n\t\n"),
            ("ControlCharsNUL", "\0\u{0001}\u{0007}\u{0008}\u{001B}"),
            ("AnsiColorSequences", "\u{001B}[31;1m赤いテキスト\u{001B}[0m"),
            ("SurrogatePairKanji", "𠮷野家で𪚥を食べる"), // 𠮷 (U+20BB7), 𪚥 (U+2A6A5 64画)
            ("EmojiSequences", "こんにちは🍣🍵🇯🇵👨‍👩‍👧‍👦🎉"),
            ("OnlySymbols", "!@#$%^&*()_+~|}{[]:;?><,./"),
            ("RepeatedPunctuation", "。。。。、、、、？？？？！！！！"),
            ("UnknownKanjiOnly", "鬱鬱鬱鬱鬱鬱鬱鬱鬱鬱"),
            ("ObscureCompoundKanji", "魑魅魍魎跋扈の様相を呈す"),
            ("MixedAdversarial", "\0 　🍣東京\u{001B}[32mへ行く！\t\r\n(笑)"),
        ]

        var cIdx = 0
        while cIdx < adversarialCases.count {
            let tc = adversarialCases[cIdx]
            let features = lengthRegulator.processText(
                text: tc.text,
                normalizer: normalizer,
                prosodyModel: prosodyModel,
                vocabulary: vocabulary
            )

            // 総フレーム数と各配列の整合性確認
            XCTAssertEqual(features.f0Contour.count, features.totalFrames, "F0 count mismatch in \(tc.name)")
            XCTAssertEqual(features.voicedFlags.count, features.totalFrames, "Voiced count mismatch in \(tc.name)")

            // NaN / Inf の不在確認
            var f = 0
            while f < features.totalFrames {
                let f0 = features.f0Contour[f]
                let voiced = features.voicedFlags[f]
                XCTAssertFalse(f0.isNaN, "NaN F0 detected in \(tc.name) at frame \(f)")
                XCTAssertFalse(f0.isInfinite, "Infinite F0 detected in \(tc.name) at frame \(f)")
                XCTAssertFalse(voiced.isNaN, "NaN voiced flag detected in \(tc.name) at frame \(f)")
                f += 1
            }

            // 音素 ID の語彙範囲 [0, 64) の確認
            var p = 0
            while p < features.phoneIds.count {
                let pid = features.phoneIds[p]
                XCTAssertTrue(0 <= pid, "Negative phone ID in \(tc.name)")
                XCTAssertTrue(pid < 64, "Out of vocabulary phone ID in \(tc.name)")
                p += 1
            }

            cIdx += 1
        }
    }

    // MARK: - 2. 助詞「は/へ/を」および紛らわしい文の判定テスト

    func testParticleAndConfusingSentences() {
        // 形態素解析と助詞置換が品詞境界を正しく判別し、単語内部の「は」「へ」を
        // 破壊することなく、真の助詞のみを正確に発音変換することを検証する。

        // ケース 1: 「母は花を買う」
        // 「母(はは)」の2拍目「は」は保護され、助詞「は」のみ「わ」になり、「花」の「は」も保護される
        let text1 = "母は花を買う"
        let norm1 = normalizer.normalize(text: text1)
        let reading1 = norm1.map { $0.reading }.joined()
        XCTAssertEqual(reading1, "ははわはなおかう")

        // ケース 2: 「へびが部屋へ行く」
        // 「部屋(へや)」の「へ」は名詞内部であり保護され、助詞「へ」のみ「え」になる
        let text2 = "へびが部屋へ行く"
        let norm2 = normalizer.normalize(text: text2)
        let reading2 = norm2.map { $0.reading }.joined()
        XCTAssertTrue(reading2.contains("へやえいく"), "Reading should be へやえいく, got \(reading2)")

        // ケース 3: 助詞連続（「はははは」「へへへ」「ををを」）でクラッシュしないこと
        let textRepH = "はははは"
        let normRepH = normalizer.normalize(text: textRepH)
        XCTAssertTrue(0 < normRepH.count)

        let textRepHe = "へへへ"
        let normRepHe = normalizer.normalize(text: textRepHe)
        XCTAssertTrue(0 < normRepHe.count)

        let textRepO = "ををを"
        let normRepO = normalizer.normalize(text: textRepO)
        XCTAssertTrue(0 < normRepO.count)

        // ケース 4: 文頭に助詞文字が来るケース（文法不完全入力）
        let textStartParticle = "は水です"
        let normStart = normalizer.normalize(text: textStartParticle)
        XCTAssertTrue(0 < normStart.count)

        // ケース 5: 著名な早口言葉・同音連続
        let textSumomo = "すもももももももものうち"
        let normSumomo = normalizer.normalize(text: textSumomo)
        let readingSumomo = normSumomo.map { $0.reading }.joined()
        XCTAssertTrue(0 < readingSumomo.count)
    }

    // MARK: - 3. 数値・助数詞・極限境界値テスト

    func testNumericAndCounterBoundaryStress() {
        // 0、負数、小数、16桁境界値、極大桁数、ゼロパディング、未知助数詞が
        // 整数オーバーフローや無限ループを起こさず安全に展開されることを検証する。

        // 1. ゼロ
        XCTAssertEqual(normalizer.expandNumbersAndCounters("0"), "ぜろ")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("007"), "ぜろぜろなな")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("0人"), "ぜろにん")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("0円"), "ぜろえん")

        // 助数詞「円」「分」「秒」の連声・展開検証 (バグ検出対象)
        XCTAssertEqual(normalizer.expandNumbersAndCounters("100円"), "ひゃくえん")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("1分"), "いっぷん")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("3分"), "さんぷん")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("10秒"), "じゅうびょう")

        // 2. 12桁（兆未満の最大クラス）
        let num12 = "999999999999"
        let exp12 = normalizer.expandNumbersAndCounters(num12)
        XCTAssertTrue(exp12.contains("おく"))
        XCTAssertTrue(exp12.contains("まん"))

        // 3. 16桁（万進法 Int64 許容上限）
        let num16 = "9999999999999999"
        let exp16 = normalizer.expandNumbersAndCounters(num16)
        XCTAssertTrue(exp16.contains("ちょう"))

        // 4. 17桁以上（1桁読み上げフォールバック）
        let num17 = "10000000000000000"
        let exp17 = normalizer.expandNumbersAndCounters(num17)
        XCTAssertEqual(exp17, "いちぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろ")

        // 5. 負数・小数の表記
        let negNum = normalizer.expandNumbersAndCounters("-100円")
        XCTAssertEqual(negNum, "-ひゃくえん")

        let floatNum = normalizer.expandNumbersAndCounters("3.14")
        XCTAssertEqual(floatNum, "さん.じゅうよん")

        // 6. 未知の助数詞
        let unknownCounter = normalizer.expandNumbersAndCounters("100箱")
        XCTAssertEqual(unknownCounter, "ひゃく箱")

        // 7. 特殊助数詞の連声網羅
        XCTAssertEqual(normalizer.expandNumbersAndCounters("1匹"), "いっぴき")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("3匹"), "さんびき")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("6匹"), "ろっぴき")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("8匹"), "はっぴき")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("10匹"), "じゅっぴき")

        XCTAssertEqual(normalizer.expandNumbersAndCounters("1個"), "いっこ")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("6個"), "ろっこ")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("8個"), "はっこ")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("10個"), "じゅっこ")

        XCTAssertEqual(normalizer.expandNumbersAndCounters("1階"), "いっかい")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("3階"), "さんがい")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("1杯"), "いっぱい")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("3杯"), "さんばい")

        XCTAssertEqual(normalizer.expandNumbersAndCounters("1か月"), "いっかげつ")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("6か月"), "ろっかげつ")

        XCTAssertEqual(normalizer.expandNumbersAndCounters("20日"), "はつか")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("14日"), "じゅうよっか")
        XCTAssertEqual(normalizer.expandNumbersAndCounters("24日"), "にじゅうよっか")
    }

    // MARK: - 4. Length Regulator & Prosody 異常値耐性テスト

    func testLengthRegulatorAndProsodyAbnormalInputs() {
        // 音響モデルの推論過程で異常値（負の継続時間、空配列、異常なアクセント核）が
        // 渡された場合でも、メモリ破壊やパニックを起こさないことを検証する。

        // 1. quantizeDurations: 負数や0を含む継続時間
        let negDurations: [Float] = [-2.0, 0.0, 5.5, -1.0]
        let quantizedNeg = lengthRegulator.quantizeDurations(durations: negDurations)
        XCTAssertEqual(quantizedNeg.count, 4)
        for q in quantizedNeg {
            XCTAssertTrue(1 <= q, "Quantized duration must be at least 1 frame")
        }

        // 2. quantizeDurations: 空配列
        let emptyQuant = lengthRegulator.quantizeDurations(durations: [])
        XCTAssertTrue(emptyQuant.isEmpty)

        // 3. expand: phonemeCount = 0
        let expZeroPhonemes = lengthRegulator.expand(embeddings: [1.0, 2.0], phonemeCount: 0, durations: [1])
        XCTAssertTrue(expZeroPhonemes.isEmpty)

        // 4. expand: durations がすべて 0 または負数
        let expZeroDurs = lengthRegulator.expand(
            embeddings: [Float](repeating: 1.0, count: 128),
            phonemeCount: 2,
            durations: [0, -1]
        )
        XCTAssertTrue(expZeroDurs.isEmpty)

        // 5. computeMoraTones: 負数や範囲外の accentKernel
        let tonesNegKernel = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: -1)
        XCTAssertEqual(tonesNegKernel.count, 3)

        let tonesHugeKernel = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 999)
        XCTAssertEqual(tonesHugeKernel.count, 3)

        let tonesZeroMoras = prosodyModel.computeMoraTones(moraCount: 0, accentKernel: 1)
        XCTAssertTrue(tonesZeroMoras.isEmpty)

        let tonesNegMoras = prosodyModel.computeMoraTones(moraCount: -5, accentKernel: 1)
        XCTAssertTrue(tonesNegMoras.isEmpty)

        // 6. generateF0Contour: 空のアクセント句リスト
        let (f0Empty, voicedEmpty, totalEmpty) = prosodyModel.generateF0Contour(phrases: [], vocabulary: vocabulary)
        XCTAssertEqual(totalEmpty, 0)
        XCTAssertTrue(f0Empty.isEmpty)
        XCTAssertTrue(voicedEmpty.isEmpty)
    }

    // MARK: - 5. 長文ストレス・スケーリング耐性テスト

    func testLongTextScalingStress() {
        // 10,000文字の超長文入力において、形態素解析ラティス構築および
        // Viterbi探索がスタックオーバーフローやメモリ爆発を起こさず正常終了することを検証する。

        let sampleUnit = "東京の天気は晴れです。" // 11文字
        let repeatCount = 1000 // 11,000文字
        let longText = String(repeating: sampleUnit, count: repeatCount)

        let start = Date()
        let morphemes = normalizer.normalize(text: longText)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertTrue(0 < morphemes.count)
        // 処理が完了し、極端なメモリリークやクラッシュがないことを確認
        print("[StressTest] 11,000 chars normalized in \(elapsed)s, morphemes count: \(morphemes.count)")
    }
}
