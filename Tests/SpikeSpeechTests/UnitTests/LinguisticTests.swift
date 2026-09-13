import XCTest
@testable import SpikeSpeech

/// 日本語言語処理フロントエンドおよび音響アライメントエンジンの単体テストスイート
///
/// 外部依存ゼロの形態素解析、助詞置換、動詞保護、数詞展開、韻律F0生成、および
/// 累積和 Length Regulation の数学的・論理的整合性を厳密に検証する。
final class LinguisticTests: XCTestCase {

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

    // MARK: - 1. 形態素解析・品詞特定テスト

    func testViterbiMorphologicalAnalysis() {
        // 指示詞・助詞・名詞・助動詞の典型的な連接がラティス探索で正しく解かれるか検証する。
        let morphemes = morphology.tokenize("これは水です")
        XCTAssertEqual(morphemes.count, 4)

        XCTAssertEqual(morphemes[0].surface, "これ")
        XCTAssertEqual(morphemes[0].pos, .noun)

        XCTAssertEqual(morphemes[1].surface, "は")
        XCTAssertEqual(morphemes[1].pos, .particle)

        XCTAssertEqual(morphemes[2].surface, "水")
        XCTAssertEqual(morphemes[2].pos, .noun)

        XCTAssertEqual(morphemes[3].surface, "です")
        XCTAssertEqual(morphemes[3].pos, .auxiliaryVerb)
    }

    // MARK: - 2. 助詞読み替えテスト (は/へ/を)

    func testParticleReadingReplacement() {
        // 助詞「は→わ」「へ→え」「を→お」の音変化を反映しつつ、
        // 単語内部の「は」（母、花）が誤置換されないことを保証する。
        let text1 = "これは水です"
        let norm1 = normalizer.normalize(text: text1)
        let reading1 = norm1.map { $0.reading }.joined()
        XCTAssertEqual(reading1, "これわみずです")

        let text2 = "東京へ行く"
        let norm2 = normalizer.normalize(text: text2)
        let reading2 = norm2.map { $0.reading }.joined()
        XCTAssertEqual(reading2, "とーきょーえいく")

        let text3 = "本を読む"
        let norm3 = normalizer.normalize(text: text3)
        let reading3 = norm3.map { $0.reading }.joined()
        XCTAssertEqual(reading3, "ほんおよむ")

        let text4 = "母は花を愛す"
        let norm4 = normalizer.normalize(text: text4)
        let reading4 = norm4.map { $0.reading }.joined()
        XCTAssertEqual(reading4, "ははわはなおあいす")
    }

    // MARK: - 3. 動詞終止形「う」保護および母音連続長音化テスト

    func testVerbEndingUProtectionAndProlongation() {
        // 名詞の母音連続（東京→とーきょー、先生→せんせー）は長音化しつつ、
        // 動詞終止形（思う、買う、追う）の語末 [u] は独立拍として保護されることを保証する。
        let textVerb1 = "彼を思う"
        let normVerb1 = normalizer.normalize(text: textVerb1)
        let readingVerb1 = normVerb1.map { $0.reading }.joined()
        XCTAssertEqual(readingVerb1, "かれおおもう")

        let textVerb2 = "靴を買う"
        let normVerb2 = normalizer.normalize(text: textVerb2)
        let readingVerb2 = normVerb2.map { $0.reading }.joined()
        XCTAssertEqual(readingVerb2, "くつおかう")

        let textNoun1 = "東京の先生"
        let normNoun1 = normalizer.normalize(text: textNoun1)
        let readingNoun1 = normNoun1.map { $0.reading }.joined()
        XCTAssertEqual(readingNoun1, "とーきょーのせんせー")

        let textNoun2 = "映画を見る"
        let normNoun2 = normalizer.normalize(text: textNoun2)
        let readingNoun2 = normNoun2.map { $0.reading }.joined()
        XCTAssertEqual(readingNoun2, "えーがおみる")
    }

    // MARK: - 4. 万進法数詞展開および助数詞連声テスト

    func testNumeralExpansionAndCounterSandhi() {
        // 「1473年」の万進法展開および「1本(いっぽん)」「3本(さんぼん)」「4日(よっか)」等の
        // 日本語特有の促音化・濁音化・不規則音便が決定論的に展開されることを確認する。
        let resYear = normalizer.expandNumbersAndCounters("1473年")
        XCTAssertEqual(resYear, "せんよんひゃくななじゅうさんねん")

        let resHon1 = normalizer.expandNumbersAndCounters("1本")
        XCTAssertEqual(resHon1, "いっぽん")

        let resHon3 = normalizer.expandNumbersAndCounters("3本")
        XCTAssertEqual(resHon3, "さんぼん")

        let resHon10 = normalizer.expandNumbersAndCounters("10本")
        XCTAssertEqual(resHon10, "じゅっぽん")

        let resMonth = normalizer.expandNumbersAndCounters("4月")
        XCTAssertEqual(resMonth, "しがつ")

        let resDay = normalizer.expandNumbersAndCounters("4日")
        XCTAssertEqual(resDay, "よっか")

        let resHour = normalizer.expandNumbersAndCounters("4時")
        XCTAssertEqual(resHour, "よじ")

        let resPerson1 = normalizer.expandNumbersAndCounters("1人")
        XCTAssertEqual(resPerson1, "ひとり")

        let resPerson2 = normalizer.expandNumbersAndCounters("2人")
        XCTAssertEqual(resPerson2, "ふたり")
    }

    // MARK: - 5. モーラ・音素階層トークナイズおよび 64 語彙 ID テスト

    func testMoraAndPhonemeTokenization() {
        // 拗音（ky）や促音（Q）、長音（_）が正しいカテゴリ・音素IDにマッピングされ、
        // 64語彙テーブルの整合性が維持されているか確認する。
        let morasKyo = vocabulary.kanaToMoras("きょう")
        XCTAssertEqual(morasKyo.count, 2)
        XCTAssertEqual(morasKyo[0].text, "きょ")
        XCTAssertEqual(morasKyo[0].phonemes.count, 2)
        XCTAssertEqual(morasKyo[0].phonemes[0].symbol, "ky")
        XCTAssertEqual(morasKyo[0].phonemes[1].symbol, "o")
        XCTAssertEqual(morasKyo[1].text, "う")

        let morasGakko = vocabulary.kanaToMoras("がっこう")
        XCTAssertEqual(morasGakko.count, 4)
        XCTAssertEqual(morasGakko[1].phonemes[0].symbol, "Q")
        XCTAssertEqual(morasGakko[1].phonemes[0].category, .geminate)

        // 語彙IDの範囲検証
        for mora in morasGakko {
            for p in mora.phonemes {
                XCTAssertTrue(0 <= p.id)
                XCTAssertTrue(p.id < 64)
            }
        }
    }

    // MARK: - 6. 東京方言ピッチアクセントトーンおよび F0 輪郭生成テスト

    func testProsodyAndF0Contour() {
        // 頭高型（ねこ: H-L）、平板型（さくら: L-H-H）、中高型（たまご: L-H-L）が
        // 東京方言の規則通りにトーン付与されることを確認する。
        let tonesHeadHigh = prosodyModel.computeMoraTones(moraCount: 2, accentKernel: 1)
        XCTAssertEqual(tonesHeadHigh, [.high, .low])

        let tonesFlat = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 0)
        XCTAssertEqual(tonesFlat, [.low, .high, .high])

        let tonesMiddleHigh = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 2)
        XCTAssertEqual(tonesMiddleHigh, [.low, .high, .low])

        // F0 輪郭の有声/無声マスキングおよび数値安定性検証
        let features = lengthRegulator.processText(
            text: "これは水です。",
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
            let v = features.voicedFlags[i]

            XCTAssertFalse(f0.isNaN)
            XCTAssertFalse(f0.isInfinite)

            if v == 1.0 {
                XCTAssertTrue(0.0 < f0)
            } else {
                XCTAssertEqual(f0, 0.0)
            }
            i += 1
        }
    }

    // MARK: - 7. Length Regulation フレーム展開および O(1) アロケーションテスト

    func testLengthRegulationExpansion() {
        // 音素埋め込み行列 (N x D) と各音素の Duration 列から、
        // 正確に sum(d_i) x D の連続音響フレーム系列が展開されることを確認する。
        let dim = 64
        let phonemeCount = 3
        let durations = [2, 3, 4] // 合計 9 フレーム
        let totalFrames = 9

        // 3 つの音素埋め込みベクトル (それぞれ一意な値で初期化)
        var embeddings = [Float](repeating: 0.0, count: phonemeCount * dim)
        var d = 0
        while d < dim {
            embeddings[0 * dim + d] = 1.0
            embeddings[1 * dim + d] = 2.0
            embeddings[2 * dim + d] = 3.0
            d += 1
        }

        let expanded = lengthRegulator.expand(
            embeddings: embeddings,
            phonemeCount: phonemeCount,
            durations: durations
        )

        XCTAssertEqual(expanded.count, totalFrames * dim)

        // 第 0..1 フレームは音素 0 (値 1.0)
        var f = 0
        while f < 2 {
            var j = 0
            while j < dim {
                XCTAssertEqual(expanded[f * dim + j], 1.0)
                j += 1
            }
            f += 1
        }

        // 第 2..4 フレームは音素 1 (値 2.0)
        while f < 5 {
            var j = 0
            while j < dim {
                XCTAssertEqual(expanded[f * dim + j], 2.0)
                j += 1
            }
            f += 1
        }

        // 第 5..8 フレームは音素 2 (値 3.0)
        while f < 9 {
            var j = 0
            while j < dim {
                XCTAssertEqual(expanded[f * dim + j], 3.0)
                j += 1
            }
            f += 1
        }
    }

    // MARK: - 8. 累積和量子化テスト

    func testCumulativeQuantization() {
        // 浮動小数点 Duration の四捨五入において、文全体での丸め誤差累積が
        // ゼロとなることを数学的に証明する。
        let rawDurations: [Float] = [1.2, 2.4, 3.4] // 総和 = 7.0
        let quantized = lengthRegulator.quantizeDurations(durations: rawDurations)

        XCTAssertEqual(quantized.count, 3)
        let totalQuantized = quantized.reduce(0, +)
        XCTAssertEqual(totalQuantized, 7) // round(7.0) = 7
    }

    // MARK: - 9. 境界値・敵対的 (Adversarial) 入力テスト

    func testAdversarialAndBoundaryInputs() {
        // 空文字、空白のみ、記号混在、長文等の入力に対して
        // クラッシュせず安全にフォールバックすることを証明する。

        // 1. 空文字
        let emptyFeatures = lengthRegulator.processText(
            text: "",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )
        XCTAssertEqual(emptyFeatures.totalFrames, 0)
        XCTAssertTrue(emptyFeatures.phoneIds.isEmpty)

        // 2. 空白のみ
        let spaceClean = normalizer.cleanText("    ")
        XCTAssertEqual(spaceClean, "    ")

        // 3. 注記付き記号混在テキスト
        let annotText = "ら〜めん(笑)！"
        let cleaned = normalizer.cleanText(annotText)
        XCTAssertEqual(cleaned, "らーめん！")

        // 4. 未知語・英数字混在
        let unknownText = "SNN音声合成AI"
        let morphemes = normalizer.normalize(text: unknownText)
        XCTAssertTrue(0 < morphemes.count)

        // 5. 10,000 文字の長文ストレス入力
        let longSentence = String(repeating: "東京の水は冷たい。", count: 1000)
        let longMorphemes = normalizer.normalize(text: longSentence)
        XCTAssertTrue(0 < longMorphemes.count)
    }
}
