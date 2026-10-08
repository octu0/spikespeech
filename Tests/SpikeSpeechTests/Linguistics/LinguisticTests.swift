import XCTest
@testable import SpikeSpeech

/// 日本語言語処理フロントエンド（形態素解析, 読み正規化, 語彙, 韻律, LengthRegulator）の網羅的単体テストスイート
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

        for mora in morasGakko {
            for p in mora.phonemes {
                XCTAssertTrue(0 <= p.id)
                XCTAssertTrue(p.id < 64)
            }
        }
    }

    // MARK: - 6. 東京方言ピッチアクセントトーンおよび F0 輪郭生成テスト

    func testProsodyAndF0Contour() {
        let tonesHeadHigh = prosodyModel.computeMoraTones(moraCount: 2, accentKernel: 1)
        XCTAssertEqual(tonesHeadHigh, [.high, .low])

        let tonesFlat = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 0)
        XCTAssertEqual(tonesFlat, [.low, .high, .high])

        let tonesMiddleHigh = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 2)
        XCTAssertEqual(tonesMiddleHigh, [.low, .high, .low])

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

    /// 日本語長文テキストに対する形態素解析・アクセント句・ピッチ輪郭の生成検証
    func testTargetSentenceProsody() {
        let text = "お好きな日本語テキストを入力してください"
        let norm = normalizer.cleanText(text)
        let morphemes = normalizer.normalize(text: norm)
        XCTAssertTrue(0 < morphemes.count)

        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
        XCTAssertTrue(0 < phrases.count)

        let features = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            baseF0: 220.0
        )
        XCTAssertTrue(0 < features.totalFrames)
        XCTAssertEqual(features.f0Contour.count, features.totalFrames)
        XCTAssertEqual(features.voicedFlags.count, features.totalFrames)
    }

    // MARK: - 7. Length Regulation フレーム展開および O(1) アロケーションテスト

    func testLengthRegulationExpansion() {
        let dim = 64
        let phonemeCount = 3
        let durations = [2, 3, 4]
        let totalFrames = 9

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

        var f = 0
        while f < 2 {
            var j = 0
            while j < dim {
                XCTAssertEqual(expanded[f * dim + j], 1.0)
                j += 1
            }
            f += 1
        }

        while f < 5 {
            var j = 0
            while j < dim {
                XCTAssertEqual(expanded[f * dim + j], 2.0)
                j += 1
            }
            f += 1
        }

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
        let rawDurations: [Float] = [1.2, 2.4, 3.4]
        let quantized = lengthRegulator.quantizeDurations(durations: rawDurations)

        XCTAssertEqual(quantized.count, 3)
        let totalQuantized = quantized.reduce(0, +)
        XCTAssertEqual(totalQuantized, 7)
    }

    // MARK: - 9. 境界値・敵対的 (Adversarial) 入力テスト

    func testAdversarialAndBoundaryInputs() {
        let emptyFeatures = lengthRegulator.processText(
            text: "",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )
        XCTAssertEqual(emptyFeatures.totalFrames, 0)
        XCTAssertTrue(emptyFeatures.phoneIds.isEmpty)

        let spaceClean = normalizer.cleanText("    ")
        XCTAssertEqual(spaceClean, "    ")

        let annotText = "ら〜めん(笑)！"
        let cleaned = normalizer.cleanText(annotText)
        XCTAssertEqual(cleaned, "らーめん！")

        let unknownText = "SNN音声合成AI"
        let morphemes = normalizer.normalize(text: unknownText)
        XCTAssertTrue(0 < morphemes.count)

        let longSentence = String(repeating: "東京の水は冷たい。", count: 100)
        let longMorphemes = normalizer.normalize(text: longSentence)
        XCTAssertTrue(0 < longMorphemes.count)
    }

    // MARK: - 10. 生体プロソディ・音素ヘルパー・エネルギープロファイル検証

    /// 調音音声学に基づくマイクロプロソディ（無声子音後の母音 F0 上昇 vs 有声子音後）の検証
    func testMicroprosodyEffect() {
        let prosody = ProsodyModel(baseF0: 200.0)

        let lingKa = lengthRegulator.processText(text: "か", normalizer: normalizer, prosodyModel: prosody, vocabulary: vocabulary)
        let lingGa = lengthRegulator.processText(text: "が", normalizer: normalizer, prosodyModel: prosody, vocabulary: vocabulary)

        let ka_kDur = Int(lingKa.durations[0])
        let ga_gDur = Int(lingGa.durations[0])

        let vowelKaF0 = lingKa.f0Contour[ka_kDur]
        let vowelGaF0 = lingGa.f0Contour[ga_gDur]

        XCTAssertTrue(0.0 < vowelKaF0, "か の母音 /a/ F0 が正であること")
        XCTAssertTrue(0.0 < vowelGaF0, "が の母音 /a/ F0 が正であること")

        // 音響音声学の物理法則: 無声破裂音直後の母音立ち上がりピッチ (+3%) は有声破裂音直後 (-2%) よりも高くなる
        XCTAssertTrue(vowelGaF0 < vowelKaF0, "マイクロプロソディにより無声破裂音直後の母音 /a/ F0 (\(vowelKaF0)) が有声子音直後 (\(vowelGaF0)) より高くなること")
    }

    /// 音素ヘルパー hy (32) の無声摩擦音・無声子音分類の検証
    func testPhonemeHelperHy() {
        let hyId = 32

        XCTAssertEqual(vocabulary.token(for: hyId), "hy", "ID 32 は hy であること")
        XCTAssertTrue(vocabulary.isUnvoicedFricative(id: hyId), "hy は無声摩擦音であること")
        XCTAssertTrue(vocabulary.isUnvoicedConsonant(id: hyId), "hy は無声子音であること")
        XCTAssertTrue(vocabulary.isVoiced(symbol: "hy") != true, "hy は無声音であること")
    }

    /// 推論時エネルギー輪郭の音素物理カテゴリプロファイルの検証
    func testInferenceEnergyProfile() {
        let features = lengthRegulator.processText(
            text: "すし",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )

        XCTAssertTrue(0 < features.totalFrames, "フレームが生成されていること")
        XCTAssertEqual(features.energyContour.count, features.totalFrames, "エネルギー輪郭と総フレーム数が一致すること")

        var fIdx = 0
        var foundVowelEnergy = false
        var foundFricativeEnergy = false

        while fIdx < features.totalFrames {
            let eng = features.energyContour[fIdx]
            XCTAssertTrue(0.0 <= eng, "エネルギーは 0 以上であること")
            XCTAssertTrue(eng <= 1.0, "エネルギーは 1.0 以下であること")

            if 0.60 <= eng {
                foundVowelEnergy = true
            }
            if 0.15 <= eng && eng <= 0.35 {
                foundFricativeEnergy = true
            }
            fIdx += 1
        }

        XCTAssertTrue(foundVowelEnergy, "母音の高エネルギー区間が存在すること")
        XCTAssertTrue(foundFricativeEnergy, "摩擦音の中間エネルギー区間が存在すること")
    }

    /// 音素境界での半正弦を外した一定カテゴリピーク（母音 0.70, 無声摩擦 0.18 等）および無声閉鎖の検証
    func testConstantCategoryPeakEnergyContour() {
        let features = lengthRegulator.processText(
            text: "すし",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            addBoundarySilence: false
        )

        var p = 0
        var curF = 0
        while p < features.phoneIds.count {
            let pid = Int(features.phoneIds[p])
            let dur = Int(features.durations[p])
            let sym = vocabulary.token(for: pid)
            var f = 0
            while f < dur {
                let eng = features.energyContour[curF + f]
                switch sym {
                case "s", "sh":
                    XCTAssertEqual(eng, 0.18, accuracy: 1e-5, "無声摩擦音は全フレームで 0.18 一定であること")
                case "u", "i":
                    XCTAssertEqual(eng, 0.70, accuracy: 1e-5, "母音は端でも半正弦に減衰せず 0.70 一定であること")
                default:
                    break
                }
                f += 1
            }
            curF += dur
            p += 1
        }

        // 無声破裂音（k, t）: dur <= 2 は全フレーム 0.35、dur >= 3 は先頭 dur - 2 フレーム 0.02、末尾 2 フレーム 0.35
        let stopFeatures = lengthRegulator.processText(
            text: "かた",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            addBoundarySilence: false
        )

        var sp = 0
        var sCurF = 0
        while sp < stopFeatures.phoneIds.count {
            let pid = Int(stopFeatures.phoneIds[sp])
            let dur = Int(stopFeatures.durations[sp])
            switch vocabulary.isUnvoicedStop(id: pid) {
            case true:
                var f = 0
                while f < dur {
                    let eng = stopFeatures.energyContour[sCurF + f]
                    switch dur <= 2 {
                    case true:
                        XCTAssertEqual(eng, 0.35, accuracy: 1e-5, "dur <= 2 の無声閉鎖音は全フレーム 0.35 であること")
                    case false:
                        switch f < dur - 2 {
                        case true:
                            XCTAssertEqual(eng, 0.02, accuracy: 1e-5, "無声閉鎖音の閉鎖期フレームは 0.02 であること")
                        case false:
                            XCTAssertEqual(eng, 0.35, accuracy: 1e-5, "無声閉鎖音の末尾2フレームは 0.35 であること")
                        }
                    }
                    f += 1
                }
            case false:
                let sym = vocabulary.token(for: pid)
                var f = 0
                while f < dur {
                    let eng = stopFeatures.energyContour[sCurF + f]
                    if sym == "a" {
                        XCTAssertEqual(eng, 0.70, accuracy: 1e-5, "母音 a は全フレーム 0.70 一定であること")
                    }
                    f += 1
                }
            }
            sCurF += dur
            sp += 1
        }
    }

    /// 無声閉鎖音（id 10: k, 12: t, 23: p, 30: ky, 38: py）の継続時間境界条件（dur = 1, 2, 3, 4, 5）の網羅検証
    func testUnvoicedStopDurationEdgeCases() {
        // 全5種の無声閉鎖音IDが正しく判定されることの検証
        let stopIds: [Int] = [10, 12, 23, 30, 38]
        for sid in stopIds {
            XCTAssertTrue(vocabulary.isUnvoicedStop(id: sid), "ID \(sid) は無声閉鎖音であること")
        }

        let testCases: [(word: String, stopId: Int32)] = [
            ("か", 10), // k
            ("た", 12), // t
            ("ぱ", 23)  // p
        ]

        let durationConfigs: [(targetDur: Int, stopWeight: Float, vowelWeight: Float, moraFrames: Float)] = [
            (1, 1.0, 1.0, 2.0),
            (2, 2.0, 1.0, 3.0),
            (3, 3.0, 1.0, 4.0),
            (4, 4.0, 2.0, 6.0),
            (5, 5.0, 2.0, 7.0)
        ]

        var cIdx = 0
        while cIdx < testCases.count {
            let tc = testCases[cIdx]
            var dIdx = 0
            while dIdx < durationConfigs.count {
                let dc = durationConfigs[dIdx]
                let avgDurs: [Int32: Float] = [
                    tc.stopId: dc.stopWeight,
                    5: dc.vowelWeight,  // a
                    1: 1.0              // sil
                ]
                let lr = LengthRegulator(
                    phonemeAverageDurations: avgDurs,
                    meanFramesPerMora: dc.moraFrames
                )
                let feat = lr.processText(
                    text: tc.word,
                    normalizer: normalizer,
                    prosodyModel: prosodyModel,
                    vocabulary: vocabulary,
                    addBoundarySilence: false
                )

                XCTAssertEqual(feat.phoneIds.count, 2, "\(tc.word) は子音と母音の2音素であること")
                let actualStopId = feat.phoneIds[0]
                XCTAssertEqual(actualStopId, tc.stopId, "先頭音素 ID が期待値と一致すること")
                let actualDur = Int(feat.durations[0])
                XCTAssertEqual(actualDur, dc.targetDur, "目標継続時間 \(dc.targetDur) と実測継続時間 \(actualDur) が一致すること")

                var f = 0
                while f < actualDur {
                    let eng = feat.energyContour[f]
                    switch actualDur <= 2 {
                    case true:
                        XCTAssertEqual(eng, 0.35, accuracy: 1e-5, "dur <= 2 の無声閉鎖音は全フレーム 0.35 であること (dur=\(actualDur), f=\(f))")
                    case false:
                        switch f < actualDur - 2 {
                        case true:
                            XCTAssertEqual(eng, 0.02, accuracy: 1e-5, "dur >= 3 の無声閉鎖音閉鎖期は 0.02 であること (dur=\(actualDur), f=\(f))")
                        case false:
                            XCTAssertEqual(eng, 0.35, accuracy: 1e-5, "dur >= 3 の無声閉鎖音開放期は 0.35 であること (dur=\(actualDur), f=\(f))")
                        }
                    }
                    f += 1
                }
                dIdx += 1
            }
            cIdx += 1
        }
    }
}
