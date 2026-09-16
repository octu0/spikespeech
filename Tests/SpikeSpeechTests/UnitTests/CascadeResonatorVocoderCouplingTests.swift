import XCTest
import Foundation
@testable import SpikeSpeech

/// SNN音響特徴量とカスケード共鳴器ボコーダーの動的結合、語彙永続化、未知語フォールバックの検証スイート
final class CascadeResonatorVocoderCouplingTests: XCTestCase {

    // MARK: - 1. 語彙永続化 (Lexicon Serialization) の検証

    /// モデル重みファイル（Models/weights.json）に語彙が永続化され、正しく復元されることを検証
    func testLexiconPersistedInWeightsJson() {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let weightsPath = currentDir + "/Models/weights.json"

        let exists = fileManager.fileExists(atPath: weightsPath)
        XCTAssertTrue(exists, "Models/weights.json が存在しません")

        guard let data = try? Data(contentsOf: URL(fileURLWithPath: weightsPath)) else {
            XCTFail("weights.json の読み込みに失敗しました")
            return
        }

        guard let weights = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) else {
            XCTFail("weights.json のデコードに失敗しました")
            return
        }

        // 永続化語彙が空でなく、150語以上保持されていることを確認
        XCTAssertFalse(weights.lexicon.isEmpty, "weights.json の lexicon が空です")
        XCTAssertTrue(150 <= weights.lexicon.count, "lexicon の単語数が不足しています: \(weights.lexicon.count)")

        // 代表語彙（助詞・名詞・動詞）の存在確認
        var foundKore = false
        var foundWatashi = false
        var foundNo = false
        var lIdx = 0
        while lIdx < weights.lexicon.count {
            let entry = weights.lexicon[lIdx]
            switch entry.surface {
            case "これ":
                foundKore = true
            case "私":
                foundWatashi = true
            case "の":
                foundNo = true
            default:
                break
            }
            lIdx += 1
        }
        XCTAssertTrue(foundKore, "語彙に 'これ' が含まれていません")
        XCTAssertTrue(foundWatashi, "語彙に '私' が含まれていません")
        XCTAssertTrue(foundNo, "語彙に 'の' が含まれていません")
    }

    /// 語彙の JSON シリアライズ／デシリアライズの可逆性および後方互換性の検証
    func testLexiconCodableRoundTripAndBackwardCompatibility() {
        let sampleLexicon = [
            LexiconEntry(surface: "テスト", reading: "てすと", pos: .noun, cost: 1000),
            LexiconEntry(surface: "走る", reading: "はしる", pos: .verb, cost: 1200)
        ]

        let baseWeights = SpikingNetworkWeights.randomWeights()
        let weightsWithLex = baseWeights.withLexicon(sampleLexicon)

        guard let encodedData = try? JSONEncoder().encode(weightsWithLex) else {
            XCTFail("SpikingNetworkWeights のエンコードに失敗しました")
            return
        }

        guard let decoded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: encodedData) else {
            XCTFail("SpikingNetworkWeights のデコードに失敗しました")
            return
        }

        XCTAssertEqual(decoded.lexicon.count, sampleLexicon.count, "復元された語彙数が不一致です")
        XCTAssertEqual(decoded.lexicon[0].surface, "テスト")
        XCTAssertEqual(decoded.lexicon[0].reading, "てすと")
        XCTAssertEqual(decoded.lexicon[1].surface, "走る")

        // lexicon キーが存在しない古い JSON からの安全なデコード（後方互換性）
        guard var jsonDict = (try? JSONSerialization.jsonObject(with: encodedData)) as? [String: Any] else {
            XCTFail("JSONSerialization に失敗しました")
            return
        }
        jsonDict.removeValue(forKey: "lexicon")
        guard let legacyData = try? JSONSerialization.data(withJSONObject: jsonDict),
              let legacyDecoded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: legacyData) else {
            XCTFail("レガシー JSON のデコードに失敗しました")
            return
        }
        XCTAssertTrue(legacyDecoded.lexicon.isEmpty, "レガシーデコード時の語彙が空配列で初期化されていません")
    }

    // MARK: - 2. 未知語のひらがな・発音フォールバックと推定合成の検証

    /// 辞書未登録の漢字・アルファベット・記号を含むテキストが音素脱落せず合成されることを検証
    func testUnknownWordsHiraganaFallbackAndSynthesis() {
        let engine = SpikeSpeechEngine()

        // 1. 辞書にない難読漢字（檸檬、麒麟）
        let kanjiText = "檸檬と麒麟"
        let kanjiSamples = engine.synthesize(text: kanjiText)
        XCTAssertFalse(kanjiSamples.isEmpty, "未知漢字の合成サンプルが空です")
        // サンプル数が発話として十分な長さ（0.3秒以上 = 4800サンプル以上）であることを確認
        XCTAssertTrue(4800 <= kanjiSamples.count, "未知漢字の合成サンプル数が不足しています: \(kanjiSamples.count)")

        // 2. アルファベット混じりテキスト（AI音声合成）
        let alphabetText = "AI音声合成"
        let alphabetSamples = engine.synthesize(text: alphabetText)
        XCTAssertFalse(alphabetSamples.isEmpty, "アルファベット混じりの合成サンプルが空です")
        XCTAssertTrue(4800 <= alphabetSamples.count, "アルファベット混じりの合成サンプル数が不足しています: \(alphabetSamples.count)")

        // 3. 英語頭字語（CPU）
        let acronymText = "CPU"
        let acronymSamples = engine.synthesize(text: acronymText)
        XCTAssertFalse(acronymSamples.isEmpty, "頭字語の合成サンプルが空です")
        XCTAssertTrue(3200 <= acronymSamples.count, "頭字語の合成サンプル数が不足しています: \(acronymSamples.count)")

        // 全サンプルにおける非有限値 (NaN/Inf) の不在確認
        var s = 0
        while s < kanjiSamples.count {
            XCTAssertFalse(kanjiSamples[s].isNaN, "未知漢字サンプルで NaN を検出")
            XCTAssertFalse(kanjiSamples[s].isInfinite, "未知漢字サンプルで Inf を検出")
            s += 1
        }
    }

    // MARK: - 3. 対数 Mel スペクトルからの動的フォルマント推定の検証

    /// 放物線補間およびスペクトル重心によるフォルマント周波数の動的追従を検証
    func testEstimateFormantFromMelPeakShift() {
        let centerFreqs = SpikeSpeechEngine.melCenterFrequencies
        let melChannels = AudioConfig.melChannels
        XCTAssertEqual(centerFreqs.count, melChannels)

        // 低域ピーク (約 400Hz) を持つ人工対数 Mel スペクトルの作成
        var melLow = [Float](repeating: -6.0, count: melChannels)
        // 400Hz に最も近いチャンネルを探索
        var ch400 = 0
        var minDiff400: Float = Float.infinity
        var c = 0
        while c < melChannels {
            let diff = abs(centerFreqs[c] - 400.0)
            if diff < minDiff400 {
                minDiff400 = diff
                ch400 = c
            }
            c += 1
        }
        melLow[ch400] = 2.0
        if 0 < ch400 {
            melLow[ch400 - 1] = 0.5
        }
        if (ch400 + 1) < melChannels {
            melLow[ch400 + 1] = 0.5
        }

        // 高域ピーク (約 800Hz) を持つ人工対数 Mel スペクトルの作成
        var melHigh = [Float](repeating: -6.0, count: melChannels)
        var ch800 = 0
        var minDiff800: Float = Float.infinity
        c = 0
        while c < melChannels {
            let diff = abs(centerFreqs[c] - 800.0)
            if diff < minDiff800 {
                minDiff800 = diff
                ch800 = c
            }
            c += 1
        }
        melHigh[ch800] = 2.0
        if 0 < ch800 {
            melHigh[ch800 - 1] = 0.5
        }
        if (ch800 + 1) < melChannels {
            melHigh[ch800 + 1] = 0.5
        }

        let fallbackF1: Float = 500.0
        let fallbackB1: Float = 100.0

        let (estLowFreq, _) = SpikeSpeechEngine.estimateFormantFromMel(
            mel: melLow,
            centerFreqs: centerFreqs,
            minFreq: 200.0,
            maxFreq: 1100.0,
            fallbackFreq: fallbackF1,
            fallbackBw: fallbackB1
        )

        let (estHighFreq, _) = SpikeSpeechEngine.estimateFormantFromMel(
            mel: melHigh,
            centerFreqs: centerFreqs,
            minFreq: 200.0,
            maxFreq: 1100.0,
            fallbackFreq: fallbackF1,
            fallbackBw: fallbackB1
        )

        // 低域ピークの推定周波数が高域ピークの推定周波数より明確に低いことを確認
        XCTAssertTrue(estLowFreq < estHighFreq, "ピーク周波数の移動が動的推定に反映されていません: low=\(estLowFreq), high=\(estHighFreq)")
        XCTAssertTrue(estLowFreq < fallbackF1, "400Hz ピークが fallback より下方に推定されていません: \(estLowFreq)")
        XCTAssertTrue(fallbackF1 < estHighFreq, "800Hz ピークが fallback より上方に推定されていません: \(estHighFreq)")
    }

    /// 音声スペクトルの低域チルト（200〜400Hz の強いエネルギー）が存在しても、
    /// 基準フォルマントをアンカーとした動的変調により F1/F2 共鳴極が低域へ潰れないことを検証
    func testEstimateFormantsFromMelSpectralTiltImmunity() {
        let centerFreqs = SpikeSpeechEngine.melCenterFrequencies
        let melChannels = AudioConfig.melChannels

        // 声帯音源の低域スペクトルチルト (-6dB/oct) を模擬した自然音声 Mel スペクトルを作成
        // 200Hz 近傍が +6.0 と最大で、周波数が高くなるにつれて単調減少
        var tiltedMel = [Float](repeating: 0.0, count: melChannels)
        var c = 0
        while c < melChannels {
            let cf = max(200.0, centerFreqs[c])
            // -1.0 * logf(cf / 200.0) による低域チルト
            tiltedMel[c] = 6.0 - (1.0 * logf(cf / 200.0))
            c += 1
        }

        // 母音 /a/ (F1=800, F2=1300) の検証
        let fallbackA = FormantConfig(
            f1: 800.0, b1: 80.0,
            f2: 1300.0, b2: 100.0,
            f3: 2600.0, b3: 120.0,
            f4: 3500.0, b4: 150.0
        )
        let estA = SpikeSpeechEngine.estimateFormantsFromMel(
            mel: tiltedMel,
            fallback: fallbackA,
            centerFreqs: centerFreqs
        )

        // F1 が 200〜400Hz の低域に引きずり落とされず、800Hz の ±15% 以内に保持されていることを検証
        XCTAssertTrue(680.0 <= estA.f1, "母音 /a/ の F1 が低域チルトに潰れています: \(estA.f1)")
        XCTAssertTrue(estA.f1 <= 920.0, "母音 /a/ の F1 が許容範囲を超過しています: \(estA.f1)")
        XCTAssertTrue(1105.0 <= estA.f2, "母音 /a/ の F2 が低域チルトに潰れています: \(estA.f2)")
        XCTAssertTrue(estA.f2 <= 1495.0, "母音 /a/ の F2 が許容範囲を超過しています: \(estA.f2)")

        // 母音 /e/ (F1=500, F2=1900) の検証
        let fallbackE = FormantConfig(
            f1: 500.0, b1: 70.0,
            f2: 1900.0, b2: 90.0,
            f3: 2700.0, b3: 130.0,
            f4: 3700.0, b4: 180.0
        )
        let estE = SpikeSpeechEngine.estimateFormantsFromMel(
            mel: tiltedMel,
            fallback: fallbackE,
            centerFreqs: centerFreqs
        )

        XCTAssertTrue(425.0 <= estE.f1, "母音 /e/ の F1 が低域チルトに潰れています: \(estE.f1)")
        XCTAssertTrue(estE.f1 <= 575.0, "母音 /e/ の F1 が許容範囲を超過しています: \(estE.f1)")
        XCTAssertTrue(1615.0 <= estE.f2, "母音 /e/ の F2 が低域チルトに潰れています: \(estE.f2)")
        XCTAssertTrue(estE.f2 <= 2185.0, "母音 /e/ の F2 が許容範囲を超過しています: \(estE.f2)")
    }

    /// フルパイプライン合成時（SNN 推論残差注入後）において、
    /// 5母音（あ・い・う・え・お）の F1/F2 フォルマント極が正しく弁別され、
    /// 低域チルトによって単一の狭母音へ潰れないことを検証
    func testVowelFormantsPreservedAcrossFramesInFullPipeline() {
        let engine = SpikeSpeechEngine()
        let text = "あいうえお"
        let lf = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )
        let totalFrames = lf.totalFrames
        let framePhoneIds = engine.extractFramePhoneIds(linguisticFeatures: lf, totalFrames: totalFrames)

        let inputSeq = engine.encodeLinguisticFeatures(features: lf)
        let snnAcousticSeq = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)
        let activePrior = engine.prior(for: VoiceProfile.default.tract)
        let melChannels = AudioConfig.melChannels
        let blendedPriorSeq = engine.computeBlendedPriorSequence(
            framePhoneIds: framePhoneIds,
            activePrior: activePrior,
            melChannels: melChannels
        )

        var combinedMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalFrames)
        var tMel = 0
        while tMel < totalFrames {
            let outDim = snnAcousticSeq[tMel].count
            let copyCount = min(melChannels, outDim)
            var rawSnnMel = [Float](repeating: 0.0, count: melChannels)
            rawSnnMel.withUnsafeMutableBufferPointer { melDst in
                snnAcousticSeq[tMel].withUnsafeBufferPointer { acSrc in
                    melDst.baseAddress!.update(from: acSrc.baseAddress!, count: copyCount)
                }
            }
            var compC = 0
            while compC < melChannels {
                combinedMelSeq[tMel][compC] = blendedPriorSeq[tMel][compC] + (engine.residualScale * rawSnnMel[compC])
                compC += 1
            }
            tMel += 1
        }

        let smoothedMelSeq = engine.smoothMelSequence(
            combinedMelSeq: combinedMelSeq,
            framePhoneIds: framePhoneIds,
            melChannels: melChannels
        )

        let (frames, _) = engine.buildResonatorFrames(
            linguisticFeatures: lf,
            voice: .default,
            effectiveBaseF0: VoiceProfile.default.baseF0,
            text: text,
            melSeq: smoothedMelSeq
        )

        XCTAssertEqual(frames.count, totalFrames)

        // 各母音の定常区間（中央フレーム）を取得して F1, F2 を検証
        var curF = 0
        var pIdx = 0
        while pIdx < lf.phoneIds.count {
            let pId = Int(lf.phoneIds[pIdx])
            let dur = Int(lf.durations[pIdx])
            let midF = curF + (dur / 2)
            if midF < totalFrames {
                let rf = frames[midF]
                let f1 = rf.formants.f1
                let f2 = rf.formants.f2
                switch pId {
                case 5: // /a/
                    XCTAssertTrue(650.0 <= f1 && f1 <= 950.0, "/a/ F1 が異常値です: \(f1)")
                    XCTAssertTrue(1100.0 <= f2 && f2 <= 1500.0, "/a/ F2 が異常値です: \(f2)")
                case 6: // /i/
                    XCTAssertTrue(250.0 <= f1 && f1 <= 380.0, "/i/ F1 が異常値です: \(f1)")
                    XCTAssertTrue(1900.0 <= f2 && f2 <= 2650.0, "/i/ F2 が異常値です: \(f2)")
                case 7: // /u/
                    XCTAssertTrue(280.0 <= f1 && f1 <= 420.0, "/u/ F1 が異常値です: \(f1)")
                    XCTAssertTrue(1000.0 <= f2 && f2 <= 1400.0, "/u/ F2 が異常値です: \(f2)")
                case 8: // /e/
                    XCTAssertTrue(420.0 <= f1 && f1 <= 600.0, "/e/ F1 が異常値です: \(f1)")
                    XCTAssertTrue(1600.0 <= f2 && f2 <= 2200.0, "/e/ F2 が異常値です: \(f2)")
                case 9: // /o/
                    XCTAssertTrue(420.0 <= f1 && f1 <= 600.0, "/o/ F1 が異常値です: \(f1)")
                    XCTAssertTrue(750.0 <= f2 && f2 <= 1050.0, "/o/ F2 が異常値です: \(f2)")
                default:
                    break
                }
            }
            curF += dur
            pIdx += 1
        }
    }

    /// 対数 Mel スペクトルからの RMS ゲイン推定の検証
    func testEstimateGainFromMel() {
        let melChannels = AudioConfig.melChannels

        // 無音フロア (-7.88)
        let silenceMel = [Float](repeating: -7.88, count: melChannels)
        let silenceGain = SpikeSpeechEngine.estimateGainFromMel(mel: silenceMel)

        // 通常発話母音 (+1.5)
        let speechMel = [Float](repeating: 1.5, count: melChannels)
        let speechGain = SpikeSpeechEngine.estimateGainFromMel(mel: speechMel)

        XCTAssertTrue(silenceGain < speechGain, "発話エネルギーが無音エネルギーより大きく計算されていません")
        XCTAssertTrue(0.0 <= silenceGain, "無音ゲインが負数です")
        XCTAssertTrue(0.5 <= speechGain, "発話ゲインが過小です: \(speechGain)")
    }

    // MARK: - 4. SNN 推論結果がカスケード共鳴ボコーダー波形に直結することの検証

    /// SNN の音響特徴量（Mel 残差）がカスケード共鳴器の駆動パラメータおよび合成波形に直接反映されることを検証
    func testSNNResidualDirectlyModifiesResonatorOutput() {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let weightsPath = currentDir + "/Models/weights.json"

        var weights = SpikingNetworkWeights.randomWeights()
        if fileManager.fileExists(atPath: weightsPath) {
            if let data = try? Data(contentsOf: URL(fileURLWithPath: weightsPath)),
               let loaded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) {
                weights = loaded
            }
        }

        // 1. SNN 残差スケール 0.0 (Prior のみで駆動)
        let enginePriorOnly = SpikeSpeechEngine(weights: weights, residualScale: 0.0)
        let text = "こんにちは"
        let samplesPrior = enginePriorOnly.synthesize(text: text)

        // 2. SNN 残差スケール 1.0 (Prior + SNN 学習残差で駆動)
        let engineWithSNN = SpikeSpeechEngine(weights: weights, residualScale: 1.0)
        let samplesWithSNN = engineWithSNN.synthesize(text: text)

        XCTAssertEqual(samplesPrior.count, samplesWithSNN.count, "サンプル数が一致しません")

        // SNN 残差の有無により波形サンプルに有意な差分が生じていること（SNN 推論結果が破棄されていないこと）を検証
        var maxDiff: Float = 0.0
        var sumDiffSq: Float = 0.0
        var i = 0
        while i < samplesPrior.count {
            let diff = abs(samplesPrior[i] - samplesWithSNN[i])
            if maxDiff < diff {
                maxDiff = diff
            }
            sumDiffSq += diff * diff
            i += 1
        }

        let rmsDiff = sqrt(sumDiffSq / Float(max(1, samplesPrior.count)))
        XCTAssertTrue(1e-4 < rmsDiff, "SNN 残差がボコーダー出力波形に反映されていません (RMS 差分ゼロ): \(rmsDiff)")
    }

    /// extractFramePhoneIds が buildResonatorFrames と完全一致する時間軸アライメントを返すことを検証
    func testExtractFramePhoneIdsConsistency() {
        let engine = SpikeSpeechEngine()
        let text = "テストです"
        let lf = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )

        let extractedIds = engine.extractFramePhoneIds(linguisticFeatures: lf, totalFrames: lf.totalFrames)
        let (_, buildIds) = engine.buildResonatorFrames(
            linguisticFeatures: lf,
            voice: .default,
            effectiveBaseF0: 220.0,
            text: text
        )

        XCTAssertEqual(extractedIds.count, lf.totalFrames)
        XCTAssertEqual(buildIds.count, lf.totalFrames)

        var t = 0
        while t < lf.totalFrames {
            // なぜ長音を含め全フレーム一致を検証するか:
            // extractFramePhoneIds と buildResonatorFrames の双方で長音記号（26）が先行母音に置換され、
            // 調音結合Priorブレンドと共鳴器フレーム構築の時間軸アライメントが完全に同一であることを保証するため。
            let eId = extractedIds[t]
            let bId = buildIds[t]
            XCTAssertEqual(eId, bId, "フレーム \(t) で音素 ID が不一致です: extracted=\(eId), build=\(bId)")
            t += 1
        }
    }

    /// SpikeSpeechEngine デフォルト初期化時にもモデル重み（weights.json）の語彙が自動注入されることを検証
    func testSpikeSpeechEngineDefaultInitLoadsLexicon() {
        let engine = SpikeSpeechEngine()
        // なぜ語彙数が空でないことを確認するか:
        // 引数なしで初期化したエンジンであっても、モデル重みファイルから永続化語彙が
        // 正常にロードされ、形態素解析と読み付与が成立することを保証するため。
        XCTAssertFalse(engine.weights.lexicon.isEmpty, "デフォルト初期化された engine.weights.lexicon が空です")
        XCTAssertTrue(150 <= engine.weights.lexicon.count, "ロードされた語彙数が不足しています: \(engine.weights.lexicon.count)")

        let text = "こんにちは、世界。"
        let morphemes = engine.normalizer.normalize(text: text)
        XCTAssertFalse(morphemes.isEmpty, "正規化形態素系列が空です")

        // 「こんにちは」が単一の語彙エントリとして正しく認識されていることを検証
        var foundGreeting = false
        var mIdx = 0
        while mIdx < morphemes.count {
            if morphemes[mIdx].surface == "こんにちは" {
                foundGreeting = true
            }
            mIdx += 1
        }
        XCTAssertTrue(foundGreeting, "語彙「こんにちは」が1文字ずつ未知語分割されています")
    }

    /// 未知漢字熟語が中国語ピンインではなく日本語形態素音訳（ひらがな）として解決されることを検証
    func testUnknownKanjiJapaneseReadingTransliteration() {
        // なぜ単漢字ではなく「檸檬」「麒麟」などの熟語で検証するか:
        // ICU の kCFStringTransformToLatin は中国語ピンイン（ning meng / qi lin）に誤音訳するため、
        // 日本語ロケール形態素音訳により「れもん」「きりん」と正しく仮名化されることを検証するため。
        let lemonReading = ViterbiMorphology.fallbackReadingForKanji("檸檬")
        XCTAssertEqual(lemonReading, "れもん", "「檸檬」の読みが日本語音訓（れもん）になっていません: \(lemonReading)")

        let kirinReading = ViterbiMorphology.fallbackReadingForKanji("麒麟")
        XCTAssertEqual(kirinReading, "きりん", "「麒麟」の読みが日本語音訓（きりん）になっていません: \(kirinReading)")

        let engine = SpikeSpeechEngine()
        let morphemes = engine.normalizer.normalize(text: "檸檬")
        XCTAssertFalse(morphemes.isEmpty)
        XCTAssertEqual(morphemes[0].reading, "れもん", "正規化後の「檸檬」の読みが一致しません: \(morphemes[0].reading)")
    }

#if canImport(MLX)
    /// MLXSpikingAcousticNetwork が語彙データを保持し、exportWeights で消失しないことを検証
    func testMLXNetworkPreservesLexiconOnExport() {
        let sampleLexicon = [
            LexiconEntry(surface: "単語A", reading: "たんごえー", pos: .noun, cost: 500),
            LexiconEntry(surface: "単語B", reading: "たんごびー", pos: .verb, cost: 600)
        ]
        let baseWeights = SpikingNetworkWeights.randomWeights().withLexicon(sampleLexicon)
        let net = MLXSpikingAcousticNetwork(weights: baseWeights)

        XCTAssertEqual(net.lexicon.count, sampleLexicon.count, "MLX ネットワーク内部で語彙が保持されていません")

        let exported = net.exportWeights()
        XCTAssertEqual(exported.lexicon.count, sampleLexicon.count, "exportWeights 時に語彙が破棄されています")
        XCTAssertEqual(exported.lexicon[0].surface, "単語A")
        XCTAssertEqual(exported.lexicon[1].surface, "単語B")
    }
#endif

    /// ユーザー報告の3文章に対する形態素解析、音素系列、フォルマントの恒久検証
    func testDiagnosticSentences() {
        let engine = SpikeSpeechEngine()
        let cases: [(text: String, expectedReadingKeywords: [String])] = [
            (
                "こんにちは。スパイクスピーチによる音声合成のテストです。",
                ["こんにちわ", "すぱいく", "すぴーち", "よる", "おんせーごーせー", "てすと", "です"]
            ),
            (
                "吾輩は猫である。名前はまだ無い。",
                ["わがはい", "ねこ", "ある", "なまえ", "まだ", "ない"]
            ),
            (
                "水をマレーシアから買わなくてはならないのです。",
                ["みず", "まれーしあ", "から", "かわ", "なく", "なら", "ない", "です"]
            )
        ]

        var cIdx = 0
        while cIdx < cases.count {
            let tc = cases[cIdx]
            let text = tc.text
            let morphemes = engine.normalizer.normalize(text: text)
            XCTAssertFalse(morphemes.isEmpty, "形態素解析結果が空です: \(text)")

            let allReadings = morphemes.map { $0.reading }.joined()
            var kIdx = 0
            while kIdx < tc.expectedReadingKeywords.count {
                let kw = tc.expectedReadingKeywords[kIdx]
                XCTAssertTrue(allReadings.contains(kw), "読み [\(kw)] が含まれていません: \(allReadings) (原文: \(text))")
                kIdx += 1
            }

            let lf = engine.lengthRegulator.processText(
                text: text,
                normalizer: engine.normalizer,
                prosodyModel: engine.prosodyModel,
                vocabulary: engine.vocabulary,
                speedFactor: 1.0,
                baseF0: VoiceProfile.default.baseF0,
                applyFluctuation: false
            )
            XCTAssertTrue(0 < lf.phoneIds.count, "音素系列が空です: \(text)")
            XCTAssertTrue(0 < lf.totalFrames, "総フレーム数が0です: \(text)")

            let inputSeq = engine.encodeLinguisticFeatures(features: lf)
            let snnAcousticSeq = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)
            let framePhoneIds = engine.extractFramePhoneIds(linguisticFeatures: lf, totalFrames: lf.totalFrames)
            let activePrior = engine.prior(for: VoiceProfile.default.tract)
            let melChannels = AudioConfig.melChannels
            let blendedPriorSeq = engine.computeBlendedPriorSequence(
                framePhoneIds: framePhoneIds,
                activePrior: activePrior,
                melChannels: melChannels
            )
            var combinedMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: lf.totalFrames)
            var tMel = 0
            while tMel < lf.totalFrames {
                let outDim = snnAcousticSeq[tMel].count
                let copyCount = min(melChannels, outDim)
                var rawSnnMel = [Float](repeating: 0.0, count: melChannels)
                rawSnnMel.withUnsafeMutableBufferPointer { melDst in
                    snnAcousticSeq[tMel].withUnsafeBufferPointer { acSrc in
                        melDst.baseAddress!.update(from: acSrc.baseAddress!, count: copyCount)
                    }
                }
                var compC = 0
                while compC < melChannels {
                    combinedMelSeq[tMel][compC] = blendedPriorSeq[tMel][compC] + (engine.residualScale * rawSnnMel[compC])
                    compC += 1
                }
                tMel += 1
            }
            let smoothedMelSeq = engine.smoothMelSequence(
                combinedMelSeq: combinedMelSeq,
                framePhoneIds: framePhoneIds,
                melChannels: melChannels
            )

            let (frames, phoneIds) = engine.buildResonatorFrames(
                linguisticFeatures: lf,
                voice: .default,
                effectiveBaseF0: VoiceProfile.default.baseF0,
                text: text,
                melSeq: smoothedMelSeq
            )
            XCTAssertEqual(frames.count, lf.totalFrames, "フレーム数不一致")

            // 母音フレームにおいてフォルマントがアンカーから逸脱せず生理学的帯域内にあることを検証
            var fIdx = 0
            while fIdx < frames.count {
                let pid = phoneIds[fIdx]
                let f = frames[fIdx]
                switch pid {
                case 5: // /a/
                    XCTAssertTrue(650.0 <= f.formants.f1 && f.formants.f1 <= 950.0, "/a/ F1 逸脱: \(f.formants.f1)")
                    XCTAssertTrue(1100.0 <= f.formants.f2 && f.formants.f2 <= 1600.0, "/a/ F2 逸脱: \(f.formants.f2)")
                case 6: // /i/
                    XCTAssertTrue(250.0 <= f.formants.f1 && f.formants.f1 <= 450.0, "/i/ F1 逸脱: \(f.formants.f1)")
                    XCTAssertTrue(1800.0 <= f.formants.f2 && f.formants.f2 <= 2600.0, "/i/ F2 逸脱: \(f.formants.f2)")
                case 7: // /u/
                    XCTAssertTrue(250.0 <= f.formants.f1 && f.formants.f1 <= 450.0, "/u/ F1 逸脱: \(f.formants.f1)")
                case 8: // /e/
                    XCTAssertTrue(400.0 <= f.formants.f1 && f.formants.f1 <= 600.0, "/e/ F1 逸脱: \(f.formants.f1)")
                    XCTAssertTrue(1600.0 <= f.formants.f2 && f.formants.f2 <= 2200.0, "/e/ F2 逸脱: \(f.formants.f2)")
                case 9: // /o/
                    XCTAssertTrue(400.0 <= f.formants.f1 && f.formants.f1 <= 600.0, "/o/ F1 逸脱: \(f.formants.f1)")
                    XCTAssertTrue(700.0 <= f.formants.f2 && f.formants.f2 <= 1350.0, "/o/ F2 逸脱: \(f.formants.f2)")
                default:
                    break
                }
                fIdx += 1
            }

            // 実音声合成の実行と波形の完全性検証
            let samples = engine.synthesize(text: text)
            XCTAssertFalse(samples.isEmpty, "合成サンプルが空です: \(text)")
            var sIdx = 0
            while sIdx < samples.count {
                XCTAssertFalse(samples[sIdx].isNaN, "NaN サンプル検出: \(sIdx)")
                XCTAssertFalse(samples[sIdx].isInfinite, "Inf サンプル検出: \(sIdx)")
                sIdx += 1
            }

            cIdx += 1
        }
    }
}

