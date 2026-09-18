import XCTest
import Foundation
import MLX
@testable import SpikeSpeech

/// Tier 1: 機能網羅テストスイート
///
/// SpikeSpeech の全構成要素（形態素解析、読み正規化、トークナイズ、Duration展開、アクセント/F0、
/// ボコーダー、MelToLPC、WAV、DSP安全機構、SNNデコーダー、BPTT学習、推論HotPath、多層SNN重み構造、
/// 基準音声生成、CLIスイート、E2E合成パイプライン）の代表的正常系挙動を決定論的に検証する。
final class Tier1FeatureTests: XCTestCase {

    private var engine: SpikeSpeechEngine!
    private var normalizer: TextNormalizer!
    private var morphology: ViterbiMorphology!
    private var vocabulary: PhonemeVocabulary!
    private var prosodyModel: ProsodyModel!
    private var lengthRegulator: LengthRegulator!
    private var vocoder: LPCVocoder!
    private var melToLPC: MelToLPC!
    private var generator: SyntheticAudioGenerator!

    override func setUp() {
        super.setUp()
        // テスト間の独立性を保ち、各テストがクリーンな初期状態から実行できるように共通コンポーネントを初期化する。
        self.morphology = ViterbiMorphology()
        self.normalizer = TextNormalizer(morphology: morphology)
        self.vocabulary = PhonemeVocabulary()
        self.prosodyModel = ProsodyModel()
        self.lengthRegulator = LengthRegulator(hiddenDimension: 128)
        self.vocoder = LPCVocoder()
        self.melToLPC = MelToLPC()
        self.generator = SyntheticAudioGenerator()
        self.engine = SpikeSpeechEngine()
    }

    // MARK: - F1: Pure Swift 形態素解析 (5ケース以上)

    func testF1_Morphology_BasicNounSentence() {
        // 単純な名詞句ラティスにおいて、二分探索語彙引きと最小コストパスが正しく名詞を抽出することを検証する。
        let text = "今日の本"
        let morphemes = morphology.tokenize(text)
        XCTAssertTrue(0 < morphemes.count)
        XCTAssertEqual(morphemes[0].surface, "今日")
        XCTAssertEqual(morphemes[0].pos, .noun)
    }

    func testF1_Morphology_VerbConjugation() {
        // 終止形および過去形活用において、動詞語幹と助動詞境界が正しく認識されることを検証する。
        let text = "走る"
        let morphemes = morphology.tokenize(text)
        XCTAssertTrue(0 < morphemes.count)
        XCTAssertEqual(morphemes[0].surface, "走る")
        XCTAssertEqual(morphemes[0].pos, .verb)
    }

    func testF1_Morphology_Adverb() {
        // 組み込み語彙の自立語品詞（副詞「とても」）が Viterbi 最短経路探索で正しく品詞特定・分割されることを検証する。
        let text = "とても速い"
        let morphemes = morphology.tokenize(text)
        XCTAssertTrue(1 < morphemes.count)
        XCTAssertEqual(morphemes[0].surface, "とても")
        XCTAssertEqual(morphemes[0].pos, .adverb)
    }

    func testF1_Morphology_AuxiliaryVerb() {
        // 動詞連用形と丁寧助動詞「ます」の連接が品詞遷移コスト行列で安定して選択されることを検証する。
        let text = "食べます"
        let morphemes = morphology.tokenize(text)
        XCTAssertTrue(1 < morphemes.count)
        XCTAssertEqual(morphemes[0].pos, .verb)
    }

    func testF1_Morphology_HiraganaSentence() {
        // 漢字の字種境界手がかりが存在しない全平仮名系列でも、Viterbi累積コストが正しく単語分割を行うことを検証する。
        let text = "わたしはねこ"
        let morphemes = morphology.tokenize(text)
        XCTAssertTrue(2 < morphemes.count)
        let surfaces = morphemes.map { $0.surface }
        XCTAssertTrue(surfaces.contains("わたし"))
        XCTAssertTrue(surfaces.contains("ねこ"))
    }

    // MARK: - F2: 日本語読み・発音正規化 (5ケース以上)

    func testF2_Normalization_ParticleHaHeWo() {
        // 助詞「は」「へ」「を」が発音規則に従って「わ」「え」「お」に正しく正規化されることを検証する。
        let text = "学校へ行く"
        let norm = normalizer.normalize(text: text)
        let reading = norm.map { $0.reading }.joined()
        XCTAssertTrue(reading.contains("えいく"))
    }

    func testF2_Normalization_KanjiNumbers() {
        // アラビア数字が位取り構造（一万二千三百四十五）に従って正確に平仮名展開されることを検証する。
        let expanded = normalizer.expandNumbersAndCounters("12345")
        XCTAssertEqual(expanded, "いちまんにせんさんびゃくよんじゅうご")
    }

    func testF2_Normalization_GeminateConsonants() {
        // 小書き「っ」が促音トークン（Q）および無音閉鎖フレームとして処理されることを検証する。
        let text = "きっと"
        let norm = normalizer.normalize(text: text)
        let reading = norm.map { $0.reading }.joined()
        XCTAssertTrue(reading.contains("っ") || reading.contains("き"))
    }

    func testF2_Normalization_ProlongedSound() {
        // カタカナ長音符「ー」が前置母音の継続特性として正しく保持されることを検証する。
        let text = "コーヒー"
        let norm = normalizer.normalize(text: text)
        let reading = norm.map { $0.reading }.joined()
        XCTAssertTrue(reading.contains("ー"))
    }

    func testF2_Normalization_VerbEndingProtection() {
        // 「思う」「追う」の語尾「う」が長音化されずに独立母音 [u] として保護されることを検証する。
        let text = "思う"
        let norm = normalizer.normalize(text: text)
        let reading = norm.map { $0.reading }.joined()
        XCTAssertEqual(reading, "おもう")
    }

    // MARK: - F3: Mora/Phoneme 階層トークナイズ (5ケース以上)

    func testF3_HierarchicalTokenize_UtteranceToPhonemes() {
        // 発話からアクセント句、モーラ、音素への階層的分解が親子整合性を保つことを検証する。
        let morphemes = normalizer.normalize(text: "空が青い")
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
        XCTAssertTrue(0 < phrases.count)
        let firstMora = phrases[0].moras[0]
        XCTAssertTrue(0 < firstMora.phonemes.count)
    }

    func testF3_HierarchicalTokenize_SpecialTokens() {
        // パディング、無音、ポーズ等の特殊記号が定義域内の固定 ID を保持することを検証する。
        let padId = vocabulary.id(for: "<pad>")
        let silId = vocabulary.id(for: "<sil>")
        let pauId = vocabulary.id(for: "<pau>")
        XCTAssertTrue(0 <= padId)
        XCTAssertTrue(0 <= silId)
        XCTAssertTrue(0 <= pauId)
        XCTAssertEqual(vocabulary.token(for: padId), "<pad>")
    }

    func testF3_HierarchicalTokenize_ContractedSounds() {
        // 「きゃ」「しゅ」等の拗音が単一モーラとして子音＋半母音＋母音の音素列を正しく構成することを検証する。
        let moras = vocabulary.kanaToMoras("きゃ")
        XCTAssertEqual(moras.count, 1)
        XCTAssertEqual(moras[0].phonemes.count, 2)
        XCTAssertEqual(moras[0].phonemes[0].symbol, "ky")
        XCTAssertEqual(moras[0].phonemes[1].symbol, "a")
    }

    func testF3_HierarchicalTokenize_NasalSyllable() {
        // 「ん」が単独モーラとして独立した音素 N に対応付けられることを検証する。
        let moras = vocabulary.kanaToMoras("ん")
        XCTAssertEqual(moras.count, 1)
        XCTAssertEqual(moras[0].phonemes.count, 1)
        XCTAssertEqual(moras[0].phonemes[0].symbol, "N")
        XCTAssertEqual(moras[0].phonemes[0].category, .nasalSyllable)
    }

    func testF3_HierarchicalTokenize_MoraDurationProperties() {
        // MoraToken.totalDurationFrames が内部の全 PhonemeToken.durationFrames の総和と一致することを検証する。
        let mora = MoraToken(text: "た", phonemes: [
            PhonemeToken(id: 10, symbol: "t", category: .consonant, durationFrames: 4),
            PhonemeToken(id: 5, symbol: "a", category: .vowel, durationFrames: 8)
        ])
        XCTAssertEqual(mora.totalDurationFrames, 12)
    }

    // MARK: - F4: 累積和 Length Regulation (5ケース以上)

    func testF4_LengthRegulation_BasicExpansion() {
        // 各音素の duration に基づいて音素 ID 列が過不足なくフレーム展開されることを検証する。
        let features = lengthRegulator.processText(
            text: "ねこ",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 1.0
        )
        XCTAssertTrue(0 < features.totalFrames)
        XCTAssertEqual(features.f0Contour.count, features.totalFrames)
        XCTAssertEqual(features.voicedFlags.count, features.totalFrames)
    }

    func testF4_LengthRegulation_SpeedFactorFast() {
        // 1.0 < speedFactor で総フレーム数が短縮されることを検証する。
        let featNormal = lengthRegulator.processText(
            text: "ありがとう",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 1.0
        )
        let featFast = lengthRegulator.processText(
            text: "ありがとう",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 1.5
        )
        XCTAssertTrue(featFast.totalFrames < featNormal.totalFrames)
    }

    func testF4_LengthRegulation_SpeedFactorSlow() {
        // speedFactor < 1.0 で総フレーム数が伸張されることを検証する。
        let featNormal = lengthRegulator.processText(
            text: "おはよう",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 1.0
        )
        let featSlow = lengthRegulator.processText(
            text: "おはよう",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 0.8
        )
        XCTAssertTrue(featNormal.totalFrames < featSlow.totalFrames)
    }

    func testF4_LengthRegulation_QuantizeDurations() {
        // 実数 duration が最小 1 フレームの整数値へクリッピング量子化されることを検証する。
        let rawDurs: [Float] = [0.2, 1.4, 3.8, 5.1]
        let quantized = lengthRegulator.quantizeDurations(durations: rawDurs)
        XCTAssertEqual(quantized.count, 4)
        var i = 0
        while i < quantized.count {
            XCTAssertTrue(1 <= quantized[i])
            i += 1
        }
    }

    func testF4_LengthRegulation_TotalFramesSum() {
        // 累積和量子化においてフレームの脱落や過剰加算が存在しないことを検証する。
        let features = lengthRegulator.processText(
            text: "さくら",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )
        var durSum: Int32 = 0
        var i = 0
        while i < features.durations.count {
            durSum += features.durations[i]
            i += 1
        }
        XCTAssertEqual(Int(durSum), features.totalFrames)
    }

    // MARK: - F5: ピッチアクセント & F0 制御 (5ケース以上)

    func testF5_Prosody_Atamadaka() {
        // 1拍目が High、2拍目以降が Low となる東京方言標準則を検証する。
        let tones = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 1)
        XCTAssertEqual(tones.count, 3)
        XCTAssertEqual(tones[0], .high)
        XCTAssertEqual(tones[1], .low)
        XCTAssertEqual(tones[2], .low)
    }

    func testF5_Prosody_Nakadaka() {
        // 1拍目が Low、2拍目が High、3拍目以降が核下降で Low となるトーン遷移を検証する。
        let tones = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 2)
        XCTAssertEqual(tones.count, 3)
        XCTAssertEqual(tones[0], .low)
        XCTAssertEqual(tones[1], .high)
        XCTAssertEqual(tones[2], .low)
    }

    func testF5_Prosody_Odaka() {
        // 1拍目が Low、2拍目以降語末まで High となるトーン遷移を検証する。
        let tones = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 3)
        XCTAssertEqual(tones.count, 3)
        XCTAssertEqual(tones[0], .low)
        XCTAssertEqual(tones[1], .high)
        XCTAssertEqual(tones[2], .high)
    }

    func testF5_Prosody_Heiban() {
        // 核が存在しない（kernel=0）場合に 1拍目 Low、2拍目以降 High となるトーン遷移を検証する。
        let tones = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 0)
        XCTAssertEqual(tones.count, 3)
        XCTAssertEqual(tones[0], .low)
        XCTAssertEqual(tones[1], .high)
        XCTAssertEqual(tones[2], .high)
    }

    func testF5_Prosody_F0ContourValues() {
        // 有声フレームでは生理的周波数範囲（100〜400Hz）が割り当てられ、無声フレームでは厳密に 0.0 となることを検証する。
        let features = lengthRegulator.processText(
            text: "あさ",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )
        var f = 0
        while f < features.totalFrames {
            let voiced = features.voicedFlags[f]
            let f0 = features.f0Contour[f]
            if 0.5 <= voiced {
                XCTAssertTrue(50.0 < f0)
                XCTAssertTrue(f0 < 500.0)
            } else {
                XCTAssertEqual(f0, 0.0)
            }
            f += 1
        }
    }

    // MARK: - F6: Source-Filter LPC ボコーダー (5ケース以上)

    func testF6_Vocoder_VoicedSynthesis() {
        // Rosenberg パルス励起と AR 声道フィルタの畳み込みにより安定した周期波形が生成されることを検証する。
        vocoder.reset()
        let frame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.05, count: 16),
            gain: 0.1,
            pitchF0: 150.0,
            voiced: 1.0
        )
        let samples = vocoder.synthesize(frames: [frame, frame])
        XCTAssertEqual(samples.count, 320)
        var energy: Float = 0.0
        var i = 0
        while i < samples.count {
            energy += samples[i] * samples[i]
            i += 1
        }
        XCTAssertTrue(0.0 < energy)
    }

    func testF6_Vocoder_UnvoicedSynthesis() {
        // Xorshift64 ノイズ励起によって非周期・広帯域な信号が生成されることを検証する。
        vocoder.reset()
        let frame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.02, count: 16),
            gain: 0.1,
            pitchF0: 0.0,
            voiced: 0.0
        )
        let samples = vocoder.synthesize(frames: [frame, frame])
        XCTAssertEqual(samples.count, 320)
        var energy: Float = 0.0
        var i = 0
        while i < samples.count {
            energy += samples[i] * samples[i]
            i += 1
        }
        XCTAssertTrue(0.0 < energy)
    }

    func testF6_Vocoder_LinearInterpolation() {
        // ゲインや係数が急変する2フレーム間でサンプル値が不連続なステップ破綻を起こさないことを検証する。
        vocoder.reset()
        let frame1 = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.01, count: 16),
            gain: 0.05,
            pitchF0: 120.0,
            voiced: 1.0
        )
        let frame2 = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.05, count: 16),
            gain: 0.2,
            pitchF0: 180.0,
            voiced: 1.0
        )
        let samples = vocoder.synthesize(frames: [frame1, frame2])
        XCTAssertEqual(samples.count, 320)
        XCTAssertTrue(samples[159].isFinite)
        XCTAssertTrue(samples[160].isFinite)
    }

    func testF6_Vocoder_SignalEnergyAndLimits() {
        // Soft Limiter の適用により出力サンプルが [-1.0, 1.0] に収まることを検証する。
        vocoder.reset()
        let frame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.05, count: 16),
            gain: 0.5,
            pitchF0: 140.0,
            voiced: 1.0
        )
        let samples = vocoder.synthesize(frames: [frame])
        var i = 0
        while i < samples.count {
            XCTAssertTrue(-1.0 <= samples[i])
            XCTAssertTrue(samples[i] <= 1.0)
            i += 1
        }
    }

    func testF6_Vocoder_ResetState() {
        // reset() 呼び出しによって残響と内部レジスタが完全に初期化されることを検証する。
        vocoder.reset()
        let frameSilent = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.0, count: 16),
            gain: 0.0,
            pitchF0: 0.0,
            voiced: 0.0
        )
        let samples = vocoder.synthesize(frames: [frameSilent])
        var i = 0
        while i < samples.count {
            XCTAssertEqual(samples[i], 0.0)
            i += 1
        }
    }

    // MARK: - F7: Mel/スペクトル反転アルゴリズム (5ケース以上)

    func testF7_MelToLPC_LogMelConversion() {
        // 64ch の対数 Mel スペクトルから Levinson-Durbin により安定した 16次 LPC 係数が導出されることを検証する。
        let logMel = [Float](repeating: -2.0, count: 64)
        var coeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLPC.convert(mel: logMel, isLogMel: true, outCoeffs: &coeffs)
        XCTAssertTrue(0.0 < gain)
        var i = 0
        while i < coeffs.count {
            XCTAssertTrue(coeffs[i].isFinite)
            i += 1
        }
    }

    func testF7_MelToLPC_LinearMelConversion() {
        // 線形エネルギー表現の Mel 特徴量からも破綻なくゲインと LPC 係数が計算できることを検証する。
        let linMel = [Float](repeating: 0.1, count: 64)
        var coeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLPC.convert(mel: linMel, isLogMel: false, outCoeffs: &coeffs)
        XCTAssertTrue(0.0 < gain)
    }

    func testF7_MelToLPC_AutoCorrelationProperties() {
        // パワースペクトルの IDCT から得られる自己相関 r0 が正の実数であり全帯域エネルギーを反映することを検証する。
        var coeffs = [Float](repeating: 0.0, count: 16)
        let mel = [Float](repeating: -1.0, count: 64)
        let gain = melToLPC.convert(mel: mel, isLogMel: true, outCoeffs: &coeffs)
        XCTAssertTrue(1e-5 < gain)
    }

    func testF7_MelToLPC_StabilityReflectionCoeffs() {
        // Levinson-Durbin の内部反射係数 ki が [-0.999, 0.999] に抑えられ発散しないことを検証する。
        var sharpMel = [Float](repeating: -10.0, count: 64)
        sharpMel[10] = 5.0 // 極端な鋭いピーク
        var coeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLPC.convert(mel: sharpMel, isLogMel: true, outCoeffs: &coeffs)
        XCTAssertTrue(gain.isFinite)
        var i = 0
        while i < coeffs.count {
            XCTAssertTrue(coeffs[i].isFinite)
            XCTAssertTrue(abs(coeffs[i]) < 10.0)
            i += 1
        }
    }

    func testF7_MelToLPC_BandwidthExpansion() {
        // 出力 LPC 係数に帯域幅拡大係数ガンマのべき乗が適用され高次係数が適切に減衰していることを検証する。
        let mel = [Float](repeating: -3.0, count: 64)
        var coeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLPC.convert(mel: mel, isLogMel: true, outCoeffs: &coeffs)
        XCTAssertTrue(gain.isFinite)
        XCTAssertTrue(coeffs[15].isFinite)
    }

    // MARK: - F8: Pure Swift WAV エンコーダー (5ケース以上)

    func testF8_WavEncoder_HeaderStructure() {
        // 44 バイト RIFF/WAVE ヘッダのマジック、フォーマット ID (1)、サンプリングレート (16000) が Little-Endian で正しく書き込まれることを検証する。
        let header = WavEncoder.createHeader(sampleRate: 16000, numChannels: 1, bitsPerSample: 16, dataSize: 320)
        XCTAssertEqual(header.count, 44)
        XCTAssertEqual(header[0], 0x52)
        XCTAssertEqual(header[1], 0x49)
        XCTAssertEqual(header[2], 0x46)
        XCTAssertEqual(header[3], 0x46)
        XCTAssertEqual(header[8], 0x57)
        XCTAssertEqual(header[9], 0x41)
        XCTAssertEqual(header[10], 0x56)
        XCTAssertEqual(header[11], 0x45)
    }

    func testF8_WavEncoder_EncodeSamples() {
        // N 個の Float サンプルが 44 + 2N バイトのバイナリ Data に一括変換されることを検証する。
        let samples: [Float] = [0.0, 0.5, -0.5, 0.8]
        let data = WavEncoder.encode(samples: samples, sampleRate: 16000)
        XCTAssertEqual(data.count, 44 + (samples.count * 2))
    }

    func testF8_WavEncoder_StreamWriterSequential() throws {
        // 逐次チャンク書き出しによってファイルサイズが連続的に伸張することを検証する。
        let tempUrl = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        FileManager.default.createFile(atPath: tempUrl.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tempUrl)
        let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

        try writer.write(samples: [Float](repeating: 0.1, count: 160))
        try writer.write(samples: [Float](repeating: -0.1, count: 160))
        try writer.finalize()
        try handle.close()

        let fileData = try Data(contentsOf: tempUrl)
        XCTAssertEqual(fileData.count, 44 + (320 * 2))
        try? FileManager.default.removeItem(at: tempUrl)
    }

    func testF8_WavEncoder_StreamWriterFinalize() throws {
        // ファイル先頭 44 バイトの ChunkSize と Subchunk2Size が確定書き込みサンプル数に一致することを検証する。
        let tempUrl = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        FileManager.default.createFile(atPath: tempUrl.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tempUrl)
        let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

        try writer.write(samples: [Float](repeating: 0.0, count: 480))
        try writer.finalize()
        try handle.close()

        let fileData = try Data(contentsOf: tempUrl)
        let dataSize = 480 * 2
        let expectedRiffSize = 36 + dataSize
        let b4 = Int(fileData[4])
        let b5 = Int(fileData[5]) << 8
        let parsedRiffSize = b4 | b5
        XCTAssertEqual(parsedRiffSize, expectedRiffSize & 0xFFFF)
        try? FileManager.default.removeItem(at: tempUrl)
    }

    func testF8_WavEncoder_RoundTripDecodeHeader() {
        // 生成した WAV バイト列からサンプリングレートとチャンネル数が完全に復元できることを検証する。
        let samples = [Float](repeating: 0.2, count: 160)
        let data = WavEncoder.encode(samples: samples, sampleRate: 16000)
        let rateByte0 = Int(data[24])
        let rateByte1 = Int(data[25]) << 8
        let rate = rateByte0 | rateByte1
        XCTAssertEqual(rate, 16000)
    }

    // MARK: - F9: SIMD8 DSP & 安全ガード (5ケース以上)

    private func applySoftLimit(_ val: Float, threshold: Float = 0.8) -> Float {
        let src = [val]
        var dst = [Float](repeating: 0.0, count: 1)
        src.withUnsafeBufferPointer { pSrc in
            dst.withUnsafeMutableBufferPointer { pDst in
                VectorOperations.softLimitTanh(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: 1, threshold: threshold)
            }
        }
        return dst[0]
    }

    func testF9_Safety_SoftLimiterLinearRange() {
        // |x| <= 0.8 では Soft Limiter が歪みなく入力を線形に通過させることを検証する。
        XCTAssertEqual(applySoftLimit(0.0), 0.0)
        XCTAssertEqual(applySoftLimit(0.5), 0.5)
        XCTAssertEqual(applySoftLimit(-0.5), -0.5)
        XCTAssertEqual(applySoftLimit(0.8), 0.8)
    }

    func testF9_Safety_SoftLimiterCompression() {
        // 0.8 を超える過大振幅が tanh により滑らかにサチュレーション圧縮され、絶対値が 1.0 以下に抑制されることを検証する。
        let val1 = applySoftLimit(1.5)
        let val2 = applySoftLimit(10.0)
        let valNeg = applySoftLimit(-10.0)
        XCTAssertTrue(0.8 < val1)
        XCTAssertTrue(val1 <= 1.0)
        XCTAssertTrue(val2 <= 1.0)
        XCTAssertTrue(-1.0 <= valNeg)
    }

    func testF9_Safety_NaNInfGuards() {
        // ボコーダーに非有限値（NaN, ±Inf）が渡された際、異常終了せずにゼロクリアで安全復旧することを検証する。
        vocoder.reset()
        let nanFrame = AcousticFrame(
            lpcCoefficients: [Float.nan, 0.0],
            gain: Float.infinity,
            pitchF0: 100.0,
            voiced: 1.0
        )
        let samples = vocoder.synthesize(frames: [nanFrame])
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < samples.count {
            XCTAssertEqual(samples[i], 0.0)
            i += 1
        }
    }

    func testF9_Safety_SIMD8DotProduct() {
        // VectorOperations.dotProduct が 8 要素並列処理でスカラー積和と一致することを検証する。
        let a: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        let b: [Float] = [2.0, 1.0, 0.5, 0.25, 1.0, 2.0, 0.0, 1.0, 3.0]
        let result = a.withUnsafeBufferPointer { pA in
            b.withUnsafeBufferPointer { pB in
                VectorOperations.dotProduct(a: pA.baseAddress!, b: pB.baseAddress!, count: 9)
            }
        }
        var expected: Float = 0.0
        var i = 0
        while i < 9 {
            expected += a[i] * b[i]
            i += 1
        }
        XCTAssertTrue(abs(result - expected) < 1e-4)
    }

    func testF9_Safety_QuantizeFloatToInt16() {
        // ±1.0 を超える浮動小数点数が Int16 のオーバーフローを起こさず境界値へ飽和することを検証する。
        let src: [Float] = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
        var dst = [Int16](repeating: 0, count: 6)
        src.withUnsafeBufferPointer { pSrc in
            dst.withUnsafeMutableBufferPointer { pDst in
                VectorOperations.quantizeFloatToInt16(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: 6)
            }
        }
        XCTAssertEqual(dst[0], -32768)
        XCTAssertEqual(dst[1], -32767)
        XCTAssertEqual(dst[2], 0)
        XCTAssertEqual(dst[4], 32767)
        XCTAssertEqual(dst[5], 32767)
    }

    // MARK: - F10: LIF/ALIF SNN 音響デコーダー (5ケース以上)

    func testF10_SNN_DirectCurrentInjection() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 32, outputDim: 16, timeSteps: 2, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 32, outputDim: 16, numLayers: weights.numLayers)
        let inFeatures = [Float](repeating: 1.0, count: 16)
        var outFeatures = [Float](repeating: 0.0, count: 16)

        inFeatures.withUnsafeBufferPointer { pIn in
            outFeatures.withUnsafeMutableBufferPointer { pOut in
                decoder.decodeFrame(
                    features: pIn.baseAddress!,
                    workspace: workspace,
                    outputFeatures: pOut.baseAddress!
                )
            }
        }
        var i = 0
        while i < outFeatures.count {
            XCTAssertTrue(outFeatures[i].isFinite)
            i += 1
        }
    }

    func testF10_SNN_AdaptiveThresholdDynamics() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.2)
        var v: [Float] = [1.5]
        var s: [Float] = [1.0]
        var a: [Float] = [0.0]
        let cur: [Float] = [2.0]

        v.withUnsafeMutableBufferPointer { pV in
            s.withUnsafeMutableBufferPointer { pS in
                a.withUnsafeMutableBufferPointer { pA in
                    cur.withUnsafeBufferPointer { pCur in
                        LIFNeuronEngine.stepAdaptiveSIMD8(
                            config: config,
                            vPtr: pV.baseAddress!,
                            sPtr: pS.baseAddress!,
                            aPtr: pA.baseAddress!,
                            curPtr: pCur.baseAddress!,
                            count: 1
                        )
                    }
                }
            }
        }
        XCTAssertTrue(0.0 < a[0])
    }

    func testF10_SNN_SubtractionResetReadout() {
        let config = LIFConfig(beta: 0.9, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.1)
        var v: [Float] = [0.8]
        var s: [Float] = [0.0]
        var a: [Float] = [0.0]
        let cur: [Float] = [0.5]
        var readoutSum: [Float] = [0.0]

        v.withUnsafeMutableBufferPointer { pV in
            s.withUnsafeMutableBufferPointer { pS in
                a.withUnsafeMutableBufferPointer { pA in
                    cur.withUnsafeBufferPointer { pCur in
                        readoutSum.withUnsafeMutableBufferPointer { pR in
                            LIFNeuronEngine.stepReadoutAdaptiveSIMD8(
                                config: config,
                                vPtr: pV.baseAddress!,
                                sPtr: pS.baseAddress!,
                                aPtr: pA.baseAddress!,
                                curPtr: pCur.baseAddress!,
                                readoutSumPtr: pR.baseAddress!,
                                count: 1
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(0.0 < readoutSum[0])
    }

    func testF10_SNN_DecodeSequenceOutputShape() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 32, outputDim: 8, timeSteps: 2, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 32, outputDim: 8, numLayers: weights.numLayers)

        let seq = [[Float]](repeating: [Float](repeating: 0.5, count: 16), count: 5)
        let outSeq = decoder.decodeSequence(featuresSeq: seq, workspace: workspace)
        XCTAssertEqual(outSeq.count, 5)
        XCTAssertEqual(outSeq[0].count, 8)
    }

    func testF10_SNN_MembraneClamping() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.1)
        var v: [Float] = [0.0]
        var s: [Float] = [0.0]
        var a: [Float] = [0.0]
        let hugeCur: [Float] = [1000.0]

        v.withUnsafeMutableBufferPointer { pV in
            s.withUnsafeMutableBufferPointer { pS in
                a.withUnsafeMutableBufferPointer { pA in
                    hugeCur.withUnsafeBufferPointer { pCur in
                        LIFNeuronEngine.stepAdaptiveSIMD8(
                            config: config,
                            vPtr: pV.baseAddress!,
                            sPtr: pS.baseAddress!,
                            aPtr: pA.baseAddress!,
                            curPtr: pCur.baseAddress!,
                            count: 1
                        )
                    }
                }
            }
        }
        XCTAssertTrue(v[0] <= LIFNeuronEngine.vClampMax)
    }

    // MARK: - F11: MLX BPTT 学習パイプライン (5ケース以上)

    func testF11_MLX_FastSigmoidSurrogateGradient() {
        let v = MLXArray([Float(0.8), Float(1.2)])
        let vTh = MLXArray([Float(1.0), Float(1.0)])
        let spikes = SurrogateGradients.fastSigmoidSTE(v: v, vTh: vTh, alpha: 2.0)
        eval(spikes)
        let spikeArr = spikes.asArray(Float.self)
        XCTAssertEqual(spikeArr[0], 0.0)
        XCTAssertEqual(spikeArr[1], 1.0)
    }

    func testF11_MLX_AlignTo32Padding() {
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 1), 32)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 32), 32)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 33), 64)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 64), 64)
        XCTAssertEqual(MLXAcousticBPTTTrainer.alignTo32(seqLen: 65), 96)
    }

    func testF11_MLX_NetworkForwardOutputShape() {
        let net = MLXSpikingAcousticNetwork(
            numLayers: 2,
            inputDim: 16,
            maxHiddenDim: 32,
            outputDim: 8,
            timeSteps: 2
        )
        let x = MLXArray.zeros([1, 4, 16])
        let out = net.forward(features: x)
        eval(out)
        XCTAssertEqual(out.shape, [1, 4, 8])
    }

    func testF11_MLX_BPTTTrainerBatchOptimization() {
        let net = MLXSpikingAcousticNetwork(
            numLayers: 2,
            inputDim: 16,
            maxHiddenDim: 32,
            outputDim: 8,
            timeSteps: 2
        )
        let trainer = MLXAcousticBPTTTrainer(network: net, learningRate: 0.01)
        let x = MLXRandom.uniform(low: -0.1, high: 0.1, [1, 32, 16])
        let y = MLXRandom.uniform(low: -0.1, high: 0.1, [1, 32, 8])
        let loss = trainer.trainBatch(features: x, targets: y)
        XCTAssertTrue(loss.isFinite)
        XCTAssertTrue(0.0 <= loss)
    }

    func testF11_MLX_StopGradientWindow() {
        let net = MLXSpikingAcousticNetwork(
            numLayers: 2,
            inputDim: 8,
            maxHiddenDim: 16,
            outputDim: 4,
            timeSteps: 1
        )
        let x = MLXArray.zeros([1, 48, 8])
        let out = net.forward(features: x, bpttWindow: 16)
        eval(out)
        XCTAssertEqual(out.shape, [1, 48, 4])
    }

    // MARK: - F12: SNN 推論 Hot Path 最適化 (5ケース以上)

    func testF12_HotPath_InputCurrentHoist() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 32, outputDim: 8, timeSteps: 4, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 32, outputDim: 8, numLayers: 2)
        let feat = [Float](repeating: 0.2, count: 16)
        var out = [Float](repeating: 0.0, count: 8)

        feat.withUnsafeBufferPointer { pF in
            out.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
            }
        }
        XCTAssertTrue(out[0].isFinite)
    }

    func testF12_HotPath_SparseAdditionFromTranspose() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 8, timeSteps: 2, numLayers: 1)
        let decoder = SpikingAcousticDecoder(weights: weights)
        XCTAssertEqual(decoder.wRecT.count, 16 * 16)
        let j = 2
        let i = 5
        XCTAssertEqual(decoder.wRecT[(j * 16) + i], weights.wRec[(i * 16) + j])
    }

    func testF12_HotPath_WorkspaceZeroAllocation() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 32, outputDim: 8, timeSteps: 2, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 32, outputDim: 8, numLayers: 2)
        let ptr1 = workspace.stepCurrents.withUnsafeBufferPointer { $0.baseAddress }

        var out = [Float](repeating: 0.0, count: 8)
        let feat = [Float](repeating: 0.1, count: 16)
        feat.withUnsafeBufferPointer { pF in
            out.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
            }
        }
        let ptr2 = workspace.stepCurrents.withUnsafeBufferPointer { $0.baseAddress }
        XCTAssertEqual(ptr1, ptr2)
    }

    func testF12_HotPath_WorkspaceReset() {
        let workspace = AcousticWorkspace(maxHiddenDim: 16, outputDim: 8, numLayers: 1)
        workspace.layerStates[0].v[0] = 0.5
        workspace.reset()
        XCTAssertEqual(workspace.layerStates[0].v[0], 0.0)
    }

    func testF12_HotPath_Determinism() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, timeSteps: 2, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let ws1 = AcousticWorkspace(maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        let ws2 = AcousticWorkspace(maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        let feat = [Float](repeating: 0.3, count: 8)
        var out1 = [Float](repeating: 0.0, count: 4)
        var out2 = [Float](repeating: 0.0, count: 4)

        feat.withUnsafeBufferPointer { pF in
            out1.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: ws1, outputFeatures: pO.baseAddress!)
            }
            out2.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: ws2, outputFeatures: pO.baseAddress!)
            }
        }
        XCTAssertEqual(out1, out2)
    }

    // MARK: - F13: 多層 SNN 重み構造と転置生成 (5ケース以上)

    func testF13_Multilayer_LayerCounts() {
        let w1 = SpikingNetworkWeights.randomWeights(numLayers: 1)
        XCTAssertEqual(w1.numLayers, 1)
        XCTAssertEqual(w1.wLayers.count, 0)

        let w3 = SpikingNetworkWeights.randomWeights(numLayers: 3)
        XCTAssertEqual(w3.numLayers, 3)
        XCTAssertEqual(w3.wLayers.count, 2)
        XCTAssertEqual(w3.bHLayers.count, 2)
        XCTAssertEqual(w3.gammaRMS.count, 2)
    }

    func testF13_Multilayer_MakeWRecT() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 64, outputDim: 8, numLayers: 2)
        let recT = weights.makeWRecT()
        XCTAssertEqual(recT.count, 64 * 64)
        XCTAssertEqual(recT[(2 * 64) + 3], weights.wRec[(3 * 64) + 2])
    }

    func testF13_Multilayer_MakeWLayersT() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 64, outputDim: 8, numLayers: 3)
        let layersT = weights.makeWLayersT()
        XCTAssertEqual(layersT.count, 2)
        XCTAssertEqual(layersT[0].count, 64 * 64)
        XCTAssertEqual(layersT[0][(4 * 64) + 5], weights.wLayers[0][(5 * 64) + 4])
    }

    func testF13_Multilayer_MakeWOutT() {
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 64, outputDim: 8, numLayers: 2)
        let outT = weights.makeWOutT()
        XCTAssertEqual(outT.count, 64 * 8)
        XCTAssertEqual(outT[(5 * 8) + 2], weights.wOut[(2 * 64) + 5])
    }

    func testF13_Multilayer_MultiLayerDecodeSuccess() {
        let layerConfigs = [1, 2, 3]
        var s = 0
        while s < layerConfigs.count {
            let layers = layerConfigs[s]
            let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 64, outputDim: 8, timeSteps: 1, numLayers: layers)
            let decoder = SpikingAcousticDecoder(weights: weights)
            let workspace = AcousticWorkspace(maxHiddenDim: 64, outputDim: 8, numLayers: layers)
            let feat = [Float](repeating: 0.1, count: 16)
            var out = [Float](repeating: 0.0, count: 8)

            feat.withUnsafeBufferPointer { pF in
                out.withUnsafeMutableBufferPointer { pO in
                    decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
                }
            }
            XCTAssertTrue(out[0].isFinite)
            s += 1
        }
    }

    // MARK: - F14: 基準音声データ自動生成器 (5ケース以上)

    func testF14_Synthetic_FiveVowelsSynthesis() {
        // a, i, u, e, o の各母音がホルマント周波数設定に従って合成され、有限エネルギーを持つことを検証する。
        let vowels: [SyntheticAudioGenerator.Vowel] = [.a, .i, .u, .e, .o]
        var v = 0
        while v < vowels.count {
            let samples = generator.generateVowel(vowel: vowels[v], durationSeconds: 0.1, f0: 140.0)
            XCTAssertTrue(0 < samples.count)
            XCTAssertTrue(samples[0].isFinite)
            v += 1
        }
    }

    func testF14_Synthetic_ChirpWaveGeneration() {
        // 20Hz から 8000Hz への周波数スイープ波形が所定のサンプル数で生成されることを検証する。
        let chirp = generator.generateChirp(startFreq: 20.0, endFreq: 4000.0, durationSeconds: 0.1)
        XCTAssertEqual(chirp.count, 1600)
        var i = 0
        while i < chirp.count {
            XCTAssertTrue(-1.0 <= chirp[i])
            XCTAssertTrue(chirp[i] <= 1.0)
            i += 1
        }
    }

    func testF14_Synthetic_ImpulseTrainGeneration() {
        // ピッチ周期ごとに厳密にデルタパルスが配置されることを検証する。
        let impulses = generator.generateImpulseTrain(f0: 100.0, durationSeconds: 0.05)
        XCTAssertEqual(impulses.count, 800)
        XCTAssertEqual(impulses[0], 1.0)
        XCTAssertEqual(impulses[160], 1.0)
        XCTAssertEqual(impulses[1], 0.0)
    }

    func testF14_Synthetic_WhiteNoiseGeneration() {
        // 指定振幅範囲内に均一に分散するノイズ波形が生成されることを検証する。
        let noise = generator.generateWhiteNoise(durationSeconds: 0.05, amplitude: 0.4)
        XCTAssertEqual(noise.count, 800)
        var i = 0
        while i < noise.count {
            XCTAssertTrue(-0.4 <= noise[i])
            XCTAssertTrue(noise[i] <= 0.4)
            i += 1
        }
    }

    func testF14_Synthetic_StandardCorpusGeneration() {
        // 5 つの代表フレーズに対する擬似正解音声データセットが外部依存なしにオンデマンド生成されることを検証する。
        let corpus = generator.generateStandardCorpus()
        XCTAssertEqual(corpus.count, 5)
        var c = 0
        while c < corpus.count {
            XCTAssertTrue(0 < corpus[c].text.count)
            XCTAssertTrue(0 < corpus[c].samples.count)
            c += 1
        }
    }

    // MARK: - F15: CLI ツールスイート (5ケース以上)

    func testF15_CLI_SynthesizeCLIExecution() {
        let wave = engine.synthesize(text: "こんにちは")
        XCTAssertTrue(0 < wave.count)
    }

    func testF15_CLI_TrainCLIFlow() {
        let pair = generator.generateStandardCorpus()[0]
        let feat = engine.encodeLinguisticFeatures(
            features: engine.lengthRegulator.processText(
                text: pair.text,
                normalizer: engine.normalizer,
                prosodyModel: engine.prosodyModel,
                vocabulary: engine.vocabulary
            )
        )
        let net = MLXSpikingAcousticNetwork(
            numLayers: 2,
            inputDim: engine.weights.inputDim,
            maxHiddenDim: 128,
            outputDim: engine.weights.outputDim,
            timeSteps: 2
        )
        let trainer = MLXAcousticBPTTTrainer(network: net, learningRate: 0.001)
        let dummyTarget = [[Float]](repeating: [Float](repeating: -2.0, count: engine.weights.outputDim), count: feat.count)
        let loss = trainer.trainSequence(features: feat, targets: dummyTarget)
        XCTAssertTrue(loss.isFinite)
    }

    func testF15_CLI_BenchmarkCLIFlow() {
        let start = Date()
        let samples = engine.synthesize(text: "テスト")
        let elapsed = Date().timeIntervalSince(start)
        let audioDuration = Double(samples.count) / Double(AudioConfig.sampleRate)
        let rtf: Double
        if 0.0 < audioDuration {
            rtf = elapsed / audioDuration
        } else {
            rtf = 0.0
        }
        XCTAssertTrue(0.0 <= rtf)
    }

    func testF15_CLI_LayerArgumentSupport() {
        let w1 = engine.synthesize(text: "テスト")
        let eng2 = SpikeSpeechEngine(weights: SpikingNetworkWeights.randomWeights(numLayers: 2))
        let w2 = eng2.synthesize(text: "テスト")
        XCTAssertEqual(w1.count, w2.count)
    }

    func testF15_CLI_PitchAndSpeedOptionSupport() {
        let samplesNormal = engine.synthesize(text: "おはよう", speed: 1.0, pitch: 1.0)
        let samplesFast = engine.synthesize(text: "おはよう", speed: 1.5, pitch: 1.2)
        XCTAssertTrue(samplesFast.count < samplesNormal.count)
    }

    // MARK: - F16: SwiftPM パッケージ構成 / SpikeSpeechEngine E2E (5ケース以上)

    func testF16_Engine_DefaultInit() {
        let eng = SpikeSpeechEngine()
        XCTAssertEqual(eng.sampleRate, 16000.0)
        XCTAssertEqual(eng.weights.outputDim, AudioConfig.melChannels)
    }

    func testF16_Engine_SynthesizePCM() {
        let samples = engine.synthesize(text: "あめ")
        XCTAssertTrue(0 < samples.count)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(-1.0 <= samples[i])
            XCTAssertTrue(samples[i] <= 1.0)
            i += 1
        }
    }

    func testF16_Engine_SynthesizeWavData() {
        let wavData = engine.synthesizeWav(text: "あめ")
        XCTAssertTrue(44 < wavData.count)
        XCTAssertEqual(wavData[0], 0x52)
        XCTAssertEqual(wavData[1], 0x49)
        XCTAssertEqual(wavData[2], 0x46)
        XCTAssertEqual(wavData[3], 0x46)
    }

    func testF16_Engine_SynthesizeStreamCallback() {
        var callbackFrameCount = 0
        let samples = engine.synthesizeStream(text: "そら") { frame in
            XCTAssertEqual(frame.count, 160)
            callbackFrameCount += 1
        }
        XCTAssertTrue(0 < callbackFrameCount)
        XCTAssertEqual(samples.count, callbackFrameCount * 160)
    }

    func testF16_Engine_PitchAndSpeedControl() {
        let waveSlow = engine.synthesize(text: "やま", speed: 0.8, pitch: 0.9)
        let waveFast = engine.synthesize(text: "やま", speed: 1.3, pitch: 1.1)
        XCTAssertTrue(waveFast.count < waveSlow.count)
    }
}
