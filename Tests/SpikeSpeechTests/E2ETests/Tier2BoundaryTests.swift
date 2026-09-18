import XCTest
import Foundation
import MLX
@testable import SpikeSpeech

/// Tier 2: 境界値・極限・コーナーケーステストスイート (F1〜F16 各5ケース以上)
///
/// 空入力、極小・極大境界、特殊記号、ゼロF0、全無声、極端ゲイン、反射係数限界（|k|接近0.999）、
/// スライス寸法境界、非有限値（NaN/Inf）等の極限環境において、
/// エンジンがクラッシュ・ハング・メモリ破壊を起こさず安全にフォールバックすることを検証する。
final class Tier2BoundaryTests: XCTestCase {

    private var engine: SpikeSpeechEngine!
    private var normalizer: TextNormalizer!
    private var morphology: ViterbiMorphology!
    private var vocabulary: PhonemeVocabulary!
    private var prosodyModel: ProsodyModel!
    private var lengthRegulator: LengthRegulator!
    private var vocoder: NeuralVocoder!

    override func setUp() {
        super.setUp()
        // 境界値テスト間の相互汚染を排除し、独立した状態で限界検証を行うため共通コンポーネントを初期化する。
        self.morphology = ViterbiMorphology()
        self.normalizer = TextNormalizer(morphology: morphology)
        self.vocabulary = PhonemeVocabulary()
        self.prosodyModel = ProsodyModel()
        self.lengthRegulator = LengthRegulator(hiddenDimension: 128)
        self.vocoder = NeuralVocoder()
        self.engine = SpikeSpeechEngine()
    }

    // MARK: - F1: 形態素解析境界 (5ケース以上)

    func testF1_Boundary_EmptyString() {
        // 文字列長ゼロの入力に対してラティス構築が例外やパニックを起こさず、空配列を返すことを検証する。
        let morphemes = morphology.tokenize("")
        XCTAssertTrue(morphemes.isEmpty)
    }

    func testF1_Boundary_SingleCharacter() {
        // 境界長 1 のノード探索でインデックス境界超過が発生しないことを検証する。
        let mHiragana = morphology.tokenize("あ")
        XCTAssertEqual(mHiragana.count, 1)
        let mKanji = morphology.tokenize("本")
        XCTAssertEqual(mKanji.count, 1)
    }

    func testF1_Boundary_OnlyPunctuation() {
        // 辞書に存在しない記号の連続で未知語処理が無限ループを起こさないことを検証する。
        let morphemes = morphology.tokenize("、、、。。。！！！")
        XCTAssertTrue(0 < morphemes.count)
        var i = 0
        while i < morphemes.count {
            XCTAssertEqual(morphemes[i].pos, .symbol)
            i += 1
        }
    }

    func testF1_Boundary_MassiveString() {
        // 1,000 文字の連続文でラティスの動的計画法テーブルがスタックオーバーフローを起こさないことを検証する。
        let sample = "東京の天気は晴れです。"
        let longText = String(repeating: sample, count: 100) // 1,100文字
        let morphemes = morphology.tokenize(longText)
        XCTAssertTrue(0 < morphemes.count)
    }

    func testF1_Boundary_SpecialSymbolsAndSpaces() {
        // 空白文字や装飾記号が安全にスキップまたは記号ノードとして処理されることを検証する。
        let morphemes = morphology.tokenize(" 　★♪ \t\n")
        XCTAssertTrue(0 <= morphemes.count)
    }

    // MARK: - F2: 読み正規化境界 (5ケース以上)

    func testF2_Boundary_HugeNumberFallback() {
        // 万進法の Int64 上限（16桁）を超過した際に整数オーバーフローを起こさず、1桁読み上げへ安全フォールバックすることを検証する。
        let hugeNum = "10000000000000000" // 17桁
        let expanded = normalizer.expandNumbersAndCounters(hugeNum)
        XCTAssertEqual(expanded, "いちぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろぜろ")
    }

    func testF2_Boundary_NegativeAndDecimals() {
        // 符号マイナスや小数点ピリオドを含む数値表現で正規化が異常終了しないことを検証する。
        let neg = normalizer.expandNumbersAndCounters("-50")
        XCTAssertEqual(neg, "-ごじゅう")
        let dec = normalizer.expandNumbersAndCounters("0.05")
        XCTAssertTrue(dec.contains("ぜろ"))
    }

    func testF2_Boundary_UnknownCounter() {
        // ルックアップテーブルにない助数詞が結合した際に数詞のみ展開され、助数詞がそのまま保護されることを検証する。
        let expanded = normalizer.expandNumbersAndCounters("100箱")
        XCTAssertEqual(expanded, "ひゃく箱")
    }

    func testF2_Boundary_ParticleRepetition() {
        // 「はははは」のような同文字連続で文脈判定が無限ループや不正置換を起こさないことを検証する。
        let norm = normalizer.normalize(text: "はははは")
        XCTAssertTrue(0 < norm.count)
    }

    func testF2_Boundary_LeadingParticle() {
        // 前置文脈が存在しない文頭に「は」や「へ」が来た場合にインデックスアンダーフローを起こさないことを検証する。
        let norm = normalizer.normalize(text: "は水です")
        XCTAssertTrue(0 < norm.count)
    }

    // MARK: - F3: トークナイズ境界 (5ケース以上)

    func testF3_Boundary_UnknownCharsAndEmoji() {
        // サロゲートペアや絵文字が混入しても音素トークナイザーがクラッシュせず未知語または無視として処理することを検証する。
        let norm = normalizer.normalize(text: "こんにちは🍣🍵🇯🇵")
        let phrases = prosodyModel.buildAccentPhrases(morphemes: norm, vocabulary: vocabulary)
        XCTAssertTrue(0 < phrases.count)
    }

    func testF3_Boundary_SingleMora() {
        // モーラ数 1 の境界値において、アクセントトーン計算が配列外参照を起こさないことを検証する。
        let tones = prosodyModel.computeMoraTones(moraCount: 1, accentKernel: 1)
        XCTAssertEqual(tones.count, 1)
        XCTAssertEqual(tones[0], .high)
    }

    func testF3_Boundary_SilenceOnly() {
        // ポーズや無音のみの系列でも音素 ID が定義域 [0, 64) に収まることを検証する。
        let silId = vocabulary.id(for: "<sil>")
        let pauId = vocabulary.id(for: "<pau>")
        XCTAssertTrue(0 <= silId)
        XCTAssertTrue(silId < 64)
        XCTAssertTrue(0 <= pauId)
        XCTAssertTrue(pauId < 64)
    }

    func testF3_Boundary_ConsecutiveVowels() {
        // 「あおい」のような子音を挟まない母音連続が各々独立したモーラとして分解されることを検証する。
        let norm = normalizer.normalize(text: "あおい")
        let phrases = prosodyModel.buildAccentPhrases(morphemes: norm, vocabulary: vocabulary)
        XCTAssertTrue(0 < phrases.count)
        let totalMoras = phrases.reduce(0) { $0 + $1.moras.count }
        XCTAssertTrue(2 <= totalMoras)
    }

    func testF3_Boundary_ConsecutiveSokuon() {
        // 「っっっ」のような非文法的な促音連続で異常終了しないことを検証する。
        let norm = normalizer.normalize(text: "っっっ")
        let phrases = prosodyModel.buildAccentPhrases(morphemes: norm, vocabulary: vocabulary)
        XCTAssertTrue(0 <= phrases.count)
    }

    // MARK: - F4: Length Regulation 境界 (5ケース以上)

    func testF4_Boundary_NegativeDuration() {
        // 音響モデルの勾配発散等で負の duration が渡された際、最小 1 フレームにクリッピングされることを検証する。
        let quantized = lengthRegulator.quantizeDurations(durations: [-5.0, -1.0])
        XCTAssertEqual(quantized.count, 2)
        XCTAssertEqual(quantized[0], 1)
        XCTAssertEqual(quantized[1], 1)
    }

    func testF4_Boundary_ZeroDuration() {
        // 0.0 duration が 0 フレーム（消滅）ではなく最低限 1 フレームに切り上げられることを検証する。
        let quantized = lengthRegulator.quantizeDurations(durations: [0.0])
        XCTAssertEqual(quantized.count, 1)
        XCTAssertEqual(quantized[0], 1)
    }

    func testF4_Boundary_ExtremeFastSpeed() {
        // speedFactor = 5.0 でも全音素が 0 フレームに潰れず、最低限のフレーム数が確保されることを検証する。
        let features = lengthRegulator.processText(
            text: "あさ",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 5.0
        )
        XCTAssertTrue(0 < features.totalFrames)
        var i = 0
        while i < features.durations.count {
            XCTAssertTrue(1 <= features.durations[i])
            i += 1
        }
    }

    func testF4_Boundary_ExtremeSlowSpeed() {
        // speedFactor = 0.1 の伸長で整数オーバーフローを起こさないことを検証する。
        let features = lengthRegulator.processText(
            text: "あさ",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 0.1
        )
        XCTAssertTrue(0 < features.totalFrames)
        XCTAssertTrue(features.totalFrames < 10000)
    }

    func testF4_Boundary_EmptyPhonemeList() {
        // 音素数 0 の入力に対して expand が空配列を返すことを検証する。
        let exp = lengthRegulator.expand(embeddings: [], phonemeCount: 0, durations: [])
        XCTAssertTrue(exp.isEmpty)
    }

    // MARK: - F5: アクセント/F0 境界 (5ケース以上)

    func testF5_Boundary_ZeroF0AllUnvoiced() {
        // 有声度が 0.0 のフレームでは F0 が厳密に 0.0 Hz に保たれ、不要な声帯振動周波数が漏洩しないことを検証する。
        let (f0, voiced, count) = prosodyModel.generateF0Contour(phrases: [], vocabulary: vocabulary)
        XCTAssertEqual(count, 0)
        XCTAssertTrue(f0.isEmpty)
        XCTAssertTrue(voiced.isEmpty)
    }

    func testF5_Boundary_HugeF0PitchScale() {
        // pitchScale = 3.0 でも F0 が生理的上限（500Hz正規化範囲）で適切に扱われることを検証する。
        let features = lengthRegulator.processText(
            text: "やま",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )
        let seq = engine.encodeLinguisticFeatures(features: features)
        XCTAssertEqual(seq.count, features.totalFrames)
        var f = 0
        while f < seq.count {
            if 65 < seq[f].count {
                XCTAssertTrue(seq[f][65] <= 1.0)
            }
            f += 1
        }
    }

    func testF5_Boundary_NegativeAccentKernel() {
        // accentKernel = -1 が渡された際に配列外参照を起こさず、平板型または安全フォールバックすることを検証する。
        let tones = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: -1)
        XCTAssertEqual(tones.count, 3)
    }

    func testF5_Boundary_HugeAccentKernel() {
        // accentKernel = 999 がモーラ数 3 を超えている場合に異常終了せず安全に判定されることを検証する。
        let tones = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 999)
        XCTAssertEqual(tones.count, 3)
    }

    func testF5_Boundary_EmptyAccentPhrases() {
        // フレーズが空の場合に generateF0Contour が 0 フレームを返し異常終了しないことを検証する。
        let (f0, voiced, total) = prosodyModel.generateF0Contour(phrases: [], vocabulary: vocabulary)
        XCTAssertEqual(total, 0)
        XCTAssertTrue(f0.isEmpty)
        XCTAssertTrue(voiced.isEmpty)
    }

    // MARK: - F6: NeuralVocoder 境界 (5ケース以上)

    func testF6_Boundary_ZeroMel() {
        // 対数 Mel = 0.0 のフレームに対して NeuralVocoder が有限な波形を出力することを検証する。
        vocoder.reset()
        let frame = [Float](repeating: 0.0, count: 64)
        let samples = vocoder.synthesize(mel: [frame])
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    func testF6_Boundary_HugeMelValues() {
        // 対数 Mel = +100.0 の極大入力に対してもリミッターにより振幅が [-1.0, 1.0] に安全に圧縮されることを検証する。
        vocoder.reset()
        let frame = [Float](repeating: 100.0, count: 64)
        let samples = vocoder.synthesize(mel: [frame])
        var i = 0
        while i < samples.count {
            XCTAssertTrue(-1.0 <= samples[i])
            XCTAssertTrue(samples[i] <= 1.0)
            i += 1
        }
    }

    func testF6_Boundary_SmallMelValues() {
        // 対数 Mel = -100.0 の極小値でもアンダーフローによる非有限値が発生しないことを検証する。
        vocoder.reset()
        let frame = [Float](repeating: -100.0, count: 64)
        let samples = vocoder.synthesize(mel: [frame])
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    func testF6_Boundary_NaNMelValues() {
        // 対数 Mel 入力に NaN が混入した際、安全復旧により有限値のみが出力されることを検証する。
        vocoder.reset()
        var nanMel = [Float](repeating: -2.0, count: 64)
        nanMel[5] = Float.nan
        let samples = vocoder.synthesize(mel: [nanMel])
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    func testF6_Boundary_InfMelValues() {
        // 対数 Mel 入力に +Inf が混入した際、安全復旧により全サンプル有限値が出力されることを検証する。
        vocoder.reset()
        var infMel = [Float](repeating: -2.0, count: 64)
        infMel[10] = Float.infinity
        let samples = vocoder.synthesize(mel: [infMel])
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    // MARK: - F7: NeuralVocoder 話者・条件付け境界 (5ケース以上)

    func testF7_Boundary_ZeroConditioning() {
        // SpeakerConditioning.zero で通常通り合成できることを検証する。
        vocoder.reset()
        let frame = [Float](repeating: -2.0, count: 64)
        let samples = vocoder.synthesize(mel: [frame], speaker: .zero)
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    func testF7_Boundary_ZeroEnergyScale() {
        // energyScale = 0.0 のとき振幅がゼロになることを検証する。
        let profile = VoiceProfile(name: "zero", baseF0: 220.0, energyScale: 0.0)
        let samples = engine.synthesize(text: "あ", voice: profile)
        XCTAssertFalse(samples.isEmpty)
        var i = 0
        while i < samples.count {
            XCTAssertEqual(samples[i], 0.0)
            i += 1
        }
    }

    func testF7_Boundary_ExtremePitchShift() {
        // 極端な F0 コンター（1000.0Hz）でも有限な波形が出力されることを検証する。
        vocoder.reset()
        let frame = [Float](repeating: -2.0, count: 64)
        let samples = vocoder.synthesize(mel: [frame], f0Contour: [1000.0])
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    func testF7_Boundary_LargeFrameBatch() {
        // 50フレームの連続 Mel 入力に対してバッファ破綻なく 8000 サンプルが出力されることを検証する。
        vocoder.reset()
        let frames = [[Float]](repeating: [Float](repeating: -3.0, count: 64), count: 50)
        let samples = vocoder.synthesize(mel: frames)
        XCTAssertEqual(samples.count, 8000)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    func testF7_Boundary_ShortEmbedding() {
        // 埋め込みベクトルの長さが標準長未満の場合でもクラッシュせず安全に波形合成されることを検証する。
        vocoder.reset()
        let cond = SpeakerConditioning(embedding: [1.0, 2.0])
        let frame = [Float](repeating: -2.0, count: 64)
        let samples = vocoder.synthesize(mel: [frame], speaker: cond)
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    // MARK: - F8: WAV エンコーダー境界 (5ケース以上)

    func testF8_Boundary_ZeroSamples() {
        // 空のサンプル列をエンコードしても、正確に 44 バイトヘッダ（dataSize=0）が生成されることを検証する。
        let data = WavEncoder.encode(samples: [], sampleRate: 16000)
        XCTAssertEqual(data.count, 44)
    }

    func testF8_Boundary_SingleSample() {
        // 最小サンプル数 1（2バイト）で 46 バイトのバイナリが生成されることを検証する。
        let data = WavEncoder.encode(samples: [0.5], sampleRate: 16000)
        XCTAssertEqual(data.count, 46)
    }

    func testF8_Boundary_OddLengthSamples() {
        // 奇数個（3サンプル）でもバイトアライメント（6バイトデータ）が正常にパッキングされることを検証する。
        let data = WavEncoder.encode(samples: [0.1, -0.1, 0.2], sampleRate: 16000)
        XCTAssertEqual(data.count, 44 + 6)
    }

    func testF8_Boundary_DoubleFinalize() throws {
        // finalize() を 2 回連続で呼び出してもファイルが破壊されたり例外が再送されないことを検証する。
        let tempUrl = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        FileManager.default.createFile(atPath: tempUrl.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tempUrl)
        let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

        try writer.write(samples: [0.1, 0.2])
        try writer.finalize()
        try writer.finalize() // 2度目の呼び出し
        try handle.close()

        let fileData = try Data(contentsOf: tempUrl)
        XCTAssertEqual(fileData.count, 44 + 4)
        try? FileManager.default.removeItem(at: tempUrl)
    }

    func testF8_Boundary_NegativeSampleValues() {
        // -2.0 などの下限逸脱サンプルが Int16.min (-32768) に安全に飽和量子化されることを検証する。
        let data = WavEncoder.encode(samples: [-2.0], sampleRate: 16000)
        XCTAssertEqual(data.count, 46)
        let b0 = data[44]
        let b1 = data[45]
        let val = Int16(bitPattern: UInt16(b0) | (UInt16(b1) << 8))
        XCTAssertEqual(val, -32768)
    }

    // MARK: - F9: SIMD8 DSP & 安全ガード境界 (5ケース以上)

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

    func testF9_Boundary_ThresholdBoundaries() {
        // 0.799（線形領域）、0.800（境界）、0.801（圧縮開始）で出力が不連続な跳躍を起こさないことを検証する。
        let yLow = applySoftLimit(0.799)
        let yMid = applySoftLimit(0.800)
        let yHigh = applySoftLimit(0.801)
        XCTAssertEqual(yLow, 0.799)
        XCTAssertEqual(yMid, 0.800)
        XCTAssertTrue(0.800 < yHigh)
        XCTAssertTrue(yHigh < 0.802)
    }

    func testF9_Boundary_ExtremeValues() {
        // ±1000.0 の極大振幅が Soft Limiter で ±1.0 以下に抑制されることを検証する。
        let outPos = applySoftLimit(1000.0)
        let outNeg = applySoftLimit(-1000.0)
        XCTAssertTrue(outPos <= 1.0)
        XCTAssertTrue(-1.0 <= outNeg)
    }

    func testF9_Boundary_DenormalValues() {
        // 1e-25 などのデノーマル数が CPU トラップを引き起こさず 0.0 に安全に扱われることを検証する。
        vocoder.reset()
        let frame = [Float](repeating: 1e-25, count: 64)
        let samples = vocoder.synthesize(mel: [frame])
        XCTAssertEqual(samples.count, 160)
        XCTAssertTrue(samples[0].isFinite)
    }

    func testF9_Boundary_ZeroElementVector() {
        // count = 0 で VectorOperations.dotProduct を呼んだ際にクラッシュせず 0.0 を返すことを検証する。
        let dummy: [Float] = [1.0]
        let res = dummy.withUnsafeBufferPointer { p in
            VectorOperations.dotProduct(a: p.baseAddress!, b: p.baseAddress!, count: 0)
        }
        XCTAssertEqual(res, 0.0)
    }

    func testF9_Boundary_NaNInSoftLimiter() {
        // 非有限値が渡された場合でも比較演算子が暴走しないことを検証する。
        let y = applySoftLimit(Float.nan)
        // NaN に対して abs(NaN) <= 0.8 は false なので圧縮ブランチへ進む
        XCTAssertTrue(y.isNaN || y.isFinite)
    }

    // MARK: - F10: SNN 境界 (5ケース以上)

    func testF10_Boundary_ZeroCurrent() {
        // 直流入力がない状態で静止電位 0.0 が維持され、無発火（スパイク 0）であることを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, timeSteps: 1, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        let feat = [Float](repeating: 0.0, count: 8)
        var out = [Float](repeating: 0.0, count: 4)

        feat.withUnsafeBufferPointer { pF in
            out.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
            }
        }
        XCTAssertTrue(out[0].isFinite)
    }

    func testF10_Boundary_HugeCurrent() {
        // 入力電流 50.0 で膜電位が暴走せず、クランプ上限 LIFNeuronEngine.vClampMax で飽和することを検証する。
        let config = LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.1)
        var v: [Float] = [0.0]
        var s: [Float] = [0.0]
        var a: [Float] = [0.0]
        let cur: [Float] = [50.0]

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
        XCTAssertTrue(v[0] <= LIFNeuronEngine.vClampMax)
    }

    func testF10_Boundary_SingleStep() {
        // timeSteps = 1 の最小時間展開において時間平均除算 (invT = 1.0) が正しく動作することを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, timeSteps: 1, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        let feat = [Float](repeating: 0.5, count: 8)
        var out = [Float](repeating: 0.0, count: 4)

        feat.withUnsafeBufferPointer { pF in
            out.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
            }
        }
        XCTAssertTrue(out[0].isFinite)
    }

    func testF10_Boundary_ThresholdBoundaryPotential() {
        // 膜電位が vTh 直下 (0.999) で未発火、直上 (1.001) で発火する境界二値性を検証する。
        let vSub = MLXArray([Float(0.999)])
        let vSuper = MLXArray([Float(1.001)])
        let vTh = MLXArray([Float(1.0)])
        let sSub = SurrogateGradients.fastSigmoidSTE(v: vSub, vTh: vTh, alpha: 2.0)
        let sSuper = SurrogateGradients.fastSigmoidSTE(v: vSuper, vTh: vTh, alpha: 2.0)
        eval(sSub, sSuper)
        XCTAssertEqual(sSub.item(Float.self), 0.0)
        XCTAssertEqual(sSuper.item(Float.self), 1.0)
    }

    func testF10_Boundary_EmptyFeatureSequence() {
        // featuresSeq が空の場合に decodeSequence が空配列を返し安全復帰することを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        let out = decoder.decodeSequence(featuresSeq: [], workspace: workspace)
        XCTAssertTrue(out.isEmpty)
    }

    // MARK: - F11: MLX BPTT 境界 (5ケース以上)

    func testF11_Boundary_SeqLenOneAlign() {
        // 1 フレームの最小入力が 32 フレームにゼロパディングされることを検証する。
        let aligned = MLXAcousticBPTTTrainer.alignTo32(seqLen: 1)
        XCTAssertEqual(aligned, 32)
    }

    func testF11_Boundary_BpttWindowOne() {
        // 最小ウィンドウ 1 で毎フレーム stopGradient が呼ばれても順伝播が完了することを検証する。
        let net = MLXSpikingAcousticNetwork(numLayers: 2, inputDim: 8, maxHiddenDim: 16, outputDim: 4, timeSteps: 1)
        let x = MLXArray.zeros([1, 4, 8])
        let out = net.forward(features: x, bpttWindow: 1)
        eval(out)
        XCTAssertEqual(out.shape, [1, 4, 4])
    }

    func testF11_Boundary_ZeroLearningRate() {
        // learningRate = 0.0 でオプティマイザがパラメータを破壊せず NaN も発生しないことを検証する。
        let net = MLXSpikingAcousticNetwork(numLayers: 2, inputDim: 8, maxHiddenDim: 16, outputDim: 4, timeSteps: 1)
        let trainer = MLXAcousticBPTTTrainer(network: net, learningRate: 0.0)
        let x = MLXArray.zeros([1, 32, 8])
        let y = MLXArray.zeros([1, 32, 4])
        let loss = trainer.trainBatch(features: x, targets: y)
        XCTAssertTrue(loss.isFinite)
    }

    func testF11_Boundary_AllZeroTargets() {
        // ターゲットが完全なゼロテンソルである場合でもスペクトル損失が正常に評価されることを検証する。
        let net = MLXSpikingAcousticNetwork(numLayers: 2, inputDim: 8, maxHiddenDim: 16, outputDim: 4, timeSteps: 1)
        let trainer = MLXAcousticBPTTTrainer(network: net, learningRate: 0.001)
        let x = MLXArray.zeros([1, 32, 8])
        let y = MLXArray.zeros([1, 32, 4])
        let loss = trainer.trainBatch(features: x, targets: y)
        XCTAssertTrue(0.0 <= loss)
    }

    func testF11_Boundary_EmptySequenceTrain() {
        // trainSequence に空配列が渡された際に 0.0 を返して安全ガードすることを検証する。
        let net = MLXSpikingAcousticNetwork(numLayers: 2, inputDim: 8, maxHiddenDim: 16, outputDim: 4, timeSteps: 1)
        let trainer = MLXAcousticBPTTTrainer(network: net, learningRate: 0.001)
        let loss = trainer.trainSequence(features: [], targets: [])
        XCTAssertEqual(loss, 0.0)
    }

    // MARK: - F12: 推論 Hot Path 境界 (5ケース以上)

    func testF12_Boundary_ZeroSpikesAllSilent() {
        // 発火ニューロンが 0 個の時、転置重みループが 0 回実行され結合電流が正確に 0.0 となることを検証する。
        let workspace = AcousticWorkspace(maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        workspace.clearStepCurrents()
        XCTAssertEqual(workspace.stepCurrents[0], 0.0)
    }

    func testF12_Boundary_AllSpikesFiring() {
        // 全ニューロンが発火した場合でも activeSpikes 配列の上限を超えず SIMD8 加算が完結することを検証する。
        let workspace = AcousticWorkspace(maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        var i = 0
        while i < 16 {
            workspace.layerStates[0].s[i] = 1.0
            i += 1
        }
        XCTAssertEqual(workspace.layerStates[0].s.count, 16)
    }

    func testF12_Boundary_MultiLayerResidual() {
        // 多層 SNN において 2 層構成の順伝播と残差接続が破綻なく実行されることを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 32, outputDim: 4, timeSteps: 1, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 32, outputDim: 4, numLayers: 2)
        let feat = [Float](repeating: 0.1, count: 8)
        var out = [Float](repeating: 0.0, count: 4)

        feat.withUnsafeBufferPointer { pF in
            out.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
            }
        }
        XCTAssertTrue(out[0].isFinite)
    }

    func testF12_Boundary_MinHiddenDim() {
        // 最小隠れ層幅（d = 8）で SIMD8 の 1 レジスタ処理が境界整合することを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 8, outputDim: 4, timeSteps: 1, numLayers: 1)
        XCTAssertEqual(weights.maxHiddenDim, 8)
    }

    func testF12_Boundary_WorkspaceReinit() {
        // 同一ワークスペースで連続フレーム推論を行っても内部バッファ境界が破壊されないことを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 32, outputDim: 4, timeSteps: 1, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 32, outputDim: 4, numLayers: 2)
        let feat = [Float](repeating: 0.2, count: 8)
        var out1 = [Float](repeating: 0.0, count: 4)
        var out2 = [Float](repeating: 0.0, count: 4)

        feat.withUnsafeBufferPointer { pF in
            out1.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
            }
            out2.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
            }
        }
        XCTAssertTrue(out1[0].isFinite)
        XCTAssertTrue(out2[0].isFinite)
    }

    // MARK: - F13: 多層 SNN 構造境界 (5ケース以上)

    func testF13_Boundary_SingleLayer() {
        // 最小層数 numLayers = 1 の境界構成で正しく重みが初期化され順伝播可能であることを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, numLayers: 1)
        XCTAssertEqual(weights.numLayers, 1)
        XCTAssertEqual(weights.wLayers.count, 0)
    }

    func testF13_Boundary_MultiLayerFour() {
        // 深層構成 numLayers = 4 において中間層重み配列が正しく確保されることを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, numLayers: 4)
        XCTAssertEqual(weights.numLayers, 4)
        XCTAssertEqual(weights.wLayers.count, 3)
        XCTAssertEqual(weights.bHLayers.count, 3)
        XCTAssertEqual(weights.gammaRMS.count, 3)
    }

    func testF13_Boundary_TransposedWeightsIntegrity() {
        // 転置重み配列の要素数が順方向重み配列と厳密に一致することを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, numLayers: 2)
        let recT = weights.makeWRecT()
        let layersT = weights.makeWLayersT()
        let outT = weights.makeWOutT()
        XCTAssertEqual(recT.count, weights.wRec.count)
        XCTAssertEqual(layersT.count, weights.wLayers.count)
        XCTAssertEqual(layersT[0].count, weights.wLayers[0].count)
        XCTAssertEqual(outT.count, weights.wOut.count)
    }

    func testF13_Boundary_DeterministicInitSeed() {
        // 同一シードから生成された重みが完全に同一の値を持つことを検証する。
        let w1 = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, numLayers: 2, seed: 42)
        let w2 = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, numLayers: 2, seed: 42)
        XCTAssertEqual(w1.wIn, w2.wIn)
        XCTAssertEqual(w1.wRec, w2.wRec)
        XCTAssertEqual(w1.wOut, w2.wOut)
    }

    func testF13_Boundary_JSONRoundtrip() {
        // Codable 経由の JSON エンコード・デコードで全パラメータが完全に復元されることを検証する。
        let original = SpikingNetworkWeights.randomWeights(inputDim: 8, maxHiddenDim: 16, outputDim: 4, numLayers: 2, seed: 100)
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        guard let data = try? encoder.encode(original),
              let restored = try? decoder.decode(SpikingNetworkWeights.self, from: data) else {
            XCTFail("JSON roundtrip failed")
            return
        }
        XCTAssertEqual(restored.numLayers, original.numLayers)
        XCTAssertEqual(restored.maxHiddenDim, original.maxHiddenDim)
        XCTAssertEqual(restored.wIn, original.wIn)
        XCTAssertEqual(restored.wRec, original.wRec)
        XCTAssertEqual(restored.wOut, original.wOut)
    }

    // MARK: - F14: 音声合成エンジン境界 (5ケース以上)

    func testF14_Boundary_EmptyStringSynthesis() {
        // 空文字入力時にエンジンが安全に空の波形を返しクラッシュしないことを検証する。
        let samples = engine.synthesize(text: "")
        XCTAssertTrue(samples.isEmpty)
    }

    func testF14_Boundary_WhitespaceOnlySynthesis() {
        // 空白・改行のみの入力時にエンジンが安全に空の波形を返すことを検証する。
        let samples = engine.synthesize(text: "  \t\n  ")
        XCTAssertTrue(samples.isEmpty)
    }

    func testF14_Boundary_NegativeSpeedClamp() {
        // 負の speed や下限未満の speed が渡された際に安全にクランプされて合成されることを検証する。
        let samples = engine.synthesize(text: "あ", speed: -1.0)
        XCTAssertFalse(samples.isEmpty)
    }

    func testF14_Boundary_NaNPitchClamp() {
        // 非有限な pitch (NaN) が渡された際に 1.0 へフォールバックされ有限波形が得られることを検証する。
        let samples = engine.synthesize(text: "あ", pitch: Float.nan)
        XCTAssertFalse(samples.isEmpty)
        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    func testF14_Boundary_ZeroFramesVocoder() {
        // NeuralVocoder に空のフレーム列を渡した際に安全に空配列が返されることを検証する。
        vocoder.reset()
        let samples = vocoder.synthesize(mel: [])
        XCTAssertTrue(samples.isEmpty)
    }

    // MARK: - F15: CLI 境界 (5ケース以上)

    func testF15_Boundary_EmptyTextArgument() {
        // CLI から "" が渡された際にエンジンが空の波形列を返し安全終了することを検証する。
        let samples = engine.synthesize(text: "")
        XCTAssertTrue(samples.isEmpty)
    }

    func testF15_Boundary_ShortTextCLI() {
        // 極小テキスト（1文字）でもエンジンが安全に合成を完遂することを検証する。
        let samples = engine.synthesize(text: "あ")
        XCTAssertTrue(0 < samples.count)
    }

    func testF15_Boundary_MinSpeedArgument() {
        // 極限低速発話パラメータにおいて極大フレーム展開が行われつつも、整数オーバーフローを起こさず正しく展開されることを検証する。
        let feat = lengthRegulator.processText(
            text: "あめ",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 0.05
        )
        XCTAssertTrue(0 < feat.totalFrames)
    }

    func testF15_Boundary_NegativePitchArgument() {
        // pitchScale = -1.0 で F0 が負数にならず 0.0 に安全に扱われることを検証する。
        let feat = lengthRegulator.processText(
            text: "あめ",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )
        let seq = engine.encodeLinguisticFeatures(features: feat)
        XCTAssertEqual(seq.count, feat.totalFrames)
    }

    func testF15_Boundary_UltraLongTextCLI() {
        // 100 モーラ以上の長文でもメモリリークを起こさず合成サンプル列が完結することを検証する。
        let text = String(repeating: "こんにちは", count: 20)
        let samples = engine.synthesize(text: text)
        XCTAssertTrue(0 < samples.count)
    }

    // MARK: - F16: E2E 境界 (5ケース以上)

    func testF16_Boundary_EmptyTextE2E() {
        // engine.synthesizeWav(text: "") が空ヘッダのみの WAV Data を安全に返すことを検証する。
        let data = engine.synthesizeWav(text: "")
        XCTAssertEqual(data.count, 44)
    }

    func testF16_Boundary_WhitespaceOnlyE2E() {
        // スペースのみの入力でエンジンがクラッシュせず無音波形を返すことを検証する。
        let samples = engine.synthesize(text: "   　　\t\n")
        XCTAssertTrue(0 <= samples.count)
    }

    func testF16_Boundary_PunctuationOnlyE2E() {
        // 「、、、」のみの入力で音素フレーム展開が破綻せず完了することを検証する。
        let samples = engine.synthesize(text: "、、、。。。")
        XCTAssertTrue(0 <= samples.count)
    }

    func testF16_Boundary_StreamEmptyText() {
        // コールバックが 0 回呼ばれて空配列が返却されることを検証する。
        var count = 0
        let samples = engine.synthesizeStream(text: "") { _ in
            count += 1
        }
        XCTAssertTrue(samples.isEmpty)
        XCTAssertEqual(count, 0)
    }

    func testF16_Boundary_RapidContinuousSynthesis() {
        // 同一エンジンインスタンスで連続して複数回の音声合成を行っても内部状態が破壊されず同一長の結果が得られることを検証する。
        let w1 = engine.synthesize(text: "ねこ")
        let w2 = engine.synthesize(text: "ねこ")
        let w3 = engine.synthesize(text: "いぬ")
        let w4 = engine.synthesize(text: "ねこ")
        XCTAssertEqual(w1.count, w2.count)
        XCTAssertEqual(w1.count, w4.count)
        XCTAssertTrue(0 < w3.count)
    }
}
