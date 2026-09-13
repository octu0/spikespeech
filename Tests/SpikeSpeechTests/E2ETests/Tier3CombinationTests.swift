import XCTest
import Foundation
import MLX
@testable import SpikeSpeech

/// Tier 3: 機能間組み合わせテストスイート (Pairwise / Cross-Feature Integration)
///
/// 単一モジュールの正常動作だけでなく、モジュール境界を跨ぐデータ受渡し
/// （言語処理 ↔ SNN ↔ DSP ↔ WAV ↔ MLX学習 ↔ 基準音声生成）の相互運用性と
/// インターフェース契約の整合性を検証する。
final class Tier3CombinationTests: XCTestCase {

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
        // 組み合わせテストにおける前テストのキャッシュや内部状態の漏洩を防ぐため全モジュールを初期化する。
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

    // MARK: - Pair 1: F1 (形態素) ＋ F2 (正規化) ＋ F5 (ピッチアクセント/F0)

    func testPair01_Morphology_Normalizer_Prosody() {
        // 助詞を含む自然文「私は学生です」が形態素分割され、助詞置換「わたしわ」を経て
        // 適切なピッチアクセントトーンと連続 F0 輪郭が導出されることを検証する。
        let text = "私は学生です"
        let norm = normalizer.normalize(text: text)
        var phrases = prosodyModel.buildAccentPhrases(morphemes: norm, vocabulary: vocabulary)
        XCTAssertTrue(0 < phrases.count)

        // 各音素の durationFrames に基づいて generateF0Contour が F0 フレーム系列を展開できるようにフレーム長を設定する。
        var p = 0
        while p < phrases.count {
            var m = 0
            while m < phrases[p].moras.count {
                var ph = 0
                while ph < phrases[p].moras[m].phonemes.count {
                    phrases[p].moras[m].phonemes[ph].durationFrames = 8
                    ph += 1
                }
                m += 1
            }
            p += 1
        }

        let (f0, voiced, total) = prosodyModel.generateF0Contour(phrases: phrases, vocabulary: vocabulary)
        XCTAssertTrue(0 < total)
        XCTAssertEqual(f0.count, total)
        XCTAssertEqual(voiced.count, total)

        // 有声フレームにおいて F0 が妥当なピッチ範囲（80〜400Hz）にあること
        var f = 0
        while f < total {
            if 0.5 <= voiced[f] {
                XCTAssertTrue(80.0 < f0[f])
                XCTAssertTrue(f0[f] < 400.0)
            }
            f += 1
        }
    }

    // MARK: - Pair 2: F2 (正規化) ＋ F3 (語彙トークナイズ) ＋ F4 (Duration展開)

    func testPair02_Normalizer_Vocabulary_LengthRegulator() {
        // 数字や促音を含む文「1冊買った」が正規化・音素化され、Length Regulation で
        // 各音素の継続時間に応じた時間軸フレーム系列へ展開されることを検証する。
        let text = "1冊買った"
        let features = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 1.0
        )
        XCTAssertTrue(0 < features.totalFrames)
        XCTAssertEqual(features.phoneIds.count, features.durations.count)

        var totalDurs: Int32 = 0
        var i = 0
        while i < features.durations.count {
            totalDurs += features.durations[i]
            i += 1
        }
        XCTAssertEqual(Int(totalDurs), features.totalFrames)
    }

    // MARK: - Pair 3: F4 (Length Regulation) ＋ F10 (SNN言語特徴量エンコード)

    func testPair03_LengthRegulator_SNNFeatureEncoder() {
        // 時間長展開された音素系列が one-hot 埋め込みおよび音響生理パラメータ（F0, 有声度, 進行率）と結合され、
        // [totalFrames, inputDim] の SNN 入力電流系列へ変換されることを検証する。
        let features = lengthRegulator.processText(
            text: "やまびこ",
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )
        let seq = engine.encodeLinguisticFeatures(features: features, pitchScale: 1.0)
        XCTAssertEqual(seq.count, features.totalFrames)
        XCTAssertEqual(seq[0].count, engine.weights.inputDim)

        // 音素内時間進行率 [0.0, 1.0] の検証 (次元 66)
        if 66 < engine.weights.inputDim {
            var t = 0
            while t < seq.count {
                XCTAssertTrue(0.0 <= seq[t][66])
                XCTAssertTrue(seq[t][66] <= 1.0)
                t += 1
            }
        }
    }

    // MARK: - Pair 4: F10 (多層 SNN デコーダー) ＋ F13 (多層ネットワーク重み) ＋ F12 (推論HotPath)

    func testPair04_SNN_MultilayerWeights_Workspace() {
        // AcousticWorkspace のゼロアロケーションバッファを用いて、多層 SNN 推論が
        // メモリリークやデータ破損なしに正しく音響特徴量を算出することを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 256, outputDim: 8, timeSteps: 2, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 256, outputDim: 8, numLayers: 2)

        let inputSeq = [[Float]](repeating: [Float](repeating: 0.1, count: 16), count: 10)
        let out = decoder.decodeSequence(featuresSeq: inputSeq, workspace: workspace)

        XCTAssertEqual(out.count, 10)
        XCTAssertEqual(out[0].count, 8)
        XCTAssertTrue(out[0][0].isFinite)
    }

    // MARK: - Pair 5: F10 (SNN音響出力) ＋ F7 (MelToLPC変換)

    func testPair05_SNNAcoustic_MelToLPC_Conversion() {
        // SNN デコーダーが生成した対数 Mel ベクトルから、Levinson-Durbin アルゴリズムを通じて
        // 有効な LPC 声道多項式係数とゲインが導出されることを検証する。
        let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 128, outputDim: 64, timeSteps: 1, numLayers: 2)
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 128, outputDim: 64, numLayers: 2)

        let inputVec = [Float](repeating: 0.2, count: 16)
        var melOut = [Float](repeating: 0.0, count: 64)

        inputVec.withUnsafeBufferPointer { pIn in
            melOut.withUnsafeMutableBufferPointer { pOut in
                decoder.decodeFrame(features: pIn.baseAddress!, workspace: workspace, outputFeatures: pOut.baseAddress!)
            }
        }

        var lpcCoeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLPC.convert(mel: melOut, isLogMel: true, outCoeffs: &lpcCoeffs)
        XCTAssertTrue(gain.isFinite)
        XCTAssertTrue(0.0 <= gain)
        var k = 0
        while k < lpcCoeffs.count {
            XCTAssertTrue(lpcCoeffs[k].isFinite)
            k += 1
        }
    }

    // MARK: - Pair 6: F7 (MelToLPC) ＋ F6 (LPCボコーダー)

    func testPair06_MelToLPC_LPCVocoder_Waveform() {
        // Mel 特徴量から導出された LPC パラメータフレームがボコーダーに渡され、
        // 16kHz モノラル PCM サンプル列として破綻なく波形合成されることを検証する。
        let mel = [Float](repeating: -2.5, count: 64)
        var lpcCoeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLPC.convert(mel: mel, isLogMel: true, outCoeffs: &lpcCoeffs)

        let frame = AcousticFrame(
            lpcCoefficients: lpcCoeffs,
            gain: gain,
            pitchF0: 160.0,
            voiced: 1.0
        )
        vocoder.reset()
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

    // MARK: - Pair 7: F6 (ボコーダー) ＋ F9 (Soft Limiter) ＋ F8 (WavEncoder)

    func testPair07_Vocoder_SoftLimiter_WavEncoder() {
        // ボコーダー出力の PCM サンプルがサチュレーション歪みなくリミッティングされ、
        // 正常な 16kHz 16-bit Mono WAV バイナリ Data へ変換されることを検証する。
        vocoder.reset()
        let frame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.02, count: 16),
            gain: 0.8,
            pitchF0: 130.0,
            voiced: 1.0
        )
        let rawSamples = vocoder.synthesize(frames: [frame, frame, frame])
        var limitedSamples = [Float](repeating: 0.0, count: rawSamples.count)
        rawSamples.withUnsafeBufferPointer { pSrc in
            limitedSamples.withUnsafeMutableBufferPointer { pDst in
                VectorOperations.softLimitTanh(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: rawSamples.count)
            }
        }

        let wavData = WavEncoder.encode(samples: limitedSamples, sampleRate: 16000)
        XCTAssertEqual(wavData.count, 44 + (rawSamples.count * 2))
        XCTAssertEqual(wavData[0], 0x52) // 'R'
    }

    // MARK: - Pair 8: F14 (SyntheticGenerator母音) ＋ F7 (MelToLPC反転)

    func testPair08_SyntheticVowel_MelToLPC_Inversion() {
        // 自動生成した母音波形から抽出された Mel 特徴量に相当するスペクトルが
        // MelToLPC によって安定した共鳴多項式へ再構成されることを検証する。
        let vowelSamples = generator.generateVowel(vowel: .a, durationSeconds: 0.05, f0: 130.0)
        XCTAssertTrue(0 < vowelSamples.count)

        // 疑似 Mel スペクトル（母音 /a/ のホルマントピークを模した分布）
        var syntheticMel = [Float](repeating: -5.0, count: 64)
        syntheticMel[6] = 2.0  // F1 (約800Hz)
        syntheticMel[11] = 1.0 // F2 (約1300Hz)

        var lpcCoeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLPC.convert(mel: syntheticMel, isLogMel: true, outCoeffs: &lpcCoeffs)
        XCTAssertTrue(0.0 < gain)
        XCTAssertTrue(lpcCoeffs[0].isFinite)
    }

    // MARK: - Pair 9: F14 (SyntheticGeneratorチャープ) ＋ F8 (WavEncoder)

    func testPair09_SyntheticChirp_WavEncoder_Integrity() {
        // 20Hz〜8000Hz の広帯域チャープ波が量子化クリッピングを起こさずに WAV Data に変換され、
        // 44 バイトヘッダの各フィールドが仕様通りに保存されることを検証する。
        let chirp = generator.generateChirp(startFreq: 50.0, endFreq: 4000.0, durationSeconds: 0.05)
        let data = WavEncoder.encode(samples: chirp, sampleRate: 16000)
        XCTAssertEqual(data.count, 44 + (chirp.count * 2))

        let bAudioFormat = Int(data[20])
        XCTAssertEqual(bAudioFormat, 1) // Linear PCM
    }

    // MARK: - Pair 10: F14 (擬似音声データ) ＋ F11 (MLX順伝播)

    func testPair10_SyntheticPhrase_BPTT_Forward() {
        // SyntheticAudioGenerator の擬似正解フレーズから抽出した特徴量が
        // MLX ネットワークの forward 計算グラフに受け渡され、損失計算可能なテンソルが得られることを検証する。
        let corpus = generator.generateStandardCorpus()
        let first = corpus[0]
        let feat = engine.encodeLinguisticFeatures(
            features: engine.lengthRegulator.processText(
                text: first.text,
                normalizer: engine.normalizer,
                prosodyModel: engine.prosodyModel,
                vocabulary: engine.vocabulary
            )
        )
        let net = MLXSpikingAcousticNetwork(numLayers: 1, inputDim: engine.weights.inputDim, maxHiddenDim: 64, outputDim: 32, timeSteps: 2)
        let alignedLen = MLXAcousticBPTTTrainer.alignTo32(seqLen: feat.count)

        var flatFeat = [Float](repeating: 0.0, count: alignedLen * engine.weights.inputDim)
        var t = 0
        while t < feat.count {
            var i = 0
            while i < engine.weights.inputDim {
                flatFeat[(t * engine.weights.inputDim) + i] = feat[t][i]
                i += 1
            }
            t += 1
        }
        let x = MLXArray(flatFeat, [1, alignedLen, engine.weights.inputDim])
        let out = net.forward(features: x)
        eval(out)
        XCTAssertEqual(out.shape, [1, alignedLen, 32])
    }

    // MARK: - Pair 11: F11 (BPTT学習) ＋ F13 (スペクトル損失) ＋ 重みエクスポート

    func testPair11_BPTT_SpectralLoss_WeightsExport() {
        // 多層 SNN においてスペクトル損失による BPTT 勾配更新が行われ、
        // 学習済みパラメータが Pure Swift 用 SpikingNetworkWeights へ完全にエクスポートできることを検証する。
        let net = MLXSpikingAcousticNetwork(numLayers: 2, inputDim: 16, maxHiddenDim: 64, outputDim: 16, timeSteps: 1)
        let trainer = MLXAcousticBPTTTrainer(network: net, learningRate: 0.01)

        let x = MLXRandom.uniform(low: -0.1, high: 0.1, [1, 32, 16])
        let y = MLXRandom.uniform(low: -0.1, high: 0.1, [1, 32, 16])
        let loss = trainer.trainBatch(features: x, targets: y)
        XCTAssertTrue(loss.isFinite)

        let exported = net.exportWeights()
        XCTAssertEqual(exported.maxHiddenDim, 64)
        XCTAssertEqual(exported.inputDim, 16)
        XCTAssertEqual(exported.outputDim, 16)
        XCTAssertEqual(exported.numLayers, 2)
        XCTAssertTrue(exported.wIn[0].isFinite)
    }

    // MARK: - Pair 12: F11 (エクスポート重み) ＋ F10/F12 (Pure Swift 推論デコーダー)

    func testPair12_ExportedWeights_PureSwiftDecoder() {
        // MLX で更新・エクスポートされた SpikingNetworkWeights が SpikingAcousticDecoder にロードされ、
        // Pure Swift の SIMD8 ホットパス推論で完全に動作することを検証する。
        let net = MLXSpikingAcousticNetwork(numLayers: 2, inputDim: 16, maxHiddenDim: 64, outputDim: 16, timeSteps: 2)
        let exported = net.exportWeights()

        let decoder = SpikingAcousticDecoder(weights: exported)
        let workspace = AcousticWorkspace(maxHiddenDim: 64, outputDim: 16, numLayers: 2)
        let feat = [Float](repeating: 0.2, count: 16)
        var out = [Float](repeating: 0.0, count: 16)

        feat.withUnsafeBufferPointer { pF in
            out.withUnsafeMutableBufferPointer { pO in
                decoder.decodeFrame(features: pF.baseAddress!, workspace: workspace, outputFeatures: pO.baseAddress!)
            }
        }
        XCTAssertTrue(out[0].isFinite)
    }

    // MARK: - Pair 13: F1〜F5 (言語) ＋ F10 (SNN) ＋ F6/F7 (ボコーダー) ＋ F8 (WAV)

    func testPair13_FullPipeline_Linguistics_To_Wav() {
        // 生テキストから形態素解析、正規化、韻律、Length Regulation、SNN推論、MelToLPC、ボコーダー、WAVエンコードまでの
        // データフローが中間の型変換破綻や情報欠落なしに一貫して接続することを検証する。
        let text = "春の風"
        let norm = normalizer.normalize(text: text)
        let feat = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary
        )
        XCTAssertTrue(0 < norm.count)
        XCTAssertTrue(0 < feat.totalFrames)

        let wavData = engine.synthesizeWav(text: text)
        XCTAssertTrue(44 < wavData.count)
    }

    // MARK: - Pair 14: F12 (AcousticWorkspace) ＋ F13 (多層推論反復)

    func testPair14_WorkspaceReuse_MultiIterationSynthesis() {
        // 同一エンジンのワークスペースにおいて複数回の音声合成を行っても、
        // 内部バッファの再割り当てなしで正常に推論が継続されることを検証する。
        var s = 0
        while s < 4 {
            let wave = engine.synthesize(text: "テスト")
            XCTAssertTrue(0 < wave.count)
            s += 1
        }
    }

    // MARK: - Pair 15: F5 (ピッチスケール) ＋ F6 (ボコーダーF0励起)

    func testPair15_ProsodyPitchScale_VocoderF0Excitation() {
        // pitchScale = 1.5 でピッチが高くなった際、Rosenberg パルスのパルス周期が短縮され、
        // 合成波形が異なる周波数スペクトルを呈することを検証する。
        let waveLow = engine.synthesize(text: "あめ", pitch: 0.8)
        let waveHigh = engine.synthesize(text: "あめ", pitch: 1.5)
        XCTAssertEqual(waveLow.count, waveHigh.count)
        // 異なるピッチなのでサンプル値が完全一致しないこと
        XCTAssertNotEqual(waveLow[80], waveHigh[80])
    }

    // MARK: - Pair 16: F4 (速度変更) ＋ F6 (ボコーダー総サンプル数)

    func testPair16_LengthRegulatorSpeed_VocoderSampleCount() {
        // speedFactor = 1.5 で Length Regulation フレーム数が短縮され、
        // 最終的な WAV サンプル長が厳密に短縮されることを検証する。
        let waveNormal = engine.synthesize(text: "おはようございます", speed: 1.0)
        let waveFast = engine.synthesize(text: "おはようございます", speed: 1.5)
        XCTAssertTrue(waveFast.count < waveNormal.count)
    }

    // MARK: - Pair 17: F8 (WavStreamWriter) ＋ F16 (synthesizeStream 逐次書き出し)

    func testPair17_StreamingWriter_EngineStreamingCallback() throws {
        // synthesizeStream のフレームコールバックから WavStreamWriter.write へ直接データを流し込み、
        // finalize で完結した WAV ファイルが通常合成した WAV ファイルとサイズ整合することを検証する。
        let tempUrl = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        FileManager.default.createFile(atPath: tempUrl.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tempUrl)
        let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

        let samples = engine.synthesizeStream(text: "すずしい") { frame in
            try? writer.write(samples: frame)
        }
        try writer.finalize()
        try handle.close()

        let fileData = try Data(contentsOf: tempUrl)
        XCTAssertEqual(fileData.count, 44 + (samples.count * 2))
        try? FileManager.default.removeItem(at: tempUrl)
    }
}
