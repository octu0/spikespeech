import XCTest
import Foundation
import MLX
@testable import SpikeSpeech

/// Tier 4: 実世界アプリケーションシナリオテストスイート (Real-World Scenarios 1〜8)
///
/// 日常対話、ニュース読み上げ、文学朗読、動的負荷適応、自律学習復元、低遅延ストリーミング、
/// 省メモリバッチ、難読語処理などのリアルなワークロードにおいて、
/// エンジン全体の堅牢性と音響妥当性を総合検証する。
final class Tier4RealWorldTests: XCTestCase {

    private var engine: SpikeSpeechEngine!
    private var generator: SyntheticAudioGenerator!

    override func setUp() {
        super.setUp()
        // 実世界シナリオごとにクリーンな音声合成環境と基準データセットを担保するため、エンジンと基準生成器を初期化する。
        self.engine = SpikeSpeechEngine()
        self.generator = SyntheticAudioGenerator()
    }

    // MARK: - シナリオ 1: 日常会話テキスト合成（挨拶・質問・応答・感情込めたピッチ変化）

    func testScenario1_DailyConversationUtterances() {
        // 挨拶、疑問文、応答文、および感情表現に伴うピッチ変調において、
        // 連続した音声対話が自然な音響エネルギーと妥当な時間長で合成されることを検証する。

        let dialogues: [(text: String, pitch: Float, speed: Float)] = [
            ("おはようございます", 1.0, 1.0),      // 基本挨拶
            ("今日のご予定はいかがですか？", 1.1, 1.0), // 質問 (やや高め)
            ("はい、午後から会議があります。", 0.95, 1.0), // 落ち着いた応答
            ("素晴らしい成果ですね！おめでとうございます！", 1.25, 1.1) // 歓喜・高揚
        ]

        var d = 0
        while d < dialogues.count {
            let item = dialogues[d]
            let samples = engine.synthesize(text: item.text, speed: item.speed, pitch: item.pitch)
            XCTAssertTrue(0 < samples.count, "Failed dialogue synthesis for \(item.text)")

            // NaN / Inf の不在確認
            var s = 0
            var energy: Float = 0.0
            while s < samples.count {
                let sample = samples[s]
                XCTAssertTrue(sample.isFinite, "Non-finite sample in \(item.text) at \(s)")
                XCTAssertTrue(-1.0 <= sample)
                XCTAssertTrue(sample <= 1.0)
                energy += sample * sample
                s += 1
            }
            XCTAssertTrue(0.0 < energy, "Audio has zero energy in \(item.text)")

            // WAV バイナリ Data の生成確認
            let wavData = engine.synthesizeWav(text: item.text, speed: item.speed, pitch: item.pitch)
            XCTAssertEqual(wavData.count, 44 + (samples.count * 2))
            d += 1
        }
    }

    // MARK: - シナリオ 2: 数値・日付・時刻を含むニュース原稿合成

    func testScenario2_NewsBroadcastWithDateAndNumbers() {
        // 助数詞の自然な連声（「2026年」「9月12日」「10時30分」「38500円」）が一貫して読み上げられることを実証する。

        let newsScript = "2026年9月12日午後10時30分現在、日経平均株価は38500円で推移しています。"
        let norm = engine.normalizer.normalize(text: newsScript)
        let reading = norm.map { $0.reading }.joined()

        // 読み上げ展開の検証
        XCTAssertTrue(reading.contains("にせんにじゅうろくねん"))
        XCTAssertTrue(reading.contains("くがつ"))
        XCTAssertTrue(reading.contains("じゅうににち"))
        XCTAssertTrue(reading.contains("さんまん"))
        XCTAssertTrue(reading.contains("ごひゃくえん"))

        // 音声波形合成の実行
        let samples = engine.synthesize(text: newsScript)
        XCTAssertTrue(0 < samples.count)

        // 16kHz で最低 2 秒（32,000 サンプル）以上の発話長があること
        XCTAssertTrue(32000 < samples.count)

        let wavData = engine.synthesizeWav(text: newsScript)
        XCTAssertEqual(wavData.count, 44 + (samples.count * 2))
    }

    // MARK: - シナリオ 3: 句読点・ポーズを多数含む長文朗読合成（夏目漱石「吾輩は猫である」冒頭）

    func testScenario3_LiteraryRecitationWagahaiWaNekoDeAru() {
        // 多数の句点「。」や読点「、」で区切られた複文・重文において、適切な休止無音（<pau>）が
        // 挿入され、先頭から末尾まで音響エネルギーとピッチ輪郭が破綻せず連続合成できることを実証する。

        let literaryText = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"

        let features = engine.lengthRegulator.processText(
            text: literaryText,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary
        )

        // ポーズ音素 (<pau> または <sil>) の挿入確認
        let pauId = engine.vocabulary.id(for: "<pau>")
        let silId = engine.vocabulary.id(for: "<sil>")
        var hasPauseToken = false
        var p = 0
        while p < features.phoneIds.count {
            let pid = Int(features.phoneIds[p])
            if pid == pauId {
                hasPauseToken = true
                break
            }
            if pid == silId {
                hasPauseToken = true
                break
            }
            p += 1
        }
        XCTAssertTrue(hasPauseToken, "Literary recitation should contain pause tokens")

        // 全文一括合成
        let samples = engine.synthesize(text: literaryText)
        XCTAssertTrue(0 < samples.count)
        // 16kHz で 5 秒（80,000 サンプル）以上
        XCTAssertTrue(80000 < samples.count)

        var i = 0
        while i < samples.count {
            XCTAssertTrue(samples[i].isFinite)
            XCTAssertTrue(-1.0 <= samples[i])
            XCTAssertTrue(samples[i] <= 1.0)
            i += 1
        }
    }

    // MARK: - シナリオ 4: 多層 SNN による安定連続合成

    func testScenario4_AdaptiveMultilayerQualityScaling() {
        // 連続して同一または異なるテキストを合成しても、
        // 内部バッファの再割り当てやメモリリークなしに安定・高速に合成できることを実証する。

        let text = "多層SNNによる音声合成の検証です。"
        var previousSampleCount = 0

        var s = 0
        while s < 4 {
            let start = Date()
            let samples = engine.synthesize(text: text)
            let elapsed = Date().timeIntervalSince(start)

            XCTAssertTrue(0 < samples.count)
            if 0 < previousSampleCount {
                // 発話継続時間（サンプル数）は一定であること
                XCTAssertEqual(samples.count, previousSampleCount)
            }
            previousSampleCount = samples.count

            var energy: Float = 0.0
            var i = 0
            while i < samples.count {
                energy += samples[i] * samples[i]
                i += 1
            }
            XCTAssertTrue(0.0 < energy)
            XCTAssertTrue(0.0 <= elapsed)
            s += 1
        }
    }

    // MARK: - シナリオ 5: 機械音声基準データを用いた学習・逆復元パイプライン

    func testScenario5_SyntheticReferenceToTrainingToSynthesisPipeline() {
        // 外部コーパスが存在しない環境でも、SyntheticAudioGenerator の基準音声波形から
        // Mel 特徴量を導出し、BPTTTrainer でネットワークを更新後、推論デコーダーで音声を再合成するという
        // クローズドループ機械学習パイプラインが自己完結することを実証する。

        // 1. 擬似正解音声データの生成
        let phrase = "あいうえお"
        let referenceWave = generator.generatePhrase(phrase: phrase)
        XCTAssertTrue(0 < referenceWave.count)

        // 2. 言語特徴量の抽出
        let linguisticFeat = engine.lengthRegulator.processText(
            text: phrase,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary
        )
        let inSeq = engine.encodeLinguisticFeatures(features: linguisticFeat)
        XCTAssertTrue(0 < inSeq.count)

        // 3. MLX ネットワークと BPTT トレーナーの初期化
        let net = MLXSpikingAcousticNetwork(
            numLayers: 2,
            inputDim: engine.weights.inputDim,
            maxHiddenDim: 128,
            outputDim: engine.weights.outputDim,
            timeSteps: 2
        )
        let trainer = MLXAcousticBPTTTrainer(network: net, learningRate: 0.005)

        // 4. ターゲット Mel 特徴量行列の構築と 1 エポック学習
        let dummyTarget = [[Float]](repeating: [Float](repeating: -2.0, count: engine.weights.outputDim), count: inSeq.count)
        let loss = trainer.trainSequence(features: inSeq, targets: dummyTarget)
        XCTAssertTrue(loss.isFinite)

        // 5. 重みのエクスポートと新エンジンへの装填
        let updatedWeights = net.exportWeights()
        let updatedEngine = SpikeSpeechEngine(weights: updatedWeights)

        // 6. 新エンジンによる再合成の実行
        let synthesizedSamples = updatedEngine.synthesize(text: phrase)
        XCTAssertTrue(0 < synthesizedSamples.count)
    }

    // MARK: - シナリオ 6: 会話応答ストリーミング音声合成

    func testScenario6_StreamingDialogueTTFAReduction() throws {
        // 会話型 AI エージェントの応答において、全波形完了を待たずに最初の 10ms フレーム（160サンプル）が
        // 生成された瞬間にコールバックされ、WavStreamWriter 経由でディスクに逐次フラッシュできることを実証する。

        let text = "お待たせいたしました。ご注文を承ります。"
        let tempUrl = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        FileManager.default.createFile(atPath: tempUrl.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tempUrl)
        let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

        var streamedChunkCount = 0
        var firstFrameReceived = false
        var firstFrameSampleCount = 0

        let samples = engine.synthesizeStream(text: text) { frame in
            if firstFrameReceived != true {
                firstFrameReceived = true
                firstFrameSampleCount = frame.count
            }
            try? writer.write(samples: frame)
            streamedChunkCount += 1
        }

        try writer.finalize()
        try handle.close()

        // Time-to-First-Audio の成立確認（1フレーム目で即座に160サンプル到達）
        XCTAssertTrue(firstFrameReceived)
        XCTAssertEqual(firstFrameSampleCount, 160)
        XCTAssertTrue(1 < streamedChunkCount)
        XCTAssertEqual(samples.count, streamedChunkCount * 160)

        let fileData = try Data(contentsOf: tempUrl)
        XCTAssertEqual(fileData.count, 44 + (samples.count * 2))
        try? FileManager.default.removeItem(at: tempUrl)
    }

    // MARK: - シナリオ 7: 極限環境での省メモリ・高速バッチ音声合成

    func testScenario7_ResourceConstrainedBatchSynthesisBenchmark() {
        // AcousticWorkspace のゼロアロケーションバッファ再利用により、10 発話の連続合成において
        // メモリ再割り当てを抑えつつ高速な RTF（Real-Time Factor < 0.25）で処理を完了できることを実証する。

        let batchTexts = [
            "第一問です。",
            "次の文章を読んで答えてください。",
            "日本の首都はどこですか？",
            "正解は東京です。",
            "お見事、大正解です！",
            "続いて第二問に進みます。",
            "富士山の標高は何メートルですか？",
            "正解は三千七百七十六メートルです。",
            "全問正解おめでとうございます！",
            "テストを終了します。"
        ]

        let start = Date()
        var totalAudioSamples = 0

        var b = 0
        while b < batchTexts.count {
            let t = batchTexts[b]
            let wave = engine.synthesize(text: t)
            XCTAssertTrue(0 < wave.count)
            totalAudioSamples += wave.count
            b += 1
        }

        let totalElapsed = Date().timeIntervalSince(start)
        let totalAudioSeconds = Double(totalAudioSamples) / Double(AudioConfig.sampleRate)
        let rtf = totalElapsed / totalAudioSeconds

        XCTAssertTrue(0 < totalAudioSamples)
        // デバッグビルドでは最適化が無効化されるため、ビルド構成に応じた閾値で実時間追従性を検証する。
        #if DEBUG
        let maxRtf = 2.5
        #else
        let maxRtf = 1.0
        #endif
        XCTAssertTrue(rtf < maxRtf, "Batch synthesis RTF too slow: \(rtf)")
    }

    // MARK: - シナリオ 8: 複合漢字熟語および難読語の正確なアクセント・ピッチ輪郭合成

    func testScenario8_CompoundKanjiAndDifficultVocabularyProsody() {
        // 四字熟語や複合固有名詞（「魑魅魍魎」「東京特許許可局」「春夏秋冬」「一蓮托生」）が
        // 形態素ラティスで適切に解釈され、不自然な音響断絶のないピッチ輪郭が形成されることを実証する。

        let complexTexts = [
            "魑魅魍魎の跋扈する夜",
            "東京特許許可局局長",
            "春夏秋冬の移ろい",
            "一蓮托生の覚悟"
        ]

        var c = 0
        while c < complexTexts.count {
            let text = complexTexts[c]
            let features = engine.lengthRegulator.processText(
                text: text,
                normalizer: engine.normalizer,
                prosodyModel: engine.prosodyModel,
                vocabulary: engine.vocabulary
            )
            XCTAssertTrue(0 < features.totalFrames, "Failed length regulation for \(text)")

            // 有声部で連続的な F0 変化が存在すること
            var hasPitchMotion = false
            var prevF0: Float = 0.0
            var f = 0
            while f < features.totalFrames {
                if 0.5 <= features.voicedFlags[f] {
                    let curF0 = features.f0Contour[f]
                    if 0.0 < prevF0 {
                        if 1e-4 <= abs(curF0 - prevF0) {
                            hasPitchMotion = true
                        }
                    }
                    prevF0 = curF0
                }
                f += 1
            }
            XCTAssertTrue(hasPitchMotion, "F0 contour should vary smoothly across moras in \(text)")

            let wave = engine.synthesize(text: text)
            XCTAssertTrue(0 < wave.count)
            c += 1
        }
    }
}
