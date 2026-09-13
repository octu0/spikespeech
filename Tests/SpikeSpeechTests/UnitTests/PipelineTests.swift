import XCTest
import Foundation
@testable import SpikeSpeech

/// Milestone 4: Synthetic Data, CLI & Pipeline 統合単体テストスイート
///
/// 基準音声生成器の音響物理特性、
/// および SpikeSpeechEngine による日本語テキストから WAV 生成までの
/// E2E パイプラインの一貫性・数値健全性を自動検証する。
final class PipelineTests: XCTestCase {

    override func setUp() {
        super.setUp()
    }

    override func tearDown() {
        super.tearDown()
    }

    // MARK: - 1. SyntheticAudioGenerator 単体検証

    /// 5母音ホルマント合成の音響特性検証
    func testFiveVowelFormantSynthesis() {
        // 各母音が NaN/Inf のない安定した振幅を有し、クリッピングせず、
        // 異なる母音間で固有のホルマント波形が生成されていることを確認する。
        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let duration: Float = 0.2 // 200ms = 3,200 サンプル
        let vowels: [SyntheticAudioGenerator.Vowel] = [.a, .i, .u, .e, .o]

        var vowelWaves: [[Float]] = []

        var vIdx = 0
        while vIdx < vowels.count {
            let v = vowels[vIdx]
            let wave = generator.generateVowel(vowel: v, durationSeconds: duration, f0: 130.0)

            // サンプル数の一致確認 (3,200 サンプル)
            let expectedSamples = Int(round(duration * 16000.0))
            XCTAssertEqual(wave.count, expectedSamples, "母音 \(v.rawValue) のサンプル数が不一致です")

            // NaN / Inf およびクリッピング検査
            var maxAbs: Float = 0.0
            var sumSq: Float = 0.0
            var i = 0
            while i < wave.count {
                let s = wave[i]
                XCTAssertFalse(s.isNaN, "母音 \(v.rawValue) のサンプル \(i) で NaN を検出")
                XCTAssertFalse(s.isInfinite, "母音 \(v.rawValue) のサンプル \(i) で Inf を検出")
                let absVal = abs(s)
                if maxAbs < absVal {
                    maxAbs = absVal
                }
                sumSq += s * s
                i += 1
            }

            // 最大振幅は 1.0 以下（ソフトサチュレーションによるクリッピング防止）
            XCTAssertTrue(maxAbs <= 1.0, "母音 \(v.rawValue) で振幅クリッピングが発生: maxAbs=\(maxAbs)")
            // 有意な音響エネルギーの存在
            let rms = sqrt(sumSq / Float(wave.count))
            XCTAssertTrue(0.005 < rms, "母音 \(v.rawValue) の音響エネルギーが小さすぎます: rms=\(rms)")

            vowelWaves.append(wave)
            vIdx += 1
        }

        // 異なる母音パラメータが同一の定数波形に縮退していないことを証明する。
        let waveA = vowelWaves[0]
        let waveI = vowelWaves[1]
        var diffSum: Float = 0.0
        var i = 0
        while i < waveA.count {
            diffSum += abs(waveA[i] - waveI[i])
            i += 1
        }
        let meanDiff = diffSum / Float(waveA.count)
        XCTAssertTrue(0.01 < meanDiff, "母音 /a/ と /i/ の波形が同一です: diff=\(meanDiff)")
    }

    /// 20Hz〜8000Hz チャープ波の周波数単調増加検証
    func testChirpWaveformGeneration() {
        // 先頭部（低周波）と終端部（高周波）のゼロ交差率 (ZCR) を比較し、
        // 瞬時周波数が時間とともに連続的に増大していることを数学的に証明する。
        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let duration: Float = 0.4
        let chirp = generator.generateChirp(startFreq: 20.0, endFreq: 8000.0, durationSeconds: duration)

        let totalSamples = chirp.count
        XCTAssertEqual(totalSamples, Int(round(duration * 16000.0)))

        // 先頭 20% と 末尾 20% のゼロ交差率 (ZCR) を算出
        let segLen = totalSamples / 5

        func computeZCR(samples: [Float], start: Int, length: Int) -> Int {
            var zcr = 0
            var n = 1
            while n < length {
                let idx1 = start + n - 1
                let idx2 = start + n
                let s1 = samples[idx1]
                let s2 = samples[idx2]
                if (s1 * s2) < 0.0 {
                    zcr += 1
                }
                n += 1
            }
            return zcr
        }

        let zcrHead = computeZCR(samples: chirp, start: 0, length: segLen)
        let zcrTail = computeZCR(samples: chirp, start: totalSamples - segLen, length: segLen)

        // チャープ波の高周波末尾は低周波先頭より厳密に多くのゼロ交差を持つ
        XCTAssertTrue(zcrHead < zcrTail, "チャープ波の周波数が上昇していません: head=\(zcrHead), tail=\(zcrTail)")
    }

    /// インパルス列 (Dirac Comb) のピッチ周期性検証
    func testImpulseTrainGeneration() {
        // F0 = 100Hz において、サンプル間隔が 16000 / 100 = 160 サンプルに
        // 厳密に一致することを検証する。
        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let f0: Float = 100.0
        let duration: Float = 0.1
        let train = generator.generateImpulseTrain(f0: f0, durationSeconds: duration)

        let expectedPeriod = 160 // 16000 / 100
        var pulseIndices: [Int] = []

        var i = 0
        while i < train.count {
            if 0.5 < train[i] {
                pulseIndices.append(i)
            }
            i += 1
        }

        XCTAssertTrue(2 <= pulseIndices.count, "十分なパルスが検出されませんでした")
        var pIdx = 1
        while pIdx < pulseIndices.count {
            let interval = pulseIndices[pIdx] - pulseIndices[pIdx - 1]
            XCTAssertEqual(interval, expectedPeriod, "インパルス列の周期が理論値と一致しません")
            pIdx += 1
        }
    }

    /// ホワイトノイズの統計的性質検証
    func testWhiteNoiseGeneration() {
        // 直流オフセットがなくゼロ近傍を中心に対称に分布し、
        // 設定振幅に応じたエネルギーを保持していることを確認する。
        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let noise = generator.generateWhiteNoise(durationSeconds: 0.2, amplitude: 0.5)

        var sum: Float = 0.0
        var sumSq: Float = 0.0
        var i = 0
        while i < noise.count {
            let s = noise[i]
            sum += s
            sumSq += s * s
            i += 1
        }

        let meanVal = sum / Float(noise.count)
        let rmsVal = sqrt(sumSq / Float(noise.count))

        // 平均値がゼロ近傍 (|mean| < 0.05)
        XCTAssertTrue(abs(meanVal) < 0.05, "ホワイトノイズの直流成分が過大です: \(meanVal)")
        // 一様乱数 [-A, A] の理論 RMS は A / sqrt(3) ~= 0.5 / 1.732 = 0.288
        XCTAssertTrue(0.15 < rmsVal, "ノイズエネルギーが小さすぎます: \(rmsVal)")
        XCTAssertTrue(rmsVal < 0.45, "ノイズエネルギーが過大です: \(rmsVal)")
    }

    /// 擬似日本語フレーズ合成および基準コーパス生成検証
    func testToySpeechCorpusGeneration() {
        // 5つの代表フレーズが全て非空の波形を生成し、
        // 音素数に応じた妥当な時間長とエネルギーを持っていることを保証する。
        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let corpus = generator.generateStandardCorpus()

        XCTAssertEqual(corpus.count, 5, "基準コーパスの文数が 5 ではありません")

        var cIdx = 0
        while cIdx < corpus.count {
            let item = corpus[cIdx]
            XCTAssertFalse(item.text.isEmpty, "コーパステキストが空です")
            XCTAssertTrue(0 < item.samples.count, "コーパス波形が空です: \(item.text)")

            // NaN / Inf の完全不在確認
            var s = 0
            while s < item.samples.count {
                let v = item.samples[s]
                XCTAssertFalse(v.isNaN, "\(item.text) のサンプル \(s) で NaN を検出")
                XCTAssertFalse(v.isInfinite, "\(item.text) のサンプル \(s) で Inf を検出")
                s += 1
            }
            cIdx += 1
        }
    }

    // MARK: - 2. SpikeSpeechEngine E2E パイプライン単体・結合検証

    /// E2E 日本語テキスト音声合成 (Text -> 16kHz 16-bit WAV) の完全性検証
    func testEndToEndSpeechSynthesis() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは、世界。"

        let wavData = engine.synthesizeWav(text: text, speed: 1.0, pitch: 1.0)

        // 44 バイトヘッダ以上のサイズ
        XCTAssertTrue(44 < wavData.count, "WAV データサイズがヘッダ長以下です: \(wavData.count)")

        // ヘッダシグネチャの完全検証
        wavData.withUnsafeBytes { rawBytes in
            let ptr = rawBytes.bindMemory(to: UInt8.self)
            // "RIFF"
            XCTAssertEqual(ptr[0], 0x52)
            XCTAssertEqual(ptr[1], 0x49)
            XCTAssertEqual(ptr[2], 0x46)
            XCTAssertEqual(ptr[3], 0x46)
            // "WAVE"
            XCTAssertEqual(ptr[8], 0x57)
            XCTAssertEqual(ptr[9], 0x41)
            XCTAssertEqual(ptr[10], 0x56)
            XCTAssertEqual(ptr[11], 0x45)
            // "fmt "
            XCTAssertEqual(ptr[12], 0x66)
            XCTAssertEqual(ptr[13], 0x6D)
            XCTAssertEqual(ptr[14], 0x74)
            XCTAssertEqual(ptr[15], 0x20)
            // "data"
            XCTAssertEqual(ptr[36], 0x64)
            XCTAssertEqual(ptr[37], 0x61)
            XCTAssertEqual(ptr[38], 0x74)
            XCTAssertEqual(ptr[39], 0x61)
        }

        // PCM サンプル列の直接取得と数値検証
        let pcmSamples = engine.synthesize(text: text)
        XCTAssertTrue(0 < pcmSamples.count)

        var sIdx = 0
        while sIdx < pcmSamples.count {
            let val = pcmSamples[sIdx]
            XCTAssertFalse(val.isNaN, "E2E 合成波形で NaN を検出: index=\(sIdx)")
            XCTAssertFalse(val.isInfinite, "E2E 合成波形で Inf を検出: index=\(sIdx)")
            XCTAssertTrue(abs(val) <= 1.0, "E2E 合成波形でクリッピングを検出: val=\(val)")
            sIdx += 1
        }
    }

    /// 多層 SNN 構成での音声合成の検証
    func testMultilayerSpeechSynthesis() {
        let text = "すぱいくすぴーち"
        let layerConfigs = [1, 2, 3]

        var s = 0
        while s < layerConfigs.count {
            let layers = layerConfigs[s]
            let weights = SpikingNetworkWeights.randomWeights(numLayers: layers)
            let engine = SpikeSpeechEngine(weights: weights)
            let wav = engine.synthesizeWav(text: text)
            XCTAssertTrue(44 < wav.count, "層数 \(layers) で WAV データが生成されませんでした")
            s += 1
        }
    }

    /// 話速およびピッチ制御の検証
    func testSpeedAndPitchModulation() {
        let engine = SpikeSpeechEngine()
        let text = "あいうえお"

        let fastSamples = engine.synthesize(text: text, speed: 1.5, pitch: 1.0)
        let slowSamples = engine.synthesize(text: text, speed: 0.8, pitch: 1.0)

        // 高速発話は低速発話よりサンプル数が厳密に少ない
        XCTAssertTrue(fastSamples.count < slowSamples.count, "話速変更によるサンプル数の短縮が機能していません: fast=\(fastSamples.count), slow=\(slowSamples.count)")

        // ピッチ変更時の合成検証
        let highPitchSamples = engine.synthesize(text: text, speed: 1.0, pitch: 1.4)
        XCTAssertTrue(0 < highPitchSamples.count)
    }

    /// 境界値入力時の安全フォールバック検証
    func testEdgeCaseEmptyTextSynthesis() {
        let engine = SpikeSpeechEngine()

        let emptyWav = engine.synthesizeWav(text: "")
        XCTAssertEqual(emptyWav.count, 44, "空テキスト入力時に 44 バイトヘッダが返却されませんでした")

        let spaceWav = engine.synthesizeWav(text: "   ")
        XCTAssertTrue(44 <= spaceWav.count)
    }

    /// ストリーミング音声合成の逐次検証
    func testStreamingSynthesis() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは"

        var callbackCount = 0
        var receivedSamplesCount = 0

        let totalSamples = engine.synthesizeStream(
            text: text,
            speed: 1.0,
            pitch: 1.0
        ) { frameSamples in
            XCTAssertEqual(frameSamples.count, AudioConfig.hopSize, "フレームサンプル数が 160 ではありません")
            callbackCount += 1
            receivedSamplesCount += frameSamples.count
        }

        XCTAssertTrue(0 < callbackCount, "コールバックが一度も呼ばれませんでした")
        XCTAssertEqual(receivedSamplesCount, totalSamples.count, "ストリーミング総サンプル数が一致しません")
    }

    // MARK: - 3. Milestone 4 静的コーディング規約機械検査

    /// Milestone 4 実装ファイルのコーディング規約完全遵守検証
    func testMilestone4StaticRuleCheck() {
        // M4 で新規作成された全ファイルについて、比較演算子 `<` と `<=` のみ、
        // `else if` 禁止、三項演算子禁止などのコーディング規約完全遵守を自動検証する。
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath

        let targetPaths = [
            currentDir + "/Sources/SpikeSpeech/DSP/SyntheticAudioGenerator.swift",
            currentDir + "/Sources/SpikeSpeech/Pipeline/SpikeSpeechEngine.swift",
            currentDir + "/script/synthesize/main.swift",
            currentDir + "/script/train/main.swift",
            currentDir + "/script/benchmark/main.swift"
        ]

        let gtSym = " " + ">" + " "
        let gteSym = " " + ">=" + " "
        let elseIfSym = "else" + " " + "if"
        let ternarySym = " " + "?" + " "

        var checkedCount = 0
        var fIdx = 0
        while fIdx < targetPaths.count {
            let path = targetPaths[fIdx]
            guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
                fIdx += 1
                continue
            }

            let lines = content.components(separatedBy: .newlines)
            var lineIdx = 0
            while lineIdx < lines.count {
                let line = lines[lineIdx]
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                // コメント行のスキップ
                if trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") {
                    lineIdx += 1
                    continue
                }

                XCTAssertFalse(
                    trimmed.contains(gtSym) && trimmed.contains("->") != true,
                    "規約違反 (大なり記号) が使用されています: \(path):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(gteSym),
                    "規約違反 (大なりイコール記号) が使用されています: \(path):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(elseIfSym),
                    "規約違反 (else-if) が使用されています: \(path):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(ternarySym) && trimmed.contains("??") != true,
                    "規約違反 (三項演算子) が使用されています: \(path):\(lineIdx + 1): \(trimmed)"
                )
                lineIdx += 1
            }

            checkedCount += 1
            fIdx += 1
        }

        XCTAssertEqual(checkedCount, targetPaths.count, "走査対象ファイル数が不一致です")
        print("--- [Milestone 4 Static Rule Check] ---")
        print("検証完了ファイル数: \(checkedCount) 件 (全ファイル規約適合)")
        print("---------------------------------------")
    }

    // MARK: - 4. 速度境界値・NaN堅牢性およびサチュレーション抑制テスト

    /// speed == 0.0、負数、NaN/Inf、極端な速度指定時のクラッシュ根絶検証
    func testSpeedZeroAndExtremeValuesRobustness() {
        // 速度 0.0 や非有限値が入力された際にゼロ除算や浮動小数点キャスト例外による
        // 異常終了を防止し、安全域 [0.1, 10.0] へ適切にクランプされることを検証する。
        let regulator = LengthRegulator()
        let testSpeeds: [Float] = [0.0, -1.0, -100.0, Float.nan, Float.infinity, -Float.infinity, 0.0001, 100.0]

        var idx = 0
        while idx < testSpeeds.count {
            let spd = testSpeeds[idx]

            // 1. floatDurationFrames の計算
            let floatDur = regulator.floatDurationFrames(category: .vowel, symbol: "a", speed: spd)
            XCTAssertFalse(floatDur.isNaN, "speed=\(spd) で floatDurationFrames が NaN を返しました")
            XCTAssertFalse(floatDur.isInfinite, "speed=\(spd) で floatDurationFrames が Inf を返しました")
            XCTAssertTrue(1.0 <= floatDur, "speed=\(spd) で floatDurationFrames が 1.0 未満です")

            // 2. defaultDurationFrames の計算 (Int キャスト安全性の確認)
            let intDur = regulator.defaultDurationFrames(category: .vowel, symbol: "a", speed: spd)
            XCTAssertTrue(1 <= intDur, "speed=\(spd) で defaultDurationFrames が 1 未満です")

            idx += 1
        }

        // 3. 異常な durations 配列に対する quantizeDurations の安全性
        let abnormalDurations: [Float] = [Float.nan, Float.infinity, -5.0, 0.0, 10.0]
        let quantized = regulator.quantizeDurations(durations: abnormalDurations)
        XCTAssertEqual(quantized.count, abnormalDurations.count)
        var q = 0
        while q < quantized.count {
            XCTAssertTrue(1 <= quantized[q], "quantizeDurations で 1 未満のフレーム数が生成されました")
            q += 1
        }

        // 4. SpikeSpeechEngine E2E での speed == 0.0 安全合成
        let engine = SpikeSpeechEngine()
        let wavData = engine.synthesizeWav(text: "テスト", speed: 0.0)
        XCTAssertTrue(44 < wavData.count, "speed == 0.0 で有効な WAV データが生成されませんでした")

        let samples = engine.synthesize(text: "テスト", speed: 0.0)
        XCTAssertTrue(0 < samples.count, "speed == 0.0 で出力サンプルが空です")
    }

    /// WAV 出力振幅の適正化およびクリッピングサチュレーション抑制検証
    func testWavHeadroomAndNoSaturationClipping() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは、世界。スパイクスピーチの合成音声テストです。"

        let samples = engine.synthesize(text: text)
        XCTAssertTrue(0 < samples.count, "合成サンプルが空です")

        var maxAbs: Float = 0.0
        var sumSq: Float = 0.0
        var i = 0
        while i < samples.count {
            let s = samples[i]
            let a = abs(s)
            if maxAbs < a {
                maxAbs = a
            }
            sumSq += s * s
            i += 1
        }

        let rms = sqrt(sumSq / Float(samples.count))

        // 最大絶対値振幅がヘッドルーム 0.8801 以下であること（1.0 への張り付き皆無）
        XCTAssertTrue(maxAbs <= 0.8801, "ヘッドルームを超過するサンプル振幅を検出: maxAbs=\(maxAbs)")
        // 有意な音響エネルギーが存在すること
        XCTAssertTrue(0.1 < rms, "音響エネルギーが小さすぎます: rms=\(rms)")

        // WAV バイナリ変換後の Int16 範囲検証
        let wavData = engine.synthesizeWav(text: text)
        let sampleCount = (wavData.count - 44) / 2
        var clippingCount = 0
        wavData.withUnsafeBytes { rawBytes in
            let basePtr = rawBytes.baseAddress!.advanced(by: 44).assumingMemoryBound(to: Int16.self)
            var sIdx = 0
            while sIdx < sampleCount {
                let val = basePtr[sIdx]
                if val == 32767 || val == -32768 {
                    clippingCount += 1
                }
                sIdx += 1
            }
        }

        // サチュレーション（最大振幅張り付き）が完全にゼロであること
        XCTAssertEqual(clippingCount, 0, "WAV サンプルでクリッピングサチュレーションが検出されました: \(clippingCount)")
    }
}
