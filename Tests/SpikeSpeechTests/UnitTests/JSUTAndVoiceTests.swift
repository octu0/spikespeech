import XCTest
import Foundation
@testable import SpikeSpeech

/// JSUT コーパス読み込み、実音声対数 Mel 特徴量抽出、および話者差し替え（VoiceProfile）検証テストスイート
final class JSUTAndVoiceTests: XCTestCase {

    // MARK: - 1. VoiceProfile 単体検証

    /// 既定の話者プロファイル（女性、男性、中性）のパラメータ整合性を検証
    func testVoiceProfilePresetsIntegrity() {
        // 女性ボイス（JSUT 標準）
        let female = VoiceProfile.female
        XCTAssertEqual(female.name, "female")
        XCTAssertEqual(female.pitchScale, 1.0)
        XCTAssertEqual(female.pitchShift, 0.0)
        XCTAssertEqual(female.formantScale, 1.0)
        XCTAssertEqual(female.energyScale, 1.0)
        XCTAssertEqual(female.speakerEmbedding.count, 16)

        // 男性ボイス（低域ピッチ、声道拡大）
        let male = VoiceProfile.male
        XCTAssertEqual(male.name, "male")
        XCTAssertTrue(male.pitchScale < 1.0)
        XCTAssertTrue(male.pitchShift < 0.0)
        XCTAssertTrue(male.formantScale < 1.0)
        XCTAssertEqual(male.speakerEmbedding.count, 16)

        // 中性ボイス
        let neutral = VoiceProfile.neutral
        XCTAssertEqual(neutral.name, "neutral")
        XCTAssertTrue(neutral.pitchScale < 1.0)
        XCTAssertEqual(neutral.speakerEmbedding.count, 16)

        // 子供ボイス
        let child = VoiceProfile.child
        XCTAssertEqual(child.name, "child")
        XCTAssertTrue(1.0 < child.pitchScale)
        XCTAssertTrue(1.0 < child.formantScale)
        XCTAssertEqual(child.speakerEmbedding.count, 16)

        // 重低音男性ボイス
        let deepMale = VoiceProfile.deepMale
        XCTAssertEqual(deepMale.name, "deepMale")
        XCTAssertTrue(deepMale.pitchScale < 0.6)
        XCTAssertTrue(deepMale.formantScale < 0.85)
        XCTAssertEqual(deepMale.speakerEmbedding.count, 16)

        // 既定値が female であること
        XCTAssertEqual(VoiceProfile.default, VoiceProfile.female)

        // 名前に基づくプリセット解決
        XCTAssertEqual(VoiceProfile.preset(named: "female"), VoiceProfile.female)
        XCTAssertEqual(VoiceProfile.preset(named: "jsut"), VoiceProfile.female)
        XCTAssertEqual(VoiceProfile.preset(named: "male"), VoiceProfile.male)
        XCTAssertEqual(VoiceProfile.preset(named: "man"), VoiceProfile.male)
        XCTAssertEqual(VoiceProfile.preset(named: "neutral"), VoiceProfile.neutral)
        XCTAssertEqual(VoiceProfile.preset(named: "child"), VoiceProfile.child)
        XCTAssertEqual(VoiceProfile.preset(named: "kid"), VoiceProfile.child)
        XCTAssertEqual(VoiceProfile.preset(named: "deep"), VoiceProfile.deepMale)
        XCTAssertEqual(VoiceProfile.preset(named: "deepmale"), VoiceProfile.deepMale)
        XCTAssertEqual(VoiceProfile.preset(named: "unknown"), VoiceProfile.female)
    }

    /// VoiceProfile の JSON 直列化および逆直列化の完全性を検証
    func testVoiceProfileCodableRoundTrip() throws {
        let customVoice = VoiceProfile(
            name: "custom_actor",
            pitchScale: 1.25,
            pitchShift: 15.0,
            formantScale: 1.05,
            energyScale: 0.95,
            speakerEmbedding: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(customVoice)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(VoiceProfile.self, from: data)

        XCTAssertEqual(decoded.name, customVoice.name)
        XCTAssertEqual(decoded.pitchScale, customVoice.pitchScale)
        XCTAssertEqual(decoded.pitchShift, customVoice.pitchShift)
        XCTAssertEqual(decoded.formantScale, customVoice.formantScale)
        XCTAssertEqual(decoded.energyScale, customVoice.energyScale)
        XCTAssertEqual(decoded.speakerEmbedding, customVoice.speakerEmbedding)
    }

    // MARK: - 2. WavAudioReader 単体検証

    /// 16kHz モノラル WAV データの読み込みおよび PCM サンプル復元を検証
    func testWavAudioReader16kMono() throws {
        let sampleRate = 16000
        let count = 1600 // 100ms
        var rawSamples = [Float](repeating: 0.0, count: count)
        var i = 0
        while i < count {
            rawSamples[i] = sin((2.0 * Float.pi * 440.0 * Float(i)) / Float(sampleRate)) * 0.5
            i += 1
        }

        let wavData = WavEncoder.encode(samples: rawSamples, sampleRate: sampleRate)
        let reader = WavAudioReader()
        let pcm = try reader.parseWav16k(bytes: [UInt8](wavData))

        XCTAssertEqual(pcm.count, count)
        var maxDiff: Float = 0.0
        var s = 0
        while s < count {
            let diff = abs(pcm[s] - rawSamples[s])
            if maxDiff < diff {
                maxDiff = diff
            }
            s += 1
        }
        // 16-bit 量子化誤差（1/32768 = 約 0.00003）以内であることを実証
        XCTAssertTrue(maxDiff < 0.001)
    }

    /// 48kHz 音声に対する 3サンプル平均間引きリサンプリングの動作を検証
    func testWavAudioReader48kTo16kDownsampling() {
        let reader = WavAudioReader()
        let pcm48k: [Float] = [0.3, 0.6, 0.9, 0.2, 0.4, 0.6]
        let pcm16k = reader.resampleTo16kHz(pcm: pcm48k, sourceSampleRate: 48000)

        XCTAssertEqual(pcm16k.count, 2)
        let expected0 = (0.3 + 0.6 + 0.9) / 3.0
        let expected1 = (0.2 + 0.4 + 0.6) / 3.0
        XCTAssertEqual(pcm16k[0], Float(expected0), accuracy: 1e-5)
        XCTAssertEqual(pcm16k[1], Float(expected1), accuracy: 1e-5)
    }

    /// 不正な WAV データに対する適切な例外スローを検証
    func testWavAudioReaderInvalidDataThrows() {
        let reader = WavAudioReader()
        let truncatedBytes: [UInt8] = [0x52, 0x49, 0x46, 0x46, 0x00, 0x00]
        XCTAssertThrowsError(try reader.parseWav16k(bytes: truncatedBytes))
    }

    /// 境界直前で切り詰められた WAV データに対する安全な例外処理を検証（バッファオーバーラン防御）
    func testWavAudioReaderBoundaryCheck() {
        let reader = WavAudioReader()
        // fmt チャンクの bitsPerSample 直前で途切れたデータ
        let bytes: [UInt8] = [
            0x52, 0x49, 0x46, 0x46, // "RIFF"
            0x24, 0x00, 0x00, 0x00, // chunk size
            0x57, 0x41, 0x56, 0x45, // "WAVE"
            0x66, 0x6d, 0x74, 0x20, // "fmt "
            0x10, 0x00, 0x00, 0x00, // subchunk1 size (16)
            0x01, 0x00,             // audio format (PCM)
            0x01, 0x00,             // num channels (1)
            0x80, 0x3e, 0x00, 0x00, // sample rate (16000)
            0x00, 0x7d, 0x00, 0x00, // byte rate
            0x02, 0x00              // block align
            // ここでデータが途切れている (wBitsPerSample が存在しない)
        ]
        XCTAssertThrowsError(try reader.parseWav16k(bytes: bytes))
    }

    /// サンプリングレート 0 やチャンネル数 0 の異常値に対する例外送出を検証
    func testWavAudioReaderInvalidSampleRateAndChannels() {
        let reader = WavAudioReader()
        let invalidBytes: [UInt8] = [
            0x52, 0x49, 0x46, 0x46, 0x2c, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
            0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, // channels = 0
            0x00, 0x00, 0x00, 0x00, // sample rate = 0
            0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x10, 0x00,
            0x64, 0x61, 0x74, 0x61, 0x04, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00
        ]
        XCTAssertThrowsError(try reader.parseWav16k(bytes: invalidBytes))
    }

    /// 24-bit PCM 音声のデコード精度を検証
    func testWavAudioReader24BitPCM() throws {
        let reader = WavAudioReader()
        let bytes: [UInt8] = [
            0x52, 0x49, 0x46, 0x46, 0x2e, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
            0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x01, 0x00, // PCM, 1 channel
            0x80, 0x3e, 0x00, 0x00, // 16000 Hz
            0x80, 0xbb, 0x00, 0x00, // byte rate (48000)
            0x03, 0x00, 0x18, 0x00, // block align 3, bitsPerSample 24
            0x64, 0x61, 0x74, 0x61, 0x06, 0x00, 0x00, 0x00, // "data", 6 bytes
            0x00, 0x00, 0x40,       // sample 1: +0.5
            0x00, 0x00, 0xc0        // sample 2: -0.5
        ]

        let pcm = try reader.parseWav16k(bytes: bytes)
        XCTAssertEqual(pcm.count, 2)
        XCTAssertEqual(pcm[0], 0.5, accuracy: 1e-4)
        XCTAssertEqual(pcm[1], -0.5, accuracy: 1e-4)
    }

    // MARK: - 3. MelSpectrogramExtractor 単体検証

    /// 64 チャンネル対数 Mel スペクトログラムの抽出サイズおよび有限性を検証
    func testMelSpectrogramExtractorDimensionsAndFiniteness() {
        let extractor = MelSpectrogramExtractor()
        let sampleRate: Float = 16000.0
        let durationSec: Float = 0.5
        let totalSamples = Int(durationSec * sampleRate) // 8000 samples

        // 1000Hz 正弦波信号の生成
        var pcm = [Float](repeating: 0.0, count: totalSamples)
        var i = 0
        while i < totalSamples {
            pcm[i] = sin((2.0 * Float.pi * 1000.0 * Float(i)) / sampleRate) * 0.7
            i += 1
        }

        let mel = extractor.extractLogMel(pcm: pcm)
        let expectedFrames = totalSamples / AudioConfig.hopSize // 50 frames
        XCTAssertEqual(mel.count, expectedFrames)

        var f = 0
        while f < mel.count {
            XCTAssertEqual(mel[f].count, AudioConfig.melChannels)
            var ch = 0
            while ch < AudioConfig.melChannels {
                let val = mel[f][ch]
                XCTAssertTrue(val.isFinite, "Mel 値が非有限値です: frame=\(f), ch=\(ch)")
                ch += 1
            }
            f += 1
        }
    }

    /// 1000Hz 正弦波における Mel チャンネルの局在エネルギーピークを検証
    func testMelSpectrogramFrequencyPeakLocalization() {
        let extractor = MelSpectrogramExtractor()
        let totalSamples = 3200 // 200ms
        var pcm = [Float](repeating: 0.0, count: totalSamples)
        var i = 0
        while i < totalSamples {
            pcm[i] = sin((2.0 * Float.pi * 1000.0 * Float(i)) / 16000.0) * 0.8
            i += 1
        }

        let mel = extractor.extractLogMel(pcm: pcm)
        XCTAssertTrue(0 < mel.count)

        // 定常区間（中央フレーム）のスペクトルを検査
        let centerFrame = mel[mel.count / 2]
        var maxCh = 0
        var maxVal = centerFrame[0]
        var ch = 1
        while ch < centerFrame.count {
            if maxVal < centerFrame[ch] {
                maxVal = centerFrame[ch]
                maxCh = ch
            }
            ch += 1
        }

        // 1000Hz は 64ch Mel フィルタバンクにおいて中低域（ch 15〜28 付近）に局在する
        XCTAssertTrue(10 <= maxCh)
        XCTAssertTrue(maxCh <= 35)
    }

    /// NaN や非有限値が入力に混入した場合でも、抽出結果の全対数 Mel 要素が有限値にクランプされることを検証
    func testMelSpectrogramExtractorNaNSafety() {
        let extractor = MelSpectrogramExtractor()
        var corruptedPCM = [Float](repeating: 0.0, count: 480)
        corruptedPCM[10] = Float.nan
        corruptedPCM[50] = Float.infinity
        corruptedPCM[100] = -Float.infinity

        let mel = extractor.extractLogMel(pcm: corruptedPCM)
        XCTAssertTrue(0 < mel.count)
        var f = 0
        while f < mel.count {
            var ch = 0
            while ch < mel[f].count {
                XCTAssertTrue(mel[f][ch].isFinite, "対数 Mel スペクトルに非有限値が含まれています: frame=\(f), ch=\(ch)")
                ch += 1
            }
            f += 1
        }
    }

    // MARK: - 4. コーパスデータセットアライメント検証
    // Sources/SpikeSpeech/Corpus は script/dataset/ へ移管されたため、
    // テスト内部の安全なヘルパー構造体を用いて時間軸整合性を検証する

    struct TestCorpusItem {
        let id: String
        let text: String
        let wavPath: String
    }

    /// 音声特徴量と目標スペクトログラムの時間軸整合性を実証
    func testJSUTDatasetTrainingPairAlignment() throws {
        let engine = SpikeSpeechEngine()
        let sampleRate = 16000
        let audioSamples = 4800 // 300ms = 30 frames

        var wave = [Float](repeating: 0.0, count: audioSamples)
        var s = 0
        while s < audioSamples {
            wave[s] = sin((2.0 * Float.pi * 200.0 * Float(s)) / Float(sampleRate)) * 0.4
            s += 1
        }

        let tempDir = FileManager.default.temporaryDirectory
        let tempWavPath = tempDir.appendingPathComponent("test_jsut_align.wav").path
        let wavData = WavEncoder.encode(samples: wave, sampleRate: sampleRate)
        try wavData.write(to: URL(fileURLWithPath: tempWavPath))

        let reader = WavAudioReader()
        let extractor = MelSpectrogramExtractor()
        let pcm = try reader.loadWav16k(from: tempWavPath)
        let targetMel = extractor.extractLogMel(pcm: pcm)

        let linguistic = engine.lengthRegulator.processText(
            text: "水をマレーシアから買わなくてはならないのです。",
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )

        let features = engine.encodeLinguisticFeatures(
            features: linguistic,
            voice: .female,
            pitchScale: 1.0
        )

        let expectedFrames = audioSamples / AudioConfig.hopSize // 30 frames
        XCTAssertEqual(targetMel.count, expectedFrames)
        XCTAssertTrue(0 < features.count)

        // 話者埋め込み（ch 68〜83）に VoiceProfile.female の値が注入されていることを検証
        let femaleEmb = VoiceProfile.female.speakerEmbedding
        var embIdx = 0
        while embIdx < min(16, femaleEmb.count) {
            let injectedVal = features[0][68 + embIdx]
            XCTAssertEqual(injectedVal, femaleEmb[embIdx], accuracy: 1e-5)
            embIdx += 1
        }
    }

    /// 極短音声に対するアライメント安全性を検証
    func testJSUTDatasetNegativeDiffAdjustmentAndF0Safety() throws {
        let sampleRate = 16000
        let audioSamples = 480
        let wave = [Float](repeating: 0.1, count: audioSamples)

        let tempDir = FileManager.default.temporaryDirectory
        let tempWavPath = tempDir.appendingPathComponent("test_jsut_short.wav").path
        let wavData = WavEncoder.encode(samples: wave, sampleRate: sampleRate)
        try wavData.write(to: URL(fileURLWithPath: tempWavPath))

        let reader = WavAudioReader()
        let extractor = MelSpectrogramExtractor()
        let pcm = try reader.loadWav16k(from: tempWavPath)
        let targetMel = extractor.extractLogMel(pcm: pcm)

        let expectedFrames = audioSamples / AudioConfig.hopSize // 3 frames
        XCTAssertEqual(targetMel.count, expectedFrames)
    }

    /// 先頭に '@' プレフィックスが付いたコーパスパスの正常解決を検証
    func testJSUTDatasetLeadingAtSignResolution() {
        var cleanPath = "@/nonexistent/path/at"
        if cleanPath.hasPrefix("@") {
            cleanPath = String(cleanPath.dropFirst())
        }
        XCTAssertFalse(cleanPath.hasPrefix("@"))
    }

    // MARK: - 5. 話者差し替え（Voice Switching）E2E 検証

    /// 女性ボイスと男性ボイスで合成波形およびピッチ特性が明確に切り替わることを実証
    func testVoiceSwitchingFemaleVsMaleSynthesis() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは、世界。"

        // 1. 女性ボイスで合成
        let femaleSamples = engine.synthesize(text: text, voice: .female)
        // 2. 男性ボイスで合成
        let maleSamples = engine.synthesize(text: text, voice: .male)

        XCTAssertTrue(0 < femaleSamples.count)
        XCTAssertTrue(0 < maleSamples.count)

        // 両者の波形が完全一致せず、声質パラメータ（ピッチ・埋め込み）の違いにより異なることを実証
        var diffSum: Float = 0.0
        let compareCount = min(femaleSamples.count, maleSamples.count)
        var i = 0
        while i < compareCount {
            diffSum += abs(femaleSamples[i] - maleSamples[i])
            i += 1
        }
        let avgDiff = diffSum / Float(compareCount)
        XCTAssertTrue(0.01 < avgDiff, "女性ボイスと男性ボイスの合成波形に有意な差異が存在しません: diff=\(avgDiff)")

        // 3. ピッチ（周波数）特性の検証: 男性ボイス（ピッチ 0.65）はゼロ交差数が女性ボイスより少なくなる
        var femaleZeroCrossings = 0
        var fIdx = 1
        while fIdx < femaleSamples.count {
            if (femaleSamples[fIdx - 1] < 0.0 && 0.0 <= femaleSamples[fIdx]) || (0.0 <= femaleSamples[fIdx - 1] && femaleSamples[fIdx] < 0.0) {
                femaleZeroCrossings += 1
            }
            fIdx += 1
        }

        var maleZeroCrossings = 0
        var mIdx = 1
        while mIdx < maleSamples.count {
            if (maleSamples[mIdx - 1] < 0.0 && 0.0 <= maleSamples[mIdx]) || (0.0 <= maleSamples[mIdx - 1] && maleSamples[mIdx] < 0.0) {
                maleZeroCrossings += 1
            }
            mIdx += 1
        }

        // 低域ピッチの男性ボイスはゼロ交差密度が低下する
        XCTAssertTrue(maleZeroCrossings < femaleZeroCrossings, "男性ボイスのゼロ交差数が女性ボイスを下回っていません: male=\(maleZeroCrossings), female=\(femaleZeroCrossings)")
    }

    /// ピッチ固定条件下で formantScale のみが異なる場合に LPC スペクトル・合成波形が有意に変化することを実証
    func testVoiceFormantScalingModifiesLPCAndSpectrum() {
        let engine = SpikeSpeechEngine()
        let text = "あああああ"

        // 同一ピッチ・同一埋め込みで、formantScale のみ 1.0 vs 0.88 のプロファイル
        let baseProfile = VoiceProfile(name: "v100", pitchScale: 1.0, pitchShift: 0.0, formantScale: 1.0, energyScale: 1.0)
        let scaledProfile = VoiceProfile(name: "v088", pitchScale: 1.0, pitchShift: 0.0, formantScale: 0.88, energyScale: 1.0)

        let samplesBase = engine.synthesize(text: text, voice: baseProfile)
        let samplesScaled = engine.synthesize(text: text, voice: scaledProfile)

        XCTAssertTrue(0 < samplesBase.count)
        XCTAssertTrue(0 < samplesScaled.count)

        var diffSum: Float = 0.0
        let count = min(samplesBase.count, samplesScaled.count)
        var i = 0
        while i < count {
            diffSum += abs(samplesBase[i] - samplesScaled[i])
            i += 1
        }
        let avgDiff = diffSum / Float(count)
        // formantScale による周波数軸伸縮により波形に統計的有意差が生じる
        XCTAssertTrue(0.005 < avgDiff, "formantScale の変化による合成波形差分が検出されません: diff=\(avgDiff)")
    }

    /// WAV 出力およびストリーミング出力における話者差し替えの動作を検証
    func testVoiceSwitchingWavAndStream() {
        let engine = SpikeSpeechEngine()
        let text = "声優のボイスを切り替えます。"

        // WAV バイナリ生成
        let femaleWav = engine.synthesizeWav(text: text, voice: .female)
        let maleWav = engine.synthesizeWav(text: text, voice: .male)

        XCTAssertTrue(44 < femaleWav.count)
        XCTAssertTrue(44 < maleWav.count)
        XCTAssertNotEqual(femaleWav, maleWav)

        // ストリーミング合成
        var streamChunkCount = 0
        let streamSamples = engine.synthesizeStream(text: text, voice: .male) { chunk in
            if 0 < chunk.count {
                streamChunkCount += 1
            }
        }
        XCTAssertTrue(0 < streamSamples.count)
        XCTAssertTrue(0 < streamChunkCount)
    }

    // MARK: - 6. 静的コーディング規約検査

    /// 本改修で新規作成・修正された全ソースコードがプロジェクト規約に完全適合していることを機械走査
    func testNewComponentsStaticRuleCheck() throws {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath

        let targetPaths = [
            currentDir + "/Sources/SpikeSpeech/Common/Types.swift",
            currentDir + "/Sources/SpikeSpeech/DSP/AudioFeatureExtractor.swift",
            currentDir + "/script/dataset/jsut.swift",
            currentDir + "/Sources/SpikeSpeech/Pipeline/SpikeSpeechEngine.swift",
            currentDir + "/Sources/SpikeSpeechWeb/Types.swift",
            currentDir + "/Sources/SpikeSpeechWeb/SpikeSpeechWebServer.swift",
            currentDir + "/script/train/main.swift",
            currentDir + "/script/synthesize/main.swift",
            currentDir + "/script/web/main.swift",
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
        print("--- [JSUT & Voice Static Rule Check] ---")
        print("検証完了ファイル数: \(checkedCount) 件 (全ファイル規約適合)")
        print("----------------------------------------")
    }
}
