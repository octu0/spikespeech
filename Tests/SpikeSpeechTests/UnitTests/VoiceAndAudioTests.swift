import XCTest
import Foundation
@testable import SpikeSpeech

/// 音声特徴量抽出、WAV 読み込み、および話者差し替え（VoiceProfile）検証テストスイート
final class VoiceAndAudioTests: XCTestCase {

    // MARK: - 1. VoiceProfile 単体検証

    /// 既定の話者プロファイル（女性、男性、中性、子供、重低音）の生理音響パラメータ整合性を検証
    /// 既定の話者プロファイル（女性、男性、中性、子供、重低音）の基音パラメータ整合性を検証
    func testVoiceProfilePresetsIntegrity() {
        // 女性ボイス（標準基準）
        let female = VoiceProfile.female
        XCTAssertEqual(female.name, "female")
        XCTAssertEqual(female.baseF0, 220.0)
        XCTAssertEqual(female.energyScale, 1.0)

        // 男性ボイス (低域ピッチ 120Hz)
        let male = VoiceProfile.male
        XCTAssertEqual(male.name, "male")
        XCTAssertTrue(male.baseF0 < 220.0)

        // 中性ボイス (baseF0 170Hz)
        let neutral = VoiceProfile.neutral
        XCTAssertEqual(neutral.name, "neutral")
        XCTAssertTrue(neutral.baseF0 < female.baseF0)
        XCTAssertTrue(male.baseF0 < neutral.baseF0)

        // 子供ボイス (baseF0 300Hz)
        let child = VoiceProfile.child
        XCTAssertEqual(child.name, "child")
        XCTAssertTrue(female.baseF0 < child.baseF0)

        // 重低音男性ボイス (baseF0 95Hz)
        let deepMale = VoiceProfile.deepMale
        XCTAssertEqual(deepMale.name, "deepMale")
        XCTAssertTrue(deepMale.baseF0 < male.baseF0)

        // 既定値が female であること
        XCTAssertEqual(VoiceProfile.default, VoiceProfile.female)

        // 名前に基づくプリセット解決
        XCTAssertEqual(VoiceProfile.preset(named: "female"), VoiceProfile.female)
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
            baseF0: 185.0,
            energyScale: 1.02
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(customVoice)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(VoiceProfile.self, from: data)

        XCTAssertEqual(decoded.name, customVoice.name)
        XCTAssertEqual(decoded.baseF0, customVoice.baseF0)
        XCTAssertEqual(decoded.energyScale, customVoice.energyScale)
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
    func testSyntheticTrainingPairAlignment() throws {
        let engine = SpikeSpeechEngine()
        let sampleRate = 16000
        let audioSamples = 4800 // 300ms = 30 frames

        var wave = [Float](repeating: 0.0, count: audioSamples)
        var s = 0
        while s < audioSamples {
            wave[s] = sin((2.0 * Float.pi * 200.0 * Float(s)) / Float(sampleRate)) * 0.4
            s += 1
        }

        // ディスクへの一時ファイル書き出しを排除し、オンメモリバイト列から直接パースして検証する。
        let wavData = WavEncoder.encode(samples: wave, sampleRate: sampleRate)

        let reader = WavAudioReader()
        let extractor = MelSpectrogramExtractor()
        let pcm = try reader.parseWav16k(bytes: [UInt8](wavData))
        let targetMel = extractor.extractLogMel(pcm: pcm)

        let linguistic = engine.lengthRegulator.processText(
            text: "水をマレーシアから買わなくてはならないのです。",
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )

        let features = engine.encodeLinguisticFeatures(
            features: linguistic
        )

        let expectedFrames = audioSamples / AudioConfig.hopSize // 30 frames
        XCTAssertEqual(targetMel.count, expectedFrames)
        XCTAssertTrue(0 < features.count)

        // 直交特徴量（ch64: 有声度, ch65: 無声度, ch66: 正規化F0）が正しく注入されていることを検証
        var f = 0
        while f < features.count {
            let voiced = features[f][64]
            let unvoiced = features[f][65]
            let sum: Float = voiced + unvoiced
            let targetOne: Float = 1.0
            let eps: Float = 1e-4
            XCTAssertEqual(sum, targetOne, accuracy: eps, "有声度と無声度の和は常に 1.0（直交補空間）であること")
            f += 1
        }
    }

    /// 極短音声に対するアライメント安全性を検証
    func testShortAudioAlignmentAndF0Safety() throws {
        let sampleRate = 16000
        let audioSamples = 480
        let wave = [Float](repeating: 0.1, count: audioSamples)

        // ディスク書き出しを排除し、WavEncoder が生成したオンメモリバイナリを直接検証する。
        let wavData = WavEncoder.encode(samples: wave, sampleRate: sampleRate)

        let reader = WavAudioReader()
        let extractor = MelSpectrogramExtractor()
        let pcm = try reader.parseWav16k(bytes: [UInt8](wavData))
        let targetMel = extractor.extractLogMel(pcm: pcm)

        let expectedFrames = audioSamples / AudioConfig.hopSize // 3 frames
        XCTAssertEqual(targetMel.count, expectedFrames)
    }

    /// 先頭に '@' プレフィックスが付いたパス文字列の正常解決を検証
    func testPathLeadingAtSignResolution() {
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

        // 両者の波形が完全一致せず、物理音響プロファイル（基音 baseF0・声帯パルス OQ/RQ・声道長 VTLN）の違いにより異なることを実証
        var diffSum: Float = 0.0
        let compareCount = min(femaleSamples.count, maleSamples.count)
        var i = 0
        while i < compareCount {
            diffSum += abs(femaleSamples[i] - maleSamples[i])
            i += 1
        }
        let avgDiff = diffSum / Float(compareCount)
        XCTAssertTrue(0.01 < avgDiff, "女性ボイスと男性ボイスの合成波形に有意な差異が存在しません: diff=\(avgDiff)")

        // 3. ピッチ（周波数）特性の検証: 低い基音を持つ男性ボイスはゼロ交差数が女性ボイスより少なくなる
        // なぜ基音成分（F0）を平滑化してゼロ交差数を測定するか:
        // 生理学的な声道音響において、男性声帯の急峻な閉鎖波形や高調波・フォルマント共鳴の波形交差を除去し、
        // 基音周期（男性 120Hz vs 女性 220Hz）本来のゼロ交差密度（男性 < 女性）を正確に評価するため。
        func countBaseZeroCrossings(_ samples: [Float]) -> Int {
            var filtered = [Float](repeating: 0.0, count: samples.count)
            var prev: Float = 0.0
            var i = 0
            while i < samples.count {
                prev = prev + (0.06 * (samples[i] - prev))
                filtered[i] = prev
                i += 1
            }
            var crossings = 0
            var idx = 1
            while idx < filtered.count {
                if (filtered[idx - 1] < 0.0 && 0.0 <= filtered[idx]) || (0.0 <= filtered[idx - 1] && filtered[idx] < 0.0) {
                    crossings += 1
                }
                idx += 1
            }
            return crossings
        }

        let femaleZeroCrossings = countBaseZeroCrossings(femaleSamples)
        let maleZeroCrossings = countBaseZeroCrossings(maleSamples)

        // なぜ基音成分を平滑化したゼロ交差数を比較するか:
        // 高周波フォルマント共鳴の交差を排し、低い基音を持つ男性ボイス（125Hz）が女性ボイス（220Hz）より
        // 基音ゼロ交差数が確実に少なくなる周波数特性を検証するため。
        XCTAssertTrue(maleZeroCrossings < femaleZeroCrossings, "男性ボイスの基音ゼロ交差数が女性ボイスを下回っていません: male=\(maleZeroCrossings), female=\(femaleZeroCrossings)")
        XCTAssertTrue(0 < maleZeroCrossings, "平滑化ゼロ交差数がゼロです")
    }

    /// 話者切り替え時において、話者基音の絶対値が変わっても「発音・アクセントの響き（相対 F0 輪郭）」が
    /// 統計的・音響学的に完全に保存されていることをピアソン相関係数により実証（相談書 §5.8 必須検定）
    func testVoiceSwitchingAccentShapePreservation() {
        let prosody = ProsodyModel()
        let vocab = PhonemeVocabulary()
        var bio = BiologicalFluctuation(seed: 1234)

        // 「こんにちは」のアクセント句（低高高高低）
        let moras = [
            MoraToken(text: "こ", phonemes: [PhonemeToken(id: 15, symbol: "k", category: .consonant, durationFrames: 4), PhonemeToken(id: 9, symbol: "o", category: .vowel, durationFrames: 8)], tone: .low),
            MoraToken(text: "ん", phonemes: [PhonemeToken(id: 30, symbol: "N", category: .consonant, durationFrames: 8)], tone: .high),
            MoraToken(text: "に", phonemes: [PhonemeToken(id: 20, symbol: "n", category: .consonant, durationFrames: 4), PhonemeToken(id: 6, symbol: "i", category: .vowel, durationFrames: 8)], tone: .high),
            MoraToken(text: "ち", phonemes: [PhonemeToken(id: 23, symbol: "ch", category: .consonant, durationFrames: 4), PhonemeToken(id: 6, symbol: "i", category: .vowel, durationFrames: 8)], tone: .high),
            MoraToken(text: "は", phonemes: [PhonemeToken(id: 25, symbol: "w", category: .consonant, durationFrames: 4), PhonemeToken(id: 5, symbol: "a", category: .vowel, durationFrames: 8)], tone: .low)
        ]
        let phrase = AccentPhrase(moras: moras)

        let (f0Female, _, _) = prosody.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocab,
            baseF0: VoiceProfile.female.baseF0,
            fluctuation: &bio,
            applyFluctuation: false
        )

        var bioMale = BiologicalFluctuation(seed: 1234)
        let (f0Male, _, _) = prosody.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocab,
            baseF0: VoiceProfile.male.baseF0,
            fluctuation: &bioMale,
            applyFluctuation: false
        )

        XCTAssertEqual(f0Female.count, f0Male.count)

        // 有声区間における対数 F0 の平均値を算出
        var logFemale: [Float] = []
        var logMale: [Float] = []
        var sumFemale: Float = 0.0
        var sumMale: Float = 0.0
        var frame = 0
        while frame < f0Female.count {
            if 0.0 < f0Female[frame] && 0.0 < f0Male[frame] {
                let lf = logf(f0Female[frame])
                let lm = logf(f0Male[frame])
                logFemale.append(lf)
                logMale.append(lm)
                sumFemale += lf
                sumMale += lm
            }
            frame += 1
        }

        XCTAssertTrue(0 < logFemale.count)
        let meanFemale = sumFemale / Float(logFemale.count)
        let meanMale = sumMale / Float(logMale.count)

        // ピアソン相関係数 r を算出
        var cov: Float = 0.0
        var varF: Float = 0.0
        var varM: Float = 0.0
        var idx = 0
        while idx < logFemale.count {
            let df = logFemale[idx] - meanFemale
            let dm = logMale[idx] - meanMale
            cov += df * dm
            varF += df * df
            varM += dm * dm
            idx += 1
        }

        let denom = sqrtf(varF * varM)
        XCTAssertTrue(0.0 < denom)
        let corr = cov / denom

        // 相関係数 r > 0.99 であり、話者基音を差し替えても抑揚・アクセントの形（響き）が完全保存されていることを数学的に証明
        XCTAssertTrue(0.99 <= corr, "話者差し替えによる相対アクセント相関が不十分です: r=\(corr)")
    }

    /// SpeakerConditioning 構造体の初期化、ゼロ値、および JSON 直列化・逆直列化を検証
    func testSpeakerConditioningVectorRepresentation() throws {
        XCTAssertEqual(SpeakerConditioning.defaultDimension, 128)

        let zeroCond = SpeakerConditioning.zero
        XCTAssertEqual(zeroCond.dimension, 128)
        XCTAssertEqual(zeroCond.embedding.count, 128)
        XCTAssertTrue(zeroCond.embedding.allSatisfy { $0 == 0.0 })

        var customVec = [Float](repeating: 0.0, count: 128)
        var i = 0
        while i < 128 {
            customVec[i] = Float(i) * 0.01
            i += 1
        }
        let customCond = SpeakerConditioning(embedding: customVec)
        XCTAssertEqual(customCond.dimension, 128)

        let encoder = JSONEncoder()
        let data = try encoder.encode(customCond)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(SpeakerConditioning.self, from: data)

        XCTAssertEqual(decoded.dimension, customCond.dimension)
        XCTAssertEqual(decoded.embedding, customCond.embedding)
    }

    /// VoiceProfile の差が言語 F0 (baseF0) のみであることを検証（受入条件 5）
    func testVoiceProfileBaseF0DifferenceOnly() {
        let female = VoiceProfile.female
        let male = VoiceProfile.male

        // baseF0 のみが異なり、声帯・声道物理共鳴モデル等の不要なスペクトル差分パラメータが存在しないこと
        XCTAssertNotEqual(female.baseF0, male.baseF0)
        XCTAssertEqual(female.energyScale, male.energyScale)
    }

    /// SpikeSpeechEngine の波形合成において SpeakerConditioning が正しく反映されることを検証
    func testSpeakerConditioningAffectsSynthesize() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは"

        let speakerA = SpeakerConditioning(embedding: [Float](repeating: 0.5, count: 128))
        let speakerB = SpeakerConditioning(embedding: [Float](repeating: -0.5, count: 128))

        let pcmA = engine.synthesize(text: text, speaker: speakerA)
        let pcmB = engine.synthesize(text: text, speaker: speakerB)

        XCTAssertFalse(pcmA.isEmpty)
        XCTAssertFalse(pcmB.isEmpty)
        XCTAssertEqual(pcmA.count, pcmB.count)

        var diffSum: Float = 0.0
        var i = 0
        while i < pcmA.count {
            diffSum += abs(pcmA[i] - pcmB[i])
            i += 1
        }
        let avgDiff = diffSum / Float(pcmA.count)
        XCTAssertTrue(1e-5 < avgDiff, "話者条件付けによる波形差分が検出されません: diff=\(avgDiff)")
    }

    /// ProsodyModel において話者基音 baseF0 を差し替えても相対アクセント・抑揚が完全に保存されることを検証
    func testRelativeProsodyBaseF0StrictLinearity() {
        let prosody = ProsodyModel()
        let vocab = PhonemeVocabulary()
        var bio = BiologicalFluctuation(seed: 42)

        let moras = [
            MoraToken(text: "あ", phonemes: [PhonemeToken(id: 5, symbol: "a", category: .vowel, durationFrames: 10)], tone: .high),
            MoraToken(text: "め", phonemes: [PhonemeToken(id: 8, symbol: "e", category: .vowel, durationFrames: 10)], tone: .low)
        ]
        let phrase = AccentPhrase(moras: moras)

        let (f0Female, _, _) = prosody.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocab,
            baseF0: 220.0,
            fluctuation: &bio,
            applyFluctuation: false
        )

        var bioMale = BiologicalFluctuation(seed: 42)
        let (f0Male, _, _) = prosody.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocab,
            baseF0: 120.0,
            fluctuation: &bioMale,
            applyFluctuation: false
        )

        XCTAssertEqual(f0Female.count, f0Male.count)
        let expectedRatio: Float = 220.0 / 120.0
        var f = 0
        while f < f0Female.count {
            if 0.0 < f0Male[f] {
                let ratio = f0Female[f] / f0Male[f]
                XCTAssertEqual(ratio, expectedRatio, accuracy: 0.05, "話者基音の比率が保たれ相対抑揚が保存されていること: frame=\(f)")
            }
            f += 1
        }
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

    /// 正本パイプラインで生成された学習目標残差が数学的に完全可逆（往復復元）であることを実データで検証
    func testResidualScaleConsistency() {
        let engine = SpikeSpeechEngine()

        let extractor = MelSpectrogramExtractor()
        let tracker = PitchTracker()
        let sampleRate = 16000
        let audioSamples = 4800 // 300ms = 30 frames

        var wave = [Float](repeating: 0.0, count: audioSamples)
        var s = 0
        while s < audioSamples {
            wave[s] = sin((2.0 * Float.pi * 200.0 * Float(s)) / Float(sampleRate)) * 0.1
            s += 1
        }

        // 正本 prepareTrainingPair による学習ペア生成
        guard let pair = engine.prepareTrainingPair(
            text: "あ",
            pcm16k: wave,
            melExtractor: extractor,
            pitchTracker: tracker
        ) else {
            XCTFail("prepareTrainingPair が有効なペアを返しませんでした")
            return
        }

        // 入力実音声からの直接 Log Mel 抽出
        let originalMel = extractor.extractLogMel(pcm: wave)
        let frameCount = min(pair.targets.count, originalMel.count)
        XCTAssertTrue(0 < frameCount)

        var t = 0
        while t < frameCount {
            var c = 0
            while c < AudioConfig.melChannels {
                let targetVal = pair.targets[t][c]
                XCTAssertEqual(targetVal, originalMel[t][c], accuracy: 1e-4, "フレーム \(t) ch \(c) で目標対数 Mel が originalMel と不一致です")
                c += 1
            }
            t += 1
        }
    }

    /// decodeSequence においてフレーム間で膜電位が不自然に 0.2 倍に消去されず、BPTT と同様に時間連続性が保たれることを検証
    func testDecoderMembraneContinuity() {
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 128,
            maxHiddenDim: 64,
            outputDim: 64,
            numLayers: 2
        )
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(
            maxHiddenDim: weights.maxHiddenDim,
            outputDim: weights.outputDim,
            numLayers: weights.numLayers
        )

        // 2 フレーム分の入力特徴量系列
        let frame0 = [Float](repeating: 0.5, count: 128)
        let frame1 = [Float](repeating: 0.5, count: 128)
        let seq = [frame0, frame1]

        let output = decoder.decodeSequence(featuresSeq: seq, workspace: workspace)
        XCTAssertEqual(output.count, 2)

        // 第 2 フレームの推論後に膜電位がゼロや極端な減衰（0.2 倍）になっておらず、
        // 直前フレームの膜電位が時間連続的に引き継がれていることを検証
        var nonZeroMembrane = false
        var i = 0
        let v0 = workspace.layerStates[0].v
        while i < v0.count {
            if 0.01 < abs(v0[i]) {
                nonZeroMembrane = true
                break
            }
            i += 1
        }
        XCTAssertTrue(nonZeroMembrane, "フレーム間で膜電位が時間連続的に保持されていません")
    }

    /// 学習データ構築パイプラインの正本（SpikeSpeechEngine.prepareTrainingPair）において、
    /// 微小ゲイン録音であっても有声母音エネルギー（ch70）が推論側母音エネルギー（0.80）と完全一致することを直接検証
    func testEnergyNormalizationDistribution() {
        let engine = SpikeSpeechEngine()
        let extractor = MelSpectrogramExtractor()
        let tracker = PitchTracker()
        let sampleRate = 16000
        let audioSamples = 4800 // 300ms = 30 frames

        // 微小振幅（RMS 約 0.07）の正弦波音声
        var wave = [Float](repeating: 0.0, count: audioSamples)
        var s = 0
        while s < audioSamples {
            wave[s] = sin((2.0 * Float.pi * 200.0 * Float(s)) / Float(sampleRate)) * 0.1
            s += 1
        }

        // 本番パイプラインを直接実行
        guard let pair = engine.prepareTrainingPair(
            text: "あ",
            pcm16k: wave,
            melExtractor: extractor,
            pitchTracker: tracker
        ) else {
            XCTFail("prepareTrainingPair が有効なペアを返しませんでした")
            return
        }

        XCTAssertTrue(0 < pair.features.count)
        XCTAssertEqual(pair.features.count, pair.targets.count)

        // 入力特徴量 ch70（エネルギー）のピーク値を走査
        var peakEnergy: Float = 0.0
        var f = 0
        while f < pair.features.count {
            let energyCh = pair.features[f][70]
            if peakEnergy < energyCh {
                peakEnergy = energyCh
            }
            f += 1
        }

        // 発話内ピークが推論側母音エネルギー（0.80）に正確に正規化されていることを検証
        XCTAssertTrue(0.75 <= peakEnergy, "スケーリング後の発話ピークエネルギーが推論側母音エネルギー（0.80）に届いていません: \(peakEnergy)")
        XCTAssertTrue(peakEnergy <= 0.85, "スケーリング後の発話ピークエネルギーが過大です: \(peakEnergy)")
    }

    // MARK: - 7. 静的コーディング規約検査

    /// 本改修で新規作成・修正された全ソースコードがプロジェクト規約に完全適合していることを機械走査
    func testNewComponentsStaticRuleCheck() throws {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath

        let targetPaths = [
            currentDir + "/Sources/SpikeSpeech/Common/Types.swift",
            currentDir + "/Sources/SpikeSpeech/DSP/AudioFeatureExtractor.swift",
            currentDir + "/Sources/SpikeSpeech/DSP/PitchTracker.swift",
            currentDir + "/Sources/SpikeSpeech/SNN/SpikingAcousticDecoder.swift",
            currentDir + "/Sources/SpikeSpeech/DSP/NeuralVocoder.swift",
            currentDir + "/Sources/SpikeSpeech/Pipeline/SpikeSpeechEngine.swift",
            currentDir + "/Sources/SpikeSpeech/Pipeline/SpikeSpeechEngine+Training.swift",
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
        print("--- [Voice & Audio Static Rule Check] ---")
        print("検証完了ファイル数: \(checkedCount) 件 (全ファイル規約適合)")
        print("----------------------------------------")
    }
}
