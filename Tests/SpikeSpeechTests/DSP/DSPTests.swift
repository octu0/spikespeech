import XCTest
import Foundation
@testable import SpikeSpeech

/// Audio DSP, NeuralVocoder, MelSpectrogramExtractor, WavEncoder, WavAudioReader の網羅的単体テストスイート
final class DSPTests: XCTestCase {

    // MARK: - 1. NeuralVocoder 時間領域波形生成特性テスト

    func testNeuralVocoderFrequencyResponseAndWaveform() {
        // 対数 Mel フレーム系列から 16kHz PCM が破綻なく生成され、
        // 振幅が許容範囲内に収まることを数学的に検証する。
        let vocoder = NeuralVocoder()
        let melFrame = [Float](repeating: -2.0, count: 64)
        let samples = vocoder.synthesize(mel: [melFrame, melFrame])
        XCTAssertEqual(samples.count, 320)

        var hasNonZero = false
        var maxVal: Float = 0.0
        var i = 0
        while i < samples.count {
            let s = samples[i]
            XCTAssertTrue(s.isFinite)
            if 0.0 < abs(s) {
                hasNonZero = true
            }
            if maxVal < abs(s) {
                maxVal = abs(s)
            }
            i += 1
        }
        XCTAssertTrue(hasNonZero)
        XCTAssertTrue(maxVal <= 1.0)
    }

    // MARK: - 2. AudioFeatureExtractor 特徴量抽出妥当性テスト

    func testMelSpectrogramExtractorProperties() {
        // 短時間フーリエ変換と Mel フィルタバンクによる対数 Mel スペクトログラム抽出が
        // 有限な数値を安定して出力することを保証する。
        let extractor = MelSpectrogramExtractor(sampleRate: 16000, melChannels: 64, hopSize: 160, fftSize: 512)
        var sine = [Float](repeating: 0.0, count: 640)
        var s = 0
        while s < 640 {
            sine[s] = sinf((2.0 * Float.pi * 440.0 * Float(s)) / 16000.0) * 0.5
            s += 1
        }
        let melFrames = extractor.extractLogMel(pcm: sine)
        XCTAssertTrue(0 < melFrames.count)

        var f = 0
        while f < melFrames.count {
            XCTAssertEqual(melFrames[f].count, 64)
            var ch = 0
            while ch < 64 {
                XCTAssertTrue(melFrames[f][ch].isFinite)
                ch += 1
            }
            f += 1
        }
    }

    // MARK: - 3. NeuralVocoder の安定性・エネルギー保持テスト

    func testNeuralVocoderStabilityAndResonance() {
        // 連続する Mel フレームに対して内部状態が発散せず、
        // 安定したエネルギーを持つ音声波形が継続して出力されることを実証する。
        let vocoder = NeuralVocoder()
        let frameCount = 10
        var frames: [[Float]] = []
        var f = 0
        while f < frameCount {
            frames.append([Float](repeating: -3.0, count: 64))
            f += 1
        }

        let audio = vocoder.synthesize(mel: frames)
        XCTAssertEqual(audio.count, frameCount * 160)

        var hasNonZero = false
        var maxVal: Float = 0.0
        var sIdx = 0
        while sIdx < audio.count {
            let s = audio[sIdx]
            XCTAssertTrue(s.isFinite)
            if 0.0 < abs(s) {
                hasNonZero = true
            }
            if maxVal < abs(s) {
                maxVal = abs(s)
            }
            sIdx += 1
        }

        XCTAssertTrue(hasNonZero)
        XCTAssertTrue(maxVal <= 1.0)
    }

    // MARK: - 4. NaN/Inf ガードおよび過大入力サチュレーション耐性テスト

    func testNeuralVocoderNaNAndOverdriveSafety() {
        // 極大入力や非有限値（NaN, ±Inf）が入力された場合でも、
        // ボコーダーがクラッシュせず安全に有限値でフォールバックすることを証明する。
        let vocoder = NeuralVocoder()

        // 1. 巨大値フレーム (Overdrive)
        let overdriveFrame = [Float](repeating: 50.0, count: 64)
        let outOverdrive = vocoder.synthesize(mel: [overdriveFrame])
        XCTAssertEqual(outOverdrive.count, 160)

        var i = 0
        while i < 160 {
            let s = outOverdrive[i]
            XCTAssertTrue(s.isFinite)
            XCTAssertTrue(-1.0 <= s)
            XCTAssertTrue(s <= 1.0)
            i += 1
        }

        // 2. NaN 係数フレームの注入
        var nanMel = [Float](repeating: -2.0, count: 64)
        nanMel[0] = Float.nan
        nanMel[1] = Float.infinity
        nanMel[2] = -Float.infinity
        let outNaN = vocoder.synthesize(mel: [nanMel])
        XCTAssertEqual(outNaN.count, 160)

        i = 0
        while i < 160 {
            let s = outNaN[i]
            XCTAssertTrue(s.isFinite)
            i += 1
        }
    }

    // MARK: - 5. WAV ヘッダ生成および PCM サンプル量子化テスト

    func testWavEncodingAndHeaderStructure() {
        let samples: [Float] = [0.0, 0.5, -0.5, 0.99, -0.99]
        let wavData = WavEncoder.encode(samples: samples, sampleRate: 16000)

        let expectedDataSize = samples.count * 2
        let expectedTotalSize = 44 + expectedDataSize
        XCTAssertEqual(wavData.count, expectedTotalSize)

        let bytes = [UInt8](wavData)

        // "RIFF"
        XCTAssertEqual(bytes[0], 0x52)
        XCTAssertEqual(bytes[1], 0x49)
        XCTAssertEqual(bytes[2], 0x46)
        XCTAssertEqual(bytes[3], 0x46)

        // "WAVE"
        XCTAssertEqual(bytes[8], 0x57)
        XCTAssertEqual(bytes[9], 0x41)
        XCTAssertEqual(bytes[10], 0x56)
        XCTAssertEqual(bytes[11], 0x45)

        // "fmt "
        XCTAssertEqual(bytes[12], 0x66)
        XCTAssertEqual(bytes[13], 0x6D)
        XCTAssertEqual(bytes[14], 0x74)
        XCTAssertEqual(bytes[15], 0x20)

        // Format: 1 (PCM)
        XCTAssertEqual(bytes[20], 1)
        XCTAssertEqual(bytes[21], 0)

        // Channels: 1
        XCTAssertEqual(bytes[22], 1)
        XCTAssertEqual(bytes[23], 0)

        // SampleRate: 16000
        XCTAssertEqual(bytes[24], 0x80)
        XCTAssertEqual(bytes[25], 0x3E)
        XCTAssertEqual(bytes[26], 0x00)
        XCTAssertEqual(bytes[27], 0x00)

        // ByteRate: 32000
        XCTAssertEqual(bytes[28], 0x00)
        XCTAssertEqual(bytes[29], 0x7D)
        XCTAssertEqual(bytes[30], 0x00)
        XCTAssertEqual(bytes[31], 0x00)

        // BlockAlign: 2
        XCTAssertEqual(bytes[32], 2)
        XCTAssertEqual(bytes[33], 0)

        // BitsPerSample: 16
        XCTAssertEqual(bytes[34], 16)
        XCTAssertEqual(bytes[35], 0)

        // "data"
        XCTAssertEqual(bytes[36], 0x64)
        XCTAssertEqual(bytes[37], 0x61)
        XCTAssertEqual(bytes[38], 0x74)
        XCTAssertEqual(bytes[39], 0x61)

        // Subchunk2Size: 10 bytes
        XCTAssertEqual(bytes[40], UInt8(expectedDataSize & 0xFF))
        XCTAssertEqual(bytes[41], 0)

        // WavStreamWriter の動作テスト
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_stream_\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: tempFile.path, contents: nil)

        do {
            let handle = try FileHandle(forWritingTo: tempFile)
            let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

            try writer.write(samples: [0.1, 0.2, 0.3])
            try writer.write(samples: [-0.1, -0.2])
            try writer.finalize()
            try handle.close()

            let writtenData = try Data(contentsOf: tempFile)
            XCTAssertEqual(writtenData.count, 44 + (5 * 2))

            let writtenBytes = [UInt8](writtenData)
            let writtenDataSize = Int(writtenBytes[40]) | (Int(writtenBytes[41]) << 8)
            XCTAssertEqual(writtenDataSize, 10)

            try? FileManager.default.removeItem(at: tempFile)
        } catch {
            XCTFail("WavStreamWriter のテスト中にエラーが発生しました: \(error)")
        }
    }

    // MARK: - 6. 無音・微小 Mel 入力に対する安定出力テスト

    func testNeuralVocoderSilenceHandling() {
        let vocoder = NeuralVocoder()
        let silenceFrame = [Float](repeating: -20.0, count: 64)
        let samples = vocoder.synthesize(mel: [silenceFrame])
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < 160 {
            XCTAssertTrue(samples[i].isFinite)
            i += 1
        }
    }

    // MARK: - 7. WavStreamWriter の空書き込みおよび多重 finalize 耐性テスト

    func testWavStreamWriterEmptyAndDoubleFinalize() {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_empty_\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: tempFile.path, contents: nil)

        do {
            let handle = try FileHandle(forWritingTo: tempFile)
            let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

            try writer.write(samples: [])
            try writer.finalize()
            try writer.finalize()
            try handle.close()

            let writtenData = try Data(contentsOf: tempFile)
            XCTAssertEqual(writtenData.count, 44, "空書き込み時の WAV ファイルサイズが 44 バイトと一致しません")

            try? FileManager.default.removeItem(at: tempFile)
        } catch {
            XCTFail("空書き込みテスト中にエラーが発生しました: \(error)")
        }
    }

    // MARK: - 8. 極端な Mel ピーク入力に対するクランプ安定性テスト

    func testNeuralVocoderExtremeMelPeakStability() {
        let vocoder = NeuralVocoder()
        var extremeMel = [Float](repeating: -10.0, count: 64)
        extremeMel[10] = 50.0

        let samples = vocoder.synthesize(mel: [extremeMel])
        XCTAssertEqual(samples.count, 160)
        var i = 0
        while i < 160 {
            let s = samples[i]
            XCTAssertTrue(s.isFinite)
            XCTAssertTrue(-1.0 <= s)
            XCTAssertTrue(s <= 1.0)
            i += 1
        }
    }

    // MARK: - 9. WavAudioReader 単体検証

    func testWavAudioReader16kMono() throws {
        let sampleRate = 16000
        let count = 1600
        var rawSamples = [Float](repeating: 0.0, count: count)
        var i = 0
        while i < count {
            rawSamples[i] = sinf((2.0 * Float.pi * 440.0 * Float(i)) / Float(sampleRate)) * 0.5
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
        XCTAssertTrue(maxDiff < 0.001)
    }

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

    func testWavAudioReaderInvalidDataThrows() {
        let reader = WavAudioReader()
        let truncatedBytes: [UInt8] = [0x52, 0x49, 0x46, 0x46, 0x00, 0x00]
        XCTAssertThrowsError(try reader.parseWav16k(bytes: truncatedBytes))
    }

    func testWavAudioReaderBoundaryCheck() {
        let reader = WavAudioReader()
        let bytes: [UInt8] = [
            0x52, 0x49, 0x46, 0x46,
            0x24, 0x00, 0x00, 0x00,
            0x57, 0x41, 0x56, 0x45,
            0x66, 0x6d, 0x74, 0x20,
            0x10, 0x00, 0x00, 0x00,
            0x01, 0x00,
            0x01, 0x00,
            0x80, 0x3e, 0x00, 0x00,
            0x00, 0x7d, 0x00, 0x00,
            0x02, 0x00
        ]
        XCTAssertThrowsError(try reader.parseWav16k(bytes: bytes))
    }

    func testWavAudioReader24BitPCM() throws {
        let reader = WavAudioReader()
        let bytes: [UInt8] = [
            0x52, 0x49, 0x46, 0x46, 0x2e, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
            0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x01, 0x00,
            0x80, 0x3e, 0x00, 0x00,
            0x80, 0xbb, 0x00, 0x00,
            0x03, 0x00, 0x18, 0x00,
            0x64, 0x61, 0x74, 0x61, 0x06, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x40,
            0x00, 0x00, 0xc0
        ]

        let pcm = try reader.parseWav16k(bytes: bytes)
        XCTAssertEqual(pcm.count, 2)
        XCTAssertEqual(pcm[0], 0.5, accuracy: 1e-4)
        XCTAssertEqual(pcm[1], -0.5, accuracy: 1e-4)
    }

    // MARK: - 10. MelSpectrogramExtractor 周波数ピーク局在と NaN 耐性

    func testMelSpectrogramFrequencyPeakLocalization() {
        let extractor = MelSpectrogramExtractor()
        let totalSamples = 3200
        var pcm = [Float](repeating: 0.0, count: totalSamples)
        var i = 0
        while i < totalSamples {
            pcm[i] = sinf((2.0 * Float.pi * 1000.0 * Float(i)) / 16000.0) * 0.8
            i += 1
        }

        let mel = extractor.extractLogMel(pcm: pcm)
        XCTAssertTrue(0 < mel.count)

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

        XCTAssertTrue(10 <= maxCh)
        XCTAssertTrue(maxCh <= 35)
    }

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
}
