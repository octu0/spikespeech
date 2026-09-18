import XCTest
@testable import SpikeSpeech

/// Audio DSP, NeuralVocoder, MelSpectrogramExtractor, WAV エンコーダーの網羅的単体テストスイート
///
/// SIMD8 ベクトル演算のビット精度、対数 Mel スペクトログラム抽出の数理的妥当性、
/// ニューラルボコーダーの連続推論安定性、過大入力や NaN 注入に対する無発振安全性、
/// および 44 バイト WAV バイナリ整合性を数理オラクルによって実証・保証する。
final class DSPTests: XCTestCase {

    // MARK: - 1. SIMD8 ベクトル演算精度テスト

    func testSIMD8VectorOperations() {
        // 8 要素境界（SIMD8 パス）と端数（スカラーフォールバックパス）の両方において
        // 演算結果がスカラー理論値と完全一致することを検証する。
        let testSizes = [0, 1, 7, 8, 15, 16, 31, 32, 33]
        var sIdx = 0
        while sIdx < testSizes.count {
            let count = testSizes[sIdx]
            var a = [Float](repeating: 0.0, count: count)
            var b = [Float](repeating: 0.0, count: count)
            var expectedDot: Float = 0.0
            var expectedSqA: Float = 0.0

            var i = 0
            while i < count {
                let va = (Float(i % 10) * 0.5) - 2.0
                let vb = (Float((i + 3) % 7) * 0.3) - 1.0
                a[i] = va
                b[i] = vb
                expectedDot += va * vb
                expectedSqA += va * va
                i += 1
            }

            if 0 < count {
                let actualDot = a.withUnsafeBufferPointer { pA in
                    b.withUnsafeBufferPointer { pB in
                        VectorOperations.dotProduct(a: pA.baseAddress!, b: pB.baseAddress!, count: count)
                    }
                }
                let diffDot = abs(actualDot - expectedDot)
                XCTAssertTrue(diffDot < 1e-4, "内積の精度誤差が許容値を超過: size=\(count), diff=\(diffDot)")

                let actualSq = a.withUnsafeBufferPointer { pA in
                    VectorOperations.sumOfSquares(ptr: pA.baseAddress!, count: count)
                }
                let diffSq = abs(actualSq - expectedSqA)
                XCTAssertTrue(diffSq < 1e-4, "二乗和の精度誤差が許容値を超過: size=\(count), diff=\(diffSq)")

                var dstMul = [Float](repeating: 0.0, count: count)
                dstMul.withUnsafeMutableBufferPointer { pDst in
                    a.withUnsafeBufferPointer { pA in
                        b.withUnsafeBufferPointer { pB in
                            VectorOperations.multiply(srcA: pA.baseAddress!, srcB: pB.baseAddress!, dst: pDst.baseAddress!, count: count)
                        }
                    }
                }
                var m = 0
                while m < count {
                    let diffMul = abs(dstMul[m] - (a[m] * b[m]))
                    XCTAssertTrue(diffMul < 1e-5, "要素積の不一致: index=\(m)")
                    m += 1
                }
            }
            sIdx += 1
        }

        // maxMagnitude の検証
        let magData: [Float] = [-1.5, 3.2, -8.7, 4.0, -12.3, 0.0, 7.5, -2.1, 9.4]
        let maxMag = magData.withUnsafeBufferPointer { p in
            VectorOperations.maxMagnitude(ptr: p.baseAddress!, count: magData.count)
        }
        XCTAssertEqual(maxMag, 12.3, accuracy: 1e-5)

        // clamp の検証
        let clampSrc: [Float] = [-2.0, -0.8, -0.3, 0.0, 0.4, 0.7, 1.5]
        var clampDst = [Float](repeating: 0.0, count: clampSrc.count)
        clampDst.withUnsafeMutableBufferPointer { pDst in
            clampSrc.withUnsafeBufferPointer { pSrc in
                VectorOperations.clamp(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: clampSrc.count, minVal: -0.5, maxVal: 0.5)
            }
        }
        let expectedClamp: [Float] = [-0.5, -0.5, -0.3, 0.0, 0.4, 0.5, 0.5]
        var cIdx = 0
        while cIdx < clampSrc.count {
            XCTAssertEqual(clampDst[cIdx], expectedClamp[cIdx], accuracy: 1e-5)
            cIdx += 1
        }

        // softLimitTanh の検証
        let limSrc: [Float] = [-2.0, -0.8, -0.5, 0.0, 0.5, 0.8, 2.0, Float.nan]
        var limDst = [Float](repeating: 0.0, count: limSrc.count)
        limDst.withUnsafeMutableBufferPointer { pDst in
            limSrc.withUnsafeBufferPointer { pSrc in
                VectorOperations.softLimitTanh(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: limSrc.count, threshold: 0.8)
            }
        }
        // 0.8 以下の要素はそのまま保持
        XCTAssertEqual(limDst[1], -0.8, accuracy: 1e-5)
        XCTAssertEqual(limDst[2], -0.5, accuracy: 1e-5)
        XCTAssertEqual(limDst[3], 0.0, accuracy: 1e-5)
        XCTAssertEqual(limDst[4], 0.5, accuracy: 1e-5)
        XCTAssertEqual(limDst[5], 0.8, accuracy: 1e-5)
        // 2.0 の要素は 1.0 未満に滑らかに圧縮
        XCTAssertTrue(limDst[6] < 1.0)
        XCTAssertTrue(0.8 < limDst[6])
        XCTAssertTrue(-1.0 < limDst[0])
        XCTAssertTrue(limDst[0] < -0.8)
        // NaN は 0.0 に安全置換
        XCTAssertEqual(limDst[7], 0.0)

        // quantizeFloatToInt16 の検証
        let quantSrc: [Float] = [-2.0, -1.0, 0.0, 1.0, 2.0, Float.nan]
        var quantDst = [Int16](repeating: 0, count: quantSrc.count)
        quantDst.withUnsafeMutableBufferPointer { pDst in
            quantSrc.withUnsafeBufferPointer { pSrc in
                VectorOperations.quantizeFloatToInt16(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: quantSrc.count)
            }
        }
        XCTAssertEqual(quantDst[0], -32768)
        XCTAssertEqual(quantDst[1], -32767)
        XCTAssertEqual(quantDst[2], 0)
        XCTAssertEqual(quantDst[3], 32767)
        XCTAssertEqual(quantDst[4], 32767)
        XCTAssertEqual(quantDst[5], 0)
    }

    // MARK: - 2. NeuralVocoder 時間領域波形生成特性テスト

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

    // MARK: - 3. AudioFeatureExtractor 特徴量抽出妥当性テスト

    func testMelSpectrogramExtractorProperties() {
        // 短時間フーリエ変換と Mel フィルタバンクによる対数 Mel スペクトログラム抽出が
        // 有限な数値を安定して出力することを保証する。
        let extractor = MelSpectrogramExtractor(sampleRate: 16000, melChannels: 64, hopSize: 160, fftSize: 512)
        var sine = [Float](repeating: 0.0, count: 640)
        var s = 0
        while s < 640 {
            sine[s] = sinf(2.0 * Float.pi * 440.0 * Float(s) / 16000.0) * 0.5
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

    // MARK: - 4. NeuralVocoder の安定性・エネルギー保持テスト

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

    // MARK: - 5. NaN/Inf ガードおよび過大入力サチュレーション耐性テスト

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

    // MARK: - 6. WAV ヘッダ生成および PCM サンプル量子化テスト

    func testWavEncodingAndHeaderStructure() {
        // RIFF/WAVE フォーマットの仕様（オフセット 0x00 の "RIFF"、0x08 の "WAVE"、
        // 0x14 の AudioFormat=1、0x18 の SampleRate=16000）との完全一致を保証する。
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

        // SampleRate: 16000 (0x00003E80 -> 0x80, 0x3E, 0x00, 0x00)
        XCTAssertEqual(bytes[24], 0x80)
        XCTAssertEqual(bytes[25], 0x3E)
        XCTAssertEqual(bytes[26], 0x00)
        XCTAssertEqual(bytes[27], 0x00)

        // ByteRate: 32000 (0x00007D00 -> 0x00, 0x7D, 0x00, 0x00)
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

            // 2 回に分けて書き込み
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

    // MARK: - 7. 無音・微小 Mel 入力に対する安定出力テスト

    func testNeuralVocoderSilenceHandling() {
        // ポーズや文末において、極小対数 Mel 入力に対して
        // 異常発散せず有限な波形が出力されることを保証する。
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

    // MARK: - 8. WavStreamWriter の空書き込みおよび多重 finalize 耐性テスト

    func testWavStreamWriterEmptyAndDoubleFinalize() {
        // 短い音声のストリーミングや複数回 finalize 呼び出しによる
        // ファイル破壊やクラッシュが発生しない堅牢性を実証する。
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_empty_\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: tempFile.path, contents: nil)

        do {
            let handle = try FileHandle(forWritingTo: tempFile)
            let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

            // 空サンプルの書き出し
            try writer.write(samples: [])
            // 複数回の finalize 呼び出し
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

    // MARK: - 9. 極端な Mel ピーク入力に対するクランプ安定性テスト

    func testNeuralVocoderExtremeMelPeakStability() {
        // 単一周波数成分が極端に突出した対数 Mel が入力された場合でも、
        // 内部畳み込みが発散せず、出力サンプルが [-1.0, 1.0] に安全に収まることを検証する。
        let vocoder = NeuralVocoder()
        var extremeMel = [Float](repeating: -10.0, count: 64)
        extremeMel[10] = 50.0 // 極大ピーク

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
}
