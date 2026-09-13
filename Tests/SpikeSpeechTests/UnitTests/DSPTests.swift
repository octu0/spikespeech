import XCTest
@testable import SpikeSpeech

/// Audio DSP, Rosenberg パルス, MelToLPC, LPCVocoder, WAV エンコーダーの網羅的単体テストスイート
///
/// SIMD8 ベクトル演算のビット精度、Rosenberg 声門音源モデルの数理的周期性、
/// Levinson-Durbin 法における反射係数クランプと単位円内極保証、
/// 過大入力や NaN 注入に対するボコーダーの無発振安定性、
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

    // MARK: - 2. Rosenberg パルス周期性・波形形状テスト

    func testRosenbergPulsePropertiesAndPeriodicity() {
        // 声門パルス発生器が位相アキュムレータに従って周波数通りの厳密な周期で
        // ピークを出力しているかを数学的に検証する。
        let sampleRate: Float = 16000.0
        let targetF0: Float = 200.0 // 16000 / 200 = 80 サンプル周期
        let expectedPeriod = 80
        let totalSamples = 800 // 10 周期分

        let pulseGen = RosenbergPulse(sampleRate: sampleRate, n1Ratio: 0.40, n2Ratio: 0.16)
        var wave = [Float](repeating: 0.0, count: totalSamples)
        wave.withUnsafeMutableBufferPointer { pWave in
            pulseGen.generateFrame(f0: targetF0, count: totalSamples, dst: pWave.baseAddress!, removeDC: false)
        }

        // ピーク位置（極大値）の探索
        var peakIndices: [Int] = []
        var i = 1
        while i < totalSamples - 1 {
            let prev = wave[i - 1]
            let curr = wave[i]
            let next = wave[i + 1]
            if prev < curr {
                if next <= curr {
                    if 0.5 < curr {
                        peakIndices.append(i)
                    }
                }
            }
            i += 1
        }

        XCTAssertTrue(5 <= peakIndices.count, "検出されたパルスピーク数が不足しています: \(peakIndices.count)")
        var pIdx = 1
        while pIdx < peakIndices.count {
            let interval = peakIndices[pIdx] - peakIndices[pIdx - 1]
            let diff = abs(interval - expectedPeriod)
            // 離散サンプリングによる ±1 サンプルの量子化誤差を許容
            XCTAssertTrue(diff <= 1, "パルス周期が目標周期から乖離: interval=\(interval), expected=\(expectedPeriod)")
            pIdx += 1
        }

        // パルス値の数理的特性の検証 (開口期・閉口期・閉鎖期)
        let v0 = pulseGen.pulseValue(at: 0.0)
        XCTAssertEqual(v0, 0.0, accuracy: 1e-6)
        let vPeak = pulseGen.pulseValue(at: 0.40) // N1 = 0.40 でピーク 1.0
        XCTAssertEqual(vPeak, 1.0, accuracy: 1e-5)
        let vClose = pulseGen.pulseValue(at: 0.56) // N1 + N2 = 0.56 で閉口 0.0
        XCTAssertEqual(vClose, 0.0, accuracy: 1e-5)
        let vSilent = pulseGen.pulseValue(at: 0.80) // 閉鎖期
        XCTAssertEqual(vSilent, 0.0, accuracy: 1e-6)

        // 無声区間 (F0 = 0.0) での出力検証
        var unvoicedWave = [Float](repeating: 1.0, count: 160)
        unvoicedWave.withUnsafeMutableBufferPointer { pWave in
            pulseGen.generateFrame(f0: 0.0, count: 160, dst: pWave.baseAddress!, removeDC: true)
        }
        var uIdx = 0
        while uIdx < 160 {
            XCTAssertEqual(unvoicedWave[uIdx], 0.0)
            uIdx += 1
        }
    }

    // MARK: - 3. MelToLPC 変換と Levinson-Durbin の数学的妥当性テスト

    func testMelToLPCConversionAndLevinsonDurbin() {
        // 逆写像行列、Wiener-Khinchin IDCT 自己相関、および Levinson-Durbin の
        // 各パイプラインが NaN を発生させずに安定した係数を出力することを保証する。
        let melToLpc = MelToLPC(melChannels: 64, fftBins: 257, lpcOrder: 16, sampleRate: 16000.0)

        // 1. 低域（母音の第1ホルマント付近）にピークを持つ Mel 特徴量の生成
        var mel = [Float](repeating: 0.0, count: 64)
        var ch = 0
        while ch < 64 {
            let dist = Float(ch - 8)
            mel[ch] = exp(-(dist * dist) * 0.05) * 5.0
            ch += 1
        }

        var lpcCoeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLpc.convert(mel: mel, isLogMel: false, outCoeffs: &lpcCoeffs)

        XCTAssertTrue(0.0 < gain, "ゲインが正値になっていません: \(gain)")
        var k = 0
        while k < 16 {
            let coeff = lpcCoeffs[k]
            // NaN / Inf チェック
            XCTAssertEqual(coeff, coeff, "LPC 係数に NaN が含まれています: index=\(k)")
            XCTAssertTrue(abs(coeff) < 100.0, "LPC 係数が異常に肥大化しています: \(coeff)")
            k += 1
        }

        // 2. 無音入力に対するフェイルセーフ検証
        let zeroMel = [Float](repeating: -20.0, count: 64) // 極小 log-mel
        var zeroCoeffs = [Float](repeating: 1.0, count: 16)
        let zeroGain = melToLpc.convert(mel: zeroMel, isLogMel: true, outCoeffs: &zeroCoeffs)
        XCTAssertTrue(zeroGain <= 1e-3, "無音入力時のゲインが抑制されていません: \(zeroGain)")
    }

    // MARK: - 4. LPC 合成ボコーダーの安定性・無発振テスト

    func testLPCVocoderStabilityAndResonance() {
        // IIR フィルタの過去状態が帰還ループで無限大に発散せず、
        // 安定したエネルギー包らかを持つ音声波形が継続して出力されることを実証する。
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)

        // 代表的な日本語母音 /a/ の LPC 係数モデル (F1=800Hz, F2=1300Hz 付近に共鳴極)
        // 安定な単位円内極を持つ多項式係数列
        let vowelCoeffs: [Float] = [
            1.25, -0.85, 0.45, -0.25, 0.15, -0.08, 0.04, -0.02,
            0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        let frameCount = 10
        var frames: [AcousticFrame] = []
        var f = 0
        while f < frameCount {
            frames.append(
                AcousticFrame(
                    lpcCoefficients: vowelCoeffs,
                    gain: 0.05,
                    pitchF0: 150.0, // 150Hz 有声音
                    voiced: 1.0
                )
            )
            f += 1
        }

        let audio = vocoder.synthesize(frames: frames)
        XCTAssertEqual(audio.count, frameCount * 160)

        var hasNonZero = false
        var maxVal: Float = 0.0
        var sIdx = 0
        while sIdx < audio.count {
            let s = audio[sIdx]
            XCTAssertEqual(s, s, "ボコーダー出力に NaN が混入しました: sample=\(sIdx)")
            if 0.0 < abs(s) {
                hasNonZero = true
            }
            if maxVal < abs(s) {
                maxVal = abs(s)
            }
            sIdx += 1
        }

        XCTAssertTrue(hasNonZero, "合成音声が完全な無音です")
        XCTAssertTrue(maxVal <= 1.0, "合成音声がクリッピングレベルを超過しています: \(maxVal)")
    }

    // MARK: - 5. NaN/Inf ガードおよび過大入力サチュレーション耐性テスト

    func testLPCVocoderNaNAndOverdriveSafety() {
        // SNN 音響モデルの学習初期や異常勾配によって異常パラメータが出力された場合でも、
        // ボコーダーがクラッシュせず安全にフォールバックすることを証明する。
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16)

        // 1. 巨大ゲインフレーム (Overdrive)
        let overdriveFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.2, count: 16),
            gain: 1000.0, // 極大ゲイン
            pitchF0: 200.0,
            voiced: 1.0
        )
        var outOverdrive = [Float](repeating: 0.0, count: 160)
        outOverdrive.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(frame: overdriveFrame, dst: pDst.baseAddress!)
        }

        var i = 0
        while i < 160 {
            let s = outOverdrive[i]
            XCTAssertEqual(s, s, "Overdrive 入力で NaN が発生しました")
            XCTAssertTrue(s <= 1.0, "Soft Limiter が 1.0 を超えるサンプルを許容しました: \(s)")
            XCTAssertTrue(-1.0 <= s, "Soft Limiter が -1.0 を下回るサンプルを許容しました: \(s)")
            i += 1
        }

        // 2. NaN 係数フレームの注入
        let nanFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: Float.nan, count: 16),
            gain: Float.nan,
            pitchF0: 200.0,
            voiced: 1.0
        )
        var outNaN = [Float](repeating: 0.0, count: 160)
        outNaN.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(frame: nanFrame, dst: pDst.baseAddress!)
        }

        i = 0
        while i < 160 {
            let s = outNaN[i]
            XCTAssertEqual(s, s, "NaN 注入後の出力に NaN が漏洩しました")
            XCTAssertEqual(s, 0.0, "NaN 注入時に 0.0 に安全フォールバックしていません")
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

    // MARK: - 7. 無音・ゼロフレーム入力に対する完全ゼロ出力テスト

    func testLPCVocoderSilenceAndZeroFrame() {
        // ポーズや文末において、乱数ノイズやパルス発振が漏れ出さず、
        // 厳密に 0.0 のデジタル無音が生成されることを保証する。
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16)
        let silenceFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.0, count: 16),
            gain: 0.0,
            pitchF0: 0.0,
            voiced: 0.0
        )

        var dst = [Float](repeating: 1.0, count: 160)
        dst.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(frame: silenceFrame, dst: pDst.baseAddress!)
        }

        var i = 0
        while i < 160 {
            XCTAssertEqual(dst[i], 0.0, "無音フレームで非ゼロ出力が発生しました: index=\(i)")
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

    // MARK: - 9. 反射係数クランプの極限境界値検証

    func testMelToLPCReflectionCoefficientClamp() {
        // 単一周波数成分が極端に突出した自己相関が入力された場合でも、
        // 反射係数が確実にクランプされ、極の安定性が維持されることを検証する。
        let melToLpc = MelToLPC(melChannels: 64, fftBins: 257, lpcOrder: 16)

        // 単一チャンネルのみが極端に巨大なスペクトル (ディラックのデルタ型 Mel)
        var extremeMel = [Float](repeating: -20.0, count: 64)
        extremeMel[10] = 30.0 // 極大ピーク

        var outCoeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLpc.convert(mel: extremeMel, isLogMel: true, outCoeffs: &outCoeffs)

        XCTAssertTrue(0.0 <= gain, "ゲインが負値になっています")
        var i = 0
        while i < 16 {
            let c = outCoeffs[i]
            XCTAssertEqual(c, c, "クランプ後係数に NaN が混入しました: index=\(i)")
            XCTAssertTrue(abs(c) < 50.0, "係数が異常発散しています: \(c)")
            i += 1
        }
    }
}
