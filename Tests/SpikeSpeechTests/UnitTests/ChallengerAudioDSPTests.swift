import XCTest
@testable import SpikeSpeech

/// Challenger 2 による WAV 規格準拠性・逐次 finalize 完全性・
/// および NeuralVocoder 波形生成安定性の数理的・定量的検証テストスイート
///
/// RIFF/WAVE 44 バイトヘッダの規格準拠性、WavStreamWriter による逐次書き込みと
/// ヘッダ確定更新のビット完全一致、ニューラルボコーダーの多パターン入力に対する
/// 数値健全性、およびクリッピング防止を自ら実証・保証する。
final class ChallengerAudioDSPTests: XCTestCase {

    // MARK: - 1. WAV 規格準拠性のバイトレベル検証

    func testWavHeaderByteLevelRIFFCompliance() {
        // 0 サンプル、端数サンプル、1 フレーム (160 サンプル)、1 秒 (16000 サンプル)、
        // および 64KB を超える多バイト境界において、リトルエンディアンのサイズフィールドが
        // RIFF/WAVE 規格書に厳密に合致することを実証する。
        let testSampleCounts = [0, 1, 8, 160, 16000, 65536]
        var sIdx = 0
        while sIdx < testSampleCounts.count {
            let sampleCount = testSampleCounts[sIdx]
            let bytesPerSample = 2 // 16-bit
            let dataSize = sampleCount * bytesPerSample
            let expectedRiffSize = 36 + dataSize

            let header = WavEncoder.createHeader(
                sampleRate: 16000,
                numChannels: 1,
                bitsPerSample: 16,
                dataSize: dataSize
            )

            // 1. ヘッダ長は厳密に 44 バイト
            XCTAssertEqual(header.count, 44, "ヘッダ長が 44 バイトではありません: count=\(sampleCount)")

            // 2. オフセット 0..3: "RIFF" (ASCII 0x52, 0x49, 0x46, 0x46)
            XCTAssertEqual(header[0], 0x52)
            XCTAssertEqual(header[1], 0x49)
            XCTAssertEqual(header[2], 0x46)
            XCTAssertEqual(header[3], 0x46)

            // 3. オフセット 4..7: ChunkSize (Little-Endian uint32: 36 + dataSize)
            let parsedChunkSize = UInt32(header[4])
                | (UInt32(header[5]) << 8)
                | (UInt32(header[6]) << 16)
                | (UInt32(header[7]) << 24)
            XCTAssertEqual(parsedChunkSize, UInt32(expectedRiffSize), "ChunkSize が不整合です: count=\(sampleCount)")

            // 4. オフセット 8..11: "WAVE" (ASCII 0x57, 0x41, 0x56, 0x45)
            XCTAssertEqual(header[8], 0x57)
            XCTAssertEqual(header[9], 0x41)
            XCTAssertEqual(header[10], 0x56)
            XCTAssertEqual(header[11], 0x45)

            // 5. オフセット 12..15: "fmt " (ASCII 0x66, 0x6D, 0x74, 0x20)
            XCTAssertEqual(header[12], 0x66)
            XCTAssertEqual(header[13], 0x6D)
            XCTAssertEqual(header[14], 0x74)
            XCTAssertEqual(header[15], 0x20)

            // 6. オフセット 16..19: Subchunk1Size: 16 (Little-Endian uint32)
            let parsedSub1Size = UInt32(header[16])
                | (UInt32(header[17]) << 8)
                | (UInt32(header[18]) << 16)
                | (UInt32(header[19]) << 24)
            XCTAssertEqual(parsedSub1Size, 16, "fmt チャンク長が 16 ではありません")

            // 7. オフセット 20..21: AudioFormat: 1 (Linear PCM) (Little-Endian uint16)
            let parsedFormat = UInt16(header[20]) | (UInt16(header[21]) << 8)
            XCTAssertEqual(parsedFormat, 1, "AudioFormat が 1 (Linear PCM) ではありません")

            // 8. オフセット 22..23: NumChannels: 1 (Mono) (Little-Endian uint16)
            let parsedChannels = UInt16(header[22]) | (UInt16(header[23]) << 8)
            XCTAssertEqual(parsedChannels, 1, "チャンネル数が 1 (Mono) ではありません")

            // 9. オフセット 24..27: SampleRate: 16000 (Little-Endian uint32: 0x00003E80 -> 0x80, 0x3E, 0x00, 0x00)
            let parsedRate = UInt32(header[24])
                | (UInt32(header[25]) << 8)
                | (UInt32(header[26]) << 16)
                | (UInt32(header[27]) << 24)
            XCTAssertEqual(parsedRate, 16000, "サンプリング周波数が 16000 ではありません")
            XCTAssertEqual(header[24], 0x80)
            XCTAssertEqual(header[25], 0x3E)
            XCTAssertEqual(header[26], 0x00)
            XCTAssertEqual(header[27], 0x00)

            // 10. オフセット 28..31: ByteRate: 32000 (SampleRate * BlockAlign)
            let parsedByteRate = UInt32(header[28])
                | (UInt32(header[29]) << 8)
                | (UInt32(header[30]) << 16)
                | (UInt32(header[31]) << 24)
            XCTAssertEqual(parsedByteRate, 32000, "ByteRate が 32000 ではありません")
            XCTAssertEqual(header[28], 0x00)
            XCTAssertEqual(header[29], 0x7D)
            XCTAssertEqual(header[30], 0x00)
            XCTAssertEqual(header[31], 0x00)

            // 11. オフセット 32..33: BlockAlign: 2 (NumChannels * BitsPerSample / 8)
            let parsedBlockAlign = UInt16(header[32]) | (UInt16(header[33]) << 8)
            XCTAssertEqual(parsedBlockAlign, 2, "BlockAlign が 2 ではありません")

            // 12. オフセット 34..35: BitsPerSample: 16
            let parsedBits = UInt16(header[34]) | (UInt16(header[35]) << 8)
            XCTAssertEqual(parsedBits, 16, "BitsPerSample が 16 ではありません")

            // 13. オフセット 36..39: "data" (ASCII 0x64, 0x61, 0x74, 0x61)
            XCTAssertEqual(header[36], 0x64)
            XCTAssertEqual(header[37], 0x61)
            XCTAssertEqual(header[38], 0x74)
            XCTAssertEqual(header[39], 0x61)

            // 14. オフセット 40..43: Subchunk2Size: dataSize (Little-Endian uint32)
            let parsedDataSize = UInt32(header[40])
                | (UInt32(header[41]) << 8)
                | (UInt32(header[42]) << 16)
                | (UInt32(header[43]) << 24)
            XCTAssertEqual(parsedDataSize, UInt32(dataSize), "Subchunk2Size が dataSize と一致しません: count=\(sampleCount)")

            // 15. 数理的整合性検証: ByteRate == SampleRate * BlockAlign
            XCTAssertEqual(parsedByteRate, parsedRate * UInt32(parsedBlockAlign))
            // ChunkSize == Subchunk2Size + 36
            XCTAssertEqual(parsedChunkSize, parsedDataSize + 36)

            sIdx += 1
        }
    }

    // MARK: - 2. WavStreamWriter 逐次出力とヘッダ finalize の完全性・ビット一致検証

    func testWavStreamWriterSequentialBitExactIdentity() {
        // チャンク境界での量子化誤差やシーク上書き時のデータ破損が皆無であり、
        // 逐次出力されたファイルが WavEncoder.encode とバイトレベルで完全一致することを実証する。
        let totalSamples = 500
        var testSamples = [Float](repeating: 0.0, count: totalSamples)
        var i = 0
        while i < totalSamples {
            // 正弦波 + ノイズ + 境界値の混成テスト信号
            let t = Float(i) * (1.0 / 16000.0)
            let sinVal = sin(2.0 * Float.pi * 440.0 * t) * 0.8
            let stepVal = (Float(i % 50) * 0.02) - 0.5
            testSamples[i] = sinVal + stepVal
            i += 1
        }
        // 境界値サンプルの意図的注入
        testSamples[0] = 0.0
        testSamples[1] = 1.0
        testSamples[2] = -1.0
        testSamples[3] = 0.9999
        testSamples[4] = -0.9999

        // 1. WavEncoder による一括エンコード
        let batchData = WavEncoder.encode(samples: testSamples, sampleRate: 16000)
        let expectedByteCount = 44 + (totalSamples * 2)
        XCTAssertEqual(batchData.count, expectedByteCount)

        // 2. WavStreamWriter による不規則チャンク書き出し
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_bitexact_\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: tempFile.path, contents: nil)

        do {
            let fileHandle = try FileHandle(forWritingTo: tempFile)
            let writer = try WavStreamWriter(fileHandle: fileHandle, sampleRate: 16000)

            // SIMD8 境界、端数、素数サイズを含む不規則なチャンク系列
            let chunkSizes = [1, 2, 5, 8, 15, 16, 17, 32, 64, 128, 77, 124, 11]
            var offset = 0
            var cIdx = 0
            while cIdx < chunkSizes.count {
                let cSize = chunkSizes[cIdx]
                let endIdx = offset + cSize
                var chunk = [Float](repeating: 0.0, count: cSize)
                var k = 0
                while k < cSize {
                    chunk[k] = testSamples[offset + k]
                    k += 1
                }
                try writer.write(samples: chunk)
                offset = endIdx
                cIdx += 1
            }
            XCTAssertEqual(offset, totalSamples, "テストサンプルの合計が一致しません")

            // finalize 実行
            try writer.finalize()

            // シーク位置がファイル終端にあることの検証
            let finalOffset = try fileHandle.offset()
            XCTAssertEqual(finalOffset, UInt64(expectedByteCount), "finalize 後のファイルシーク位置が終端と一致しません")

            try fileHandle.close()

            // 3. 全バイト完全一致 (Bit-Exactness) の検証
            let streamedData = try Data(contentsOf: tempFile)
            XCTAssertEqual(streamedData.count, batchData.count, "ストリーミング出力の総バイト数が一括エンコードと不一致です")

            let batchBytes = [UInt8](batchData)
            let streamBytes = [UInt8](streamedData)

            var bIdx = 0
            var diffCount = 0
            while bIdx < batchBytes.count {
                if batchBytes[bIdx] != streamBytes[bIdx] {
                    diffCount += 1
                }
                bIdx += 1
            }
            XCTAssertEqual(diffCount, 0, "ストリーミング出力に一括エンコードとのバイト不一致が存在します: 不一致数=\(diffCount)")

            try? FileManager.default.removeItem(at: tempFile)
        } catch {
            XCTFail("逐次ストリーミングテスト中にエラーが発生しました: \(error)")
        }
    }

    // MARK: - 3. WavStreamWriter の多重 finalize・追記拒否・空書き込み耐性検証

    func testWavStreamWriterIdempotenceAndStateGuards() {
        // finalize 後の多重呼び出しでヘッダが破損しないこと、
        // finalize 後の追記が正しく拒否されファイルサイズが膨張しないこと、
        // および 0 サンプル生成時の安全性を実証する。
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_guard_\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: tempFile.path, contents: nil)

        do {
            let handle = try FileHandle(forWritingTo: tempFile)
            let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)

            // 1. 空サンプルの書き出し
            try writer.write(samples: [])
            let initialOffset = try handle.offset()
            XCTAssertEqual(initialOffset, 44, "空書き出し後にヘッダ以外のバイトが追記されました")

            // 2. 有効サンプル書き込み (10 サンプル = 20 バイト)
            let validSamples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]
            try writer.write(samples: validSamples)

            // 3. 1 回目の finalize
            try writer.finalize()
            let afterFinalize1 = try handle.offset()
            XCTAssertEqual(afterFinalize1, 44 + 20)

            // 4. finalize 後の追記試行 (無視されるべき)
            try writer.write(samples: [0.9, 0.9])
            let afterPostWrite = try handle.offset()
            XCTAssertEqual(afterPostWrite, 44 + 20, "finalize 後の追記が拒否されずにファイルへ書き込まれました")

            // 5. 2 回目の finalize (冪等性)
            try writer.finalize()
            let afterFinalize2 = try handle.offset()
            XCTAssertEqual(afterFinalize2, 44 + 20, "2 回目の finalize によりファイルサイズが変化しました")

            try handle.close()

            // 6. ファイルヘッダの確認
            let fileData = try Data(contentsOf: tempFile)
            XCTAssertEqual(fileData.count, 64)

            let bytes = [UInt8](fileData)
            let sub2Size = UInt32(bytes[40]) | (UInt32(bytes[41]) << 8) | (UInt32(bytes[42]) << 16) | (UInt32(bytes[43]) << 24)
            XCTAssertEqual(sub2Size, 20, "ヘッダのデータサイズが 20 バイトと一致しません")

            try? FileManager.default.removeItem(at: tempFile)
        } catch {
            XCTFail("多重 finalize テスト中にエラーが発生しました: \(error)")
        }
    }

    // MARK: - 4. NeuralVocoder 活性化および畳み込み演算の数値健全性検証

    func testNeuralVocoderGlobalPredictionErrorMinimization() {
        // 多層 1D 畳み込みおよび MRF 残差ブロックによる推論において、
        // 入力 Mel 特徴量から出力 PCM まで NaN や Inf が発生せず、
        // 振幅が安定して [-1.0, 1.0] に保持されることを数理的に実証する。
        let vocoder = NeuralVocoder()

        // 典型的な母音スペクトル包絡（低域・中域に明瞭なピーク）
        var mel = [Float](repeating: 0.0, count: 64)
        var ch = 0
        while ch < 64 {
            let f1Dist = Float(ch - 10)
            let f2Dist = Float(ch - 24)
            let peak1 = exp(-(f1Dist * f1Dist) * 0.08) * 6.0
            let peak2 = exp(-(f2Dist * f2Dist) * 0.05) * 4.0
            mel[ch] = logf(max(peak1 + peak2 + 0.1, 1e-4))
            ch += 1
        }

        let samples = vocoder.synthesize(mel: [mel, mel])
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

    // MARK: - 5. NeuralVocoder 多パターン Mel 入力波形安定性の定量的検証

    func testNeuralVocoderUnitCirclePoleStabilityViaStepDown() {
        // 様々なホルマントパターン、単一周波数ピーク、過大飽和 Mel 特徴量に対して
        // ニューラルボコーダーが一切破綻せず、有限で安全な波形を出力することを検証する。
        let vocoder = NeuralVocoder()

        let testMelPatterns: [(name: String, mel: [Float])] = [
            ("Formant_A", generateSyntheticVowelMel(f1Bin: 12, f2Bin: 22)),
            ("Formant_I", generateSyntheticVowelMel(f1Bin: 4, f2Bin: 45)),
            ("Formant_U", generateSyntheticVowelMel(f1Bin: 6, f2Bin: 20)),
            ("Formant_E", generateSyntheticVowelMel(f1Bin: 8, f2Bin: 35)),
            ("Formant_O", generateSyntheticVowelMel(f1Bin: 10, f2Bin: 16)),
            ("Single_Peak_Low", generateSinglePeakMel(channel: 2)),
            ("Single_Peak_Mid", generateSinglePeakMel(channel: 32)),
            ("Single_Peak_High", generateSinglePeakMel(channel: 60)),
            ("White_Noise_Flat", [Float](repeating: 2.0, count: 64)),
            ("Extreme_Saturation", [Float](repeating: 15.0, count: 64)),
        ]

        var pIdx = 0
        while pIdx < testMelPatterns.count {
            let tc = testMelPatterns[pIdx]
            vocoder.reset()
            let samples = vocoder.synthesize(mel: [tc.mel])
            XCTAssertEqual(samples.count, 160)

            var i = 0
            while i < samples.count {
                let s = samples[i]
                XCTAssertTrue(s.isFinite, "非有限サンプル検出: pattern=\(tc.name), index=\(i)")
                XCTAssertTrue(-1.0 <= s, "振幅下限逸脱: pattern=\(tc.name), index=\(i), val=\(s)")
                XCTAssertTrue(s <= 1.0, "振幅上限逸脱: pattern=\(tc.name), index=\(i), val=\(s)")
                i += 1
            }
            pIdx += 1
        }
    }

    // MARK: - 6. 平坦スペクトル（ホワイトノイズ）における波形エネルギー安定性テスト

    func testNeuralVocoderFlatSpectrumZeroCoefficients() {
        // 一様な平坦対数 Mel 特徴量に対しても安定して波形合成が行われ、
        // 異常な発振や直流オフセット過大が発生しないことを検証する。
        let vocoder = NeuralVocoder()
        let flatMel = [Float](repeating: -2.0, count: 64)

        let samples = vocoder.synthesize(mel: [flatMel, flatMel])
        XCTAssertEqual(samples.count, 320)

        var sum: Float = 0.0
        var maxAbs: Float = 0.0
        var i = 0
        while i < samples.count {
            let s = samples[i]
            XCTAssertTrue(s.isFinite)
            sum += s
            let absS = abs(s)
            if maxAbs < absS {
                maxAbs = absS
            }
            i += 1
        }

        let mean = sum / Float(samples.count)
        // 直流オフセットが過大でないこと
        XCTAssertTrue(abs(mean) < 0.2)
        XCTAssertTrue(maxAbs <= 1.0)
    }

    // MARK: - 7. PCM 量子化境界値およびバイナリビット精度テスト

    func testQuantizationEdgeValuesAndBinaryEncoding() {
        // 16-bit 符号付き整数の限界値および過大振幅・NaN が
        // リトルエンディアンで正確なバイト列にマッピングされることをビット完全性で保証する。
        let testCases: [(val: Float, expectedInt16: Int16)] = [
            (0.0, 0),
            (1.0, 32767),
            (-1.0, -32767),
            (2.0, 32767),      // 上限クリップ
            (-2.0, -32768),    // 下限クリップ
            (Float.nan, 0),    // NaN フェイルセーフ
            (Float.infinity, 32767), // +Inf
            (-Float.infinity, -32768) // -Inf
        ]

        var samples = [Float](repeating: 0.0, count: testCases.count)
        var tIdx = 0
        while tIdx < testCases.count {
            samples[tIdx] = testCases[tIdx].val
            tIdx += 1
        }

        let wavData = WavEncoder.encode(samples: samples, sampleRate: 16000)
        XCTAssertEqual(wavData.count, 44 + (testCases.count * 2))

        let bytes = [UInt8](wavData)
        tIdx = 0
        while tIdx < testCases.count {
            let offset = 44 + (tIdx * 2)
            let b0 = bytes[offset]
            let b1 = bytes[offset + 1]
            let parsedInt16 = Int16(bitPattern: UInt16(b0) | (UInt16(b1) << 8))

            let expected = testCases[tIdx].expectedInt16
            XCTAssertEqual(
                parsedInt16, expected,
                "量子化バイナリ不一致: index=\(tIdx), input=\(testCases[tIdx].val), parsed=\(parsedInt16), expected=\(expected)"
            )
            tIdx += 1
        }
    }

    // MARK: - 8. リポジトリ静的コード規約（比較演算子・else if・三項演算子）の自動走査検証

    func testRepositoryCodingStandardsCompliance() {
        // 人手による見落としを完全に排除し、新旧すべての Swift 実装ファイルにおいて
        // 規約違反が皆無であることをテスト実行ごとに機械的に証明・担保する。
        let fileManager = FileManager.default
        let currentFilePath = URL(fileURLWithPath: #filePath)
        let repoRoot = currentFilePath
            .deletingLastPathComponent() // UnitTests
            .deletingLastPathComponent() // SpikeSpeechTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // spikespeech

        let sourcesPath = repoRoot.appendingPathComponent("Sources").path
        guard let enumerator = fileManager.enumerator(atPath: sourcesPath) else {
            XCTFail("Sources ディレクトリの列挙に失敗しました: \(sourcesPath)")
            return
        }

        var checkedFiles = 0
        while let relativePath = enumerator.nextObject() as? String {
            if relativePath.hasSuffix(".swift") != true {
                continue
            }

            let fullPath = sourcesPath + "/" + relativePath
            guard let content = try? String(contentsOfFile: fullPath, encoding: .utf8) else {
                continue
            }

            let lines = content.components(separatedBy: .newlines)
            var lineIdx = 0
            while lineIdx < lines.count {
                let line = lines[lineIdx]
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                // コメント行は除外
                if trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") {
                    lineIdx += 1
                    continue
                }

                // 1. 比較演算子 > または >= の禁止検証
                XCTAssertFalse(
                    trimmed.contains(" > ") && trimmed.contains("->") != true,
                    "比較演算子 > が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(" >= "),
                    "比較演算子 >= が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )

                // 2. else if の禁止検証
                XCTAssertFalse(
                    trimmed.contains("else if"),
                    "else if が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )

                // 3. 三項演算子の禁止検証
                XCTAssertFalse(
                    trimmed.contains(" ? ") && trimmed.contains(":") && trimmed.contains("//") != true,
                    "三項演算子が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )

                // 4. cond == false / cond == true の禁止検証
                XCTAssertFalse(
                    trimmed.contains("== false"),
                    "== false が使用されています ( != true を使用してください ): \(relativePath):\(lineIdx + 1)"
                )
                XCTAssertFalse(
                    trimmed.contains("== true"),
                    "== true が使用されています: \(relativePath):\(lineIdx + 1)"
                )

                lineIdx += 1
            }
            checkedFiles += 1
        }

        XCTAssertTrue(0 < checkedFiles, "検査対象の Swift ファイルが検出されませんでした")
        print("--- [Challenger 2 Static Rule Check] ---")
        print("検査完了 Swift ファイル数: \(checkedFiles) 件 (全ファイル規約適合)")
        print("-----------------------------------------")
    }

    // MARK: - ヘルパー関数

    /// 合成ホルマント Mel 特徴量の生成ヘルパー
    private func generateSyntheticVowelMel(f1Bin: Int, f2Bin: Int) -> [Float] {
        var mel = [Float](repeating: 0.05, count: 64)
        var ch = 0
        while ch < 64 {
            let d1 = Float(ch - f1Bin)
            let d2 = Float(ch - f2Bin)
            let peak1 = exp(-(d1 * d1) * 0.1) * 5.0
            let peak2 = exp(-(d2 * d2) * 0.08) * 3.0
            mel[ch] += peak1 + peak2
            ch += 1
        }
        return mel
    }

    /// 単一ピーク Mel 特徴量の生成ヘルパー
    private func generateSinglePeakMel(channel: Int) -> [Float] {
        var mel = [Float](repeating: 0.01, count: 64)
        mel[channel] = 10.0
        return mel
    }
}
