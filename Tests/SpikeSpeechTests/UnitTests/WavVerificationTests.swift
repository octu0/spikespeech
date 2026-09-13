import XCTest
@testable import SpikeSpeech

/// 生成された WAV 音声の RIFF ヘッダ仕様および PCM サンプル品質の網羅的検証テストスイート
final class WavVerificationTests: XCTestCase {

    /// 合成された音声波形から WAV データを生成し、RIFF/WAVE フォーマットの規格適合性を検証する。
    func testWavFormatAndPcmIntegrity() {
        // 基準となる 1 秒間の 440Hz 正弦波テスト信号を生成する
        let sampleRate = 16000
        let sampleCount = 16000
        var samples = [Float](repeating: 0.0, count: sampleCount)
        var i = 0
        while i < sampleCount {
            let t = Float(i) / Float(sampleRate)
            samples[i] = sin(2.0 * Float.pi * 440.0 * t) * 0.5
            i += 1
        }

        let wavData = WavEncoder.encode(samples: samples, sampleRate: sampleRate)
        let expectedTotalSize = 44 + (sampleCount * 2)
        XCTAssertEqual(wavData.count, expectedTotalSize, "WAV データ長がヘッダと PCM データの合計サイズと一致しません")

        // 1. RIFF マジックナンバーの検証
        let riff = String(data: wavData.subdata(in: 0..<4), encoding: .ascii)
        XCTAssertEqual(riff, "RIFF", "RIFF マジックナンバーが一致しません")

        // 2. ChunkSize (ファイルサイズ - 8) の検証
        let chunkSize = wavData.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self).littleEndian }
        let expectedChunkSize = UInt32(wavData.count - 8)
        XCTAssertEqual(chunkSize, expectedChunkSize, "ChunkSize がファイルサイズ - 8 と一致しません")

        // 3. WAVE マジックナンバーの検証
        let wave = String(data: wavData.subdata(in: 8..<12), encoding: .ascii)
        XCTAssertEqual(wave, "WAVE", "WAVE マジックナンバーが一致しません")

        // 4. fmt チャンクの検証
        let fmt = String(data: wavData.subdata(in: 12..<16), encoding: .ascii)
        XCTAssertEqual(fmt, "fmt ", "fmt マジックナンバーが一致しません")

        let fmtSize = wavData.withUnsafeBytes { $0.load(fromByteOffset: 16, as: UInt32.self).littleEndian }
        XCTAssertEqual(fmtSize, 16, "PCM fmt チャンクサイズは 16 である必要があります")

        let audioFormat = wavData.withUnsafeBytes { $0.load(fromByteOffset: 20, as: UInt16.self).littleEndian }
        XCTAssertEqual(audioFormat, 1, "リニア PCM の AudioFormat は 1 である必要があります")

        let numChannels = wavData.withUnsafeBytes { $0.load(fromByteOffset: 22, as: UInt16.self).littleEndian }
        XCTAssertEqual(numChannels, 1, "モノラル音声のチャンネル数は 1 である必要があります")

        let rate = wavData.withUnsafeBytes { $0.load(fromByteOffset: 24, as: UInt32.self).littleEndian }
        XCTAssertEqual(rate, UInt32(sampleRate), "サンプリングレートが一致しません")

        let byteRate = wavData.withUnsafeBytes { $0.load(fromByteOffset: 28, as: UInt32.self).littleEndian }
        XCTAssertEqual(byteRate, UInt32(sampleRate * 2), "バイトレート (サンプリングレート * 2) が一致しません")

        let blockAlign = wavData.withUnsafeBytes { $0.load(fromByteOffset: 32, as: UInt16.self).littleEndian }
        XCTAssertEqual(blockAlign, 2, "16-bit モノラルの BlockAlign は 2 である必要があります")

        let bitsPerSample = wavData.withUnsafeBytes { $0.load(fromByteOffset: 34, as: UInt16.self).littleEndian }
        XCTAssertEqual(bitsPerSample, 16, "ビット深度は 16 である必要があります")

        // 5. data チャンクの検証
        let dataMagic = String(data: wavData.subdata(in: 36..<40), encoding: .ascii)
        XCTAssertEqual(dataMagic, "data", "data マジックナンバーが一致しません")

        let dataSize = wavData.withUnsafeBytes { $0.load(fromByteOffset: 40, as: UInt32.self).littleEndian }
        let expectedDataSize = UInt32(sampleCount * 2)
        XCTAssertEqual(dataSize, expectedDataSize, "data チャンクサイズがサンプルデータサイズと一致しません")

        // 6. PCM サンプルの統計量解析
        var decodedSamples = [Int16](repeating: 0, count: sampleCount)
        wavData.withUnsafeBytes { ptr in
            let base = ptr.baseAddress!.advanced(by: 44).assumingMemoryBound(to: Int16.self)
            decodedSamples.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: base, count: sampleCount)
            }
        }

        var minSample: Int16 = 0
        var maxSample: Int16 = 0
        var sumSq: Double = 0.0
        var clippingCount = 0

        var idx = 0
        while idx < sampleCount {
            let val = decodedSamples[idx]
            if maxSample < val {
                maxSample = val
            }
            if val < minSample {
                minSample = val
            }
            if val == 32767 || val == -32768 {
                clippingCount += 1
            }
            let f = Double(val) / 32768.0
            sumSq += f * f
            idx += 1
        }

        let rms = sqrt(sumSq / Double(sampleCount))
        XCTAssertEqual(clippingCount, 0, "クリッピングが発生してはならない")
        XCTAssertTrue(0.005 < rms, "無音ではなく十分な音響エネルギーが存在すること")
    }

    /// Phase B 監査で合成された実機 WAV ファイル群のヘッダ規格適合性および波形統計量検証
    func testPhaseBAuditedWavFilesCompliance() throws {
        let testCases: [(path: String, text: String)] = [
            ("/tmp/test_phase_b_1.wav", "こんにちは、音声合成の世界へようこそ。"),
            ("/tmp/test_phase_b_short.wav", "はい"),
            ("/tmp/test_phase_b_question.wav", "明日の天気はどうですか？"),
            ("/tmp/test_phase_b_sokuon.wav", "もっとゆっくり走ってください")
        ]

        let engine = SpikeSpeechEngine()
        let fileManager = FileManager.default

        var fIdx = 0
        while fIdx < testCases.count {
            let tc = testCases[fIdx]
            let path = tc.path
            let url = URL(fileURLWithPath: path)

            let data: Data
            if fileManager.fileExists(atPath: path) {
                data = try Data(contentsOf: url)
                try? fileManager.removeItem(at: url)
            } else {
                data = engine.synthesizeWav(text: tc.text)
            }

            XCTAssertTrue(44 <= data.count, "WAV ヘッダサイズ未満です: \(path)")

            // 1. RIFF
            let riff = String(data: data.subdata(in: 0..<4), encoding: .ascii)
            XCTAssertEqual(riff, "RIFF")

            let chunkSize = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self).littleEndian }
            XCTAssertEqual(chunkSize, UInt32(data.count - 8))

            // 2. WAVE
            let wave = String(data: data.subdata(in: 8..<12), encoding: .ascii)
            XCTAssertEqual(wave, "WAVE")

            // 3. fmt
            let fmt = String(data: data.subdata(in: 12..<16), encoding: .ascii)
            XCTAssertEqual(fmt, "fmt ")

            let fmtSize = data.withUnsafeBytes { $0.load(fromByteOffset: 16, as: UInt32.self).littleEndian }
            XCTAssertEqual(fmtSize, 16)

            let audioFormat = data.withUnsafeBytes { $0.load(fromByteOffset: 20, as: UInt16.self).littleEndian }
            XCTAssertEqual(audioFormat, 1)

            let numChannels = data.withUnsafeBytes { $0.load(fromByteOffset: 22, as: UInt16.self).littleEndian }
            XCTAssertEqual(numChannels, 1)

            let sampleRate = data.withUnsafeBytes { $0.load(fromByteOffset: 24, as: UInt32.self).littleEndian }
            XCTAssertEqual(sampleRate, 16000)

            let byteRate = data.withUnsafeBytes { $0.load(fromByteOffset: 28, as: UInt32.self).littleEndian }
            XCTAssertEqual(byteRate, 32000)

            let blockAlign = data.withUnsafeBytes { $0.load(fromByteOffset: 32, as: UInt16.self).littleEndian }
            XCTAssertEqual(blockAlign, 2)

            let bitsPerSample = data.withUnsafeBytes { $0.load(fromByteOffset: 34, as: UInt16.self).littleEndian }
            XCTAssertEqual(bitsPerSample, 16)

            // 4. data
            let dataMagic = String(data: data.subdata(in: 36..<40), encoding: .ascii)
            XCTAssertEqual(dataMagic, "data")

            let dataSize = data.withUnsafeBytes { $0.load(fromByteOffset: 40, as: UInt32.self).littleEndian }
            XCTAssertEqual(dataSize, UInt32(data.count - 44))

            // 5. PCM 統計量解析
            let sampleCount = Int(dataSize / 2)
            XCTAssertTrue(0 < sampleCount)

            var decodedSamples = [Int16](repeating: 0, count: sampleCount)
            data.withUnsafeBytes { ptr in
                let base = ptr.baseAddress!.advanced(by: 44).assumingMemoryBound(to: Int16.self)
                decodedSamples.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.update(from: base, count: sampleCount)
                }
            }

            var minSample: Int16 = 0
            var maxSample: Int16 = 0
            var sumSq: Double = 0.0
            var clippingCount = 0

            var idx = 0
            while idx < sampleCount {
                let val = decodedSamples[idx]
                if maxSample < val {
                    maxSample = val
                }
                if val < minSample {
                    minSample = val
                }
                if val == 32767 || val == -32768 {
                    clippingCount += 1
                }
                let f = Double(val) / 32768.0
                sumSq += f * f
                idx += 1
            }

            let rms = sqrt(sumSq / Double(sampleCount))
            print("--- [Phase B Wav Statistics: \(path)] ---")
            print("サンプル数: \(sampleCount)")
            print("最小サンプル値: \(minSample), 最大サンプル値: \(maxSample)")
            print("クリッピング数: \(clippingCount)")
            print("RMS エネルギー: \(rms)")

            // 無音 (RMS == 0) の不発生
            XCTAssertTrue(0.001 < rms, "無音（エネルギー不足）を検出しました: \(path), rms=\(rms)")
            // クリッピング (過大振幅) の不発生
            XCTAssertEqual(clippingCount, 0, "クリッピングが発生しました: \(path)")

            fIdx += 1
        }
    }
}
