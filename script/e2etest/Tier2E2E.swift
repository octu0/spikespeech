import Foundation
import SpikeSpeech

/// Tier 2: 境界値 E2E テスト
func runTier2Tests() {
    print("--- Running Tier 2 Boundary E2E Tests ---")

    let engine = SpikeSpeechEngine()
    let vocoder = NeuralVocoder()
    let regulator = LengthRegulator()

    // 1. 空文字合成
    let emptyWav = engine.synthesizeWav(text: "")
    e2eAssertEqual(emptyWav.count, 44, "Tier2: empty text produces 44 bytes header")

    let spaceWav = engine.synthesizeWav(text: "     ")
    e2eAssertTrue(44 <= spaceWav.count, "Tier2: spaces produce valid header")

    // 2. 極端な話速指定クランプ
    let fastDur = regulator.phonemeDuration(phoneId: 5, speedFactor: 100.0)
    e2eAssertTrue(1.0 <= fastDur, "Tier2: extreme fast speed clamped to 1.0 frame min")

    let slowDur = regulator.phonemeDuration(phoneId: 5, speedFactor: -5.0)
    e2eAssertTrue(1.0 <= slowDur, "Tier2: negative speed clamped safely")

    let nanDur = regulator.phonemeDuration(phoneId: 5, speedFactor: Float.nan)
    e2eAssertFalse(nanDur.isNaN, "Tier2: NaN speed does not return NaN")

    // 3. ボコーダー NaN / Inf 入力耐性
    vocoder.reset()
    var nanMel = [Float](repeating: -2.0, count: 64)
    nanMel[0] = Float.nan
    nanMel[1] = Float.infinity
    nanMel[2] = -Float.infinity

    let outPcm = vocoder.synthesize(mel: [nanMel])
    e2eAssertEqual(outPcm.count, 160, "Tier2: vocoder produces 160 samples on NaN frame")
    var i = 0
    var allFinite = true
    while i < outPcm.count {
        if outPcm[i].isFinite != true {
            allFinite = false
        }
        i += 1
    }
    e2eAssertTrue(allFinite, "Tier2: all vocoder output samples are finite")

    // 4. WavEncoder ゼロサンプル
    let zeroWav = WavEncoder.encode(samples: [], sampleRate: 16000)
    e2eAssertEqual(zeroWav.count, 44, "Tier2: WavEncoder encodes empty samples to 44 bytes")

    // 5. WavStreamWriter 複数回 finalize 耐性
    let tempDir = FileManager.default.temporaryDirectory
    let tempFile = tempDir.appendingPathComponent("tier2_test_\(UUID().uuidString).wav")
    FileManager.default.createFile(atPath: tempFile.path, contents: nil)

    do {
        let handle = try FileHandle(forWritingTo: tempFile)
        let writer = try WavStreamWriter(fileHandle: handle, sampleRate: 16000)
        try writer.finalize()
        try writer.finalize()
        try handle.close()

        let writtenData = try Data(contentsOf: tempFile)
        e2eAssertEqual(writtenData.count, 44, "Tier2: WavStreamWriter double finalize produces 44 bytes")
        try? FileManager.default.removeItem(at: tempFile)
    } catch {
        E2ETestContext.shared.recordFail("Tier2: WavStreamWriter threw: \(error)")
    }
}
