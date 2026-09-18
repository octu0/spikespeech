import Foundation
import SpikeSpeech

/// Tier 4: 実環境 E2E テスト
func runTier4Tests() {
    print("--- Running Tier 4 Real-World E2E Tests ---")

    let engine = SpikeSpeechEngine()

    // 1. 実世界日本語対話文の合成
    let conversationTexts = [
        "こんにちは！本日の天気は晴れのち曇り、気温は24度です。",
        "お会計は、合計で3,450円になります。ポイントカードはお持ちですか？",
        "次の交差点を右折したあと、およそ300メートル直進してください。"
    ]

    var cIdx = 0
    while cIdx < conversationTexts.count {
        let text = conversationTexts[cIdx]
        let pcm = engine.synthesize(text: text)
        e2eAssertTrue(0 < pcm.count, "Tier4: real world dialogue \(cIdx) non-empty")

        var i = 0
        var allFinite = true
        while i < pcm.count {
            if pcm[i].isFinite != true {
                allFinite = false
            }
            i += 1
        }
        e2eAssertTrue(allFinite, "Tier4: dialogue \(cIdx) samples finite")
        cIdx += 1
    }

    // 2. 外部コーパス依存テスト（-d オプション必須、埋め込みパス禁止）
    if let corpusDir = E2ETestContext.shared.corpusDirectory {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: corpusDir) {
            print("  [Corpus] Evaluating corpus at: \(corpusDir)")
            let reader = WavAudioReader()
            let melExtractor = MelSpectrogramExtractor()
            let tracker = PitchTracker()

            // コーパス直下の WAV ファイルを最大 3 件検査
            if let enumerator = fileManager.enumerator(atPath: corpusDir) {
                var checkedCount = 0
                while let subPath = enumerator.nextObject() as? String {
                    if subPath.hasSuffix(".wav") && checkedCount < 3 {
                        let fullPath = corpusDir + "/" + subPath
                        do {
                            let pcm = try reader.loadWav16k(from: fullPath)
                            let mel = melExtractor.extractLogMel(pcm: pcm)
                            let pitch = tracker.track(pcm: pcm)
                            e2eAssertTrue(0 < pcm.count, "Tier4: corpus wav pcm non-empty")
                            e2eAssertTrue(0 < mel.count, "Tier4: corpus mel non-empty")
                            e2eAssertTrue(0 < pitch.frameCount, "Tier4: corpus pitch non-empty")
                            checkedCount += 1
                        } catch {
                            E2ETestContext.shared.recordFail("Tier4: failed to read corpus wav: \(fullPath)")
                        }
                    }
                }
                print("  [Corpus] Tested \(checkedCount) corpus files")
            }
        } else {
            E2ETestContext.shared.recordSkip("Corpus directory does not exist: \(corpusDir)")
        }
    } else {
        E2ETestContext.shared.recordSkip("No -d / --corpus-dir specified. Skipping corpus verification.")
    }
}
