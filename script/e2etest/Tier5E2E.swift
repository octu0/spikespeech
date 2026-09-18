import Foundation
import SpikeSpeech

/// Tier 5: 敵対的 (Adversarial) E2E テスト
func runTier5Tests() {
    print("--- Running Tier 5 Adversarial E2E Tests ---")

    let engine = SpikeSpeechEngine()

    // 1. 絵文字や特殊記号混在テキスト
    let adversarialTexts = [
        "ラーメン🍜最高！(笑) www 100%オススメ✨",
        "★☆◆◇【超特価】9,999,999,999,999円！！！",
        "!@#$%^&*()_+|~-=`{}[]:\";'<>?,./",
        "αβγδε こんにちは 12345 漢字カタカナひらがな",
        "・・・・・・・・・・・・"
    ]

    var aIdx = 0
    while aIdx < adversarialTexts.count {
        let text = adversarialTexts[aIdx]
        let pcm = engine.synthesize(text: text)
        var i = 0
        var allFinite = true
        while i < pcm.count {
            if pcm[i].isFinite != true {
                allFinite = false
            }
            i += 1
        }
        e2eAssertTrue(allFinite, "Tier5: adversarial text \(aIdx) produces finite samples without crashing")
        aIdx += 1
    }

    // 2. 極端なパラメータの組み合わせ
    let extremeWav1 = engine.synthesizeWav(text: "急いで話します", speed: 5.0, pitch: 2.0)
    e2eAssertTrue(44 <= extremeWav1.count, "Tier5: max speed and pitch synthesis")

    let extremeWav2 = engine.synthesizeWav(text: "ゆっくり話します", speed: 0.2, pitch: 0.5)
    e2eAssertTrue(44 <= extremeWav2.count, "Tier5: min speed and pitch synthesis")
}
