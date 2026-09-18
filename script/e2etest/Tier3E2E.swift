import Foundation
import SpikeSpeech

/// Tier 3: 組み合わせ E2E テスト
func runTier3Tests() {
    print("--- Running Tier 3 Combination E2E Tests ---")

    let engine = SpikeSpeechEngine()

    // 1. 話速 × ピッチの組み合わせ合成
    let speeds: [Float] = [0.5, 1.0, 1.8]
    let pitches: [Float] = [0.8, 1.0, 1.3]

    var sIdx = 0
    while sIdx < speeds.count {
        let spd = speeds[sIdx]
        var pIdx = 0
        while pIdx < pitches.count {
            let pit = pitches[pIdx]
            let pcm = engine.synthesize(text: "テスト音声", speed: spd, pitch: pit)
            e2eAssertTrue(0 < pcm.count, "Tier3: speed=\(spd) pitch=\(pit) non-empty")
            pIdx += 1
        }
        sIdx += 1
    }

    // 2. 話者プロファイル × テキストパターンの組み合わせ
    let profiles: [VoiceProfile] = [.female, .male, .neutral, .child, .deepMale]
    let texts: [String] = ["あ", "おはようございます。", "12345円です。"]

    var prIdx = 0
    while prIdx < profiles.count {
        let prof = profiles[prIdx]
        var tIdx = 0
        while tIdx < texts.count {
            let txt = texts[tIdx]
            let wav = engine.synthesizeWav(text: txt, voice: prof)
            e2eAssertTrue(44 < wav.count, "Tier3: profile=\(prof.name) text=\(txt) produces valid wav")
            tIdx += 1
        }
        prIdx += 1
    }

    // 3. 多層 SNN 層数 × 発話の組み合わせ
    let layerConfigs = [1, 2, 3]
    var lIdx = 0
    while lIdx < layerConfigs.count {
        let layers = layerConfigs[lIdx]
        let weights = SpikingNetworkWeights.randomWeights(numLayers: layers)
        let multiEngine = SpikeSpeechEngine(weights: weights)
        let wav = multiEngine.synthesizeWav(text: "多層ニューラルネットワーク")
        e2eAssertTrue(44 < wav.count, "Tier3: layers=\(layers) synthesis")
        lIdx += 1
    }
}
