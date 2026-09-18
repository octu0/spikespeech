import Foundation
import SpikeSpeech

/// Phase C: 敵対的ストレステスト
func runPhaseCTests() {
    print("--- Running Phase C Adversarial E2E Tests ---")

    let engine = SpikeSpeechEngine()
    let vocoder = NeuralVocoder()

    // 1. 異常対数 Mel の直接合成サニティ
    vocoder.reset()
    let frameCount = 8
    var corruptedFrames: [[Float]] = []
    var f = 0
    while f < frameCount {
        var row = [Float](repeating: -3.0, count: 64)
        if f == 2 {
            row[5] = Float.nan
        }
        if f == 4 {
            row[10] = Float.infinity
        }
        if f == 6 {
            row[15] = -Float.infinity
        }
        corruptedFrames.append(row)
        f += 1
    }

    let pcm = vocoder.synthesize(mel: corruptedFrames)
    e2eAssertEqual(pcm.count, frameCount * 160, "PhaseC: output count matches hop size")

    var i = 0
    var allFinite = true
    while i < pcm.count {
        if pcm[i].isFinite != true {
            allFinite = false
        }
        i += 1
    }
    e2eAssertTrue(allFinite, "PhaseC: vocoder NaN/Inf recovery produces finite samples")

    // 2. 連続合成時のメモリおよび状態の完全リセット
    let text = "あいうえお"
    let run1 = engine.synthesize(text: text)
    let run2 = engine.synthesize(text: text)
    e2eAssertEqual(run1.count, run2.count, "PhaseC: deterministic sample count")
    e2eAssertEqual(run1, run2, "PhaseC: state non-pollution identical waveform")
}
