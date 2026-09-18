import XCTest
import Foundation
@testable import SpikeSpeech

/// PitchTracker（基本周波数 F0 および有声判定）の単体テストスイート
final class PitchTrackerTests: XCTestCase {

    /// 既知の合成正弦波（150Hz, 220Hz, 330Hz, 440Hz）に対する F0 抽出精度の厳格検証
    ///
    /// 放物線補間が機能し、離散サンプリングの限界を超えて誤差 ±2.0Hz 以内で
    /// 真の基本周波数を正確に特定できることを数学的に検証する。
    func testPureSineToneTracking() {
        let tracker = PitchTracker(sampleRate: 16000.0)
        let testFrequencies: [Float] = [150.0, 220.0, 330.0, 440.0]
        let sampleRate: Float = 16000.0
        let durationSamples = 4800 // 0.3 秒 (30 フレーム)

        var idx = 0
        while idx < testFrequencies.count {
            let targetF0 = testFrequencies[idx]
            var pcm = [Float](repeating: 0.0, count: durationSamples)

            var s = 0
            while s < durationSamples {
                let t = Float(s) / sampleRate
                pcm[s] = 0.5 * sinf(2.0 * Float.pi * targetF0 * t)
                s += 1
            }

            let result = tracker.track(pcm: pcm)
            XCTAssertTrue(0 < result.frameCount, "フレーム数がゼロです")

            // 定常区間（中央フレーム群）の F0 精度を検査
            var validF0List: [Float] = []
            var f = 5
            let endF = result.frameCount - 5
            while f < endF {
                if 0.5 <= result.voiced[f] {
                    validF0List.append(result.f0[f])
                }
                f += 1
            }

            XCTAssertTrue(10 <= validF0List.count, "\(targetF0)Hz の有声フレーム数が不足しています")

            // 中央値の算出
            validF0List.sort()
            let medianF0 = validF0List[validF0List.count / 2]
            let absError = abs(medianF0 - targetF0)

            // 誤差は ±2.0 Hz 以内（相対誤差 1% 未満）であることを厳格に検証
            XCTAssertTrue(absError <= 2.0, "ターゲット周波数 \(targetF0)Hz に対する誤差が過大です: median=\(medianF0), error=\(absError)")
            idx += 1
        }
    }

    /// 無音および相関のないホワイトノイズに対する有声無声判定の検証
    ///
    /// 声帯振動が存在しない区間で F0 が誤検出されず、確実に F0=0.0 かつ voiced=0.0 となることを保証する。
    func testSilenceAndNoiseHandling() {
        let tracker = PitchTracker(sampleRate: 16000.0)

        // 1. 完全無音
        let silence = [Float](repeating: 0.0, count: 3200)
        let silenceResult = tracker.track(pcm: silence)
        var f = 0
        while f < silenceResult.frameCount {
            XCTAssertTrue(silenceResult.voiced[f] <= 1e-6, "無音区間で有声判定されました: f=\(f)")
            XCTAssertTrue(silenceResult.f0[f] <= 1e-6, "無音区間でF0が誤検出されました: f=\(f)")
            f += 1
        }

        // 2. ホワイトノイズ（周期性のない乱数信号）
        var noise = [Float](repeating: 0.0, count: 4800)
        var rng: UInt64 = 0x1234_5678_9ABC_DEF0
        var s = 0
        while s < noise.count {
            rng ^= (rng << 13)
            rng ^= (rng >> 7)
            rng ^= (rng << 17)
            let u32 = UInt32(truncatingIfNeeded: rng)
            noise[s] = ((Float(u32) / 4294967295.0) * 0.4) - 0.2
            s += 1
        }

        let noiseResult = tracker.track(pcm: noise)
        var voicedCount = 0
        f = 0
        while f < noiseResult.frameCount {
            if 0.5 <= noiseResult.voiced[f] {
                voicedCount += 1
            }
            f += 1
        }

        // ホワイトノイズでは周期相関が低いため、有声誤検出フレーム比率は極小（15% 未満）であることを検証
        let voicedRatio = Float(voicedCount) / Float(max(1, noiseResult.frameCount))
        XCTAssertTrue(voicedRatio <= 0.15, "ホワイトノイズに対する有声誤検出が過大です: \(voicedRatio)")
    }

    /// 短時間 RMS エネルギー抽出の単調性検証
    func testEnergyExtraction() {
        let tracker = PitchTracker(sampleRate: 16000.0)
        let sampleRate: Float = 16000.0
        let count = 3200

        var pcmLow = [Float](repeating: 0.0, count: count)
        var pcmHigh = [Float](repeating: 0.0, count: count)
        var s = 0
        while s < count {
            let t = Float(s) / sampleRate
            pcmLow[s] = 0.1 * sinf(2.0 * Float.pi * 200.0 * t)
            pcmHigh[s] = 0.4 * sinf(2.0 * Float.pi * 200.0 * t)
            s += 1
        }

        let resLow = tracker.track(pcm: pcmLow)
        let resHigh = tracker.track(pcm: pcmHigh)

        XCTAssertTrue(0 < resLow.frameCount, "フレーム数がゼロです")
        XCTAssertTrue(resLow.energy[5] < resHigh.energy[5], "振幅の増加に対して抽出エネルギーが増加していません")
    }

    /// 合成複合音（基音 220Hz + 高調波）に対するトラッキング精度検証
    func testHarmonicSignalTracking() {
        let tracker = PitchTracker(sampleRate: 16000.0)
        let sampleRate: Float = 16000.0
        let count = 4800
        var pcm = [Float](repeating: 0.0, count: count)

        var s = 0
        while s < count {
            let t = Float(s) / sampleRate
            let base = 0.5 * sinf(2.0 * Float.pi * 220.0 * t)
            let h2 = 0.3 * sinf(2.0 * Float.pi * 440.0 * t)
            let h3 = 0.15 * sinf(2.0 * Float.pi * 660.0 * t)
            pcm[s] = base + h2 + h3
            s += 1
        }

        let res = tracker.track(pcm: pcm)
        XCTAssertTrue(10 <= res.frameCount)

        var midF0List: [Float] = []
        var f = 5
        let endF = res.frameCount - 5
        while f < endF {
            if 0.5 <= res.voiced[f] {
                midF0List.append(res.f0[f])
            }
            f += 1
        }

        XCTAssertTrue(5 <= midF0List.count)
        midF0List.sort()
        let medianF0 = midF0List[midF0List.count / 2]
        let diff = abs(medianF0 - 220.0)
        XCTAssertTrue(diff <= 2.0, "調波複合音で基音 220Hz が検出されていません: median=\(medianF0)")
    }
}
