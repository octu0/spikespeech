import XCTest
@testable import SpikeSpeech

/// 生体ゆらぎ（1/f ピンクノイズ、Jitter、Shimmer、発話継続時間ゆらぎ）の音響生理学的検証テスト
final class BiologicalFluctuationTests: XCTestCase {

    /// 1/f ピンクノイズの出力範囲および非自明性の検証
    func testPinkNoiseDistribution() {
        var bio = BiologicalFluctuation(seed: 12345)
        var minVal: Float = 100.0
        var maxVal: Float = -100.0
        var sum: Float = 0.0
        let count = 5000

        var i = 0
        while i < count {
            let sample = bio.nextPink()
            if sample < minVal {
                minVal = sample
            }
            if maxVal < sample {
                maxVal = sample
            }
            sum += sample
            i += 1
        }

        XCTAssertTrue(-1.0 <= minVal, "ピンクノイズ下限は -1.0 以上であること")
        XCTAssertTrue(maxVal <= 1.0, "ピンクノイズ上限は 1.0 以下であること")
        XCTAssertTrue(minVal < maxVal, "静止値ではなく変動していること")

        let mean = sum / Float(count)
        XCTAssertTrue(abs(mean) < 0.20, "大数法則により平均値は 0 近傍に分布すること (平均: \(mean))")
    }

    /// 声帯ピッチゆらぎ (Jitter) および振幅ゆらぎ (Shimmer) の適正範囲検証
    func testJitterAndShimmer() {
        var bio = BiologicalFluctuation(seed: 54321)
        let baseF0: Float = 220.0
        let baseGain: Float = 0.15

        var minJitterF0: Float = baseF0 * 2.0
        var maxJitterF0: Float = 0.0

        var minShimmerGain: Float = baseGain * 2.0
        var maxShimmerGain: Float = 0.0

        var i = 0
        while i < 500 {
            let jF0 = bio.computePitchJitter(baseF0: baseF0)
            if jF0 < minJitterF0 {
                minJitterF0 = jF0
            }
            if maxJitterF0 < jF0 {
                maxJitterF0 = jF0
            }

            let sGain = bio.computeAmplitudeShimmer(baseGain: baseGain)
            if sGain < minShimmerGain {
                minShimmerGain = sGain
            }
            if maxShimmerGain < sGain {
                maxShimmerGain = sGain
            }

            i += 1
        }

        // Jitter は ±1.2% 以内（220Hz に対しておよそ 217Hz 〜 223Hz）
        XCTAssertTrue(baseF0 * 0.985 <= minJitterF0, "Jitter 下限が正常であること")
        XCTAssertTrue(maxJitterF0 <= baseF0 * 1.015, "Jitter 上限が正常であること")

        // Shimmer は ±2.5% 以内（0.15 に対しておよそ 0.146 〜 0.154）
        XCTAssertTrue(baseGain * 0.97 <= minShimmerGain, "Shimmer 下限が正常であること")
        XCTAssertTrue(maxShimmerGain <= baseGain * 1.03, "Shimmer 上限が正常であること")
    }

    /// 発話継続時間（Duration）の生体テンポゆらぎの検証
    func testDurationFluctuation() {
        var bio = BiologicalFluctuation(seed: 99999)
        let baseFrames = 12

        var minDur = baseFrames * 2
        var maxDur = 0

        var i = 0
        while i < 200 {
            let dur = bio.computeDurationFluctuation(baseFrames: baseFrames)
            if dur < minDur {
                minDur = dur
            }
            if maxDur < dur {
                maxDur = dur
            }
            i += 1
        }

        XCTAssertTrue(1 <= minDur, "時間長は 1 フレーム以上であること")
        XCTAssertTrue(minDur <= maxDur, "最小値 <= 最大値 であること")
        // ±8% の伸縮のため 11〜13 フレーム程度に分布
        XCTAssertTrue(10 <= minDur, "極端な短縮が起きないこと")
        XCTAssertTrue(maxDur <= 14, "極端な伸長が起きないこと")
    }

    /// テキストハッシュシードのステートレス決定論性と発話多様性の検証
    func testTextSeedDeterminism() {
        let textA1 = "こんにちは、世界。"
        let textA2 = "こんにちは、世界。"
        let textB = "こんばんは、世界。"

        let seedA1 = BiologicalFluctuation.seed(from: textA1)
        let seedA2 = BiologicalFluctuation.seed(from: textA2)
        let seedB = BiologicalFluctuation.seed(from: textB)

        XCTAssertEqual(seedA1, seedA2, "同一テキストからは同一の 64-bit シードが生成されること")
        XCTAssertTrue(seedA1 != seedB, "異なるテキストからは異なるシードが生成されること")

        var bio1 = BiologicalFluctuation(seed: seedA1)
        var bio2 = BiologicalFluctuation(seed: seedA2)

        var step = 0
        while step < 100 {
            let val1 = bio1.nextPink()
            let val2 = bio2.nextPink()
            XCTAssertEqual(val1, val2, "同一シードから生成されるピンクノイズ系列が完全一致すること")
            step += 1
        }
    }

    /// 1/f ピンクノイズの単位分散特性の検証
    func testPinkNoiseUnitVariance() {
        var bio = BiologicalFluctuation(seed: 42)
        let n = 5000
        var sum: Float = 0.0
        var sumSq: Float = 0.0

        var i = 0
        while i < n {
            let v = bio.nextPink()
            sum += v
            sumSq += v * v
            i += 1
        }

        let mean = sum / Float(n)
        let variance = (sumSq / Float(n)) - (mean * mean)

        XCTAssertTrue(abs(mean) < 0.15, "平均値は 0 近傍であること")
        XCTAssertTrue(0.45 <= variance, "単位分散の切断理論値 (約 0.527) に合致すること (実測分散: \(variance))")
        XCTAssertTrue(variance <= 0.65, "分散が過大にならないこと (実測分散: \(variance))")
    }
}
