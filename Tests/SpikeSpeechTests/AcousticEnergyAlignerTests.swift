import XCTest
@testable import SpikeSpeech

final class AcousticEnergyAlignerTests: XCTestCase {
    var aligner: AcousticEnergyAligner!
    var lengthRegulator: LengthRegulator!
    var normalizer: TextNormalizer!
    var prosodyModel: ProsodyModel!
    var vocabulary: PhonemeVocabulary!

    override func setUp() {
        super.setUp()
        aligner = AcousticEnergyAligner()
        lengthRegulator = LengthRegulator(hiddenDimension: 64)
        normalizer = TextNormalizer()
        prosodyModel = ProsodyModel()
        vocabulary = PhonemeVocabulary()
    }

    /// 短時間平滑化エネルギーが非負であり、エネルギーの谷（振幅低下区間）が正確に検出されることを検証
    func testSmoothedEnergyDetectsValleys() {
        let hopSize = 160
        let frameCount = 10
        // フレーム 0..3: 高振幅 (0.5), フレーム 4..5: 低振幅 (0.01), フレーム 6..9: 高振幅 (0.5)
        var pcm = [Float](repeating: 0.0, count: frameCount * hopSize)
        var f = 0
        while f < frameCount {
            let amp: Float
            switch f {
            case 4, 5:
                amp = 0.01
            default:
                amp = 0.5
            }
            var s = 0
            while s < hopSize {
                let idx = f * hopSize + s
                pcm[idx] = amp
                s += 1
            }
            f += 1
        }

        let smoothed = aligner.computeSmoothedEnergy(
            pcm: pcm,
            hopSize: hopSize,
            startFrame: 0,
            frameCount: frameCount
        )

        XCTAssertEqual(smoothed.count, frameCount)
        // 谷（フレーム 4, 5）のエネルギーがフレーム 1, 2 や 7, 8 より小さいことを検証
        XCTAssertLessThan(smoothed[4], smoothed[1])
        XCTAssertLessThan(smoothed[5], smoothed[8])
    }

    /// モーラ境界が厳密に単調増加し、各モーラの最小フレーム数が保証されることを検証
    func testMoraBoundariesMonotonicAndValid() {
        let frameCount = 50
        var energy = [Float](repeating: 0.01, count: frameCount)
        // 適当なエネルギー変動を作成
        var i = 0
        while i < frameCount {
            let val = sinf(Float(i) * 0.5) + 1.1
            energy[i] = val
            i += 1
        }

        let moraCount = 5
        let minFrames = [2, 2, 2, 2, 2]

        let boundaries = aligner.findMoraBoundaries(
            smoothedEnergy: energy,
            moraCount: moraCount,
            moraMinFrames: minFrames
        )

        XCTAssertEqual(boundaries.count, moraCount + 1)
        XCTAssertEqual(boundaries[0], 0)
        XCTAssertEqual(boundaries[moraCount], frameCount)

        var m = 0
        while m < moraCount {
            let start = boundaries[m]
            let end = boundaries[m + 1]
            XCTAssertLessThanOrEqual(start + minFrames[m], end)
            m += 1
        }
    }

    /// 非対称波形において等時間割りと異なり、エネルギーの偏りに応じたモーラ境界が算出されることを検証
    func testEnergyDrivenBoundariesDifferFromUniformTimeDivision() {
        let hopSize = 160
        let frameCount = 40
        // 前半 20 フレーム: 高振幅 (0.8), 後半 20 フレーム: 低振幅 (0.05)
        var pcm = [Float](repeating: 0.0, count: frameCount * hopSize)
        var f = 0
        while f < frameCount {
            let amp: Float
            if f < 20 {
                amp = 0.8
            } else {
                amp = 0.05
            }
            var s = 0
            while s < hopSize {
                pcm[f * hopSize + s] = amp
                s += 1
            }
            f += 1
        }

        let smoothed = aligner.computeSmoothedEnergy(
            pcm: pcm,
            hopSize: hopSize,
            startFrame: 0,
            frameCount: frameCount
        )

        let moraCount = 2
        let minFrames = [1, 1]
        let boundaries = aligner.findMoraBoundaries(
            smoothedEnergy: smoothed,
            moraCount: moraCount,
            moraMinFrames: minFrames
        )

        // 等時間割りの場合の中央境界は 20。
        // 前半が高エネルギーのため、累積エネルギーの 50% は 20 よりも手前（前進）で達成される。
        let midBoundary = boundaries[1]
        XCTAssertNotEqual(midBoundary, 20)
        XCTAssertLessThan(midBoundary, 20)
    }

    /// alignPhonemes により全音素フレームの合計が発話区間総フレーム数と厳密に一致することを検証
    func testAlignPhonemesConservesTotalSpeechFrames() {
        let text = "こんにちは"
        let morphemes = normalizer.normalize(text: text)
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)

        let hopSize = 160
        let speechFrames = 75
        let leadSilence = 5
        let totalPcmFrames = leadSilence + speechFrames + 10
        var pcm = [Float](repeating: 0.1, count: totalPcmFrames * hopSize)

        // 簡単な振幅変調を付与
        var i = 0
        while i < pcm.count {
            pcm[i] = 0.2 + 0.1 * sinf(Float(i) * 0.01)
            i += 1
        }

        let durations = aligner.alignPhonemes(
            pcm: pcm,
            hopSize: hopSize,
            leadSilence: leadSilence,
            speechFrames: speechFrames,
            phrases: phrases,
            lengthRegulator: lengthRegulator
        )

        XCTAssertNotNil(durations)
        if let d = durations {
            var sum = 0
            var idx = 0
            while idx < d.count {
                XCTAssertLessThanOrEqual(1, d[idx])
                sum += d[idx]
                idx += 1
            }
            XCTAssertEqual(sum, speechFrames)
        }
    }
}
