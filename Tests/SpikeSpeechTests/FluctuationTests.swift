import XCTest
@testable import SpikeSpeech

/// 生体ゆらぎ（1/f ピンクノイズ、Jitter、Shimmer、マイクロプロソディ）の音響生理学的検証テスト
final class FluctuationTests: XCTestCase {

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

    /// 調音音声学に基づくマイクロプロソディ（無声子音後の母音 F0 上昇 vs 有声子音後）の検証
    func testMicroprosodyEffect() {
        let normalizer = TextNormalizer()
        let prosody = ProsodyModel(baseF0: 200.0)
        let vocab = PhonemeVocabulary()
        let regulator = LengthRegulator()

        // 「か」(無声破裂音 /k/ + 母音 /a/) と 「が」(有声破裂音 /g/ + 母音 /a/)
        let lingKa = regulator.processText(text: "か", normalizer: normalizer, prosodyModel: prosody, vocabulary: vocab)
        let lingGa = regulator.processText(text: "が", normalizer: normalizer, prosodyModel: prosody, vocabulary: vocab)

        // /a/ 母音の開始フレームを正確に特定（/k/ の継続時間後、および /g/ の継続時間後）
        let ka_kDur = Int(lingKa.durations[0])
        let ga_gDur = Int(lingGa.durations[0])

        let vowelKaF0 = lingKa.f0Contour[ka_kDur]
        let vowelGaF0 = lingGa.f0Contour[ga_gDur]

        XCTAssertTrue(0.0 < vowelKaF0, "か の母音 /a/ F0 が正であること")
        XCTAssertTrue(0.0 < vowelGaF0, "が の母音 /a/ F0 が正であること")

        // 音響音声学の物理法則: 無声破裂音直後の母音立ち上がりピッチ (+3%) は有声破裂音直後 (-2%) よりも高くなる
        XCTAssertTrue(vowelGaF0 < vowelKaF0, "マイクロプロソディにより無声破裂音直後の母音 /a/ F0 (\(vowelKaF0)) が有声子音直後 (\(vowelGaF0)) より高くなること")
    }

    /// テキストハッシュシードのステートレス決定論性と発話多様性の検証
    /// なぜ検証するか:
    /// 同一テキストの複数回合成においてビット単位で完全再現され（ステートレス性）、
    /// かつテキストが異なれば別系列のゆらぎが生成されることを自動テストでロックするため。
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

    /// 1/f ピンクノイズの真の単位分散特性（余分な 0.5 減衰の排除）の検証
    /// なぜ検証するか:
    /// Jitter (±1.2%) や Shimmer (±2.5%) の公称強度が半分に減衰せず、
    /// 正確な標準偏差（std ≈ 1.0）として発揮されることを保証するため。
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
        // なぜ 0.45〜0.65 の範囲を検証するか:
        // 単位分散正規化された乱数列を [-1.0, 1.0] に安全クリップした場合、
        // 裾野の切断効果により数学的理論分散はおよそ 0.527 となる。
        // 旧実装（余分な 0.5 減衰乗算による分散約 0.25）から約 2 倍向上し、
        // 理論値（0.527）と整合していることを厳密に検証する。
        XCTAssertTrue(0.45 <= variance, "単位分散の切断理論値 (約 0.527) に合致し、旧実装の過度な減衰 (0.25) がないこと (実測分散: \(variance))")
        XCTAssertTrue(variance <= 0.65, "分散が過大にならないこと (実測分散: \(variance))")
    }

    /// ΔF0 のフレーム間変化率と無声マスクの検証
    /// なぜ検証するか:
    /// 前フレームとの差分が有声無声境界で正しくマスクされ、有声継続区間で一定ピッチ時に
    /// ±1.0 に飽和せず 0.0 に安定することを証明するため。
    func testDeltaF0ScaleSymmetry() {
        let engine = SpikeSpeechEngine()

        // 200Hz 一定の F0 を持つ有声フレーム系列
        let f0Const: [Float] = [0.0, 200.0, 200.0, 200.0, 0.0]
        let voicedFlags: [Float] = [0.0, 1.0, 1.0, 1.0, 0.0]
        let energy: [Float] = [0.0, 0.7, 0.7, 0.7, 0.0]
        let phoneIds: [Int32] = [1, 5, 5, 5, 1] // sil, a, a, a, sil
        let durations: [Int32] = [1, 1, 1, 1, 1]

        let ling = LinguisticFeatures(
            phoneIds: phoneIds,
            durations: durations,
            f0Contour: f0Const,
            voicedFlags: voicedFlags,
            energyContour: energy,
            totalFrames: 5
        )

        let encoded = engine.encodeLinguisticFeatures(features: ling)

        // フレーム 0: 無声 -> deltaF0 (ch67) = 0.0
        XCTAssertEqual(encoded[0][67], 0.0, accuracy: 1e-4, "無声フレームでの deltaF0 は 0.0 であること")

        // フレーム 1: 無声から有声への立ち上がり -> 無声境界マスクにより deltaF0 = 0.0
        XCTAssertEqual(encoded[1][67], 0.0, accuracy: 1e-4, "有声開始フレームでの deltaF0 は 0.0 であること")

        // フレーム 2: 有声継続かつ F0 一定 (200Hz) -> 差分 0.0（飽和バグ解消の証明）
        XCTAssertEqual(encoded[2][67], 0.0, accuracy: 1e-4, "有声継続一定ピッチでの deltaF0 は 0.0 であること")

        // フレーム 3: 有声継続かつ F0 一定 (200Hz) -> 差分 0.0
        XCTAssertEqual(encoded[3][67], 0.0, accuracy: 1e-4, "有声継続一定ピッチでの deltaF0 は 0.0 であること")

        // フレーム 4: 有声から無声への移行 -> deltaF0 = 0.0
        XCTAssertEqual(encoded[4][67], 0.0, accuracy: 1e-4, "無声フレームでの deltaF0 は 0.0 であること")

        // ch65 の直交特徴量（無声度 = 1.0 - voiced）の検証
        XCTAssertEqual(encoded[0][65], 1.0, accuracy: 1e-4, "無声フレームの ch65 は 1.0 であること")
        XCTAssertEqual(encoded[2][65], 0.0, accuracy: 1e-4, "有声フレームの ch65 は 0.0 であること")
    }

    /// 音素ヘルパー hy (32) の無声摩擦音・無声子音分類の検証
    /// なぜ検証するか:
    /// 「ひゃ」「ひゅ」「ひょ」の子音 hy が漏れなく無声摩擦音・無声子音として分類されることを保証するため。
    func testPhonemeHelperHy() {
        let vocab = PhonemeVocabulary()
        let hyId = 32

        XCTAssertEqual(vocab.token(for: hyId), "hy", "ID 32 は hy であること")
        XCTAssertTrue(vocab.isUnvoicedFricative(id: hyId), "hy は無声摩擦音であること")
        XCTAssertTrue(vocab.isUnvoicedConsonant(id: hyId), "hy は無声子音であること")
        XCTAssertTrue(vocab.isVoiced(symbol: "hy") != true, "hy は無声音であること")
    }

    /// 推論時エネルギー輪郭の音素物理カテゴリプロファイルの検証
    /// なぜ検証するか:
    /// 一律固定値（0.60/0.10）ではなく、休止・破裂音閉鎖・摩擦音・母音の
    /// 音響物理エネルギー勾配が形成されていることを保証するため。
    func testInferenceEnergyProfile() {
        let regulator = LengthRegulator()
        let normalizer = TextNormalizer()
        let prosody = ProsodyModel()
        let vocab = PhonemeVocabulary()

        // 「すし」(s, u, sh, i)
        let features = regulator.processText(
            text: "すし",
            normalizer: normalizer,
            prosodyModel: prosody,
            vocabulary: vocab
        )

        XCTAssertTrue(0 < features.totalFrames, "フレームが生成されていること")
        XCTAssertEqual(features.energyContour.count, features.totalFrames, "エネルギー輪郭と総フレーム数が一致すること")

        // 各フレームのエネルギーが適正な物理範囲 [0.0, 1.0] に収まっていること
        var fIdx = 0
        var foundVowelEnergy = false
        var foundFricativeEnergy = false

        while fIdx < features.totalFrames {
            let eng = features.energyContour[fIdx]
            XCTAssertTrue(0.0 <= eng, "エネルギーは 0 以上であること")
            XCTAssertTrue(eng <= 1.0, "エネルギーは 1.0 以下であること")

            if 0.60 <= eng {
                foundVowelEnergy = true
            }
            if 0.15 <= eng && eng <= 0.35 {
                foundFricativeEnergy = true
            }
            fIdx += 1
        }

        XCTAssertTrue(foundVowelEnergy, "母音の高エネルギー区間が存在すること")
        XCTAssertTrue(foundFricativeEnergy, "摩擦音の中間エネルギー区間が存在すること")
    }
}
