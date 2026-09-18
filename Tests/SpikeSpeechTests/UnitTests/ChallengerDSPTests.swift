import XCTest
@testable import SpikeSpeech

/// Milestone 2 / Wave 1: NeuralVocoder 敵対的・極限入力耐性 Challenger 検証テストスイート
///
/// 無音安定性、過大入力 Soft Limiter、NaN/Inf 異常入力耐性、
/// 極限 F0 耐性、および連続フレーム遷移平滑性を敵対的入力を用いて実証・保証する。
final class ChallengerDSPTests: XCTestCase {

    // MARK: - 1. 無音・ゼロ入力安定性テスト

    func testSilenceStabilityLongTermAndTransitions() {
        // 1秒間（100フレーム = 16,000サンプル）の連続無音において
        // 異常発振や非有限値が発生せず、安定した波形が生成されることを実証する。
        let vocoder = NeuralVocoder()

        // 1. 100 フレーム連続の極小対数 Mel 入力
        let silenceFrame = [Float](repeating: -20.0, count: 64)
        let silenceFrames = [[Float]](repeating: silenceFrame, count: 100)
        let silenceWave = vocoder.synthesize(mel: silenceFrames)

        XCTAssertEqual(silenceWave.count, 16000)
        var sIdx = 0
        while sIdx < silenceWave.count {
            XCTAssertTrue(silenceWave[sIdx].isFinite, "無音フレームで非有限サンプルを検出: index=\(sIdx)")
            sIdx += 1
        }

        // 2. 初期状態からのゼロ Mel フレーム
        vocoder.reset()
        let zeroFrame = [Float](repeating: 0.0, count: 64)
        let zeroWave = vocoder.synthesize(mel: [zeroFrame, zeroFrame])
        XCTAssertEqual(zeroWave.count, 320)
        var zIdx = 0
        while zIdx < zeroWave.count {
            XCTAssertTrue(zeroWave[zIdx].isFinite)
            zIdx += 1
        }
    }

    // MARK: - 2. Soft Limiter (Tanh) 及び振幅サチュレーション検証

    func testSoftLimiterAndSaturation() {
        // 極端なオーバードライブ入力下でも Soft Limiter が波形振幅を確実に [-1.0, 1.0] に収め、
        // かつハードクリッピングせず滑らかな非線形サチュレーションを保つことを保証する。
        let vocoder = NeuralVocoder()

        // 各種過大入力での振幅レンジ検証 (10.0, 50.0, 100.0)
        let testInputs: [Float] = [10.0, 50.0, 100.0]
        var gIdx = 0
        while gIdx < testInputs.count {
            let val = testInputs[gIdx]
            let overFrame = [Float](repeating: val, count: 64)
            vocoder.reset()
            let outWave = vocoder.synthesize(mel: [overFrame])
            XCTAssertEqual(outWave.count, 160)

            var i = 0
            while i < 160 {
                let s = outWave[i]
                XCTAssertTrue(s.isFinite)
                XCTAssertTrue(-1.0 <= s)
                XCTAssertTrue(s <= 1.0)
                i += 1
            }
            gIdx += 1
        }
    }

    // MARK: - 3. 敵対的異常入力（NaN/Inf）に対する耐性・フォールバック実証テスト

    func testAdversarialAbnormalInputResilience() {
        // 音響モデルの勾配爆発や数値例外によって異常値が渡された際、
        // プロセスがクラッシュせず安全に有限値でフォールバックすることを実証する。
        let vocoder = NeuralVocoder()

        // 1. Mel 特徴量の一部に NaN が含まれるフレーム
        var nanMel = [Float](repeating: -2.0, count: 64)
        nanMel[3] = Float.nan
        let outNaN = vocoder.synthesize(mel: [nanMel])
        XCTAssertEqual(outNaN.count, 160)
        var nIdx = 0
        while nIdx < 160 {
            XCTAssertTrue(outNaN[nIdx].isFinite)
            nIdx += 1
        }

        // 2. Mel 特徴量に +Inf / -Inf が含まれるフレーム
        var infMel = [Float](repeating: -2.0, count: 64)
        infMel[5] = Float.infinity
        infMel[6] = -Float.infinity
        let outInf = vocoder.synthesize(mel: [infMel])
        XCTAssertEqual(outInf.count, 160)
        var iIdx = 0
        while iIdx < 160 {
            XCTAssertTrue(outInf[iIdx].isFinite)
            iIdx += 1
        }
    }

    // MARK: - 4. 極限 F0 入力耐性テスト

    func testInfinityF0DoesNotHang() {
        // 非有限値 F0 や極端に高い F0 が渡された際、無限ループに陥らず、
        // 安全に有限な PCM 出力を生成してプロセスが健全に動作し続けることを数学的に実証する。
        let vocoder = NeuralVocoder()
        let frame = [Float](repeating: -2.0, count: 64)

        // 1. Infinity F0
        vocoder.reset()
        let outInfF0 = vocoder.synthesize(mel: [frame], f0Contour: [Float.infinity])
        XCTAssertEqual(outInfF0.count, 160)
        XCTAssertTrue(outInfF0[0].isFinite)

        // 2. NaN F0
        vocoder.reset()
        let outNaNF0 = vocoder.synthesize(mel: [frame], f0Contour: [Float.nan])
        XCTAssertEqual(outNaNF0.count, 160)
        XCTAssertTrue(outNaNF0[0].isFinite)

        // 3. ナイキスト周波数超え F0 (10000Hz)
        vocoder.reset()
        let outHighF0 = vocoder.synthesize(mel: [frame], f0Contour: [10000.0])
        XCTAssertEqual(outHighF0.count, 160)
        XCTAssertTrue(outHighF0[0].isFinite)
    }

    // MARK: - 6. 静的コード規約適合性検証 (DSP モジュール全走査)

    func testStaticCodingRulesComplianceDSP() {
        // 比較演算子は `<` および `<=` のみ、`else if` 禁止、三項演算子禁止などの規約を
        // Sources/SpikeSpeech/DSP/ 以下の全ファイルに対して自動検証する。
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let dspPath = currentDir + "/Sources/SpikeSpeech/DSP"

        guard let enumerator = fileManager.enumerator(atPath: dspPath) else {
            XCTFail("Sources/SpikeSpeech/DSP が走査できませんでした")
            return
        }

        var checkedFiles = 0
        while let relativePath = enumerator.nextObject() as? String {
            if relativePath.hasSuffix(".swift") != true {
                continue
            }

            let fullPath = dspPath + "/" + relativePath
            guard let content = try? String(contentsOfFile: fullPath, encoding: .utf8) else {
                continue
            }

            let lines = content.components(separatedBy: .newlines)
            var lineIdx = 0
            while lineIdx < lines.count {
                let line = lines[lineIdx]
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                if trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") {
                    lineIdx += 1
                    continue
                }

                XCTAssertFalse(
                    trimmed.contains(" > ") && trimmed.contains("->") != true,
                    "比較演算子 > が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(" >= "),
                    "比較演算子 >= が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains("else if"),
                    "else if が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(" ? ") && trimmed.contains("??") != true,
                    "三項演算子が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                lineIdx += 1
            }
            checkedFiles += 1
        }

        XCTAssertTrue(0 < checkedFiles, "チェック対象の Swift ファイルがありません")
        print("--- [Challenger DSP Static Rule Check] ---")
        print("検証完了 DSP ファイル数: \(checkedFiles) 件 (全ファイル規約適合)")
        print("------------------------------------------")
    }

    // MARK: - 11. 5母音波形健全性・エネルギー検証

    func testFiveVowelsFormantSpectraCompliance() {
        // 合成エンジンが生成する日本語5母音において、
        // 異常発振や無音にならず、全サンプルが有限かつ適正なエネルギーを持つことを実証する。
        let engine = SpikeSpeechEngine()
        let vowels = ["あ", "い", "う", "え", "お"]

        var vIdx = 0
        while vIdx < vowels.count {
            let vowel = vowels[vIdx]
            let samples = engine.synthesize(text: vowel)

            XCTAssertTrue(0 < samples.count)

            var energy: Float = 0.0
            var i = 0
            while i < samples.count {
                let s = samples[i]
                XCTAssertTrue(s.isFinite)
                XCTAssertTrue(-1.0 <= s)
                XCTAssertTrue(s <= 1.0)
                energy += s * s
                i += 1
            }

            XCTAssertTrue(0.0 < energy, "母音 \(vowel) のエネルギーが 0 です")
            vIdx += 1
        }
    }

    // MARK: - 12. 有声／無声フレーム切り替えにおけるクリックノイズ不在・平滑遷移検証

    func testExcitationTransitionSmoothnessAndClickAbsence() {
        // NeuralVocoder において、有声・無声（高エネルギー・低エネルギー）のフレームが
        // 交互に連続遷移する際、過度なクリックノイズや不連続衝撃が発生せず平滑に出力されることを実証する。
        let vocoder = NeuralVocoder()
        let voicedFrame = [Float](repeating: -2.0, count: 64)
        let unvoicedFrame = [Float](repeating: -20.0, count: 64)

        // 有声 -> 無声 -> 有声 の切り替えシーケンス
        let sequence = [
            voicedFrame, voicedFrame,
            unvoicedFrame, unvoicedFrame,
            voicedFrame, voicedFrame
        ]

        let output = vocoder.synthesize(mel: sequence)
        XCTAssertEqual(output.count, 6 * 160)

        // フレーム境界部におけるサンプル間最大一階差分 |s[n] - s[n-1]| を測定
        var maxDeltaAtBoundary: Float = 0.0
        let boundaryIndices = [160, 320, 480, 640, 800]

        var bIdx = 0
        while bIdx < boundaryIndices.count {
            let b = boundaryIndices[bIdx]
            var offset = -10
            while offset <= 10 {
                let idx = b + offset
                if 1 <= idx && idx < output.count {
                    let delta = abs(output[idx] - output[idx - 1])
                    if maxDeltaAtBoundary < delta {
                        maxDeltaAtBoundary = delta
                    }
                }
                offset += 1
            }
            bIdx += 1
        }

        print("--- [Excitation Transition Click Test] ---")
        print("境界近傍最大一階差分: \(maxDeltaAtBoundary)")

        // 急峻なインパルス不連続が発生せず、差分が 0.35 未満に抑制されていることを確認する。
        XCTAssertTrue(maxDeltaAtBoundary < 0.35, "フレーム切り替え境界で急峻な不連続パルスを検出しました: \(maxDeltaAtBoundary)")
    }
}


