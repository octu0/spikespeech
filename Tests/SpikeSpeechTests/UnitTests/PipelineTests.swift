import XCTest
import Foundation
@testable import SpikeSpeech

/// Milestone 4: Synthetic Data, CLI & Pipeline 統合単体テストスイート
///
/// 基準音声生成器の音響物理特性、
/// および SpikeSpeechEngine による日本語テキストから WAV 生成までの
/// E2E パイプラインの一貫性・数値健全性を自動検証する。
final class PipelineTests: XCTestCase {

    override func setUp() {
        super.setUp()
    }

    override func tearDown() {
        super.tearDown()
    }
    // MARK: - 1. SpikeSpeechEngine E2E パイプライン単体・結合検証

    /// E2E 日本語テキスト音声合成 (Text -> 16kHz 16-bit WAV) の完全性検証
    func testEndToEndSpeechSynthesis() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは、世界。"

        let wavData = engine.synthesizeWav(text: text, speed: 1.0, pitch: 1.0)

        // 44 バイトヘッダ以上のサイズ
        XCTAssertTrue(44 < wavData.count, "WAV データサイズがヘッダ長以下です: \(wavData.count)")

        // ヘッダシグネチャの完全検証
        wavData.withUnsafeBytes { rawBytes in
            let ptr = rawBytes.bindMemory(to: UInt8.self)
            // "RIFF"
            XCTAssertEqual(ptr[0], 0x52)
            XCTAssertEqual(ptr[1], 0x49)
            XCTAssertEqual(ptr[2], 0x46)
            XCTAssertEqual(ptr[3], 0x46)
            // "WAVE"
            XCTAssertEqual(ptr[8], 0x57)
            XCTAssertEqual(ptr[9], 0x41)
            XCTAssertEqual(ptr[10], 0x56)
            XCTAssertEqual(ptr[11], 0x45)
            // "fmt "
            XCTAssertEqual(ptr[12], 0x66)
            XCTAssertEqual(ptr[13], 0x6D)
            XCTAssertEqual(ptr[14], 0x74)
            XCTAssertEqual(ptr[15], 0x20)
            // "data"
            XCTAssertEqual(ptr[36], 0x64)
            XCTAssertEqual(ptr[37], 0x61)
            XCTAssertEqual(ptr[38], 0x74)
            XCTAssertEqual(ptr[39], 0x61)
        }

        // PCM サンプル列の直接取得と数値検証
        let pcmSamples = engine.synthesize(text: text)
        XCTAssertTrue(0 < pcmSamples.count)

        var sIdx = 0
        while sIdx < pcmSamples.count {
            let val = pcmSamples[sIdx]
            XCTAssertFalse(val.isNaN, "E2E 合成波形で NaN を検出: index=\(sIdx)")
            XCTAssertFalse(val.isInfinite, "E2E 合成波形で Inf を検出: index=\(sIdx)")
            XCTAssertTrue(abs(val) <= 1.0, "E2E 合成波形でクリッピングを検出: val=\(val)")
            sIdx += 1
        }
    }

    /// 多層 SNN 構成での音声合成の検証
    func testMultilayerSpeechSynthesis() {
        let text = "すぱいくすぴーち"
        let layerConfigs = [1, 2, 3]

        var s = 0
        while s < layerConfigs.count {
            let layers = layerConfigs[s]
            let weights = SpikingNetworkWeights.randomWeights(numLayers: layers)
            let engine = SpikeSpeechEngine(weights: weights)
            let wav = engine.synthesizeWav(text: text)
            XCTAssertTrue(44 < wav.count, "層数 \(layers) で WAV データが生成されませんでした")
            s += 1
        }
    }

    /// 話速およびピッチ制御の検証
    func testSpeedAndPitchModulation() {
        let engine = SpikeSpeechEngine()
        let text = "あいうえお"

        let fastSamples = engine.synthesize(text: text, speed: 1.5, pitch: 1.0)
        let slowSamples = engine.synthesize(text: text, speed: 0.8, pitch: 1.0)

        // 高速発話は低速発話よりサンプル数が厳密に少ない
        XCTAssertTrue(fastSamples.count < slowSamples.count, "話速変更によるサンプル数の短縮が機能していません: fast=\(fastSamples.count), slow=\(slowSamples.count)")

        // ピッチ変更時の合成検証
        let highPitchSamples = engine.synthesize(text: text, speed: 1.0, pitch: 1.4)
        XCTAssertTrue(0 < highPitchSamples.count)
    }

    /// 境界値入力時の安全フォールバック検証
    func testEdgeCaseEmptyTextSynthesis() {
        let engine = SpikeSpeechEngine()

        let emptyWav = engine.synthesizeWav(text: "")
        XCTAssertEqual(emptyWav.count, 44, "空テキスト入力時に 44 バイトヘッダが返却されませんでした")

        let spaceWav = engine.synthesizeWav(text: "   ")
        XCTAssertTrue(44 <= spaceWav.count)
    }

    /// ストリーミング音声合成の逐次検証
    func testStreamingSynthesis() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは"

        var callbackCount = 0
        var receivedSamplesCount = 0

        let totalSamples = engine.synthesizeStream(
            text: text,
            speed: 1.0,
            pitch: 1.0
        ) { frameSamples in
            XCTAssertEqual(frameSamples.count, AudioConfig.hopSize, "フレームサンプル数が 160 ではありません")
            callbackCount += 1
            receivedSamplesCount += frameSamples.count
        }

        XCTAssertTrue(0 < callbackCount, "コールバックが一度も呼ばれませんでした")
        XCTAssertEqual(receivedSamplesCount, totalSamples.count, "ストリーミング総サンプル数が一致しません")
    }

    // MARK: - 3. Milestone 4 静的コーディング規約機械検査

    /// Milestone 4 実装ファイルのコーディング規約完全遵守検証
    func testMilestone4StaticRuleCheck() {
        // M4 で新規作成された全ファイルについて、比較演算子 `<` と `<=` のみ、
        // `else if` 禁止、三項演算子禁止などのコーディング規約完全遵守を自動検証する。
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath

        let targetPaths = [
            currentDir + "/Sources/SpikeSpeech/DSP/NeuralVocoder.swift",
            currentDir + "/Sources/SpikeSpeech/Pipeline/SpikeSpeechEngine.swift",
            currentDir + "/script/synthesize/main.swift",
            currentDir + "/script/train/main.swift",
            currentDir + "/script/benchmark/main.swift"
        ]

        let gtSym = " " + ">" + " "
        let gteSym = " " + ">=" + " "
        let elseIfSym = "else" + " " + "if"
        let ternarySym = " " + "?" + " "

        var checkedCount = 0
        var fIdx = 0
        while fIdx < targetPaths.count {
            let path = targetPaths[fIdx]
            guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
                fIdx += 1
                continue
            }

            let lines = content.components(separatedBy: .newlines)
            var lineIdx = 0
            while lineIdx < lines.count {
                let line = lines[lineIdx]
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                // コメント行のスキップ
                if trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") {
                    lineIdx += 1
                    continue
                }

                XCTAssertFalse(
                    trimmed.contains(gtSym) && trimmed.contains("->") != true,
                    "規約違反 (大なり記号) が使用されています: \(path):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(gteSym),
                    "規約違反 (大なりイコール記号) が使用されています: \(path):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(elseIfSym),
                    "規約違反 (else-if) が使用されています: \(path):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(ternarySym) && trimmed.contains("??") != true,
                    "規約違反 (三項演算子) が使用されています: \(path):\(lineIdx + 1): \(trimmed)"
                )
                lineIdx += 1
            }

            checkedCount += 1
            fIdx += 1
        }

        XCTAssertEqual(checkedCount, targetPaths.count, "走査対象ファイル数が不一致です")
        print("--- [Milestone 4 Static Rule Check] ---")
        print("検証完了ファイル数: \(checkedCount) 件 (全ファイル規約適合)")
        print("---------------------------------------")
    }

    // MARK: - 4. 速度境界値・NaN堅牢性およびサチュレーション抑制テスト

    /// speed == 0.0、負数、NaN/Inf、極端な速度指定時のクラッシュ根絶検証
    func testSpeedZeroAndExtremeValuesRobustness() {
        // 速度 0.0 や非有限値が入力された際にゼロ除算や浮動小数点キャスト例外による
        // 異常終了を防止し、安全域 [0.1, 10.0] へ適切にクランプされることを検証する。
        let regulator = LengthRegulator()
        let testSpeeds: [Float] = [0.0, -1.0, -100.0, Float.nan, Float.infinity, -Float.infinity, 0.0001, 100.0]

        var idx = 0
        while idx < testSpeeds.count {
            let spd = testSpeeds[idx]

            // 1. floatDurationFrames の計算
            let floatDur = regulator.floatDurationFrames(category: .vowel, symbol: "a", speed: spd)
            XCTAssertFalse(floatDur.isNaN, "speed=\(spd) で floatDurationFrames が NaN を返しました")
            XCTAssertFalse(floatDur.isInfinite, "speed=\(spd) で floatDurationFrames が Inf を返しました")
            XCTAssertTrue(1.0 <= floatDur, "speed=\(spd) で floatDurationFrames が 1.0 未満です")

            // 2. defaultDurationFrames の計算 (Int キャスト安全性の確認)
            let intDur = regulator.defaultDurationFrames(category: .vowel, symbol: "a", speed: spd)
            XCTAssertTrue(1 <= intDur, "speed=\(spd) で defaultDurationFrames が 1 未満です")

            idx += 1
        }

        // 3. 異常な durations 配列に対する quantizeDurations の安全性
        let abnormalDurations: [Float] = [Float.nan, Float.infinity, -5.0, 0.0, 10.0]
        let quantized = regulator.quantizeDurations(durations: abnormalDurations)
        XCTAssertEqual(quantized.count, abnormalDurations.count)
        var q = 0
        while q < quantized.count {
            XCTAssertTrue(1 <= quantized[q], "quantizeDurations で 1 未満のフレーム数が生成されました")
            q += 1
        }

        // 4. SpikeSpeechEngine E2E での speed == 0.0 安全合成
        let engine = SpikeSpeechEngine()
        let wavData = engine.synthesizeWav(text: "テスト", speed: 0.0)
        XCTAssertTrue(44 < wavData.count, "speed == 0.0 で有効な WAV データが生成されませんでした")

        let samples = engine.synthesize(text: "テスト", speed: 0.0)
        XCTAssertTrue(0 < samples.count, "speed == 0.0 で出力サンプルが空です")
    }

    /// WAV 出力振幅の適正化およびクリッピングサチュレーション抑制検証
    func testWavHeadroomAndNoSaturationClipping() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは、世界。スパイクスピーチの合成音声テストです。"

        let samples = engine.synthesize(text: text)
        XCTAssertTrue(0 < samples.count, "合成サンプルが空です")

        var maxAbs: Float = 0.0
        var sumSq: Float = 0.0
        var i = 0
        while i < samples.count {
            let s = samples[i]
            let a = abs(s)
            if maxAbs < a {
                maxAbs = a
            }
            sumSq += s * s
            i += 1
        }

        let rms = sqrt(sumSq / Float(samples.count))

        // 最大絶対値振幅がヘッドルーム 0.8801 以下であること（1.0 への張り付き皆無）
        XCTAssertTrue(maxAbs <= 0.8801, "ヘッドルームを超過するサンプル振幅を検出: maxAbs=\(maxAbs)")
        // 有意な音響エネルギーが存在すること
        XCTAssertTrue(0.1 < rms, "音響エネルギーが小さすぎます: rms=\(rms)")

        // WAV バイナリ変換後の Int16 範囲検証
        let wavData = engine.synthesizeWav(text: text)
        let sampleCount = (wavData.count - 44) / 2
        var clippingCount = 0
        wavData.withUnsafeBytes { rawBytes in
            let basePtr = rawBytes.baseAddress!.advanced(by: 44).assumingMemoryBound(to: Int16.self)
            var sIdx = 0
            while sIdx < sampleCount {
                let val = basePtr[sIdx]
                if val == 32767 || val == -32768 {
                    clippingCount += 1
                }
                sIdx += 1
            }
        }

        // サチュレーション（最大振幅張り付き）が完全にゼロであること
        XCTAssertEqual(clippingCount, 0, "WAV サンプルでクリッピングサチュレーションが検出されました: \(clippingCount)")
    }
}
