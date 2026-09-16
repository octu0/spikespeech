import XCTest
import Foundation
@testable import SpikeSpeech

/// 現代的ニューラルボコーダー（NeuralVocoder）単体機能および数値安定性検証スイート
final class NeuralVocoderTests: XCTestCase {

    // MARK: - 1. 設定パラメータおよび初期化の整合性

    /// サンプリングレート、フレーム幅、およびチャンネル数の整合性を検証
    func testNeuralVocoderConfigDefaults() {
        let config = NeuralVocoderConfig()
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.hopSize, 160)
        XCTAssertEqual(config.melChannels, 64)
        XCTAssertEqual(config.hiddenChannels, 16)
    }

    /// ニューラルボコーダー重み構造体の決定論的初期化と JSON シリアライズ完全可逆性を検証
    func testNeuralVocoderWeightsInitializationAndCodable() throws {
        let config = NeuralVocoderConfig()
        let weights = NeuralVocoderWeights.randomWeights(config: config, seed: 42)

        XCTAssertEqual(weights.config, config)
        XCTAssertFalse(weights.convPreWeight.isEmpty)
        XCTAssertFalse(weights.res1Conv1Weight.isEmpty)
        XCTAssertFalse(weights.res2Conv1Weight.isEmpty)
        XCTAssertFalse(weights.convPostWeight.isEmpty)

        // JSON エンコードおよびデコードのラウンドトリップ検証
        let encoder = JSONEncoder()
        let data = try encoder.encode(weights)
        XCTAssertFalse(data.isEmpty)

        let decoder = JSONDecoder()
        let restored = try decoder.decode(NeuralVocoderWeights.self, from: data)

        XCTAssertEqual(restored.config, weights.config)
        XCTAssertEqual(restored.convPreWeight.count, weights.convPreWeight.count)
        XCTAssertEqual(restored.res1Conv1Weight.count, weights.res1Conv1Weight.count)
        XCTAssertEqual(restored.res2Conv1Weight.count, weights.res2Conv1Weight.count)
        XCTAssertEqual(restored.convPostWeight.count, weights.convPostWeight.count)
        XCTAssertEqual(restored.harmonicBias, weights.harmonicBias)
    }

    /// 重みパラメータが推論ホットパス上で実際に評価され出力波形に寄与することを検証
    /// なぜこのテストを行うか:
    /// 重み構造体が定義されているにもかかわらず推論時に無視・バイパスされる
    /// 「形骸化したニューラル実装」の混入を恒久的に防ぐため。
    func testNeuralVocoderWeightsActuallyAffectSynthesis() {
        let config = NeuralVocoderConfig()
        let weightsA = NeuralVocoderWeights.randomWeights(config: config, seed: 42)
        let weightsB = NeuralVocoderWeights.randomWeights(config: config, seed: 9999)

        let vocoderA = NeuralVocoder(weights: weightsA)
        let vocoderB = NeuralVocoder(weights: weightsB)

        let frameCount = 6
        let melSeq = [[Float]](repeating: [Float](repeating: 0.5, count: 64), count: frameCount)
        let f0s = [Float](repeating: 220.0, count: frameCount)
        let vFlags = [Float](repeating: 1.0, count: frameCount)

        vocoderA.reset()
        let pcmA = vocoderA.synthesize(mel: melSeq, f0Contour: f0s, voicedFlags: vFlags, voice: .female)

        vocoderB.reset()
        let pcmB = vocoderB.synthesize(mel: melSeq, f0Contour: f0s, voicedFlags: vFlags, voice: .female)

        XCTAssertEqual(pcmA.count, pcmB.count)

        var diffSum: Float = 0.0
        var i = 0
        while i < pcmA.count {
            diffSum += abs(pcmA[i] - pcmB[i])
            i += 1
        }
        let avgDiff = diffSum / Float(pcmA.count)
        XCTAssertTrue(1e-4 < avgDiff, "重みパラメータの差異が出力波形に反映されていません: \(avgDiff)")
    }

    // MARK: - 2. 単一フレームおよび複数フレームの波形合成

    /// 単一フレーム入力に対する正確な 160 サンプル生成および有限値（NaN/Inf 不在）を検証
    func testNeuralVocoderSynthesizeSingleFrame() {
        let vocoder = NeuralVocoder()
        let melFrame = [Float](repeating: 1.0, count: 64)

        let pcm = vocoder.synthesize(
            mel: [melFrame],
            f0Contour: [220.0],
            voicedFlags: [1.0],
            voice: .female
        )

        // 1フレーム = 160 サンプル
        XCTAssertEqual(pcm.count, 160)

        // 全サンプルが [-1.0, 1.0] の有限値であることを検証
        var s = 0
        while s < pcm.count {
            let val = pcm[s]
            XCTAssertFalse(val.isNaN, "単一フレーム合成で NaN を検出: \(s)")
            XCTAssertFalse(val.isInfinite, "単一フレーム合成で Inf を検出: \(s)")
            XCTAssertTrue(-1.0 <= val, "波形振幅が下限 -1.0 を下回っています: \(val)")
            XCTAssertTrue(val <= 1.0, "波形振幅が上限 1.0 を超えています: \(val)")
            s += 1
        }
    }

    /// 複数フレーム（10フレーム = 100ms = 1600サンプル）の波形合成と連続性を検証
    func testNeuralVocoderSynthesizeMultiFrame() {
        let vocoder = NeuralVocoder()
        let frameCount = 10
        var melSeq: [[Float]] = []
        var fIdx = 0
        while fIdx < frameCount {
            melSeq.append([Float](repeating: 0.5, count: 64))
            fIdx += 1
        }

        let pcm = vocoder.synthesize(
            mel: melSeq,
            f0Contour: [Float](repeating: 220.0, count: frameCount),
            voicedFlags: [Float](repeating: 1.0, count: frameCount),
            voice: .female
        )

        XCTAssertEqual(pcm.count, frameCount * 160)

        var s = 0
        while s < pcm.count {
            let val = pcm[s]
            XCTAssertFalse(val.isNaN, "複数フレーム合成で NaN を検出: \(s)")
            XCTAssertFalse(val.isInfinite, "複数フレーム合成で Inf を検出: \(s)")
            s += 1
        }

        // サンプル間の急激な不連続ステップ（クリック音）が存在しないことを検証
        var maxDelta: Float = 0.0
        var i = 1
        while i < pcm.count {
            let delta = abs(pcm[i] - pcm[i - 1])
            if maxDelta < delta {
                maxDelta = delta
            }
            i += 1
        }
        // 連続サンプル間の差分が 1.0 未満であることを確認
        XCTAssertTrue(maxDelta < 1.0, "隣接サンプル間に異常なステップ不連続を検出: \(maxDelta)")
    }

    // MARK: - 3. 話者条件（男性 vs 女性）の音響特性差分

    /// 男性 (120Hz) と女性 (220Hz) の話者条件により異なる波形が生成されることを検証
    func testNeuralVocoderSpeakerConditioning() {
        let vocoder = NeuralVocoder()
        let frameCount = 8
        let melSeq = [[Float]](repeating: [Float](repeating: 1.0, count: 64), count: frameCount)

        vocoder.reset()
        let femalePcm = vocoder.synthesize(
            mel: melSeq,
            f0Contour: [Float](repeating: 220.0, count: frameCount),
            voicedFlags: [Float](repeating: 1.0, count: frameCount),
            voice: .female
        )

        vocoder.reset()
        let malePcm = vocoder.synthesize(
            mel: melSeq,
            f0Contour: [Float](repeating: 120.0, count: frameCount),
            voicedFlags: [Float](repeating: 1.0, count: frameCount),
            voice: .male
        )

        XCTAssertEqual(femalePcm.count, malePcm.count)

        // 男声と女声でピッチ周期・倍音構造に有意な差分が存在することを検証
        var diffSum: Float = 0.0
        var i = 0
        while i < femalePcm.count {
            diffSum += abs(femalePcm[i] - malePcm[i])
            i += 1
        }
        let avgDiff = diffSum / Float(femalePcm.count)
        XCTAssertTrue(1e-3 < avgDiff, "男性と女性の間で出力波形に差分が存在しません: \(avgDiff)")
    }

    // MARK: - 4. 無音・ポーズ区間のエネルギー制御

    /// 極小 Mel エネルギー（無音フロア）入力時に出力波形が低エネルギーになることを検証
    func testNeuralVocoderZeroMelSilence() {
        let vocoder = NeuralVocoder()
        let frameCount = 5
        let silenceMel = [[Float]](repeating: [Float](repeating: -8.0, count: 64), count: frameCount)

        let pcm = vocoder.synthesize(
            mel: silenceMel,
            f0Contour: [Float](repeating: 0.0, count: frameCount),
            voicedFlags: [Float](repeating: 0.0, count: frameCount),
            voice: .female
        )

        XCTAssertEqual(pcm.count, frameCount * 160)

        // RMS エネルギーが 0.05 未満に抑制されていることを検証
        var sumSq: Float = 0.0
        var s = 0
        while s < pcm.count {
            sumSq += pcm[s] * pcm[s]
            s += 1
        }
        let rms = sqrtf(sumSq / Float(pcm.count))
        XCTAssertTrue(rms < 0.05, "無音入力時のエネルギーが過大です: \(rms)")
    }

    // MARK: - 5. ストリーミング逐次合成の整合性

    /// `synthesizeFrame` による逐次合成がバッファ境界を正しく満たすことを検証
    func testNeuralVocoderStreamingFrameSynthesis() {
        let vocoder = NeuralVocoder()
        let melFrame = [Float](repeating: 0.8, count: 64)
        var buffer = [Float](repeating: 0.0, count: 160)

        buffer.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(
                melFrame: melFrame,
                f0: 220.0,
                voiced: 1.0,
                voice: .female,
                dst: pDst.baseAddress!
            )
        }

        // バッファがゼロクリアされたままでなく、有効なサンプルが書き込まれていることを検証
        var nonZeroCount = 0
        var i = 0
        while i < buffer.count {
            if abs(buffer[i]) < 1e-6 {
                // 微小値
            } else {
                nonZeroCount += 1
            }
            i += 1
        }
        XCTAssertTrue(100 <= nonZeroCount, "フレーム逐次合成で有効サンプルが書き込まれていません")
    }

    // MARK: - 6. オンメモリ音声合成品質検証（ディスク書き出しゼロ）

    /// 女性・男性の音声合成 WAV がオンメモリで正しく生成され、フォーマット規格を満たすことを検証
    /// なぜディスク書き出しを排除するか:
    /// コーディング規約「テストのオンメモリ化」「絶対パスを残さない」を厳格に遵守するため。
    func testNeuralVocoderEvaluationAudioInMemory() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは、音声合成の世界へようこそ。自然な日本語の音声をお届けします。"

        let femaleWav = engine.synthesizeWav(text: text, voice: .female)
        let maleWav = engine.synthesizeWav(text: text, voice: .male)

        XCTAssertTrue(44 < femaleWav.count)
        XCTAssertTrue(44 < maleWav.count)

        // RIFF ヘッダ検証 (オンメモリバイト列)
        let fHeader = String(data: femaleWav.subdata(in: 0..<4), encoding: .ascii)
        let mHeader = String(data: maleWav.subdata(in: 0..<4), encoding: .ascii)
        XCTAssertEqual(fHeader, "RIFF")
        XCTAssertEqual(mHeader, "RIFF")

        // 男声と女声で出力データ長が同等であり、サンプル内容が異なることを検証
        XCTAssertEqual(femaleWav.count, maleWav.count)
        XCTAssertNotEqual(femaleWav, maleWav)
    }
}
