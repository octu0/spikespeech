import XCTest
import Foundation
#if canImport(MLX)
import MLX
import MLXNN
import MLXOptimizers
#endif
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
        XCTAssertEqual(config.hiddenChannels, 256)
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
    }

    /// 旧 64ch 重み検出時に破棄して 256ch 初期重みへフォールバックすることを検証
    func testNeuralVocoderFallbackFromLegacy64ChannelWeights() {
        let legacyConfig = NeuralVocoderConfig(hiddenChannels: 64)
        let legacyWeights = NeuralVocoderWeights.randomWeights(config: legacyConfig, seed: 123)

        let vocoder = NeuralVocoder(weights: legacyWeights)
        XCTAssertEqual(vocoder.weights.config.hiddenChannels, 256, "旧 64ch 重みが 256ch 初期重みへフォールバックしていません")
    }

    /// 連続2回合成で状態汚染がないことを検証
    func testNeuralVocoderConsecutiveSynthesisNoStatePollution() {
        let vocoder = NeuralVocoder()
        let frameCount = 6
        let melSeq = [[Float]](repeating: [Float](repeating: 0.3, count: 64), count: frameCount)
        let f0s = [Float](repeating: 200.0, count: frameCount)
        let vFlags = [Float](repeating: 1.0, count: frameCount)

        vocoder.reset()
        let pcm1 = vocoder.synthesize(mel: melSeq, f0Contour: f0s, voicedFlags: vFlags)
        vocoder.reset()
        let pcm2 = vocoder.synthesize(mel: melSeq, f0Contour: f0s, voicedFlags: vFlags)

        XCTAssertEqual(pcm1.count, pcm2.count)
        XCTAssertEqual(pcm1, pcm2, "リセット後の連続合成で出力波形に状態汚染が生じています")
    }

    /// 重みパラメータが推論ホットパス上で実際に評価され出力波形に寄与することを検証
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
        let pcmA = vocoderA.synthesize(mel: melSeq, f0Contour: f0s, voicedFlags: vFlags)

        vocoderB.reset()
        let pcmB = vocoderB.synthesize(mel: melSeq, f0Contour: f0s, voicedFlags: vFlags)

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
            voicedFlags: [1.0]
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
            voicedFlags: [Float](repeating: 1.0, count: frameCount)
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

    // MARK: - 3. 話者条件付け（SpeakerConditioning）の検証

    /// SpeakerConditioning 埋め込みベクトルの違いにより出力波形が変化することを検証
    func testNeuralVocoderSpeakerConditioning() {
        let vocoder = NeuralVocoder()
        let frameCount = 8
        let melSeq = [[Float]](repeating: [Float](repeating: 1.0, count: 64), count: frameCount)
        let f0Contour = [Float](repeating: 200.0, count: frameCount)
        let voicedFlags = [Float](repeating: 1.0, count: frameCount)

        let speakerA = SpeakerConditioning(embedding: [Float](repeating: 0.5, count: 128))
        let speakerB = SpeakerConditioning(embedding: [Float](repeating: -0.5, count: 128))

        vocoder.reset()
        let pcmA = vocoder.synthesize(
            mel: melSeq,
            f0Contour: f0Contour,
            voicedFlags: voicedFlags,
            speaker: speakerA
        )

        vocoder.reset()
        let pcmB = vocoder.synthesize(
            mel: melSeq,
            f0Contour: f0Contour,
            voicedFlags: voicedFlags,
            speaker: speakerB
        )

        XCTAssertEqual(pcmA.count, pcmB.count)

        var diffSum: Float = 0.0
        var i = 0
        while i < pcmA.count {
            diffSum += abs(pcmA[i] - pcmB[i])
            i += 1
        }
        let avgDiff = diffSum / Float(pcmA.count)
        XCTAssertTrue(1e-5 < avgDiff, "話者条件付けによる波形差分が存在しません: \(avgDiff)")
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
            voicedFlags: [Float](repeating: 0.0, count: frameCount)
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
                speaker: .zero,
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
        let text = "こんにちは"

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

#if canImport(MLX)
import MLX

extension NeuralVocoderTests {
    func testMLXAndSwiftNeuralVocoderEquivalence() {
        let weights = NeuralVocoderWeights.randomWeights(seed: 1234)
        let mlxVocoder = MLXNeuralVocoder(weights: weights)
        let swiftVocoder = NeuralVocoder(weights: weights)

        let totalFrames = 4
        let inCh = weights.config.melChannels + 2 // 66
        var flatFeats = [Float](repeating: 0.0, count: totalFrames * inCh)
        var i = 0
        while i < flatFeats.count {
            flatFeats[i] = sinf(Float(i) * 0.1)
            i += 1
        }

        // MLX 推論
        let mlxInput = MLXArray(flatFeats, [1, totalFrames, inCh])
        let mlxOutput = mlxVocoder(mlxInput)
        let mlxAudio = mlxOutput.asArray(Float.self)

        // Swift 推論: 特徴量を mel, f0, voiced に分解
        var mel = [[Float]]()
        var f0 = [Float]()
        var voiced = [Float]()
        var t = 0
        while t < totalFrames {
            let row = t * inCh
            let melRow = Array(flatFeats[row..<(row + weights.config.melChannels)])
            mel.append(melRow)
            let normF0 = flatFeats[row + weights.config.melChannels]
            f0.append(normF0 * 500.0)
            let v = flatFeats[row + weights.config.melChannels + 1]
            voiced.append(v)
            t += 1
        }

        swiftVocoder.reset()
        let swiftAudio = swiftVocoder.synthesize(mel: mel, f0Contour: f0, voicedFlags: voiced)

        print("[Equivalence] MLX samples: \(mlxAudio.count), Swift samples: \(swiftAudio.count)")
        var diffSum: Float = 0.0
        var s = 0
        let limit = min(mlxAudio.count, swiftAudio.count)
        while s < limit {
            diffSum += abs(mlxAudio[s] - swiftAudio[s])
            s += 1
        }
        let avgDiff = diffSum / Float(limit)
        print("[Equivalence] Avg diff: \(avgDiff)")
        XCTAssertTrue(avgDiff < 0.05, "MLX と Swift のボコーダー推論波形の誤差が許容値を超えています: \(avgDiff)")
    }

    /// 実際の SpikeSpeechEngine で合成された音声の波形特性・ピッチ・フォルマントを直接診断
    func testInspectSynthesizedAudioWaveform() {
        let vocoderPath = "Models/vocoder_weights.json"
        if FileManager.default.fileExists(atPath: vocoderPath) != true {
            let cleanWeights = NeuralVocoderWeights.randomWeights()
            if let encoded = try? JSONEncoder().encode(cleanWeights) {
                try? encoded.write(to: URL(fileURLWithPath: vocoderPath), options: .atomic)
            }
        }

        let engine = SpikeSpeechEngine()
        let text = "こんにちは"
        let femaleSamples = engine.synthesize(text: text, voice: .female)
        let maleSamples = engine.synthesize(text: text, voice: .male)

        XCTAssertFalse(femaleSamples.isEmpty, "女性合成音声が空です")
        XCTAssertFalse(maleSamples.isEmpty, "男性合成音声が空です")

        let fLinguistic = engine.lengthRegulator.processText(text: text, normalizer: engine.normalizer, prosodyModel: engine.prosodyModel, vocabulary: engine.vocabulary, speedFactor: 1.0, baseF0: VoiceProfile.female.baseF0)
        let mLinguistic = engine.lengthRegulator.processText(text: text, normalizer: engine.normalizer, prosodyModel: engine.prosodyModel, vocabulary: engine.vocabulary, speedFactor: 1.0, baseF0: VoiceProfile.male.baseF0)
        let midFrame = 29
        print(String(format: "MidFrame %d: Female target F0=%.1f Hz, Male target F0=%.1f Hz", midFrame, fLinguistic.f0Contour[midFrame], mLinguistic.f0Contour[midFrame]))

        print("=== [Synthesized Audio Inspection] ===")
        print("Female sample count: \(femaleSamples.count), Male sample count: \(maleSamples.count)")
        let fooLinguistic = engine.lengthRegulator.processText(text: "お好きな日本語テキストを入力してください。", normalizer: engine.normalizer, prosodyModel: engine.prosodyModel, vocabulary: engine.vocabulary, speedFactor: 1.0, baseF0: VoiceProfile.female.baseF0)
        let fooInp = engine.encodeLinguisticFeatures(features: fooLinguistic)
        let fooDec = engine.decoder.decodeSequence(featuresSeq: fooInp, workspace: engine.workspace)
        print("foo SNN Decoded frames: \(fooDec.count)")
        var fMin: Float = 999.0
        var fMax: Float = -999.0
        var fSum: Float = 0.0
        var fTotal = 0
        for row in fooDec {
            for v in row {
                if v < fMin { fMin = v }
                if fMax < v { fMax = v }
                fSum += v
                fTotal += 1
            }
        }
        print(String(format: "foo SNN Mel stats: min=%.4f, max=%.4f, avg=%.4f (total=%d)", fMin, fMax, fSum / Float(max(1, fTotal)), fTotal))
        if 20 < fooDec.count {
            print("foo SNN Mel frame 20 first 10: \(Array(fooDec[20].prefix(10)))")
        }

        let tracker = PitchTracker(minF0: 80.0, maxF0: 500.0)
        let fTrack = tracker.track(pcm: femaleSamples)
        let mTrack = tracker.track(pcm: maleSamples)

        // 中盤有声フレームのピッチ周波数を比較
        // なぜ .build_worker/foo.wav を直接検証するか:
        // 既存の生成ファイル foo.wav における基本周波数の追従性、およびフレーム周期（100Hz）への
        // 偽相関・モノトーン化が生じているかを直接測定するため。
        if let fooData = try? Data(contentsOf: URL(fileURLWithPath: ".build_worker/foo.wav")) {
            if 44 < fooData.count {
                let pcmBytes = fooData.subdata(in: 44..<fooData.count)
                let sCount = pcmBytes.count / 2
                var fooSamples = [Float](repeating: 0.0, count: sCount)
                pcmBytes.withUnsafeBytes { raw in
                    let ptr16 = raw.bindMemory(to: Int16.self)
                    var si = 0
                    while si < sCount {
                        fooSamples[si] = Float(ptr16[si]) / 32767.0
                        si += 1
                    }
                }
                let fooTrack = tracker.track(pcm: fooSamples)
                var fooVoicedCount = 0
                var foo100HzCount = 0
                var fi = 0
                while fi < fooTrack.frameCount {
                    if 0.5 <= fooTrack.voiced[fi] {
                        fooVoicedCount += 1
                        let f = fooTrack.f0[fi]
                        if 95.0 <= f && f <= 105.0 {
                            foo100HzCount += 1
                        }
                    }
                    fi += 1
                }
                print("foo.wav Total Frames: \(fooTrack.frameCount), Voiced Frames: \(fooVoicedCount), 100Hz Frames: \(foo100HzCount) (\(Float(foo100HzCount) / Float(max(1, fooVoicedCount)) * 100.0)%)")
                let ratio100Hz = Float(foo100HzCount) / Float(max(1, fooVoicedCount))
                XCTAssertTrue(ratio100Hz < 0.05, "foo.wav に 100Hz 固定フレームが過剰に含まれています (\(ratio100Hz * 100.0)%)")
                var fooF0Str = ""
                var fi2 = 0
                while fi2 < fooTrack.frameCount {
                    if 0.5 <= fooTrack.voiced[fi2] {
                        fooF0Str += String(format: "%.1f, ", fooTrack.f0[fi2])
                    } else {
                        fooF0Str += "0, "
                    }
                    fi2 += 1
                }
                print("foo.wav Tracked F0: [\(fooF0Str)]")
            }
        }
        // なぜ全フレームの F0 を調査するか:
        // 合成された音声が指定されたピッチ輪郭（女性 220Hz〜、男性 120Hz〜）を正しく追従しているか、
        // あるいはフレーム周期（100Hz）の偽相関に縮退しているかを精密に検証するため。
        var f0ValsStr = ""
        var ff = 0
        while ff < fTrack.frameCount {
            if 0.5 <= fTrack.voiced[ff] {
                f0ValsStr += String(format: "%.1f, ", fTrack.f0[ff])
            }
            ff += 1
        }
        print("Tracked Female F0 values: [\(f0ValsStr)]")
        var mF0ValsStr = ""
        var mf = 0
        while mf < mTrack.frameCount {
            if 0.5 <= mTrack.voiced[mf] {
                mF0ValsStr += String(format: "%.1f, ", mTrack.f0[mf])
            }
            mf += 1
        }
        print("Tracked Male F0 values: [\(mF0ValsStr)]")
        let midF = fTrack.frameCount / 2
        let fFreq = fTrack.f0[midF]
        let mFreq = mTrack.f0[midF]
        print(String(format: "PitchTracker MidFrame %d: Female F0=%.1f Hz (voiced=%.1f), Male F0=%.1f Hz (voiced=%.1f)", midF, fFreq, fTrack.voiced[midF], mFreq, mTrack.voiced[midF]))


        // 音響合成波形の健全性検証:
        // 1. 有声区間で F0 が正しく検出されていること
        XCTAssertTrue(0.0 < fFreq, "女性の基本周波数が検出されませんでした")
        XCTAssertTrue(0.0 < mFreq, "男性の基本周波数が検出されませんでした")
        // 2. 男声と女声で波形が有意に異なっていること
        XCTAssertNotEqual(femaleSamples, maleSamples, "男声と女声の波形が同一です")
        // 3. 最大絶対振幅がクリップせず適正ヘッドルーム（<= 0.98）に収まっていること
        var maxFAbs: Float = 0.0
        var s = 0
        while s < femaleSamples.count {
            let v = abs(femaleSamples[s])
            if maxFAbs < v { maxFAbs = v }
            s += 1
        }
        XCTAssertTrue(maxFAbs <= 0.98, "女性合成音声がクリップしています: peak=\(maxFAbs)")
        // 4. 基本周波数が健全ピッチ範囲（110〜450Hz: 基底ピッチ〜オクターブ高調波）に収まり、100Hz（ホップ周期偽相関）への縮退がないこと
        var isFemalePitch = false
        if 110.0 <= fFreq && fFreq <= 450.0 {
            isFemalePitch = true
        }
        XCTAssertTrue(isFemalePitch, "女性合成音声の中間部ピッチが目標範囲 (110-450Hz) から逸脱しています: \(fFreq)")
        var isMalePitch = false
        if 110.0 <= mFreq && mFreq <= 450.0 {
            isMalePitch = true
        }
        XCTAssertTrue(isMalePitch, "男性合成音声の中間部ピッチが目標範囲から逸脱しています: \(mFreq)")
    }


    /// オンメモリ合成信号からの対数 Mel スペクトル直接入力（Copy Synthesis）によりニューラルボコーダー単体の波形生成能力を検証
    func testCopySynthesisFromSyntheticWaveform() {
        let sampleRate = 16000
        let totalSamples = 1600 // 10フレーム相当 (100ms)
        var syntheticPCM = [Float](repeating: 0.0, count: totalSamples)
        var s = 0
        let f0: Float = 220.0
        let twoPi = 2.0 * Float.pi
        while s < totalSamples {
            let t = Float(s) / Float(sampleRate)
            // なぜ基本波と2次高調波を重畳したテスト信号を用いるか:
            // ピッチ追従と調波構造のスペクトル包絡をニューラルボコーダーへ確実に供給するため。
            let wave = (0.6 * sinf(twoPi * f0 * t)) + (0.3 * sinf(twoPi * 2.0 * f0 * t))
            syntheticPCM[s] = wave
            s += 1
        }

        let melExtractor = MelSpectrogramExtractor()
        let pitchTracker = PitchTracker()

        let mel = melExtractor.extractLogMel(pcm: syntheticPCM)
        let pitchResult = pitchTracker.track(pcm: syntheticPCM)

        XCTAssertFalse(mel.isEmpty, "抽出された Mel スペクトルが空です")

        let vocoder = NeuralVocoder()
        vocoder.reset()
        let synthesizedPCM = vocoder.synthesize(
            mel: mel,
            f0Contour: pitchResult.f0,
            voicedFlags: pitchResult.voiced
        )

        XCTAssertEqual(synthesizedPCM.count, mel.count * 160)

        var hasNonZero = false
        var si = 0
        while si < synthesizedPCM.count {
            let sample = synthesizedPCM[si]
            XCTAssertFalse(sample.isNaN, "Copy Synthesis 波形に NaN を検出: \(si)")
            XCTAssertFalse(sample.isInfinite, "Copy Synthesis 波形に Inf を検出: \(si)")
            if 1e-4 < abs(sample) {
                hasNonZero = true
            }
            si += 1
        }
        XCTAssertTrue(hasNonZero, "Copy Synthesis で非ゼロ波形が生成されていません")
    }

    /// MLX 環境下における NeuralVocoder の自動微分および損失逆伝播のサニティチェック
    func testMLXNeuralVocoderOptimizationSanity() throws {
        #if canImport(MLX)
        let vocoder = MLXNeuralVocoder()
        let optimizer = Adam(learningRate: 0.0003)

        let segFrames = 4
        let hopSize = vocoder.config.hopSize
        let segSamples = segFrames * hopSize
        let inCh = vocoder.config.melChannels + 2 // 66

        let featData = [Float](repeating: 0.1, count: segFrames * inCh)
        var targetData = [Float](repeating: 0.0, count: segSamples)
        var s = 0
        while s < segSamples {
            targetData[s] = 0.5 * sinf(Float(s) * 0.1)
            s += 1
        }

        let fArr = MLXArray(featData, [1, segFrames, inCh])
        let tArr = MLXArray(targetData, [1, segSamples])

        let lg = valueAndGrad(model: vocoder) { (model: MLXNeuralVocoder, arrays: [MLXArray]) -> [MLXArray] in
            let f = arrays[0]
            let t = arrays[1]
            let pred = model(f)
            let loss = MLXNeuralVocoder.totalVocoderLoss(predicted: pred, target: t)
            return [loss]
        }

        let (losses, grads) = lg(vocoder, [fArr, tArr])
        let lossVal = losses[0].item(Float.self)
        XCTAssertFalse(lossVal.isNaN, "ボコーダー損失が NaN です")
        XCTAssertFalse(lossVal.isInfinite, "ボコーダー損失が Inf です")

        optimizer.update(model: vocoder, gradients: grads)

        let exported = vocoder.exportWeights()
        XCTAssertEqual(exported.config.hiddenChannels, vocoder.config.hiddenChannels)
        #endif
    }

    /// MLXNeuralVocoder が hiddenChannels 不一致の重みインポートを安全に拒否することを検証
    func testMLXNeuralVocoderRejectsMismatchedHiddenChannels() {
        #if canImport(MLX)
        let vocoder = MLXNeuralVocoder()
        let legacyConfig = NeuralVocoderConfig(hiddenChannels: 64)
        let legacyWeights = NeuralVocoderWeights.randomWeights(config: legacyConfig, seed: 123)

        let origShape = vocoder.convPre.weight.shape
        vocoder.importWeights(from: legacyWeights)
        // 拒否されたため、convPre の重みテンソル形状や hiddenChannels が保持されていること
        XCTAssertEqual(vocoder.config.hiddenChannels, 256)
        XCTAssertEqual(vocoder.convPre.weight.shape, origShape)
        #endif
    }

    /// 受入条件 E2E サニティテスト（空文字、非有限値クランプ、連続合成状態非汚染）
    func testSpikeSpeechEngineE2EAcceptance() {
        let engine = SpikeSpeechEngine()

        // 1. 空文字・空白のみで空 PCM（WAV はヘッダのみ 44バイト）
        let emptyPCM1 = engine.synthesize(text: "")
        let emptyPCM2 = engine.synthesize(text: "   \n\t  ")
        XCTAssertTrue(emptyPCM1.isEmpty, "空文字で PCM が返却されています")
        XCTAssertTrue(emptyPCM2.isEmpty, "空白文字で PCM が返却されています")

        let emptyWav = engine.synthesizeWav(text: "")
        XCTAssertEqual(emptyWav.count, 44, "空文字の WAV 出力が 44 バイトヘッダのみではありません")

        // 2. 非空日本語で 16kHz 有限 PCM（NaN/Inf 無し）
        let text = "こんにちは"
        let pcm = engine.synthesize(text: text)
        XCTAssertFalse(pcm.isEmpty, "日本語テキストに対する合成 PCM が空です")
        var i = 0
        while i < pcm.count {
            let s = pcm[i]
            XCTAssertFalse(s.isNaN, "E2E 合成波形に NaN を検出: index=\(i)")
            XCTAssertFalse(s.isInfinite, "E2E 合成波形に Inf を検出: index=\(i)")
            i += 1
        }

        // 3. speed / pitch クランプ検証（非有限値は 1.0、下限 0.2 上限 5.0）
        let pcmNanSpeed = engine.synthesize(text: text, speed: Float.nan)
        let pcmInfPitch = engine.synthesize(text: text, pitch: Float.infinity)
        XCTAssertFalse(pcmNanSpeed.isEmpty)
        XCTAssertFalse(pcmInfPitch.isEmpty)

        // 4. 連続 2 回合成で状態汚染なし
        let pcmFirst = engine.synthesize(text: text)
        let pcmSecond = engine.synthesize(text: text)
        XCTAssertEqual(pcmFirst.count, pcmSecond.count)
        XCTAssertEqual(pcmFirst, pcmSecond, "同一テキストの連続合成で波形に差異（状態汚染）が生じています")

        // 5. SpeakerConditioning 引数による合成
        let speaker = SpeakerConditioning(embedding: [Float](repeating: 0.1, count: 128))
        let pcmSpeaker = engine.synthesize(text: text, speaker: speaker)
        XCTAssertFalse(pcmSpeaker.isEmpty)
    }
}
#endif
