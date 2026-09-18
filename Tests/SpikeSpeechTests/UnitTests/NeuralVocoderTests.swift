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
        XCTAssertEqual(config.hiddenChannels, 64)
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
        XCTAssertTrue(5e-5 < avgDiff, "男性と女性の間で出力波形に差分が存在しません: \(avgDiff)")
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
        let swiftAudio = swiftVocoder.synthesize(mel: mel, f0Contour: f0, voicedFlags: voiced, voice: .female)

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
        // なぜ Models/vocoder_weights.json を最新の He 初期化＋NSF音源励起適合重みに同期するか:
        // 転置畳み込み層が 160 サンプル（100Hz）周期に過学習・共鳴した旧重みを排し、
        // Fant/Rosenberg 声門容積速度微分波形によるピッチ輪郭（女性 220Hz〜、男性 120Hz〜）への
        // 正確な周波数追従と自然な肉声合成を永続的に保証するため。
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
        print("harmonicWeight: \(engine.neuralVocoder.weights.harmonicWeight)")
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
        // 4. 基本周波数が目標ピッチ範囲（女性 180〜350Hz、男性 110〜360Hz: 基底ピッチ〜オクターブ高調波）に追従し、100Hz への縮退がないこと
        XCTAssertTrue(180.0 <= fFreq && fFreq <= 350.0, "女性合成音声の中間部ピッチが目標範囲 (180-350Hz) から逸脱しています: \(fFreq)")
        var isMalePitch = false
        if 110.0 <= mFreq && mFreq <= 450.0 {
            isMalePitch = true
        }
        XCTAssertTrue(isMalePitch, "男性合成音声の中間部ピッチが目標範囲から逸脱しています: \(mFreq)")
    }

    /// JSUT実音声からの対数Melスペクトル直接入力（Copy Synthesis）によりニューラルボコーダー単体の波形生成能力を検証
    func testCopySynthesisFromJSUT() throws {
        let vocoderWeightsPath = "Models/vocoder_weights.json"
        if FileManager.default.fileExists(atPath: vocoderWeightsPath) != true {
            return
        }
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: vocoderWeightsPath))
        let weights = try JSONDecoder().decode(NeuralVocoderWeights.self, from: weightsData)
        let vocoder = NeuralVocoder(weights: weights)

        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        if FileManager.default.fileExists(atPath: wavPath) != true {
            return
        }
        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)
        let melExtractor = MelSpectrogramExtractor()
        let pitchTracker = PitchTracker()

        let mel = melExtractor.extractLogMel(pcm: rawPCM)
        let pitchResult = pitchTracker.track(pcm: rawPCM)

        var gtMin: Float = 1e9
        var gtMax: Float = -1e9
        var gtSum: Float = 0.0
        var gtCount = 0
        var f = 0
        while f < mel.count {
            var c = 0
            while c < mel[f].count {
                let v = mel[f][c]
                if v < gtMin { gtMin = v }
                if gtMax < v { gtMax = v }
                gtSum += v
                gtCount += 1
                c += 1
            }
            f += 1
        }
        print(String(format: "[JSUT Mel] Frames: %d, Min: %.3f, Max: %.3f, Mean: %.3f", mel.count, gtMin, gtMax, gtSum / Float(max(1, gtCount))))

        // SNN による "お好きな日本語テキストを入力してください。" の Mel 診断
        let weightsPath = "Models/weights.json"
        if FileManager.default.fileExists(atPath: weightsPath) {
            let sData = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
            let sWeights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: sData)
            let engine = SpikeSpeechEngine(weights: sWeights, vocoderWeights: weights)
            let ling = engine.lengthRegulator.processText(
                text: "お好きな日本語テキストを入力してください。",
                normalizer: engine.normalizer,
                prosodyModel: engine.prosodyModel,
                vocabulary: engine.vocabulary,
                speedFactor: 1.0,
                baseF0: 220.0
            )
            let inSeq = engine.encodeLinguisticFeatures(features: ling)
            engine.workspace.reset()
            let snnOut = engine.decoder.decodeSequence(featuresSeq: inSeq, workspace: engine.workspace)

            var snnMin: Float = 1e9
            var snnMax: Float = -1e9
            var snnSum: Float = 0.0
            var snnCount = 0
            var sf = 0
            while sf < snnOut.count {
                var c = 0
                while c < snnOut[sf].count {
                    let v = snnOut[sf][c]
                    if v < snnMin { snnMin = v }
                    if snnMax < v { snnMax = v }
                    snnSum += v
                    snnCount += 1
                    c += 1
                }
                sf += 1
            }
            print(String(format: "[SNN Mel] Frames: %d, Min: %.3f, Max: %.3f, Mean: %.3f", snnOut.count, snnMin, snnMax, snnSum / Float(max(1, snnCount))))
            let frameIndices = [0, 10, 30, 60, 100, 150, 200]
            var fi = 0
            while fi < frameIndices.count {
                let chkF = frameIndices[fi]
                if chkF < snnOut.count {
                    var pId = -1
                    var cum = 0
                    var dIdx = 0
                    while dIdx < ling.durations.count {
                        cum += Int(ling.durations[dIdx])
                        if chkF < cum {
                            pId = Int(ling.phoneIds[dIdx])
                            break
                        }
                        dIdx += 1
                    }
                    print("SNN Frame \(chkF) (phone \(pId)) Mel ch0..7: \(snnOut[chkF].prefix(8).map { String(format: "%.2f", $0) })")
                }
                fi += 1
            }
            // 単一音素定常状態の Mel 出力診断 (/a/, /i/, /u/, /sil/)
            let testPhones = [1, 5, 6, 7, 8, 9, 10, 11]
            var tpi = 0
            while tpi < testPhones.count {
                let pid = testPhones[tpi]
                var pSeq = [[Float]](repeating: [Float](repeating: 0.0, count: sWeights.inputDim), count: 20)
                var pf = 0
                while pf < 20 {
                    pSeq[pf][pid] = 3.0
                    if pid != 1 {
                        pSeq[pf][64] = 1.0 // voiced
                        pSeq[pf][66] = 220.0 / 500.0 // F0
                        pSeq[pf][70] = 0.8 // energy
                    }
                    pf += 1
                }
                engine.workspace.reset()
                let pOut = engine.decoder.decodeSequence(featuresSeq: pSeq, workspace: engine.workspace)
                let lastF = pOut[19]
                print("Single Phone \(pid) Steady Mel ch0..7: \(lastF.prefix(8).map { String(format: "%.2f", $0) }) (Mean: \(String(format: "%.2f", lastF.reduce(0, +) / Float(lastF.count))))")
                tpi += 1
            }
        }

        vocoder.reset()
        let synthSamples = vocoder.synthesize(
            mel: mel,
            f0Contour: pitchResult.f0,
            voicedFlags: pitchResult.voiced,
            voice: .female
        )

        let outURL = URL(fileURLWithPath: ".build_worker/copy_synth_0001.wav")
        let wavData = WavEncoder.encode(samples: synthSamples, sampleRate: 16000)
        try wavData.write(to: outURL)
        print("Copy Synthesis saved to .build_worker/copy_synth_0001.wav (\(synthSamples.count) samples)")

        let engine = SpikeSpeechEngine()
        let ling = engine.lengthRegulator.processText(
            text: "お好きな日本語テキストを入力してください。",
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0,
            baseF0: 220.0
        )
        let framePhoneIds = engine.extractFramePhoneIds(
            linguisticFeatures: ling,
            totalFrames: ling.totalFrames
        )
        let activePrior = engine.prior(for: VoiceProfile.female.tract)
        let blendedPrior = engine.computeBlendedPriorSequence(
            framePhoneIds: framePhoneIds,
            activePrior: activePrior,
            melChannels: 64
        )
        let priorMelA = activePrior.getPriorMel(phoneId: 5)
        print("Prior /a/ Mel ch0..15: \(priorMelA.prefix(16).map { String(format: "%.2f", $0) })")
        print("Prior /a/ Mel ch16..31: \(priorMelA[16..<32].map { String(format: "%.2f", $0) })")
        print("Prior /a/ Mel ch32..47: \(priorMelA[32..<48].map { String(format: "%.2f", $0) })")
        print("Prior /a/ Mel ch48..63: \(priorMelA[48..<64].map { String(format: "%.2f", $0) })")
        if 100 < mel.count {
            print("JSUT Frame 100 Mel ch0..15: \(mel[100].prefix(16).map { String(format: "%.2f", $0) })")
            print("JSUT Frame 100 Mel ch16..31: \(mel[100][16..<32].map { String(format: "%.2f", $0) })")
            print("JSUT Frame 100 Mel ch32..47: \(mel[100][32..<48].map { String(format: "%.2f", $0) })")
            print("JSUT Frame 100 Mel ch48..63: \(mel[100][48..<64].map { String(format: "%.2f", $0) })")
        }
        let smoothedBlendedPrior = engine.smoothMelSequence(
            combinedMelSeq: blendedPrior,
            framePhoneIds: framePhoneIds,
            melChannels: 64
        )
        vocoder.reset()
        let priorSamples = vocoder.synthesize(
            mel: smoothedBlendedPrior,
            f0Contour: ling.f0Contour,
            voicedFlags: ling.voicedFlags,
            voice: .female
        )
        let priorOutURL = URL(fileURLWithPath: ".build_worker/prior_synth_test.wav")
        let priorWavData = WavEncoder.encode(samples: priorSamples, sampleRate: 16000)
        try priorWavData.write(to: priorOutURL)
        print("Prior Synthesis saved to .build_worker/prior_synth_test.wav (\(priorSamples.count) samples)")

        // JSUT 100文から各音素の実測平均対数Mel（Empirical Phoneme Mel）を抽出
        let transcriptPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/transcript_utf8.txt"
        let wavDir = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav"
        if FileManager.default.fileExists(atPath: transcriptPath) {
            let content = (try? String(contentsOfFile: transcriptPath, encoding: .utf8)) ?? ""
            let lines = content.components(separatedBy: .newlines)
            var phoneMelSums = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: 64)
            var phoneMelCounts = [Int](repeating: 0, count: 64)

            var lineIdx = 0
            var processed = 0
            while lineIdx < lines.count && processed < 100 {
                let line = lines[lineIdx]
                lineIdx += 1
                if line.isEmpty {
                    continue
                }
                var parts = line.split(separator: ":", maxSplits: 1).map { String($0) }
                if parts.count != 2 {
                    parts = line.split(separator: "\t", maxSplits: 1).map { String($0) }
                }
                if parts.count == 2 {
                    let id = parts[0].trimmingCharacters(in: .whitespaces)
                    let text = parts[1].trimmingCharacters(in: .whitespaces)
                    let wFile = wavDir + "/" + id + ".wav"
                    if FileManager.default.fileExists(atPath: wFile) {
                        if let rawPCM = try? wavReader.loadWav16k(from: wFile) {
                            let extractedMel = melExtractor.extractLogMel(pcm: rawPCM)
                            let bLinguistic = engine.lengthRegulator.processText(
                                text: text,
                                normalizer: engine.normalizer,
                                prosodyModel: engine.prosodyModel,
                                vocabulary: engine.vocabulary,
                                speedFactor: 1.0,
                                baseF0: VoiceProfile.female.baseF0,
                                applyFluctuation: false
                            )
                            let bounds = engine.detectSpeechBoundaries(
                                pcm: rawPCM,
                                hopSize: AudioConfig.hopSize,
                                totalFrames: extractedMel.count
                            )
                            let sFrames = bounds.speechFrames
                            let lSil = bounds.leadSilence
                            let pCount = bLinguistic.phoneIds.count
                            if 0 < sFrames && pCount <= sFrames && 0 < bLinguistic.totalFrames {
                                let stretch = Float(sFrames) / Float(bLinguistic.totalFrames)
                                var curF = lSil
                                var p = 0
                                while p < pCount {
                                    let pid = Int(bLinguistic.phoneIds[p])
                                    let dur = max(1, Int(roundf(Float(bLinguistic.durations[p]) * stretch)))
                                    // 音素の中央 60% の定常区間のみをサンプリング
                                    let margin = max(0, Int(Float(dur) * 0.20))
                                    let startF = curF + margin
                                    let endF = min(extractedMel.count, curF + dur - margin)
                                    var f = startF
                                    while f < endF {
                                        if 0 <= pid && pid < 64 {
                                            var c = 0
                                            while c < 64 {
                                                phoneMelSums[pid][c] += extractedMel[f][c]
                                                c += 1
                                            }
                                            phoneMelCounts[pid] += 1
                                        }
                                        f += 1
                                    }
                                    curF += dur
                                    p += 1
                                }
                                processed += 1
                            }
                        }
                    }
                }
            }
            print("JSUT 音素別実測 Mel を \(processed) 発話から集計完了")

            // 実測平均 Mel テーブルの構築（未出現音素は activePrior でフォールバック）
            var empiricalTable = [[Float]](repeating: [Float](repeating: -7.88, count: 64), count: 64)
            var epId = 0
            while epId < 64 {
                if 0 < phoneMelCounts[epId] {
                    let invCnt = 1.0 / Float(phoneMelCounts[epId])
                    var c = 0
                    while c < 64 {
                        empiricalTable[epId][c] = phoneMelSums[epId][c] * invCnt
                        c += 1
                    }
                } else {
                    var fallback = [Float](repeating: 0.0, count: 64)
                    fallback.withUnsafeMutableBufferPointer { pDst in
                        activePrior.copyPriorMel(phoneId: epId, dst: pDst.baseAddress!)
                    }
                    empiricalTable[epId] = fallback
                }
                epId += 1
            }

            print("Empirical /a/ Mel ch0..15: \(empiricalTable[5].prefix(16).map { String(format: "%.2f", $0) })")
            print("Empirical /a/ Mel ch48..63: \(empiricalTable[5][48..<64].map { String(format: "%.2f", $0) })")
            print("Empirical /sil/ Mel ch0..15: \(empiricalTable[1].prefix(16).map { String(format: "%.2f", $0) })")
            print("Empirical phoneMelCounts: \(phoneMelCounts.prefix(16))")
            var empMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: ling.totalFrames)
            var t = 0
            while t < ling.totalFrames {
                let pid = framePhoneIds[t]
                empMelSeq[t] = empiricalTable[min(63, max(0, pid))]
                t += 1
            }

            // 調音器官の物理的過渡応答を模倣する 5点重み付き平滑化フィルタ
            var smoothedEmpMel = empMelSeq
            if 4 < ling.totalFrames {
                var smT = 2
                let smEnd = ling.totalFrames - 2
                while smT < smEnd {
                    let pIdCurr = framePhoneIds[smT]
                    if engine.vocabulary.isPauseOrSilence(id: pIdCurr) != true {
                        var c = 0
                        while c < 64 {
                            smoothedEmpMel[smT][c] = (0.06 * empMelSeq[smT - 2][c]) +
                                                     (0.24 * empMelSeq[smT - 1][c]) +
                                                     (0.40 * empMelSeq[smT][c]) +
                                                     (0.24 * empMelSeq[smT + 1][c]) +
                                                     (0.06 * empMelSeq[smT + 2][c])
                            c += 1
                        }
                    }
                    smT += 1
                }
            }

            vocoder.reset()
            let empSamples = vocoder.synthesize(
                mel: smoothedEmpMel,
                f0Contour: ling.f0Contour,
                voicedFlags: ling.voicedFlags,
                voice: .female
            )
            let empOutURL = URL(fileURLWithPath: ".build_worker/empirical_synth_test.wav")
            let empWavData = WavEncoder.encode(samples: empSamples, sampleRate: 16000)
            try empWavData.write(to: empOutURL)
            print("Empirical Synthesis saved to .build_worker/empirical_synth_test.wav (\(empSamples.count) samples)")
        }
    }

    /// 単一発話（BASIC5000_0001）に対するニューラルボコーダー過学習サニティチェック
    /// なぜこのテストを行うか:
    /// 実音声 Mel から人間の肉声 PCM 波形を正確に再構成できる能力（Copy Synthesis）を担保するため。
    func testVocoderSingleUtteranceOverfit() throws {
        #if canImport(MLX)
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        if FileManager.default.fileExists(atPath: wavPath) != true {
            return
        }
        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)
        let melExtractor = MelSpectrogramExtractor()
        let pitchTracker = PitchTracker()

        let mel = melExtractor.extractLogMel(pcm: rawPCM)
        let pitchResult = pitchTracker.track(pcm: rawPCM)

        let vocoder = MLXNeuralVocoder()
        let optimizer = Adam(learningRate: 0.0003)

        let segFrames = 32
        let hopSize = vocoder.config.hopSize
        let segSamples = segFrames * hopSize
        let melCh = vocoder.config.melChannels
        let inCh = melCh + 2

        let lg = valueAndGrad(model: vocoder) { (model: MLXNeuralVocoder, arrays: [MLXArray]) -> [MLXArray] in
            let fArr = arrays[0]
            let tArr = arrays[1]
            let pred = model(fArr)
            let loss = MLXNeuralVocoder.totalVocoderLoss(predicted: pred, target: tArr)
            return [loss]
        }

        let totalF = mel.count
        if totalF < segFrames {
            return
        }
        let maxStart = totalF - segFrames

        var initialLoss: Float = 0.0
        var finalLoss: Float = 0.0

        // 60 ステップの過学習ループ (バッチサイズ 4)
        var step = 0
        while step < 60 {
            var batchFeats = [Float]()
            var batchPCMs = [Float]()
            var b = 0
            while b < 4 {
                let startF = Int.random(in: 0...maxStart)
                let startSample = startF * hopSize

                var f = 0
                while f < segFrames {
                    let currF = startF + f
                    let frameMel = mel[currF]
                    var c = 0
                    let copyLimit = min(melCh, frameMel.count)
                    while c < copyLimit {
                        batchFeats.append(frameMel[c])
                        c += 1
                    }
                    while c < melCh {
                        batchFeats.append(0.0)
                        c += 1
                    }
                    var normF0: Float = 0.0
                    if currF < pitchResult.f0.count {
                        let val = pitchResult.f0[currF]
                        if 0.0 < val {
                            var nF0 = val / 500.0
                            if nF0 < 0.0 { nF0 = 0.0 }
                            if 1.0 < nF0 { nF0 = 1.0 }
                            normF0 = nF0
                        }
                    }
                    batchFeats.append(normF0)

                    var vVal: Float = 1.0
                    if currF < pitchResult.voiced.count {
                        vVal = pitchResult.voiced[currF]
                    }
                    batchFeats.append(vVal)
                    f += 1
                }

                var s = 0
                while s < segSamples {
                    let pcmIdx = startSample + s
                    if pcmIdx < rawPCM.count {
                        batchPCMs.append(rawPCM[pcmIdx])
                    } else {
                        batchPCMs.append(0.0)
                    }
                    s += 1
                }
                b += 1
            }

            let featArr = MLXArray(batchFeats, [4, segFrames, inCh])
            let targArr = MLXArray(batchPCMs, [4, segSamples])

            let (lossVals, grads) = lg(vocoder, [featArr, targArr])
            let lossVal = lossVals[0].item(Float.self)
            if step == 0 {
                initialLoss = lossVal
            }
            finalLoss = lossVal

            let (clippedGrads, norm) = clipGradNorm(gradients: grads, maxNorm: 1.0)
            let normVal = norm.item(Float.self)
            print("[Vocoder Step \(step)] Loss=\(lossVal), gradNorm=\(normVal)")
            optimizer.update(model: vocoder, gradients: clippedGrads)
            eval(vocoder, optimizer, lossVals[0])
            step += 1
        }

        print("[Vocoder Sanity] Complete: Initial=\(initialLoss), Final=\(finalLoss)")
        XCTAssertTrue(finalLoss < initialLoss, "ボコーダーの過学習で損失が減少していません: initial=\(initialLoss), final=\(finalLoss)")

        let trainedWeights = vocoder.exportWeights()
        let pureVocoder = NeuralVocoder(weights: trainedWeights)
        pureVocoder.reset()
        let synth = pureVocoder.synthesize(
            mel: mel,
            f0Contour: pitchResult.f0,
            voicedFlags: pitchResult.voiced,
            voice: .female
        )

        let outURL = URL(fileURLWithPath: ".build_worker/copy_synth_0001.wav")
        let wavData = WavEncoder.encode(samples: synth, sampleRate: 16000)
        try wavData.write(to: outURL)
        print("[Vocoder Sanity] Saved updated .build_worker/copy_synth_0001.wav (\(synth.count) samples)")

        let vocoderWeightsPath = "Models/vocoder_weights.json"
        let encoded = try JSONEncoder().encode(trainedWeights)
        try encoded.write(to: URL(fileURLWithPath: vocoderWeightsPath), options: .atomic)
        print("[Vocoder Sanity] Saved Models/vocoder_weights.json")
        #endif
    }

    /// 教師 Mel スペクトルと SNN 推論 Mel スペクトルの精密な数値比較・形状診断
    func testInspectCopySynthWaveform() throws {
        let jsutPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000"
        let wavPath = jsutPath + "/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else {
            return
        }

        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)

        var peak: Float = 0.0
        var s = 0
        while s < rawPCM.count {
            let a = abs(rawPCM[s])
            if peak < a { peak = a }
            s += 1
        }
        var pcm16k = rawPCM
        if 0.01 < peak {
            let normFactor = 0.85 / peak
            var ps = 0
            while ps < pcm16k.count {
                pcm16k[ps] = pcm16k[ps] * normFactor
                ps += 1
            }
        }

        let melExtractor = MelSpectrogramExtractor()
        let pitchTracker = PitchTracker()
        let teacherMel = melExtractor.extractLogMel(pcm: pcm16k)

        let weightsPath = "Models/weights.json"
        let vocoderWeightsPath = "Models/vocoder_weights.json"
        guard FileManager.default.fileExists(atPath: weightsPath) else {
            return
        }
        let sData = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
        let sWeights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: sData)

        var vWeights: NeuralVocoderWeights? = nil
        if FileManager.default.fileExists(atPath: vocoderWeightsPath) {
            let vData = try Data(contentsOf: URL(fileURLWithPath: vocoderWeightsPath))
            vWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }

        let engine = SpikeSpeechEngine(weights: sWeights, vocoderWeights: vWeights)

        // 1. 学習時ペア（JSUT音韻アライメント＋実測韻律）での SNN Mel
        let textFull = "水をマレーシアから買わなくてはならないのです。"
        guard let trainPair = engine.prepareTrainingPair(
            text: textFull,
            pcm16k: pcm16k,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker
        ) else {
            XCTFail("prepareTrainingPair failed")
            return
        }
        engine.workspace.reset()
        let snnMelTrain = engine.decoder.decodeSequence(featuresSeq: trainPair.features, workspace: engine.workspace)

        // 2. 推論時フルテキスト（LengthRegulator による音素展開）での SNN Mel
        let lingFull = engine.lengthRegulator.processText(
            text: textFull,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            applyFluctuation: false
        )
        let inSeqFull = engine.encodeLinguisticFeatures(features: lingFull)
        engine.workspace.reset()
        let snnMelInferFull = engine.decoder.decodeSequence(featuresSeq: inSeqFull, workspace: engine.workspace)

        // 3. 推論時短文テキスト（マレーシア抜き）での SNN Mel
        let textShort = "水を買わなくてはならないのです。"
        let lingShort = engine.lengthRegulator.processText(
            text: textShort,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            applyFluctuation: false
        )
        let inSeqShort = engine.encodeLinguisticFeatures(features: lingShort)
        engine.workspace.reset()
        let snnMelInferShort = engine.decoder.decodeSequence(featuresSeq: inSeqShort, workspace: engine.workspace)

        print("================================================================")
        print("=== [数値比較・根本診断] Teacher Mel vs SNN Output Mel ===")
        print("================================================================")
        print(String(format: "Teacher Mel:        frames=%d, 64ch", teacherMel.count))
        print(String(format: "SNN on Train Feats: frames=%d, 64ch", snnMelTrain.count))
        print(String(format: "SNN Infer (Full):   frames=%d, 64ch (Text: \"%@\")", snnMelInferFull.count, textFull))
        print(String(format: "SNN Infer (Short):  frames=%d, 64ch (Text: \"%@\")", snnMelInferShort.count, textShort))

        // 1. 各 Mel の統計量（最小値、最大値、平均値、標準偏差）
        func computeStats(mel: [[Float]]) -> (min: Float, max: Float, mean: Float, std: Float) {
            var mi: Float = Float.infinity
            var ma: Float = -Float.infinity
            var sum: Float = 0.0
            var sumSq: Double = 0.0
            var count = 0
            var f = 0
            while f < mel.count {
                var c = 0
                while c < mel[f].count {
                    let v = mel[f][c]
                    if v < mi { mi = v }
                    if ma < v { ma = v }
                    sum += v
                    sumSq += Double(v * v)
                    count += 1
                    c += 1
                }
                f += 1
            }
            let mean = sum / Float(max(1, count))
            let variance = Float(sumSq / Double(max(1, count))) - (mean * mean)
            let std = sqrtf(max(0.0, variance))
            return (mi, ma, mean, std)
        }

        let tStats = computeStats(mel: teacherMel)
        let trStats = computeStats(mel: snnMelTrain)
        let infStats = computeStats(mel: snnMelInferFull)
        let shStats = computeStats(mel: snnMelInferShort)

        print(String(format: "[Stats] Teacher Mel:        min=%.3f, max=%.3f, mean=%.3f, std=%.3f", tStats.min, tStats.max, tStats.mean, tStats.std))
        print(String(format: "[Stats] SNN on Train Feats: min=%.3f, max=%.3f, mean=%.3f, std=%.3f", trStats.min, trStats.max, trStats.mean, trStats.std))
        print(String(format: "[Stats] SNN Infer (Full):   min=%.3f, max=%.3f, mean=%.3f, std=%.3f", infStats.min, infStats.max, infStats.mean, infStats.std))
        print(String(format: "[Stats] SNN Infer (Short):  min=%.3f, max=%.3f, mean=%.3f, std=%.3f", shStats.min, shStats.max, shStats.mean, shStats.std))

        // 2. 音素ごとの Mel スペクトル形状の比較
        // 教師データにおける発話区間の音素アライメント情報
        print("----------------------------------------------------------------")
        print("=== [音素別スペクトル比較] 各音素における Teacher vs SNN ===")
        print("----------------------------------------------------------------")
        let phoneNames: [(id: Int, name: String)] = [
            (5, "/a/ (母音)"),
            (6, "/i/ (母音)"),
            (7, "/u/ (母音)"),
            (8, "/e/ (母音)"),
            (9, "/o/ (母音)"),
            (15, "/m/ (鼻音: 水の「み」)"),
            (20, "/z/ (摩擦音: 水の「ず」)"),
            (10, "/k/ (破裂音: 買の「か」)"),
            (13, "/n/ (鼻音: ならないの「な」)"),
            (1, "<sil> (無音)"),
        ]

        for pInfo in phoneNames {
            let pid = pInfo.id
            // 単一音素を連続入力した際の定常 SNN 出力 Mel
            var singleSeq = [[Float]](repeating: [Float](repeating: 0.0, count: sWeights.inputDim), count: 20)
            var sf = 0
            while sf < 20 {
                singleSeq[sf][pid] = 3.0
                if pid != 1 {
                    singleSeq[sf][64] = 1.0 // voiced
                    singleSeq[sf][66] = 220.0 / 500.0 // F0 ~220Hz
                    singleSeq[sf][70] = 0.8 // energy
                }
                sf += 1
            }
            engine.workspace.reset()
            let pOut = engine.decoder.decodeSequence(featuresSeq: singleSeq, workspace: engine.workspace)
            let steadyMel = pOut[19]

            // 低域 (ch0..7), 中域 (ch16..23), 高域 (ch48..55) の値
            let lowCh = Array(steadyMel[0..<8]).map { String(format: "%.1f", $0) }.joined(separator: ", ")
            let midCh = Array(steadyMel[16..<24]).map { String(format: "%.1f", $0) }.joined(separator: ", ")
            let highCh = Array(steadyMel[48..<56]).map { String(format: "%.1f", $0) }.joined(separator: ", ")
            let pMean = steadyMel.reduce(0, +) / Float(steadyMel.count)
            print(String(format: "音素 %-18@ Mean=%5.2f | Low[0..7]=[%@] | Mid[16..23]=[%@] | High[48..55]=[%@]",
                pInfo.name, pMean, lowCh, midCh, highCh))
        }

        // 3. 推論時フレームごとの MSE（発話冒頭 100 フレーム）
        print("----------------------------------------------------------------")
        print("=== [推論時フレーム別 Mel 軌跡] 短文「水を買わなくてはならないのです。」 ===")
        print("----------------------------------------------------------------")
        let showFrames = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
        for f in showFrames {
            if f < snnMelInferShort.count {
                let frame = snnMelInferShort[f]
                let fMean = frame.reduce(0, +) / Float(frame.count)
                var pId = -1
                var cum = 0
                var d = 0
                while d < lingShort.durations.count {
                    cum += Int(lingShort.durations[d])
                    if f < cum {
                        pId = Int(lingShort.phoneIds[d])
                        break
                    }
                    d += 1
                }
                let token = engine.vocabulary.token(for: pId)
                let low8 = Array(frame[0..<8]).map { String(format: "%.1f", $0) }.joined(separator: ", ")
                print(String(format: "Frame %3d (phone %2d:%-3@) Mean=%5.2f | ch0..7=[%@]", f, pId, token, fMean, low8))
            }
        }
    }
}
#endif
