import XCTest
import Foundation
@testable import SpikeSpeech

/// SpikeSpeechEngine 音声合成パイプラインの単体テストスイート
final class PipelineTests: XCTestCase {

    // MARK: - 1. SpikeSpeechEngine 基本合成テスト

    /// 日本語テキスト音声合成 (Text -> 16kHz 16-bit WAV) の検証
    func testEndToEndSpeechSynthesis() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは、世界。"

        let wavData = engine.synthesizeWav(text: text, speed: 1.0, pitch: 1.0)
        XCTAssertTrue(44 < wavData.count, "WAV データサイズがヘッダ長以下です: \(wavData.count)")

        wavData.withUnsafeBytes { rawBytes in
            let ptr = rawBytes.bindMemory(to: UInt8.self)
            XCTAssertEqual(ptr[0], 0x52) // R
            XCTAssertEqual(ptr[1], 0x49) // I
            XCTAssertEqual(ptr[2], 0x46) // F
            XCTAssertEqual(ptr[3], 0x46) // F
            XCTAssertEqual(ptr[8], 0x57) // W
            XCTAssertEqual(ptr[9], 0x41) // A
            XCTAssertEqual(ptr[10], 0x56) // V
            XCTAssertEqual(ptr[11], 0x45) // E
            XCTAssertEqual(ptr[12], 0x66) // f
            XCTAssertEqual(ptr[13], 0x6D) // m
            XCTAssertEqual(ptr[14], 0x74) // t
            XCTAssertEqual(ptr[15], 0x20)
            XCTAssertEqual(ptr[36], 0x64) // d
            XCTAssertEqual(ptr[37], 0x61) // a
            XCTAssertEqual(ptr[38], 0x74) // t
            XCTAssertEqual(ptr[39], 0x61) // a
        }

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

        XCTAssertTrue(fastSamples.count < slowSamples.count, "話速変更によるサンプル数の短縮が機能していません: fast=\(fastSamples.count), slow=\(slowSamples.count)")

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

    /// 速度・ピッチの異常値・極端値に対する堅牢性検証
    func testSpeedZeroAndExtremeValuesRobustness() {
        let regulator = LengthRegulator()
        let testSpeeds: [Float] = [0.0, -1.0, -100.0, Float.nan, Float.infinity, -Float.infinity, 0.0001, 100.0]

        var idx = 0
        while idx < testSpeeds.count {
            let spd = testSpeeds[idx]

            let floatDur = regulator.floatDurationFrames(category: .vowel, symbol: "a", speed: spd)
            XCTAssertFalse(floatDur.isNaN, "speed=\(spd) で floatDurationFrames が NaN を返しました")
            XCTAssertFalse(floatDur.isInfinite, "speed=\(spd) で floatDurationFrames が Inf を返しました")
            XCTAssertTrue(1.0 <= floatDur, "speed=\(spd) で floatDurationFrames が 1.0 未満です")

            let intDur = regulator.defaultDurationFrames(category: .vowel, symbol: "a", speed: spd)
            XCTAssertTrue(1 <= intDur, "speed=\(spd) で defaultDurationFrames が 1 未満です")

            idx += 1
        }

        let abnormalDurations: [Float] = [Float.nan, Float.infinity, -5.0, 0.0, 10.0]
        let quantized = regulator.quantizeDurations(durations: abnormalDurations)
        XCTAssertEqual(quantized.count, abnormalDurations.count)
        var q = 0
        while q < quantized.count {
            XCTAssertTrue(1 <= quantized[q], "quantizeDurations で 1 未満のフレーム数が生成されました")
            q += 1
        }

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

        XCTAssertTrue(maxAbs <= 0.8801, "ヘッドルームを超過するサンプル振幅を検出: maxAbs=\(maxAbs)")
        XCTAssertTrue(0.1 < rms, "音響エネルギーが小さすぎます: rms=\(rms)")

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

        XCTAssertEqual(clippingCount, 0, "WAV サンプルでクリッピングサチュレーションが検出されました: \(clippingCount)")
    }

    // MARK: - 2. 話者切り替え（Voice Profile / Speaker Conditioning）検証

    func testVoiceSwitchingFemaleVsMaleSynthesis() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは、世界。"

        let femaleSamples = engine.synthesize(text: text, voice: .female)
        let maleSamples = engine.synthesize(text: text, voice: .male)

        XCTAssertTrue(0 < femaleSamples.count)
        XCTAssertTrue(0 < maleSamples.count)

        var diffSum: Float = 0.0
        let compareCount = min(femaleSamples.count, maleSamples.count)
        var i = 0
        while i < compareCount {
            diffSum += abs(femaleSamples[i] - maleSamples[i])
            i += 1
        }
        let avgDiff = diffSum / Float(compareCount)
        XCTAssertTrue(0.01 < avgDiff, "女性ボイスと男性ボイスの合成波形に有意な差異が存在しません: diff=\(avgDiff)")
    }

    func testVoiceSwitchingAccentShapePreservation() {
        let prosody = ProsodyModel()
        let vocab = PhonemeVocabulary()
        var bio = BiologicalFluctuation(seed: 1234)

        let moras = [
            MoraToken(text: "こ", phonemes: [PhonemeToken(id: 15, symbol: "k", category: .consonant, durationFrames: 4), PhonemeToken(id: 9, symbol: "o", category: .vowel, durationFrames: 8)], tone: .low),
            MoraToken(text: "ん", phonemes: [PhonemeToken(id: 30, symbol: "N", category: .consonant, durationFrames: 8)], tone: .high),
            MoraToken(text: "に", phonemes: [PhonemeToken(id: 20, symbol: "n", category: .consonant, durationFrames: 4), PhonemeToken(id: 6, symbol: "i", category: .vowel, durationFrames: 8)], tone: .high),
            MoraToken(text: "ち", phonemes: [PhonemeToken(id: 23, symbol: "ch", category: .consonant, durationFrames: 4), PhonemeToken(id: 6, symbol: "i", category: .vowel, durationFrames: 8)], tone: .high),
            MoraToken(text: "は", phonemes: [PhonemeToken(id: 25, symbol: "w", category: .consonant, durationFrames: 4), PhonemeToken(id: 5, symbol: "a", category: .vowel, durationFrames: 8)], tone: .low)
        ]
        let phrase = AccentPhrase(moras: moras)

        let (f0Female, _, _) = prosody.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocab,
            baseF0: VoiceProfile.female.baseF0,
            fluctuation: &bio,
            applyFluctuation: false
        )

        var bioMale = BiologicalFluctuation(seed: 1234)
        let (f0Male, _, _) = prosody.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocab,
            baseF0: VoiceProfile.male.baseF0,
            fluctuation: &bioMale,
            applyFluctuation: false
        )

        XCTAssertEqual(f0Female.count, f0Male.count)

        var logFemale: [Float] = []
        var logMale: [Float] = []
        var sumFemale: Float = 0.0
        var sumMale: Float = 0.0
        var frame = 0
        while frame < f0Female.count {
            if 0.0 < f0Female[frame] && 0.0 < f0Male[frame] {
                let lf = logf(f0Female[frame])
                let lm = logf(f0Male[frame])
                logFemale.append(lf)
                logMale.append(lm)
                sumFemale += lf
                sumMale += lm
            }
            frame += 1
        }

        XCTAssertTrue(0 < logFemale.count)
        let meanFemale = sumFemale / Float(logFemale.count)
        let meanMale = sumMale / Float(logMale.count)

        var cov: Float = 0.0
        var varF: Float = 0.0
        var varM: Float = 0.0
        var idx = 0
        while idx < logFemale.count {
            let df = logFemale[idx] - meanFemale
            let dm = logMale[idx] - meanMale
            cov += df * dm
            varF += df * df
            varM += dm * dm
            idx += 1
        }

        let denom = sqrtf(varF * varM)
        XCTAssertTrue(0.0 < denom)
        let corr = cov / denom

        XCTAssertTrue(0.99 <= corr, "話者差し替えによる相対アクセント相関が不十分です: r=\(corr)")
    }

    func testSpeakerConditioningAffectsSynthesize() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは"

        let speakerA = SpeakerConditioning(embedding: [Float](repeating: 0.5, count: 128))
        let speakerB = SpeakerConditioning(embedding: [Float](repeating: -0.5, count: 128))

        let pcmA = engine.synthesize(text: text, speaker: speakerA)
        let pcmB = engine.synthesize(text: text, speaker: speakerB)

        XCTAssertFalse(pcmA.isEmpty)
        XCTAssertFalse(pcmB.isEmpty)
        XCTAssertEqual(pcmA.count, pcmB.count)

        var diffSum: Float = 0.0
        var i = 0
        while i < pcmA.count {
            diffSum += abs(pcmA[i] - pcmB[i])
            i += 1
        }
        let avgDiff = diffSum / Float(pcmA.count)
        XCTAssertTrue(1e-5 < avgDiff, "話者条件付けによる波形差分が検出されません: diff=\(avgDiff)")
    }

    func testRelativeProsodyBaseF0StrictLinearity() {
        let prosody = ProsodyModel()
        let vocab = PhonemeVocabulary()
        var bio = BiologicalFluctuation(seed: 42)

        let moras = [
            MoraToken(text: "あ", phonemes: [PhonemeToken(id: 5, symbol: "a", category: .vowel, durationFrames: 10)], tone: .high),
            MoraToken(text: "め", phonemes: [PhonemeToken(id: 8, symbol: "e", category: .vowel, durationFrames: 10)], tone: .low)
        ]
        let phrase = AccentPhrase(moras: moras)

        let (f0Female, _, _) = prosody.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocab,
            baseF0: 220.0,
            fluctuation: &bio,
            applyFluctuation: false
        )

        var bioMale = BiologicalFluctuation(seed: 42)
        let (f0Male, _, _) = prosody.generateF0Contour(
            phrases: [phrase],
            vocabulary: vocab,
            baseF0: 120.0,
            fluctuation: &bioMale,
            applyFluctuation: false
        )

        XCTAssertEqual(f0Female.count, f0Male.count)
        let expectedRatio: Float = 220.0 / 120.0
        var f = 0
        while f < f0Female.count {
            if 0.0 < f0Male[f] {
                let ratio = f0Female[f] / f0Male[f]
                XCTAssertEqual(ratio, expectedRatio, accuracy: 0.05, "話者基音の比率が保たれ相対抑揚が保存されていること: frame=\(f)")
            }
            f += 1
        }
    }

    func testVoiceSwitchingWavAndStream() {
        let engine = SpikeSpeechEngine()
        let text = "声優のボイスを切り替えます。"

        let femaleWav = engine.synthesizeWav(text: text, voice: .female)
        let maleWav = engine.synthesizeWav(text: text, voice: .male)

        XCTAssertTrue(44 < femaleWav.count)
        XCTAssertTrue(44 < maleWav.count)
        XCTAssertNotEqual(femaleWav, maleWav)

        var streamChunkCount = 0
        let streamSamples = engine.synthesizeStream(text: text, voice: .male) { chunk in
            if 0 < chunk.count {
                streamChunkCount += 1
            }
        }
        XCTAssertTrue(0 < streamSamples.count)
        XCTAssertTrue(0 < streamChunkCount)
    }

    // MARK: - 3. SpikeSpeechEngine 短い単体サニティテスト（空文字、クランプ、連続2回）

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

    // MARK: - 4. 特徴量エンコーディング検証

    func testDeltaF0ScaleSymmetry() {
        let engine = SpikeSpeechEngine()

        let f0Const: [Float] = [0.0, 200.0, 200.0, 200.0, 0.0]
        let voicedFlags: [Float] = [0.0, 1.0, 1.0, 1.0, 0.0]
        let energy: [Float] = [0.0, 0.7, 0.7, 0.7, 0.0]
        let phoneIds: [Int32] = [1, 5, 5, 5, 1]
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

        XCTAssertEqual(encoded[0][67], 0.0, accuracy: 1e-4, "無声フレームでの deltaF0 は 0.0 であること")
        XCTAssertEqual(encoded[1][67], 0.0, accuracy: 1e-4, "有声開始フレームでの deltaF0 は 0.0 であること")
        XCTAssertEqual(encoded[2][67], 0.0, accuracy: 1e-4, "有声継続一定ピッチでの deltaF0 は 0.0 であること")
        XCTAssertEqual(encoded[3][67], 0.0, accuracy: 1e-4, "有声継続一定ピッチでの deltaF0 は 0.0 であること")
        XCTAssertEqual(encoded[4][67], 0.0, accuracy: 1e-4, "無声フレームでの deltaF0 は 0.0 であること")

        XCTAssertEqual(encoded[0][65], 1.0, accuracy: 1e-4, "無声フレームの ch65 は 1.0 であること")
        XCTAssertEqual(encoded[2][65], 0.0, accuracy: 1e-4, "有声フレームの ch65 は 0.0 であること")
    }
}
