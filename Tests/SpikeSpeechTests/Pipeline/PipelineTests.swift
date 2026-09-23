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

            let floatDur = regulator.phonemeDuration(phoneId: 5, speedFactor: spd)
            XCTAssertFalse(floatDur.isNaN, "speed=\(spd) で phonemeDuration が NaN を返しました")
            XCTAssertFalse(floatDur.isInfinite, "speed=\(spd) で phonemeDuration が Inf を返しました")
            XCTAssertTrue(1.0 <= floatDur, "speed=\(spd) で phonemeDuration が 1.0 未満です")

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

        XCTAssertEqual(encoded[0][195], 0.0, accuracy: 1e-4, "無声フレームでの deltaF0 は 0.0 であること")
        XCTAssertEqual(encoded[1][195], 0.0, accuracy: 1e-4, "有声開始フレームでの deltaF0 は 0.0 であること")
        XCTAssertEqual(encoded[2][195], 0.0, accuracy: 1e-4, "有声継続一定ピッチでの deltaF0 は 0.0 であること")
        XCTAssertEqual(encoded[3][195], 0.0, accuracy: 1e-4, "有声継続一定ピッチでの deltaF0 は 0.0 であること")
        XCTAssertEqual(encoded[4][195], 0.0, accuracy: 1e-4, "無声フレームでの deltaF0 は 0.0 であること")

        XCTAssertEqual(encoded[0][193], 1.0, accuracy: 1e-4, "無声フレームの ch193 は 1.0 であること")
        XCTAssertEqual(encoded[2][193], 0.0, accuracy: 1e-4, "有声フレームの ch193 は 0.0 であること")
    }

    /// 受入基準テスト: 同一の音素 N であっても後続音素 (n vs p) によって直後 one-hot が異なることを検証
    /// なぜこのテストが必要か:
    /// 「銀杏」の「んな」と「甲板」の「んぱ」のように、後続子音に応じた調音結合フォルマントの
    /// 差異を SNN 音響モデルが捉えるための入力文脈分離を数学的に保証するため。
    func testPhonemeContextTriphoneOneHotDifferences() {
        let engine = SpikeSpeechEngine()

        // 1. 「かんな」: k (10), a (5), N (24), n (13), a (5)
        let phoneIdsKanna: [Int32] = [10, 5, 24, 13, 5]
        let durationsKanna: [Int32] = [2, 4, 3, 2, 4]
        let totalFKanna = 15
        let lingKanna = LinguisticFeatures(
            phoneIds: phoneIdsKanna,
            durations: durationsKanna,
            f0Contour: [Float](repeating: 220.0, count: totalFKanna),
            voicedFlags: [Float](repeating: 1.0, count: totalFKanna),
            energyContour: [Float](repeating: 0.7, count: totalFKanna),
            totalFrames: totalFKanna
        )
        let encodedKanna = engine.encodeLinguisticFeatures(features: lingKanna)

        // 2. 「かんぱ」: k (10), a (5), N (24), p (23), a (5)
        let phoneIdsKanpa: [Int32] = [10, 5, 24, 23, 5]
        let durationsKanpa: [Int32] = [2, 4, 3, 2, 4]
        let totalFKanpa = 15
        let lingKanpa = LinguisticFeatures(
            phoneIds: phoneIdsKanpa,
            durations: durationsKanpa,
            f0Contour: [Float](repeating: 220.0, count: totalFKanpa),
            voicedFlags: [Float](repeating: 1.0, count: totalFKanpa),
            energyContour: [Float](repeating: 0.7, count: totalFKanpa),
            totalFrames: totalFKanpa
        )
        let encodedKanpa = engine.encodeLinguisticFeatures(features: lingKanpa)

        // N の区間は両発話ともオフセット 2 + 4 = 6 フレーム目から 3 フレーム (f = 6, 7, 8)
        let frameN = 6

        // 現在の音素 one-hot (ch 0 ..< 64): 両者とも N (id=24) で一致
        XCTAssertEqual(encodedKanna[frameN][24], 3.0)
        XCTAssertEqual(encodedKanpa[frameN][24], 3.0)

        // 直前の音素 one-hot (ch 64 ..< 128): 両者とも a (id=5, ch 64+5=69) で一致
        XCTAssertEqual(encodedKanna[frameN][69], 3.0)
        XCTAssertEqual(encodedKanpa[frameN][69], 3.0)

        // 直後の音素 one-hot (ch 128 ..< 192):
        // かんな は n (id=13, ch 128+13=141) が 3.0
        XCTAssertEqual(encodedKanna[frameN][141], 3.0, accuracy: 1e-4)
        XCTAssertEqual(encodedKanna[frameN][151], 0.0, accuracy: 1e-4)

        // かんぱ は p (id=23, ch 128+23=151) が 3.0
        XCTAssertEqual(encodedKanpa[frameN][151], 3.0, accuracy: 1e-4)
        XCTAssertEqual(encodedKanpa[frameN][141], 0.0, accuracy: 1e-4)

        // 直後音素ベクトル全体の不一致を厳密に検証
        let nextSliceKanna = Array(encodedKanna[frameN][128..<192])
        let nextSliceKanpa = Array(encodedKanpa[frameN][128..<192])
        XCTAssertNotEqual(nextSliceKanna, nextSliceKanpa, "同一音素 N に対する直後音素 one-hot ベクトルが同一です（分離失敗）")

        // 文頭 (frame 0: k) の直前音素が <sil> (id=1, ch 64+1=65) であること
        XCTAssertEqual(encodedKanna[0][65], 3.0, accuracy: 1e-4, "文頭音素の直前は <sil> であること")

        // 文末 (frame 14: a) の直後音素が <sil> (id=1, ch 128+1=129) であること
        XCTAssertEqual(encodedKanna[14][129], 3.0, accuracy: 1e-4, "文末音素の直後は <sil> であること")
    }

    /// 受入基準検証: 長音符「ー」が 2文字拗音判定に誤爆せず独立したモーラとして維持されること
    func testKanaToMorasDoesNotMergeProlongedSound() {
        let vocab = PhonemeVocabulary()

        // 「いい」のモーラ分割（長音化後「いー」となっても 2 モーラであること）
        let morasIi = vocab.kanaToMoras("いい")
        XCTAssertEqual(morasIi.count, 2, "「いい」が 2 モーラとして分割されていません: count=\(morasIi.count)")

        // 「いー」のモーラ分割
        let morasI_ = vocab.kanaToMoras("いー")
        XCTAssertEqual(morasI_.count, 2, "「いー」が 2 モーラとして分割されていません: count=\(morasI_.count)")

        // 「きょう」のモーラ分割（「きょ」と「ー」の 2 モーラ）
        let morasKyo = vocab.kanaToMoras("きょう")
        XCTAssertEqual(morasKyo.count, 2, "「きょう」が 2 モーラとして分割されていません: count=\(morasKyo.count)")
        XCTAssertEqual(morasKyo[0].text, "きょ")

        // 「きょうはいい天気です」（10 モーラ）
        let normalizer = TextNormalizer(morphology: ViterbiMorphology())
        let morphemes = normalizer.normalize(text: "今日はいい天気です")
        let prosodyModel = ProsodyModel()
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocab)
        let totalMoras = phrases.reduce(0) { $0 + $1.moras.count }
        XCTAssertEqual(totalMoras, 10, "「今日はいい天気です」の総モーラ数が 10 ではありません: \(totalMoras)")
    }

    /// 受入基準検証: 推論本体フレーム合計が round(meanFramesPerMora × モーラ数) に厳格一致すること
    func testSpeechBodyDurationExactMatchAcceptanceCriteria() throws {
        var weights = SpikingNetworkWeights.standardInit(inputDim: 256, maxHiddenDim: 256, numLayers: 4)
        if let data = try? Data(contentsOf: URL(fileURLWithPath: "Models/weights.json")),
           let loaded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) {
            weights = loaded
        }
        let engine = SpikeSpeechEngine(weights: weights)

        // 1. 「今日はいい天気です」: 10 モーラ × 16 フレーム = 160 フレーム (1.60 秒)
        // 受入基準: 発話本体 1.45–1.75 秒
        let textTenki = "今日はいい天気です"
        let featuresTenki = engine.lengthRegulator.processText(
            text: textTenki,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            applyFluctuation: true,
            addBoundarySilence: true
        )
        // 先頭・末尾の境界無音（<sil>: ID=1）を除いた発話本体フレーム数を算出
        var bodyFramesTenki = 0
        var fIdx = 0
        while fIdx < featuresTenki.phoneIds.count {
            let pid = featuresTenki.phoneIds[fIdx]
            if pid != Int32(PhonemeVocabulary.silId) {
                bodyFramesTenki += Int(featuresTenki.durations[fIdx])
            }
            fIdx += 1
        }
        let bodySecTenki = Float(bodyFramesTenki) / 100.0
        print("[受入基準検証] 「今日はいい天気です」 発話本体: \(bodyFramesTenki) frames (\(bodySecTenki)s), 全体: \(featuresTenki.totalFrames) frames (\(Float(featuresTenki.totalFrames)/100.0)s)")
        XCTAssertTrue(145 <= bodyFramesTenki, "天気の本体フレーム数 \(bodyFramesTenki) が 145 未満です")
        XCTAssertTrue(bodyFramesTenki <= 175, "天気の本体フレーム数 \(bodyFramesTenki) が 175 超です")
        XCTAssertEqual(bodyFramesTenki, 160, "天気の本体フレーム数が目標 160 (10モーラ×16) と完全一致していません")

        // 2. 「こんにちは」: 5 モーラ × 16 フレーム = 80 フレーム (0.80 秒)
        // 受入基準: 発話本体 0.70–0.95 秒
        let textKonnichiwa = "こんにちは"
        let featuresKonnichiwa = engine.lengthRegulator.processText(
            text: textKonnichiwa,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            applyFluctuation: true,
            addBoundarySilence: true
        )
        var bodyFramesKonnichiwa = 0
        fIdx = 0
        while fIdx < featuresKonnichiwa.phoneIds.count {
            let pid = featuresKonnichiwa.phoneIds[fIdx]
            if pid != Int32(PhonemeVocabulary.silId) {
                bodyFramesKonnichiwa += Int(featuresKonnichiwa.durations[fIdx])
            }
            fIdx += 1
        }
        let bodySecKonnichiwa = Float(bodyFramesKonnichiwa) / 100.0
        print("[受入基準検証] 「こんにちは」 発話本体: \(bodyFramesKonnichiwa) frames (\(bodySecKonnichiwa)s), 全体: \(featuresKonnichiwa.totalFrames) frames (\(Float(featuresKonnichiwa.totalFrames)/100.0)s)")
        XCTAssertTrue(70 <= bodyFramesKonnichiwa, "こんにちは本体フレーム数 \(bodyFramesKonnichiwa) が 70 未満です")
        XCTAssertTrue(bodyFramesKonnichiwa <= 95, "こんにちは本体フレーム数 \(bodyFramesKonnichiwa) が 95 超です")
        XCTAssertEqual(bodyFramesKonnichiwa, 80, "こんにちは本体フレーム数が目標 80 (5モーラ×16) と完全一致していません")
    }
}
