import Foundation
#if canImport(MLX)
import MLX
#endif
import SpikeSpeech

/// Tier 1: 機能網羅 E2E テスト
func runTier1Tests() {
    print("--- Running Tier 1 Feature E2E Tests ---")

    let morphology = ViterbiMorphology()
    let normalizer = TextNormalizer(morphology: morphology)
    let vocabulary = PhonemeVocabulary()
    let prosodyModel = ProsodyModel()
    let lengthRegulator = LengthRegulator(hiddenDimension: 128)
    let vocoder = NeuralVocoder()
    let engine = SpikeSpeechEngine()

    // F1: 形態素解析
    let m1 = morphology.tokenize("今日の本")
    e2eAssertTrue(0 < m1.count, "F1: morphology token count")
    if 0 < m1.count {
        e2eAssertEqual(m1[0].surface, "今日", "F1: surface")
        e2eAssertEqual(m1[0].pos, .noun, "F1: pos")
    }

    let m2 = morphology.tokenize("走る")
    e2eAssertTrue(0 < m2.count, "F1: verb token count")
    if 0 < m2.count {
        e2eAssertEqual(m2[0].surface, "走る", "F1: verb surface")
        e2eAssertEqual(m2[0].pos, .verb, "F1: verb pos")
    }

    // F2: 読み・発音正規化
    let norm1 = normalizer.normalize(text: "学校へ行く")
    let reading1 = norm1.map { $0.reading }.joined()
    e2eAssertTrue(reading1.contains("えいく"), "F2: particle he to e")

    let expandedNum = normalizer.expandNumbersAndCounters("12345")
    e2eAssertEqual(expandedNum, "いちまんにせんさんびゃくよんじゅうご", "F2: number expansion")

    let normVerb = normalizer.normalize(text: "思う")
    let readingVerb = normVerb.map { $0.reading }.joined()
    e2eAssertEqual(readingVerb, "おもう", "F2: verb ending u protection")

    // F3: トークナイズ
    let moras = vocabulary.kanaToMoras("きゃ")
    e2eAssertEqual(moras.count, 1, "F3: contracted sound mora count")
    if 0 < moras.count {
        e2eAssertEqual(moras[0].phonemes.count, 2, "F3: phoneme count")
        if 1 < moras[0].phonemes.count {
            e2eAssertEqual(moras[0].phonemes[0].symbol, "ky", "F3: phoneme 0")
            e2eAssertEqual(moras[0].phonemes[1].symbol, "a", "F3: phoneme 1")
        }
    }

    // F4: Length Regulation
    let features = lengthRegulator.processText(
        text: "ねこ",
        normalizer: normalizer,
        prosodyModel: prosodyModel,
        vocabulary: vocabulary,
        speedFactor: 1.0
    )
    e2eAssertTrue(0 < features.totalFrames, "F4: totalFrames positive")
    e2eAssertEqual(features.f0Contour.count, features.totalFrames, "F4: f0 count")

    // F5: ピッチアクセント
    let tonesHeadHigh = prosodyModel.computeMoraTones(moraCount: 3, accentKernel: 1)
    e2eAssertEqual(tonesHeadHigh.count, 3, "F5: tone count")
    if 2 < tonesHeadHigh.count {
        e2eAssertEqual(tonesHeadHigh[0], .high, "F5: tone 0")
        e2eAssertEqual(tonesHeadHigh[1], .low, "F5: tone 1")
        e2eAssertEqual(tonesHeadHigh[2], .low, "F5: tone 2")
    }

    // F6: NeuralVocoder 直接波形合成
    vocoder.reset()
    let melFrame = [Float](repeating: -2.0, count: 64)
    let vocoderOut = vocoder.synthesize(mel: [melFrame, melFrame])
    e2eAssertEqual(vocoderOut.count, 320, "F6: vocoder samples count")

    // F8: WavEncoder
    let testSamples: [Float] = [0.0, 0.5, -0.5, 0.8]
    let wavBytes = WavEncoder.encode(samples: testSamples, sampleRate: 16000)
    e2eAssertEqual(wavBytes.count, 44 + (testSamples.count * 2), "F8: wav bytes count")

    // F10: SNN デコーダー
    let weights = SpikingNetworkWeights.randomWeights(inputDim: 16, maxHiddenDim: 32, outputDim: 8, timeSteps: 2, numLayers: 2)
    let decoder = SpikingAcousticDecoder(weights: weights)
    let workspace = AcousticWorkspace(maxHiddenDim: 32, outputDim: 8, numLayers: 2)
    let inSeq = [[Float]](repeating: [Float](repeating: 0.5, count: 16), count: 4)
    let decodedSeq = decoder.decodeSequence(featuresSeq: inSeq, workspace: workspace)
    e2eAssertEqual(decodedSeq.count, 4, "F10: decoded frames count")

    #if canImport(MLX)
    // F11: MLX BPTT
    let v = MLXArray([Float(0.8), Float(1.2)])
    let vTh = MLXArray([Float(1.0), Float(1.0)])
    let spikes = SurrogateGradients.fastSigmoidSTE(v: v, vTh: vTh, alpha: 2.0)
    let sArr = spikes.asArray(Float.self)
    e2eAssertEqual(sArr[0], 0.0, "F11: spike 0")
    e2eAssertEqual(sArr[1], 1.0, "F11: spike 1")
    #endif

    // E2E パイプライン総合
    let fullWav = engine.synthesizeWav(text: "こんにちは、世界。")
    e2eAssertTrue(44 < fullWav.count, "F14: E2E wav synthesis non-empty")
}
