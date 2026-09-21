import Foundation

/// SpikeSpeech E2E 音声合成エンジン
///
/// テキスト正規化、形態素解析、韻律推定、時間長アライメント、多層 SNN 音響推論、
/// およびニューラルボコーダーによる波形出力までの一連の処理を外部依存ゼロで高速に実行する。
public final class SpikeSpeechEngine: @unchecked Sendable {

    public let normalizer: TextNormalizer
    public let prosodyModel: ProsodyModel
    public let prosodyPredictor: ProsodyPredictor
    public let vocabulary: PhonemeVocabulary
    public let lengthRegulator: LengthRegulator
    public let decoder: SpikingAcousticDecoder
    public let neuralVocoder: NeuralVocoder
    public let workspace: AcousticWorkspace
    public let weights: SpikingNetworkWeights
    public let sampleRate: Float

    /// 初期化
    public init(
        weights: SpikingNetworkWeights? = nil,
        sampleRate: Float = Float(AudioConfig.sampleRate),
        vocoderWeights: NeuralVocoderWeights? = nil
    ) {
        // なぜ weights が未指定の場合に Models/weights.json を自動ロードするか:
        // 引数なしの SpikeSpeechEngine() をインスタンス化した場合でも、
        // 学習済みの実音声 SNN 音響モデル重みを自動で読み込み、本来の自然な肉声合成を実行できるようにするため。
        let baseWeights: SpikingNetworkWeights
        switch weights {
        case .some(let w):
            baseWeights = w
        case .none:
            let defaultWeightsPath = "Models/weights.json"
            var loaded: SpikingNetworkWeights? = nil
            if FileManager.default.fileExists(atPath: defaultWeightsPath) {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: defaultWeightsPath)) {
                    loaded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data)
                }
            }
            switch loaded {
            case .some(let w):
                baseWeights = w
            case .none:
                baseWeights = SpikingNetworkWeights.randomWeights()
            }
        }

        // なぜ baseWeights.lexicon が空の場合にデフォルト語彙を注入するか:
        // 引数なしの SpikeSpeechEngine() や lexicon がシリアライズされていない重みでも
        // 永続化された学習語彙知識（Models/weights.json）を自動ロードし、
        // 正常な形態素解析と読み推定を実行できるようにするため。
        let effectiveWeights: SpikingNetworkWeights
        if baseWeights.lexicon.isEmpty != true {
            effectiveWeights = baseWeights
        } else {
            let defaultLex = ViterbiMorphology.loadDefaultLexicon()
            effectiveWeights = baseWeights.withLexicon(defaultLex)
        }

        self.weights = effectiveWeights
        self.sampleRate = sampleRate

        // なぜ effectiveWeights.lexicon を注入するか:
        // ソースコード内にハードコードされた辞書を排し、モデルの学習重みとともに永続化された
        // 語彙知識を形態素解析器へ供給して正確な分かち書きと読み推定を行うため。
        let morphology = ViterbiMorphology(lexicon: effectiveWeights.lexicon)
        self.normalizer = TextNormalizer(morphology: morphology)
        self.prosodyModel = ProsodyModel()
        self.prosodyPredictor = ProsodyPredictor(weights: effectiveWeights.prosodyWeights)
        self.vocabulary = PhonemeVocabulary()
        self.lengthRegulator = LengthRegulator(
            hiddenDimension: effectiveWeights.inputDim,
            phonemeAverageDurations: effectiveWeights.phonemeAverageDurations
        )
        self.decoder = SpikingAcousticDecoder(weights: effectiveWeights)
        self.neuralVocoder = NeuralVocoder(weights: vocoderWeights)
        self.workspace = AcousticWorkspace(
            maxHiddenDim: effectiveWeights.maxHiddenDim,
            outputDim: effectiveWeights.outputDim,
            numLayers: effectiveWeights.numLayers
        )
    }

    /// 言語特徴量から SNN 入力フレーム特徴量系列を生成する。
    ///
    /// なぜ未学習の話者埋め込み（ch68-83）を排除し直交特徴量のみにするか:
    /// 単一話者学習において手書き話者埋め込みは単なる定数直流バイアスとして吸収され、
    /// 推論時に他話者の未学習ベクトルを注入すると膜電位をランダムに歪める有害電流となるため。
    /// 話者同一性は基音（baseF0）および NeuralVocoder への SpeakerConditioning で制御し、
    /// SNN は音素・有声度・F0・進行度・エネルギーの純粋な音響特徴量に専念させる。
    public func encodeLinguisticFeatures(
        features: LinguisticFeatures
    ) -> [[Float]] {
        let totalFrames = features.totalFrames
        if totalFrames <= 0 {
            return []
        }

        let inDim = weights.inputDim
        var seq = [[Float]](repeating: [Float](repeating: 0.0, count: inDim), count: totalFrames)

        var currentFrameOffset = 0
        let phoneCount = features.phoneIds.count

        var p = 0
        while p < phoneCount {
            let pid = Int(features.phoneIds[p])
            let duration = Int(features.durations[p])

            if duration <= 0 {
                p += 1
                continue
            }

            var f = 0
            while f < duration {
                let frameIdx = currentFrameOffset + f
                if totalFrames <= frameIdx {
                    break
                }

                var rawF0: Float = 0.0
                if frameIdx < features.f0Contour.count {
                    rawF0 = features.f0Contour[frameIdx]
                }
                var voiced: Float = 0.0
                if frameIdx < features.voicedFlags.count {
                    voiced = features.voicedFlags[frameIdx]
                }
                var f0 = rawF0
                if f0 < 0.0 {
                    f0 = 0.0
                }

                // 1. 音素 ID の One-Hot 符号化
                // 背景電流に埋もれず膜電位の閾値を確実に突破できるよう、音素発火電流を注入する
                if 0 <= pid {
                    if pid < 64 {
                        if pid < inDim {
                            seq[frameIdx][pid] = 3.0
                        }
                    }
                }

                // 2. 韻律および音響生理学的特徴の付加
                if 64 < inDim {
                    seq[frameIdx][64] = voiced * 1.0
                }
                if 65 < inDim {
                    // ch64 の有声度に対する直交抑制電流として機能させる無声度 (1.0 - voiced)
                    let unvoiced = 1.0 - voiced
                    seq[frameIdx][65] = unvoiced * 1.0
                }
                if 66 < inDim {
                    var normF0 = f0 / 500.0
                    if normF0 < 0.0 {
                        normF0 = 0.0
                    }
                    if 1.0 < normF0 {
                        normF0 = 1.0
                    }
                    // F0 ピッチ周波数を 500Hz 基準で [0.0, 1.0] に安全正規化して供給
                    seq[frameIdx][66] = normF0 * 1.0
                }
                if 67 < inDim {
                    // なぜ前フレーム F0 との差分を 50Hz スケールでクリップするか:
                    // 急峻なピッチ変動（抑揚アクセント境界）を SNN に直接知らせるため。
                    var deltaF0: Float = 0.0
                    if 0.5 <= voiced && 0 < frameIdx {
                        let prevIdx = frameIdx - 1
                        var prevVoiced: Float = 0.0
                        if prevIdx < features.voicedFlags.count {
                            prevVoiced = features.voicedFlags[prevIdx]
                        }
                        if 0.5 <= prevVoiced && prevIdx < features.f0Contour.count {
                            let rawPrevF0 = features.f0Contour[prevIdx]
                            var prevF0 = rawPrevF0
                            if prevF0 < 0.0 {
                                prevF0 = 0.0
                            }
                            deltaF0 = (f0 - prevF0) / 50.0
                        }
                    }
                    var clampedDelta = deltaF0
                    if clampedDelta < -1.0 {
                        clampedDelta = -1.0
                    }
                    if 1.0 < clampedDelta {
                        clampedDelta = 1.0
                    }
                    seq[frameIdx][67] = clampedDelta * 1.0
                }
                if 68 < inDim {
                    let progress = Float(f) / Float(max(1, duration))
                    seq[frameIdx][68] = progress * 1.0
                }
                if 69 < inDim {
                    let rate = 10.0 / Float(max(1, duration))
                    seq[frameIdx][69] = min(1.0, rate) * 1.0
                }
                if 70 < inDim {
                    var engVal: Float = 0.50
                    if frameIdx < features.energyContour.count {
                        engVal = features.energyContour[frameIdx]
                    }
                    if 1.0 < engVal {
                        engVal = 1.0
                    }
                    seq[frameIdx][70] = engVal * 1.0
                }

                f += 1
            }

            currentFrameOffset += duration
            p += 1
        }

        return seq
    }


    /// 各フレームが無音・休止・促音・無声破裂音の閉鎖期（気流遮断）に該当するか判定するマスクを構築する。
    /// なぜ音素アライメント情報から厳密に無音判定を行うか:
    /// 閉鎖期に微小なノイズが漏洩すると耳障りなヒス・濁音感を生むため、物理的原理に基づき完全無音化する。
    internal func computeFrameSilenceMask(
        linguisticFeatures: LinguisticFeatures,
        totalFrames: Int
    ) -> [Bool] {
        var silenceMask = [Bool](repeating: true, count: totalFrames)
        var curF = 0
        var pIdx = 0
        let pCount = min(linguisticFeatures.phoneIds.count, linguisticFeatures.durations.count)
        while pIdx < pCount {
            let pid = Int(linguisticFeatures.phoneIds[pIdx])
            let dur = Int(linguisticFeatures.durations[pIdx])

            let isPause = vocabulary.isPauseOrSilence(id: pid)

            var f = 0
            while f < dur {
                let frameIdx = curF + f
                if frameIdx < totalFrames {
                    switch isPause {
                    case true:
                        silenceMask[frameIdx] = true
                    case false:
                        silenceMask[frameIdx] = false
                    }
                }
                f += 1
            }
            curF += dur
            pIdx += 1
        }
        return silenceMask
    }

    /// 無声破裂音の解放バースト期（最終フレーム）を判定するマスクを算出する
    /// なぜバースト期を特定するか:
    /// 無声破裂音（k, t, p）の解放バーストは急峻な過渡アタックであり、
    /// 定常的なホワイトノイズ（> 0.02 RMS）への肥大化を抑止して自然な子音アタック知覚を担保するため。
    internal func computeStopBurstMask(
        linguisticFeatures: LinguisticFeatures,
        totalFrames: Int
    ) -> [Bool] {
        var burstMask = [Bool](repeating: false, count: totalFrames)
        var curF = 0
        var pIdx = 0
        let pCount = min(linguisticFeatures.phoneIds.count, linguisticFeatures.durations.count)
        while pIdx < pCount {
            let pid = Int(linguisticFeatures.phoneIds[pIdx])
            let dur = Int(linguisticFeatures.durations[pIdx])
            let isStop = vocabulary.isUnvoicedStop(id: pid)
            if isStop {
                let burstFrame = curF + dur - 1
                if burstFrame < totalFrames {
                    burstMask[burstFrame] = true
                }
            }
            curF += dur
            pIdx += 1
        }
        return burstMask
    }


    /// 日本語テキストから 16kHz モノラル PCM 浮動小数点サンプル列を合成する。
    @discardableResult
    public func synthesize(
        text: String,
        voice: VoiceProfile = .default,
        speed: Float = 1.0,
        pitch: Float = 1.0,
        speaker: SpeakerConditioning = .zero
    ) -> [Float] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        var safeSpeed = speed
        if safeSpeed.isFinite != true {
            safeSpeed = 1.0
        }
        if safeSpeed < 0.2 {
            safeSpeed = 0.2
        }
        if 5.0 < safeSpeed {
            safeSpeed = 5.0
        }
        var safePitch = pitch
        if safePitch.isFinite != true {
            safePitch = 1.0
        }
        if safePitch < 0.20 {
            safePitch = 0.20
        }
        if 5.00 < safePitch {
            safePitch = 5.00
        }

        // なぜ推論前に両状態をリセットするか:
        // 前の発話の残差膜電位やボコーダー内部バッファの漏洩（状態汚染）を確実に根絶するため。
        workspace.reset()
        neuralVocoder.reset()

        // 1. テキスト正規化および言語韻律処理（学習済み韻律予測器を主経路とし、話者基音 baseF0 にユーザー指定 pitch を一元反映）
        let effectiveBaseF0 = voice.baseF0 * safePitch
        let linguisticFeatures = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            prosodyPredictor: prosodyPredictor,
            speedFactor: safeSpeed,
            baseF0: effectiveBaseF0,
            addBoundarySilence: true
        )

        let totalFrames = linguisticFeatures.totalFrames
        if totalFrames <= 0 {
            return []
        }

        // 2. 多層 SNN 音響デコーダー推論（内部状態・膜電位の一貫性を更新し Mel 残差系列を生成）
        let inputSeq = encodeLinguisticFeatures(features: linguisticFeatures)
        let snnAcousticSeq = decoder.decodeSequence(
            featuresSeq: inputSeq,
            workspace: workspace
        )

        // 3. Mel スペクトル系列の供給
        // なぜ SNN 音響モデル出力をそのままニューラルボコーダーへ供給するか:
        // SNN は対数 Mel スペクトルを出力するように設計されており、単一パイプラインとして直結することで歪みを防ぐため。
        let melChannels = AudioConfig.melChannels
        var melSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalFrames)
        var t = 0
        while t < totalFrames {
            if t < snnAcousticSeq.count {
                let outDim = snnAcousticSeq[t].count
                let copyCount = min(melChannels, outDim)
                melSeq[t].withUnsafeMutableBufferPointer { melDst in
                    snnAcousticSeq[t].withUnsafeBufferPointer { acSrc in
                        melDst.baseAddress!.update(from: acSrc.baseAddress!, count: copyCount)
                    }
                }
            }
            t += 1
        }

        // 4. ニューラルボコーダーによる 16kHz PCM 波形展開
        var rawSamples = neuralVocoder.synthesize(
            mel: melSeq,
            f0Contour: linguisticFeatures.f0Contour,
            voicedFlags: linguisticFeatures.voicedFlags,
            speaker: speaker
        )

        // 5. 無音・休止・促音区間および無声破裂音の閉鎖期における完全ゼロミュート
        // なぜ閉鎖期・ポーズをゼロクリアするか:
        // 調音生理学において無声破裂音（k, t, p）の閉鎖期および休止・ポーズは気流が完全遮断され音響エネルギーが物理的に 0 であるため。
        let silenceMask = computeFrameSilenceMask(
            linguisticFeatures: linguisticFeatures,
            totalFrames: totalFrames
        )
        let frameSize = AudioConfig.hopSize
        var fIdx = 0
        while fIdx < totalFrames {
            let isSilence = silenceMask[fIdx]
            switch isSilence {
            case true:
                var prevIsSilence = true
                if 0 < fIdx {
                    prevIsSilence = silenceMask[fIdx - 1]
                }
                let startSample = fIdx * frameSize
                let endSample = min(rawSamples.count, startSample + frameSize)
                switch prevIsSilence {
                case false:
                    let invN = 1.0 / Float(frameSize)
                    var s = startSample
                    while s < endSample {
                        let sampleOffset = s - startSample
                        let fade = 1.0 - (Float(sampleOffset) * invN)
                        rawSamples[s] = rawSamples[s] * fade
                        s += 1
                    }
                case true:
                    var s = startSample
                    while s < endSample {
                        rawSamples[s] = 0.0
                        s += 1
                    }
                }
            case false:
                break
            }
            fIdx += 1
        }

        // 6. ヘッドルーム正規化（可聴音圧の確保と安全マージン）および話者エネルギースケーリング
        // なぜ話者のエネルギースケーリングを最終波形に乗算するか:
        // 音響モデルやボコーダー内部の振幅レンジを歪めることなく、
        // ユーザー指定の VoiceProfile.energyScale に応じた全体ゲイン・ゼロエネルギー消音を忠実に実現するため。
        if voice.energyScale <= 0.0 {
            var s = 0
            while s < rawSamples.count {
                rawSamples[s] = 0.0
                s += 1
            }
        } else {
            let targetPeak: Float = 0.85 * voice.energyScale
            var currentPeak: Float = 0.0
            var pIdx = 0
            while pIdx < rawSamples.count {
                let absVal = abs(rawSamples[pIdx])
                if currentPeak < absVal {
                    currentPeak = absVal
                }
                pIdx += 1
            }
            if 0.01 < currentPeak {
                var normScale = targetPeak / currentPeak
                if 6.0 < normScale {
                    normScale = 6.0
                }
                var s = 0
                while s < rawSamples.count {
                    var scaled = rawSamples[s] * normScale
                    if targetPeak < scaled {
                        scaled = targetPeak
                    }
                    if scaled < -targetPeak {
                        scaled = -targetPeak
                    }
                    rawSamples[s] = scaled
                    s += 1
                }
            }
        }

        // 7. 発話開始および終了境界における DAC ポップノイズ防止のマイクロフェード（160サンプル / 10ms）
        let fadeLen = 160
        if (fadeLen * 2) <= rawSamples.count {
            let invFade: Float = 1.0 / Float(fadeLen)
            var s = 0
            while s < fadeLen {
                let factor = Float(s) * invFade
                rawSamples[s] = rawSamples[s] * factor
                s += 1
            }
            let endOffset = rawSamples.count - fadeLen
            s = 0
            while s < fadeLen {
                let factor = Float(fadeLen - 1 - s) * invFade
                rawSamples[endOffset + s] = rawSamples[endOffset + s] * factor
                s += 1
            }
        }

        return rawSamples
    }

    /// 日本語テキストから 16kHz 16-bit Mono WAV バイナリデータを生成する。
    @discardableResult
    public func synthesizeWav(
        text: String,
        voice: VoiceProfile = .default,
        speed: Float = 1.0,
        pitch: Float = 1.0,
        speaker: SpeakerConditioning = .zero
    ) -> Data {
        let samples = synthesize(text: text, voice: voice, speed: speed, pitch: pitch, speaker: speaker)
        return WavEncoder.encode(samples: samples, sampleRate: Int(sampleRate))
    }

    /// テキストを句読点（句点・感嘆符・疑問符・改行）境界で分割する
    private func splitIntoSentences(text: String) -> [String] {
        var sentences = [String]()
        var current = ""
        for char in text {
            current.append(char)
            switch char {
            case "。", "！", "？", "\n", ".", "!", "?":
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty != true {
                    sentences.append(current)
                }
                current = ""
            default:
                break
            }
        }
        let tailTrimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if tailTrimmed.isEmpty != true {
            sentences.append(current)
        }
        return sentences
    }

    /// ストリーミング音声合成を実行する。
    ///
    /// - Parameters:
    ///   - text: 入力日本語テキスト
    ///   - voice: 声プロファイル
    ///   - speed: 話速係数
    ///   - pitch: ピッチ係数
    ///   - speaker: 話者条件付け埋め込み
    ///   - isCancelled: 早期脱出判定クロージャ。true を返した場合は以降のフレーム生成を中断する。
    ///   - onFrame: フレームごとの PCM サンプル送出クロージャ
    @discardableResult
    public func synthesizeStream(
        text: String,
        voice: VoiceProfile = .default,
        speed: Float = 1.0,
        pitch: Float = 1.0,
        speaker: SpeakerConditioning = .zero,
        isCancelled: (@Sendable () -> Bool)? = nil,
        onFrame: (([Float]) -> Void)?
    ) -> [Float] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        switch isCancelled {
        case .some(let cancel):
            if cancel() {
                return []
            }
        case .none:
            break
        }

        var safeSpeed = speed
        if safeSpeed.isFinite != true {
            safeSpeed = 1.0
        }
        if safeSpeed < 0.2 {
            safeSpeed = 0.2
        }
        if 5.0 < safeSpeed {
            safeSpeed = 5.0
        }

        var safePitch = pitch
        if safePitch.isFinite != true {
            safePitch = 1.0
        }
        if safePitch < 0.20 {
            safePitch = 0.20
        }
        if 5.00 < safePitch {
            safePitch = 5.00
        }

        let frameSize = AudioConfig.hopSize

        switch onFrame {
        case .none:
            return synthesize(
                text: text,
                voice: voice,
                speed: safeSpeed,
                pitch: safePitch,
                speaker: speaker
            )

        case .some(let callback):
            let sentences = splitIntoSentences(text: text)
            var allSamples = [Float]()
            var sIdx = 0
            while sIdx < sentences.count {
                switch isCancelled {
                case .some(let cancel):
                    if cancel() {
                        return allSamples
                    }
                case .none:
                    break
                }

                let sentText = sentences[sIdx]
                let sentSamples = synthesize(
                    text: sentText,
                    voice: voice,
                    speed: safeSpeed,
                    pitch: safePitch,
                    speaker: speaker
                )

                var offset = 0
                let sentTotal = sentSamples.count
                while offset < sentTotal {
                    switch isCancelled {
                    case .some(let cancel):
                        if cancel() {
                            return allSamples
                        }
                    case .none:
                        break
                    }

                    let remaining = sentTotal - offset
                    let count = min(frameSize, remaining)
                    var frameBuffer = [Float](repeating: 0.0, count: frameSize)
                    var s = 0
                    while s < count {
                        frameBuffer[s] = sentSamples[offset + s]
                        s += 1
                    }
                    callback(frameBuffer)
                    allSamples.append(contentsOf: sentSamples[offset..<(offset + count)])
                    offset += count
                    usleep(100)
                }

                sIdx += 1
            }

            return allSamples
        }
    }
}
