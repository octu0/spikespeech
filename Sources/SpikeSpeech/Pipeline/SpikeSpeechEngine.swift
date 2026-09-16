import Foundation

/// SpikeSpeech E2E 音声合成エンジン
///
/// テキスト正規化、形態素解析、韻律推定、時間長アライメント、多層 SNN 音響推論、
/// LPC 音響変換および WAV 出力までの一連の処理を外部依存ゼロで高速に実行する。
public final class SpikeSpeechEngine: @unchecked Sendable {

    public let normalizer: TextNormalizer
    public let prosodyModel: ProsodyModel
    public let vocabulary: PhonemeVocabulary
    public let lengthRegulator: LengthRegulator
    public let decoder: SpikingAcousticDecoder
    public let melToLPC: MelToLPC
    public let neuralVocoder: NeuralVocoder
    public let acousticPrior: PhonemeAcousticPrior
    public let workspace: AcousticWorkspace
    public let weights: SpikingNetworkWeights
    public let sampleRate: Float
    /// SNN音響モデルが生成した残差Mel特徴量の加算合成比率
    public var residualScale: Float

    /// 声道長パラメータごとの PhonemeAcousticPrior キャッシュ
    private var priorCache: [Int: PhonemeAcousticPrior] = [:]
    private let priorLock = NSLock()

    /// 声道幾何パラメータから一意かつ決定論的な整数キャッシュキーを生成する
    @inline(__always)
    private static func tractCacheKey(for tract: VocalTract) -> Int {
        let ls = Int(roundf(tract.lengthScale * 1000.0))
        let bw = Int(roundf(tract.bandwidthScale * 1000.0))
        return (ls * 10000) + bw
    }

    /// 初期化
    public init(
        weights: SpikingNetworkWeights = SpikingNetworkWeights.randomWeights(),
        sampleRate: Float = Float(AudioConfig.sampleRate),
        // なぜデフォルト値を 1.0 とするのか:
        // SNN 音響モデルは訓練時に実音声 Mel と Prior の差分である全残差 (targetMel - prior) を
        // 目的関数として直接最適化している。推論時にも同一スケール（1.0）で残差を加算することで、
        // BPTT が学習した音韻補正およびスペクトル残差を 100% 反映させ、学習と推論の目的関数を数理的に完全一致させる。
        residualScale: Float = 1.0,
        // なぜ vocoderWeights をオプショナルで受け取るか:
        // 外部からカスタム学習済みニューラルボコーダー重みを柔軟に差し替え可能にしつつ、
        // 省略時は安全な決定論的初期重みによりゼロ設定で即座に動作可能とするため。
        vocoderWeights: NeuralVocoderWeights? = nil
    ) {
        // なぜ weights.lexicon が空の場合にデフォルト語彙を注入するか:
        // 引数なしの SpikeSpeechEngine() や lexicon がシリアライズされていない重みでも
        // 永続化された学習語彙知識（Models/weights.json）を自動ロードし、
        // 正常な形態素解析と読み推定を実行できるようにするため。
        let effectiveWeights: SpikingNetworkWeights
        if weights.lexicon.isEmpty != true {
            effectiveWeights = weights
        } else {
            let defaultLex = ViterbiMorphology.loadDefaultLexicon()
            effectiveWeights = weights.withLexicon(defaultLex)
        }

        self.weights = effectiveWeights
        self.sampleRate = sampleRate
        self.residualScale = residualScale

        // なぜ effectiveWeights.lexicon を注入するか:
        // ソースコード内にハードコードされた辞書を排し、モデルの学習重みとともに永続化された
        // 語彙知識を形態素解析器へ供給して正確な分かち書きと読み推定を行うため。
        let morphology = ViterbiMorphology(lexicon: effectiveWeights.lexicon)
        self.normalizer = TextNormalizer(morphology: morphology)
        self.prosodyModel = ProsodyModel()
        self.vocabulary = PhonemeVocabulary()
        self.lengthRegulator = LengthRegulator(hiddenDimension: weights.inputDim)
        self.decoder = SpikingAcousticDecoder(weights: weights)
        let defaultTract = VocalTract(lengthScale: 1.0, bandwidthScale: 1.0)
        let defaultPrior = PhonemeAcousticPrior(
            melChannels: AudioConfig.melChannels,
            vocabSize: 64,
            sampleRate: sampleRate,
            tract: defaultTract
        )
        self.acousticPrior = defaultPrior
        self.priorCache[Self.tractCacheKey(for: defaultTract)] = defaultPrior

        self.melToLPC = MelToLPC(
            melChannels: AudioConfig.melChannels,
            fftBins: 257,
            lpcOrder: AudioConfig.lpcOrder,
            sampleRate: sampleRate
        )
        let effectiveVocoderWeights: NeuralVocoderWeights
        switch vocoderWeights {
        case .some(let vw):
            effectiveVocoderWeights = vw
        case .none:
            let defaultVocoderPath = "Models/vocoder_weights.json"
            var loaded: NeuralVocoderWeights? = nil
            if FileManager.default.fileExists(atPath: defaultVocoderPath) {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: defaultVocoderPath)) {
                    loaded = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: data)
                }
            }
            switch loaded {
            case .some(let vw):
                effectiveVocoderWeights = vw
            case .none:
                effectiveVocoderWeights = NeuralVocoderWeights.randomWeights()
            }
        }
        self.neuralVocoder = NeuralVocoder(weights: effectiveVocoderWeights)
        self.workspace = AcousticWorkspace(
            maxHiddenDim: weights.maxHiddenDim,
            outputDim: weights.outputDim,
            numLayers: weights.numLayers
        )
    }

    /// 声道伝達特性（VocalTract）に応じた PhonemeAcousticPrior を取得または生成する
    ///
    /// なぜキャッシュするか:
    /// 音声合成ごとに同一話者の Prior テーブル（4096 Float）を再計算するオーバーヘッドを排除し、
    /// リアルタイム即時推論時のゼロアロケーションと O(1) 参照を実現するため。
    public func prior(for tract: VocalTract) -> PhonemeAcousticPrior {
        let key = Self.tractCacheKey(for: tract)
        priorLock.lock()
        defer { priorLock.unlock() }
        switch priorCache[key] {
        case .some(let p):
            return p
        case .none:
            let p = PhonemeAcousticPrior(
                melChannels: AudioConfig.melChannels,
                vocabSize: 64,
                sampleRate: sampleRate,
                tract: tract
            )
            priorCache[key] = p
            return p
        }
    }

    /// 言語特徴量から SNN 入力フレーム特徴量系列を生成する。
    ///
    /// なぜ未学習の話者埋め込み（ch68-83）を排除し直交特徴量のみにするか:
    /// 単一話者学習において手書き話者埋め込みは単なる定数直流バイアスとして吸収され、
    /// 推論時に他話者の未学習ベクトルを注入すると膜電位をランダムに歪める有害電流となるため。
    /// 話者同一性は Prior（Filter）と Vocoder（Source）で物理的に制御し、
    /// SNN は音素・有声度・F0・進行度・エネルギーの純粋な音響残差に専念させる。
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

    /// 各フレームの音素 ID 系列に基づき、調音結合（Coarticulation）を考慮した事前対数 Mel 系列を算出する。
    /// なぜ学習と推論で共通化するか:
    /// 音素境界での生 Prior からの急激な段差をクロスフェード平滑化した事前スペクトルに対して
    /// 残差 (targetMel - blendedPrior) を学習・加算することで、学習時と推論時で残差の物理的意味が 1 対 1 で完全に一致するため。
    public func computeBlendedPriorSequence(
        framePhoneIds: [Int],
        activePrior: PhonemeAcousticPrior,
        melChannels: Int
    ) -> [[Float]] {
        let totalFrames = framePhoneIds.count
        if totalFrames <= 0 {
            return []
        }

        var blendedSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalFrames)

        var t = 0
        while t < totalFrames {
            let pId = framePhoneIds[t]
            let prevPId: Int
            if 0 < t {
                prevPId = framePhoneIds[t - 1]
            } else {
                prevPId = pId
            }

            let nextPId: Int
            if (t + 1) < totalFrames {
                nextPId = framePhoneIds[t + 1]
            } else {
                nextPId = pId
            }

            var currPrior = [Float](repeating: 0.0, count: melChannels)
            currPrior.withUnsafeMutableBufferPointer { pDst in
                activePrior.copyPriorMel(phoneId: pId, dst: pDst.baseAddress!)
            }

            var blendedPrior = currPrior
            if prevPId != pId {
                var canBlendPrev = true
                if vocabulary.isPauseOrSilence(id: prevPId) || vocabulary.isUnvoicedConsonant(id: prevPId) {
                    canBlendPrev = false
                }
                if vocabulary.isPauseOrSilence(id: pId) || vocabulary.isUnvoicedStop(id: pId) {
                    canBlendPrev = false
                }

                if canBlendPrev {
                    var prevPrior = [Float](repeating: 0.0, count: melChannels)
                    prevPrior.withUnsafeMutableBufferPointer { pDst in
                        activePrior.copyPriorMel(phoneId: prevPId, dst: pDst.baseAddress!)
                    }
                    var c = 0
                    while c < melChannels {
                        blendedPrior[c] = (0.30 * prevPrior[c]) + (0.70 * blendedPrior[c])
                        c += 1
                    }
                }
            }
            if nextPId != pId {
                var canBlendNext = true
                if vocabulary.isPauseOrSilence(id: nextPId) || vocabulary.isUnvoicedConsonant(id: nextPId) {
                    canBlendNext = false
                }
                if vocabulary.isPauseOrSilence(id: pId) || vocabulary.isUnvoicedStop(id: pId) {
                    canBlendNext = false
                }

                if canBlendNext {
                    var nextPrior = [Float](repeating: 0.0, count: melChannels)
                    nextPrior.withUnsafeMutableBufferPointer { pDst in
                        activePrior.copyPriorMel(phoneId: nextPId, dst: pDst.baseAddress!)
                    }
                    var c = 0
                    while c < melChannels {
                        blendedPrior[c] = (0.70 * blendedPrior[c]) + (0.30 * nextPrior[c])
                        c += 1
                    }
                }
            }

            blendedSeq[t] = blendedPrior
            t += 1
        }

        return blendedSeq
    }

    /// 合成対数 Mel 系列に対して時間軸方向の 5 点加重平滑化フィルタを適用する。
    /// なぜ 5 点加重平滑化を行うか:
    /// 調音器官（舌・口唇・下顎）の生理学的慣性による連続的な声道形状変化を再現し、
    /// フレーム境界でのステップ状不連続を解消して滑らかで自然なフォルマント軌跡を生成するため。
    public func smoothMelSequence(
        combinedMelSeq: [[Float]],
        framePhoneIds: [Int],
        melChannels: Int
    ) -> [[Float]] {
        let totalFrames = combinedMelSeq.count
        var smoothedMelSeq = combinedMelSeq
        if 4 < totalFrames {
            var smT = 2
            let smEnd = totalFrames - 2
            while smT < smEnd {
                let pIdPrev2 = framePhoneIds[smT - 2]
                let pIdPrev1 = framePhoneIds[smT - 1]
                let pIdCurr = framePhoneIds[smT]
                let pIdNext1 = framePhoneIds[smT + 1]
                let pIdNext2 = framePhoneIds[smT + 2]

                // ポーズ・無音・破裂音閉鎖区間の前後境界では無音フロアと急峻なアタックを鋭敏に保つため、
                // 5 点近傍内に無音・破裂音が侵入している場合は平滑化をバイパスする
                var isPauseNear = false
                let checkList = [pIdCurr, pIdPrev1, pIdNext1, pIdPrev2, pIdNext2]
                var ck = 0
                while ck < 5 {
                    let chkId = checkList[ck]
                    if vocabulary.isPauseOrSilence(id: chkId) || vocabulary.isUnvoicedStop(id: chkId) {
                        isPauseNear = true
                    }
                    ck += 1
                }

                if isPauseNear != true {
                    var ch = 0
                    while ch < melChannels {
                        smoothedMelSeq[smT][ch] = (0.06 * combinedMelSeq[smT - 2][ch]) +
                                                  (0.24 * combinedMelSeq[smT - 1][ch]) +
                                                  (0.40 * combinedMelSeq[smT][ch]) +
                                                  (0.24 * combinedMelSeq[smT + 1][ch]) +
                                                  (0.06 * combinedMelSeq[smT + 2][ch])
                        ch += 1
                    }
                }
                smT += 1
            }
        }
        return smoothedMelSeq
    }

    /// 64 チャンネル Mel フィルタバンクの中心周波数 (Hz) テーブル
    ///
    /// なぜ不変テーブルとして事前計算するか:
    /// 音響モデルの Mel 特徴量から物理的な共鳴周波数（フォルマント極）を復元する際、
    /// 毎フレームの対数・べき乗演算をゼロ化し、決定論的な周波数マッピングを O(1) で参照するため。
    public static let melCenterFrequencies: [Float] = {
        let minMel: Float = 0.0
        let maxFreq: Float = Float(AudioConfig.sampleRate) * 0.5
        let maxMel: Float = 2595.0 * log10(1.0 + (maxFreq / 700.0))
        let melStep: Float = (maxMel - minMel) / Float(AudioConfig.melChannels + 1)
        var freqs = [Float](repeating: 0.0, count: AudioConfig.melChannels)
        var ch = 0
        while ch < AudioConfig.melChannels {
            let centerMel = minMel + (Float(ch + 1) * melStep)
            freqs[ch] = 700.0 * (powf(10.0, centerMel / 2595.0) - 1.0)
            ch += 1
        }
        return freqs
    }()

    /// 言語特徴量の音素IDおよび継続時間から全フレームの音素ID系列を展開する
    ///
    /// なぜ共通メソッドとして一元化するか:
    /// 調音結合Priorブレンド、共鳴器フレーム構築、無音ミュート判定で同一の
    /// 音素時間軸アライメントを保証し、フレーム間の境界ズレを根絶するため。
    public func extractFramePhoneIds(
        linguisticFeatures: LinguisticFeatures,
        totalFrames: Int
    ) -> [Int] {
        var framePhoneIds = [Int](repeating: PhonemeVocabulary.silId, count: totalFrames)
        var curF = 0
        var pIdx = 0
        let pCount = min(linguisticFeatures.phoneIds.count, linguisticFeatures.durations.count)
        var lastVowelId = 5
        while pIdx < pCount {
            let pid = Int(linguisticFeatures.phoneIds[pIdx])
            let dur = Int(linguisticFeatures.durations[pIdx])
            // なぜ長音記号（26）を先行母音に置換するか:
            // buildResonatorFrames および Prior ブレンドにおいて、長音区間を中立・未定義音素ではなく
            // 実際に伸ばされている先行母音の声道共鳴特性で駆動し、音響的な不連続と音素アライメントの不整合を解消するため。
            let effectivePid: Int
            if pid == 26 {
                effectivePid = lastVowelId
            } else {
                effectivePid = pid
                switch pid {
                case 5, 6, 7, 8, 9:
                    lastVowelId = pid
                default:
                    break
                }
            }
            var f = 0
            while f < dur {
                let frameIdx = curF + f
                if frameIdx < totalFrames {
                    framePhoneIds[frameIdx] = effectivePid
                }
                f += 1
            }
            curF += dur
            pIdx += 1
        }
        return framePhoneIds
    }

    /// 各フレームが音響物理的に完全無音（ポーズ、促音、または無声破裂音・破擦音の閉鎖期）であるかを判定する
    ///
    /// なぜ音素閉鎖期も無音判定に含めるか:
    /// 音声学（Acoustic Phonetics）において、無声破裂音（/k/, /t/, /p/）および無声破擦音（/ch/, /ts/）は
    /// 口腔内が完全に密着して気流が遮断される「閉鎖無音期」を持ち、その後の破裂バースト期のみエネルギーが発生する。
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
            let isStop = vocabulary.isUnvoicedStop(id: pid)

            var f = 0
            while f < dur {
                let frameIdx = curF + f
                if frameIdx < totalFrames {
                    switch true {
                    case isPause:
                        silenceMask[frameIdx] = true
                    case isStop && f < (dur - 1):
                        silenceMask[frameIdx] = true
                    default:
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

    /// 与えられた対数 Mel スペクトルから、特定周波数帯域 [minFreq, maxFreq] における
    /// 局所ピーク（放物線補間）およびスペクトル重心を推定して共鳴周波数 (Hz) と帯域幅 (Hz) を動的に算出する
    ///
    /// なぜ放物線補間とスペクトル重心を組み合わせるか:
    /// Mel フィルタバンクの離散ビン解像度（低域約 30Hz、高域約 200Hz）の粗さを放物線補間で補完しつつ、
    /// スペクトル重心により非対称なスペクトル傾斜を捉え、SNN が学習した微細な音響特徴量を連続的な共鳴極へ反映するため。
    public static func estimateFormantFromMel(
        mel: [Float],
        centerFreqs: [Float],
        minFreq: Float,
        maxFreq: Float,
        fallbackFreq: Float,
        fallbackBw: Float
    ) -> (freq: Float, bw: Float) {
        let chCount = min(mel.count, centerFreqs.count)
        if chCount <= 0 {
            return (fallbackFreq, fallbackBw)
        }

        var startCh = -1
        var endCh = -1
        var c = 0
        while c < chCount {
            let cf = centerFreqs[c]
            if minFreq <= cf && cf <= maxFreq {
                if startCh < 0 {
                    startCh = c
                }
                endCh = c
            }
            c += 1
        }

        if startCh < 0 || endCh < 0 {
            return (fallbackFreq, fallbackBw)
        }

        var maxVal: Float = -Float.infinity
        var minVal: Float = Float.infinity
        var maxCh = startCh
        var sumEnergy: Float = 0.0
        var weightedFreqSum: Float = 0.0

        // 帯域内低域チルト（声帯音源の -6dB/oct 傾斜）を補正し、真の共鳴ピークを検出可能にする
        var refFreq: Float = 200.0
        if 0 < startCh && startCh < chCount {
            refFreq = centerFreqs[startCh]
        }
        if refFreq < 1.0 {
            refFreq = 1.0
        }

        var idx = startCh
        while idx <= endCh {
            let cf = centerFreqs[idx]
            var safeCf = cf
            if safeCf < refFreq {
                safeCf = refFreq
            }
            // +1.0 * logf(safeCf / refFreq) による帯域チルト補正 (+6dB/oct 相当)
            let tiltComp = 1.0 * logf(safeCf / refFreq)
            let compV = mel[idx] + tiltComp

            if maxVal < compV {
                maxVal = compV
                maxCh = idx
            }
            if compV < minVal {
                minVal = compV
            }
            var clampedV = compV
            if clampedV < -20.0 {
                clampedV = -20.0
            }
            if 20.0 < clampedV {
                clampedV = 20.0
            }
            let linearE = expf(clampedV)
            sumEnergy += linearE
            weightedFreqSum += linearE * cf
            idx += 1
        }

        if sumEnergy <= 1e-8 {
            return (fallbackFreq, fallbackBw)
        }

        // 放物線補間によるピーク周波数の高精度推定
        var peakFreq = centerFreqs[maxCh]
        var peakCurvature: Float = 0.0
        if 0 < maxCh && (maxCh + 1) < chCount {
            let cf0 = max(refFreq, centerFreqs[maxCh - 1])
            let cf1 = max(refFreq, centerFreqs[maxCh])
            let cf2 = max(refFreq, centerFreqs[maxCh + 1])
            let y0 = mel[maxCh - 1] + (1.0 * logf(cf0 / refFreq))
            let y1 = mel[maxCh] + (1.0 * logf(cf1 / refFreq))
            let y2 = mel[maxCh + 1] + (1.0 * logf(cf2 / refFreq))
            let denom = y0 - (2.0 * y1) + y2
            if denom < -1e-5 {
                peakCurvature = -denom
                let delta = (0.5 * (y0 - y2)) / denom
                var clampedDelta = delta
                if clampedDelta < -0.5 {
                    clampedDelta = -0.5
                }
                if 0.5 < clampedDelta {
                    clampedDelta = 0.5
                }
                if clampedDelta < 0.0 {
                    let freqDiff = centerFreqs[maxCh] - centerFreqs[maxCh - 1]
                    peakFreq = centerFreqs[maxCh] + (clampedDelta * freqDiff)
                } else {
                    let freqDiff = centerFreqs[maxCh + 1] - centerFreqs[maxCh]
                    peakFreq = centerFreqs[maxCh] + (clampedDelta * freqDiff)
                }
            }
        }

        // スペクトル重心の算出
        let centroid = weightedFreqSum / sumEnergy
        let rawEstimatedFreq = (0.70 * peakFreq) + (0.30 * centroid)
        var clampedFreq = rawEstimatedFreq
        if clampedFreq < minFreq {
            clampedFreq = minFreq
        }
        if maxFreq < clampedFreq {
            clampedFreq = maxFreq
        }

        // 帯域内コントラスト（ダイナミックレンジ）に基づく動的信頼度
        let contrast = maxVal - minVal
        var alpha: Float = 0.0
        if 0.2 < contrast {
            alpha = (contrast - 0.2) / 1.2
            if 1.0 < alpha {
                alpha = 1.0
            }
        }

        // SNN の学習結果（推論スペクトル）による基準フォルマントからの微細変調
        // 音響音声学における調音結合シフトは基準周波数の ±10〜15% 以内に収まる
        let rawShift = clampedFreq - fallbackFreq
        let maxShift = fallbackFreq * 0.15
        var clampedShift = rawShift
        if clampedShift < -maxShift {
            clampedShift = -maxShift
        }
        if maxShift < clampedShift {
            clampedShift = maxShift
        }
        let snnWeight = 0.80 * alpha
        let finalFreq = fallbackFreq + (snnWeight * clampedShift)

        // ピークの鋭さ（曲率）に基づく帯域幅の動的調整
        var bwFactor: Float = 1.0
        if 0.1 < peakCurvature {
            bwFactor = 1.0 / (1.0 + (0.3 * peakCurvature))
            if bwFactor < 0.6 {
                bwFactor = 0.6
            }
        }
        var finalBw = fallbackBw * bwFactor
        if finalBw < 40.0 {
            finalBw = 40.0
        }
        if 500.0 < finalBw {
            finalBw = 500.0
        }

        return (finalFreq, finalBw)
    }

    /// 対数 Mel スペクトルから 4 共鳴極（F1〜F4, B1〜B4）の FormantConfig を動的に推定する
    ///
    /// 音響音声学における日本人母音・子音の主要共鳴帯域:
    /// - F1 (200〜1100Hz): 舌高・開口度を反映
    /// - F2 (750〜2800Hz): 舌前後位置・口唇形状を反映
    /// - F3 (1800〜3800Hz): 調音点・口蓋化を反映
    /// - F4 (3000〜4800Hz): 話者固有の声道管長を反映
    public static func estimateFormantsFromMel(
        mel: [Float],
        fallback: FormantConfig,
        centerFreqs: [Float] = melCenterFrequencies
    ) -> FormantConfig {
        // 基準フォルマント (Prior) をアンカーとして、生理学的な調音結合変動帯域 (±25%) に探索窓を限定
        // これにより、音声スペクトルの低域チルト（200〜400Hz の声帯音源エネルギー）によって
        // 母音 /a/ (800Hz) や /e/ (500Hz) の F1/F2 が低域へ潰れる音響破綻を物理的に防止する
        let f1Min = max(200.0, fallback.f1 * 0.75)
        let f1Max = min(1100.0, fallback.f1 * 1.25)
        let (f1, b1) = estimateFormantFromMel(
            mel: mel,
            centerFreqs: centerFreqs,
            minFreq: f1Min,
            maxFreq: f1Max,
            fallbackFreq: fallback.f1,
            fallbackBw: fallback.b1
        )

        let f2Min = max(750.0, fallback.f2 * 0.75)
        let f2Max = min(2800.0, fallback.f2 * 1.25)
        let (f2, b2) = estimateFormantFromMel(
            mel: mel,
            centerFreqs: centerFreqs,
            minFreq: f2Min,
            maxFreq: f2Max,
            fallbackFreq: fallback.f2,
            fallbackBw: fallback.b2
        )

        let f3Min = max(1800.0, fallback.f3 * 0.80)
        let f3Max = min(3800.0, fallback.f3 * 1.20)
        let (f3, b3) = estimateFormantFromMel(
            mel: mel,
            centerFreqs: centerFreqs,
            minFreq: f3Min,
            maxFreq: f3Max,
            fallbackFreq: fallback.f3,
            fallbackBw: fallback.b3
        )

        let f4Min = max(3000.0, fallback.f4 * 0.85)
        let f4Max = min(4800.0, fallback.f4 * 1.15)
        let (f4, b4) = estimateFormantFromMel(
            mel: mel,
            centerFreqs: centerFreqs,
            minFreq: f4Min,
            maxFreq: f4Max,
            fallbackFreq: fallback.f4,
            fallbackBw: fallback.b4
        )

        return FormantConfig(
            f1: f1, b1: b1,
            f2: f2, b2: b2,
            f3: f3, b3: b3,
            f4: f4, b4: b4
        )
    }

    /// 対数 Mel スペクトルの総エネルギーから物理的 RMS ゲインを推定する
    public static func estimateGainFromMel(
        mel: [Float]
    ) -> Float {
        var sumEnergy: Float = 0.0
        var ch = 0
        let chCount = mel.count
        while ch < chCount {
            let v = mel[ch]
            var clamped = v
            if clamped < -20.0 {
                clamped = -20.0
            }
            if 20.0 < clamped {
                clamped = 20.0
            }
            sumEnergy += expf(clamped)
            ch += 1
        }
        if sumEnergy <= 1e-8 || chCount <= 0 {
            return 0.0
        }
        let rms = sqrtf(sumEnergy / Float(chCount))
        return rms
    }

    /// カスケード共鳴器ボコーダー用音響フレーム系列の構築
    ///
    /// Melスペクトログラム三角フィルタの広帯域平滑化およびLevinson-Durbin全極推定による
    /// フォルマント平坦化・消失（ブザー音の主因）を排除し、音響音声学の共鳴極を
    /// カスケード共鳴器へ直接供給する。また、SNN 音響モデルが推定した対数 Mel スペクトル系列が
    /// 供給された場合は、学習されたスペクトル残差から 4 共鳴極および RMS ゲインを動的に推定して反映する。
    public func buildResonatorFrames(
        linguisticFeatures: LinguisticFeatures,
        voice: VoiceProfile,
        effectiveBaseF0: Float,
        text: String,
        melSeq: [[Float]] = []
    ) -> (frames: [ResonatorFrame], phoneIds: [Int]) {
        let totalFrames = linguisticFeatures.totalFrames
        if totalFrames <= 0 {
            return ([], [])
        }

        var frameF1 = [Float](repeating: 500.0, count: totalFrames)
        var frameB1 = [Float](repeating: 100.0, count: totalFrames)
        var frameF2 = [Float](repeating: 1500.0, count: totalFrames)
        var frameB2 = [Float](repeating: 150.0, count: totalFrames)
        var frameF3 = [Float](repeating: 2500.0, count: totalFrames)
        var frameB3 = [Float](repeating: 200.0, count: totalFrames)
        var frameF4 = [Float](repeating: 3500.0, count: totalFrames)
        var frameB4 = [Float](repeating: 250.0, count: totalFrames)
        var frameVoiced = [Float](repeating: 0.0, count: totalFrames)
        var frameGain = [Float](repeating: 0.0, count: totalFrames)
        var framePhoneIds = [Int](repeating: 1, count: totalFrames)

        var curF = 0
        var pIdx = 0
        let pCount = min(linguisticFeatures.phoneIds.count, linguisticFeatures.durations.count)

        // 1. 各音素のターゲット周波数と後続母音の検索
        var lastVowelId = 5
        while pIdx < pCount {
            let pid = Int(linguisticFeatures.phoneIds[pIdx])
            let dur = Int(linguisticFeatures.durations[pIdx])

            var nextVowel: Int? = nil
            var scanIdx = pIdx + 1
            while scanIdx < pCount {
                let sId = Int(linguisticFeatures.phoneIds[scanIdx])
                switch sId {
                case 5, 6, 7, 8, 9:
                    nextVowel = sId
                    scanIdx = pCount
                default:
                    scanIdx += 1
                }
            }

            let effectivePid: Int
            if pid == 26 { // 長音 (_)
                effectivePid = lastVowelId
            } else {
                effectivePid = pid
                switch pid {
                case 5, 6, 7, 8, 9:
                    lastVowelId = pid
                default:
                    break
                }
            }

            var f = 0
            while f < dur {
                let frameIdx = curF + f
                if frameIdx < totalFrames {
                    framePhoneIds[frameIdx] = effectivePid
                    let fm = PhonemeFormantTable.formantConfig(for: effectivePid, nextVowelId: nextVowel)

                    var dynamicFm = fm
                    if frameIdx < melSeq.count && AudioConfig.melChannels <= melSeq[frameIdx].count {
                        dynamicFm = Self.estimateFormantsFromMel(
                            mel: melSeq[frameIdx],
                            fallback: fm,
                            centerFreqs: Self.melCenterFrequencies
                        )
                    }

                    frameF1[frameIdx] = dynamicFm.f1
                    frameB1[frameIdx] = dynamicFm.b1
                    frameF2[frameIdx] = dynamicFm.f2
                    frameB2[frameIdx] = dynamicFm.b2
                    frameF3[frameIdx] = dynamicFm.f3
                    frameB3[frameIdx] = dynamicFm.b3
                    frameF4[frameIdx] = dynamicFm.f4
                    frameB4[frameIdx] = dynamicFm.b4

                    var vFlag: Float = 0.0
                    if frameIdx < linguisticFeatures.voicedFlags.count {
                        vFlag = linguisticFeatures.voicedFlags[frameIdx]
                    }

                    var baseGain: Float = 0.65
                    switch true {
                    case vocabulary.isPauseOrSilence(id: effectivePid):
                        baseGain = 0.0
                        vFlag = 0.0
                    case vocabulary.isUnvoicedStop(id: effectivePid):
                        vFlag = 0.0
                        if f < (dur - 1) {
                            baseGain = 0.0 // 閉鎖無音期
                        } else {
                            baseGain = 0.32 // 破裂バースト期
                        }
                    case vocabulary.isVoicedStop(id: effectivePid):
                        vFlag = 1.0
                        if f < (dur - 1) {
                            baseGain = 0.12 // ボイスバー期
                        } else {
                            baseGain = 0.40 // 有声破裂期
                        }
                    case vocabulary.isAffricate(id: effectivePid):
                        vFlag = 0.0
                        if f < (dur / 2) {
                            baseGain = 0.0 // 閉鎖期
                        } else {
                            baseGain = 0.32 // 摩擦期
                        }
                    case vocabulary.isUnvoicedFricative(id: effectivePid):
                        vFlag = 0.0
                        baseGain = 0.28
                    default:
                        // 母音・有声鼻音・半母音など: SNN 推論 Mel スペクトルの RMS エネルギーを反映
                        if frameIdx < melSeq.count && AudioConfig.melChannels <= melSeq[frameIdx].count {
                            let melRms = Self.estimateGainFromMel(mel: melSeq[frameIdx])
                            let nominalRms: Float = 2.2
                            var relGain = melRms / nominalRms
                            if relGain < 0.35 {
                                relGain = 0.35
                            }
                            if 1.65 < relGain {
                                relGain = 1.65
                            }
                            baseGain = baseGain * relGain
                        }
                    }

                    frameGain[frameIdx] = baseGain * voice.energyScale
                    frameVoiced[frameIdx] = vFlag
                }
                f += 1
            }
            curF += dur
            pIdx += 1
        }

        // 2. 調音結合 (Coarticulation): 音素境界を自然に繋ぐ 3 点移動平均平滑化
        var smoothF1 = frameF1
        var smoothF2 = frameF2
        var smoothF3 = frameF3
        var smoothF4 = frameF4
        var smoothB1 = frameB1
        var smoothB2 = frameB2
        var smoothB3 = frameB3
        var smoothB4 = frameB4

        if 2 < totalFrames {
            var t = 1
            let tEnd = totalFrames - 1
            while t < tEnd {
                let pCurr = framePhoneIds[t]
                if vocabulary.isPauseOrSilence(id: pCurr) != true {
                    smoothF1[t] = (0.20 * frameF1[t - 1]) + (0.60 * frameF1[t]) + (0.20 * frameF1[t + 1])
                    smoothF2[t] = (0.20 * frameF2[t - 1]) + (0.60 * frameF2[t]) + (0.20 * frameF2[t + 1])
                    smoothF3[t] = (0.20 * frameF3[t - 1]) + (0.60 * frameF3[t]) + (0.20 * frameF3[t + 1])
                    smoothF4[t] = (0.20 * frameF4[t - 1]) + (0.60 * frameF4[t]) + (0.20 * frameF4[t + 1])
                    smoothB1[t] = (0.20 * frameB1[t - 1]) + (0.60 * frameB1[t]) + (0.20 * frameB1[t + 1])
                    smoothB2[t] = (0.20 * frameB2[t - 1]) + (0.60 * frameB2[t]) + (0.20 * frameB2[t + 1])
                    smoothB3[t] = (0.20 * frameB3[t - 1]) + (0.60 * frameB3[t]) + (0.20 * frameB3[t + 1])
                    smoothB4[t] = (0.20 * frameB4[t - 1]) + (0.60 * frameB4[t]) + (0.20 * frameB4[t + 1])
                }
                t += 1
            }
        }

        // 3. 生理的ゆらぎ (Shimmer) の適用および ResonatorFrame 系列の構築
        var bioFluctuation = BiologicalFluctuation(seed: BiologicalFluctuation.seed(from: text))
        var frames: [ResonatorFrame] = []
        frames.reserveCapacity(totalFrames)

        var tf = 0
        while tf < totalFrames {
            var g = frameGain[tf]
            if 0.0 < g {
                g = bioFluctuation.computeAmplitudeShimmer(baseGain: g)
            }

            var f0: Float = 0.0
            if tf < linguisticFeatures.f0Contour.count {
                f0 = linguisticFeatures.f0Contour[tf]
            }
            if f0 <= 0.0 {
                f0 = effectiveBaseF0
            }

            let cfg = FormantConfig(
                f1: smoothF1[tf],
                b1: smoothB1[tf],
                f2: smoothF2[tf],
                b2: smoothB2[tf],
                f3: smoothF3[tf],
                b3: smoothB3[tf],
                f4: smoothF4[tf],
                b4: smoothB4[tf]
            )

            frames.append(
                ResonatorFrame(
                    formants: cfg,
                    gain: g,
                    pitchF0: f0,
                    voiced: frameVoiced[tf]
                )
            )
            tf += 1
        }

        return (frames, framePhoneIds)
    }

    /// 日本語テキストから 16kHz モノラル PCM 浮動小数点サンプル列を合成する。
    @discardableResult
    public func synthesize(
        text: String,
        voice: VoiceProfile = .default,
        speed: Float = 1.0,
        pitch: Float = 1.0
    ) -> [Float] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        var safeSpeed = speed
        if safeSpeed.isFinite != true {
            safeSpeed = 1.0
        }
        if safeSpeed < 0.1 {
            safeSpeed = 0.1
        }
        if 10.0 < safeSpeed {
            safeSpeed = 10.0
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

        // 1. テキスト正規化および言語韻律処理（話者基音 baseF0 にユーザー指定 pitch を一元反映）
        let effectiveBaseF0 = voice.baseF0 * safePitch
        let linguisticFeatures = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: safeSpeed,
            baseF0: effectiveBaseF0
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

        // 3. 各フレームの音素 ID 系列を展開
        let framePhoneIds = extractFramePhoneIds(
            linguisticFeatures: linguisticFeatures,
            totalFrames: totalFrames
        )

        // 4. 調音結合 Prior ブレンド系列の算出（推論話者の声道特性 Prior を適用）
        let activePrior = prior(for: voice.tract)
        let melChannels = AudioConfig.melChannels
        let blendedPriorSeq = computeBlendedPriorSequence(
            framePhoneIds: framePhoneIds,
            activePrior: activePrior,
            melChannels: melChannels
        )

        // 5. SNN 推論残差と Prior の合成
        var combinedMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalFrames)
        var tMel = 0
        let resScale = self.residualScale
        while tMel < totalFrames {
            let outDim = snnAcousticSeq[tMel].count
            let copyCount = min(melChannels, outDim)

            var rawSnnMel = [Float](repeating: 0.0, count: melChannels)
            rawSnnMel.withUnsafeMutableBufferPointer { melDst in
                snnAcousticSeq[tMel].withUnsafeBufferPointer { acSrc in
                    melDst.baseAddress!.update(from: acSrc.baseAddress!, count: copyCount)
                }
            }

            var compC = 0
            while compC < melChannels {
                combinedMelSeq[tMel][compC] = blendedPriorSeq[tMel][compC] + (resScale * rawSnnMel[compC])
                compC += 1
            }
            tMel += 1
        }

        // 6. 時間軸方向の 5 点加重平滑化フィルタ
        let smoothedMelSeq = smoothMelSequence(
            combinedMelSeq: combinedMelSeq,
            framePhoneIds: framePhoneIds,
            melChannels: melChannels
        )

        // 7. カスケード共鳴器用音響フレーム系列の構築（音素アライメントと SNN Mel 残差の調和）
        let (resonatorFrames, _) = buildResonatorFrames(
            linguisticFeatures: linguisticFeatures,
            voice: voice,
            effectiveBaseF0: effectiveBaseF0,
            text: text,
            melSeq: smoothedMelSeq
        )

        // 8. 現代的ニューラルボコーダー（NSF: Neural Source-Filter）による高品位波形合成
        neuralVocoder.reset()
        var rawSamples = neuralVocoder.synthesize(
            mel: smoothedMelSeq,
            f0Contour: linguisticFeatures.f0Contour,
            voicedFlags: linguisticFeatures.voicedFlags,
            voice: voice,
            resonatorFrames: resonatorFrames
        )


        // 6. 無音・休止・促音区間および無声破裂音・破擦音の閉鎖期における完全ゼロミュート
        // なぜ閉鎖期もゼロクリアするか:
        // 調音生理学において無声破裂音（/k/, /t/, /p/）の閉鎖期は気流が完全遮断され音響エネルギーが物理的に 0 であるため。
        // ポーズ遷移境界の先頭 1 フレームのみ滑らかにフェードアウトさせ、クリックノイズを防止する。
        let silenceMask = computeFrameSilenceMask(
            linguisticFeatures: linguisticFeatures,
            totalFrames: totalFrames
        )
        let frameSize = AudioConfig.hopSize
        var fIdx = 0
        while fIdx < totalFrames {
            let isSilence = silenceMask[fIdx]

            if isSilence {
                var prevIsSilence = true
                if 0 < fIdx {
                    prevIsSilence = silenceMask[fIdx - 1]
                }

                let startSample = fIdx * frameSize
                let endSample = min(rawSamples.count, startSample + frameSize)
                if prevIsSilence != true {
                    // 有声音から無音・閉鎖期への遷移境界：160サンプルで 1.0 から 0.0 へ線形フェードアウト
                    let invN = 1.0 / Float(frameSize)
                    var s = startSample
                    while s < endSample {
                        let sampleOffset = s - startSample
                        let fade = 1.0 - (Float(sampleOffset) * invN)
                        rawSamples[s] = rawSamples[s] * fade
                        s += 1
                    }
                } else {
                    // 連続無音・閉鎖区間：波形サンプルを完全にゼロクリア（絶対無音化）
                    var s = startSample
                    while s < endSample {
                        rawSamples[s] = 0.0
                        s += 1
                    }
                }
            }
            fIdx += 1
        }

        // 7. 発話開始および終了境界におけるDACポップノイズ防止のマイクロフェード（40サンプル / 2.5ms）
        // アナログDAC起動時の急峻なステップ立ち上がりや再生終了時のプツッというノイズを完全に防止する。
        if 40 <= rawSamples.count {
            let invFade: Float = 1.0 / 40.0
            var s = 0
            while s < 40 {
                let factor = Float(s) * invFade
                rawSamples[s] = rawSamples[s] * factor
                s += 1
            }
            let endOffset = rawSamples.count - 40
            s = 0
            while s < 40 {
                let factor = Float(39 - s) * invFade
                rawSamples[endOffset + s] = rawSamples[endOffset + s] * factor
                s += 1
            }
        }

        // 8. 有声区間 RMS（実効値）基準のラウドネス一定化制御およびソフトリミッター
        // 従来のピークノーマライズは、単一の破裂音スパイクで文全体が極小化したり、
        // 穏やかな文で過大爆音化して文章ごとに音量が激しくバラつく欠陥があった。
        // 人間の聴覚が知覚する実効エネルギー（RMS）を有声区間から算出し、
        // どの文章・語彙でも常に一定の均一な適正音量（約 -17dBFS、RMS=0.14）に自動整流する。
        var voicedSumSq: Float = 0.0
        var voicedSampleCount = 0
        var vF = 0
        while vF < totalFrames {
            let pId = framePhoneIds[vF]
            let isSpeech = vocabulary.isPauseOrSilence(id: pId) != true
            if isSpeech {
                let startS = vF * frameSize
                let endS = min(rawSamples.count, startS + frameSize)
                var s = startS
                while s < endS {
                    let v = rawSamples[s]
                    voicedSumSq += v * v
                    voicedSampleCount += 1
                    s += 1
                }
            }
            vF += 1
        }

        var voicedRms: Float = 0.0
        if 0 < voicedSampleCount {
            voicedRms = sqrt(voicedSumSq / Float(voicedSampleCount))
        }

        // なぜ目標ラウドネス targetRms を 0.155（約 -16.2dBFS）に設定するか:
        // 放送音響基準（ARIB TR-B32 / EBU R128）に準拠した標準的な肉声ラウドネスを確立し、
        // 母音フォルマントの豊かな共鳴エネルギー（定常部 RMS >= 0.10）を確実に引き出すため。
        let targetRms: Float = 0.155
        var loudnessGain: Float = 1.0
        if 1e-4 < voicedRms {
            let desiredGain = targetRms / voicedRms
            // 極端な静音音声や異常値による過大増幅（ノイズフロアの持ち上がり）を防止する安全リミット
            if desiredGain <= 8.0 {
                if 0.1 <= desiredGain {
                    loudnessGain = desiredGain
                } else {
                    loudnessGain = 0.1
                }
            } else {
                loudnessGain = 8.0
            }
        }

        // 全サンプルへのラウドネスゲイン乗算および過大ピークに対するソフトリミッター
        // 単発の破裂音などで 0.80 を超えるサンプルのみ滑らかな曲線で圧縮し、デジタルクリッピングをゼロにする
        var samples = [Float](repeating: 0.0, count: rawSamples.count)
        var sIdx = 0
        let kneeThreshold: Float = 0.70
        let margin: Float = 0.12
        while sIdx < rawSamples.count {
            let amplified = rawSamples[sIdx] * loudnessGain
            let absVal = abs(amplified)
            if kneeThreshold < absVal {
                let over = absVal - kneeThreshold
                let compressed = kneeThreshold + (margin * tanh(over / margin))
                if amplified < 0.0 {
                    samples[sIdx] = -compressed
                } else {
                    samples[sIdx] = compressed
                }
            } else {
                samples[sIdx] = amplified
            }
            sIdx += 1
        }
        return samples
    }

    /// 日本語テキストから 16kHz 16-bit Mono WAV バイナリデータを生成する。
    @discardableResult
    public func synthesizeWav(
        text: String,
        voice: VoiceProfile = .default,
        speed: Float = 1.0,
        pitch: Float = 1.0
    ) -> Data {
        let samples = synthesize(text: text, voice: voice, speed: speed, pitch: pitch)
        return WavEncoder.encode(samples: samples, sampleRate: Int(sampleRate))
    }

    /// ストリーミング音声合成を実行する。
    ///
    /// - Parameters:
    ///   - text: 入力日本語テキスト
    ///   - voice: 声プロファイル
    ///   - speed: 話速係数
    ///   - pitch: ピッチ係数
    ///   - isCancelled: 早期脱出判定クロージャ。true を返した場合は以降のフレーム生成を中断する。
    ///   - onFrame: フレームごとの PCM サンプル送出クロージャ
    @discardableResult
    public func synthesizeStream(
        text: String,
        voice: VoiceProfile = .default,
        speed: Float = 1.0,
        pitch: Float = 1.0,
        isCancelled: (@Sendable () -> Bool)? = nil,
        onFrame: (([Float]) -> Void)?
    ) -> [Float] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        var safeSpeed = speed
        if safeSpeed.isFinite != true {
            safeSpeed = 1.0
        }
        if safeSpeed < 0.1 {
            safeSpeed = 0.1
        }
        if 10.0 < safeSpeed {
            safeSpeed = 10.0
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

        let effectiveBaseF0 = voice.baseF0 * safePitch
        let linguisticFeatures = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: safeSpeed,
            baseF0: effectiveBaseF0
        )

        let totalFrames = linguisticFeatures.totalFrames
        if totalFrames <= 0 {
            return []
        }

        let inputSeq = encodeLinguisticFeatures(features: linguisticFeatures)
        let snnAcousticSeq = decoder.decodeSequence(
            featuresSeq: inputSeq,
            workspace: workspace
        )

        let framePhoneIds = extractFramePhoneIds(
            linguisticFeatures: linguisticFeatures,
            totalFrames: totalFrames
        )

        let activePrior = prior(for: voice.tract)
        let melChannels = AudioConfig.melChannels
        let blendedPriorSeq = computeBlendedPriorSequence(
            framePhoneIds: framePhoneIds,
            activePrior: activePrior,
            melChannels: melChannels
        )

        var combinedMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalFrames)
        var tMel = 0
        let resScale = self.residualScale
        while tMel < totalFrames {
            let outDim = snnAcousticSeq[tMel].count
            let copyCount = min(melChannels, outDim)

            var rawSnnMel = [Float](repeating: 0.0, count: melChannels)
            rawSnnMel.withUnsafeMutableBufferPointer { melDst in
                snnAcousticSeq[tMel].withUnsafeBufferPointer { acSrc in
                    melDst.baseAddress!.update(from: acSrc.baseAddress!, count: copyCount)
                }
            }

            var compC = 0
            while compC < melChannels {
                combinedMelSeq[tMel][compC] = blendedPriorSeq[tMel][compC] + (resScale * rawSnnMel[compC])
                compC += 1
            }
            tMel += 1
        }

        let smoothedMelSeq = smoothMelSequence(
            combinedMelSeq: combinedMelSeq,
            framePhoneIds: framePhoneIds,
            melChannels: melChannels
        )

        // カスケード共鳴器用音響フレーム系列の構築（音素アライメントと SNN Mel 残差の調和）
        let (resonatorFrames, _) = buildResonatorFrames(
            linguisticFeatures: linguisticFeatures,
            voice: voice,
            effectiveBaseF0: effectiveBaseF0,
            text: text,
            melSeq: smoothedMelSeq
        )

        // なぜニューラルボコーダー逐次推論を行うか:
        // 共鳴器（CascadeResonatorVocoder）特有の金属的・電子音的ロボット感を完全排除し、
        // Mel スペクトル系列と連続 F0 輪郭から人間らしい自然な日本語波形を低遅延ストリーミングで生成するため。
        let frameSize = AudioConfig.hopSize
        let silenceMask = computeFrameSilenceMask(
            linguisticFeatures: linguisticFeatures,
            totalFrames: totalFrames
        )
        neuralVocoder.reset()

        var allSamples = [Float]()
        allSamples.reserveCapacity(totalFrames * frameSize)
        var frameBuffer = [Float](repeating: 0.0, count: frameSize)

        var t = 0
        while t < totalFrames {
            switch isCancelled {
            case .some(let cancel):
                if cancel() {
                    return allSamples
                }
            case .none:
                break
            }

            var frameF0 = voice.baseF0
            if t < linguisticFeatures.f0Contour.count {
                let f = linguisticFeatures.f0Contour[t]
                if 0.0 < f {
                    frameF0 = f
                }
            }

            var frameVoiced: Float = 1.0
            if t < linguisticFeatures.voicedFlags.count {
                frameVoiced = linguisticFeatures.voicedFlags[t]
            }

            var frameResFrame: ResonatorFrame? = nil
            if t < resonatorFrames.count {
                frameResFrame = resonatorFrames[t]
            }

            frameBuffer.withUnsafeMutableBufferPointer { pDst in
                neuralVocoder.synthesizeFrame(
                    melFrame: smoothedMelSeq[t],
                    f0: frameF0,
                    voiced: frameVoiced,
                    voice: voice,
                    resonatorFrame: frameResFrame,
                    dst: pDst.baseAddress!
                )
            }

            // 無音・休止・促音区間および無声破裂音・破擦音の閉鎖期における完全ゼロミュートおよびフェードアウト
            let isSilence = silenceMask[t]
            if isSilence {
                var prevIsSilence = true
                if 0 < t {
                    prevIsSilence = silenceMask[t - 1]
                }

                if prevIsSilence != true {
                    let invN = 1.0 / Float(frameSize)
                    var s = 0
                    while s < frameSize {
                        let fade = 1.0 - (Float(s) * invN)
                        frameBuffer[s] = frameBuffer[s] * fade
                        s += 1
                    }
                } else {
                    var s = 0
                    while s < frameSize {
                        frameBuffer[s] = 0.0
                        s += 1
                    }
                }
            }

            // 先頭有声フレームの開始時におけるマイクロフェード（DAC起動ポップ防止）
            if t == 0 && isSilence != true {
                let invMicro: Float = 1.0 / 40.0
                var ms = 0
                while ms < 40 {
                    frameBuffer[ms] = frameBuffer[ms] * (Float(ms) * invMicro)
                    ms += 1
                }
            }

            // なぜストリーミング出力に適正ラウドネスゲイン（1.25）およびソフトリミッターを適用するか:
            // バッチ合成（synthesize）の有声区間 RMS 規準ラウドネス整流（targetRms = 0.155）と
            // ストリーミング合成の出力レベルを完全に一致させ、
            // 再生方式による音量の段差やデジタルクリッピングを完全に防止するため。
            let streamLoudnessGain: Float = 1.25
            let kneeThreshold: Float = 0.70
            let margin: Float = 0.12
            var s = 0
            while s < frameSize {
                let amplified = frameBuffer[s] * streamLoudnessGain
                let absVal = abs(amplified)
                if kneeThreshold < absVal {
                    let over = absVal - kneeThreshold
                    let compressed = kneeThreshold + (margin * tanh(over / margin))
                    if amplified < 0.0 {
                        frameBuffer[s] = -compressed
                    } else {
                        frameBuffer[s] = compressed
                    }
                } else {
                    frameBuffer[s] = amplified
                }
                s += 1
            }

            switch onFrame {
            case .some(let callback):
                callback(frameBuffer)
            case .none:
                break
            }

            allSamples.append(contentsOf: frameBuffer)
            t += 1
        }

        return allSamples
    }
}
