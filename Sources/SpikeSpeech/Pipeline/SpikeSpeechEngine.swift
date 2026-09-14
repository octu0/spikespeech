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
    public let vocoder: LPCVocoder
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
        // 大規模コーパス（500文以上）の学習過程で生じるSNNの高周波スペクトルゆらぎやざらつきノイズがボコーダーに過大に混入するのを防ぎ、
        // 事前フォルマントアンカー（Prior）の明瞭な母音共鳴を基盤として、子音やアクセントの質感のみを自然に付与するためデフォルト値を0.20とする。
        residualScale: Float = 0.20
    ) {
        self.weights = weights
        self.sampleRate = sampleRate
        self.residualScale = residualScale

        self.normalizer = TextNormalizer()
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
        self.vocoder = LPCVocoder(
            sampleRate: sampleRate,
            frameSize: AudioConfig.hopSize,
            lpcOrder: AudioConfig.lpcOrder,
            deEmphasisCoeff: 0.95
        )
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

    /// 日本語テキストから 16kHz モノラル PCM 浮動小数点サンプル列を合成する。
    @discardableResult
    public func synthesize(
        text: String,
        voice: VoiceProfile = .default,
        speed: Float = 1.0,
        pitch: Float = 1.0
    ) -> [Float] {
        if text.isEmpty {
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

        // 2. 言語特徴量のフレーム系列展開（F0 二重補正を完全撤廃）
        let inputSeq = encodeLinguisticFeatures(features: linguisticFeatures)

        // 3. 多層 SNN 音響デコーダー推論
        let acousticSeq = decoder.decodeSequence(
            featuresSeq: inputSeq,
            workspace: workspace
        )

        // 各フレームに対応する音素 ID、音素内オフセット、および音素継続フレーム数の時間軸展開マップの構築
        var framePhoneIds = [Int](repeating: 1, count: totalFrames)
        var framePhoneOffsets = [Int](repeating: 0, count: totalFrames)
        var framePhoneDurations = [Int](repeating: 1, count: totalFrames)
        var curFrame = 0
        var pIdx = 0
        let pCount = min(linguisticFeatures.phoneIds.count, linguisticFeatures.durations.count)
        while pIdx < pCount {
            let pId = linguisticFeatures.phoneIds[pIdx]
            let d = Int(linguisticFeatures.durations[pIdx])
            var f = 0
            while f < d {
                let frameIdx = curFrame + f
                if frameIdx < totalFrames {
                    framePhoneIds[frameIdx] = Int(pId)
                    framePhoneOffsets[frameIdx] = f
                    framePhoneDurations[frameIdx] = d
                }
                f += 1
            }
            curFrame += d
            pIdx += 1
        }

        // 4. 話者声道特性（VocalTract）に基づく真の VTLN Prior と SNN 残差の統合
        let activePrior = prior(for: voice.tract)
        let melChannels = melToLPC.melChannels
        var frames: [AcousticFrame] = []
        frames.reserveCapacity(totalFrames)

        // 全フレームの合成対数 Mel スペクトログラム作業バッファ [totalFrames * melChannels]
        var combinedMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalFrames)

        var t = 0
        while t < totalFrames {
            let outDim = acousticSeq[t].count
            let copyCount = min(melChannels, outDim)

            var rawSnnMel = [Float](repeating: 0.0, count: melChannels)
            rawSnnMel.withUnsafeMutableBufferPointer { melDst in
                acousticSeq[t].withUnsafeBufferPointer { acSrc in
                    melDst.baseAddress!.update(from: acSrc.baseAddress!, count: copyCount)
                }
            }

            // 音素境界における自然な調音結合 (Coarticulation) クロスフェードの算出
            let pId = framePhoneIds[t]
            let prevPId: Int
            if 0 < t {
                prevPId = framePhoneIds[t - 1]
            } else {
                prevPId = pId
            }

            let nextPId: Int
            if t + 1 < totalFrames {
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

            let resScale = self.residualScale
            var compC = 0
            while compC < melChannels {
                combinedMelSeq[t][compC] = blendedPrior[compC] + (resScale * rawSnnMel[compC])
                compC += 1
            }
            t += 1
        }

        // 時間軸方向の 5 点加重平滑化フィルタ (Temporal Coarticulation Smoothing: [0.06, 0.24, 0.40, 0.24, 0.06])
        // 調音器官（舌・唇・下顎）の生理学的慣性による連続的な声道形状変化を再現し、
        // フレーム境界でのステップ状不連続を解消して滑らかで自然なフォルマント軌跡を生成する
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
                // 5点近傍内に無音・破裂音が侵入している場合は平滑化をバイパスする
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

        var bioFluctuation = BiologicalFluctuation(seed: BiologicalFluctuation.seed(from: text))
        t = 0
        while t < totalFrames {
            let pId = framePhoneIds[t]
            let melVec = smoothedMelSeq[t]

            var lpcCoeffs = [Float](repeating: 0.0, count: AudioConfig.lpcOrder)
            let baseGain = melToLPC.convert(mel: melVec, isLogMel: true, outCoeffs: &lpcCoeffs)
            let gain = baseGain * voice.energyScale
            // 調音生理学（音声学）に基づく子音エネルギーと閉鎖区間の厳密制御
            // 人間の調音器官（舌・口蓋・唇）の物理的作用を模倣し、無声破裂音の閉鎖期（前半）での
            // 無声乱数ノイズ漏洩（「こ」等の発音開始前に鳴り響く「ザー」という異音）を根絶する。
            var effectiveGain = gain
            let pOffset = framePhoneOffsets[t]
            let pDur = framePhoneDurations[t]

            switch true {
            case vocabulary.isPauseOrSilence(id: pId):
                // 無音・休止・促音区間（<sil>, <pau>, <pad>, Q）:
                // 呼気・声帯振動が完全に遮断されているため絶対無音とする。
                effectiveGain = 0.0

            case vocabulary.isUnvoicedStop(id: pId):
                // 無声破裂音: k, t, p, ky, py
                // 舌や口唇が気流を完全に閉塞する「閉鎖期」（最終フレーム前）は物理的に音響エネルギーがゼロである。
                // 閉鎖期に無声乱数を注入すると母音立ち上がり前に「ザー」という異音となるため完全ミュート（0.0）とし、
                // 気流が急激に開放される直前の最後の 1 フレーム（破裂バースト）のみシャープに適正ゲインを付与する。
                if pOffset < (pDur - 1) {
                    effectiveGain = 0.0
                } else {
                    effectiveGain = min(0.18, gain * 0.50)
                }

            case vocabulary.isVoicedStop(id: pId):
                // 有声破裂音: g, d, b, gy, by
                // 閉鎖期は気流通過ノイズがゼロで微弱な低周波声帯振動（ボイスバー）のみ存在するため、
                // 閉鎖期は極小ゲインとし、最後の 1 フレームで破裂バーストを付与する。
                if pOffset < (pDur - 1) {
                    effectiveGain = min(0.04, gain * 0.15)
                } else {
                    effectiveGain = min(0.18, gain * 0.50)
                }

            case vocabulary.isAffricate(id: pId):
                // 無声破擦音: ch, ts
                // 閉鎖期から摩擦期への過渡的二相構造。前半（閉鎖区間）は呼気遮断のため無音（0.0）とし、
                // 後半（摩擦区間）のみ摩擦ノイズを発生させる。
                let halfDur = pDur / 2
                if pOffset < halfDur {
                    effectiveGain = 0.0
                } else {
                    effectiveGain = min(0.20, gain * 0.60)
                }

            case vocabulary.isUnvoicedFricative(id: pId):
                // 無声摩擦音: s, h, sh
                // 定常的な気流摩擦音。耳障りな過大ヒスノイズの突出を防止するため適正上限でクリップする。
                effectiveGain = min(0.22, gain * 0.65)

            default:
                break
            }

            // 生理学的 1/f 振幅ゆらぎ（Shimmer / 呼吸筋の微小な息の強弱）の適用
            if 0.0 < effectiveGain {
                effectiveGain = bioFluctuation.computeAmplitudeShimmer(baseGain: effectiveGain)
            }

            // 話者の絶対基音に基づく F0 周波数の直接適用（二重乗算を撤廃）
            var f0: Float = 0.0
            if t < linguisticFeatures.f0Contour.count {
                f0 = max(0.0, linguisticFeatures.f0Contour[t])
            }
            let voiced = linguisticFeatures.voicedFlags[t]

            frames.append(
                AcousticFrame(
                    lpcCoefficients: lpcCoeffs,
                    gain: effectiveGain,
                    pitchF0: f0,
                    voiced: voiced
                )
            )
            t += 1
        }

        // 5. Source-Filter LPC ボコーダーによる波形合成（話者声門パラメータを動的適用）
        vocoder.reset()
        vocoder.apply(glottal: voice.glottal)
        var rawSamples = vocoder.synthesize(frames: frames)

        // 6. 無音・休止・促音区間（<sil>, <pau>, <pad>, Q）におけるクリック防止フェードアウトと完全ゼロミュート
        // 人間の聴覚は文末や句読点ポーズ、促音での微小なヒスノイズ（ざーというホワイトノイズ）に極めて敏感であるため、
        // ポーズ区間では物理的にエネルギーを完全ゼロ（絶対無音）にする。
        // ただし、直前の有声サンプルから急峻に 0 にクリップすると不連続クリックノイズが発生するため、
        // 遷移境界の先頭 1 フレーム（10ms / 160サンプル）のみ滑らかにフェードアウトさせ、2フレーム目以降を完全ゼロクリアする。
        let frameSize = AudioConfig.hopSize
        var fIdx = 0
        while fIdx < totalFrames {
            let pId = framePhoneIds[fIdx]
            let isSilence = vocabulary.isPauseOrSilence(id: pId)

            if isSilence {
                var prevIsSilence = true
                if 0 < fIdx {
                    let prevPid = framePhoneIds[fIdx - 1]
                    prevIsSilence = vocabulary.isPauseOrSilence(id: prevPid)
                }

                let startSample = fIdx * frameSize
                let endSample = min(rawSamples.count, startSample + frameSize)
                if prevIsSilence != true {
                    // 有声音からポーズへの遷移境界：160サンプルで 1.0 から 0.0 へ線形フェードアウト
                    let invN = 1.0 / Float(frameSize)
                    var s = startSample
                    while s < endSample {
                        let sampleOffset = s - startSample
                        let fade = 1.0 - (Float(sampleOffset) * invN)
                        rawSamples[s] = rawSamples[s] * fade
                        s += 1
                    }
                } else {
                    // 連続ポーズ区間：波形サンプルを完全にゼロクリア（絶対無音化）
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

        let targetRms: Float = 0.14
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
        let kneeThreshold: Float = 0.80
        let margin: Float = 0.15
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
        if text.isEmpty {
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
        let acousticSeq = decoder.decodeSequence(
            featuresSeq: inputSeq,
            workspace: workspace
        )

        let activePrior = prior(for: voice.tract)
        let melChannels = melToLPC.melChannels
        let frameSize = AudioConfig.hopSize
        vocoder.reset()
        vocoder.apply(glottal: voice.glottal)

        // 各フレームに対応する音素 ID、音素内オフセット、および音素継続フレーム数の時間軸展開マップの構築
        var framePhoneIds = [Int](repeating: 1, count: totalFrames)
        var framePhoneOffsets = [Int](repeating: 0, count: totalFrames)
        var framePhoneDurations = [Int](repeating: 1, count: totalFrames)
        var curFrame = 0
        var pIdx = 0
        let pCount = min(linguisticFeatures.phoneIds.count, linguisticFeatures.durations.count)
        while pIdx < pCount {
            let pId = linguisticFeatures.phoneIds[pIdx]
            let d = Int(linguisticFeatures.durations[pIdx])
            var f = 0
            while f < d {
                let frameIdx = curFrame + f
                if frameIdx < totalFrames {
                    framePhoneIds[frameIdx] = Int(pId)
                    framePhoneOffsets[frameIdx] = f
                    framePhoneDurations[frameIdx] = d
                }
                f += 1
            }
            curFrame += d
            pIdx += 1
        }

        // 全フレームの合成対数 Mel スペクトログラム作業バッファ [totalFrames * melChannels]
        var combinedMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalFrames)

        var tMel = 0
        while tMel < totalFrames {
            let outDim = acousticSeq[tMel].count
            let copyCount = min(melChannels, outDim)

            var rawSnnMel = [Float](repeating: 0.0, count: melChannels)
            rawSnnMel.withUnsafeMutableBufferPointer { melDst in
                acousticSeq[tMel].withUnsafeBufferPointer { acSrc in
                    melDst.baseAddress!.update(from: acSrc.baseAddress!, count: copyCount)
                }
            }

            let pId = framePhoneIds[tMel]
            let prevPId: Int
            if 0 < tMel {
                prevPId = framePhoneIds[tMel - 1]
            } else {
                prevPId = pId
            }

            let nextPId: Int
            if tMel + 1 < totalFrames {
                nextPId = framePhoneIds[tMel + 1]
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

            let resScale = self.residualScale
            var compC = 0
            while compC < melChannels {
                combinedMelSeq[tMel][compC] = blendedPrior[compC] + (resScale * rawSnnMel[compC])
                compC += 1
            }
            tMel += 1
        }

        // 時間軸方向の 5 点加重平滑化フィルタ
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

        var allSamples = [Float](repeating: 0.0, count: totalFrames * frameSize)
        var frameBuffer = [Float](repeating: 0.0, count: frameSize)
        var lpcCoeffs = [Float](repeating: 0.0, count: AudioConfig.lpcOrder)
        var bioFluctuation = BiologicalFluctuation(seed: BiologicalFluctuation.seed(from: text))

        var t = 0
        while t < totalFrames {
            // キャンセル要求があれば以降のフレーム合成を即座に打ち切り、不要な CPU 演算とメモリ送出を抑止する
            if let isCancelled = isCancelled, isCancelled() {
                break
            }

            let pId = framePhoneIds[t]
            let melVec = smoothedMelSeq[t]

            let baseGain = melToLPC.convert(mel: melVec, isLogMel: true, outCoeffs: &lpcCoeffs)
            let gain = baseGain * voice.energyScale

            // 調音生理学（音声学）に基づく子音エネルギーと閉鎖区間の厳密制御
            var effectiveGain = gain
            let pOffset = framePhoneOffsets[t]
            let pDur = framePhoneDurations[t]

            switch true {
            case vocabulary.isPauseOrSilence(id: pId):
                effectiveGain = 0.0

            case vocabulary.isUnvoicedStop(id: pId):
                // 無声破裂音: k, t, p, ky, py
                // 閉鎖期は完全無音（0.0）、最後の1フレームのみ破裂バースト
                if pOffset < (pDur - 1) {
                    effectiveGain = 0.0
                } else {
                    effectiveGain = min(0.18, gain * 0.50)
                }

            case vocabulary.isVoicedStop(id: pId):
                // 有声破裂音: g, d, b, gy, by
                if pOffset < (pDur - 1) {
                    effectiveGain = min(0.04, gain * 0.15)
                } else {
                    effectiveGain = min(0.18, gain * 0.50)
                }

            case vocabulary.isAffricate(id: pId):
                // 無声破擦音: ch, ts
                let halfDur = pDur / 2
                if pOffset < halfDur {
                    effectiveGain = 0.0
                } else {
                    effectiveGain = min(0.20, gain * 0.60)
                }

            case vocabulary.isUnvoicedFricative(id: pId):
                // 無声摩擦音: s, h, sh
                effectiveGain = min(0.22, gain * 0.65)

            default:
                break
            }

            // 生理学的 1/f 振幅ゆらぎ（Shimmer / 呼吸筋の微小な息の強弱）の適用
            if 0.0 < effectiveGain {
                effectiveGain = bioFluctuation.computeAmplitudeShimmer(baseGain: effectiveGain)
            }

            // 話者の絶対基音に基づく F0 周波数の直接適用（二重乗算を撤廃）
            var f0: Float = 0.0
            if t < linguisticFeatures.f0Contour.count {
                f0 = max(0.0, linguisticFeatures.f0Contour[t])
            }
            let voiced = linguisticFeatures.voicedFlags[t]

            let acousticFrame = AcousticFrame(
                lpcCoefficients: lpcCoeffs,
                gain: effectiveGain,
                pitchF0: f0,
                voiced: voiced
            )

            frameBuffer.withUnsafeMutableBufferPointer { fBuf in
                vocoder.synthesizeFrame(frame: acousticFrame, dst: fBuf.baseAddress!)
            }

            let headroomMargin: Float = 0.88
            var f = 0
            while f < frameSize {
                frameBuffer[f] = frameBuffer[f] * headroomMargin
                f += 1
            }

            // 無音・休止・促音区間（<sil>, <pau>, <pad>, Q）におけるクリック防止フェードアウトと完全ゼロミュート
            var isSilence = false
            switch pId {
            case PhonemeVocabulary.silId, PhonemeVocabulary.pauId, PhonemeVocabulary.padId, PhonemeVocabulary.qId:
                isSilence = true
            default:
                break
            }

            if isSilence {
                var prevIsSilence = true
                if 0 < t {
                    let prevPid = framePhoneIds[t - 1]
                    switch prevPid {
                    case PhonemeVocabulary.silId, PhonemeVocabulary.pauId, PhonemeVocabulary.padId, PhonemeVocabulary.qId:
                        prevIsSilence = true
                    default:
                        prevIsSilence = false
                    }
                }

                if prevIsSilence != true {
                    // 有声音からポーズへの遷移境界：160サンプルで 1.0 から 0.0 へ線形フェードアウト
                    let invN = 1.0 / Float(frameSize)
                    var s = 0
                    while s < frameSize {
                        let fade = 1.0 - (Float(s) * invN)
                        frameBuffer[s] = frameBuffer[s] * fade
                        s += 1
                    }
                } else {
                    // 連続ポーズ区間：波形サンプルを完全にゼロクリア（絶対無音化）
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

            let offset = t * frameSize
            allSamples.withUnsafeMutableBufferPointer { dstBuf in
                frameBuffer.withUnsafeBufferPointer { srcBuf in
                    dstBuf.baseAddress!.advanced(by: offset).update(from: srcBuf.baseAddress!, count: frameSize)
                }
            }

            onFrame?(frameBuffer)
            t += 1
        }

        // キャンセルにより早期脱出した場合は、生成が完了した有効フレーム分のサンプル数にバッファを切り詰める
        let generatedSamplesCount = t * frameSize
        if generatedSamplesCount < allSamples.count {
            allSamples.removeSubrange(generatedSamplesCount..<allSamples.count)
        }

        return allSamples
    }
}
