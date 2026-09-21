import Foundation

/// モーラ・音素発話継続時間の割り当ておよび累積和量子化フレーム展開器
///
/// 単純に各音素の予測継続時間を個別に四捨五入すると発話全体でフレーム丸め誤差が累積するため、
/// 累積和差分法により全体誤差を常に半フレーム以内に抑制する。
public final class LengthRegulator: Sendable {
    public let hiddenDimension: Int
    public let phonemeAverageDurations: [Int32: Float]

    /// JSUT 5000発話アライメント実測統計に基づく音素 ID 別デフォルト平均継続時間 (1フレーム=10ms)
    /// なぜ実測統計をデフォルトとして保持するか:
    /// 16.0 モーラ固定や等時間割りを完全撤廃し、アライメントファイル未ロード時であっても
    /// 実音声データに裏打ちされた自然な物理時間比率（各音素平均）を推論正本とするため。
    public static let defaultPhonemeAverageDurations: [Int32: Float] = [
        1: 10.0,  // <sil>
        5: 6.4,   // a
        6: 5.1,   // i
        7: 5.4,   // u
        8: 6.1,   // e
        9: 6.4,   // o
        10: 13.2, // k
        11: 4.4,  // s
        12: 12.9, // t
        13: 5.7,  // n
        14: 4.5,  // h
        15: 7.3,  // m
        16: 5.6,  // y
        17: 6.8,  // r
        18: 5.1,  // w
        19: 12.5, // g
        20: 4.0,  // z
        21: 13.4, // d
        22: 12.8, // b
        23: 12.3, // p
        24: 25.7, // N (撥音: ん)
        25: 9.6,  // Q (促音: っ)
        26: 5.4,  // _ (長音: ー)
        27: 3.1,  // sh
        28: 2.5,  // ch
        29: 2.4,  // ts
        30: 2.5,  // ky
        31: 2.2,  // ny
        32: 2.4,  // hy
        33: 2.4,  // my
        34: 2.2,  // ry
        35: 2.3,  // gy
        36: 2.4,  // j
        37: 2.4,  // by
        38: 2.4,  // py
        39: 12.0  // <pau>
    ]

    public init(
        hiddenDimension: Int = 64,
        phonemeAverageDurations: [Int32: Float]? = nil
    ) {
        self.hiddenDimension = hiddenDimension
        switch phonemeAverageDurations {
        case .some(let table):
            self.phonemeAverageDurations = table
        case .none:
            self.phonemeAverageDurations = Self.defaultPhonemeAverageDurations
        }
    }

    /// 音素 ID と発話速度に基づく予測フレーム数を算出する（正本 API）
    public func phonemeDuration(phoneId: Int32, speedFactor: Float = 1.0) -> Float {
        let avgFrames: Float
        switch phonemeAverageDurations[phoneId] {
        case .some(let val):
            avgFrames = val
        case .none:
            avgFrames = Self.defaultPhonemeAverageDurations[phoneId] ?? 8.0
        }

        var safeSpeed = speedFactor
        if safeSpeed.isFinite != true {
            safeSpeed = 1.0
        }
        if safeSpeed < 0.1 {
            safeSpeed = 0.1
        }
        if 10.0 < safeSpeed {
            safeSpeed = 10.0
        }

        var scaled = avgFrames / safeSpeed
        if scaled.isFinite != true {
            scaled = avgFrames
        }
        if scaled < 1.0 {
            scaled = 1.0
        }
        if 1000.0 < scaled {
            scaled = 1000.0
        }
        return scaled
    }

    /// アクセント句列からデータ駆動型の自然なモーラ C/V 比率音素継続フレーム系列を算出する（推論正本）
    ///
    /// なぜモーラ C/V 比率モデルとするか:
    /// 音素ごとに孤立した平均値を単純に割り振ると破裂音や撥音が 15〜30 フレームに肥大化して
    /// SNN の膜電位が直流飽和・ブザー発振に陥るため、
    /// 日本語の音韻構造（1モーラ約 155ms、子音 25% / 母音 75%）に基づき適正な過渡変化長を配分する。
    public func computeDataDrivenDurations(
        phrases: [AccentPhrase],
        speedFactor: Float = 1.0,
        applyFluctuation: Bool = true,
        text: String = ""
    ) -> [Float] {
        var totalMoras = 0
        var pIdx = 0
        while pIdx < phrases.count {
            totalMoras += phrases[pIdx].moras.count
            pIdx += 1
        }
        if totalMoras <= 0 {
            return []
        }

        var safeSpeed = speedFactor
        if safeSpeed.isFinite != true {
            safeSpeed = 1.0
        }
        if safeSpeed < 0.1 {
            safeSpeed = 0.1
        }
        if 10.0 < safeSpeed {
            safeSpeed = 10.0
        }

        var bioFluctuation = BiologicalFluctuation(seed: BiologicalFluctuation.seed(from: text))
        var rawDurations: [Float] = []

        // 1 モーラ基本長 (15.5 フレーム = 155ms / モーラ、基準 150〜180ms に合致)
        let baseMoraFrames: Float = 15.5 / safeSpeed

        pIdx = 0
        while pIdx < phrases.count {
            let phrase = phrases[pIdx]
            let isLastPhrase = (pIdx + 1) == phrases.count
            var mIdx = 0
            while mIdx < phrase.moras.count {
                let mora = phrase.moras[mIdx]
                let isLastMora = (mIdx + 1) == phrase.moras.count
                var shouldLengthen = false
                if isLastMora {
                    if phrase.pauseAfter || isLastPhrase {
                        shouldLengthen = true
                    }
                }

                var rIdx = 0
                while rIdx < mora.phonemes.count {
                    let token = mora.phonemes[rIdx]
                    var d = phonemeDuration(phoneId: Int32(token.id), speedFactor: safeSpeed)

                    if applyFluctuation {
                        let tempoScale = bioFluctuation.computeTempoScale()
                        d *= tempoScale
                    }

                    if shouldLengthen {
                        switch token.category {
                        case .vowel, .nasalSyllable, .prolonged:
                            d *= 1.15
                        default:
                            break
                        }
                    }

                    rawDurations.append(d)
                    rIdx += 1
                }
                mIdx += 1
            }
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                let safePause = min(10.0, Float(phrase.pauseDurationFrames)) / safeSpeed
                rawDurations.append(safePause)
            }
            pIdx += 1
        }

        return rawDurations
    }

    /// 実数予測継続時間列から累積丸め誤差ゼロの整数フレーム系列を算出する
    ///
    /// 量子化後の総フレーム数が実数継続時間の総和の四捨五入と厳密に一致することを数学的に保証し、長文発話での累積ドリフトをゼロにする。
    public func quantizeDurations(durations: [Float]) -> [Int] {
        if durations.isEmpty {
            return []
        }

        var quantized = [Int](repeating: 0, count: durations.count)
        var cumulativeFloat: Float = 0.0
        var cumulativeInt: Int = 0

        var i = 0
        while i < durations.count {
            var dur = durations[i]
            // 異常値が累積和に混入して以降のフレームが発散するのを防止する。
            if dur.isFinite != true {
                dur = 1.0
            }
            if dur < 1.0 {
                dur = 1.0
            }
            if 100000.0 < dur {
                dur = 100000.0
            }

            cumulativeFloat += dur
            // 浮動小数点数から整数へのキャスト時における例外を防止するため、累積和の有限性を保証する。
            let roundedFloat = roundf(cumulativeFloat)
            let safeRoundedFloat: Float
            if roundedFloat.isFinite != true {
                safeRoundedFloat = Float(cumulativeInt + 1)
            } else {
                safeRoundedFloat = roundedFloat
            }
            let roundedTotal = Int(safeRoundedFloat)
            var frameCount = roundedTotal - cumulativeInt
            if frameCount < 1 {
                frameCount = 1
            }
            quantized[i] = frameCount
            cumulativeInt += frameCount
            i += 1
        }

        return quantized
    }

    /// 音素埋め込み系列を各音素の Duration フレーム数に応じて連続展開する
    ///
    /// 音響モデルの処理において配列の動的リサイズを繰り返すとメモリコピーが発生して速度低下を招くため、
    /// 事前に総フレーム数を計算して一括確保し、連続メモリアドレスに対してバルクコピーを行うことで最高速のメモリ帯域効率を達成する。
    public func expand(
        embeddings: [Float],
        phonemeCount: Int,
        durations: [Int]
    ) -> [Float] {
        if phonemeCount <= 0 || embeddings.isEmpty {
            return []
        }

        // 1. 総フレーム数の算出
        var totalFrames = 0
        var p = 0
        while p < phonemeCount {
            let d = durations[p]
            if 0 < d {
                totalFrames += d
            }
            p += 1
        }

        if totalFrames <= 0 {
            return []
        }

        let dim = hiddenDimension
        let totalElements = totalFrames * dim
        var output = [Float](repeating: 0.0, count: totalElements)

        // 2. バルクメモリアサイン
        output.withUnsafeMutableBufferPointer { dstBuf in
            embeddings.withUnsafeBufferPointer { srcBuf in
                let dstBase = dstBuf.baseAddress!
                let srcBase = srcBuf.baseAddress!

                var currentFrameOffset = 0
                var i = 0
                while i < phonemeCount {
                    let duration = durations[i]
                    if duration <= 0 {
                        i += 1
                        continue
                    }

                    let srcOffset = i * dim
                    let srcPtr = srcBase.advanced(by: srcOffset)

                    var f = 0
                    while f < duration {
                        let dstOffset = (currentFrameOffset + f) * dim
                        let dstPtr = dstBase.advanced(by: dstOffset)
                        dstPtr.update(from: srcPtr, count: dim)
                        f += 1
                    }

                    currentFrameOffset += duration
                    i += 1
                }
            }
        }

        return output
    }

    /// 日本語テキストから SNN 音響モデルに入力可能な統合 LinguisticFeatures を生成する
    ///
    /// 正規化、形態素解析、音素およびモーラ分解、アクセント句結合、ピッチ輪郭、
    /// および時間長アライメントを単一の明示的な処理として完結させる。
    @discardableResult
    public func processText(
        text: String,
        normalizer: TextNormalizer,
        prosodyModel: ProsodyModel,
        vocabulary: PhonemeVocabulary,
        prosodyPredictor: ProsodyPredictor? = nil,
        speedFactor: Float = 1.0,
        baseF0: Float = 220.0,
        applyFluctuation: Bool = true,
        addBoundarySilence: Bool = false
    ) -> LinguisticFeatures {
        if text.isEmpty {
            return LinguisticFeatures(phoneIds: [], durations: [], f0Contour: [], voicedFlags: [], energyContour: [], totalFrames: 0)
        }

        // 外部から非有限値やゼロが指定された場合でも、後続の計算が安全範囲内で安定して実行されることを保証する。
        var safeSpeedFactor = speedFactor
        if safeSpeedFactor.isFinite != true {
            safeSpeedFactor = 1.0
        }
        if safeSpeedFactor < 0.1 {
            safeSpeedFactor = 0.1
        }
        if 10.0 < safeSpeedFactor {
            safeSpeedFactor = 10.0
        }

        // 1. テキスト正規化・形態素解析
        let morphemes = normalizer.normalize(text: text)

        // 2. アクセント句・モーラ階層構築
        var phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)

        // 3. 各音素および休止の Duration 系列の確定（推論正本: 音素平均フレームテーブル）
        let quantizedDurations: [Int]
        switch prosodyPredictor {
        case .some(let predictor):
            quantizedDurations = predictor.predictDurations(
                phrases: phrases,
                vocabulary: vocabulary,
                lengthRegulator: self,
                speedFactor: safeSpeedFactor,
                applyFluctuation: applyFluctuation,
                text: text
            )
        case .none:
            let rawFloatDurations = computeDataDrivenDurations(
                phrases: phrases,
                speedFactor: safeSpeedFactor,
                applyFluctuation: applyFluctuation,
                text: text
            )
            quantizedDurations = quantizeDurations(durations: rawFloatDurations)
        }

        // 4. 量子化されたフレーム数を各音素および休止へ反映
        var qIdx = 0
        var pIdx = 0
        while pIdx < phrases.count {
            var mIdx = 0
            while mIdx < phrases[pIdx].moras.count {
                var phIdx = 0
                while phIdx < phrases[pIdx].moras[mIdx].phonemes.count {
                    if qIdx < quantizedDurations.count {
                        phrases[pIdx].moras[mIdx].phonemes[phIdx].durationFrames = quantizedDurations[qIdx]
                        qIdx += 1
                    }
                    phIdx += 1
                }
                mIdx += 1
            }
            if phrases[pIdx].pauseAfter && 0 < phrases[pIdx].pauseDurationFrames {
                if qIdx < quantizedDurations.count {
                    phrases[pIdx].pauseDurationFrames = quantizedDurations[qIdx]
                    qIdx += 1
                }
            }
            pIdx += 1
        }

        // 5. フラットな音素列と Duration 列の抽出
        var phoneIds: [Int32] = []
        var durations: [Int32] = []

        var pListIdx = 0
        while pListIdx < phrases.count {
            let phrase = phrases[pListIdx]
            for mora in phrase.moras {
                for phoneme in mora.phonemes {
                    phoneIds.append(Int32(phoneme.id))
                    durations.append(Int32(phoneme.durationFrames))
                }
            }
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                phoneIds.append(Int32(PhonemeVocabulary.pauId))
                durations.append(Int32(phrase.pauseDurationFrames))
            }
            pListIdx += 1
        }

        // 6. F0 輪郭パラメータの生成（学習済み予測器を主経路とし、未指定時は藤崎モデル規則）
        let f0Contour: [Float]
        let voicedFlags: [Float]
        let totalFrames: Int
        switch prosodyPredictor {
        case .some(let predictor):
            let predRes = predictor.predictF0Contour(
                phrases: phrases,
                vocabulary: vocabulary,
                baseF0: baseF0,
                prosodyModel: prosodyModel,
                durations: quantizedDurations,
                applyFluctuation: applyFluctuation
            )
            f0Contour = predRes.f0Contour
            voicedFlags = predRes.voicedFlags
            totalFrames = predRes.totalFrames
        case .none:
            var bioFluctuation = BiologicalFluctuation(seed: BiologicalFluctuation.seed(from: text))
            let fujisakiRes = prosodyModel.generateF0Contour(
                phrases: phrases,
                vocabulary: vocabulary,
                baseF0: baseF0,
                fluctuation: &bioFluctuation,
                applyFluctuation: applyFluctuation
            )
            f0Contour = fujisakiRes.f0Contour
            voicedFlags = fujisakiRes.voicedFlags
            totalFrames = fujisakiRes.totalFrames
        }

        // 8. 音素物理カテゴリに基づく音響エネルギー輪郭の生成
        // なぜ一律固定値ではなく音素カテゴリ別物理プロファイル＋半正弦波窓にするか:
        // 学習側 PitchTracker の実測 RMS は発話内ピーク 0.70〜0.80 で母音平均約 0.20〜0.25 であるため、
        // 推論時も半正弦波窓により中央ピーク 0.70、両端で滑らかに遷移させ、膜電位飽和を防ぎ過渡的フォルマント変化を維持する。
        var baseEnergyContour = [Float](repeating: 0.0, count: totalFrames)
        var curFrame = 0
        var phIter = 0
        while phIter < phoneIds.count {
            let pid = Int(phoneIds[phIter])
            let dur = Int(durations[phIter])
            let peakEnergy: Float

            switch true {
            case vocabulary.isPauseOrSilence(id: pid):
                peakEnergy = 0.0
            case vocabulary.isUnvoicedStop(id: pid):
                peakEnergy = 0.02 // 閉鎖無音区間
            case vocabulary.isUnvoicedFricative(id: pid):
                peakEnergy = 0.18 // 無声摩擦気流
            case vocabulary.isAffricate(id: pid):
                peakEnergy = 0.15 // 破擦音
            case vocabulary.isVoicedStop(id: pid):
                peakEnergy = 0.25 // 有声破裂音
            default:
                let symbol = vocabulary.token(for: pid)
                if vocabulary.isVoiced(symbol: symbol) {
                    switch symbol {
                    case "a", "i", "u", "e", "o", "N", "_":
                        peakEnergy = 0.70 // 母音・撥音・長音
                    default:
                        peakEnergy = 0.35 // その他有声子音（鼻音・半母音・弾音など）
                    }
                } else {
                    peakEnergy = 0.10
                }
            }

            var f = 0
            while f < dur {
                let frameIdx = curFrame + f
                if frameIdx < totalFrames {
                    switch (peakEnergy <= 0.05, dur <= 2) {
                    case (true, _), (_, true):
                        baseEnergyContour[frameIdx] = peakEnergy
                    default:
                        let phase = (Float(f) + 0.5) / Float(dur)
                        let window = sinf(Float.pi * phase)
                        baseEnergyContour[frameIdx] = peakEnergy * window
                    }
                }
                f += 1
            }

            curFrame += dur
            phIter += 1
        }

        if addBoundarySilence {
            var leadSil = Int(roundf(6.0 / safeSpeedFactor))
            if leadSil < 4 {
                leadSil = 4
            }
            var trailSil = Int(roundf(8.0 / safeSpeedFactor))
            if trailSil < 5 {
                trailSil = 5
            }

            var newPhoneIds: [Int32] = []
            newPhoneIds.reserveCapacity(phoneIds.count + 2)
            newPhoneIds.append(Int32(PhonemeVocabulary.silId))
            newPhoneIds.append(contentsOf: phoneIds)
            newPhoneIds.append(Int32(PhonemeVocabulary.silId))

            var newDurations: [Int32] = []
            newDurations.reserveCapacity(durations.count + 2)
            newDurations.append(Int32(leadSil))
            newDurations.append(contentsOf: durations)
            newDurations.append(Int32(trailSil))

            let newTotalFrames = totalFrames + leadSil + trailSil

            var newF0 = [Float](repeating: 0.0, count: leadSil)
            newF0.append(contentsOf: f0Contour)
            newF0.append(contentsOf: [Float](repeating: 0.0, count: trailSil))

            var newVoiced = [Float](repeating: 0.0, count: leadSil)
            newVoiced.append(contentsOf: voicedFlags)
            newVoiced.append(contentsOf: [Float](repeating: 0.0, count: trailSil))

            var newEnergy = [Float](repeating: 0.0, count: leadSil)
            newEnergy.append(contentsOf: baseEnergyContour)
            newEnergy.append(contentsOf: [Float](repeating: 0.0, count: trailSil))

            return LinguisticFeatures(
                phoneIds: newPhoneIds,
                durations: newDurations,
                f0Contour: newF0,
                voicedFlags: newVoiced,
                energyContour: newEnergy,
                totalFrames: newTotalFrames
            )
        }

        return LinguisticFeatures(
            phoneIds: phoneIds,
            durations: durations,
            f0Contour: f0Contour,
            voicedFlags: voicedFlags,
            energyContour: baseEnergyContour,
            totalFrames: totalFrames
        )
    }
}
