import Foundation

/// 学習済み韻律予測器（Duration & F0）
///
/// 規則表・藤崎規則モデルを初期特徴量・Prior としつつ、
/// 学習されたニューラル予測重みにより自然な音素継続長と豊かな F0 抑揚輪郭を主経路として生成する。
public final class ProsodyPredictor: Sendable {
    public let weights: ProsodyWeights?

    public init(weights: ProsodyWeights? = nil) {
        self.weights = weights
    }

    /// 各音素の継続時間特徴量ベクトルを構築する
    public static func extractDurationFeatures(
        token: PhonemeToken,
        mora: MoraToken,
        phrase: AccentPhrase,
        isLastMoraInPhrase: Bool,
        isLastPhonemeInMora: Bool,
        ruleDuration: Float,
        inputDim: Int = 72
    ) -> [Float] {
        var feat = [Float](repeating: 0.0, count: inputDim)

        // 1. 音素 ID One-Hot (0..<64)
        let pid = token.id
        if 0 <= pid {
            if pid < 64 {
                if pid < inputDim {
                    feat[pid] = 1.0
                }
            }
        }

        // 2. 音素カテゴリ (64..67)
        if 64 < inputDim {
            switch token.category {
            case .vowel:
                feat[64] = 1.0
            case .consonant, .contracted:
                if 65 < inputDim {
                    feat[65] = 1.0
                }
            case .pause:
                if 66 < inputDim {
                    feat[66] = 1.0
                }
            case .geminate, .nasalSyllable, .prolonged:
                if 67 < inputDim {
                    feat[67] = 1.0
                }
            }
        }

        // 3. アクセントトーン (68)
        if 68 < inputDim {
            switch mora.tone {
            case .high:
                feat[68] = 1.0
            case .low:
                feat[68] = 0.0
            }
        }

        // 4. 句末フラグ (69)
        if 69 < inputDim {
            switch isLastMoraInPhrase {
            case true:
                feat[69] = 1.0
            case false:
                feat[69] = 0.0
            }
        }

        // 5. モーラ末フラグ (70)
        if 70 < inputDim {
            switch isLastPhonemeInMora {
            case true:
                feat[70] = 1.0
            case false:
                feat[70] = 0.0
            }
        }

        // 6. 規則 duration (71)
        if 71 < inputDim {
            var normD = ruleDuration / 20.0
            if normD < 0.0 {
                normD = 0.0
            }
            if 5.0 < normD {
                normD = 5.0
            }
            feat[71] = normD
        }

        return feat
    }

    /// 各フレームの F0 特徴量ベクトルを構築する
    public static func extractF0Features(
        phoneId: Int,
        phonemeProgress: Float,
        duration: Int,
        fujisakiF0: Float,
        isHighTone: Bool,
        phraseProgress: Float,
        isVoiced: Bool,
        sentenceProgress: Float,
        isQuestion: Bool,
        isAccentNucleus: Bool = false,
        isFlatAccent: Bool = false,
        isVowel: Bool = false,
        inputDim: Int = 76
    ) -> [Float] {
        var feat = [Float](repeating: 0.0, count: inputDim)

        // 1. 音素 ID One-Hot (0..<64)
        if 0 <= phoneId {
            if phoneId < 64 {
                if phoneId < inputDim {
                    feat[phoneId] = 1.0
                }
            }
        }

        // 2. 音素内進行度 (64)
        if 64 < inputDim {
            var p = phonemeProgress
            if p < 0.0 { p = 0.0 }
            if 1.0 < p { p = 1.0 }
            feat[64] = p
        }

        // 3. 音素 duration (65)
        if 65 < inputDim {
            let durScale = 10.0 / Float(max(1, duration))
            var d = durScale
            if 2.0 < d { d = 2.0 }
            feat[65] = d
        }

        // 4. 藤崎モデル大局的対数比率特徴量 (66)
        // なぜ対数比率特徴量を渡すか:
        // 完成 F0 の掛け算 Prior は完全撤廃したが、文頭上昇・句下降・アクセント核のマクロな抑揚コンテキストを
        // 入力特徴量として提供することで、全発話の平均への退行を防ぎ目標 MAE < 20 Hz への収束を可能にするため。
        if 66 < inputDim {
            var fujiNorm: Float = 0.0
            if 0.0 < fujisakiF0 {
                fujiNorm = logf(fujisakiF0 / 220.0)
            }
            if fujiNorm < -1.5 {
                fujiNorm = -1.5
            }
            if 1.5 < fujiNorm {
                fujiNorm = 1.5
            }
            feat[66] = fujiNorm
        }

        // 5. アクセントトーン (67)
        if 67 < inputDim {
            switch isHighTone {
            case true:
                feat[67] = 1.0
            case false:
                feat[67] = 0.0
            }
        }

        // 6. 句内進行度 (68)
        if 68 < inputDim {
            var pp = phraseProgress
            if pp < 0.0 { pp = 0.0 }
            if 1.0 < pp { pp = 1.0 }
            feat[68] = pp
        }

        // 7. 有声フラグ (69)
        if 69 < inputDim {
            switch isVoiced {
            case true:
                feat[69] = 1.0
            case false:
                feat[69] = 0.0
            }
        }

        // 8. 文内進行度 (70)
        if 70 < inputDim {
            var sp = sentenceProgress
            if sp < 0.0 { sp = 0.0 }
            if 1.0 < sp { sp = 1.0 }
            feat[70] = sp
        }

        // 9. 疑問文フラグ (71)
        if 71 < inputDim {
            switch isQuestion {
            case true:
                feat[71] = 1.0
            case false:
                feat[71] = 0.0
            }
        }

        // 10. アクセント核フラグ (72)
        if 72 < inputDim {
            switch isAccentNucleus {
            case true:
                feat[72] = 1.0
            case false:
                feat[72] = 0.0
            }
        }

        // 11. 平板型アクセントフラグ (73)
        if 73 < inputDim {
            switch isFlatAccent {
            case true:
                feat[73] = 1.0
            case false:
                feat[73] = 0.0
            }
        }

        // 12. 藤崎対数オフセット (74)
        if 74 < inputDim {
            var logOffset: Float = 0.0
            if 0.0 < fujisakiF0 {
                logOffset = logf(fujisakiF0 / 220.0)
            }
            if logOffset < -1.5 { logOffset = -1.5 }
            if 1.5 < logOffset { logOffset = 1.5 }
            feat[74] = logOffset
        }

        // 13. 母音フラグ (75)
        if 75 < inputDim {
            switch isVowel {
            case true:
                feat[75] = 1.0
            case false:
                feat[75] = 0.0
            }
        }

        return feat
    }

    /// 音素系列から学習済みモデルにより音素フレーム数を予測する
    public func predictDurations(
        phrases: [AccentPhrase],
        vocabulary: PhonemeVocabulary,
        lengthRegulator: LengthRegulator,
        speedFactor: Float = 1.0,
        applyFluctuation: Bool = true,
        text: String = ""
    ) -> [Int] {
        let rawFloatDurations = lengthRegulator.computeDataDrivenDurations(
            phrases: phrases,
            speedFactor: speedFactor,
            applyFluctuation: applyFluctuation,
            text: text
        )
        return lengthRegulator.quantizeDurations(durations: rawFloatDurations)
    }

    /// フレーム特徴から学習済みモデルにより F0 輪郭 [Hz] を予測する
    public func predictF0Contour(
        phrases: [AccentPhrase],
        vocabulary: PhonemeVocabulary,
        baseF0: Float,
        prosodyModel: ProsodyModel,
        durations: [Int] = [],
        applyFluctuation: Bool = true
    ) -> (f0Contour: [Float], voicedFlags: [Float], totalFrames: Int) {
        // durations が渡されている場合、フレーズ内の各音素 durationFrames に反映した workingPhrases を生成する
        var workingPhrases = phrases
        if durations.isEmpty != true {
            var qIdx = 0
            var p = 0
            while p < workingPhrases.count {
                var m = 0
                while m < workingPhrases[p].moras.count {
                    var ph = 0
                    while ph < workingPhrases[p].moras[m].phonemes.count {
                        if qIdx < durations.count {
                            workingPhrases[p].moras[m].phonemes[ph].durationFrames = durations[qIdx]
                            qIdx += 1
                        }
                        ph += 1
                    }
                    m += 1
                }
                if workingPhrases[p].pauseAfter && 0 < workingPhrases[p].pauseDurationFrames {
                    if qIdx < durations.count {
                        workingPhrases[p].pauseDurationFrames = durations[qIdx]
                        qIdx += 1
                    }
                }
                p += 1
            }
        }

        // まず藤崎モデル規則輪郭を取得（Prior として使用）
        var bioFluctuation = BiologicalFluctuation(seed: 2026)
        let (fujisakiF0, voicedFlags, totalFrames) = prosodyModel.generateF0Contour(
            phrases: workingPhrases,
            vocabulary: vocabulary,
            baseF0: baseF0,
            fluctuation: &bioFluctuation,
            applyFluctuation: applyFluctuation
        )

        let fWeights = weights?.f0Weights
        guard let fw = fWeights else {
            return (fujisakiF0, voicedFlags, totalFrames)
        }

        var resultF0 = [Float](repeating: 0.0, count: totalFrames)
        if totalFrames <= 0 {
            return (resultF0, voicedFlags, totalFrames)
        }

        let inD = fw.inputDim
        let hidD = fw.hiddenDim

        // 1. 全フレームの特徴量ベクトルを平坦化配列として収集 [totalFrames * inD]
        var allFeatures = [Float](repeating: 0.0, count: totalFrames * inD)
        var isVoicedFrame = [Bool](repeating: false, count: totalFrames)

        var curF = 0
        var pIdx = 0
        while pIdx < workingPhrases.count {
            let phrase = workingPhrases[pIdx]
            let phraseTotalFrames = phrase.moras.reduce(0) { total, mora in
                total + mora.phonemes.reduce(0) { $0 + $1.durationFrames }
            }
            let phraseStartF = curF

            var mIdx = 0
            while mIdx < phrase.moras.count {
                let mora = phrase.moras[mIdx]
                let isHigh = (mora.tone == .high)

                var phIdx = 0
                while phIdx < mora.phonemes.count {
                    let token = mora.phonemes[phIdx]
                    let dur = token.durationFrames
                    let isVoiced = vocabulary.isVoiced(symbol: token.symbol)

                    var f = 0
                    while f < dur {
                        let frameIdx = curF + f
                        if frameIdx < totalFrames {
                            let baseF = fujisakiF0[frameIdx]
                            var vFlag: Float = 0.0
                            if frameIdx < voicedFlags.count {
                                vFlag = voicedFlags[frameIdx]
                            }

                            let pProg = Float(f) / Float(max(1, dur))
                            let phraseFrame = (curF - phraseStartF) + f
                            let phProg = Float(phraseFrame) / Float(max(1, phraseTotalFrames))
                            let sProg = Float(frameIdx) / Float(max(1, totalFrames))
                            let isAccentNucleus = mora.isAccentKernel
                            let isFlatAccent = phrase.moras.allSatisfy { $0.isAccentKernel != true }
                            let isVowel = (token.category == .vowel)

                            let feat = ProsodyPredictor.extractF0Features(
                                phoneId: token.id,
                                phonemeProgress: pProg,
                                duration: dur,
                                fujisakiF0: baseF,
                                isHighTone: isHigh,
                                phraseProgress: phProg,
                                isVoiced: isVoiced,
                                sentenceProgress: sProg,
                                isQuestion: phrase.isQuestion,
                                isAccentNucleus: isAccentNucleus,
                                isFlatAccent: isFlatAccent,
                                isVowel: isVowel,
                                inputDim: inD
                            )

                            let featOffset = frameIdx * inD
                            var fi = 0
                            while fi < inD {
                                allFeatures[featOffset + fi] = feat[fi]
                                fi += 1
                            }

                            if 0.5 <= vFlag && 0.0 < baseF {
                                isVoicedFrame[frameIdx] = true
                            }
                        }
                        f += 1
                    }
                    curF += dur
                    phIdx += 1
                }
                mIdx += 1
            }

            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                curF += phrase.pauseDurationFrames
            }
            pIdx += 1
        }

        // 2. 初段線形射影 + LeakyReLU: H0 [totalFrames * hidD]
        var h0 = [Float](repeating: 0.0, count: totalFrames * hidD)
        var t = 0
        while t < totalFrames {
            let featRow = t * inD
            let hRow = t * hidD
            var h = 0
            while h < hidD {
                var dot = fw.b1[h]
                let wRow = h * inD
                var j = 0
                while j < inD {
                    dot += fw.w1[wRow + j] * allFeatures[featRow + j]
                    j += 1
                }
                var act = dot
                if dot < 0.0 {
                    act = dot * 0.1
                }
                h0[hRow + h] = act
                h += 1
            }
            t += 1
        }

        // 3. 時間畳み込み（1D Depthwise Conv, K=5 / K=3, replicate padding）+ 残差加算: M [totalFrames * hidD]
        var mConv = [Float](repeating: 0.0, count: totalFrames * hidD)
        let kernelSize = fw.wConv.count / max(1, hidD)
        t = 0
        while t < totalFrames {
            let hRowCur = t * hidD
            switch kernelSize {
            case 5:
                var tM2 = t - 2
                if tM2 < 0 {
                    tM2 = 0
                }
                var tM1 = t - 1
                if tM1 < 0 {
                    tM1 = 0
                }
                var tP1 = t + 1
                if totalFrames <= tP1 {
                    tP1 = totalFrames - 1
                }
                var tP2 = t + 2
                if totalFrames <= tP2 {
                    tP2 = totalFrames - 1
                }

                let rM2 = tM2 * hidD
                let rM1 = tM1 * hidD
                let rP1 = tP1 * hidD
                let rP2 = tP2 * hidD

                var h = 0
                while h < hidD {
                    let w0 = fw.wConv[(0 * hidD) + h]
                    let w1 = fw.wConv[(1 * hidD) + h]
                    let w2 = fw.wConv[(2 * hidD) + h]
                    let w3 = fw.wConv[(3 * hidD) + h]
                    let w4 = fw.wConv[(4 * hidD) + h]

                    let z0 = h0[rM2 + h] * w0
                    let z1 = h0[rM1 + h] * w1
                    let z2 = h0[hRowCur + h] * w2
                    let z3 = h0[rP1 + h] * w3
                    let z4 = h0[rP2 + h] * w4
                    let zConv = (z0 + z1) + (z2 + z3) + z4

                    let sumVal = h0[hRowCur + h] + zConv
                    var actM = sumVal
                    if sumVal < 0.0 {
                        actM = sumVal * 0.1
                    }
                    mConv[hRowCur + h] = actM
                    h += 1
                }
            default:
                var tPrev = t - 1
                if tPrev < 0 {
                    tPrev = 0
                }
                var tNext = t + 1
                if totalFrames <= tNext {
                    tNext = totalFrames - 1
                }

                let hRowPrev = tPrev * hidD
                let hRowNext = tNext * hidD

                var h = 0
                while h < hidD {
                    let w0 = fw.wConv[(0 * hidD) + h]
                    let w1 = fw.wConv[(1 * hidD) + h]
                    let w2 = fw.wConv[(2 * hidD) + h]

                    let z0 = h0[hRowPrev + h] * w0
                    let z1 = h0[hRowCur + h] * w1
                    let z2 = h0[hRowNext + h] * w2
                    let zConv = (z0 + z1) + z2

                    let sumVal = h0[hRowCur + h] + zConv
                    var actM = sumVal
                    if sumVal < 0.0 {
                        actM = sumVal * 0.1
                    }
                    mConv[hRowCur + h] = actM
                    h += 1
                }
            }
            t += 1
        }

        // 4. 終段線形射影
        // 新モデル (Wave 2c) は初期値 log(220) ≈ 5.39 の対数 Hz 直接予測モデルであり、
        // 旧モデル (b2 ≈ 0) は藤崎モデルとの対数比率オフセット予測モデルである。
        // b2[0] のスケールで両者を明確に判別し、後方互換性と新仕様を完全両立する。
        let isDirectLogHzModel = 3.0 <= fw.b2[0]
        let pitchShiftLog = logf(max(0.1, baseF0 / 220.0))
        let minLogF0 = logf(max(50.0, baseF0 * 0.60)) // baseF0 220.0 -> 132.0 Hz
        let maxLogF0 = logf(min(450.0, baseF0 * 1.75)) // baseF0 220.0 -> 385.0 Hz

        t = 0
        while t < totalFrames {
            if isVoicedFrame[t] {
                let mRow = t * hidD
                var outVal = fw.b2[0]
                var h = 0
                while h < hidD {
                    outVal += fw.w2[h] * mConv[mRow + h]
                    h += 1
                }

                switch isDirectLogHzModel {
                case true:
                    // Wave 2c 新仕様: 対数 Hz 直接予測
                    var shiftedLogHz = outVal + pitchShiftLog
                    if shiftedLogHz < minLogF0 {
                        shiftedLogHz = minLogF0
                    }
                    if maxLogF0 < shiftedLogHz {
                        shiftedLogHz = maxLogF0
                    }
                    resultF0[t] = expf(shiftedLogHz)
                case false:
                    // 旧仕様後方互換: オフセット予測
                    var safeOffset = outVal
                    if safeOffset < -0.8 {
                        safeOffset = -0.8
                    }
                    if 0.8 < safeOffset {
                        safeOffset = 0.8
                    }
                    let baseF = fujisakiF0[t]
                    var predF0 = baseF * expf(safeOffset)
                    let minF0 = max(45.0, baseF0 * 0.40)
                    let maxF0 = min(600.0, baseF0 * 2.50)
                    if predF0 < minF0 {
                        predF0 = minF0
                    }
                    if maxF0 < predF0 {
                        predF0 = maxF0
                    }
                    resultF0[t] = predF0
                }
            } else {
                resultF0[t] = 0.0
            }
            t += 1
        }

        // 5. 有声区間の時間的平滑化（3点加重移動平均 [0.20, 0.60, 0.20]）
        // なぜ平滑化を行うか:
        // フレーム単位の予測における局所的な高周波ジッターや不自然なスパイクを抑え、
        // 生理学的な声帯筋の滑らかなピッチ推移（C1連続性）を保証するため。
        switch isDirectLogHzModel {
        case true:
            var smoothedF0 = resultF0
            var i = 1
            while i < totalFrames - 1 {
                if isVoicedFrame[i - 1] && isVoicedFrame[i] && isVoicedFrame[i + 1] {
                    smoothedF0[i] = (0.20 * resultF0[i - 1]) + (0.60 * resultF0[i]) + (0.20 * resultF0[i + 1])
                }
                i += 1
            }
            resultF0 = smoothedF0
        case false:
            break
        }

        return (resultF0, voicedFlags, totalFrames)
    }
}
