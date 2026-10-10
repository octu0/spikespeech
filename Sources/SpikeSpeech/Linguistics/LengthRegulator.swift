import Foundation

/// モーラ・音素発話継続時間の割り当ておよび累積和量子化フレーム展開器
///
/// 単純に各音素の予測継続時間を個別に四捨五入すると発話全体でフレーム丸め誤差が累積するため、
/// 累積和差分法により全体誤差を常に半フレーム以内に抑制する。
public final class LengthRegulator: Sendable {
    public let hiddenDimension: Int
    public let phonemeAverageDurations: [Int32: Float]
    public let meanFramesPerMora: Float
    public static let defaultMeanFramesPerMora: Float = 16.0 // 実測 ~160ms/モーラ

    /// アライメント実測統計に基づく音素 ID 別デフォルト平均継続時間 (1フレーム=10ms)
    /// なぜ実測統計をデフォルトとして保持するか:
    /// 16.0 モーラ固定や等時間割りを完全撤廃し、アライメントファイル未ロード時であっても
    /// 実音声データに裏打ちされた自然な物理時間比率（各音素平均）を推論正本とするため。
    public static let defaultPhonemeAverageDurations: [Int32: Float] = [
        1: 10.0,  // <sil>
        5: 7.9,   // a
        6: 6.9,   // i
        7: 6.1,   // u
        8: 7.1,   // e
        9: 7.9,   // o
        10: 6.5,  // k
        11: 7.7,  // s
        12: 6.0,  // t
        13: 6.5,  // n
        14: 5.8,  // h
        15: 7.0,  // m
        16: 7.6,  // y
        17: 7.7,  // r
        18: 7.2,  // w
        19: 6.5,  // g
        20: 7.1,  // z
        21: 6.1,  // d
        22: 6.2,  // b
        23: 6.9,  // p
        24: 7.1,  // N (撥音: ん)
        25: 9.1,  // Q (促音: っ)
        26: 8.6,  // _ (長音: ー)
        27: 8.1,  // sh
        28: 7.5,  // ch
        29: 6.1,  // ts
        30: 7.9,  // ky
        31: 6.2,  // ny
        32: 7.6,  // hy
        33: 8.7,  // my
        34: 7.1,  // ry
        35: 7.4,  // gy
        36: 7.3,  // j
        37: 5.1,  // by
        38: 4.8,  // py
        39: 5.6   // <pau>
    ]

    public init(
        hiddenDimension: Int = 64,
        phonemeAverageDurations: [Int32: Float]? = nil,
        meanFramesPerMora: Float? = nil
    ) {
        self.hiddenDimension = hiddenDimension
        switch phonemeAverageDurations {
        case .some(let table):
            self.phonemeAverageDurations = table
        case .none:
            self.phonemeAverageDurations = Self.defaultPhonemeAverageDurations
        }
        switch meanFramesPerMora {
        case .some(let m):
            self.meanFramesPerMora = m
        case .none:
            self.meanFramesPerMora = Self.defaultMeanFramesPerMora
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

    /// アクセント句列から実測音素平均の比率を維持したまま、文全体を会話速度（meanFramesPerMora × モーラ数）へスケーリングした継続フレーム系列を算出する（推論正本）
    ///
    /// なぜ文全体線形スケーリングとするか:
    /// モーラごとの均等割り（16フレーム箱）は音素間の自然な比率を破壊し、一方単純加算は母音短縮により早口化を招くため、
    /// MAS 反復自己収束で計測された各音素の精密な比率（子音閉鎖、母音、撥音、促音、長音）を 100% 保持したまま、
    /// 文全体に単一の共通係数を乗じて発話本体を目標会話速度（meanFramesPerMora × モーラ数）に合致させる。
    public func computeDataDrivenDurations(
        phrases: [AccentPhrase],
        speedFactor: Float = 1.0,
        applyFluctuation: Bool = true,
        text: String = "",
        meanFramesPerMora: Float? = nil
    ) -> [Float] {
        var rawDurations: [Float] = []
        var totalMoras = 0
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

        var pIdx = 0
        while pIdx < phrases.count {
            let phrase = phrases[pIdx]
            totalMoras += phrase.moras.count
            var mIdx = 0
            while mIdx < phrase.moras.count {
                let mora = phrase.moras[mIdx]
                var phIdx = 0
                while phIdx < mora.phonemes.count {
                    let token = mora.phonemes[phIdx]
                    let avgDur = phonemeDuration(phoneId: Int32(token.id), speedFactor: safeSpeed)
                    rawDurations.append(avgDur)
                    phIdx += 1
                }
                mIdx += 1
            }
            pIdx += 1
        }

        if rawDurations.isEmpty || totalMoras <= 0 {
            return rawDurations
        }

        var rawSum: Float = 0.0
        var rI = 0
        while rI < rawDurations.count {
            rawSum += rawDurations[rI]
            rI += 1
        }

        let effectiveMoraRate: Float
        switch meanFramesPerMora {
        case .some(let m):
            effectiveMoraRate = m
        case .none:
            effectiveMoraRate = self.meanFramesPerMora
        }

        // 目標本体フレーム数 (effectiveMoraRate / safeSpeed * totalMoras)
        let targetBodyFrames = (effectiveMoraRate / safeSpeed) * Float(totalMoras)
        var scaleFactor: Float = 1.0
        if 0.001 < rawSum {
            scaleFactor = targetBodyFrames / rawSum
        }

        var scaledDurations: [Float] = []
        var sI = 0
        while sI < rawDurations.count {
            var d = rawDurations[sI] * scaleFactor
            if d < 1.0 {
                d = 1.0
            }
            scaledDurations.append(d)
            sI += 1
        }

        return scaledDurations
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
        addBoundarySilence: Bool = false,
        meanFramesPerMora: Float? = nil
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
                text: text,
                meanFramesPerMora: meanFramesPerMora
            )
        case .none:
            let rawFloatDurations = computeDataDrivenDurations(
                phrases: phrases,
                speedFactor: safeSpeedFactor,
                applyFluctuation: applyFluctuation,
                text: text,
                meanFramesPerMora: meanFramesPerMora
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
            pIdx += 1
        }

        // 5. フラットな音素列と Duration 列の抽出（各音素平均の絶対長をそのまま反映）
        var phoneIds: [Int32] = []
        var durations: [Int32] = []
        var phoneAccent: [[Float]] = []

        var pListIdx = 0
        while pListIdx < phrases.count {
            let phrase = phrases[pListIdx]
            let moraCount = max(1, phrase.moras.count)
            var moraIdx = 0
            for mora in phrase.moras {
                let toneVal: Float = (mora.tone == .high) ? 1.0 : 0.0
                let posVal: Float = Float(moraIdx) / Float(moraCount)
                let kernelVal: Float = mora.isAccentKernel ? 1.0 : 0.0
                for phoneme in mora.phonemes {
                    phoneIds.append(Int32(phoneme.id))
                    durations.append(Int32(phoneme.durationFrames))
                    phoneAccent.append([toneVal, posVal, kernelVal])
                }
                moraIdx += 1
            }
            pListIdx += 1
        }

        // 6. F0 輪郭パラメータの生成（学習済み予測器を主経路とし、未指定時は藤崎モデル規則）
        let rawF0Contour: [Float]
        let rawVoicedFlags: [Float]
        let finalIntDurations = durations.map { Int($0) }
        switch prosodyPredictor {
        case .some(let predictor):
            let predRes = predictor.predictF0Contour(
                phrases: phrases,
                vocabulary: vocabulary,
                baseF0: baseF0,
                prosodyModel: prosodyModel,
                durations: finalIntDurations,
                applyFluctuation: applyFluctuation
            )
            rawF0Contour = predRes.f0Contour
            rawVoicedFlags = predRes.voicedFlags
        case .none:
            var bioFluctuation = BiologicalFluctuation(seed: BiologicalFluctuation.seed(from: text))
            let fujisakiRes = prosodyModel.generateF0Contour(
                phrases: phrases,
                vocabulary: vocabulary,
                baseF0: baseF0,
                fluctuation: &bioFluctuation,
                applyFluctuation: applyFluctuation
            )
            rawF0Contour = fujisakiRes.f0Contour
            rawVoicedFlags = fujisakiRes.voicedFlags
        }

        // 7. 発話本体総フレーム数の厳格な整合性保証
        // なぜ durations.reduce の合計値を totalFrames とするのか:
        // 外部の予測器や藤崎規則が不要なポーズや丸め誤差で異なるフレーム数を返した場合でも、
        // 音素 One-Hot 展開長（durations の総和）と F0/voiced/energy 輪郭のフレーム数を 100% 厳格に一致させ、
        // フレーム長不一致による音素崩れや末尾の無音ゴースト（足された pau）を根本排除するため。
        var bodyTotalFrames = 0
        var bI = 0
        while bI < durations.count {
            bodyTotalFrames += Int(durations[bI])
            bI += 1
        }
        let totalFrames = bodyTotalFrames

        var f0Contour = rawF0Contour
        if f0Contour.count < totalFrames {
            let lastF0: Float
            switch f0Contour.last {
            case .some(let v):
                lastF0 = v
            case .none:
                lastF0 = baseF0
            }
            let diff = totalFrames - f0Contour.count
            f0Contour.append(contentsOf: [Float](repeating: lastF0, count: diff))
        }
        if totalFrames < f0Contour.count {
            f0Contour = Array(f0Contour.prefix(totalFrames))
        }

        var voicedFlags = rawVoicedFlags
        if voicedFlags.count < totalFrames {
            let diff = totalFrames - voicedFlags.count
            voicedFlags.append(contentsOf: [Float](repeating: 0.0, count: diff))
        }
        if totalFrames < voicedFlags.count {
            voicedFlags = Array(voicedFlags.prefix(totalFrames))
        }

        // 8. 音素物理カテゴリに基づく音響エネルギー輪郭の生成
        // なぜ半正弦波窓を外しカテゴリピーク一定値とするか:
        // 音素境界での半正弦波窓減衰（端で約1割まで低下）による発音途切れ・デコボコを排除するため、
        // 音素持続時間内の全フレームにカテゴリピーク値を一定値として配置する（音素間線形補間は行わない）。
        var baseEnergyContour = [Float](repeating: 0.0, count: totalFrames)
        var curFrame = 0
        var phIter = 0
        while phIter < phoneIds.count {
            let pid = Int(phoneIds[phIter])
            let dur = Int(durations[phIter])

            switch vocabulary.isUnvoicedStop(id: pid) {
            case true:
                switch dur <= 2 {
                case true:
                    var f = 0
                    while f < dur {
                        let frameIdx = curFrame + f
                        if frameIdx < totalFrames {
                            baseEnergyContour[frameIdx] = 0.35
                        }
                        f += 1
                    }
                case false:
                    // 3 <= dur: 先頭 dur - 2 フレームを 0.02、末尾 2 フレームを 0.35
                    let closureFrames = dur - 2
                    var f = 0
                    while f < closureFrames {
                        let frameIdx = curFrame + f
                        if frameIdx < totalFrames {
                            baseEnergyContour[frameIdx] = 0.02
                        }
                        f += 1
                    }
                    while f < dur {
                        let frameIdx = curFrame + f
                        if frameIdx < totalFrames {
                            baseEnergyContour[frameIdx] = 0.35
                        }
                        f += 1
                    }
                }
            case false:
                let peakEnergy: Float
                switch true {
                case vocabulary.isPauseOrSilence(id: pid):
                    peakEnergy = 0.0
                case vocabulary.isUnvoicedFricative(id: pid):
                    peakEnergy = 0.18 // 無声摩擦気流
                case vocabulary.isAffricate(id: pid):
                    peakEnergy = 0.15 // 破擦音
                case vocabulary.isVoicedStop(id: pid):
                    peakEnergy = 0.25 // 有声破裂音
                default:
                    let symbol = vocabulary.token(for: pid)
                    switch vocabulary.isVoiced(symbol: symbol) {
                    case true:
                        switch symbol {
                        case "a", "i", "u", "e", "o", "N", "_":
                            peakEnergy = 0.70 // 母音・撥音・長音
                        default:
                            peakEnergy = 0.35 // その他有声子音（鼻音・半母音・弾音など）
                        }
                    case false:
                        peakEnergy = 0.10
                    }
                }

                var f = 0
                while f < dur {
                    let frameIdx = curFrame + f
                    if frameIdx < totalFrames {
                        baseEnergyContour[frameIdx] = peakEnergy
                    }
                    f += 1
                }
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

            var newAccent: [[Float]] = [[0.0, 0.0, 0.0]]
            newAccent.append(contentsOf: phoneAccent)
            newAccent.append([0.0, 0.0, 0.0])

            return LinguisticFeatures(
                phoneIds: newPhoneIds,
                durations: newDurations,
                f0Contour: newF0,
                voicedFlags: newVoiced,
                energyContour: newEnergy,
                totalFrames: newTotalFrames,
                phoneAccent: newAccent
            )
        }

        return LinguisticFeatures(
            phoneIds: phoneIds,
            durations: durations,
            f0Contour: f0Contour,
            voicedFlags: voicedFlags,
            energyContour: baseEnergyContour,
            totalFrames: totalFrames,
            phoneAccent: phoneAccent
        )
    }
}
