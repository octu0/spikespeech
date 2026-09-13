import Foundation

/// モーラ・音素発話継続時間の割り当ておよび累積和量子化フレーム展開器
///
/// 単純に各音素の予測継続時間を個別に四捨五入すると発話全体でフレーム丸め誤差が累積するため、
/// 累積和差分法により全体誤差を常に半フレーム以内に抑制する。
public final class LengthRegulator: Sendable {
    public let hiddenDimension: Int

    public init(hiddenDimension: Int = 64) {
        self.hiddenDimension = hiddenDimension
    }

    /// 音素カテゴリに応じた連続浮動小数点継続時間を算出する
    ///
    /// 累積和量子化に入力する浮動小数点継続時間列を生成し、
    /// 丸め誤差の累積による発話時間ドリフトをゼロにする。
    public func floatDurationFrames(category: PhonemeCategory, symbol: String, speed: Float = 1.0) -> Float {
        var baseFrames: Float = 8.0

        switch category {
        case .vowel:
            // 狭母音 (i, u) は開口度が小さく短め (約 70ms)、広母音 (a, o, e) は明瞭な調音のため長め (約 85ms) に設定
            switch symbol {
            case "i", "u":
                baseFrames = 7.5
            default:
                baseFrames = 8.5
            }
        case .consonant:
            // 摩擦音 (s, sh, h) は気流維持のため長め (約 50ms)、破裂音 (k, t, p) は急峻な破裂のため短め (約 35ms)
            switch symbol {
            case "s", "sh", "h", "z", "j":
                baseFrames = 5.0
            case "k", "t", "p", "g", "d", "b":
                baseFrames = 3.5
            default:
                baseFrames = 4.0
            }
        case .contracted:
            baseFrames = 5.0
        case .geminate:
            // 促音 (っ) は明瞭な音節境界知覚のため十分な無音閉鎖間隔 (約 105ms) を確保
            baseFrames = 10.5
        case .nasalSyllable:
            baseFrames = 8.0
        case .prolonged:
            baseFrames = 9.0
        case .pause:
            if symbol == "<sil>" {
                baseFrames = 30.0
            } else {
                baseFrames = 15.0
            }
        }

        // ゼロ除算や非有限値によるフレーム展開破綻、および整数キャスト時の実行時例外を防止するため、速度値を安全な実数範囲に制限する。
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

        var scaled = baseFrames / safeSpeed
        if scaled.isFinite != true {
            scaled = baseFrames
        }
        if scaled < 1.0 {
            scaled = 1.0
        }
        if 1000.0 < scaled {
            scaled = 1000.0
        }
        return scaled
    }

    /// 音素カテゴリに応じた標準継続フレーム数を算出する
    ///
    /// 音韻生理学に基づき、促音の閉鎖持続時間や母音の定常部など音素ごとの特性に応じたフレーム数を算出する。
    public func defaultDurationFrames(category: PhonemeCategory, symbol: String, speed: Float = 1.0) -> Int {
        let scaled = floatDurationFrames(category: category, symbol: symbol, speed: speed)
        // 浮動小数点数から整数へのキャスト時に非有限値が混入した場合の実行時例外を防止する。
        if scaled.isFinite != true {
            return 8
        }
        var rounded = roundf(scaled)
        if rounded < 1.0 {
            rounded = 1.0
        }
        if 100000.0 < rounded {
            rounded = 100000.0
        }
        return Int(rounded)
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
        speedFactor: Float = 1.0
    ) -> LinguisticFeatures {
        if text.isEmpty {
            return LinguisticFeatures(phoneIds: [], durations: [], f0Contour: [], voicedFlags: [], totalFrames: 0)
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

        // 3. 各音素および休止の浮動小数点 Duration 系列の収集
        var rawFloatDurations: [Float] = []
        var pIdx = 0
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

                var phIdx = 0
                while phIdx < mora.phonemes.count {
                    let token = mora.phonemes[phIdx]
                    var scaled = floatDurationFrames(category: token.category, symbol: token.symbol, speed: safeSpeedFactor)
                    // 文末または読点ポーズ直前のモーラを自然に約1.25倍伸張（Phrase-Final Lengthening）
                    if shouldLengthen {
                        switch token.category {
                        case .vowel, .nasalSyllable, .prolonged:
                            scaled *= 1.25
                        default:
                            break
                        }
                    }
                    rawFloatDurations.append(scaled)
                    phIdx += 1
                }
                mIdx += 1
            }
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                rawFloatDurations.append(Float(phrase.pauseDurationFrames))
            }
            pIdx += 1
        }

        // 4. 累積和量子化による整数フレーム確定 (累積丸めドリフトゼロ)
        let quantizedDurations = quantizeDurations(durations: rawFloatDurations)

        // 5. 量子化されたフレーム数を各音素および休止へ反映
        var qIdx = 0
        pIdx = 0
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

        // 6. フラットな音素列と Duration 列の抽出
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

        // 7. F0 輪郭パラメータの生成
        let (f0Contour, voicedFlags, totalFrames) = prosodyModel.generateF0Contour(
            phrases: phrases,
            vocabulary: vocabulary
        )

        return LinguisticFeatures(
            phoneIds: phoneIds,
            durations: durations,
            f0Contour: f0Contour,
            voicedFlags: voicedFlags,
            totalFrames: totalFrames
        )
    }
}
