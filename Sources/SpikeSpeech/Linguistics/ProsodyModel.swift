import Foundation

/// 東京方言ピッチアクセント則および藤崎モデル風 F0 連続ピッチ輪郭生成器
///
/// SNN音響モデルにおいて声帯振動の基本周波数を滑らかな直流電流注入として
/// ニューロン膜電位に供給することで、機械的な音程段差を排除し自然な抑揚を実現する。
public final class ProsodyModel: Sendable {
    public let baseF0: Float       // 基準ピッチ周波数 (女性声の標準 220.0 Hz)
    public let phraseAmp: Float    // フレーズ成分立ち上がり振幅 (0.18)
    public let phraseDecay: Float  // フレーズ成分減衰率 (2.0 / sec)
    public let accentAmp: Float    // アクセント成分振幅 (0.20)
    public let accentAttack: Float // アクセント立ち上がり係数 (15.0 / sec)
    public let accentDecay: Float  // アクセント立ち下がり減衰係数 (16.0 / sec)

    public init(
        baseF0: Float = 220.0,
        phraseAmp: Float = 0.18,
        phraseDecay: Float = 2.0,
        accentAmp: Float = 0.20,
        accentAttack: Float = 15.0,
        accentDecay: Float = 16.0
    ) {
        self.baseF0 = baseF0
        self.phraseAmp = phraseAmp
        self.phraseDecay = phraseDecay
        self.accentAmp = accentAmp
        self.accentAttack = accentAttack
        self.accentDecay = accentDecay
    }

    /// 東京方言アクセント規則に従って各モーラの High/Low トーンを算出する
    ///
    /// 東京方言の原則に基づき、頭高型、平板型、中高型、尾高型を決定論的に割り振る。
    public func computeMoraTones(moraCount: Int, accentKernel: Int) -> [AccentTone] {
        if moraCount <= 0 {
            return []
        }
        var tones = [AccentTone](repeating: .low, count: moraCount)

        if moraCount == 1 {
            switch accentKernel {
            case 1:
                tones[0] = .high
            default:
                tones[0] = .low
            }
            return tones
        }

        switch accentKernel {
        case 1:
            // 頭高型: 1拍目のみ High, 2拍目以降は Low
            tones[0] = .high
        case 0:
            // 平板型: 1拍目は Low, 2拍目以降はすべて High
            var i = 1
            while i < moraCount {
                tones[i] = .high
                i += 1
            }
        default:
            // 中高型・尾高型: 1拍目は Low, 2拍目から核まで High, 核の後は Low
            var i = 1
            while i < moraCount {
                if i < accentKernel {
                    tones[i] = .high
                }
                i += 1
            }
        }

        return tones
    }

    /// 形態素系列からアクセント句系列を構築する
    ///
    /// 自立語に付属語が連なることで発話句を形成し、
    /// 句境界で呼吸圧成分のリセットを行う。
    public func buildAccentPhrases(morphemes: [Morpheme], vocabulary: PhonemeVocabulary) -> [AccentPhrase] {
        if morphemes.isEmpty {
            return []
        }

        var phrases: [AccentPhrase] = []
        var currentMoras: [MoraToken] = []
        var currentKernel: Int16 = 0

        var mIdx = 0
        while mIdx < morphemes.count {
            let m = morphemes[mIdx]

            // 句読点・記号の場合は現在の句を完了し、ポーズ付きで閉じる
            if m.pos == .symbol {
                if currentMoras.isEmpty != true {
                    // 現在の句にトーンを適用
                    let tones = computeMoraTones(moraCount: currentMoras.count, accentKernel: Int(currentKernel))
                    var k = 0
                    while k < currentMoras.count {
                        currentMoras[k].tone = tones[k]
                        k += 1
                    }
                    var pauseDur = 30
                    if m.surface == "、" {
                        pauseDur = 15
                    }
                    var isQuestion = false
                    if m.surface == "？" || m.surface == "?" {
                        isQuestion = true
                    }
                    phrases.append(AccentPhrase(moras: currentMoras, pauseAfter: true, pauseDurationFrames: pauseDur, isQuestion: isQuestion))
                    currentMoras = []
                    currentKernel = 0
                }
                mIdx += 1
                continue
            }

            // 新たな自立語（名詞、動詞、形容詞、接続詞、感動詞）で句を分割
            var startsNewPhrase = false
            switch m.pos {
            case .noun, .verb, .adjective, .conjunction, .interjection:
                if currentMoras.isEmpty != true {
                    startsNewPhrase = true
                }
            default:
                break
            }

            if startsNewPhrase {
                let tones = computeMoraTones(moraCount: currentMoras.count, accentKernel: Int(currentKernel))
                var k = 0
                while k < currentMoras.count {
                    currentMoras[k].tone = tones[k]
                    k += 1
                }
                phrases.append(AccentPhrase(moras: currentMoras, pauseAfter: false, pauseDurationFrames: 0, isQuestion: false))
                currentMoras = []
                currentKernel = 0
            }

            let wordMoras = vocabulary.kanaToMoras(m.reading)
            if currentMoras.isEmpty {
                currentKernel = m.accentKernel
            }
            currentMoras.append(contentsOf: wordMoras)

            mIdx += 1
        }

        if currentMoras.isEmpty != true {
            let tones = computeMoraTones(moraCount: currentMoras.count, accentKernel: Int(currentKernel))
            var k = 0
            while k < currentMoras.count {
                currentMoras[k].tone = tones[k]
                k += 1
            }
            var isQuestion = false
            if let lastMora = currentMoras.last {
                if lastMora.text == "か" {
                    isQuestion = true
                }
            }
            phrases.append(AccentPhrase(moras: currentMoras, pauseAfter: false, pauseDurationFrames: 0, isQuestion: isQuestion))
        }

        return phrases
    }

    /// 各フレームの目標 F0 周波数 [Hz] および有声フラグ系列を生成する
    ///
    /// 無声子音、促音、ポーズ区間では物理的に声帯振動が存在しないため、
    /// 目標周波数および有声フラグをゼロにしてSNNデコーダーに正しく伝達する。
    /// 各フレームの目標 F0 周波数 [Hz] および有声フラグ系列を生成する
    ///
    /// 無声子音、促音、ポーズ区間では物理的に声帯振動が存在しないため、
    /// 目標周波数および有声フラグをゼロにしてSNNデコーダーに正しく伝達する。
    public func generateF0Contour(
        phrases: [AccentPhrase],
        vocabulary: PhonemeVocabulary,
        seed: UInt64 = 0x1234_5678_9abc_def0,
        applyFluctuation: Bool = true
    ) -> (f0Contour: [Float], voicedFlags: [Float], totalFrames: Int) {
        var bioFluctuation = BiologicalFluctuation(seed: seed)
        return generateF0Contour(
            phrases: phrases,
            vocabulary: vocabulary,
            fluctuation: &bioFluctuation,
            applyFluctuation: applyFluctuation
        )
    }

    /// 各フレームの目標 F0 周波数 [Hz] および有声フラグ系列を生成する（生体ゆらぎインスタンス共有版）
    /// なぜ生体ゆらぎインスタンスを共有するか:
    /// テンポゆらぎ、ピッチジッター、シマーが同一の生体ゆらぎ（1/f ピンクノイズ）系列を
    /// 継続して消費することで、呼気圧・声帯振動の生理学的連動性を維持するため。
    public func generateF0Contour(
        phrases: [AccentPhrase],
        vocabulary: PhonemeVocabulary,
        fluctuation: inout BiologicalFluctuation,
        applyFluctuation: Bool = true
    ) -> (f0Contour: [Float], voicedFlags: [Float], totalFrames: Int) {
        var f0List: [Float] = []
        var voicedList: [Float] = []

        var pIdx = 0
        var phraseScale: Float = 1.0
        var globalFrame = 0
        var sentenceFrame = 0
        var voicedStreak = 0
        var prevPhonemeSymbol = ""

        while pIdx < phrases.count {
            let phrase = phrases[pIdx]
            let moras = phrase.moras
            let effPhraseAmp = phraseAmp * phraseScale

            var frameInPhrase = 0
            var highStartFrame: Int? = nil
            var fallStartFrame: Int? = nil
            var lastAccentAtFall: Float = 0.0
            var wasHigh = false

            var mIdx = 0
            while mIdx < moras.count {
                let mora = moras[mIdx]
                let isHigh = (mora.tone == .high)
                let isLastMoraInPhrase = (mIdx + 1) == moras.count

                var phIdx = 0
                while phIdx < mora.phonemes.count {
                    let phoneme = mora.phonemes[phIdx]
                    let duration = phoneme.durationFrames
                    let isVoicedPhoneme = vocabulary.isVoiced(symbol: phoneme.symbol)
                    let isLastPhonemeInMora = (phIdx + 1) == mora.phonemes.count

                    var f = 0
                    while f < duration {
                        let t = Float(frameInPhrase) * 0.010 // 1フレーム = 10ms
                        let tSentence = Float(sentenceFrame) * 0.010

                        // 1. フレーズ成分: 句頭でインパルス応答的に立ち上がり、指数関数的減衰
                        let phraseComp = effPhraseAmp * expf(-phraseDecay * t)

                        // 2. アクセント成分: High トーン区間に立ち上がりステップ応答、Low 遷移時に指数減衰平滑化
                        var accentComp: Float = 0.0
                        if isHigh {
                            if wasHigh != true {
                                highStartFrame = frameInPhrase
                                wasHigh = true
                            }
                            let startFrame: Int
                            switch highStartFrame {
                            case .some(let sf):
                                startFrame = sf
                            case .none:
                                startFrame = frameInPhrase
                            }
                            let tRel = Float(frameInPhrase - startFrame) * 0.010
                            accentComp = accentAmp * (1.0 - expf(-accentAttack * tRel))
                            lastAccentAtFall = accentComp
                        } else {
                            if wasHigh {
                                fallStartFrame = frameInPhrase
                                wasHigh = false
                            }
                            switch fallStartFrame {
                            case .some(let fallFrame):
                                let tFall = Float(frameInPhrase - fallFrame) * 0.010
                                accentComp = lastAccentAtFall * expf(-accentDecay * tFall)
                            case .none:
                                accentComp = 0.0
                            }
                        }

                        // 疑問文における文末上昇調（Interrogative Rising Tone）の付与
                        if phrase.isQuestion && isLastMoraInPhrase && isLastPhonemeInMora {
                            let relPos = Float(f) / Float(max(1, duration))
                            accentComp += 0.20 * (relPos * relPos)
                        }

                        // 3. 生理学的呼気圧低下モデル (Declination / 文単位の自然降下線)
                        // 人間の発話では文頭から文末に向けて呼気圧が徐々に減衰するため、緩やかな下降傾斜を付加する。
                        // 文境界（句点・長ポーズ）で呼気圧がリセットされる生理学的機構を再現する。
                        let declination = expf(-0.05 * min(4.0, tSentence))

                        // 4. 調音音声学に基づくマイクロプロソディ (Microprosody / 子音牽引ピッチ効果)
                        // 先行する無声子音の気圧解放や有声子音の負荷により、母音立ち上がりの F0 が過渡的に変動する
                        var microprosodyScale: Float = 1.0
                        if isVoicedPhoneme && f < 3 {
                            let decay = Float(3 - f) / 3.0
                            switch prevPhonemeSymbol {
                            case "k", "ky", "t", "ch", "ts", "p", "py", "s", "sh", "h", "hy":
                                // 無声破裂・摩擦音後: 声門下圧の上昇によりピッチが一時的に跳ね上がる (+3.0%)
                                microprosodyScale = 1.0 + (0.030 * decay)
                            case "g", "gy", "d", "b", "by", "z", "j", "m", "my", "n", "ny", "r", "ry":
                                // 有声子音・鼻音後: 声帯への音響負荷によりピッチが低域から立ち上がる (-2.0%)
                                microprosodyScale = 1.0 - (0.020 * decay)
                            default:
                                break
                            }
                        }

                        // 5. 母音固有基本周波数 (Intrinsic Vowel Pitch / IF0)
                        // 音響音声学における舌根挙上と喉頭牽引の相互作用により、狭母音 (i, u) は広母音よりわずかにピッチが高くなる
                        var intrinsicScale: Float = 1.0
                        switch phoneme.symbol {
                        case "i", "u":
                            intrinsicScale = 1.035
                        default:
                            break
                        }

                        if isVoicedPhoneme {
                            voicedStreak += 1
                            // 6. F0 周波数の合成と 1/f 生体ピッチゆらぎ (Jitter)
                            // なぜ有声フレーム内のみで Jitter を算出するか:
                            // 無声フレームで疑似乱数ジェネレータを進めると、無声子音の長さに応じて
                            // 後続母音のピッチ位相が不自然にずれる現象を防止するため。
                            let rawF0 = baseF0 * expf(phraseComp + accentComp) * declination * intrinsicScale * microprosodyScale
                            var f0 = rawF0
                            if applyFluctuation {
                                f0 = fluctuation.computePitchJitter(baseF0: rawF0)
                            }

                            // 有声化開始アタック (Onset Glottal Attack)
                            // 無声から有声へ切り替わる先頭 2 フレームで声帯振動がわずかに低域から立ち上がる生理学的アタック
                            if voicedStreak <= 2 {
                                let onsetScale: Float = 0.94 + (0.03 * Float(voicedStreak))
                                f0 = f0 * onsetScale
                            }

                            // 60Hz〜480Hz の適正有声帯域内に確実にクランプ
                            if f0 < 60.0 {
                                f0 = 60.0
                            }
                            if 480.0 < f0 {
                                f0 = 480.0
                            }

                            f0List.append(f0)
                            voicedList.append(1.0)
                        } else {
                            voicedStreak = 0
                            f0List.append(0.0)
                            voicedList.append(0.0)
                        }

                        frameInPhrase += 1
                        globalFrame += 1
                        sentenceFrame += 1
                        f += 1
                    }
                    prevPhonemeSymbol = phoneme.symbol
                    phIdx += 1
                }
                mIdx += 1
            }

            // 次のアクセント句へのダウンステップ（Catathesis）の適用および文境界での呼吸圧リセット
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                phraseScale = 1.0
                sentenceFrame = 0 // 文境界ポーズで呼気圧・微小ゆらぎ時間をリセット
            } else {
                let nextScale = phraseScale * 0.85
                if 0.68 <= nextScale {
                    phraseScale = nextScale
                } else {
                    phraseScale = 0.68
                }
            }

            // 後続ポーズ（読点・句点）フレームの追加
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                var pf = 0
                while pf < phrase.pauseDurationFrames {
                    f0List.append(0.0)
                    voicedList.append(0.0)
                    pf += 1
                }
            }

            pIdx += 1
        }

        let total = f0List.count
        return (f0List, voicedList, total)
    }
}
