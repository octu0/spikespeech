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
    public func generateF0Contour(
        phrases: [AccentPhrase],
        vocabulary: PhonemeVocabulary
    ) -> (f0Contour: [Float], voicedFlags: [Float], totalFrames: Int) {
        var f0List: [Float] = []
        var voicedList: [Float] = []

        var pIdx = 0
        var phraseScale: Float = 1.0
        var globalFrame = 0
        var sentenceFrame = 0
        var voicedStreak = 0

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

                        // 4. 自然な生体声帯微小ゆらぎ (Organic Glottal Micro-variation)
                        // 単一周波数の過剰な正弦波変調（ビブラート感・ロボット感）を排除し、
                        // 非調和な超低周波数の合成による微小な自然ゆらぎ（振幅 0.5% 未満）を付与する。
                        let jitter1 = 0.003 * sin(2.0 * Float.pi * 1.73 * tSentence)
                        let jitter2 = 0.002 * sin(2.0 * Float.pi * 3.11 * tSentence)
                        let jitter = 1.0 + jitter1 + jitter2

                        // 5. 母音固有基本周波数 (Intrinsic Vowel Pitch / IF0)
                        // 音響音声学における舌根挙上と喉頭牽引の相互作用により、狭母音 (i, u) は広母音よりわずかにピッチが高くなる
                        var intrinsicScale: Float = 1.0
                        switch phoneme.symbol {
                        case "i", "u":
                            intrinsicScale = 1.035
                        default:
                            break
                        }

                        // 6. F0 周波数の合成
                        var f0 = baseF0 * expf(phraseComp + accentComp) * declination * jitter * intrinsicScale

                        if isVoicedPhoneme {
                            voicedStreak += 1
                            // 有声化開始アタック (Onset Glottal Attack)
                            // 無声から有声へ切り替わる先頭 2 フレームで声帯振動がわずかに低域から立ち上がる生理学的アタック
                            if voicedStreak <= 2 {
                                let onsetScale: Float = 0.94 + (0.03 * Float(voicedStreak))
                                f0 = f0 * onsetScale
                            }

                            // 50Hz〜500Hz の適正有声帯域内に確実にクランプ
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
