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
        phraseDecay: Float = 0.70,
        accentAmp: Float = 0.32,
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
                    // 現在の句にトーンとアクセント核を適用
                    let tones = computeMoraTones(moraCount: currentMoras.count, accentKernel: Int(currentKernel))
                    var k = 0
                    while k < currentMoras.count {
                        currentMoras[k].tone = tones[k]
                        if 0 < currentKernel && (k + 1) == Int(currentKernel) {
                            currentMoras[k].isAccentKernel = true
                        } else {
                            currentMoras[k].isAccentKernel = false
                        }
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
                    if 0 < currentKernel && (k + 1) == Int(currentKernel) {
                        currentMoras[k].isAccentKernel = true
                    } else {
                        currentMoras[k].isAccentKernel = false
                    }
                    k += 1
                }
                // 自立語境界では句分割を行いつつ、ポーズ時間は 0 に設定する。
                // 理由: 学習時データにおいて文中の自立語間ポーズは存在せず、
                // ポーズを挿入すると SNN が未知のフレーム・Mel 遷移を出力して発音がブザー化するため。
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
                if 0 < currentKernel && (k + 1) == Int(currentKernel) {
                    currentMoras[k].isAccentKernel = true
                } else {
                    currentMoras[k].isAccentKernel = false
                }
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
    /// なぜ baseF0 を引数で受け取れるようにするか:
    /// 話者の絶対基音周波数は声質（VoiceProfile）に属し、言語処理・アクセント句の相対抑揚（句成分・アクセント成分・文降下・マイクロプロソディ・ジッター）
    /// とは直交するため、推論時に話者の絶対基音を直接注入して自然な F0 輪郭を生成できるようにするため。
    public func generateF0Contour(
        phrases: [AccentPhrase],
        vocabulary: PhonemeVocabulary,
        baseF0: Float = 220.0,
        fluctuation: inout BiologicalFluctuation,
        applyFluctuation: Bool = true
    ) -> (f0Contour: [Float], voicedFlags: [Float], totalFrames: Int) {
        let effectiveBaseF0 = baseF0

        var latentF0List: [Float] = []
        var voicedList: [Float] = []

        var pIdx = 0
        var phraseScale: Float = 1.0
        var globalFrame = 0
        var sentenceFrame = 0
        var breathGroupFrame = 0
        var voicedStreak = 0
        var prevPhonemeSymbol = ""

        // 輪状甲状筋の質量・弾性・粘性（マス・スプリング・ダンパー系）を模した2次臨界減衰系状態変数
        // なぜ2次臨界減衰系を採用するか:
        // 従来の1次指数遅れ（折れ線応答）による急峻な階段ピッチ段差や不自然な角を完全に排除し、
        // 藤崎モデルに基づく滑らかなS字立ち上がりと自然な筋弛緩減衰（C1級連続性）を実現するため。
        // また、頭高型アクセント（第1拍がHigh）では発話開始時点ですでに声帯筋が緊張しているため、
        // 初期アクセント値を目標値に設定して第1拍の十分な高音立ち上がりを保証する。
        var accentVal: Float = 0.0
        var accentVel: Float = 0.0
        if let firstPhrase = phrases.first, let firstMora = firstPhrase.moras.first, firstMora.tone == .high {
            accentVal = accentAmp * phraseScale
        }
        let omegaAccent: Float = 22.0 // 自然角周波数 (約60msで目標アクセントへ滑らかに収束)
        let dt: Float = 0.010        // 1フレーム = 10ms

        while pIdx < phrases.count {
            let phrase = phrases[pIdx]
            let moras = phrase.moras
            let effPhraseAmp = phraseAmp * phraseScale

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
                        let tBreath = Float(breathGroupFrame) * 0.010
                        let tSentence = Float(sentenceFrame) * 0.010

                        // 1. 呼気段落（Breath Group）に基づくフレーズ成分
                        // なぜ単語ごとではなく呼気段落（文頭・ポーズ境界）単位で減衰させるか:
                        // ポーズのない同一文内の単語境界でフレーズ成分を0リセットすると、単語ごとにピッチが+20%跳ね上がる
                        // ロボット特有の階段状イントネーションになるため、呼気圧全体を文全体で有機的に減衰させる。
                        let phraseComp = effPhraseAmp * expf(-phraseDecay * tBreath)

                        // 2. 生理学的2次臨界減衰系によるアクセント成分の追従
                        // なぜ2次臨界減衰フィルタを用いるか:
                        // 輪状甲状筋の生体慣性を再現し、目標ピッチへの加速と減速を連続的（S字カーブ）に行うため。
                        var targetAccent: Float = 0.0
                        if isHigh {
                            targetAccent = accentAmp * phraseScale
                        }
                        if phrase.isQuestion && isLastMoraInPhrase && isLastPhonemeInMora {
                            let relPos = Float(f) / Float(max(1, duration))
                            targetAccent += 0.22 * (relPos * relPos)
                        }

                        let accel = (omegaAccent * omegaAccent * (targetAccent - accentVal)) - (2.0 * omegaAccent * accentVel)
                        accentVel += accel * dt
                        accentVal += accentVel * dt

                        // 3. 生理学的呼気圧低下モデル (Declination)
                        let declination = expf(-0.045 * min(4.0, tSentence))

                        // 4. 調音音声学に基づくマイクロプロソディ (Microprosody)
                        // なぜ直線減少ではなく時定数15msの滑らかな過渡応答にするか:
                        // 先行子音解放時の気圧変化が母音開始部に連続的に合流する生理学的過渡現象を忠実に再現するため。
                        var microprosodyScale: Float = 1.0
                        if isVoicedPhoneme && f < 3 {
                            let decay = expf(-Float(f) * 0.6)
                            switch prevPhonemeSymbol {
                            case "k", "ky", "t", "ch", "ts", "p", "py", "s", "sh", "h", "hy":
                                microprosodyScale = 1.0 + (0.025 * decay)
                            case "g", "gy", "d", "b", "by", "z", "j", "m", "my", "n", "ny", "r", "ry":
                                microprosodyScale = 1.0 - (0.015 * decay)
                            default:
                                break
                            }
                        }

                        // 5. 母音固有基本周波数 (Intrinsic Vowel Pitch / IF0)
                        var intrinsicScale: Float = 1.0
                        switch phoneme.symbol {
                        case "i", "u":
                            intrinsicScale = 1.030
                        default:
                            break
                        }

                        if isVoicedPhoneme {
                            voicedStreak += 1
                        } else {
                            voicedStreak = 0
                        }

                        let rawF0 = effectiveBaseF0 * expf(phraseComp + accentVal) * declination * intrinsicScale * microprosodyScale
                        var f0 = rawF0
                        if applyFluctuation {
                            f0 = fluctuation.computePitchJitter(baseF0: rawF0)
                        }

                        let minF0 = max(45.0, effectiveBaseF0 * 0.45)
                        let maxF0 = min(600.0, effectiveBaseF0 * 2.20)
                        if f0 < minF0 {
                            f0 = minF0
                        }
                        if maxF0 < f0 {
                            f0 = maxF0
                        }

                        latentF0List.append(f0)
                        if isVoicedPhoneme {
                            voicedList.append(1.0)
                        } else {
                            voicedList.append(0.0)
                        }

                        breathGroupFrame += 1
                        globalFrame += 1
                        sentenceFrame += 1
                        f += 1
                    }
                    prevPhonemeSymbol = phoneme.symbol
                    phIdx += 1
                }
                mIdx += 1
            }

            // 次のアクセント句へのダウンステップ（Catathesis）および文境界・読点での呼気圧リセット
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                phraseScale = 1.0
                sentenceFrame = 0
                breathGroupFrame = 0 // ポーズ境界で呼気圧・フレーズ成分を新規立ち上げ
                accentVal = 0.0
                accentVel = 0.0
                let nextPIdx = pIdx + 1
                if nextPIdx < phrases.count {
                    let nextPhrase = phrases[nextPIdx]
                    if let nextMora = nextPhrase.moras.first, nextMora.tone == .high {
                        accentVal = accentAmp * phraseScale
                    }
                }
            } else {
                let nextScale = phraseScale * 0.94
                if 0.70 <= nextScale {
                    phraseScale = nextScale
                } else {
                    phraseScale = 0.70
                }
            }

            // 後続ポーズ（読点・句点）フレームの追加
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                var pf = 0
                while pf < phrase.pauseDurationFrames {
                    latentF0List.append(effectiveBaseF0)
                    voicedList.append(0.0)
                    pf += 1
                }
            }

            pIdx += 1
        }

        // 6. 連続潜因ピッチ（Latent F0）に対する 5 点ガウシアン加重平滑化
        // なぜ有声/無声の切断前に全体平滑化を行うか:
        // 無声子音を挟む前後でピッチ目標が断絶するのを防ぎ、声帯制御筋の連続的緊張変化（C1級連続性）を保証するため。
        let totalCount = latentF0List.count
        var smoothedLatent = latentF0List
        if 4 < totalCount {
            var i = 2
            let endIdx = totalCount - 2
            while i < endIdx {
                smoothedLatent[i] = (0.06 * latentF0List[i - 2]) +
                                    (0.24 * latentF0List[i - 1]) +
                                    (0.40 * latentF0List[i]) +
                                    (0.24 * latentF0List[i + 1]) +
                                    (0.06 * latentF0List[i + 2])
                i += 1
            }
        }

        // 7. 有声フラグに基づく F0 マスキング（無声区間は厳密に 0.0）
        var f0List = [Float](repeating: 0.0, count: totalCount)
        var i = 0
        while i < totalCount {
            if 0.5 <= voicedList[i] {
                f0List[i] = smoothedLatent[i]
            } else {
                f0List[i] = 0.0
            }
            i += 1
        }

        return (f0List, voicedList, totalCount)
    }
}
