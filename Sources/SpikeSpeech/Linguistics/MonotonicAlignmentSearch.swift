import Foundation

/// 教師 Mel スペクトログラムと音素系列の単調動的計画法アライメント (Monotonic Alignment Search: MAS)
///
/// 手書きフォルマント表（makeMel等）を一切持たず、教師 WAV から抽出された実測対数 Mel スペクトログラムの
/// 音素別平均プロトタイプと各フレームの対数尤度最大化により、大局的単調パスを探索して音素境界を推定する。
public final class MonotonicAlignmentSearch: Sendable {

    /// 音素別 64 チャネル対数 Mel 実測音響プロトタイプ
    public struct PhonemeMelPrototype: Sendable {
        /// 64 チャネルの対数 Mel 平均ベクトル（教師 WAV 実測平均）
        public var meanMel: [Float]
        /// 有声度実測平均 (0.0: 無声, 1.0: 有声)
        public var voiced: Float
        /// 集計された観測フレーム数
        public var frameCount: Float

        public init(meanMel: [Float], voiced: Float, frameCount: Float = 1.0) {
            self.meanMel = meanMel
            self.voiced = voiced
            self.frameCount = frameCount
        }
    }

    public let prototypes: [Int: PhonemeMelPrototype]

    public init(prototypes: [Int: PhonemeMelPrototype] = [:]) {
        self.prototypes = prototypes
    }

    /// 各音素の標準的な相対継続時間重み（仮分割用）
    public static func priorWeight(for phoneme: PhonemeToken) -> Float {
        switch phoneme.category {
        case .vowel:
            return 8.0
        case .consonant:
            switch phoneme.symbol {
            case "s", "sh", "h", "z", "j", "ch", "ts":
                return 5.0
            case "k", "t", "p", "g", "d", "b":
                return 4.0
            default:
                return 4.5
            }
        case .nasalSyllable:
            return 10.0
        case .geminate:
            return 8.0
        case .prolonged:
            return 9.0
        case .pause:
            return 12.0
        case .contracted:
            return 4.0
        }
    }

    /// 発話区間のフレーム総数をモーラ等時性と音素比率により初期仮分割する
    ///
    /// なぜ手書き固定表ではなく仮分割から開始するか:
    /// 音素ラベルの初期推定区間を教師 WAV の全体長から比例配分で設定し、
    /// その区間内の実測教師 Mel を平均することで、人手バイアスゼロの実音声音素プロトタイプを自律生成するため。
    public static func initialBootstrapDurations(
        totalFrames: Int,
        phonemes: [PhonemeToken]
    ) -> [Int] {
        let n = phonemes.count
        if totalFrames <= 0 || n <= 0 {
            return []
        }
        if totalFrames < n {
            return [Int](repeating: 1, count: n)
        }

        var weights = [Float](repeating: 1.0, count: n)
        var sumWeight: Float = 0.0
        var i = 0
        while i < n {
            let w = priorWeight(for: phonemes[i])
            weights[i] = w
            sumWeight += w
            i += 1
        }
        if sumWeight <= 0.001 {
            sumWeight = Float(n)
        }

        var durations = [Int](repeating: 1, count: n)
        var cumulativeFloat: Float = 0.0
        var cumulativeInt = 0

        var idx = 0
        while idx < n {
            let ratio = weights[idx] / sumWeight
            let expectedF = Float(totalFrames) * ratio
            cumulativeFloat += expectedF

            let targetInt: Int
            if idx == n - 1 {
                targetInt = totalFrames
            } else {
                targetInt = Int(roundf(cumulativeFloat))
            }

            var dur = targetInt - cumulativeInt
            if dur < 1 {
                dur = 1
            }
            durations[idx] = dur
            cumulativeInt += dur
            idx += 1
        }

        // 丸めによる余剰または不足の補正
        var curSum = 0
        var cIdx = 0
        while cIdx < n {
            curSum += durations[cIdx]
            cIdx += 1
        }
        if curSum < totalFrames {
            durations[n - 1] += (totalFrames - curSum)
        } else {
            var diff = curSum - totalFrames
            var r = n - 1
            while 0 < diff && 0 <= r {
                if 1 < durations[r] {
                    let canReduce = durations[r] - 1
                    let reduce = min(diff, canReduce)
                    durations[r] -= reduce
                    diff -= reduce
                }
                r -= 1
            }
        }

        return durations
    }

    /// 教師 Mel 系列と音素列のアライメント結果から、音素 ID ごとの実測 Mel プロトタイプを集計・更新する
    public static func accumulatePrototypes(
        utterances: [(mel: [[Float]], voiced: [Float], phonemes: [PhonemeToken], durations: [Int])],
        existing: [Int: PhonemeMelPrototype] = [:]
    ) -> [Int: PhonemeMelPrototype] {
        var melSums: [Int: [Float]] = [:]
        var voicedSums: [Int: Float] = [:]
        var counts: [Int: Float] = [:]

        var uIdx = 0
        while uIdx < utterances.count {
            let utt = utterances[uIdx]
            let mel = utt.mel
            let voiced = utt.voiced
            let phones = utt.phonemes
            let durs = utt.durations

            var curFrame = 0
            var p = 0
            while p < phones.count && p < durs.count {
                let pid = phones[p].id
                let dur = durs[p]
                let endF = min(mel.count, curFrame + dur)

                var f = curFrame
                while f < endF {
                    let mFrame = mel[f]
                    let vVal: Float
                    if f < voiced.count {
                        vVal = voiced[f]
                    } else {
                        vVal = 0.0
                    }

                    var curMelSum = melSums[pid] ?? [Float](repeating: 0.0, count: AudioConfig.melChannels)
                    var c = 0
                    while c < AudioConfig.melChannels && c < mFrame.count {
                        curMelSum[c] += mFrame[c]
                        c += 1
                    }
                    melSums[pid] = curMelSum
                    voicedSums[pid] = (voicedSums[pid] ?? 0.0) + vVal
                    counts[pid] = (counts[pid] ?? 0.0) + 1.0

                    f += 1
                }
                curFrame += dur
                p += 1
            }
            uIdx += 1
        }

        var updated: [Int: PhonemeMelPrototype] = existing
        for (pid, cnt) in counts {
            if 0.0 < cnt {
                let mSum = melSums[pid] ?? [Float](repeating: -6.0, count: AudioConfig.melChannels)
                let vSum = voicedSums[pid] ?? 0.0

                var meanM = [Float](repeating: 0.0, count: AudioConfig.melChannels)
                var c = 0
                while c < AudioConfig.melChannels {
                    meanM[c] = mSum[c] / cnt
                    c += 1
                }
                let meanV = vSum / cnt
                updated[pid] = PhonemeMelPrototype(meanMel: meanM, voiced: meanV, frameCount: cnt)
            }
        }

        return updated
    }

    /// 音素トークンとフレーム対数 Mel の適合度対数尤度を算出する
    public func frameLogLikelihood(
        phoneId: Int,
        frameMel: [Float],
        frameVoiced: Float,
        globalMeanMel: [Float]? = nil
    ) -> Float {
        let meanM: [Float]
        let vTarget: Float

        switch prototypes[phoneId] {
        case .some(let p):
            meanM = p.meanMel
            vTarget = p.voiced
        case .none:
            switch globalMeanMel {
            case .some(let gm):
                meanM = gm
            case .none:
                meanM = [Float](repeating: -6.0, count: AudioConfig.melChannels)
            }
            vTarget = 0.5
        }

        let melDim = min(frameMel.count, meanM.count)
        var sumSqDiff: Float = 0.0

        var c = 0
        while c < melDim {
            let diff = frameMel[c] - meanM[c]
            sumSqDiff += diff * diff
            c += 1
        }

        let normDist = sumSqDiff / Float(max(1, melDim))
        var score = -normDist

        // 有声度整合性ペナルティ
        let voicedDiff = abs(frameVoiced - vTarget)
        score -= voicedDiff * 2.0

        return score
    }

    /// 教師 Mel 系列と音素トークン列の大局的単調動的計画法アライメント (MAS)
    ///
    /// - Parameters:
    ///   - mel: 発話区間の対数 Mel スペクトログラム系列 (長さ T, 各要素 64ch)
    ///   - voiced: 各フレームの有声度系列 (長さ T, 0.0〜1.0)
    ///   - phonemes: 発話区間の音素トークン系列 (長さ N)
    ///   - meanFramesPerMora: コーパスまたはモデルの平均モーラフレーム数 (~16.0)
    /// - Returns: 各音素の確定フレーム数配列 (長さ N, 総和が T と厳密一致)
    public func align(
        mel: [[Float]],
        voiced: [Float],
        phonemes: [PhonemeToken],
        meanFramesPerMora: Float = 16.0
    ) -> [Int]? {
        let tTotal = mel.count
        let nTotal = phonemes.count

        if tTotal <= 0 || nTotal <= 0 {
            return nil
        }
        if tTotal < nTotal {
            // フレーム数が音素数未満の場合は各音素に最低 1F を配分できない
            return nil
        }

        // 発話内 Mel 平均（未登録音素のフォールバック用）
        var gMel = [Float](repeating: 0.0, count: AudioConfig.melChannels)
        var tf = 0
        while tf < tTotal {
            var c = 0
            while c < AudioConfig.melChannels && c < mel[tf].count {
                gMel[c] += mel[tf][c]
                c += 1
            }
            tf += 1
        }
        var c = 0
        while c < AudioConfig.melChannels {
            gMel[c] = gMel[c] / Float(max(1, tTotal))
            c += 1
        }

        // 1. 各音素と全フレームの音響対数尤度行列を事前計算
        var logLikelihoods = [[Float]](repeating: [Float](repeating: 0.0, count: tTotal), count: nTotal)
        var i = 0
        while i < nTotal {
            let pid = phonemes[i].id
            var t = 0
            while t < tTotal {
                let v: Float
                if t < voiced.count {
                    v = voiced[t]
                } else {
                    v = 0.0
                }
                logLikelihoods[i][t] = frameLogLikelihood(
                    phoneId: pid,
                    frameMel: mel[t],
                    frameVoiced: v,
                    globalMeanMel: gMel
                )
                t += 1
            }
            i += 1
        }

        // 2. 単調動的計画法 (MAS: Monotonic Alignment Search)
        // Q[i][t]: 音素 i がフレーム t を消費した時点での最大累積スコア
        // transition[i][t]: 0: 自己遷移 (同一音素継続), 1: 前進遷移 (直前音素から切替)
        let negInf: Float = -1.0e18
        var qTable = [[Float]](repeating: [Float](repeating: negInf, count: tTotal), count: nTotal)
        var transition = [[UInt8]](repeating: [UInt8](repeating: 0, count: tTotal), count: nTotal)

        // 初期化: 音素 0
        qTable[0][0] = logLikelihoods[0][0]
        var t0 = 1
        let maxT0 = tTotal - (nTotal - 1)
        while t0 < maxT0 {
            qTable[0][t0] = qTable[0][t0 - 1] + logLikelihoods[0][t0]
            transition[0][t0] = 0
            t0 += 1
        }

        // 漸化式更新: ph = 1..<nTotal
        var ph = 1
        while ph < nTotal {
            let minT = ph
            let maxT = tTotal - (nTotal - ph)

            var curT = minT
            while curT <= maxT {
                let score = logLikelihoods[ph][curT]

                let stayScore = qTable[ph][curT - 1]
                let advanceScore = qTable[ph - 1][curT - 1]

                if stayScore < advanceScore {
                    qTable[ph][curT] = advanceScore + score
                    transition[ph][curT] = 1
                } else {
                    qTable[ph][curT] = stayScore + score
                    transition[ph][curT] = 0
                }
                curT += 1
            }
            ph += 1
        }

        // 3. バックトラックによる最適パスの確定
        var durations = [Int](repeating: 0, count: nTotal)
        var curPh = nTotal - 1
        var curFrame = tTotal - 1

        while 0 <= curFrame {
            durations[curPh] += 1
            switch transition[curPh][curFrame] {
            case 1:
                curPh -= 1
            default:
                break
            }
            curFrame -= 1
        }

        // 検証: 全音素が 1F 以上、かつ総和が tTotal と完全一致
        var sumDurs = 0
        var chk = 0
        while chk < nTotal {
            if durations[chk] < 1 {
                return nil
            }
            sumDurs += durations[chk]
            chk += 1
        }
        if sumDurs != tTotal {
            return nil
        }

        return durations
    }

    /// MAS 反復集計用発話アイテム
    public struct AlignmentInputItem: Sendable {
        public let utteranceId: String
        public let leadSilence: Int
        public let trailSilence: Int
        public let totalSpeechFrames: Int
        public let mel: [[Float]]
        public let voiced: [Float]
        public let phonemes: [PhonemeToken]

        public init(
            utteranceId: String,
            leadSilence: Int,
            trailSilence: Int,
            totalSpeechFrames: Int,
            mel: [[Float]],
            voiced: [Float],
            phonemes: [PhonemeToken]
        ) {
            self.utteranceId = utteranceId
            self.leadSilence = leadSilence
            self.trailSilence = trailSilence
            self.totalSpeechFrames = totalSpeechFrames
            self.mel = mel
            self.voiced = voiced
            self.phonemes = phonemes
        }
    }

    /// 複数発話の教師 Mel から音素プロトタイプを自律学習し、2〜3 周の反復 MAS により大局的に自己収束したアライメントを算出する
    public static func iterativelyAlign(
        items: [AlignmentInputItem],
        iterations: Int = 3,
        meanFramesPerMora: Float = 16.0
    ) -> [UtteranceAlignment] {
        if items.isEmpty {
            return []
        }

        // Round 0: 初期仮分割（meanFramesPerMora と音素比率による配分）
        print("MAS 反復集計: Round 0 (初期仮分割から第 1 世代音素プロトタイプを算出中...)")
        var currentUtterances: [(mel: [[Float]], voiced: [Float], phonemes: [PhonemeToken], durations: [Int])] = []
        var i = 0
        while i < items.count {
            let item = items[i]
            let bootstrapDurs = initialBootstrapDurations(
                totalFrames: item.totalSpeechFrames,
                phonemes: item.phonemes
            )
            if bootstrapDurs.count == item.phonemes.count {
                currentUtterances.append((mel: item.mel, voiced: item.voiced, phonemes: item.phonemes, durations: bootstrapDurs))
            }
            i += 1
        }

        var prototypes = accumulatePrototypes(utterances: currentUtterances)
        print("  第 1 世代プロトタイプ算出完了: \(prototypes.count) 音素カテゴリ集計済")

        // Round 1 ..< iterations: MAS アライメントとプロトタイプ再集計の反復
        var iter = 1
        while iter < iterations {
            print("MAS 反復集計: Round \(iter) (MAS 単調動的計画法による境界最適化中...)")
            let aligner = MonotonicAlignmentSearch(prototypes: prototypes)
            var updatedUtterances: [(mel: [[Float]], voiced: [Float], phonemes: [PhonemeToken], durations: [Int])] = []

            var u = 0
            while u < items.count {
                let item = items[u]
                if let durs = aligner.align(
                    mel: item.mel,
                    voiced: item.voiced,
                    phonemes: item.phonemes,
                    meanFramesPerMora: meanFramesPerMora
                ) {
                    updatedUtterances.append((mel: item.mel, voiced: item.voiced, phonemes: item.phonemes, durations: durs))
                }
                u += 1
            }

            prototypes = accumulatePrototypes(utterances: updatedUtterances)
            print("  Round \(iter) 完了: \(updatedUtterances.count)/\(items.count) 発話収束、\(prototypes.count) 音素プロトタイプ更新")
            iter += 1
        }

        // 最終パス: 確定した自己収束プロトタイプによる最終アライメント
        print("MAS 反復集計: 最終パス (収束プロトタイプによる確定アライメント抽出)")
        let finalAligner = MonotonicAlignmentSearch(prototypes: prototypes)
        var results: [UtteranceAlignment] = []

        var fIdx = 0
        while fIdx < items.count {
            let item = items[fIdx]
            let finalDurs: [Int]
            if let aligned = finalAligner.align(
                mel: item.mel,
                voiced: item.voiced,
                phonemes: item.phonemes,
                meanFramesPerMora: meanFramesPerMora
            ) {
                finalDurs = aligned
            } else {
                // 万一失敗した場合は仮分割をフォールバックとして採用
                finalDurs = initialBootstrapDurations(
                    totalFrames: item.totalSpeechFrames,
                    phonemes: item.phonemes
                )
            }

            var phList: [PhonemeAlignment] = []
            var p = 0
            while p < item.phonemes.count && p < finalDurs.count {
                phList.append(PhonemeAlignment(
                    symbol: item.phonemes[p].symbol,
                    phoneId: Int32(item.phonemes[p].id),
                    durationFrames: finalDurs[p]
                ))
                p += 1
            }

            results.append(UtteranceAlignment(
                utteranceId: item.utteranceId,
                leadSilenceFrames: item.leadSilence,
                trailSilenceFrames: item.trailSilence,
                totalSpeechFrames: item.totalSpeechFrames,
                phonemes: phList
            ))
            fIdx += 1
        }

        print("MAS 反復集計完了: \(results.count) 発話のアライメントが確定しました")
        return results
    }
}
