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

    /// 各音素カテゴリの物理的最小継続時間フレーム数 (1フレーム=10ms)
    ///
    /// なぜ 1 フレーム音素を禁止し母音 4F / 子音 3F を下限とするか:
    /// 人が発音する際、子音調音には最低 30ms、母音共鳴には最低 40ms の物理的時間が必要であり、
    /// 単調 DP が特定音素を 1 フレーム (10ms) に押し潰して余剰時間を母音に寄せる縮退現象を構造的に根絶するため。
    public static func minDuration(for phoneme: PhonemeToken) -> Int {
        switch phoneme.category {
        case .pause:
            return 1
        case .vowel, .prolonged:
            return 4
        default:
            return 3
        }
    }

    /// 各音素の標準最大許容継続時間フレーム数 (20 フレーム = 200ms)
    public static let standardMaxDuration: Int = 20

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

    /// 発話区間のフレーム総数を制約（母音>=4F, 子音>=3F, 最大<=20F）と音素比率により初期仮分割する
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

        var minDurs = [Int](repeating: 1, count: n)
        var sumMin = 0
        var i = 0
        while i < n {
            let md = minDuration(for: phonemes[i])
            minDurs[i] = md
            sumMin += md
            i += 1
        }

        if totalFrames < sumMin {
            return []
        }

        let maxPerPhone: Int
        let avgPerPhone = Float(totalFrames) / Float(max(1, n))
        if 10.0 < avgPerPhone {
            maxPerPhone = max(standardMaxDuration, Int(ceilf(avgPerPhone * 1.5)))
        } else {
            maxPerPhone = standardMaxDuration
        }

        let sumMax = maxPerPhone * n
        if sumMax < totalFrames {
            return []
        }

        var durations = minDurs
        var remaining = totalFrames - sumMin

        var weights = [Float](repeating: 1.0, count: n)
        var sumWeight: Float = 0.0
        var wIdx = 0
        while wIdx < n {
            let w = priorWeight(for: phonemes[wIdx])
            weights[wIdx] = w
            sumWeight += w
            wIdx += 1
        }
        if sumWeight <= 0.001 {
            sumWeight = Float(n)
        }

        while 0 < remaining {
            var activeWeight: Float = 0.0
            var c = 0
            while c < n {
                if durations[c] < maxPerPhone {
                    activeWeight += weights[c]
                }
                c += 1
            }
            if activeWeight <= 0.001 {
                break
            }

            var distributed = 0
            var dIdx = 0
            while dIdx < n && 0 < remaining {
                if durations[dIdx] < maxPerPhone {
                    let share = max(1, Int(roundf(Float(remaining) * (weights[dIdx] / activeWeight))))
                    let space = maxPerPhone - durations[dIdx]
                    let add = min(share, min(space, remaining))
                    if 0 < add {
                        durations[dIdx] += add
                        remaining -= add
                        distributed += add
                    }
                }
                dIdx += 1
            }
            if distributed == 0 {
                var rIdx = 0
                while rIdx < n && 0 < remaining {
                    if durations[rIdx] < maxPerPhone {
                        durations[rIdx] += 1
                        remaining -= 1
                    }
                    rIdx += 1
                }
            }
        }

        var curSum = 0
        var chk = 0
        while chk < n {
            curSum += durations[chk]
            chk += 1
        }
        if curSum != totalFrames {
            return []
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
    ///   - maxDurationPerPhoneme: 1音素あたりの最大フレーム数（nil時は通常20F、極小音素数時は適応上限）
    /// - Returns: 各音素の確定フレーム数配列 (長さ N, 総和が T と厳密一致)
    public func align(
        mel: [[Float]],
        voiced: [Float],
        phonemes: [PhonemeToken],
        meanFramesPerMora: Float = 16.0,
        maxDurationPerPhoneme: Int? = nil
    ) -> [Int]? {
        let tTotal = mel.count
        let nTotal = phonemes.count

        if tTotal <= 0 || nTotal <= 0 {
            return nil
        }

        let maxDurLimit: Int
        switch maxDurationPerPhoneme {
        case .some(let m):
            maxDurLimit = m
        case .none:
            let avgPerPhone = Float(tTotal) / Float(max(1, nTotal))
            if 10.0 < avgPerPhone {
                // 合成単体テスト等で極小音素数に対し大フレーム数が渡された場合
                maxDurLimit = max(Self.standardMaxDuration, Int(ceilf(avgPerPhone * 1.5)))
            } else {
                // 通常発話: 1 音素 20 フレーム超は不採用（設計書 2 項）
                maxDurLimit = Self.standardMaxDuration
            }
        }

        var minDurs = [Int](repeating: 1, count: nTotal)
        var maxDurs = [Int](repeating: maxDurLimit, count: nTotal)
        var sumMin = 0
        var sumMax = 0

        var p = 0
        while p < nTotal {
            let md = Self.minDuration(for: phonemes[p])
            minDurs[p] = md
            maxDurs[p] = maxDurLimit
            sumMin += md
            sumMax += maxDurLimit
            p += 1
        }

        if tTotal < sumMin || sumMax < tTotal {
            return nil
        }

        // 累積最小/最大継続時間配列（探索枝刈り用）
        var prefixMin = [Int](repeating: 0, count: nTotal)
        var prefixMax = [Int](repeating: 0, count: nTotal)
        var suffixMin = [Int](repeating: 0, count: nTotal + 1)

        var runMin = 0
        var runMax = 0
        var i = 0
        while i < nTotal {
            runMin += minDurs[i]
            runMax += maxDurs[i]
            prefixMin[i] = runMin
            prefixMax[i] = runMax
            i += 1
        }

        var sufRun = 0
        var sIdx = nTotal - 1
        while 0 <= sIdx {
            sufRun += minDurs[sIdx]
            suffixMin[sIdx] = sufRun
            sIdx -= 1
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

        // 1. 各音素と全フレームの対数尤度行列の計算
        var logLikelihoods = [[Float]](repeating: [Float](repeating: 0.0, count: tTotal), count: nTotal)
        var phIdx = 0
        while phIdx < nTotal {
            let pid = phonemes[phIdx].id
            var t = 0
            while t < tTotal {
                let v: Float
                if t < voiced.count {
                    v = voiced[t]
                } else {
                    v = 0.0
                }
                logLikelihoods[phIdx][t] = frameLogLikelihood(
                    phoneId: pid,
                    frameMel: mel[t],
                    frameVoiced: v,
                    globalMeanMel: gMel
                )
                t += 1
            }
            phIdx += 1
        }

        // 2. 対数尤度の累積和配列（区間スコア O(1) 算出用）
        // prefLL[ph][k] = sum(logLikelihoods[ph][0 ..< k])
        var prefLL = [[Float]](repeating: [Float](repeating: 0.0, count: tTotal + 1), count: nTotal)
        var ph = 0
        while ph < nTotal {
            var k = 0
            var accum: Float = 0.0
            while k < tTotal {
                accum += logLikelihoods[ph][k]
                prefLL[ph][k + 1] = accum
                k += 1
            }
            ph += 1
        }

        // 3. 制約付き単調動的計画法 (Constrained MAS)
        // dp[i][k]: 音素 0...i をフレーム 0..<k に割り当てたときの最大対数尤度
        // bestD[i][k]: その最大スコアを達成した音素 i のフレーム数 d
        let negInf: Float = -1.0e18
        var dp = [[Float]](repeating: [Float](repeating: negInf, count: tTotal + 1), count: nTotal)
        var bestD = [[UInt8]](repeating: [UInt8](repeating: 0, count: tTotal + 1), count: nTotal)

        // 初期化: 音素 0
        let k0Min = minDurs[0]
        let k0Max = min(maxDurs[0], tTotal - suffixMin[1])
        var k0 = k0Min
        while k0 <= k0Max {
            dp[0][k0] = prefLL[0][k0]
            bestD[0][k0] = UInt8(k0)
            k0 += 1
        }

        // 漸化式更新: ph = 1 ..< nTotal
        var curPh = 1
        while curPh < nTotal {
            let kMin = prefixMin[curPh]
            let kMax = min(prefixMax[curPh], tTotal - suffixMin[curPh + 1])

            var curK = kMin
            while curK <= kMax {
                let dMin = max(minDurs[curPh], curK - prefixMax[curPh - 1])
                let dMax = min(maxDurs[curPh], curK - prefixMin[curPh - 1])

                var bestScore = negInf
                var bestDur: UInt8 = 0

                var d = dMin
                while d <= dMax {
                    let prevK = curK - d
                    let prevScore = dp[curPh - 1][prevK]
                    if -1.0e17 < prevScore {
                        let segScore = prefLL[curPh][curK] - prefLL[curPh][prevK]
                        let totalScore = prevScore + segScore
                        if bestScore < totalScore {
                            bestScore = totalScore
                            bestDur = UInt8(d)
                        }
                    }
                    d += 1
                }

                dp[curPh][curK] = bestScore
                bestD[curPh][curK] = bestDur
                curK += 1
            }
            curPh += 1
        }

        // 4. バックトラックによる最適持続時間の確定
        if dp[nTotal - 1][tTotal] <= -1.0e17 {
            return nil
        }

        var durations = [Int](repeating: 0, count: nTotal)
        var traceK = tTotal
        var tracePh = nTotal - 1

        while 0 <= tracePh {
            let d = Int(bestD[tracePh][traceK])
            if d < minDurs[tracePh] || maxDurs[tracePh] < d {
                return nil
            }
            durations[tracePh] = d
            traceK -= d
            tracePh -= 1
        }

        if traceK != 0 {
            return nil
        }

        // 5. 厳格受入検証:
        // - 無音以外の 1 フレーム音素は禁止（母音 >= 4, 子音 >= 3）
        // - 1 音素が maxDurLimit を超えない
        // - 総和が tTotal と完全一致
        var sumDurs = 0
        var chkIdx = 0
        while chkIdx < nTotal {
            let dur = durations[chkIdx]
            let reqMin = minDurs[chkIdx]
            if dur < reqMin {
                return nil
            }
            if maxDurs[chkIdx] < dur {
                return nil
            }
            sumDurs += dur
            chkIdx += 1
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

            if finalDurs.count == item.phonemes.count {
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
            }
            fIdx += 1
        }

        print("MAS 反復集計完了: \(results.count) 発話のアライメントが確定しました")
        return results
    }
}
