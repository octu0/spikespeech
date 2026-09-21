import Foundation

/// 実音声の音響特徴（Mel、有声度、スペクトル急変、周波数帯域比率）に基づく動的計画法 Forced Aligner
///
/// 音素音響カテゴリと音声フレームの適合度対数尤度を算出し、大局的動的計画法（DP）によって
/// 音素の開始・終了境界を物理的に決定する。
public final class AcousticForcedAligner: Sendable {
    public init() {}

    /// 単一発話に対する音響特徴の抽出構造体
    public struct FrameAcousticFeatures: Sendable {
        public let power: Float
        public let voiced: Float
        public let spectralFlux: Float
        public let highFreqRatio: Float
        public let lowFreqRatio: Float
    }

    /// 発話区間の全フレームに対する音響特徴を抽出する
    public func extractFeatures(
        pcm: [Float],
        hopSize: Int,
        startFrame: Int,
        frameCount: Int,
        pitchTracker: PitchTracker,
        melExtractor: MelSpectrogramExtractor
    ) -> [FrameAcousticFeatures] {
        if frameCount <= 0 {
            return []
        }

        let pcmLength = pcm.count
        let pitchResult = pitchTracker.track(pcm: pcm)
        let logMel = melExtractor.extractLogMel(pcm: pcm)

        var features = [FrameAcousticFeatures](repeating: FrameAcousticFeatures(power: 0.0, voiced: 0.0, spectralFlux: 0.0, highFreqRatio: 0.0, lowFreqRatio: 0.0), count: frameCount)

        var t = 0
        while t < frameCount {
            let absFrame = startFrame + t
            let sampleStart = absFrame * hopSize
            let sampleEnd = min(pcmLength, sampleStart + hopSize)

            var power: Float = 1.0e-9
            if sampleStart < sampleEnd {
                var sumSq: Float = 0.0
                var s = sampleStart
                while s < sampleEnd {
                    let v = pcm[s]
                    sumSq += v * v
                    s += 1
                }
                power = sumSq / Float(sampleEnd - sampleStart)
            }

            var voiced: Float = 0.0
            if absFrame < pitchResult.frameCount {
                voiced = pitchResult.voiced[absFrame]
            }

            var flux: Float = 0.0
            var highRatio: Float = 0.0
            var lowRatio: Float = 0.0

            if absFrame < logMel.count {
                let curMel = logMel[absFrame]
                if 0 < absFrame && absFrame - 1 < logMel.count {
                    let prevMel = logMel[absFrame - 1]
                    var diffSqSum: Float = 0.0
                    var c = 0
                    while c < AudioConfig.melChannels {
                        let diff = curMel[c] - prevMel[c]
                        diffSqSum += diff * diff
                        c += 1
                    }
                    flux = sqrtf(diffSqSum)
                }

                // 線形パワー近似による帯域比率算出
                var lowSum: Float = 0.0
                var highSum: Float = 0.0
                var totalSum: Float = 0.0
                var ch = 0
                while ch < AudioConfig.melChannels {
                    let linP = expf(curMel[ch])
                    totalSum += linP
                    if ch < 16 {
                        lowSum += linP
                    }
                    if 32 <= ch {
                        highSum += linP
                    }
                    ch += 1
                }
                if 1.0e-9 < totalSum {
                    lowRatio = lowSum / totalSum
                    highRatio = highSum / totalSum
                }
            }

            features[t] = FrameAcousticFeatures(
                power: power,
                voiced: voiced,
                spectralFlux: flux,
                highFreqRatio: highRatio,
                lowFreqRatio: lowRatio
            )
            t += 1
        }

        return features
    }

    /// 音素トークンとフレーム音響特徴のマッチングスコア（適合度対数尤度）を算出する
    public func acousticScore(phoneme: PhonemeToken, feature: FrameAcousticFeatures) -> Float {
        var score: Float = 0.0

        switch phoneme.category {
        case .vowel, .prolonged:
            // 母音・長音: 高有声度、高エネルギー、低周波集中、低 Flux
            score += feature.voiced * 2.5
            if 0.005 < feature.power {
                score += 1.5
            }
            score += feature.lowFreqRatio * 1.5
            score -= feature.highFreqRatio * 1.5
            score -= min(2.0, feature.spectralFlux * 0.2)

        case .consonant:
            switch phoneme.symbol {
            case "s", "sh", "h", "z", "j", "ch":
                // 摩擦音・破擦音: 高周波比率大、無声または弱有声
                score += feature.highFreqRatio * 3.5
                if feature.voiced < 0.5 {
                    score += 1.0
                }
                if 0.001 < feature.power {
                    score += 0.8
                }
            case "k", "t", "p", "g", "d", "b":
                // 破裂音: 閉鎖（低パワー）またはバースト（高 Flux）
                if feature.power < 0.003 {
                    score += 2.0
                }
                score += min(2.5, feature.spectralFlux * 0.4)
                if feature.voiced < 0.6 {
                    score += 0.5
                }
            default:
                // その他子音 (m, n, r, w, y): 有声低周波
                score += feature.voiced * 1.5
                score += feature.lowFreqRatio * 1.5
            }

        case .geminate:
            // 促音 (っ): 極低パワー閉鎖、無声
            if feature.power < 0.001 {
                score += 3.5
            } else {
                if feature.power < 0.004 {
                    score += 1.5
                } else {
                    score -= 3.0
                }
            }
            if feature.voiced < 0.2 {
                score += 1.5
            } else {
                score -= 2.0
            }

        case .nasalSyllable:
            // 撥音 (ん): 有声、低周波集中、高周波遮断
            score += feature.voiced * 2.0
            score += feature.lowFreqRatio * 2.5
            score -= feature.highFreqRatio * 2.5
            if 0.002 < feature.power {
                score += 1.0
            }

        case .pause:
            if feature.power < 0.001 {
                score += 4.0
            }
            if feature.voiced < 0.1 {
                score += 2.0
            }

        case .contracted:
            score += feature.voiced * 1.5
        }

        return score
    }

    /// 各音素の最小・最大許容フレーム長を定義する
    public func durationBounds(for phoneme: PhonemeToken) -> (minDur: Int, maxDur: Int) {
        switch phoneme.category {
        case .geminate:
            return (minDur: 4, maxDur: 20)
        case .nasalSyllable, .prolonged:
            return (minDur: 4, maxDur: 25)
        case .vowel:
            return (minDur: 3, maxDur: 30)
        case .consonant:
            switch phoneme.symbol {
            case "s", "sh", "h", "z", "j", "ch":
                return (minDur: 2, maxDur: 18)
            case "k", "t", "p", "g", "d", "b":
                return (minDur: 2, maxDur: 14)
            default:
                return (minDur: 2, maxDur: 16)
            }
        case .pause:
            return (minDur: 3, maxDur: 40)
        case .contracted:
            return (minDur: 2, maxDur: 12)
        }
    }

    /// 動的計画法 (DP) による大局的最適音素アライメントを実行する
    ///
    /// - Parameters:
    ///   - features: 発話区間の音響特徴系列 (長さ frameCount)
    ///   - phonemes: 発話区間の音素系列 (長さ P)
    /// - Returns: 各音素の確定フレーム数配列 (長さ P, 総和が frameCount と厳密一致)
    public func align(
        features: [FrameAcousticFeatures],
        phonemes: [PhonemeToken]
    ) -> [Int]? {
        let frameCount = features.count
        let phoneCount = phonemes.count

        if frameCount <= 0 || phoneCount <= 0 {
            return nil
        }

        // 各音素の最小・最大フレーム境界の事前計算
        var minDurs = [Int](repeating: 0, count: phoneCount)
        var maxDurs = [Int](repeating: 0, count: phoneCount)
        var sumMin = 0

        var p = 0
        while p < phoneCount {
            let (minD, maxD) = durationBounds(for: phonemes[p])
            minDurs[p] = minD
            maxDurs[p] = maxD
            sumMin += minD
            p += 1
        }

        // フレーム数が最小要求を満たさない場合は安全に緩和
        if frameCount < sumMin {
            var reduced = 0
            p = 0
            while p < phoneCount {
                minDurs[p] = 1
                reduced += 1
                p += 1
            }
            if frameCount < reduced {
                return nil
            }
        }

        // 接尾辞最小フレーム数合計
        var suffixMin = [Int](repeating: 0, count: phoneCount + 1)
        var sIdx = phoneCount - 1
        while 0 <= sIdx {
            suffixMin[sIdx] = suffixMin[sIdx + 1] + minDurs[sIdx]
            sIdx -= 1
        }

        // 音素-フレームスコア行列の事前計算
        var scoreMatrix = [[Float]](repeating: [Float](repeating: 0.0, count: frameCount), count: phoneCount)
        var pi = 0
        while pi < phoneCount {
            let ph = phonemes[pi]
            var fi = 0
            while fi < frameCount {
                scoreMatrix[pi][fi] = acousticScore(phoneme: ph, feature: features[fi])
                fi += 1
            }
            pi += 1
        }

        // スコア累積和テーブル (音素区間スコア O(1) 算出用)
        var cumScores = [[Float]](repeating: [Float](repeating: 0.0, count: frameCount + 1), count: phoneCount)
        pi = 0
        while pi < phoneCount {
            var cSum: Float = 0.0
            var fi = 0
            while fi < frameCount {
                cSum += scoreMatrix[pi][fi]
                cumScores[pi][fi + 1] = cSum
                fi += 1
            }
            pi += 1
        }

        // DP テーブル: dp[phoneIdx][endFrame]
        let negInf: Float = -1.0e18
        var dp = [[Float]](repeating: [Float](repeating: negInf, count: frameCount + 1), count: phoneCount)
        var trace = [[Int]](repeating: [Int](repeating: 0, count: frameCount + 1), count: phoneCount)

        // 初期化: 音素 0
        let min0 = minDurs[0]
        let max0 = min(maxDurs[0], frameCount - suffixMin[1])
        var f0 = min0
        while f0 <= max0 {
            let segScore = cumScores[0][f0] - cumScores[0][0]
            dp[0][f0] = segScore
            trace[0][f0] = f0
            f0 += 1
        }

        // 漸化式更新: 音素 1..<phoneCount
        var i = 1
        while i < phoneCount {
            let minDur = minDurs[i]
            let maxDur = maxDurs[i]
            let remMinAfter = suffixMin[i + 1]

            // この音素までの最小累積フレーム数
            var prefixMin = 0
            var k = 0
            while k <= i {
                prefixMin += minDurs[k]
                k += 1
            }

            let tStart = prefixMin
            let tEnd = frameCount - remMinAfter

            var t = tStart
            while t <= tEnd {
                var bestVal = negInf
                var bestDur = minDur

                // 前の音素の終了フレーム prevT = t - dur
                let durStart = minDur
                let durEnd = min(maxDur, t)

                var dur = durStart
                while dur <= durEnd {
                    let prevT = t - dur
                    let prevScore = dp[i - 1][prevT]
                    if negInf < prevScore {
                        let segScore = cumScores[i][t] - cumScores[i][prevT]
                        let totalVal = prevScore + segScore
                        if bestVal < totalVal {
                            bestVal = totalVal
                            bestDur = dur
                        }
                    }
                    dur += 1
                }

                dp[i][t] = bestVal
                trace[i][t] = bestDur
                t += 1
            }
            i += 1
        }

        // バックトラッキング
        var durations = [Int](repeating: 0, count: phoneCount)
        var currT = frameCount
        var bIdx = phoneCount - 1
        while 0 <= bIdx {
            let chosenDur = trace[bIdx][currT]
            if chosenDur <= 0 {
                // 安全フォールバック
                durations[bIdx] = max(1, minDurs[bIdx])
                currT -= durations[bIdx]
            } else {
                durations[bIdx] = chosenDur
                currT -= chosenDur
            }
            bIdx -= 1
        }

        // 総和の厳密一致検査と端数調整
        var totalAllocated = 0
        var d = 0
        while d < durations.count {
            totalAllocated += durations[d]
            d += 1
        }

        let diff = frameCount - totalAllocated
        if diff != 0 && 0 < durations.count {
            let lastI = durations.count - 1
            durations[lastI] += diff
        }

        return durations
    }
}
