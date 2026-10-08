#if canImport(MLX)
import Foundation
import MLX

/// SNN 音響モデル学習用のスペクトル回帰損失関数
///
/// 音声合成の音響モデルは離散分類ではなく連続スペクトルの多次元回帰問題であるため、
/// 周波数ビンごとの幾何学的距離を安定して最小化する。
public enum AcousticLossFunctions {

    /// L1 スペクトル再構成損失
    ///
    /// L2 損失と比較して外れ値に対する勾配が穏やかであり、スペクトルのフォルマント
    /// ピークや高周波帯域の微細構造を過度につぶさずに鮮明に再構成する。
    public static func spectralL1Loss(
        predicted: MLXArray,
        target: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let diff = abs(predicted - target)

        if let m = mask {
            let mExpanded: MLXArray
            if m.ndim < diff.ndim {
                mExpanded = expandedDimensions(m, axis: -1)
            } else {
                mExpanded = m
            }
            let maskedDiff = diff * mExpanded
            let featureDim = Float(predicted.shape[predicted.ndim - 1])
            let totalValid = sum(mExpanded) * featureDim
            return sum(maskedDiff) / (totalValid + 1e-5)
        }

        return mean(diff)
    }

    /// フレーム間差分スペクトル損失
    ///
    /// 静的スペクトルの絶対値だけでなく時間方向の動的変化をターゲットに一致させ、
    /// フレーム間の遷移における不連続性やクリックノイズを抑制する。
    public static func spectralDeltaLoss(
        predicted: MLXArray,
        target: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let pDelta = predicted[0..., 1..., 0...] - predicted[0..., ..<(-1), 0...]
        let tDelta = target[0..., 1..., 0...] - target[0..., ..<(-1), 0...]

        var deltaMask: MLXArray? = nil
        if let m = mask {
            deltaMask = m[0..., 1...]
        }

        return spectralL1Loss(predicted: pDelta, target: tDelta, mask: deltaMask)
    }

    /// 音素区間の検出
    ///
    /// 区間は ch 199 (AudioConfig.pulseChannel) が 0.5 を超えるフレームで切る。
    public static func findPhonemeIntervals(
        features: [[Float]],
        rawLen: Int
    ) -> [(start: Int, end: Int)] {
        if rawLen <= 0 || features.isEmpty {
            return []
        }
        let safeLen = min(rawLen, features.count)
        var pulseIndices: [Int] = []
        var t = 0
        while t < safeLen {
            if AudioConfig.pulseChannel < features[t].count {
                if 0.5 < features[t][AudioConfig.pulseChannel] {
                    pulseIndices.append(t)
                }
            }
            t += 1
        }

        var cutPoints: [Int] = []
        if pulseIndices.isEmpty || pulseIndices[0] != 0 {
            cutPoints.append(0)
        }
        for p in pulseIndices {
            cutPoints.append(p)
        }
        cutPoints.append(safeLen)

        var intervals: [(start: Int, end: Int)] = []
        var i = 0
        while i + 1 < cutPoints.count {
            let s = cutPoints[i]
            let e = cutPoints[i + 1]
            if s < e {
                intervals.append((start: s, end: e))
            }
            i += 1
        }
        return intervals
    }

    /// 音素区間残差用の中心化射影行列 (Centering Matrix) を構築する。
    ///
    /// 長さ 2 フレーム以上の区間ごとに、予測と教師からそれぞれの区間平均を引き (P = I - 1/L 1 1^T)、
    /// その差の L1 を計算可能にする。
    public static func buildCenteringMatrix(
        intervals: [(start: Int, end: Int)],
        alignedLen: Int
    ) -> (matrix: [Float], totalResidualFrames: Int) {
        if alignedLen <= 0 {
            return (matrix: [], totalResidualFrames: 0)
        }
        var flatP = [Float](repeating: 0.0, count: alignedLen * alignedLen)
        var totalResFrames = 0

        var idx = 0
        while idx < intervals.count {
            let seg = intervals[idx]
            let segStart = max(0, min(alignedLen, seg.start))
            let segEnd = max(segStart, min(alignedLen, seg.end))
            let len = segEnd - segStart
            if 2 <= len {
                let invLen = 1.0 / Float(len)
                var r = segStart
                while r < segEnd {
                    var c = segStart
                    while c < segEnd {
                        let flatIdx = (r * alignedLen) + c
                        if r == c {
                            flatP[flatIdx] = 1.0 - invLen
                        } else {
                            flatP[flatIdx] = -invLen
                        }
                        c += 1
                    }
                    r += 1
                }
                totalResFrames += len
            }
            idx += 1
        }

        return (matrix: flatP, totalResidualFrames: totalResFrames)
    }

    /// 音素区間残差損失 (中心化射影行列版)
    public static func phonemeResidualLoss(
        predicted: MLXArray,
        target: MLXArray,
        centeringMatrix: MLXArray,
        totalResidualFrames: MLXArray
    ) -> MLXArray {
        let diff = predicted - target
        let residualDiff = matmul(centeringMatrix, diff)
        let featureDim = Float(predicted.shape[predicted.ndim - 1])
        let totalValid = totalResidualFrames * featureDim
        return sum(abs(residualDiff)) / (totalValid + 1e-5)
    }

    /// 音素区間残差損失 (平均形 μ 教師版)
    ///
    /// 予測の区間中心化と μ の L1 を、有効フレーム数 × 特徴量次元 (64) で割った損失。
    public static func phonemeResidualLossWithMu(
        predicted: MLXArray,
        muTarget: MLXArray,
        centeringMatrix: MLXArray,
        totalResidualFrames: MLXArray
    ) -> MLXArray {
        let pCentered = matmul(centeringMatrix, predicted)
        let residualDiff = pCentered - muTarget
        let featureDim = Float(predicted.shape[predicted.ndim - 1])
        let totalValid = totalResidualFrames * featureDim
        return sum(abs(residualDiff)) / (totalValid + 1e-5)
    }

    /// 音素区間残差損失 (区間リスト直接版)
    public static func phonemeResidualLoss(
        predicted: MLXArray,
        target: MLXArray,
        intervals: [(start: Int, end: Int)]
    ) -> MLXArray {
        var residualSum = MLXArray(0.0)
        var totalFrames = 0
        let featureDim = Float(predicted.shape[predicted.ndim - 1])

        var i = 0
        while i < intervals.count {
            let seg = intervals[i]
            let maxT = predicted.shape[1]
            let segStart = max(0, min(maxT, seg.start))
            let segEnd = max(segStart, min(maxT, seg.end))
            let len = segEnd - segStart
            if 2 <= len {
                let pSeg = predicted[0..., segStart..<segEnd, 0...]
                let tSeg = target[0..., segStart..<segEnd, 0...]
                let pMean = mean(pSeg, axis: 1, keepDims: true)
                let tMean = mean(tSeg, axis: 1, keepDims: true)
                let pRes = pSeg - pMean
                let tRes = tSeg - tMean
                let segDiff = abs(pRes - tRes)
                residualSum = residualSum + sum(segDiff)
                totalFrames += len
            }
            i += 1
        }

        if totalFrames <= 0 {
            return MLXArray(0.0)
        }
        let totalValid = Float(totalFrames) * featureDim
        return residualSum / (totalValid + 1e-5)
    }
}

/// 三つ組音素キー: (直前音素 ID, 現在音素 ID, 直後音素 ID)
public struct TripletKey: Hashable, Sendable {
    public let prevId: Int
    public let currId: Int
    public let nextId: Int

    public init(prevId: Int, currId: Int, nextId: Int) {
        self.prevId = prevId
        self.currId = currId
        self.nextId = nextId
    }
}

/// 三つ組の平均形実測テーブル
///
/// 教師対数メルと音素区間から実測された三つ組の平均残差軌跡（8点 x 64次元）を保持する。
public struct TripletAverageTable: Sendable {
    public let table: [TripletKey: [[Float]]]

    public init(table: [TripletKey: [[Float]]]) {
        self.table = table
    }

    /// 特徴量フレームから三つ組キー (直前, 現在, 直後) を抽出する
    public static func extractKey(features: [[Float]], segStart: Int) -> TripletKey {
        var currId = 1
        var prevId = 1
        var nextId = 1
        if segStart < features.count {
            let frame = features[segStart]
            // 現在 ID: ch 0 ..< 64
            var maxCurr: Float = 1.0
            var i = 0
            while i < 64 {
                if i < frame.count {
                    if maxCurr < frame[i] {
                        maxCurr = frame[i]
                        currId = i
                    }
                }
                i += 1
            }
            // 直前 ID: ch 64 ..< 128
            var maxPrev: Float = 1.0
            var j = 0
            while j < 64 {
                let ch = 64 + j
                if ch < frame.count {
                    if maxPrev < frame[ch] {
                        maxPrev = frame[ch]
                        prevId = j
                    }
                }
                j += 1
            }
            // 直後 ID: ch 128 ..< 192
            var maxNext: Float = 1.0
            var k = 0
            while k < 64 {
                let ch = 128 + k
                if ch < frame.count {
                    if maxNext < frame[ch] {
                        maxNext = frame[ch]
                        nextId = k
                    }
                }
                k += 1
            }
        }
        return TripletKey(prevId: prevId, currId: currId, nextId: nextId)
    }

    /// 5,000 発話の訓練データから三つ組平均形テーブルを構築する
    public static func build(
        trainingData: [(features: [[Float]], targets: [[Float]])],
        minCount: Int = 8
    ) -> TripletAverageTable {
        var keyCounts: [TripletKey: Int] = [:]
        var keySums: [TripletKey: [[Float]]] = [:]
        let melCh = AudioConfig.melChannels

        var dIdx = 0
        while dIdx < trainingData.count {
            let pair = trainingData[dIdx]
            let rawLen = min(pair.features.count, pair.targets.count)
            let intervals = AcousticLossFunctions.findPhonemeIntervals(features: pair.features, rawLen: rawLen)

            var sIdx = 0
            while sIdx < intervals.count {
                let seg = intervals[sIdx]
                let len = seg.end - seg.start
                if 2 <= len {
                    let key = extractKey(features: pair.features, segStart: seg.start)

                    // 1. 各区間で教師メルから区間平均を引く
                    var meanMel = [Float](repeating: 0.0, count: melCh)
                    var c = 0
                    while c < melCh {
                        var sumC: Float = 0.0
                        var t = 0
                        while t < len {
                            sumC += pair.targets[seg.start + t][c]
                            t += 1
                        }
                        meanMel[c] = sumC / Float(len)
                        c += 1
                    }

                    // 2. 位置 k/7 (k = 0..7) の 8 点へ線形補間
                    var interp8 = [[Float]](repeating: [Float](repeating: 0.0, count: melCh), count: 8)
                    var k = 0
                    while k < 8 {
                        let continuousPos = (Float(k) / 7.0) * Float(len - 1)
                        let s0 = Int(continuousPos)
                        var s1 = s0 + 1
                        if len <= s1 {
                            s1 = len - 1
                        }
                        let alpha = continuousPos - Float(s0)
                        var chIdx = 0
                        while chIdx < melCh {
                            let res0 = pair.targets[seg.start + s0][chIdx] - meanMel[chIdx]
                            let res1 = pair.targets[seg.start + s1][chIdx] - meanMel[chIdx]
                            interp8[k][chIdx] = (1.0 - alpha) * res0 + alpha * res1
                            chIdx += 1
                        }
                        k += 1
                    }

                    // 3. キーごとに集計
                    let curCount = keyCounts[key] ?? 0
                    keyCounts[key] = curCount + 1
                    var curSum = keySums[key] ?? [[Float]](repeating: [Float](repeating: 0.0, count: melCh), count: 8)
                    var ki = 0
                    while ki < 8 {
                        var ci = 0
                        while ci < melCh {
                            curSum[ki][ci] += interp8[ki][ci]
                            ci += 1
                        }
                        ki += 1
                    }
                    keySums[key] = curSum
                }
                sIdx += 1
            }
            dIdx += 1
        }

        // 出現が minCount (8回) 以上のキーだけ平均を残し、8 点の平均を引いて中心化
        var resultTable: [TripletKey: [[Float]]] = [:]
        for (key, count) in keyCounts {
            if count < minCount {
                continue
            }
            if let sumArr = keySums[key] {
                var avgArr = [[Float]](repeating: [Float](repeating: 0.0, count: melCh), count: 8)
                let invCount = 1.0 / Float(count)
                var k = 0
                while k < 8 {
                    var c = 0
                    while c < melCh {
                        avgArr[k][c] = sumArr[k][c] * invCount
                        c += 1
                    }
                    k += 1
                }

                // 8 点の平均を引いて中心化
                var centered = [[Float]](repeating: [Float](repeating: 0.0, count: melCh), count: 8)
                var c = 0
                while c < melCh {
                    var sumK: Float = 0.0
                    var k = 0
                    while k < 8 {
                        sumK += avgArr[k][c]
                        k += 1
                    }
                    let mean8 = sumK / 8.0
                    k = 0
                    while k < 8 {
                        centered[k][c] = avgArr[k][c] - mean8
                        k += 1
                    }
                    c += 1
                }
                resultTable[key] = centered
            }
        }

        return TripletAverageTable(table: resultTable)
    }

    /// 各発話の各フレームに対する平均形 μ 系列を生成する
    public func generateMuSequence(
        features: [[Float]],
        rawLen: Int,
        alignedLen: Int
    ) -> [[Float]] {
        let melCh = AudioConfig.melChannels
        var mu = [[Float]](repeating: [Float](repeating: 0.0, count: melCh), count: alignedLen)
        let intervals = AcousticLossFunctions.findPhonemeIntervals(features: features, rawLen: rawLen)

        var sIdx = 0
        while sIdx < intervals.count {
            let seg = intervals[sIdx]
            let len = seg.end - seg.start
            if 2 <= len {
                let key = Self.extractKey(features: features, segStart: seg.start)
                if let entry = table[key] {
                    // 8 点を区間長へ線形補間
                    var interp = [[Float]](repeating: [Float](repeating: 0.0, count: melCh), count: len)
                    var f = 0
                    while f < len {
                        let continuousK = (Float(f) / Float(len - 1)) * 7.0
                        let k0 = Int(continuousK)
                        var k1 = k0 + 1
                        if 8 <= k1 {
                            k1 = 7
                        }
                        let alpha = continuousK - Float(k0)
                        var c = 0
                        while c < melCh {
                            interp[f][c] = (1.0 - alpha) * entry[k0][c] + alpha * entry[k1][c]
                            c += 1
                        }
                        f += 1
                    }

                    // もう一度平均を引いたものを μ とする
                    var c = 0
                    while c < melCh {
                        var sumF: Float = 0.0
                        var fi = 0
                        while fi < len {
                            sumF += interp[fi][c]
                            fi += 1
                        }
                        let meanF = sumF / Float(len)
                        fi = 0
                        while fi < len {
                            let frameIdx = seg.start + fi
                            if frameIdx < alignedLen {
                                mu[frameIdx][c] = interp[fi][c] - meanF
                            }
                            fi += 1
                        }
                        c += 1
                    }
                }
            }
            sIdx += 1
        }
        return mu
    }
}
#endif
