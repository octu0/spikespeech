import Foundation

/// 実音声信号の短時間エネルギー特性に基づき、モーラ境界を物理的に同期・決定する音響アライメント器
///
/// 等時間割り（speechFrames / M）を完全撤廃し、WAV 音声信号の平滑化エネルギー（RMS パワー）から
/// 累積エネルギー等分および局所エネルギー谷スナップ（Local Valley Snapping）によってモーラ境界を決定する。
public final class AcousticEnergyAligner: Sendable {
    public init() {}

    /// 発話区間の短時間平滑化 RMS パワー系列を算出する
    ///
    /// - Parameters:
    ///   - pcm: 16kHz PCM 音声信号
    ///   - hopSize: フレームシフトサンプル数 (通常 160 サンプル = 10ms)
    ///   - startFrame: 発話区間の開始絶対フレーム (leadSilence)
    ///   - frameCount: 発話区間の総フレーム数 (speechFrames)
    /// - Returns: 各フレームの平滑化エネルギー系列 (長さ frameCount)
    public func computeSmoothedEnergy(
        pcm: [Float],
        hopSize: Int,
        startFrame: Int,
        frameCount: Int
    ) -> [Float] {
        if frameCount <= 0 {
            return []
        }
        if hopSize <= 0 {
            return [Float](repeating: 0.0, count: frameCount)
        }

        var rawPowers = [Float](repeating: 0.0, count: frameCount)
        let pcmLength = pcm.count

        var i = 0
        while i < frameCount {
            let absFrame = startFrame + i
            let sampleStart = absFrame * hopSize
            let sampleEnd = min(pcmLength, sampleStart + hopSize)

            if sampleEnd <= sampleStart {
                rawPowers[i] = 1.0e-9
            } else {
                var sumSq: Float = 0.0
                var s = sampleStart
                while s < sampleEnd {
                    let val = pcm[s]
                    sumSq += val * val
                    s += 1
                }
                let count = Float(sampleEnd - sampleStart)
                rawPowers[i] = (sumSq / count) + 1.0e-9
            }
            i += 1
        }

        // 3 フレーム移動平均（前・現在・後）による平滑化
        var smoothed = [Float](repeating: 0.0, count: frameCount)
        var k = 0
        while k < frameCount {
            var sum: Float = 0.0
            var count: Float = 0.0

            if 0 <= k - 1 {
                sum += rawPowers[k - 1]
                count += 1.0
            }
            sum += rawPowers[k]
            count += 1.0
            if k + 1 < frameCount {
                sum += rawPowers[k + 1]
                count += 1.0
            }

            smoothed[k] = sum / count
            k += 1
        }

        return smoothed
    }

    /// 累積エネルギー等分および局所エネルギー谷スナップによりモーラ境界を算出する
    ///
    /// - Parameters:
    ///   - smoothedEnergy: 平滑化エネルギー系列 (長さ frameCount)
    ///   - moraCount: モーラ数 M
    ///   - moraMinFrames: 各モーラに必要な最小フレーム数配列 (長さ M)
    /// - Returns: モーラ境界インデックス配列 (長さ M + 1, 0 から frameCount まで単調増加)
    public func findMoraBoundaries(
        smoothedEnergy: [Float],
        moraCount: Int,
        moraMinFrames: [Int]
    ) -> [Int] {
        let frameCount = smoothedEnergy.count
        if moraCount <= 0 || frameCount <= 0 {
            return []
        }
        if moraCount == 1 {
            return [0, frameCount]
        }

        var boundaries = [Int](repeating: 0, count: moraCount + 1)
        boundaries[0] = 0
        boundaries[moraCount] = frameCount

        // 累積エネルギー系列の算出
        var cumEnergy = [Float](repeating: 0.0, count: frameCount)
        var cumSum: Float = 0.0
        var i = 0
        while i < frameCount {
            cumSum += smoothedEnergy[i]
            cumEnergy[i] = cumSum
            i += 1
        }

        let totalEnergy = cumSum
        let avgMoraFrames = max(1, frameCount / moraCount)
        // 局所探索窓の幅（平均モーラ長の 1/4、最大 4 フレーム = 40ms）
        let windowDelta = max(1, min(4, avgMoraFrames / 4))

        // 残りモーラに配分すべき後方最小フレーム数合計のプレフィックス和
        var suffixMinFrames = [Int](repeating: 0, count: moraCount + 1)
        var sIdx = moraCount - 1
        while 0 <= sIdx {
            suffixMinFrames[sIdx] = suffixMinFrames[sIdx + 1] + moraMinFrames[sIdx]
            sIdx -= 1
        }

        var m = 1
        while m < moraCount {
            // 累積エネルギー等分点の探索
            let targetEnergy = totalEnergy * (Float(m) / Float(moraCount))
            var anchorFrame = 0
            var k = 0
            while k < frameCount {
                if targetEnergy <= cumEnergy[k] {
                    anchorFrame = k
                    break
                }
                k += 1
            }

            // 探索窓 [anchorFrame - windowDelta, anchorFrame + windowDelta] における局所エネルギー谷の探索
            let searchStart = max(0, anchorFrame - windowDelta)
            let searchEnd = min(frameCount - 1, anchorFrame + windowDelta)

            var bestValleyFrame = anchorFrame
            var minEnergy = Float.greatestFiniteMagnitude

            var cand = searchStart
            while cand <= searchEnd {
                let e = smoothedEnergy[cand]
                if e < minEnergy {
                    minEnergy = e
                    bestValleyFrame = cand
                }
                cand += 1
            }

            // 前後モーラの最小フレーム数制約に基づく安全クランプ
            let minBoundary = boundaries[m - 1] + moraMinFrames[m - 1]
            let maxBoundary = frameCount - suffixMinFrames[m]

            var chosenBoundary = bestValleyFrame
            if chosenBoundary < minBoundary {
                chosenBoundary = minBoundary
            }
            if maxBoundary < chosenBoundary {
                chosenBoundary = maxBoundary
            }

            boundaries[m] = chosenBoundary
            m += 1
        }

        return boundaries
    }

    /// 音響エネルギーアライメントにより各音素の確定フレーム数を算出する
    ///
    /// - Parameters:
    ///   - pcm: 16kHz PCM 音声信号
    ///   - hopSize: 160
    ///   - leadSilence: 発話先頭無音フレーム数
    ///   - speechFrames: 発話区間総フレーム数
    ///   - phrases: アクセント句列
    ///   - lengthRegulator: 音素比率モデル算出インスタンス
    /// - Returns: 各音素の確定継続フレーム数配列 (全合計が speechFrames と厳密一致)
    public func alignPhonemes(
        pcm: [Float],
        hopSize: Int,
        leadSilence: Int,
        speechFrames: Int,
        phrases: [AccentPhrase],
        lengthRegulator: LengthRegulator
    ) -> [Int]? {
        // 1. 全モーラの収集
        var allMoras: [MoraToken] = []
        var pIdx = 0
        while pIdx < phrases.count {
            var mIdx = 0
            while mIdx < phrases[pIdx].moras.count {
                allMoras.append(phrases[pIdx].moras[mIdx])
                mIdx += 1
            }
            pIdx += 1
        }

        let moraCount = allMoras.count
        if moraCount <= 0 || speechFrames <= 0 {
            return nil
        }

        // 各モーラの音素数および全音素数の確認
        var moraMinFrames = [Int](repeating: 0, count: moraCount)
        var totalPhoneCount = 0
        var m = 0
        while m < moraCount {
            let phoneCount = allMoras[m].phonemes.count
            moraMinFrames[m] = max(1, phoneCount)
            totalPhoneCount += moraMinFrames[m]
            m += 1
        }

        if speechFrames < totalPhoneCount {
            return nil
        }

        // 2. 短時間平滑化エネルギーの算出
        let smoothedEnergy = computeSmoothedEnergy(
            pcm: pcm,
            hopSize: hopSize,
            startFrame: leadSilence,
            frameCount: speechFrames
        )

        // 3. モーラ境界の決定
        let boundaries = findMoraBoundaries(
            smoothedEnergy: smoothedEnergy,
            moraCount: moraCount,
            moraMinFrames: moraMinFrames
        )

        if boundaries.count != moraCount + 1 {
            return nil
        }

        // 4. 各モーラ内の音素比率配分
        var resultDurations: [Int] = []
        m = 0
        while m < moraCount {
            let moraStart = boundaries[m]
            let moraEnd = boundaries[m + 1]
            let moraLen = max(moraMinFrames[m], moraEnd - moraStart)

            let phonemes = allMoras[m].phonemes
            let ratios = lengthRegulator.moraPhonemeRatios(phonemes: phonemes)

            switch phonemes.count {
            case 1:
                resultDurations.append(moraLen)
            case 2:
                let r0 = ratios[0]
                var d0 = Int(roundf(Float(moraLen) * r0))
                if d0 < 1 {
                    d0 = 1
                }
                var d1 = moraLen - d0
                if d1 < 1 {
                    d1 = 1
                    d0 = moraLen - 1
                }
                resultDurations.append(d0)
                resultDurations.append(d1)
            default:
                // 3音素以上（拗音など）
                var allocatedSum = 0
                var ph = 0
                while ph < phonemes.count - 1 {
                    var d = Int(roundf(Float(moraLen) * ratios[ph]))
                    if d < 1 {
                        d = 1
                    }
                    resultDurations.append(d)
                    allocatedSum += d
                    ph += 1
                }
                var lastD = moraLen - allocatedSum
                if lastD < 1 {
                    lastD = 1
                }
                resultDurations.append(lastD)
            }
            m += 1
        }

        // 合計フレーム数と speechFrames の厳密一致検査
        var totalSum = 0
        var r = 0
        while r < resultDurations.count {
            totalSum += resultDurations[r]
            r += 1
        }

        let diff = speechFrames - totalSum
        if diff != 0 && 0 < resultDurations.count {
            // 端数（最後の音素に吸収）
            let lastIdx = resultDurations.count - 1
            resultDurations[lastIdx] += diff
        }

        return resultDurations
    }
}
