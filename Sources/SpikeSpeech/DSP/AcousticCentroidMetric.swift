import Foundation

/// スペクトル重心および余弦類似度客観指標計測モジュール
///
/// 設計仕様（design_cfc_acoustic.md）に準拠:
/// 窓 512、ホップ 160、Hann。
/// フレームの平均振幅が、そのファイルの 55 パーセンタイルを超えるものを有声とする。
/// 連続する有声フレームについて、200–1500 Hz および 200–4000 Hz のスペクトル重心の差の絶対値の中央値を取る。
public enum AcousticCentroidMetric {

    public struct Result: Sendable {
        public let cosRatio: Float
        public let centroidMedian200to4000: Float
        public let centroidMedian200to1500: Float
        public let voicedCount: Int

        public init(
            cosRatio: Float,
            centroidMedian200to4000: Float,
            centroidMedian200to1500: Float,
            voicedCount: Int
        ) {
            self.cosRatio = cosRatio
            self.centroidMedian200to4000 = centroidMedian200to4000
            self.centroidMedian200to1500 = centroidMedian200to1500
            self.voicedCount = voicedCount
        }
    }

    public static func measure(pcm: [Float]) -> Result {
        if pcm.isEmpty {
            return Result(cosRatio: 0.0, centroidMedian200to4000: 0.0, centroidMedian200to1500: 0.0, voicedCount: 0)
        }
        let winSize = 512
        let hopSize = 160
        if pcm.count < winSize {
            return Result(cosRatio: 0.0, centroidMedian200to4000: 0.0, centroidMedian200to1500: 0.0, voicedCount: 0)
        }
        let frameCount = max(1, (pcm.count - winSize) / hopSize)
        let sampleRate: Float = 16000.0

        var hann = [Float](repeating: 0.0, count: winSize)
        var n = 0
        while n < winSize {
            hann[n] = 0.5 * (1.0 - cosf((2.0 * Float.pi * Float(n)) / Float(winSize)))
            n += 1
        }

        let extractor = MelSpectrogramExtractor(
            sampleRate: sampleRate,
            melChannels: AudioConfig.melChannels,
            hopSize: hopSize,
            frameSize: winSize,
            fftSize: winSize
        )

        var frameAmplitudes = [Float](repeating: 0.0, count: frameCount)
        var frameMags = [[Float]](repeating: [Float](repeating: 0.0, count: 257), count: frameCount)
        var frameCentroids4k = [Float](repeating: 0.0, count: frameCount)
        var frameCentroids1k5 = [Float](repeating: 0.0, count: frameCount)

        let binHz = sampleRate / Float(winSize)
        let kMin = 7     // ~218.75 Hz
        let kMax4k = 128 // ~4000 Hz
        let kMax1k5 = 48 // ~1500 Hz

        var f = 0
        while f < frameCount {
            let sampleStart = f * hopSize
            var realBuf = [Float](repeating: 0.0, count: winSize)
            var imagBuf = [Float](repeating: 0.0, count: winSize)

            var ampSum: Float = 0.0
            var s = 0
            while s < winSize {
                let pcmIdx = sampleStart + s
                var sampleVal: Float = 0.0
                if pcmIdx < pcm.count {
                    sampleVal = pcm[pcmIdx]
                }
                ampSum += abs(sampleVal)
                realBuf[s] = sampleVal * hann[s]
                imagBuf[s] = 0.0
                s += 1
            }
            frameAmplitudes[f] = ampSum / Float(winSize)

            extractor.computeFFT(real: &realBuf, imag: &imagBuf)

            var num4k: Float = 0.0
            var den4k: Float = 0.0
            var num1k5: Float = 0.0
            var den1k5: Float = 0.0

            var b = 0
            while b <= 256 {
                let r = realBuf[b]
                let im = imagBuf[b]
                let mag = sqrtf(r * r + im * im)
                frameMags[f][b] = mag
                let freq = Float(b) * binHz
                if kMin <= b && b <= kMax4k {
                    num4k += freq * mag
                    den4k += mag
                }
                if kMin <= b && b <= kMax1k5 {
                    num1k5 += freq * mag
                    den1k5 += mag
                }
                b += 1
            }

            if 1e-6 < den4k {
                frameCentroids4k[f] = num4k / den4k
            }
            if 1e-6 < den1k5 {
                frameCentroids1k5[f] = num1k5 / den1k5
            }
            f += 1
        }

        let sortedAmps = frameAmplitudes.sorted()
        var p55Idx = Int(Float(frameCount) * 0.55)
        if frameCount <= p55Idx {
            p55Idx = frameCount - 1
        }
        let threshold55 = sortedAmps[p55Idx]

        var voicedPairs = 0
        var cosOver99Count = 0
        var centroidDiffs4k: [Float] = []
        var centroidDiffs1k5: [Float] = []

        var t = 1
        while t < frameCount {
            let isVoicedCurr = (threshold55 < frameAmplitudes[t])
            let isVoicedPrev = (threshold55 < frameAmplitudes[t - 1])
            if isVoicedCurr && isVoicedPrev {
                voicedPairs += 1

                var dot: Float = 0.0
                var normA: Float = 0.0
                var normB: Float = 0.0
                var k = kMin
                while k <= kMax4k {
                    let a = frameMags[t][k]
                    let b = frameMags[t - 1][k]
                    dot += a * b
                    normA += a * a
                    normB += b * b
                    k += 1
                }
                let denom = sqrtf(normA) * sqrtf(normB)
                var cosSim: Float = 0.0
                if 1e-6 < denom {
                    cosSim = dot / denom
                }
                if 0.99 < cosSim {
                    cosOver99Count += 1
                }

                centroidDiffs4k.append(abs(frameCentroids4k[t] - frameCentroids4k[t - 1]))
                centroidDiffs1k5.append(abs(frameCentroids1k5[t] - frameCentroids1k5[t - 1]))
            }
            t += 1
        }

        centroidDiffs4k.sort()
        var medDiff4k: Float = 0.0
        if centroidDiffs4k.isEmpty != true {
            medDiff4k = centroidDiffs4k[centroidDiffs4k.count / 2]
        }

        centroidDiffs1k5.sort()
        var medDiff1k5: Float = 0.0
        if centroidDiffs1k5.isEmpty != true {
            medDiff1k5 = centroidDiffs1k5[centroidDiffs1k5.count / 2]
        }

        var cosRatio: Float = 0.0
        if 0 < voicedPairs {
            cosRatio = Float(cosOver99Count) / Float(voicedPairs)
        }

        return Result(
            cosRatio: cosRatio,
            centroidMedian200to4000: medDiff4k,
            centroidMedian200to1500: medDiff1k5,
            voicedCount: voicedPairs
        )
    }

    public struct SegmentAnalysis: Sendable {
        public let cosRatio: Float
        public let centroidMedian200to1500: Float
        public let f0Median: Float
        public let sampleCount: Int

        public init(
            cosRatio: Float,
            centroidMedian200to1500: Float,
            f0Median: Float,
            sampleCount: Int
        ) {
            self.cosRatio = cosRatio
            self.centroidMedian200to1500 = centroidMedian200to1500
            self.f0Median = f0Median
            self.sampleCount = sampleCount
        }
    }

    /// 設計仕様書に準拠した F0 系列計測:
    /// 32 ms の Hann 窓、ホップ 320 サンプル、70–350 Hz の自己相関。
    /// ピークがゼロ遅れの 0.35 倍未満の窓は捨て、残った有声 F0 の配列を返す。
    public static func measureF0Values(
        pcm: [Float],
        sampleRate: Float = 16000.0,
        interpolateParabolic: Bool = false
    ) -> [Float] {
        if pcm.isEmpty {
            return []
        }
        let winSize = 512
        let hopSize = 320
        if pcm.count < winSize {
            return []
        }

        var hann = [Float](repeating: 0.0, count: winSize)
        var n = 0
        while n < winSize {
            hann[n] = 0.5 * (1.0 - cosf((2.0 * Float.pi * Float(n)) / Float(winSize)))
            n += 1
        }

        let minLag = Int(floor(sampleRate / 350.0)) // 45
        let maxLag = Int(ceil(sampleRate / 70.0))   // 229

        var validF0s: [Float] = []
        var start = 0
        while start + winSize <= pcm.count {
            var r0: Float = 0.0
            var winBuf = [Float](repeating: 0.0, count: winSize)
            var s = 0
            while s < winSize {
                let v = pcm[start + s] * hann[s]
                winBuf[s] = v
                r0 += v * v
                s += 1
            }

            if 1e-6 < r0 {
                var maxR: Float = -Float.greatestFiniteMagnitude
                var bestLag = minLag

                var lag = minLag
                while lag <= maxLag {
                    var r: Float = 0.0
                    var k = 0
                    let limit = winSize - lag
                    while k < limit {
                        r += winBuf[k] * winBuf[k + lag]
                        k += 1
                    }
                    if maxR < r {
                        maxR = r
                        bestLag = lag
                    }
                    lag += 1
                }

                if 0.35 * r0 <= maxR {
                    var f0: Float = sampleRate / Float(bestLag)
                    if interpolateParabolic {
                        if minLag < bestLag && bestLag < maxLag {
                            var rPrev: Float = 0.0
                            var rNext: Float = 0.0
                            var k = 0
                            let limitPrev = winSize - (bestLag - 1)
                            while k < limitPrev {
                                rPrev += winBuf[k] * winBuf[k + bestLag - 1]
                                k += 1
                            }
                            k = 0
                            let limitNext = winSize - (bestLag + 1)
                            while k < limitNext {
                                rNext += winBuf[k] * winBuf[k + bestLag + 1]
                                k += 1
                            }
                            let denom = 2.0 * (2.0 * maxR - rPrev - rNext)
                            if 1e-6 < denom {
                                var delta = (rNext - rPrev) / denom
                                if delta < -0.5 {
                                    delta = -0.5
                                }
                                if 0.5 < delta {
                                    delta = 0.5
                                }
                                let fineLag = Float(bestLag) + delta
                                if 0.0 < fineLag {
                                    f0 = sampleRate / fineLag
                                }
                            }
                        }
                    }
                    validF0s.append(f0)
                }
            }
            start += hopSize
        }
        return validF0s
    }

    /// 設計仕様書に準拠した F0 中央値計測
    public static func measureF0Median(
        pcm: [Float],
        sampleRate: Float = 16000.0,
        interpolateParabolic: Bool = false
    ) -> Float {
        var vals = measureF0Values(pcm: pcm, sampleRate: sampleRate, interpolateParabolic: interpolateParabolic)
        if vals.isEmpty {
            return 0.0
        }
        vals.sort()
        return vals[vals.count / 2]
    }

    /// 設計仕様書に準拠した有声 F0 標準偏差計測
    public static func measureF0StdDev(
        pcm: [Float],
        sampleRate: Float = 16000.0,
        interpolateParabolic: Bool = false
    ) -> Float {
        let vals = measureF0Values(pcm: pcm, sampleRate: sampleRate, interpolateParabolic: interpolateParabolic)
        if vals.isEmpty {
            return 0.0
        }
        var sum: Float = 0.0
        var i = 0
        while i < vals.count {
            sum += vals[i]
            i += 1
        }
        let mean = sum / Float(vals.count)
        var sumSq: Float = 0.0
        i = 0
        while i < vals.count {
            let diff = vals[i] - mean
            sumSq += diff * diff
            i += 1
        }
        return sqrtf(sumSq / Float(vals.count))
    }

    /// ファイルをサンプル数の半分で前後に分け、前半と後半の停止割合、200–1500 Hz重心、F0中央値を計測する
    public static func measureHalves(
        pcm: [Float],
        sampleRate: Float = 16000.0,
        interpolateParabolic: Bool = false
    ) -> (first: SegmentAnalysis, second: SegmentAnalysis) {
        if pcm.isEmpty {
            let zero = SegmentAnalysis(cosRatio: 0.0, centroidMedian200to1500: 0.0, f0Median: 0.0, sampleCount: 0)
            return (zero, zero)
        }
        let half = pcm.count / 2
        let firstPCM = Array(pcm[0..<half])
        let secondPCM = Array(pcm[half..<pcm.count])

        let firstMetric = measure(pcm: firstPCM)
        let firstF0 = measureF0Median(pcm: firstPCM, sampleRate: sampleRate, interpolateParabolic: interpolateParabolic)
        let firstSeg = SegmentAnalysis(
            cosRatio: firstMetric.cosRatio,
            centroidMedian200to1500: firstMetric.centroidMedian200to1500,
            f0Median: firstF0,
            sampleCount: firstPCM.count
        )

        let secondMetric = measure(pcm: secondPCM)
        let secondF0 = measureF0Median(pcm: secondPCM, sampleRate: sampleRate, interpolateParabolic: interpolateParabolic)
        let secondSeg = SegmentAnalysis(
            cosRatio: secondMetric.cosRatio,
            centroidMedian200to1500: secondMetric.centroidMedian200to1500,
            f0Median: secondF0,
            sampleCount: secondPCM.count
        )

        return (firstSeg, secondSeg)
    }

    /// 振幅エンベロープの 6–12 Hz 変調割合の計測:
    /// ホップ 160、窓 320 の RMS。
    /// ピークの 0.06 倍と 0.010 の大きい方以上のフレームを本体とし、最初の本体フレームから最後の本体フレームまでを使う。
    /// 本体の平均を引き、Hann を掛け、実数 FFT する。フレーム間隔は 10 ms。
    /// 6 Hz 以上 12 Hz 未満のパワー和を、2 Hz 以上 25 Hz 未満のパワー和で割る。
    public static func measureModulation6to12Ratio(pcm: [Float]) -> Float {
        let winSize = 320
        let hopSize = 160
        if pcm.count < winSize {
            return 0.0
        }
        let totalFrames = max(1, (pcm.count - winSize) / hopSize + 1)
        var frameRms = [Float](repeating: 0.0, count: totalFrames)
        var maxRms: Float = 0.0

        var f = 0
        while f < totalFrames {
            let start = f * hopSize
            var sumSq: Float = 0.0
            var s = 0
            while s < winSize {
                let v = pcm[start + s]
                sumSq += v * v
                s += 1
            }
            let rms = sqrtf(sumSq / Float(winSize))
            frameRms[f] = rms
            if maxRms < rms {
                maxRms = rms
            }
            f += 1
        }

        let threshold = max(0.010, maxRms * 0.06)
        var firstBody = -1
        var lastBody = -1
        var sf = 0
        while sf < totalFrames {
            if threshold <= frameRms[sf] {
                if firstBody < 0 {
                    firstBody = sf
                }
                lastBody = sf
            }
            sf += 1
        }

        if firstBody < 0 || lastBody < firstBody {
            return 0.0
        }

        let bodyLen = lastBody - firstBody + 1
        if bodyLen < 4 {
            return 0.0
        }

        var sumBody: Float = 0.0
        var bIdx = 0
        while bIdx < bodyLen {
            sumBody += frameRms[firstBody + bIdx]
            bIdx += 1
        }
        let meanBody = sumBody / Float(bodyLen)

        var winBuf = [Float](repeating: 0.0, count: bodyLen)
        var n = 0
        while n < bodyLen {
            let hann = 0.5 * (1.0 - cosf((2.0 * Float.pi * Float(n)) / Float(bodyLen)))
            winBuf[n] = (frameRms[firstBody + n] - meanBody) * hann
            n += 1
        }

        let halfLen = bodyLen / 2
        var power2to25: Float = 0.0
        var power6to12: Float = 0.0

        var k = 0
        while k <= halfLen {
            let freq = (Float(k) * 100.0) / Float(bodyLen)
            let in2to25 = (2.0 <= freq && freq < 25.0)
            let in6to12 = (6.0 <= freq && freq < 12.0)
            if in2to25 || in6to12 {
                var realSum: Float = 0.0
                var imagSum: Float = 0.0
                var j = 0
                let angleStep = (2.0 * Float.pi * Float(k)) / Float(bodyLen)
                while j < bodyLen {
                    let angle = angleStep * Float(j)
                    let v = winBuf[j]
                    realSum += v * cosf(angle)
                    imagSum -= v * sinf(angle)
                    j += 1
                }
                let power = (realSum * realSum) + (imagSum * imagSum)
                if in2to25 {
                    power2to25 += power
                }
                if in6to12 {
                    power6to12 += power
                }
            }
            k += 1
        }

        var ratio: Float = 0.0
        if 1e-12 < power2to25 {
            ratio = power6to12 / power2to25
        }
        return ratio
    }

    /// 発話本体区間における深いエネルギー落ち込み（閉鎖期・無音トラフ）の検出
    ///
    /// なぜ 80ms (8フレーム) 以上の落ち込みを計測するか:
    /// 音声合成において不自然な途切れや息継ぎ様の無音ギャップ（deep dips）の発生頻度と最大長を客観評価するため。
    public static func measureDeepDips(
        pcm: [Float]
    ) -> (dips: [(startSec: Float, lengthMs: Int, avgRms: Float)], maxDipLengthMs: Int) {
        if pcm.isEmpty {
            return ([], 0)
        }
        let winSize = 320
        let hopSize = 160
        if pcm.count < winSize {
            return ([], 0)
        }
        let totalFrames = max(1, (pcm.count - winSize) / hopSize + 1)
        var frameRms = [Float](repeating: 0.0, count: totalFrames)
        var maxRms: Float = 0.0

        var f = 0
        while f < totalFrames {
            let start = f * hopSize
            var sumSq: Float = 0.0
            var s = 0
            while s < winSize {
                let pcmIdx = start + s
                var v: Float = 0.0
                if pcmIdx < pcm.count {
                    v = pcm[pcmIdx] * 32768.0
                }
                sumSq += v * v
                s += 1
            }
            let rms = sqrtf(sumSq / Float(winSize))
            frameRms[f] = rms
            if maxRms < rms {
                maxRms = rms
            }
            f += 1
        }

        let bodyThreshold = max(0.010 * 32768.0, maxRms * 0.06)
        var firstBody = -1
        var lastBody = -1
        var sf = 0
        while sf < totalFrames {
            if bodyThreshold <= frameRms[sf] {
                if firstBody < 0 {
                    firstBody = sf
                }
                lastBody = sf
            }
            sf += 1
        }

        let dipThreshold = max(0.008 * 32768.0, maxRms * 0.12)
        var dips: [(startSec: Float, lengthMs: Int, avgRms: Float)] = []
        var maxDipLen = 0

        if 0 <= firstBody && firstBody <= lastBody {
            var curStart = -1
            var curCount = 0
            var curSumRms: Float = 0.0

            var bf = firstBody
            while bf <= lastBody {
                let rmsVal = frameRms[bf]
                if rmsVal < dipThreshold {
                    if curStart < 0 {
                        curStart = bf
                        curCount = 0
                        curSumRms = 0.0
                    }
                    curCount += 1
                    curSumRms += rmsVal
                } else {
                    if 0 <= curStart {
                        if 8 <= curCount {
                            let avg = curSumRms / Float(curCount)
                            if avg < 800.0 {
                                let startSec = Float(curStart * hopSize) / 16000.0
                                let lenMs = curCount * 10
                                if maxDipLen < lenMs {
                                    maxDipLen = lenMs
                                }
                                dips.append((startSec: startSec, lengthMs: lenMs, avgRms: avg))
                            }
                        }
                        curStart = -1
                    }
                }
                bf += 1
            }
            if 0 <= curStart && 8 <= curCount {
                let avg = curSumRms / Float(curCount)
                if avg < 800.0 {
                    let startSec = Float(curStart * hopSize) / 16000.0
                    let lenMs = curCount * 10
                    if maxDipLen < lenMs {
                        maxDipLen = lenMs
                    }
                    dips.append((startSec: startSec, lengthMs: lenMs, avgRms: avg))
                }
            }
        }

        return (dips: dips, maxDipLengthMs: maxDipLen)
    }

    /// 有声区間 F0 標準偏差の計測 (Hz)
    public static func measureVoicedF0StdDev(
        pcm: [Float],
        sampleRate: Float = 16000.0
    ) -> Float {
        if pcm.isEmpty {
            return 0.0
        }
        let tracker = PitchTracker(sampleRate: sampleRate)
        let res = tracker.track(pcm: pcm)
        var voicedF0s: [Float] = []
        var i = 0
        while i < res.frameCount {
            if 0.5 <= res.voiced[i] && 70.0 <= res.f0[i] {
                voicedF0s.append(res.f0[i])
            }
            i += 1
        }
        if voicedF0s.count <= 1 {
            return 0.0
        }
        var sum: Float = 0.0
        var j = 0
        while j < voicedF0s.count {
            sum += voicedF0s[j]
            j += 1
        }
        let mean = sum / Float(voicedF0s.count)
        var sumSq: Float = 0.0
        j = 0
        while j < voicedF0s.count {
            let diff = voicedF0s[j] - mean
            sumSq += diff * diff
            j += 1
        }
        return sqrtf(sumSq / Float(voicedF0s.count))
    }
}

