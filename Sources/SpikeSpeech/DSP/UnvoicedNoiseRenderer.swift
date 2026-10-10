import Foundation

/// 無声区間の雑音励振レンダラ
///
/// なぜ無声フレームをボコーダ出力ではなく雑音で生成するか:
/// 本プロジェクトのニューラルボコーダは入力（メル・F0・有声フラグ）から波形を決定論的に写像する畳み込み生成器で、
/// 雑音源も敵対的学習も持たない。そのため摩擦音・無声化母音・語尾の息といった本来は非周期の区間を
/// 疑似周期的なブザー音として生成し、「発話の後にロボットの残響が付く」ように聞こえていた
/// （合成音の語尾でピッチ推定器が 60〜170 Hz の周期を検出する一方、教師は無声）。
/// 無声フレームは、音響モデルが予測した対数メル包絡で整形した白色雑音に置き換え、
/// 有声フレームとはフレーム境界で線形にクロスフェードする。
public enum UnvoicedNoiseRenderer {

    /// 無声フレームの雑音信号と、サンプル単位の混合重み（1.0 = 雑音、0.0 = ボコーダ出力）を生成する
    /// - Parameters:
    ///   - mel: 対数メル系列 [T][melChannels]
    ///   - voicedFlags: フレームごとの有声フラグ（0.5 以上で有声）
    ///   - extractor: 学習時と同一設定のメル抽出器（フィルタバンクと対数の定義を共有する）
    ///   - silenceLogMelMean: この値以下の平均対数メルは無音とみなし雑音を入れない
    public static func render(
        mel: [[Float]],
        voicedFlags: [Float],
        extractor: MelSpectrogramExtractor,
        silenceLogMelMean: Float = -6.0,
        seed: UInt64 = 2026
    ) -> (noise: [Float], weight: [Float]) {
        let totalFrames = mel.count
        let hop = extractor.hopSize
        let fftSize = extractor.fftSize
        let fftBins = extractor.fftBins
        let totalSamples = totalFrames * hop
        var noise = [Float](repeating: 0.0, count: totalSamples)
        var weight = [Float](repeating: 0.0, count: totalSamples)
        if totalFrames <= 0 {
            return (noise, weight)
        }

        // 1. 無声かつ非無音のフレームを選ぶ
        var isNoiseFrame = [Bool](repeating: false, count: totalFrames)
        var f = 0
        while f < totalFrames {
            var voiced: Float = 1.0
            if f < voicedFlags.count {
                voiced = voicedFlags[f]
            }
            var mean: Float = 0.0
            if mel[f].isEmpty != true {
                mean = mel[f].reduce(0.0, +) / Float(mel[f].count)
            }
            if voiced < 0.5 && silenceLogMelMean < mean {
                isNoiseFrame[f] = true
            }
            f += 1
        }

        // 2. 合成窓（Hann, fftSize）と重なり加算の正規化係数
        var synthWindow = [Float](repeating: 0.0, count: fftSize)
        var n = 0
        while n < fftSize {
            synthWindow[n] = 0.5 * (1.0 - cosf((2.0 * Float.pi * Float(n)) / Float(fftSize)))
            n += 1
        }
        var windowSum: Float = 0.0
        n = 0
        while n < fftSize {
            windowSum += synthWindow[n]
            n += 1
        }
        let olaNorm = Float(hop) / max(1e-6, windowSum)

        // 3. フレームごとに「メル包絡 × 乱数位相」のスペクトルを逆 FFT して重なり加算（1 回目）
        func synthesizeFrames(gains: [Float], into out: inout [Float]) {
            var re = [Float](repeating: 0.0, count: fftSize)
            var im = [Float](repeating: 0.0, count: fftSize)
            var f = 0
            while f < totalFrames {
                if isNoiseFrame[f] {
                    let power = extractor.melToLinearPower(logMel: mel[f])
                    var rng = seed &+ (UInt64(f) &* 0x9E3779B97F4A7C15)
                    var k = 0
                    while k < fftSize {
                        re[k] = 0.0
                        im[k] = 0.0
                        k += 1
                    }
                    k = 0
                    while k < fftBins {
                        rng ^= rng << 13
                        rng ^= rng >> 7
                        rng ^= rng << 17
                        let phase = Float(rng >> 40) / Float(1 << 24) * 2.0 * Float.pi
                        let mag = sqrtf(max(0.0, power[k])) * gains[f]
                        re[k] = mag * cosf(phase)
                        im[k] = mag * sinf(phase)
                        if 0 < k && k < (fftSize / 2) {
                            re[fftSize - k] = re[k]
                            im[fftSize - k] = -im[k]
                        }
                        k += 1
                    }
                    im[0] = 0.0
                    im[fftSize / 2] = 0.0
                    inverseFFT(real: &re, imag: &im)
                    // フレーム中心を解析フレーム（start = f * hop, 長さ frameSize）の中心に合わせる
                    let center = f * hop + extractor.frameSize / 2
                    let start = center - fftSize / 2
                    var i = 0
                    while i < fftSize {
                        let idx = start + i
                        if 0 <= idx && idx < totalSamples {
                            out[idx] += re[i] * synthWindow[i] * olaNorm
                        }
                        i += 1
                    }
                }
                f += 1
            }
        }

        var gains = [Float](repeating: 1.0, count: totalFrames)
        synthesizeFrames(gains: gains, into: &noise)

        // 4. 較正: 生成雑音の対数メルを同じ抽出器で測り、目標との平均差からフレーム利得を補正して再合成
        // なぜ較正するか: 擬似逆・窓・重なり加算の定数を厳密に追わなくても、
        // 学習時と同じ抽出器で測った対数メルの平均が目標と一致すれば音量とバランスが揃うため。
        let measured = extractor.extractLogMel(pcm: noise)
        f = 0
        while f < totalFrames {
            if isNoiseFrame[f] && f < measured.count {
                let target = mel[f].reduce(0.0, +) / Float(max(1, mel[f].count))
                let got = measured[f].reduce(0.0, +) / Float(max(1, measured[f].count))
                var g = expf((target - got) * 0.5)
                if g.isFinite != true {
                    g = 1.0
                }
                gains[f] = min(8.0, max(0.125, g))
            }
            f += 1
        }
        noise = [Float](repeating: 0.0, count: totalSamples)
        synthesizeFrames(gains: gains, into: &noise)

        // 5. 混合重み（フレーム境界で 1 フレーム幅の線形ランプ）
        f = 0
        while f < totalFrames {
            let cur: Float = isNoiseFrame[f] ? 1.0 : 0.0
            var next: Float = cur
            if (f + 1) < totalFrames {
                next = isNoiseFrame[f + 1] ? 1.0 : 0.0
            }
            var i = 0
            while i < hop {
                let x = Float(i) / Float(hop)
                // フレーム前半は現フレームの値、後半で次フレームへ向けて遷移
                var w = cur
                if 0.5 <= x {
                    w = cur + (next - cur) * ((x - 0.5) * 2.0)
                }
                weight[f * hop + i] = w
                i += 1
            }
            f += 1
        }
        return (noise, weight)
    }

    /// 基数 2 の反復逆 FFT（長さは 2 のべき乗）
    static func inverseFFT(real: inout [Float], imag: inout [Float]) {
        let n = real.count
        // 共役 → 順 FFT → 共役 / n
        var i = 0
        while i < n {
            imag[i] = -imag[i]
            i += 1
        }
        forwardFFT(real: &real, imag: &imag)
        let inv = 1.0 / Float(n)
        i = 0
        while i < n {
            real[i] = real[i] * inv
            imag[i] = -imag[i] * inv
            i += 1
        }
    }

    static func forwardFFT(real: inout [Float], imag: inout [Float]) {
        let n = real.count
        var j = 0
        var i = 1
        while i < n {
            var bit = n >> 1
            while 0 < (j & bit) {
                j ^= bit
                bit >>= 1
            }
            j ^= bit
            if i < j {
                real.swapAt(i, j)
                imag.swapAt(i, j)
            }
            i += 1
        }
        var len = 2
        while len <= n {
            let ang = -2.0 * Float.pi / Float(len)
            let wRe = cosf(ang)
            let wIm = sinf(ang)
            var start = 0
            while start < n {
                var cRe: Float = 1.0
                var cIm: Float = 0.0
                var k = 0
                while k < len / 2 {
                    let aRe = real[start + k]
                    let aIm = imag[start + k]
                    let bRe = real[start + k + len / 2] * cRe - imag[start + k + len / 2] * cIm
                    let bIm = real[start + k + len / 2] * cIm + imag[start + k + len / 2] * cRe
                    real[start + k] = aRe + bRe
                    imag[start + k] = aIm + bIm
                    real[start + k + len / 2] = aRe - bRe
                    imag[start + k + len / 2] = aIm - bIm
                    let nRe = cRe * wRe - cIm * wIm
                    cIm = cRe * wIm + cIm * wRe
                    cRe = nRe
                    k += 1
                }
                start += len
            }
            len <<= 1
        }
    }
}
