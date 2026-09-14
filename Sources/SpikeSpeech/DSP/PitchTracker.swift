import Foundation

/// ピッチ追跡およびエネルギー抽出の結果
public struct PitchTrackResult: Sendable {
    /// 各フレームの基本周波数 F0 [Hz]（無声区間は 0.0）
    public let f0: [Float]

    /// 各フレームの有声無声フラグ（有声: 1.0, 無声: 0.0）
    public let voiced: [Float]

    /// 各フレームの短時間 RMS 音響エネルギー（0.0 〜 1.0）
    public let energy: [Float]

    /// 総フレーム数
    public let frameCount: Int

    public init(f0: [Float], voiced: [Float], energy: [Float], frameCount: Int) {
        self.f0 = f0
        self.voiced = voiced
        self.energy = energy
        self.frameCount = frameCount
    }
}

/// Pure Swift による正規化交差相関法 (NCCF) 高精度ピッチトラッカー & 音響エネルギー抽出器
///
/// なぜ自己相関法 (NCCF) を採用するか:
/// 1. 外部 C / Python ライブラリ（WORLD, REAPER, PyWorld 等）への依存を完全に排除し、
///    macOS / Linux (Cloud Run) のあらゆる環境で同一の音響韻律特徴量を安定抽出するため。
/// 2. 人間の音声信号において、声帯振動の基本周波数 F0 は波形の周期自己相関の極大点として最も頑健に現れる。
/// 3. 放物線補間（Parabolic Interpolation）を組み合わせることで、
///    16kHz サンプリング時の離散ラグ量子化誤差（数 Hz 〜 10 数 Hz の段差）を解消し、
///    サブサンプル精度の滑らかで生々しい実測 F0 コンターを復元するため。
public final class PitchTracker: @unchecked Sendable {

    /// サンプリング周波数 [Hz]
    public let sampleRate: Float

    /// フレーム長 [サンプル] (25ms = 400 サンプル @ 16kHz)
    public let frameLength: Int

    /// フレームシフト (ホップサイズ) [サンプル] (10ms = 160 サンプル @ 16kHz)
    public let hopSize: Int

    /// 検出最小ピッチ [Hz] (男性の極低音域 60Hz に対応)
    public let minF0: Float

    /// 検出最大ピッチ [Hz] (女性・小児の高音域 500Hz に対応)
    public let maxF0: Float

    /// 探索最小ラグ [サンプル]
    private let minLag: Int

    /// 探索最大ラグ [サンプル]
    private let maxLag: Int

    /// 有声判定の相関閾値
    private let voicingThreshold: Float

    public init(
        sampleRate: Float = 16000.0,
        frameLength: Int = 400,
        hopSize: Int = AudioConfig.hopSize, // 160
        minF0: Float = 60.0,
        maxF0: Float = 500.0,
        voicingThreshold: Float = 0.35
    ) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.minF0 = minF0
        self.maxF0 = maxF0
        self.voicingThreshold = voicingThreshold

        // サンプリング周波数と探索周波数範囲から探索ラグ範囲を確定
        // 60Hz -> 16000 / 60 = 266 サンプル
        // 500Hz -> 16000 / 500 = 32 サンプル
        var lagLow = Int(floor(sampleRate / maxF0))
        if lagLow < 2 {
            lagLow = 2
        }
        self.minLag = lagLow

        var lagHigh = Int(ceil(sampleRate / minF0))
        if (frameLength - 2) < lagHigh {
            lagHigh = frameLength - 2
        }
        self.maxLag = lagHigh
    }

    /// 16kHz PCM 波形全体からフレームごとの実測 F0・有声度・エネルギーを抽出する
    public func track(pcm: [Float]) -> PitchTrackResult {
        if pcm.isEmpty {
            return PitchTrackResult(f0: [], voiced: [], energy: [], frameCount: 0)
        }

        let frameCount = max(1, pcm.count / hopSize)
        var f0List = [Float](repeating: 0.0, count: frameCount)
        var voicedList = [Float](repeating: 0.0, count: frameCount)
        var energyList = [Float](repeating: 0.0, count: frameCount)

        // フレームごとの分析バッファ
        var frameBuf = [Float](repeating: 0.0, count: frameLength)

        var f = 0
        while f < frameCount {
            let startSample = f * hopSize

            // フレームサンプルの収集
            var s = 0
            while s < frameLength {
                let pcmIdx = startSample + s
                if pcmIdx < pcm.count {
                    frameBuf[s] = pcm[pcmIdx]
                } else {
                    frameBuf[s] = 0.0
                }
                s += 1
            }

            let result = analyzeFrame(samples: frameBuf)
            f0List[f] = result.f0
            voicedList[f] = result.voiced
            energyList[f] = result.energy

            f += 1
        }


        // 時間的連続性に基づく F0 メディアン平滑化 (孤立したピッチ誤検出スパイクの除去)
        var smoothF0 = f0List
        if 2 < frameCount {
            var m = 1
            let mEnd = frameCount - 1
            while m < mEnd {
                // 有声区間内でのみ 3 点メディアンを適用
                let v0 = voicedList[m - 1]
                let v1 = voicedList[m]
                let v2 = voicedList[m + 1]
                if 0.5 <= v0 && 0.5 <= v1 && 0.5 <= v2 {
                    let p0 = f0List[m - 1]
                    let p1 = f0List[m]
                    let p2 = f0List[m + 1]
                    // 3 値の中央値選択
                    var med = p1
                    switch true {
                    case (p0 <= p1 && p1 <= p2) || (p2 <= p1 && p1 <= p0):
                        med = p1
                    case (p1 <= p0 && p0 <= p2) || (p2 <= p0 && p0 <= p1):
                        med = p0
                    default:
                        med = p2
                    }
                    smoothF0[m] = med
                }
                m += 1
            }
        }

        return PitchTrackResult(
            f0: smoothF0,
            voiced: voicedList,
            energy: energyList,
            frameCount: frameCount
        )
    }

    /// 単一フレーム（400サンプル）のピッチおよびエネルギー解析
    private func analyzeFrame(samples: [Float]) -> (f0: Float, voiced: Float, energy: Float) {
        // 1. 短時間 RMS エネルギーの計算
        var sumSq: Float = 0.0
        var s = 0
        while s < frameLength {
            let v = samples[s]
            sumSq += v * v
            s += 1
        }
        let rms = sqrtf(sumSq / Float(frameLength))

        // 背景ノイズ・微弱無音区間（約 -46 dBFS 以下）の早期足切り
        // 声帯振動の物理的エネルギーが存在しない無音区間での相関誤検出を防止する
        // なぜ純粋な短時間 RMS（min(1.0, rms)）を返すか:
        // 低レベル DSP 抽出器として人為的な 3 倍増幅を行わず、物理的な信号振幅（0.0〜1.0）を誠実に保持することで、
        // 学習側パイプラインでのピーク正規化や推論側のエネルギー条件付けと数学的にクリーンに連携させるため。
        if rms < 0.005 {
            return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
        }

        // 2. 正規化交差相関 (NCCF: Normalized Cross-Correlation) の計算
        // なぜ生サンプル直接の NCCF を用いるか:
        // Hanning 窓を掛けた信号同士の自己相関を取ると、ラグ tau が大きくなるにつれて
        // 窓の重なり形状によるテーパリング減衰（Window Tapering Bias）が生じ、
        // 相関ピークがより小さいラグ（より高い周波数）側へ数 Hz 偏る。
        // 区間長 (frameLength - tau) の生サンプル同士で Pearson 余弦類似度を直接算出することで、
        // 窓関数による周波数バイアスをゼロにし、真の基本周期を正確に特定する。
        let corrSize = maxLag + 2
        var nccf = [Float](repeating: 0.0, count: corrSize)

        var tau = minLag
        while tau <= maxLag {
            let corrLen = frameLength - tau
            var sumCross: Float = 0.0
            var sumEnergy0: Float = 0.0
            var sumEnergyTau: Float = 0.0

            var n = 0
            while n < corrLen {
                let x0 = samples[n]
                let xTau = samples[n + tau]
                sumCross += x0 * xTau
                sumEnergy0 += x0 * x0
                sumEnergyTau += xTau * xTau
                n += 1
            }

            let denom = sqrtf(sumEnergy0 * sumEnergyTau)
            if 1e-8 < denom {
                nccf[tau] = sumCross / denom
            } else {
                nccf[tau] = 0.0
            }

            tau += 1
        }

        // 3. 局所極大ピーク群の検出と大域最大値の特定
        var peakLags: [Int] = []
        var peakVals: [Float] = []
        var globalMaxPeak: Float = -1.0

        var t = minLag + 1
        while t < maxLag {
            let val = nccf[t]
            let prevVal = nccf[t - 1]
            let nextVal = nccf[t + 1]

            // 局所極大値の検出
            if prevVal < val && nextVal <= val {
                peakLags.append(t)
                peakVals.append(val)
                if globalMaxPeak < val {
                    globalMaxPeak = val
                }
            }
            t += 1
        }

        // 有声判定: 大域最大相関が閾値未満の場合は無声音とする
        if globalMaxPeak < voicingThreshold || peakLags.isEmpty {
            return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
        }

        // 4. 最短有意ラグ（First Significant Peak）による真の基本周期確定
        // なぜ最短ラグを選択するか:
        // 基本周波数 F0 は信号の「最小周期（最短ラグ）」である。
        // 自己相関関数では 2 周期、3 周期（2tau, 3tau）でも同等以上の相関ピークが周期的に出現するため、
        // 単純に最大値だけを選ぶとオクターブ低周波誤認（ピッチ半減や 3 分の 1 化）を引き起こす。
        // 大域的最大相関の 85% 以上の十分な周期強度を持つ最初の（最短ラグの）極大ピークを採用することで、
        // 多重周期のオクターブ跳躍を根絶し、真の基本周波数を確実に特定する。
        let peakThreshold = globalMaxPeak * 0.85
        var bestLag = 0
        var pIdx = 0
        while pIdx < peakLags.count {
            let pLag = peakLags[pIdx]
            let pVal = peakVals[pIdx]
            if peakThreshold <= pVal {
                bestLag = pLag
                break
            }
            pIdx += 1
        }

        if bestLag <= 0 {
            bestLag = peakLags[0]
        }

        // 6. 放物線補間（Parabolic Interpolation）によるサブサンプル精度ラグ推定
        // 離散ラグ bestLag の前後 3 点から極大放物線の頂点 delta を計算
        let y0 = nccf[bestLag - 1]
        let y1 = nccf[bestLag]
        let y2 = nccf[bestLag + 1]
        let denomParabola = (y0 - (2.0 * y1)) + y2

        var delta: Float = 0.0
        if 1e-6 < abs(denomParabola) {
            delta = (y0 - y2) / (2.0 * denomParabola)
            if delta < -0.5 {
                delta = -0.5
            }
            if 0.5 < delta {
                delta = 0.5
            }
        }

        let preciseLag = Float(bestLag) + delta
        if preciseLag <= 0.0 {
            return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
        }

        let extractedF0 = sampleRate / preciseLag

        // 有効ピッチ範囲の最終防壁検査
        if extractedF0 < minF0 || maxF0 < extractedF0 {
            return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
        }

        return (
            f0: extractedF0,
            voiced: 1.0,
            energy: min(1.0, rms)
        )
    }
}
