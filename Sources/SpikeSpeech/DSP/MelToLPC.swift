import Foundation

/// 64ch Mel 特徴量から線形パワースペクトル復元、自己相関算出、
/// および Levinson-Durbin 法による 16次 LPC 係数導出を行う変換器
///
/// SNN音響モデルが生成した聴覚的解像度の高い64チャンネルMel表現から
/// 物理音響モデルである全極AR声道フィルタの共鳴パラメータへ決定論的に変換し、
/// 軽量かつ明瞭な波形合成を可能にする。
public final class MelToLPC: @unchecked Sendable {

    public let melChannels: Int
    public let fftBins: Int
    public let lpcOrder: Int

    /// Mel フィルタバンクの線形再構成重み行列
    private var inversionMatrix: [Float]
    /// 逆コサイン変換テーブル
    private var idctTable: [Float]

    /// 内部作業領域
    private var linearPowerSpectrum: [Float]
    private var autoCorr: [Float]
    private var lpcA: [Float]
    private var lpcPrevA: [Float]
    private var linearMel: [Float]

    /// 初期化
    ///
    /// 実行時における三角関数演算や行列除算を排除し、
    /// 単純な内積演算のみで高速駆動する。
    public init(
        melChannels: Int = AudioConfig.melChannels, // 64
        fftBins: Int = 257,                        // 512 FFT points
        lpcOrder: Int = AudioConfig.lpcOrder,      // 16
        sampleRate: Float = 16000.0
    ) {
        self.melChannels = melChannels
        self.fftBins = fftBins
        self.lpcOrder = lpcOrder

        self.inversionMatrix = [Float](repeating: 0.0, count: fftBins * melChannels)
        self.idctTable = [Float](repeating: 0.0, count: (lpcOrder + 1) * fftBins)
        self.linearPowerSpectrum = [Float](repeating: 0.0, count: fftBins)
        self.autoCorr = [Float](repeating: 0.0, count: lpcOrder + 1)
        self.lpcA = [Float](repeating: 0.0, count: lpcOrder + 1)
        self.lpcPrevA = [Float](repeating: 0.0, count: lpcOrder + 1)
        self.linearMel = [Float](repeating: 0.0, count: melChannels)

        buildInversionMatrix(sampleRate: sampleRate)
        buildIDCTTable()
    }

    /// Mel フィルタバンクの逆写像行列を生成
    ///
    /// 帯域エネルギー保存則を満たし、低域から高域までのパワースペクトル傾斜が
    /// 反転時に不自然に歪むのを防ぐ。
    private func buildInversionMatrix(sampleRate: Float) {
        let minMel: Float = 0.0
        let maxFreq = sampleRate * 0.5
        let maxMel = 2595.0 * log10(1.0 + (maxFreq / 700.0))

        // Mel チャンネルの中心周波数
        var melPoints = [Float](repeating: 0.0, count: melChannels + 2)
        let melStep = (maxMel - minMel) / Float(melChannels + 1)
        var m = 0
        while m < melChannels + 2 {
            melPoints[m] = minMel + (Float(m) * melStep)
            m += 1
        }

        // 周波数ビンごとの重み計算
        var k = 0
        while k < fftBins {
            let freq = (Float(k) * maxFreq) / Float(fftBins - 1)
            let mel = 2595.0 * log10(1.0 + (freq / 700.0))

            var ch = 0
            var weightSum: Float = 0.0
            while ch < melChannels {
                let left = melPoints[ch]
                let center = melPoints[ch + 1]
                let right = melPoints[ch + 2]

                var w: Float = 0.0
                if left <= mel {
                    if mel <= center {
                        let span = center - left
                        if 1e-6 < span {
                            w = (mel - left) / span
                        }
                    } else {
                        if mel <= right {
                            let span = right - center
                            if 1e-6 < span {
                                w = (right - mel) / span
                            }
                        }
                    }
                }
                inversionMatrix[(k * melChannels) + ch] = w
                weightSum += w
                ch += 1
            }

            // フィルタの重なり部分でエネルギーが二重加算されスペクトル包絡に凹凸が生じるのを防ぐ正規化。
            if 1e-6 < weightSum {
                let invSum = 1.0 / weightSum
                ch = 0
                while ch < melChannels {
                    inversionMatrix[(k * melChannels) + ch] *= invSum
                    ch += 1
                }
            }
            k += 1
        }
    }

    /// 逆コサイン変換テーブルの作成
    ///
    /// パワースペクトルの逆変換が自己相関関数になるヴィーナー・ヒンチンの定理に基づき、
    /// 時間領域信号を経由せずスペクトルから直接自己相関を計算する。
    private func buildIDCTTable() {
        let normFactor: Float = 1.0 / Float(fftBins)
        var tau = 0
        while tau <= lpcOrder {
            var k = 0
            while k < fftBins {
                let angle = (Float.pi * Float(k * tau)) / Float(fftBins - 1)
                idctTable[(tau * fftBins) + k] = cos(angle) * normFactor
                k += 1
            }
            tau += 1
        }
    }

    /// 64ch Mel 特徴量から LPC 係数とゲインを計算
    ///
    /// SNN音響モデルの学習損失で標準的な対数スケールと
    /// ボコーダーの物理エネルギー演算における線形スケールの双方に対応する。
    public func convert(
        mel: [Float],
        isLogMel: Bool = true,
        outCoeffs: inout [Float]
    ) -> Float {
        // 音響モデルの勾配発散等により非有限値が入力された際に不正値の伝播を防ぎ、安全な無音を返す。
        var melIdx = 0
        let inMelCount = mel.count
        while melIdx < inMelCount {
            if mel[melIdx].isFinite != true {
                var i = 0
                while i < lpcOrder {
                    if i < outCoeffs.count {
                        outCoeffs[i] = 0.0
                    }
                    i += 1
                }
                return 0.0
            }
            melIdx += 1
        }

        // 1. 必要に応じて指数変換し、線形 Mel エネルギーを確保
        var ch = 0
        while ch < melChannels {
            let v = mel[ch]
            if isLogMel {
                // 極端な出力値による浮動小数点のオーバーフローやアンダーフローを防止する。
                var clamped = v
                if clamped < -20.0 {
                    clamped = -20.0
                }
                if 20.0 < clamped {
                    clamped = 20.0
                }
                linearMel[ch] = exp(clamped)
            } else {
                if v < 0.0 {
                    linearMel[ch] = 0.0
                } else {
                    linearMel[ch] = v
                }
            }
            ch += 1
        }

        // 2. 線形パワースペクトルの復元
        linearMel.withUnsafeBufferPointer { melBuf in
            let melPtr = melBuf.baseAddress!
            inversionMatrix.withUnsafeBufferPointer { invBuf in
                let invPtr = invBuf.baseAddress!
                linearPowerSpectrum.withUnsafeMutableBufferPointer { specBuf in
                    let specPtr = specBuf.baseAddress!
                    var k = 0
                    while k < fftBins {
                        let rowPtr = invPtr.advanced(by: k * melChannels)
                        let val = VectorOperations.dotProduct(a: rowPtr, b: melPtr, count: melChannels)
                        // スペクトルの負値およびアルゴリズム内部でのゼロ除算を防止する微小フロア。
                        if val < 1e-8 {
                            specPtr[k] = 1e-8
                        } else {
                            specPtr[k] = val
                        }
                        k += 1
                    }
                }
            }
        }

        // 3. 逆コサイン変換による自己相関の導出
        linearPowerSpectrum.withUnsafeBufferPointer { specBuf in
            let specPtr = specBuf.baseAddress!
            idctTable.withUnsafeBufferPointer { tableBuf in
                let tablePtr = tableBuf.baseAddress!
                autoCorr.withUnsafeMutableBufferPointer { rBuf in
                    let rPtr = rBuf.baseAddress!
                    var tau = 0
                    while tau <= lpcOrder {
                        let rowPtr = tablePtr.advanced(by: tau * fftBins)
                        rPtr[tau] = VectorOperations.dotProduct(a: specPtr, b: rowPtr, count: fftBins)
                        tau += 1
                    }
                }
            }
        }

        let r0 = autoCorr[0]
        if r0 < 1e-10 {
            // 無音区間で不要な発振を起こさず完全な無音状態を確定させる。
            var i = 0
            while i < lpcOrder {
                outCoeffs[i] = 0.0
                i += 1
            }
            return 0.0
        }

        // 単一周波数ピーク等の特異スペクトルによって自己相関行列が病的に縮退し、反射係数が過剰接近して発散するのを防ぐノイズフロア付加。
        autoCorr[0] *= 1.002

        // 4. Levinson-Durbin アルゴリズム
        var e = autoCorr[0]
        var i = 1
        while i <= lpcOrder {
            lpcA[i] = 0.0
            lpcPrevA[i] = 0.0
            i += 1
        }

        i = 1
        while i <= lpcOrder {
            var s: Float = 0.0
            var j = 1
            while j < i {
                s += lpcA[j] * autoCorr[i - j]
                j += 1
            }

            var ki = (autoCorr[i] - s) / e

            // 全極フィルタの極が単位円の境界へ到達して無限大発振を起こすのを防ぐため、反射係数を安定領域にクランプする。
            if ki < -0.999 {
                ki = -0.999
            }
            if 0.999 < ki {
                ki = 0.999
            }

            lpcPrevA.withUnsafeMutableBufferPointer { prevBuf in
                lpcA.withUnsafeBufferPointer { curBuf in
                    prevBuf.baseAddress!.update(from: curBuf.baseAddress!, count: lpcOrder + 1)
                }
            }

            lpcA[i] = ki
            j = 1
            while j < i {
                lpcA[j] = lpcPrevA[j] - (ki * lpcPrevA[i - j])
                j += 1
            }

            e = e * (1.0 - (ki * ki))
            if e < 1e-10 {
                e = 1e-10
            }
            i += 1
        }

        // 5. 出力係数の書き込みと帯域幅拡大
        // 声道共鳴フィルタの極を単位円の内側へ均等に引き込み、急峻すぎる共鳴によるリンギングや係数の過大化を抑制する帯域幅拡大。
        let gamma: Float = 0.98
        var currentGamma: Float = 1.0
        i = 1
        while i <= lpcOrder {
            currentGamma *= gamma
            outCoeffs[i - 1] = lpcA[i] * currentGamma
            i += 1
        }

        let gain = sqrt(e)
        return gain
    }
}
