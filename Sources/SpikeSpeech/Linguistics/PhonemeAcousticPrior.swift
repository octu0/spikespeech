import Foundation

/// 音響音声学に基づく日本語 64 音素の標準対数 Mel スペクトル事前アンカー (Phoneme Acoustic Prior)
///
/// SNN 音響モデルがゼロ学習時や過渡期においても安定したフォルマント共鳴を形成できるよう、
/// 各音素固有の標準共鳴スペクトル（母音の F1/F2 フォルマント、鼻音共鳴、無音フロア）を事前知識として保持する。
/// SNN はこの事前スペクトルからの「残差（Residual）」を学習することで、直流オフセットへの勾配消費を排除し、
/// 話者個性や肉声の微細なニュアンスの最適化に 100% 専念できる。
public final class PhonemeAcousticPrior: @unchecked Sendable {

    public let melChannels: Int
    public let vocabSize: Int
    public let tract: VocalTract

    /// 音素 ID ごとの事前対数 Mel スペクトルテーブル [vocabSize * melChannels]
    private let priorTable: [Float]

    public init(
        melChannels: Int = AudioConfig.melChannels, // 64
        vocabSize: Int = 64,
        sampleRate: Float = 16000.0,
        fftBins: Int = 257,
        tract: VocalTract = VocalTract()
    ) {
        self.melChannels = melChannels
        self.vocabSize = vocabSize
        self.tract = tract

        // 1. Mel フィルタバンク中心周波数・帯域重みの算出
        let maxFreq = sampleRate * 0.5
        let maxMel = 2595.0 * log10(1.0 + (maxFreq / 700.0))
        var melPoints = [Float](repeating: 0.0, count: melChannels + 2)
        let melStep = maxMel / Float(melChannels + 1)
        var mp = 0
        while mp < melChannels + 2 {
            melPoints[mp] = Float(mp) * melStep
            mp += 1
        }

        var filterbank = [Float](repeating: 0.0, count: fftBins * melChannels)
        var binIdx = 0
        while binIdx < fftBins {
            let freq = (Float(binIdx) * maxFreq) / Float(fftBins - 1)
            let mel = 2595.0 * log10(1.0 + (freq / 700.0))

            var ch = 0
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
                filterbank[(binIdx * melChannels) + ch] = w
                ch += 1
            }
            binIdx += 1
        }

        // 2. 音素ごとの線形パワースペクトル合成と対数 Mel 変換
        // 実音声の収録環境ノイズフロア（対数 Mel 平均約 -7.88）と厳密に整合させ、
        // 無音区間での残差ペナルティをゼロ化して発話区間の学習効率を最大化する。
        let silenceEnergyFloor: Float = 3.8e-4
        let silenceLogMelFloor: Float = logf(silenceEnergyFloor)
        var table = [Float](repeating: silenceLogMelFloor, count: vocabSize * melChannels)

        let vtl = tract.lengthScale
        let bwScale = tract.bandwidthScale

        var pId = 0
        while pId < vocabSize {
            var powerSpec = [Float](repeating: silenceEnergyFloor, count: fftBins)

            // 音素固有のフォルマント共鳴パラメータ定義（Hz 領域での真の声道長スケーリング VTLN を適用）
            // なぜ Mel ビン伸縮ではなく Hz 領域でスケールするか:
            // Fant の音響音声学モデルに基づき、声道長比 α に応じて共鳴ピーク（F1, F2, F3）が周波数軸上で平行移動し、
            // 母音の音韻同一性（F2/F1 比）を 100% 保存しながら話者声道のサイズ差のみを正確に表現するため。
            switch pId {
            case 5: // /a/ (あ): F1=800, F2=1300, F3=2600
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (800.0 * vtl, 90.0 * bwScale, 1.0),
                        (1300.0 * vtl, 110.0 * bwScale, 0.6),
                        (2600.0 * vtl, 160.0 * bwScale, 0.18)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 6: // /i/ (い): F1=300, F2=2300, F3=3000
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (300.0 * vtl, 70.0 * bwScale, 0.9),
                        (2300.0 * vtl, 130.0 * bwScale, 0.5),
                        (3000.0 * vtl, 180.0 * bwScale, 0.18)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 7: // /u/ (う): F1=360, F2=1200, F3=2400
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (360.0 * vtl, 70.0 * bwScale, 0.85),
                        (1200.0 * vtl, 100.0 * bwScale, 0.4),
                        (2400.0 * vtl, 150.0 * bwScale, 0.12)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 8: // /e/ (え): F1=500, F2=1900, F3=2600
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (500.0 * vtl, 80.0 * bwScale, 0.95),
                        (1900.0 * vtl, 120.0 * bwScale, 0.5),
                        (2600.0 * vtl, 160.0 * bwScale, 0.18)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 9: // /o/ (お): F1=500, F2=900, F3=2500
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (500.0 * vtl, 80.0 * bwScale, 0.95),
                        (900.0 * vtl, 90.0 * bwScale, 0.65),
                        (2500.0 * vtl, 150.0 * bwScale, 0.12)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 10, 11, 12, 14, 27, 28, 29: // 無声子音 (k, s, t, h, sh, ch, ts): 高域摩擦ノイズ
                let cutoffFreq = 2000.0 * vtl
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    if cutoffFreq <= freq {
                        powerSpec[b] = 0.05 * (freq / maxFreq)
                    }
                    b += 1
                }
            case 13, 15, 24: // 鼻音 (n, m, N): 低域共鳴集中
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (250.0 * vtl, 100.0 * bwScale, 0.7),
                        (1000.0 * vtl, 200.0 * bwScale, 0.15)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.7
                )
            case 16, 17, 18, 19, 20, 21, 22, 23, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38: // 有声子音・拗音・長音: 一般有声共鳴
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (450.0 * vtl, 120.0 * bwScale, 0.6),
                        (1600.0 * vtl, 180.0 * bwScale, 0.25)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.8
                )
            default:
                // 無音・休止 (0, 1, 2, 3, 4, 25, 39..63): 実音声フロアのまま
                break
            }

            // Mel フィルタバンク積算と対数化
            var c = 0
            while c < melChannels {
                var bandEnergy: Float = 0.0
                var b = 0
                while b < fftBins {
                    bandEnergy += powerSpec[b] * filterbank[(b * melChannels) + c]
                    b += 1
                }
                var clamped = bandEnergy
                if clamped < silenceEnergyFloor {
                    clamped = silenceEnergyFloor
                }
                table[(pId * melChannels) + c] = logf(clamped)
                c += 1
            }

            pId += 1
        }

        self.priorTable = table
    }

    /// 二次共鳴フィルタ伝達関数に基づくパワースペクトルの合成
    ///
    /// 音響音声学における声道音響伝達特性を忠実に再現し、
    /// フォルマント周波数近傍の滑らかな共鳴山と声門気流のスペクトル傾斜を付与する。
    private static func synthesizeFormantSpectrum(
        powerSpec: inout [Float],
        formants: [(freq: Float, bw: Float, gain: Float)],
        maxFreq: Float,
        fftBins: Int,
        tiltDecay: Float
    ) {
        var b = 0
        while b < fftBins {
            let f = (Float(b) * maxFreq) / Float(fftBins - 1)
            // 声門励起のスペクトル傾斜 (-6dB/oct)
            let fNorm = f / 100.0
            let tilt = 1.0 / (1.0 + (fNorm * (1.0 - tiltDecay)))

            var resSum: Float = 0.001
            var fIdx = 0
            while fIdx < formants.count {
                let fmt = formants[fIdx]
                let f0 = fmt.freq
                // 帯域幅が極端に小さい場合の 0/0 による NaN 発生を二重防護するため下限 10.0 Hz にクランプ
                var safeBw = fmt.bw
                if safeBw < 10.0 {
                    safeBw = 10.0
                }
                let g = fmt.gain
                // Cauchy-Lorentz 共鳴形状
                let diff = f - f0
                let denom = 1.0 + ((diff * diff) / ((safeBw * 0.5) * (safeBw * 0.5)))
                resSum += g / denom
                fIdx += 1
            }

            // 実音声の母音パワースペクトル水準（Mel対数値で +1.0〜+3.5）と整合させるためのエネルギースケール
            let speechPowerScale: Float = 15.0
            powerSpec[b] = resSum * tilt * speechPowerScale
            b += 1
        }
    }

    /// 指定音素 ID の事前対数 Mel スペクトル（64次元）を作業バッファへ高速一括転送する
    @inline(__always)
    public func copyPriorMel(phoneId: Int, dst: UnsafeMutablePointer<Float>) {
        var safeId = phoneId
        if safeId < 0 {
            safeId = 0
        }
        if vocabSize <= safeId {
            safeId = 0
        }
        let offset = safeId * melChannels
        priorTable.withUnsafeBufferPointer { srcBuf in
            dst.update(from: srcBuf.baseAddress!.advanced(by: offset), count: melChannels)
        }
    }

    /// 指定音素 ID の事前対数 Mel スペクトル配列を取得する
    public func getPriorMel(phoneId: Int) -> [Float] {
        var result = [Float](repeating: 0.0, count: melChannels)
        result.withUnsafeMutableBufferPointer { buf in
            copyPriorMel(phoneId: phoneId, dst: buf.baseAddress!)
        }
        return result
    }
}
