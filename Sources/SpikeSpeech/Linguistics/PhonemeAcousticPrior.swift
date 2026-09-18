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
        tract: VocalTract = VocalTract(),
        baseF0: Float = 220.0
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
            case 5: // /a/ (あ): F0=baseF0, F1=850, F2=1350, F3=2850, F4=3800, F5=4700
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (baseF0, 40.0 * bwScale, 0.45),
                        (850.0 * vtl, 80.0 * bwScale, 1.60),
                        (1350.0 * vtl, 90.0 * bwScale, 0.60),
                        (2850.0 * vtl, 140.0 * bwScale, 0.25),
                        (3800.0 * vtl, 180.0 * bwScale, 0.10),
                        (4700.0 * vtl, 220.0 * bwScale, 0.05)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 6: // /i/ (い): F0=baseF0, F1=280, F2=2350, F3=3200, F4=4100
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (baseF0, 35.0 * bwScale, 0.45),
                        (280.0 * vtl, 60.0 * bwScale, 1.50),
                        (2350.0 * vtl, 110.0 * bwScale, 0.60),
                        (3200.0 * vtl, 150.0 * bwScale, 0.25),
                        (4100.0 * vtl, 200.0 * bwScale, 0.10)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 7: // /u/ (う): F0=baseF0, F1=340, F2=1350, F3=2450, F4=3600
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (baseF0, 40.0 * bwScale, 0.45),
                        (340.0 * vtl, 65.0 * bwScale, 1.30),
                        (1350.0 * vtl, 95.0 * bwScale, 0.50),
                        (2450.0 * vtl, 140.0 * bwScale, 0.22),
                        (3600.0 * vtl, 180.0 * bwScale, 0.08)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 8: // /e/ (え): F0=baseF0, F1=500, F2=1950, F3=2800, F4=3700
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (baseF0, 40.0 * bwScale, 0.45),
                        (500.0 * vtl, 75.0 * bwScale, 1.50),
                        (1950.0 * vtl, 110.0 * bwScale, 0.60),
                        (2800.0 * vtl, 150.0 * bwScale, 0.25),
                        (3700.0 * vtl, 190.0 * bwScale, 0.10)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 9: // /o/ (お): F0=baseF0, F1=500, F2=950, F3=2600, F4=3600
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (baseF0, 40.0 * bwScale, 0.45),
                        (500.0 * vtl, 75.0 * bwScale, 1.50),
                        (950.0 * vtl, 85.0 * bwScale, 0.70),
                        (2600.0 * vtl, 140.0 * bwScale, 0.22),
                        (3600.0 * vtl, 180.0 * bwScale, 0.08)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.85
                )
            case 10: // /k/: 軟口蓋破裂音解放バースト (Release Burst)
                let kCenter = 2600.0 * vtl
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    let diff = freq - kCenter
                    let normDiff = (diff * diff) / (350.0 * 350.0)
                    var burst = 0.45 / (1.0 + (normDiff * normDiff))
                    if 3500.0 * vtl <= freq {
                        let norm = (freq - (3500.0 * vtl)) / (maxFreq - (3500.0 * vtl))
                        burst += 0.20 * norm
                    }
                    powerSpec[b] = burst
                    b += 1
                }
            case 11: // /s/: 歯茎摩擦音（4kHz〜8kHz の強力なヒスノイズ）
                let sCutoff = 3800.0 * vtl
                let sSpan = maxFreq - sCutoff
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    if sCutoff <= freq {
                        let norm = (freq - sCutoff) / sSpan
                        powerSpec[b] = 0.50 * norm
                    }
                    b += 1
                }
            case 12: // /t/: 歯茎破裂音解放バースト (Release Burst)
                let tCenter = 4200.0 * vtl
                let tCutoff = 3500.0 * vtl
                let tSpan = maxFreq - tCutoff
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    let diff = freq - tCenter
                    let normDiff = (diff * diff) / (500.0 * 500.0)
                    var burst = 0.50 / (1.0 + (normDiff * normDiff))
                    if tCutoff <= freq {
                        let norm = (freq - tCutoff) / tSpan
                        burst += 0.25 * norm
                    }
                    powerSpec[b] = burst
                    b += 1
                }
            case 14: // /h/: 声門摩擦音（気息音）
                let hCenter = 3000.0 * vtl
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    if 1200.0 * vtl <= freq && freq <= 5500.0 * vtl {
                        let dev = abs(freq - hCenter) / (3000.0 * vtl)
                        powerSpec[b] = 0.25 * (1.0 - dev)
                    }
                    b += 1
                }
            case 23: // /p/: 両唇破裂音解放バースト
                let pCenter = 1200.0 * vtl
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    let diff = freq - pCenter
                    let normDiff = (diff * diff) / (400.0 * 400.0)
                    powerSpec[b] = 0.40 / (1.0 + (normDiff * normDiff))
                    b += 1
                }
            case 27: // /sh/: 歯茎硬口蓋摩擦音
                let shCutoff = 2600.0 * vtl
                let shSpan = maxFreq - shCutoff
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    if shCutoff <= freq {
                        let norm = (freq - shCutoff) / shSpan
                        powerSpec[b] = 0.55 * norm
                    }
                    b += 1
                }
            case 28, 29: // /ch/, /ts/: 破擦音
                let affCutoff = 2800.0 * vtl
                let affSpan = maxFreq - affCutoff
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    if affCutoff <= freq {
                        let norm = (freq - affCutoff) / affSpan
                        powerSpec[b] = 0.50 * norm
                    }
                    b += 1
                }
            case 13, 15, 24: // 鼻音 (n, m, N): 低域共鳴集中
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (baseF0, 45.0 * bwScale, 0.60),
                        (280.0 * vtl, 60.0 * bwScale, 1.30),
                        (1000.0 * vtl, 180.0 * bwScale, 0.10)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.65
                )
            case 20, 36: // /z/, /j/: 有声摩擦音
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (baseF0, 50.0 * bwScale, 0.50),
                        (450.0 * vtl, 110.0 * bwScale, 0.45),
                        (1800.0 * vtl, 160.0 * bwScale, 0.20)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.80
                )
                let zCutoff = 3500.0 * vtl
                let zSpan = maxFreq - zCutoff
                var b = 0
                while b < fftBins {
                    let freq = (Float(b) * maxFreq) / Float(fftBins - 1)
                    if zCutoff <= freq {
                        powerSpec[b] += 0.25 * ((freq - zCutoff) / zSpan)
                    }
                    b += 1
                }
            case 16, 17, 18, 19, 21, 22, 26, 30, 31, 32, 33, 34, 35, 37, 38: // 有声子音・拗音・長音: 一般有声共鳴
                Self.synthesizeFormantSpectrum(
                    powerSpec: &powerSpec,
                    formants: [
                        (baseF0, 50.0 * bwScale, 0.55),
                        (450.0 * vtl, 110.0 * bwScale, 0.55),
                        (1600.0 * vtl, 160.0 * bwScale, 0.22)
                    ],
                    maxFreq: maxFreq,
                    fftBins: fftBins,
                    tiltDecay: 0.80
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

    /// 音響音声学に基づくパワースペクトルの合成（フォルマント急峻共鳴および深谷 antiresonance 特性）
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

            var resSum: Float = 0.00005
            var fIdx = 0
            while fIdx < formants.count {
                let fmt = formants[fIdx]
                let f0 = fmt.freq
                var safeBw = fmt.bw
                if safeBw < 10.0 {
                    safeBw = 10.0
                }
                let g = fmt.gain
                let diff = f - f0
                let halfBw = safeBw * 0.5
                let normDiff = (diff * diff) / (halfBw * halfBw)
                // 4次急峻共鳴減衰により、フォルマント山と谷（ゼロ点）の深さ（-6.0〜-7.5dB）を正確に再現
                let denom = (1.0 + normDiff) * (1.0 + normDiff)
                resSum += g / denom
                fIdx += 1
            }

            // 母音パワースペクトル水準の調整
            let speechPowerScale: Float = 12.0
            var pVal = resSum * tilt * speechPowerScale

            // 3500Hz 以上の有声帯域における自然な声帯呼気性気息成分（Aspiration Noise）
            if 3500.0 <= f {
                let normF = (f - 3500.0) / (maxFreq - 3500.0)
                let breath = 0.016 * (1.0 - (normF * 0.35))
                pVal += breath
            }

            powerSpec[b] = pVal
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
