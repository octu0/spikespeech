import Foundation

/// 基準音声データ自動生成器
///
/// 外部音声ファイルへの依存を排除し、
/// CI環境やローカルマシン上で完全再現可能な基準音声およびテスト波形を
/// 音響生理学モデルから決定論的に合成する。
public final class SyntheticAudioGenerator: @unchecked Sendable {

    /// 日本語基本 5 母音
    public enum Vowel: String, Sendable, CaseIterable {
        case a = "a"
        case i = "i"
        case u = "u"
        case e = "e"
        case o = "o"
    }

    /// ホルマント共鳴設定
    public struct FormantConfig: Sendable, Equatable {
        public let f1: Float
        public let b1: Float
        public let f2: Float
        public let b2: Float
        public let f3: Float
        public let b3: Float
        public let f4: Float
        public let b4: Float

        public init(
            f1: Float, b1: Float,
            f2: Float, b2: Float,
            f3: Float, b3: Float,
            f4: Float, b4: Float
        ) {
            self.f1 = f1
            self.b1 = b1
            self.f2 = f2
            self.b2 = b2
            self.f3 = f3
            self.b3 = b3
            self.f4 = f4
            self.b4 = b4
        }
    }

    public let sampleRate: Float
    private var rngState: UInt64 = 5489

    /// 初期化
    ///
    /// 音響設定の標準サンプリング周波数と整合させ、
    /// SNN音響モデルおよびボコーダーと同一のナイキスト周波数で動作させる。
    public init(sampleRate: Float = Float(AudioConfig.sampleRate)) {
        self.sampleRate = sampleRate
    }

    /// 日本語 5 母音の生理学的標準ホルマント周波数テーブルを取得
    ///
    /// 日本語母音の第一ホルマント、第二ホルマント、
    /// および第三と第四ホルマントを再現し、明瞭な母音差を形成する。
    public func formantConfig(for vowel: Vowel) -> FormantConfig {
        switch vowel {
        case .a:
            // 低舌および後舌母音
            return FormantConfig(
                f1: 800.0, b1: 80.0,
                f2: 1300.0, b2: 100.0,
                f3: 2600.0, b3: 120.0,
                f4: 3500.0, b4: 150.0
            )
        case .i:
            // 高舌および前舌母音
            return FormantConfig(
                f1: 300.0, b1: 60.0,
                f2: 2300.0, b2: 100.0,
                f3: 3000.0, b3: 150.0,
                f4: 3800.0, b4: 200.0
            )
        case .u:
            // 高舌および後舌円唇母音
            return FormantConfig(
                f1: 350.0, b1: 70.0,
                f2: 1200.0, b2: 100.0,
                f3: 2500.0, b3: 140.0,
                f4: 3600.0, b4: 180.0
            )
        case .e:
            // 中舌および前舌母音
            return FormantConfig(
                f1: 500.0, b1: 70.0,
                f2: 1900.0, b2: 90.0,
                f3: 2700.0, b3: 130.0,
                f4: 3700.0, b4: 180.0
            )
        case .o:
            // 中舌および後舌円唇母音
            return FormantConfig(
                f1: 500.0, b1: 70.0,
                f2: 900.0, b2: 80.0,
                f3: 2600.0, b3: 120.0,
                f4: 3500.0, b4: 150.0
            )
        }
    }

    /// 高速 Xorshift64 疑似乱数生成器
    ///
    /// テスト実行ごとの乱数シード差による変動を排除し、
    /// 同一のノイズ波形を確実に再現可能にする。
    @inline(__always)
    private func nextRandomFloat() -> Float {
        rngState ^= (rngState << 13)
        rngState ^= (rngState >> 7)
        rngState ^= (rngState << 17)
        let u32 = UInt32(truncatingIfNeeded: rngState)
        return (Float(u32) * (2.0 / 4294967295.0)) - 1.0
    }

    /// 2次 IIR デジタル共鳴器
    ///
    /// 声道の音響管結合モデルに基づき、
    /// 直列配置した共鳴器によってホルマント周波数間の自然なゼロ点と減衰特性を再現する。
    private final class ResonatorStage {
        private let a1: Float
        private let a2: Float
        private let b0: Float
        private var y1: Float = 0.0
        private var y2: Float = 0.0

        init(freq: Float, bandwidth: Float, sampleRate: Float) {
            // 指定した帯域幅をインパルス不変系に対応させる極半径計算。
            let r = exp(-Float.pi * bandwidth / sampleRate)
            let theta = 2.0 * Float.pi * freq / sampleRate
            self.a1 = 2.0 * r * cos(theta)
            self.a2 = -(r * r)
            // 直列カスケード共鳴器において直流ゲインを正規化し、多段接続時の急激な減衰を防ぎつつ共鳴ピークを強調する。
            self.b0 = 1.0 - (2.0 * r * cos(theta)) + (r * r)
        }

        @inline(__always)
        func step(input: Float) -> Float {
            let y0 = (b0 * input) + (a1 * y1) + (a2 * y2)
            y2 = y1
            y1 = y0
            return y0
        }

        func reset() {
            y1 = 0.0
            y2 = 0.0
        }
    }

    // MARK: - 1. 5母音ホルマント合成

    /// 日本語母音ホルマント音声を合成
    ///
    /// 正弦波加算合成では声門開閉の倍音構造が失われるため、
    /// 非対称パルス気流を声道フィルタの4段カスケード共鳴器で濾過して自然な母音波形を得る。
    public func generateVowel(
        vowel: Vowel,
        durationSeconds: Float = 0.5,
        f0: Float = 130.0
    ) -> [Float] {
        if durationSeconds <= 0.0 {
            return []
        }
        let totalSamples = Int(round(durationSeconds * sampleRate))
        if totalSamples <= 0 {
            return []
        }

        let cfg = formantConfig(for: vowel)
        let r1 = ResonatorStage(freq: cfg.f1, bandwidth: cfg.b1, sampleRate: sampleRate)
        let r2 = ResonatorStage(freq: cfg.f2, bandwidth: cfg.b2, sampleRate: sampleRate)
        let r3 = ResonatorStage(freq: cfg.f3, bandwidth: cfg.b3, sampleRate: sampleRate)
        let r4 = ResonatorStage(freq: cfg.f4, bandwidth: cfg.b4, sampleRate: sampleRate)

        let pulseGenerator = RosenbergPulse(sampleRate: sampleRate)
        var output = [Float](repeating: 0.0, count: totalSamples)

        // 音の開始および終了時における急峻なステップ変化によるクリックノイズを防止する。
        let rampSamples = min(160, totalSamples / 4)

        var n = 0
        while n < totalSamples {
            let glottal = pulseGenerator.nextSample(f0: f0, removeDC: true)
            let s1 = r1.step(input: glottal)
            let s2 = r2.step(input: s1)
            let s3 = r3.step(input: s2)
            var s4 = r4.step(input: s3)

            // エンベロープ窓の乗算
            if n < rampSamples {
                let gain = Float(n) / Float(rampSamples)
                s4 *= gain
            }
            let tailStart = totalSamples - rampSamples
            if tailStart <= n {
                let gain = Float(totalSamples - 1 - n) / Float(rampSamples)
                s4 *= gain
            }

            // フィルタの過渡応答による振幅超過を安全に圧縮する。
            let absVal = abs(s4)
            let limited: Float
            if absVal <= 0.8 {
                limited = s4
            } else {
                let excess = absVal - 0.8
                let compressed = 0.8 + (0.2 * tanh(excess * 5.0))
                if s4 < 0.0 {
                    limited = -compressed
                } else {
                    limited = compressed
                }
            }

            output[n] = limited
            n += 1
        }

        return output
    }

    // MARK: - 2. 周波数チャープ波

    /// 直線周波数スイープを合成
    ///
    /// 20Hzから8000Hzまでの全周波数成分を時系列で走査し、
    /// Melフィルタバンク全チャンネルの受容野応答および群遅延特性を網羅検証する。
    public func generateChirp(
        startFreq: Float = 20.0,
        endFreq: Float = 8000.0,
        durationSeconds: Float = 0.5
    ) -> [Float] {
        if durationSeconds <= 0.0 {
            return []
        }
        let totalSamples = Int(round(durationSeconds * sampleRate))
        if totalSamples <= 0 {
            return []
        }

        var output = [Float](repeating: 0.0, count: totalSamples)
        let twoPi = 2.0 * Float.pi
        let deltaF = endFreq - startFreq

        var n = 0
        while n < totalSamples {
            let t = Float(n) / sampleRate
            let normalizedT = t / durationSeconds
            // 瞬時周波数が線形変化する数学的連続位相を厳密に生成する。
            let phase = twoPi * ((startFreq * t) + (0.5 * deltaF * normalizedT * t))
            output[n] = sin(phase) * 0.8
            n += 1
        }

        return output
    }

    // MARK: - 3. インパルス列

    /// 周期インパルス列を合成
    ///
    /// ピッチ周期ごとに厳密にデルタ関数を配置し、
    /// ボコーダーや自己相関アルゴリズムのインパルス応答および位相遅延を精密測定する。
    public func generateImpulseTrain(
        f0: Float = 100.0,
        durationSeconds: Float = 0.5
    ) -> [Float] {
        if durationSeconds <= 0.0 || f0 <= 0.0 {
            return []
        }
        let totalSamples = Int(round(durationSeconds * sampleRate))
        if totalSamples <= 0 {
            return []
        }

        var output = [Float](repeating: 0.0, count: totalSamples)
        let periodSamples = max(1, Int(round(sampleRate / f0)))

        var n = 0
        while n < totalSamples {
            if (n % periodSamples) == 0 {
                output[n] = 1.0
            } else {
                output[n] = 0.0
            }
            n += 1
        }

        return output
    }

    // MARK: - 4. ホワイトノイズ

    /// ホワイトノイズ波形を合成
    ///
    /// 無声子音の励起信号特性、および平坦なパワースペクトル入力時の
    /// フィルタ伝達関数の逆同定テストを実施する。
    public func generateWhiteNoise(
        durationSeconds: Float = 0.5,
        amplitude: Float = 0.5
    ) -> [Float] {
        if durationSeconds <= 0.0 {
            return []
        }
        let totalSamples = Int(round(durationSeconds * sampleRate))
        if totalSamples <= 0 {
            return []
        }

        var output = [Float](repeating: 0.0, count: totalSamples)
        let clampedAmp = min(1.0, max(0.0, amplitude))

        var n = 0
        while n < totalSamples {
            output[n] = nextRandomFloat() * clampedAmp
            n += 1
        }

        return output
    }

    // MARK: - 5. 擬似日本語フレーズ数式合成

    /// 日本語テキストに対応する擬似音声波形を数式合成
    ///
    /// 学習ループにおいて外部音声ファイルを読み込まずに
    /// 音素、母音、子音、ポーズの時系列変化を持つ正解音声信号をオンデマンド生成する。
    public func generatePhrase(
        phrase: String,
        speed: Float = 1.0,
        baseF0: Float = 130.0
    ) -> [Float] {
        if phrase.isEmpty {
            return []
        }

        let normalizer = TextNormalizer()
        let prosody = ProsodyModel()
        let vocab = PhonemeVocabulary()
        let regulator = LengthRegulator()

        let features = regulator.processText(
            text: phrase,
            normalizer: normalizer,
            prosodyModel: prosody,
            vocabulary: vocab,
            speedFactor: speed
        )

        let totalFrames = features.totalFrames
        if totalFrames <= 0 {
            return []
        }

        let hopSize = AudioConfig.hopSize // 160 samples (10ms)
        let totalSamples = totalFrames * hopSize
        var output = [Float](repeating: 0.0, count: totalSamples)

        // 音素ごとのサンプル区間を展開
        var currentSampleOffset = 0
        let phoneCount = features.phoneIds.count

        var p = 0
        while p < phoneCount {
            let phoneId = Int(features.phoneIds[p])
            let durationFrames = Int(features.durations[p])
            let phoneSamples = durationFrames * hopSize

            if phoneSamples <= 0 {
                p += 1
                continue
            }

            // 現在のフレームインデックスから F0 と有声度を抽出
            let frameIdx = min(currentSampleOffset / hopSize, totalFrames - 1)
            var currentF0 = features.f0Contour[frameIdx]
            if currentF0 <= 0.0 {
                currentF0 = baseF0
            }

            // 音素種別に応じた合成波形生成
            var phoneWave: [Float] = []

            switch phoneId {
            case 5: // "a"
                phoneWave = generateVowel(vowel: .a, durationSeconds: Float(phoneSamples) / sampleRate, f0: currentF0)
            case 6: // "i"
                phoneWave = generateVowel(vowel: .i, durationSeconds: Float(phoneSamples) / sampleRate, f0: currentF0)
            case 7: // "u"
                phoneWave = generateVowel(vowel: .u, durationSeconds: Float(phoneSamples) / sampleRate, f0: currentF0)
            case 8: // "e"
                phoneWave = generateVowel(vowel: .e, durationSeconds: Float(phoneSamples) / sampleRate, f0: currentF0)
            case 9: // "o"
                phoneWave = generateVowel(vowel: .o, durationSeconds: Float(phoneSamples) / sampleRate, f0: currentF0)
            case 11, 27, 29: // "s", "sh", "ts" (無声摩擦音: 高域ノイズ)
                phoneWave = generateWhiteNoise(durationSeconds: Float(phoneSamples) / sampleRate, amplitude: 0.3)
            case 24: // "N" (撥音: 低域共鳴)
                phoneWave = generateVowel(vowel: .u, durationSeconds: Float(phoneSamples) / sampleRate, f0: currentF0 * 0.9)
            case 0, 1, 39: // "<pad>", "<sil>", "<pau>" (無音区間)
                phoneWave = [Float](repeating: 0.0, count: phoneSamples)
            default:
                // その他子音・特殊拍: 弱めの母音または低振幅ノイズ
                if 0.5 <= features.voicedFlags[frameIdx] {
                    phoneWave = generateVowel(vowel: .o, durationSeconds: Float(phoneSamples) / sampleRate, f0: currentF0)
                } else {
                    phoneWave = generateWhiteNoise(durationSeconds: Float(phoneSamples) / sampleRate, amplitude: 0.2)
                }
            }

            // バルクコピーによる出力バッファへの配置
            let copyCount = min(phoneSamples, phoneWave.count)
            if 0 < copyCount {
                output.withUnsafeMutableBufferPointer { dstBuf in
                    phoneWave.withUnsafeBufferPointer { srcBuf in
                        let dstPtr = dstBuf.baseAddress!.advanced(by: currentSampleOffset)
                        dstPtr.update(from: srcBuf.baseAddress!, count: copyCount)
                    }
                }
            }

            currentSampleOffset += phoneSamples
            p += 1
        }

        return output
    }

    // MARK: - 6. 基準音声データセット生成

    /// 学習および検証用の基準音声データセットを一括生成
    ///
    /// 母音単独、挨拶、技術用語、地名、日常表現の主要な音韻およびモーラ構造を包括する。
    public func generateStandardCorpus() -> [(text: String, samples: [Float])] {
        let corpusTexts = [
            "あいうえお",
            "こんにちは",
            "すぱいくすぴーち",
            "とうきょう",
            "ありがとう"
        ]

        var corpus: [(text: String, samples: [Float])] = []
        var i = 0
        while i < corpusTexts.count {
            let t = corpusTexts[i]
            let wave = generatePhrase(phrase: t)
            corpus.append((text: t, samples: wave))
            i += 1
        }

        return corpus
    }
}
