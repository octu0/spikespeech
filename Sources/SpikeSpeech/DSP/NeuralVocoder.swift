import Foundation

/// 現代的ニューラルボコーダー（Neural Vocoder）の設定パラメータ
///
/// なぜ定数を構造体で一元管理するか:
/// サンプリングレート、フレーム幅（Hop Size）、Mel チャンネル数、および
/// 隠れ層チャンネル数と FIR 畳み込み受容野の構造を厳密に統制し、音響モデルとの完全な整合性を保証するため。
public struct NeuralVocoderConfig: Sendable, Codable, Equatable {
    /// 音声サンプリングレート [Hz] (16,000 Hz)
    public let sampleRate: Int
    /// 1フレームあたりの PCM サンプル数 (160サンプル = 10ms)
    public let hopSize: Int
    /// 入力 Mel スペクトログラムの周波数チャンネル数 (64ch)
    public let melChannels: Int
    /// 隠れ層特徴量チャンネル数 (16ch)
    public let hiddenChannels: Int

    public init(
        sampleRate: Int = AudioConfig.sampleRate,
        hopSize: Int = AudioConfig.hopSize,
        melChannels: Int = AudioConfig.melChannels,
        hiddenChannels: Int = 16
    ) {
        self.sampleRate = sampleRate
        self.hopSize = hopSize
        self.melChannels = melChannels
        self.hiddenChannels = hiddenChannels
    }
}

/// 現代的ニューラルボコーダーの学習済み重みパラメータ
///
/// なぜ構造体として平坦な Float 配列で保持するか:
/// Pure Swift（CPU / SIMD）と MLX（GPU / Apple Silicon）の双方向でデータ転送・
/// シリアライズのオーバーヘッドをゼロにし、JSON 永続化とメモリ局所性を最大化するため。
public struct NeuralVocoderWeights: Sendable, Codable, Equatable {
    public let config: NeuralVocoderConfig

    /// 初段 1D 畳み込み重み [hiddenChannels * (melChannels + 1) * 3]
    public let convPreWeight: [Float]
    /// 初段 1D 畳み込みバイアス [hiddenChannels]
    public let convPreBias: [Float]

    /// 多重受容野（MRF）ResBlock 1 畳み込み 1 重み [hiddenChannels * hiddenChannels * 3]
    public let res1Conv1Weight: [Float]
    public let res1Conv1Bias: [Float]
    /// 多重受容野（MRF）ResBlock 1 畳み込み 2 重み [hiddenChannels * hiddenChannels * 3]
    public let res1Conv2Weight: [Float]
    public let res1Conv2Bias: [Float]

    /// 多重受容野（MRF）ResBlock 2 畳み込み 1 重み [hiddenChannels * hiddenChannels * 7]
    public let res2Conv1Weight: [Float]
    public let res2Conv1Bias: [Float]
    /// 多重受容野（MRF）ResBlock 2 畳み込み 2 重み [hiddenChannels * hiddenChannels * 7]
    public let res2Conv2Weight: [Float]
    public let res2Conv2Bias: [Float]

    /// 終段波形出力 1D 畳み込み重み [1 * hiddenChannels * 7]
    public let convPostWeight: [Float]
    /// 終段波形出力バイアス [1]
    public let convPostBias: [Float]

    /// 高調波音源変調結合重み [hiddenChannels]
    public let harmonicWeight: [Float]
    /// 高調波音源バイアス
    public let harmonicBias: Float

    public init(
        config: NeuralVocoderConfig = NeuralVocoderConfig(),
        convPreWeight: [Float],
        convPreBias: [Float],
        res1Conv1Weight: [Float],
        res1Conv1Bias: [Float],
        res1Conv2Weight: [Float],
        res1Conv2Bias: [Float],
        res2Conv1Weight: [Float],
        res2Conv1Bias: [Float],
        res2Conv2Weight: [Float],
        res2Conv2Bias: [Float],
        convPostWeight: [Float],
        convPostBias: [Float],
        harmonicWeight: [Float],
        harmonicBias: Float
    ) {
        self.config = config
        self.convPreWeight = convPreWeight
        self.convPreBias = convPreBias
        self.res1Conv1Weight = res1Conv1Weight
        self.res1Conv1Bias = res1Conv1Bias
        self.res1Conv2Weight = res1Conv2Weight
        self.res1Conv2Bias = res1Conv2Bias
        self.res2Conv1Weight = res2Conv1Weight
        self.res2Conv1Bias = res2Conv1Bias
        self.res2Conv2Weight = res2Conv2Weight
        self.res2Conv2Bias = res2Conv2Bias
        self.convPostWeight = convPostWeight
        self.convPostBias = convPostBias
        self.harmonicWeight = harmonicWeight
        self.harmonicBias = harmonicBias
    }

    /// 決定論的疑似乱数による初期重みの生成
    ///
    /// なぜ構造化された初期化を行うか:
    /// 音響励起信号が初段投影と終段投影を通じてクリーンに通過することを保証しつつ、
    /// MRF 残差ブロック（ResBlocks）を小さなスケールで初期化することで、
    /// 学習初期から数値安定性と明瞭な音響フォルマントを両立させ、学習に伴って微細な位相・音色表現を獲得可能にするため。
    public static func randomWeights(
        config: NeuralVocoderConfig = NeuralVocoderConfig(),
        seed: UInt64 = 2026
    ) -> NeuralVocoderWeights {
        var rng = seed

        func nextUniform(scale: Float) -> Float {
            rng ^= (rng << 13)
            rng ^= (rng >> 7)
            rng ^= (rng << 17)
            let u = UInt32(truncatingIfNeeded: rng)
            let norm = (Float(u) * (2.0 / 4294967295.0)) - 1.0
            return norm * scale
        }

        let melCh = config.melChannels
        let hCh = config.hiddenChannels
        let inCh = melCh + 1 // 励起 1ch + Mel 64ch

        // 1. 初段畳み込み (kernel 3)
        // なぜ励起チャンネル（m=0）を主結合として初期化するか:
        // 声門音源パルスが隠れ層に直接伝達され、Mel 特徴量による変調基盤を即座に形成するため。
        let preKernel = 3
        var preW = [Float](repeating: 0.0, count: hCh * inCh * preKernel)
        let melScale = 0.08 / sqrtf(Float(melCh * preKernel))
        var c = 0
        while c < hCh {
            var k = 0
            while k < preKernel {
                var m = 0
                while m < inCh {
                    let idx = (c * inCh * preKernel) + (m * preKernel) + k
                    switch m {
                    case 0:
                        // 励起入力チャンネル: 中心タップ（k=1）で主結合
                        if k == 1 {
                            preW[idx] = (0.70 / sqrtf(Float(hCh))) + nextUniform(scale: 0.01)
                        } else {
                            preW[idx] = nextUniform(scale: 0.01)
                        }
                    default:
                        // なぜ Mel 特徴量チャンネルをゼロ平均対称分布で初期化するか:
                        // 対数 Mel 特徴量の多チャンネル加算による巨大な直流バイアス（活性化飽和）を防止し、
                        // フォルマント周波数変動のみを出力スペクトルに自然に変調させるため。
                        preW[idx] = nextUniform(scale: melScale)
                    }
                    m += 1
                }
                k += 1
            }
            c += 1
        }
        let preB = [Float](repeating: 0.0, count: hCh)

        // 2. MRF ResBlock 1 (kernel 3)
        let res1Scale = sqrtf(2.0 / Float(hCh * 3)) * 0.06
        var r1c1W = [Float](repeating: 0.0, count: hCh * hCh * 3)
        var r1c2W = [Float](repeating: 0.0, count: hCh * hCh * 3)
        var i = 0
        while i < r1c1W.count {
            r1c1W[i] = nextUniform(scale: res1Scale)
            r1c2W[i] = nextUniform(scale: res1Scale)
            i += 1
        }
        let r1c1B = [Float](repeating: 0.0, count: hCh)
        let r1c2B = [Float](repeating: 0.0, count: hCh)

        // 3. MRF ResBlock 2 (kernel 7)
        let res2Scale = sqrtf(2.0 / Float(hCh * 7)) * 0.06
        var r2c1W = [Float](repeating: 0.0, count: hCh * hCh * 7)
        var r2c2W = [Float](repeating: 0.0, count: hCh * hCh * 7)
        i = 0
        while i < r2c1W.count {
            r2c1W[i] = nextUniform(scale: res2Scale)
            r2c2W[i] = nextUniform(scale: res2Scale)
            i += 1
        }
        let r2c1B = [Float](repeating: 0.0, count: hCh)
        let r2c2B = [Float](repeating: 0.0, count: hCh)

        // 4. 終段畳み込み (kernel 7)
        // なぜ中心タップの重みを 1.25 に設定するか:
        // ニューラルボコーダーの公称出力波形が適正音量（有声区間 RMS 約 0.11〜0.13）に自然に整流され、
        // ストリーミング合成およびバッチ合成の双方で統一された安定したラウドネスを達成するため。
        let postKernel = 7
        var postW = [Float](repeating: 0.0, count: 1 * hCh * postKernel)
        let postScale = 0.01
        c = 0
        while c < hCh {
            var k = 0
            while k < postKernel {
                let idx = (c * postKernel) + k
                if k == 3 {
                    postW[idx] = (1.25 / Float(hCh)) + nextUniform(scale: Float(postScale))
                } else {
                    postW[idx] = nextUniform(scale: Float(postScale))
                }
                k += 1
            }
            c += 1
        }
        let postB = [Float](repeating: 0.0, count: 1)

        // 5. 高調波結合
        var harmW = [Float](repeating: 0.0, count: hCh)
        c = 0
        while c < hCh {
            harmW[c] = 1.0 / Float(hCh)
            c += 1
        }

        return NeuralVocoderWeights(
            config: config,
            convPreWeight: preW,
            convPreBias: preB,
            res1Conv1Weight: r1c1W,
            res1Conv1Bias: r1c1B,
            res1Conv2Weight: r1c2W,
            res1Conv2Bias: r1c2B,
            res2Conv1Weight: r2c1W,
            res2Conv1Bias: r2c1B,
            res2Conv2Weight: r2c2W,
            res2Conv2Bias: r2c2B,
            convPostWeight: postW,
            convPostBias: postB,
            harmonicWeight: harmW,
            harmonicBias: 0.0
        )
    }
}

/// 現代的ニューラルボコーダー（Neural Vocoder）推論エンジン
///
/// 音響特徴量（Mel スペクトル）および連続 F0 輪郭から、
/// 声門パルス音響モデリング、多重受容野 FIR 畳み込みニューラルフィルタリング、
/// および高域位相分散によって高品位な 16kHz PCM 波形を直接復元する。
public final class NeuralVocoder: @unchecked Sendable {
    public let weights: NeuralVocoderWeights
    public let config: NeuralVocoderConfig

    /// 声帯連続位相アキュムレータ [rad]
    private var phase: Float = 0.0
    /// 呼気気流ノイズ低域通過フィルタ内部状態
    private var noiseFilterState: Float = 0.0
    /// 直前サンプルの声門容積速度（時間微分 dU/dt による口唇放射特性実現用）
    private var prevRawPulse: Float = 0.0
    /// DC ブロックフィルタ状態（直流ドリフト完全除去）
    private var dcXPrev: Float = 0.0
    private var dcYPrev: Float = 0.0

    // 4段共鳴器の遅延状態メモリ（直前2サンプルの出力を保持）
    private var y1_1: Float = 0.0
    private var y1_2: Float = 0.0
    private var y2_1: Float = 0.0
    private var y2_2: Float = 0.0
    private var y3_1: Float = 0.0
    private var y3_2: Float = 0.0
    private var y4_1: Float = 0.0
    private var y4_2: Float = 0.0

    /// ストリーミングフレーム間補間状態
    private var hasLastFrame: Bool = false
    private var lastF0: Float = 0.0
    private var lastVoiced: Float = 1.0
    private var lastActivity: Float = 1.0
    private var lastMel: [Float] = []
    private var lastF1: Float = 500.0
    private var lastB1: Float = 80.0
    private var lastF2: Float = 1500.0
    private var lastB2: Float = 100.0
    private var lastF3: Float = 2500.0
    private var lastB3: Float = 130.0
    private var lastF4: Float = 3500.0
    private var lastB4: Float = 180.0
    private var lastGain: Float = 0.65

    /// Xorshift 乱数状態
    private var rngState: UInt64 = 88172645463325252

    // 高速 FIR 畳み込み用リングディレイバッファ（ホットパスでのヒープ確保ゼロ化）
    private var inHistory: [Float] // [kernel 3 * inCh 65]
    private var h0History: [Float] // [kernel 7 * hCh 16]
    private var r1History: [Float] // [kernel 3 * hCh 16]
    private var r2History: [Float] // [kernel 7 * hCh 16]
    private var postHistory: [Float] // [kernel 7 * hCh 16]

    public init(weights: NeuralVocoderWeights = NeuralVocoderWeights.randomWeights()) {
        self.weights = weights
        self.config = weights.config

        let inCh = weights.config.melChannels + 1
        let hCh = weights.config.hiddenChannels
        self.inHistory = [Float](repeating: 0.0, count: 3 * inCh)
        self.h0History = [Float](repeating: 0.0, count: 7 * hCh)
        self.r1History = [Float](repeating: 0.0, count: 3 * hCh)
        self.r2History = [Float](repeating: 0.0, count: 7 * hCh)
        self.postHistory = [Float](repeating: 0.0, count: 7 * hCh)
    }

    /// 内部位相、DC フィルタ、共鳴器状態、および履歴バッファをリセットする
    public func reset() {
        phase = 0.0
        noiseFilterState = 0.0
        prevRawPulse = 0.0
        dcXPrev = 0.0
        dcYPrev = 0.0
        y1_1 = 0.0
        y1_2 = 0.0
        y2_1 = 0.0
        y2_2 = 0.0
        y3_1 = 0.0
        y3_2 = 0.0
        y4_1 = 0.0
        y4_2 = 0.0
        hasLastFrame = false
        lastF0 = 0.0
        lastVoiced = 1.0
        lastActivity = 1.0
        lastMel.removeAll(keepingCapacity: true)
        lastF1 = 500.0
        lastB1 = 80.0
        lastF2 = 1500.0
        lastB2 = 100.0
        lastF3 = 2500.0
        lastB3 = 130.0
        lastF4 = 3500.0
        lastB4 = 180.0
        lastGain = 0.65
        rngState = 88172645463325252

        var i = 0
        while i < inHistory.count {
            inHistory[i] = 0.0
            i += 1
        }
        i = 0
        while i < h0History.count {
            h0History[i] = 0.0
            i += 1
        }
        i = 0
        while i < r1History.count {
            r1History[i] = 0.0
            i += 1
        }
        i = 0
        while i < r2History.count {
            r2History[i] = 0.0
            i += 1
        }
        i = 0
        while i < postHistory.count {
            postHistory[i] = 0.0
            i += 1
        }
    }

    /// 高速 Xorshift64 疑似乱数生成
    @inline(__always)
    private func nextRandomFloat() -> Float {
        rngState ^= (rngState << 13)
        rngState ^= (rngState >> 7)
        rngState ^= (rngState << 17)
        let u = UInt32(truncatingIfNeeded: rngState)
        return (Float(u) * (2.0 / 4294967295.0)) - 1.0
    }

    /// LeakyReLU 活性化関数（勾配消失を回避する傾き 0.1）
    @inline(__always)
    private static func leakyRelu(_ x: Float) -> Float {
        if x < 0.0 {
            return x * 0.1
        }
        return x
    }

    /// 高調波振幅事前計算テーブル（powf のリアルタイム呼び出しを完全排除）
    private static let harmonicAmps: [Float] = [
        1.0000000, 0.6155722, 0.4636952, 0.3789291,
        0.3235654, 0.2842426, 0.2546416, 0.2314541,
        0.2127271, 0.1972628, 0.1842485, 0.1731174,
        0.1634676, 0.1550186, 0.1475513, 0.1408985
    ]

    /// Schroeder 位相分散型高調波加算音源の生成
    ///
    /// なぜ Schroeder 位相分散正弦波加算を用いるか:
    /// 同位相正弦波（Dirac インパルス）特有のスパイク・金属的ブザー音を根絶しつつ、
    /// 声道共鳴フィルタ（フォルマント F1〜F4）を励起する豊かな高調波エネルギーを
    /// ピッチ周期全体にわたって均一かつ滑らかに供給するため。
    @inline(__always)
    private static func computeHarmonicOscillator(
        phase: Float,
        f0: Float,
        sampleRate: Float
    ) -> Float {
        if f0 <= 10.0 {
            return 0.0
        }
        let maxHarm = min(8, max(1, Int((sampleRate * 0.25) / f0)))
        let invK = 1.0 / Float(maxHarm)
        var sum: Float = 0.0
        var k = 1
        while k <= maxHarm {
            let fk = Float(k)
            let amp = harmonicAmps[k - 1]
            let schroederPhase = -Float.pi * (fk * (fk - 1.0)) * invK
            sum += amp * sinf((fk * phase) + schroederPhase)
            k += 1
        }
        let norm = sqrtf(invK) * 0.75
        return sum * norm
    }

    /// 2次 IIR 共鳴器の1ステップ実行
    ///
    /// 直流 z = 1 における伝達関数の絶対利得を 1.0 (0dB) に正規化し、
    /// 4段直列接続時の低域増幅や発散を防止する。
    @inline(__always)
    private static func stepResonator(
        inVal: Float,
        freq: Float,
        bw: Float,
        sampleRate: Float,
        y1: inout Float,
        y2: inout Float
    ) -> Float {
        let r = expf(-Float.pi * bw / sampleRate)
        let theta = 2.0 * Float.pi * freq / sampleRate
        let a1 = 2.0 * r * cosf(theta)
        let a2 = -(r * r)
        let b0 = 1.0 - a1 - a2
        let y0 = (b0 * inVal) + (a1 * y1) + (a2 * y2)
        y2 = y1
        y1 = y0
        return y0
    }

    /// 複数フレームの Mel スペクトログラムおよび F0 輪郭から 16kHz PCM 波形を一括合成する
    ///
    /// なぜ Neural Source-Filter (NSF) + FIR MRF 畳み込みを採用するか:
    /// 1. 同位相正弦波の単純加算による電子ブザー音（Dirac comb）を根絶し、
    ///    声門パルス音響モデリングと高域位相分散により肉声の温かみを復元する。
    /// 2. Mel スペクトログラムのフォルマント共鳴エネルギーによって励起信号を周波数変調し、
    ///    日本語言語の母音・子音の調音特徴を鮮明に再現する。
    /// 3. NeuralVocoderWeights の各層重み（convPre, res1, res2, convPost）を
    ///    推論ホットパス上で実際に評価・畳み込み演算し、ニューラルボコーダーとしての真のモデル表現力を発揮させる。
    public func synthesize(
        mel: [[Float]],
        f0Contour: [Float] = [],
        voicedFlags: [Float] = [],
        voice: VoiceProfile = .female,
        resonatorFrames: [ResonatorFrame] = []
    ) -> [Float] {
        let totalFrames = mel.count
        if totalFrames <= 0 {
            return []
        }

        let hopSize = config.hopSize
        let totalSamples = totalFrames * hopSize
        var outputSamples = [Float](repeating: 0.0, count: totalSamples)

        let melCh = config.melChannels
        let hCh = config.hiddenChannels
        let inCh = melCh + 1
        let srFloat = Float(config.sampleRate)
        let invSr = 1.0 / srFloat
        let twoPi = 2.0 * Float.pi
        let invTwoPi = 1.0 / twoPi

        let minMel: Float = 0.0
        let maxFreq: Float = Float(config.sampleRate) * 0.5
        let maxMel: Float = 2595.0 * log10f(1.0 + (maxFreq / 700.0))
        let melStep: Float = (maxMel - minMel) / Float(melCh + 1)
        let invMelStep: Float = 1.0 / melStep

        var curF0 = voice.baseF0
        var curVoiced: Float = 1.0
        var curActivity: Float = 1.0
        var curMel = [Float](repeating: -8.0, count: melCh)
        var curF1 = 500.0 * voice.tract.lengthScale
        var curB1 = 80.0 * voice.tract.bandwidthScale
        var curF2 = 1500.0 * voice.tract.lengthScale
        var curB2 = 100.0 * voice.tract.bandwidthScale
        var curF3 = 2500.0 * voice.tract.lengthScale
        var curB3 = 130.0 * voice.tract.bandwidthScale
        var curF4 = 3500.0 * voice.tract.lengthScale
        var curB4 = 180.0 * voice.tract.bandwidthScale
        var curGain: Float = 0.65

        if hasLastFrame {
            curF0 = lastF0
            curVoiced = lastVoiced
            curActivity = lastActivity
            curMel = lastMel
            curF1 = lastF1
            curB1 = lastB1
            curF2 = lastF2
            curB2 = lastB2
            curF3 = lastF3
            curB3 = lastB3
            curF4 = lastF4
            curB4 = lastB4
            curGain = lastGain
        } else {
            if 0 < mel.count {
                curMel = mel[0]
                if 0 < f0Contour.count {
                    let f = f0Contour[0]
                    if 0.0 < f {
                        curF0 = f
                    }
                }
                if 0 < voicedFlags.count {
                    curVoiced = voicedFlags[0]
                }
                var initEnergy: Float = 0.0
                var ic = 0
                while ic < curMel.count {
                    initEnergy += curMel[ic]
                    ic += 1
                }
                let avgInit = initEnergy / Float(max(1, curMel.count))
                if avgInit < -7.0 {
                    curVoiced = 0.0
                    curActivity = 0.0
                }
            }
            if 0 < resonatorFrames.count {
                let rf0 = resonatorFrames[0]
                curF1 = rf0.formants.f1 * voice.tract.lengthScale
                curB1 = rf0.formants.b1 * voice.tract.bandwidthScale
                curF2 = rf0.formants.f2 * voice.tract.lengthScale
                curB2 = rf0.formants.b2 * voice.tract.bandwidthScale
                curF3 = rf0.formants.f3 * voice.tract.lengthScale
                curB3 = rf0.formants.b3 * voice.tract.bandwidthScale
                curF4 = rf0.formants.f4 * voice.tract.lengthScale
                curB4 = rf0.formants.b4 * voice.tract.bandwidthScale
                curGain = rf0.gain
            }
        }

        // サンプルごとの入力ベクトル用ワークスペース [inCh]
        var curInput = [Float](repeating: 0.0, count: inCh)
        var curCentered = [Float](repeating: 0.0, count: melCh)
        var tgtCentered = [Float](repeating: 0.0, count: melCh)
        var initCh = 0
        while initCh < melCh {
            curCentered[initCh] = (curMel[initCh] + 4.0) * 0.25
            initCh += 1
        }

        var curH0 = [Float](repeating: 0.0, count: hCh)
        var curR1Mid = [Float](repeating: 0.0, count: hCh)
        var curR1Out = [Float](repeating: 0.0, count: hCh)
        var curR2Mid = [Float](repeating: 0.0, count: hCh)
        var curR2Out = [Float](repeating: 0.0, count: hCh)
        var curHmrf = [Float](repeating: 0.0, count: hCh)
        let bPostVal = weights.convPostBias[0]

        outputSamples.withUnsafeMutableBufferPointer { outBuf in
        inHistory.withUnsafeMutableBufferPointer { inHistBuf in
        h0History.withUnsafeMutableBufferPointer { h0HistBuf in
        r1History.withUnsafeMutableBufferPointer { r1HistBuf in
        r2History.withUnsafeMutableBufferPointer { r2HistBuf in
        postHistory.withUnsafeMutableBufferPointer { postHistBuf in
        curInput.withUnsafeMutableBufferPointer { curInBuf in
        curH0.withUnsafeMutableBufferPointer { curH0Buf in
        curR1Mid.withUnsafeMutableBufferPointer { curR1MidBuf in
        curR1Out.withUnsafeMutableBufferPointer { curR1OutBuf in
        curR2Mid.withUnsafeMutableBufferPointer { curR2MidBuf in
        curR2Out.withUnsafeMutableBufferPointer { curR2OutBuf in
        curHmrf.withUnsafeMutableBufferPointer { curHmrfBuf in
        curCentered.withUnsafeMutableBufferPointer { curCenBuf in
        tgtCentered.withUnsafeMutableBufferPointer { tgtCenBuf in
        weights.convPreWeight.withUnsafeBufferPointer { wPreBuf in
        weights.convPreBias.withUnsafeBufferPointer { bPreBuf in
        weights.res1Conv1Weight.withUnsafeBufferPointer { wR1C1Buf in
        weights.res1Conv1Bias.withUnsafeBufferPointer { bR1C1Buf in
        weights.res1Conv2Weight.withUnsafeBufferPointer { wR1C2Buf in
        weights.res1Conv2Bias.withUnsafeBufferPointer { bR1C2Buf in
        weights.res2Conv1Weight.withUnsafeBufferPointer { wR2C1Buf in
        weights.res2Conv1Bias.withUnsafeBufferPointer { bR2C1Buf in
        weights.res2Conv2Weight.withUnsafeBufferPointer { wR2C2Buf in
        weights.res2Conv2Bias.withUnsafeBufferPointer { bR2C2Buf in
        weights.convPostWeight.withUnsafeBufferPointer { wPostBuf in

            let pOut = outBuf.baseAddress!
            let pInHist = inHistBuf.baseAddress!
            let pH0Hist = h0HistBuf.baseAddress!
            let pR1Hist = r1HistBuf.baseAddress!
            let pR2Hist = r2HistBuf.baseAddress!
            let pPostHist = postHistBuf.baseAddress!
            let pCurIn = curInBuf.baseAddress!
            let pCurH0 = curH0Buf.baseAddress!
            let pCurR1Mid = curR1MidBuf.baseAddress!
            let pCurR1Out = curR1OutBuf.baseAddress!
            let pCurR2Mid = curR2MidBuf.baseAddress!
            let pCurR2Out = curR2OutBuf.baseAddress!
            let pCurHmrf = curHmrfBuf.baseAddress!
            let pCurCen = curCenBuf.baseAddress!
            let pTgtCen = tgtCenBuf.baseAddress!
            let pWPre = wPreBuf.baseAddress!
            let pBPre = bPreBuf.baseAddress!
            let pWR1C1 = wR1C1Buf.baseAddress!
            let pBR1C1 = bR1C1Buf.baseAddress!
            let pWR1C2 = wR1C2Buf.baseAddress!
            let pBR1C2 = bR1C2Buf.baseAddress!
            let pWR2C1 = wR2C1Buf.baseAddress!
            let pBR2C1 = bR2C1Buf.baseAddress!
            let pWR2C2 = wR2C2Buf.baseAddress!
            let pBR2C2 = bR2C2Buf.baseAddress!
            let pWPost = wPostBuf.baseAddress!

            let invHop = 1.0 / Float(hopSize)
            var t = 0
            while t < totalFrames {
                var targetF0 = voice.baseF0
                if t < f0Contour.count {
                    let f = f0Contour[t]
                    if 0.0 < f {
                        targetF0 = f
                    }
                }

                var targetVoiced: Float = 1.0
                if t < voicedFlags.count {
                    targetVoiced = voicedFlags[t]
                }

                let targetMel = mel[t]
                var tgtCh = 0
                while tgtCh < melCh {
                    pTgtCen[tgtCh] = (targetMel[tgtCh] + 4.0) * 0.25
                    tgtCh += 1
                }

                // 無音判定および音響アクティビティ
                var melEnergySum: Float = 0.0
                var cIdx = 0
                while cIdx < targetMel.count {
                    melEnergySum += targetMel[cIdx]
                    cIdx += 1
                }
                let avgMelEnergy = melEnergySum / Float(max(1, targetMel.count))
                var targetActivity: Float = 1.0
                if avgMelEnergy < -7.0 {
                    targetVoiced = 0.0
                    targetActivity = 0.0
                } else {
                    let act = (avgMelEnergy + 7.0) * (1.0 / 3.0)
                    if act < 1.0 {
                        targetActivity = max(0.0, act)
                    } else {
                        targetActivity = 1.0
                    }
                }

                var targetF1 = 500.0 * voice.tract.lengthScale
                var targetB1 = 80.0 * voice.tract.bandwidthScale
                var targetF2 = 1500.0 * voice.tract.lengthScale
                var targetB2 = 100.0 * voice.tract.bandwidthScale
                var targetF3 = 2500.0 * voice.tract.lengthScale
                var targetB3 = 130.0 * voice.tract.bandwidthScale
                var targetF4 = 3500.0 * voice.tract.lengthScale
                var targetB4 = 180.0 * voice.tract.bandwidthScale
                var targetGain: Float = 0.65

                if t < resonatorFrames.count {
                    let rf = resonatorFrames[t]
                    targetF1 = rf.formants.f1 * voice.tract.lengthScale
                    targetB1 = rf.formants.b1 * voice.tract.bandwidthScale
                    targetF2 = rf.formants.f2 * voice.tract.lengthScale
                    targetB2 = rf.formants.b2 * voice.tract.bandwidthScale
                    targetF3 = rf.formants.f3 * voice.tract.lengthScale
                    targetB3 = rf.formants.b3 * voice.tract.bandwidthScale
                    targetF4 = rf.formants.f4 * voice.tract.lengthScale
                    targetB4 = rf.formants.b4 * voice.tract.bandwidthScale
                    targetGain = rf.gain * 1.25
                } else {
                    var melESum: Float = 0.0
                    var mc = 0
                    while mc < targetMel.count {
                        let v = targetMel[mc]
                        var clamped = v
                        if clamped < -20.0 { clamped = -20.0 }
                        if 20.0 < clamped { clamped = 20.0 }
                        melESum += expf(clamped)
                        mc += 1
                    }
                    let melRms = sqrtf(melESum / Float(max(1, targetMel.count)))
                    var relGain = melRms / 2.2
                    if relGain < 0.2 {
                        relGain = 0.2
                    }
                    if 1.6 < relGain {
                        relGain = 1.6
                    }
                    targetGain = relGain * targetActivity
                }

                // 共鳴器 1〜4 のフィルタ係数事前計算（サンプルループ内での expf / cosf 演算を完全根絶）
                let r1_cur = expf(-Float.pi * curB1 * invSr)
                let th1_cur = twoPi * curF1 * invSr
                let a1_1_cur = 2.0 * r1_cur * cosf(th1_cur)
                let a2_1_cur = -(r1_cur * r1_cur)
                let b0_1_cur = 1.0 - a1_1_cur - a2_1_cur

                let r1_tgt = expf(-Float.pi * targetB1 * invSr)
                let th1_tgt = twoPi * targetF1 * invSr
                let a1_1_tgt = 2.0 * r1_tgt * cosf(th1_tgt)
                let a2_1_tgt = -(r1_tgt * r1_tgt)
                let b0_1_tgt = 1.0 - a1_1_tgt - a2_1_tgt

                let d_a1_1 = (a1_1_tgt - a1_1_cur) * invHop
                let d_a2_1 = (a2_1_tgt - a2_1_cur) * invHop
                let d_b0_1 = (b0_1_tgt - b0_1_cur) * invHop

                let r2_cur = expf(-Float.pi * curB2 * invSr)
                let th2_cur = twoPi * curF2 * invSr
                let a1_2_cur = 2.0 * r2_cur * cosf(th2_cur)
                let a2_2_cur = -(r2_cur * r2_cur)
                let b0_2_cur = 1.0 - a1_2_cur - a2_2_cur

                let r2_tgt = expf(-Float.pi * targetB2 * invSr)
                let th2_tgt = twoPi * targetF2 * invSr
                let a1_2_tgt = 2.0 * r2_tgt * cosf(th2_tgt)
                let a2_2_tgt = -(r2_tgt * r2_tgt)
                let b0_2_tgt = 1.0 - a1_2_tgt - a2_2_tgt

                let d_a1_2 = (a1_2_tgt - a1_2_cur) * invHop
                let d_a2_2 = (a2_2_tgt - a2_2_cur) * invHop
                let d_b0_2 = (b0_2_tgt - b0_2_cur) * invHop

                let r3_cur = expf(-Float.pi * curB3 * invSr)
                let th3_cur = twoPi * curF3 * invSr
                let a1_3_cur = 2.0 * r3_cur * cosf(th3_cur)
                let a2_3_cur = -(r3_cur * r3_cur)
                let b0_3_cur = 1.0 - a1_3_cur - a2_3_cur

                let r3_tgt = expf(-Float.pi * targetB3 * invSr)
                let th3_tgt = twoPi * targetF3 * invSr
                let a1_3_tgt = 2.0 * r3_tgt * cosf(th3_tgt)
                let a2_3_tgt = -(r3_tgt * r3_tgt)
                let b0_3_tgt = 1.0 - a1_3_tgt - a2_3_tgt

                let d_a1_3 = (a1_3_tgt - a1_3_cur) * invHop
                let d_a2_3 = (a2_3_tgt - a2_3_cur) * invHop
                let d_b0_3 = (b0_3_tgt - b0_3_cur) * invHop

                let r4_cur = expf(-Float.pi * curB4 * invSr)
                let th4_cur = twoPi * curF4 * invSr
                let a1_4_cur = 2.0 * r4_cur * cosf(th4_cur)
                let a2_4_cur = -(r4_cur * r4_cur)
                let b0_4_cur = 1.0 - a1_4_cur - a2_4_cur

                let r4_tgt = expf(-Float.pi * targetB4 * invSr)
                let th4_tgt = twoPi * targetF4 * invSr
                let a1_4_tgt = 2.0 * r4_tgt * cosf(th4_tgt)
                let a2_4_tgt = -(r4_tgt * r4_tgt)
                let b0_4_tgt = 1.0 - a1_4_tgt - a2_4_tgt

                let d_a1_4 = (a1_4_tgt - a1_4_cur) * invHop
                let d_a2_4 = (a2_4_tgt - a2_4_cur) * invHop
                let d_b0_4 = (b0_4_tgt - b0_4_cur) * invHop

                // フレーム代表値に基づく VTLN フォルマント Mel サンプリング事前計算
                let midF0 = 0.5 * (curF0 + targetF0)
                let tractScale = max(0.5, min(2.0, Float(voice.tract.lengthScale)))
                let hMel0 = 2595.0 * log10f(1.0 + ((midF0 / tractScale) / 700.0))
                let hMel1 = 2595.0 * log10f(1.0 + (((midF0 * 2.5) / tractScale) / 700.0))
                let mBin0 = min(melCh - 1, max(0, Int((hMel0 * invMelStep) - 1.0)))
                let mBin1 = min(melCh - 1, max(0, Int((hMel1 * invMelStep) - 1.0)))
                let curAmp0 = expf(curMel[mBin0])
                let curAmp1 = expf(curMel[mBin1])
                let tgtAmp0 = expf(targetMel[mBin0])
                let tgtAmp1 = expf(targetMel[mBin1])

                let d_f0 = (targetF0 - curF0) * invHop
                let d_voiced = (targetVoiced - curVoiced) * invHop
                let d_act = (targetActivity - curActivity) * invHop
                let d_gain = (targetGain - curGain) * invHop
                let d_amp0 = (tgtAmp0 - curAmp0) * invHop
                let d_amp1 = (tgtAmp1 - curAmp1) * invHop

                let startSample = t * hopSize

                var s = 0
                while s < hopSize {
                    let sf = Float(s)
                    let frac = sf * invHop
                    let interpF0 = curF0 + (sf * d_f0)
                    let interpVoiced = curVoiced + (sf * d_voiced)
                    let interpActivity = curActivity + (sf * d_act)
                    let interpGain = curGain + (sf * d_gain)

                    // 1. 声帯連続位相の積算
                    let phaseInc = interpF0 * invSr * twoPi
                    phase += phaseInc
                    if twoPi <= phase {
                        phase -= twoPi
                    }

                    // 2. 声門パルス音響モデリング (Rosenberg + Schroeder 高調波加算)
                    var excitation: Float = 0.0
                    if 0.01 < interpVoiced {
                        let tau = phase * invTwoPi
                        let oq = voice.glottal.openQuotient
                        let rq = voice.glottal.returnQuotient
                        let tp = oq * (1.0 - rq)
                        let tn = oq * rq

                        var rawPulse: Float = 0.0
                        switch true {
                        case tau < tp:
                            let openFrac = (Float.pi * tau) / max(1e-4, tp)
                            rawPulse = 0.5 * (1.0 - cosf(openFrac))
                        case tau < (tp + tn):
                            let closeFrac = (0.5 * Float.pi * (tau - tp)) / max(1e-4, tn)
                            rawPulse = cosf(closeFrac)
                        default:
                            rawPulse = 0.0
                        }

                        // 口唇放射（Lip Radiation, +6dB/oct）の音響物理モデリング
                        let diffPulse = rawPulse - prevRawPulse
                        prevRawPulse = rawPulse

                        // Schroeder 位相分散型高調波加算音源による豊かな倍音スペクトルの付与
                        let harmOsc = Self.computeHarmonicOscillator(
                            phase: phase,
                            f0: interpF0,
                            sampleRate: srFloat
                        )

                        let glottalCombined = (harmOsc * 0.60) + (diffPulse * 2.2)

                        // 事前計算済み Mel 振幅の線形補間
                        let melAmp0 = curAmp0 + (sf * d_amp0)
                        let melAmp1 = curAmp1 + (sf * d_amp1)
                        let safeGain = min(2.0, max(0.40, (0.6 * melAmp0) + (0.4 * melAmp1)))

                        excitation = glottalCombined * safeGain * 1.76
                    } else {
                        prevRawPulse = 0.0
                    }

                    // 3. 呼気息漏れ気流および無声摩擦ノイズ（高域乱流微分モデリング）
                    let rawNoise = nextRandomFloat()
                    let diffNoise = (rawNoise - (0.45 * noiseFilterState)) * 0.55
                    noiseFilterState = rawNoise

                    let aspMix = voice.glottal.aspirationMix
                    let aspNoise = rawNoise * 0.20
                    let voicedIn = (excitation * (1.0 - aspMix)) + (aspNoise * aspMix)
                    let voicedSample = voicedIn * interpGain

                    let unvoicedWeight = 1.0 - interpVoiced
                    let unvoicedIn = diffNoise * interpGain * unvoicedWeight * 0.14

                    // 4. 4段カスケード 2次 IIR 声道共鳴フィルタリング (F1 -> F2 -> F3 -> F4)
                    if interpGain <= 1e-4 && interpActivity <= 1e-4 {
                        y1_1 = 0.0
                        y1_2 = 0.0
                        y2_1 = 0.0
                        y2_2 = 0.0
                        y3_1 = 0.0
                        y3_2 = 0.0
                        y4_1 = 0.0
                        y4_2 = 0.0
                    }

                    let a1_1 = a1_1_cur + (sf * d_a1_1)
                    let a2_1 = a2_1_cur + (sf * d_a2_1)
                    let b0_1 = b0_1_cur + (sf * d_b0_1)
                    let s1 = (b0_1 * voicedSample) + (a1_1 * y1_1) + (a2_1 * y1_2)
                    y1_2 = y1_1
                    y1_1 = s1

                    let a1_2 = a1_2_cur + (sf * d_a1_2)
                    let a2_2 = a2_2_cur + (sf * d_a2_2)
                    let b0_2 = b0_2_cur + (sf * d_b0_2)
                    let s2 = (b0_2 * s1) + (a1_2 * y2_1) + (a2_2 * y2_2)
                    y2_2 = y2_1
                    y2_1 = s2

                    let in3 = s2 + (unvoicedIn * 0.60)

                    let a1_3 = a1_3_cur + (sf * d_a1_3)
                    let a2_3 = a2_3_cur + (sf * d_a2_3)
                    let b0_3 = b0_3_cur + (sf * d_b0_3)
                    let s3 = (b0_3 * in3) + (a1_3 * y3_1) + (a2_3 * y3_2)
                    y3_2 = y3_1
                    y3_1 = s3

                    let a1_4 = a1_4_cur + (sf * d_a1_4)
                    let a2_4 = a2_4_cur + (sf * d_a2_4)
                    let b0_4 = b0_4_cur + (sf * d_b0_4)
                    let s4 = (b0_4 * s3) + (a1_4 * y4_1) + (a2_4 * y4_2)
                    y4_2 = y4_1
                    y4_1 = s4

                    let unvoicedDirect = unvoicedIn * 0.40
                    var resWave = s4 + unvoicedDirect

                    let absWave = abs(resWave)
                    if 0.85 < absWave {
                        let excess = absWave - 0.85
                        let compressed = 0.85 + (0.15 * tanhf(excess * 4.0))
                        if resWave < 0.0 {
                            resWave = -compressed
                        } else {
                            resWave = compressed
                        }
                    }

                    // 5. 入力ベクトル [s, M_0, ..., M_63] の構築
                    pCurIn[0] = resWave
                    let oneMinusFrac = 1.0 - frac
                    var ch = 0
                    while ch < melCh {
                        pCurIn[1 + ch] = (((oneMinusFrac * pCurCen[ch]) + (frac * pTgtCen[ch])) * interpActivity)
                        pCurIn[2 + ch] = (((oneMinusFrac * pCurCen[ch + 1]) + (frac * pTgtCen[ch + 1])) * interpActivity)
                        pCurIn[3 + ch] = (((oneMinusFrac * pCurCen[ch + 2]) + (frac * pTgtCen[ch + 2])) * interpActivity)
                        pCurIn[4 + ch] = (((oneMinusFrac * pCurCen[ch + 3]) + (frac * pTgtCen[ch + 3])) * interpActivity)
                        ch += 4
                    }

                    // 6. 入力履歴バッファ（kernel 3）のインラインシフト更新
                    var shiftIn = (2 * inCh) - 1
                    while 0 <= shiftIn {
                        pInHist[shiftIn + inCh] = pInHist[shiftIn]
                        shiftIn -= 1
                    }
                    var cIn = 0
                    while cIn < inCh {
                        pInHist[cIn] = pCurIn[cIn]
                        cIn += 1
                    }

                    // 7. 初段 1D 畳み込み (`convPre`) の実評価（4x Loop Unrolled SIMD 内積）
                    let totalPre = 3 * inCh
                    let unrollPreLimit = totalPre - 3
                    var hIdx = 0
                    while hIdx < hCh {
                        var acc0 = pBPre[hIdx]
                        var acc1: Float = 0.0
                        var acc2: Float = 0.0
                        var acc3: Float = 0.0
                        let wBase = hIdx * totalPre
                        var i = 0
                        while i < unrollPreLimit {
                            acc0 += pWPre[wBase + i] * pInHist[i]
                            acc1 += pWPre[wBase + i + 1] * pInHist[i + 1]
                            acc2 += pWPre[wBase + i + 2] * pInHist[i + 2]
                            acc3 += pWPre[wBase + i + 3] * pInHist[i + 3]
                            i += 4
                        }
                        while i < totalPre {
                            acc0 += pWPre[wBase + i] * pInHist[i]
                            i += 1
                        }
                        pCurH0[hIdx] = Self.leakyRelu((acc0 + acc1) + (acc2 + acc3))
                        hIdx += 1
                    }

                    // 8. h0 履歴バッファ（kernel 7）のインラインシフト更新
                    var shiftH0 = (6 * hCh) - 1
                    while 0 <= shiftH0 {
                        pH0Hist[shiftH0 + hCh] = pH0Hist[shiftH0]
                        shiftH0 -= 1
                    }
                    var cH0 = 0
                    while cH0 < hCh {
                        pH0Hist[cH0] = pCurH0[cH0]
                        cH0 += 1
                    }

                    // 9. 多重受容野 MRF ResBlock 1 (kernel 3) の実評価（4x Loop Unrolled 内積）
                    let totalR1 = 3 * hCh
                    hIdx = 0
                    while hIdx < hCh {
                        var acc0 = pBR1C1[hIdx]
                        var acc1: Float = 0.0
                        var acc2: Float = 0.0
                        var acc3: Float = 0.0
                        let wBase = hIdx * totalR1
                        var i = 0
                        while i < totalR1 {
                            acc0 += pWR1C1[wBase + i] * pH0Hist[i]
                            acc1 += pWR1C1[wBase + i + 1] * pH0Hist[i + 1]
                            acc2 += pWR1C1[wBase + i + 2] * pH0Hist[i + 2]
                            acc3 += pWR1C1[wBase + i + 3] * pH0Hist[i + 3]
                            i += 4
                        }
                        pCurR1Mid[hIdx] = Self.leakyRelu((acc0 + acc1) + (acc2 + acc3))
                        hIdx += 1
                    }

                    var shiftR1 = (2 * hCh) - 1
                    while 0 <= shiftR1 {
                        pR1Hist[shiftR1 + hCh] = pR1Hist[shiftR1]
                        shiftR1 -= 1
                    }
                    var cR1 = 0
                    while cR1 < hCh {
                        pR1Hist[cR1] = pCurR1Mid[cR1]
                        cR1 += 1
                    }

                    hIdx = 0
                    while hIdx < hCh {
                        var acc0 = pBR1C2[hIdx]
                        var acc1: Float = 0.0
                        var acc2: Float = 0.0
                        var acc3: Float = 0.0
                        let wBase = hIdx * totalR1
                        var i = 0
                        while i < totalR1 {
                            acc0 += pWR1C2[wBase + i] * pR1Hist[i]
                            acc1 += pWR1C2[wBase + i + 1] * pR1Hist[i + 1]
                            acc2 += pWR1C2[wBase + i + 2] * pR1Hist[i + 2]
                            acc3 += pWR1C2[wBase + i + 3] * pR1Hist[i + 3]
                            i += 4
                        }
                        pCurR1Out[hIdx] = (acc0 + acc1) + (acc2 + acc3)
                        hIdx += 1
                    }

                    // 10. 多重受容野 MRF ResBlock 2 (kernel 7) の実評価（4x Loop Unrolled 内積）
                    let totalR2 = 7 * hCh
                    hIdx = 0
                    while hIdx < hCh {
                        var acc0 = pBR2C1[hIdx]
                        var acc1: Float = 0.0
                        var acc2: Float = 0.0
                        var acc3: Float = 0.0
                        let wBase = hIdx * totalR2
                        var i = 0
                        while i < totalR2 {
                            acc0 += pWR2C1[wBase + i] * pH0Hist[i]
                            acc1 += pWR2C1[wBase + i + 1] * pH0Hist[i + 1]
                            acc2 += pWR2C1[wBase + i + 2] * pH0Hist[i + 2]
                            acc3 += pWR2C1[wBase + i + 3] * pH0Hist[i + 3]
                            i += 4
                        }
                        pCurR2Mid[hIdx] = Self.leakyRelu((acc0 + acc1) + (acc2 + acc3))
                        hIdx += 1
                    }

                    var shiftR2 = (6 * hCh) - 1
                    while 0 <= shiftR2 {
                        pR2Hist[shiftR2 + hCh] = pR2Hist[shiftR2]
                        shiftR2 -= 1
                    }
                    var cR2 = 0
                    while cR2 < hCh {
                        pR2Hist[cR2] = pCurR2Mid[cR2]
                        cR2 += 1
                    }

                    hIdx = 0
                    while hIdx < hCh {
                        var acc0 = pBR2C2[hIdx]
                        var acc1: Float = 0.0
                        var acc2: Float = 0.0
                        var acc3: Float = 0.0
                        let wBase = hIdx * totalR2
                        var i = 0
                        while i < totalR2 {
                            acc0 += pWR2C2[wBase + i] * pR2Hist[i]
                            acc1 += pWR2C2[wBase + i + 1] * pR2Hist[i + 1]
                            acc2 += pWR2C2[wBase + i + 2] * pR2Hist[i + 2]
                            acc3 += pWR2C2[wBase + i + 3] * pR2Hist[i + 3]
                            i += 4
                        }
                        pCurR2Out[hIdx] = (acc0 + acc1) + (acc2 + acc3)
                        hIdx += 1
                    }

                    // 11. MRF 残差結合: H_mrf = H_0 + ResBlock1 + ResBlock2
                    hIdx = 0
                    while hIdx < hCh {
                        pCurHmrf[hIdx] = pCurH0[hIdx] + pCurR1Out[hIdx] + pCurR2Out[hIdx]
                        hIdx += 1
                    }

                    var shiftPost = (6 * hCh) - 1
                    while 0 <= shiftPost {
                        pPostHist[shiftPost + hCh] = pPostHist[shiftPost]
                        shiftPost -= 1
                    }
                    var cPost = 0
                    while cPost < hCh {
                        pPostHist[cPost] = pCurHmrf[cPost]
                        cPost += 1
                    }

                    // 12. 終段 1D 畳み込み (`convPost`) の実評価（4x Loop Unrolled 内積）
                    var postAcc0 = bPostVal
                    var postAcc1: Float = 0.0
                    var postAcc2: Float = 0.0
                    var postAcc3: Float = 0.0
                    var iPost = 0
                    while iPost < totalR2 {
                        postAcc0 += pWPost[iPost] * pPostHist[iPost]
                        postAcc1 += pWPost[iPost + 1] * pPostHist[iPost + 1]
                        postAcc2 += pWPost[iPost + 2] * pPostHist[iPost + 2]
                        postAcc3 += pWPost[iPost + 3] * pPostHist[iPost + 3]
                        iPost += 4
                    }
                    let finalOut = (postAcc0 + postAcc1) + (postAcc2 + postAcc3)

                    // 13. NSF 音響物理共鳴とニューラル残差の統合
                    let neuralWave = finalOut * 0.35
                    let combined = (resWave * 1.0) + neuralWave
                    let rawVal = tanhf(combined * voice.energyScale * 1.25)
                    let dcR: Float = 0.995
                    let dcOut = rawVal - dcXPrev + (dcR * dcYPrev)
                    dcXPrev = rawVal
                    dcYPrev = dcOut

                    var sampleVal = dcOut
                    if sampleVal < -1.0 {
                        sampleVal = -1.0
                    }
                    if 1.0 < sampleVal {
                        sampleVal = 1.0
                    }

                    if interpVoiced <= 0.001 && interpActivity <= 0.001 {
                        pOut[startSample + s] = 0.0
                        dcXPrev = 0.0
                        dcYPrev = 0.0
                    } else {
                        pOut[startSample + s] = sampleVal
                    }

                    s += 1
                }

                curF0 = targetF0
                curVoiced = targetVoiced
                curActivity = targetActivity
                curMel = targetMel
                curF1 = targetF1
                curB1 = targetB1
                curF2 = targetF2
                curB2 = targetB2
                curF3 = targetF3
                curB3 = targetB3
                curF4 = targetF4
                curB4 = targetB4
                curGain = targetGain

                var chC = 0
                while chC < melCh {
                    pCurCen[chC] = pTgtCen[chC]
                    chC += 1
                }
                t += 1
            }
        }}}}}}}}}}}}}}}}}}}}}}}}}}

        // 次回フレーム合成時の補間連続性を維持
        hasLastFrame = true
        lastF0 = curF0
        lastVoiced = curVoiced
        lastActivity = curActivity
        lastMel = curMel
        lastF1 = curF1
        lastB1 = curB1
        lastF2 = curF2
        lastB2 = curB2
        lastF3 = curF3
        lastB3 = curB3
        lastF4 = curF4
        lastB4 = curB4
        lastGain = curGain

        return outputSamples
    }

    /// 1フレーム単位の逐次合成（低遅延ストリーミング対応）
    public func synthesizeFrame(
        melFrame: [Float],
        f0: Float = 0.0,
        voiced: Float = 1.0,
        voice: VoiceProfile = .female,
        resonatorFrame: ResonatorFrame? = nil,
        dst: UnsafeMutablePointer<Float>
    ) {
        let rFrames: [ResonatorFrame]
        switch resonatorFrame {
        case .some(let rf):
            rFrames = [rf]
        case .none:
            rFrames = []
        }
        let frameSamples = synthesize(
            mel: [melFrame],
            f0Contour: [f0],
            voicedFlags: [voiced],
            voice: voice,
            resonatorFrames: rFrames
        )
        let copyCount = min(config.hopSize, frameSamples.count)
        var i = 0
        while i < copyCount {
            dst[i] = frameSamples[i]
            i += 1
        }
    }
}
