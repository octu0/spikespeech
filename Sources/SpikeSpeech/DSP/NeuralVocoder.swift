import Foundation

/// 現代的ニューラルボコーダー（Neural Vocoder）の設定パラメータ
///
/// サンプリングレート、フレーム幅（Hop Size）、Mel チャンネル数、および
/// 隠れ層チャンネル数（64ch）と多重受容野（MRF）畳み込み構造を一元管理し、
/// 音響モデルおよび MLX 学習基盤との整合性を保証する。
public struct NeuralVocoderConfig: Sendable, Codable, Equatable {
    /// 音声サンプリングレート [Hz] (16,000 Hz)
    public let sampleRate: Int
    /// 1フレームあたりの PCM サンプル数 (160サンプル = 10ms)
    public let hopSize: Int
    /// 入力 Mel スペクトログラムの周波数チャンネル数 (64ch)
    public let melChannels: Int
    /// 隠れ層特徴量チャンネル数 (64ch)
    /// なぜ 64ch に設定するか:
    /// 16ch の玩具規模では 64ch Mel の音響情報・フォルマント包絡を保持できず、
    /// 64ch かつ SIMD8（8 レーン）ベクトル演算との整合性を最大化し、
    /// 高速な Pure Swift 推論と豊かな肉声表現力を両立するため。
    public let hiddenChannels: Int

    public init(
        sampleRate: Int = AudioConfig.sampleRate,
        hopSize: Int = AudioConfig.hopSize,
        melChannels: Int = AudioConfig.melChannels,
        hiddenChannels: Int = 64
    ) {
        self.sampleRate = sampleRate
        self.hopSize = hopSize
        self.melChannels = melChannels
        self.hiddenChannels = hiddenChannels
    }
}

/// 現代的ニューラルボコーダーの学習済み重みパラメータ
///
/// Pure Swift（CPU / SIMD8）と MLX（GPU / Apple Silicon）の間で
/// ゼロコピーかつ相互運用可能な平坦な Float 配列として保持し、
/// JSON 永続化とキャッシュ局所性を最大化する。
public struct NeuralVocoderWeights: Sendable, Codable, Equatable {
    public let config: NeuralVocoderConfig

    /// 初段 1D 畳み込み重み [hiddenChannels * kernelPre * inChannels] (kernel 7, inChannels 66)
    public let convPreWeight: [Float]
    public let convPreBias: [Float]

    /// アップサンプリング Stage 1 転置畳み込み重み [hiddenChannels * kernelUp1 * hiddenChannels] (stride 10, kernel 20)
    public let up1Weight: [Float]
    public let up1Bias: [Float]

    /// MRF 1 ResBlock 1 (kernel 3, dilation 1, 3)
    public let res1Conv1Weight: [Float]
    public let res1Conv1Bias: [Float]
    public let res1Conv2Weight: [Float]
    public let res1Conv2Bias: [Float]

    /// MRF 1 ResBlock 2 (kernel 7, dilation 1, 3)
    public let res2Conv1Weight: [Float]
    public let res2Conv1Bias: [Float]
    public let res2Conv2Weight: [Float]
    public let res2Conv2Bias: [Float]

    /// アップサンプリング Stage 2 転置畳み込み重み [hiddenChannels * kernelUp2 * hiddenChannels] (stride 4, kernel 8)
    public let up2Weight: [Float]
    public let up2Bias: [Float]

    /// MRF 2 ResBlock 1 (kernel 3, dilation 1, 3)
    public let res3Conv1Weight: [Float]
    public let res3Conv1Bias: [Float]
    public let res3Conv2Weight: [Float]
    public let res3Conv2Bias: [Float]

    /// MRF 2 ResBlock 2 (kernel 7, dilation 1, 3)
    public let res4Conv1Weight: [Float]
    public let res4Conv1Bias: [Float]
    public let res4Conv2Weight: [Float]
    public let res4Conv2Bias: [Float]

    /// アップサンプリング Stage 3 転置畳み込み重み [hiddenChannels * kernelUp3 * hiddenChannels] (stride 4, kernel 8)
    public let up3Weight: [Float]
    public let up3Bias: [Float]

    /// MRF 3 ResBlock 1 (kernel 3, dilation 1, 3)
    public let res5Conv1Weight: [Float]
    public let res5Conv1Bias: [Float]
    public let res5Conv2Weight: [Float]
    public let res5Conv2Bias: [Float]

    /// MRF 3 ResBlock 2 (kernel 7, dilation 1, 3)
    public let res6Conv1Weight: [Float]
    public let res6Conv1Bias: [Float]
    public let res6Conv2Weight: [Float]
    public let res6Conv2Bias: [Float]

    /// 終段波形出力 1D 畳み込み重み [1 * kernelPost * hiddenChannels] (kernel 7)
    public let convPostWeight: [Float]
    public let convPostBias: [Float]

    /// 後方互換性用（推論では未使用）
    public let harmonicWeight: [Float]
    public let harmonicBias: Float

    public init(
        config: NeuralVocoderConfig = NeuralVocoderConfig(),
        convPreWeight: [Float],
        convPreBias: [Float],
        up1Weight: [Float],
        up1Bias: [Float],
        res1Conv1Weight: [Float],
        res1Conv1Bias: [Float],
        res1Conv2Weight: [Float],
        res1Conv2Bias: [Float],
        res2Conv1Weight: [Float],
        res2Conv1Bias: [Float],
        res2Conv2Weight: [Float],
        res2Conv2Bias: [Float],
        up2Weight: [Float],
        up2Bias: [Float],
        res3Conv1Weight: [Float],
        res3Conv1Bias: [Float],
        res3Conv2Weight: [Float],
        res3Conv2Bias: [Float],
        res4Conv1Weight: [Float],
        res4Conv1Bias: [Float],
        res4Conv2Weight: [Float],
        res4Conv2Bias: [Float],
        up3Weight: [Float],
        up3Bias: [Float],
        res5Conv1Weight: [Float],
        res5Conv1Bias: [Float],
        res5Conv2Weight: [Float],
        res5Conv2Bias: [Float],
        res6Conv1Weight: [Float],
        res6Conv1Bias: [Float],
        res6Conv2Weight: [Float],
        res6Conv2Bias: [Float],
        convPostWeight: [Float],
        convPostBias: [Float],
        harmonicWeight: [Float] = [],
        harmonicBias: Float = 0.0
    ) {
        self.config = config
        self.convPreWeight = convPreWeight
        self.convPreBias = convPreBias
        self.up1Weight = up1Weight
        self.up1Bias = up1Bias
        self.res1Conv1Weight = res1Conv1Weight
        self.res1Conv1Bias = res1Conv1Bias
        self.res1Conv2Weight = res1Conv2Weight
        self.res1Conv2Bias = res1Conv2Bias
        self.res2Conv1Weight = res2Conv1Weight
        self.res2Conv1Bias = res2Conv1Bias
        self.res2Conv2Weight = res2Conv2Weight
        self.res2Conv2Bias = res2Conv2Bias
        self.up2Weight = up2Weight
        self.up2Bias = up2Bias
        self.res3Conv1Weight = res3Conv1Weight
        self.res3Conv1Bias = res3Conv1Bias
        self.res3Conv2Weight = res3Conv2Weight
        self.res3Conv2Bias = res3Conv2Bias
        self.res4Conv1Weight = res4Conv1Weight
        self.res4Conv1Bias = res4Conv1Bias
        self.res4Conv2Weight = res4Conv2Weight
        self.res4Conv2Bias = res4Conv2Bias
        self.up3Weight = up3Weight
        self.up3Bias = up3Bias
        self.res5Conv1Weight = res5Conv1Weight
        self.res5Conv1Bias = res5Conv1Bias
        self.res5Conv2Weight = res5Conv2Weight
        self.res5Conv2Bias = res5Conv2Bias
        self.res6Conv1Weight = res6Conv1Weight
        self.res6Conv1Bias = res6Conv1Bias
        self.res6Conv2Weight = res6Conv2Weight
        self.res6Conv2Bias = res6Conv2Bias
        self.convPostWeight = convPostWeight
        self.convPostBias = convPostBias
        self.harmonicWeight = harmonicWeight
        self.harmonicBias = harmonicBias
    }

    /// 決定論的疑似乱数による初期重みの生成
    ///
    /// なぜ He 正規分布初期化を採用するか:
    /// 未学習状態であっても数値発散・勾配消失を防ぎ、
    /// 多重受容野アップサンプリングにおいて振幅スケールを安定して維持するため。
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
        let inCh = melCh + 2 // Mel 64ch + F0 1ch + Voiced 1ch

        // 1. convPre: kernel 7, inCh -> hCh
        let preK = 7
        let preScale = sqrtf(2.0 / Float(inCh * preK)) * 1.0
        var preW = [Float](repeating: 0.0, count: hCh * preK * inCh)
        var i = 0
        while i < preW.count {
            preW[i] = nextUniform(scale: preScale)
            i += 1
        }
        let preB = [Float](repeating: 0.0, count: hCh)

        // 2. Stage 1: up1 (stride 10, kernel 20, padding 5)
        let up1K = 20
        let up1Scale = sqrtf(2.0 / Float(hCh * up1K)) * 1.0
        var up1W = [Float](repeating: 0.0, count: hCh * up1K * hCh)
        i = 0
        while i < up1W.count {
            up1W[i] = nextUniform(scale: up1Scale)
            i += 1
        }
        let up1B = [Float](repeating: 0.0, count: hCh)

        // MRF 1 ResBlock 1 (kernel 3)
        let r1Scale = sqrtf(2.0 / Float(hCh * 3)) * 0.4
        var r1c1W = [Float](repeating: 0.0, count: hCh * 3 * hCh)
        var r1c2W = [Float](repeating: 0.0, count: hCh * 3 * hCh)
        i = 0
        while i < r1c1W.count {
            r1c1W[i] = nextUniform(scale: r1Scale)
            r1c2W[i] = nextUniform(scale: r1Scale)
            i += 1
        }
        let r1c1B = [Float](repeating: 0.0, count: hCh)
        let r1c2B = [Float](repeating: 0.0, count: hCh)

        // MRF 1 ResBlock 2 (kernel 7)
        let r2Scale = sqrtf(2.0 / Float(hCh * 7)) * 0.4
        var r2c1W = [Float](repeating: 0.0, count: hCh * 7 * hCh)
        var r2c2W = [Float](repeating: 0.0, count: hCh * 7 * hCh)
        i = 0
        while i < r2c1W.count {
            r2c1W[i] = nextUniform(scale: r2Scale)
            r2c2W[i] = nextUniform(scale: r2Scale)
            i += 1
        }
        let r2c1B = [Float](repeating: 0.0, count: hCh)
        let r2c2B = [Float](repeating: 0.0, count: hCh)

        // 3. Stage 2: up2 (stride 4, kernel 8, padding 2)
        let up2K = 8
        let up2Scale = sqrtf(2.0 / Float(hCh * up2K)) * 1.0
        var up2W = [Float](repeating: 0.0, count: hCh * up2K * hCh)
        i = 0
        while i < up2W.count {
            up2W[i] = nextUniform(scale: up2Scale)
            i += 1
        }
        let up2B = [Float](repeating: 0.0, count: hCh)

        // MRF 2 ResBlock 1 (kernel 3)
        var r3c1W = [Float](repeating: 0.0, count: hCh * 3 * hCh)
        var r3c2W = [Float](repeating: 0.0, count: hCh * 3 * hCh)
        i = 0
        while i < r3c1W.count {
            r3c1W[i] = nextUniform(scale: r1Scale)
            r3c2W[i] = nextUniform(scale: r1Scale)
            i += 1
        }
        let r3c1B = [Float](repeating: 0.0, count: hCh)
        let r3c2B = [Float](repeating: 0.0, count: hCh)

        // MRF 2 ResBlock 2 (kernel 7)
        var r4c1W = [Float](repeating: 0.0, count: hCh * 7 * hCh)
        var r4c2W = [Float](repeating: 0.0, count: hCh * 7 * hCh)
        i = 0
        while i < r4c1W.count {
            r4c1W[i] = nextUniform(scale: r2Scale)
            r4c2W[i] = nextUniform(scale: r2Scale)
            i += 1
        }
        let r4c1B = [Float](repeating: 0.0, count: hCh)
        let r4c2B = [Float](repeating: 0.0, count: hCh)

        // 4. Stage 3: up3 (stride 4, kernel 8, padding 2)
        var up3W = [Float](repeating: 0.0, count: hCh * up2K * hCh)
        i = 0
        while i < up3W.count {
            up3W[i] = nextUniform(scale: up2Scale)
            i += 1
        }
        let up3B = [Float](repeating: 0.0, count: hCh)

        // MRF 3 ResBlock 1 (kernel 3)
        var r5c1W = [Float](repeating: 0.0, count: hCh * 3 * hCh)
        var r5c2W = [Float](repeating: 0.0, count: hCh * 3 * hCh)
        i = 0
        while i < r5c1W.count {
            r5c1W[i] = nextUniform(scale: r1Scale)
            r5c2W[i] = nextUniform(scale: r1Scale)
            i += 1
        }
        let r5c1B = [Float](repeating: 0.0, count: hCh)
        let r5c2B = [Float](repeating: 0.0, count: hCh)

        // MRF 3 ResBlock 2 (kernel 7)
        var r6c1W = [Float](repeating: 0.0, count: hCh * 7 * hCh)
        var r6c2W = [Float](repeating: 0.0, count: hCh * 7 * hCh)
        i = 0
        while i < r6c1W.count {
            r6c1W[i] = nextUniform(scale: r2Scale)
            r6c2W[i] = nextUniform(scale: r2Scale)
            i += 1
        }
        let r6c1B = [Float](repeating: 0.0, count: hCh)
        let r6c2B = [Float](repeating: 0.0, count: hCh)

        // 5. convPost: kernel 7, hCh -> 1
        let postK = 7
        let postScale = sqrtf(2.0 / Float(hCh * postK)) * 1.0
        var postW = [Float](repeating: 0.0, count: 1 * postK * hCh)
        i = 0
        while i < postW.count {
            postW[i] = nextUniform(scale: postScale)
            i += 1
        }
        let postB = [Float](repeating: 0.0, count: 1)

        return NeuralVocoderWeights(
            config: config,
            convPreWeight: preW,
            convPreBias: preB,
            up1Weight: up1W,
            up1Bias: up1B,
            res1Conv1Weight: r1c1W,
            res1Conv1Bias: r1c1B,
            res1Conv2Weight: r1c2W,
            res1Conv2Bias: r1c2B,
            res2Conv1Weight: r2c1W,
            res2Conv1Bias: r2c1B,
            res2Conv2Weight: r2c2W,
            res2Conv2Bias: r2c2B,
            up2Weight: up2W,
            up2Bias: up2B,
            res3Conv1Weight: r3c1W,
            res3Conv1Bias: r3c1B,
            res3Conv2Weight: r3c2W,
            res3Conv2Bias: r3c2B,
            res4Conv1Weight: r4c1W,
            res4Conv1Bias: r4c1B,
            res4Conv2Weight: r4c2W,
            res4Conv2Bias: r4c2B,
            up3Weight: up3W,
            up3Bias: up3B,
            res5Conv1Weight: r5c1W,
            res5Conv1Bias: r5c1B,
            res5Conv2Weight: r5c2W,
            res5Conv2Bias: r5c2B,
            res6Conv1Weight: r6c1W,
            res6Conv1Bias: r6c1B,
            res6Conv2Weight: r6c2W,
            res6Conv2Bias: r6c2B,
            convPostWeight: postW,
            convPostBias: postB
        )
    }
}

/// 現代的ニューラルボコーダー（Neural Vocoder）Pure Swift 推論エンジン
///
/// 従来の音響物理共鳴器（カスケード IIR フィルタ）や Rosenberg 声門パルス潜在加算を完全撤廃し、
/// 64チャンネル絶対 log-Mel スペクトログラムおよび連続 F0 / 有声フラグから、
/// 多重受容野（MRF: kernel 3, 7 × dilation 1, 3）残差ブロックと階層的転置畳み込みアップサンプリング
/// （10x, 4x, 4x = 160倍）によって直接時間領域 16kHz PCM 波形を高速生成する。
public final class NeuralVocoder: @unchecked Sendable {
    public let weights: NeuralVocoderWeights
    public let config: NeuralVocoderConfig

    // 作業用メモリバッファ（ホットパスでのヒープ確保ゼロ化）
    private var bufPre: [Float] = []
    private var bufUp1: [Float] = []
    private var bufMid1: [Float] = []
    private var bufR1: [Float] = []
    private var bufR2: [Float] = []
    private var bufUp2: [Float] = []
    private var bufMid2: [Float] = []
    private var bufR3: [Float] = []
    private var bufR4: [Float] = []
    private var bufUp3: [Float] = []
    private var bufMid3: [Float] = []
    private var bufR5: [Float] = []
    private var bufR6: [Float] = []

    // NSF (Neural Source-Filter) 位相蓄積器、乱流気流ノイズ乱数シード、および平滑化包絡線
    private var phase: Float = 0.0
    private var noiseRng: UInt32 = 20260917
    private var smoothEnv: Float = 0.0

    public init(weights: NeuralVocoderWeights? = nil) {
        let w: NeuralVocoderWeights
        switch weights {
        case .some(let explicit):
            w = explicit
        case .none:
            let defaultVocoderPath = "Models/vocoder_weights.json"
            var loaded: NeuralVocoderWeights? = nil
            if FileManager.default.fileExists(atPath: defaultVocoderPath) {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: defaultVocoderPath)) {
                    loaded = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: data)
                }
            }
            switch loaded {
            case .some(let vw):
                w = vw
            case .none:
                w = NeuralVocoderWeights.randomWeights()
            }
        }
        self.weights = w
        self.config = w.config
    }

    /// 内部推論バッファのリセット
    public func reset() {
        phase = 0.0
        noiseRng = 20260917
        smoothEnv = 0.0
    }

    @inline(__always)
    private static func leakyRelu(_ x: Float) -> Float {
        if x < 0.0 {
            return x * 0.1
        }
        return x
    }



    /// 1D 畳み込み演算（ゼロパディングおよび膨張係数 Dilation 対応）
    ///
    /// なぜ Dilation（膨張畳み込み）を導入するか:
    /// パラメータ数および積和演算量を増加させることなく受容野を指数関数的に拡大し、
    /// 16kHz PCM におけるピッチ周期（40〜160サンプル）と音韻遷移の微細時間構造を完全にカバーするため。
    /// 1D 畳み込み演算（ゼロパディングおよび膨張係数 Dilation 対応）
    ///
    /// なぜ境界領域と内部領域を分離（Hoist）し 64ch 展開を行うか:
    /// ループ深部から if 条件分岐を完全排除し、Hot Path 指針（ボイラープレートを恐れず内部・境界分離）に
    /// 従い、dotProduct64 の SIMD8 デュアルアキュムレータと組み合わせて演算スループットを最大化するため。
    private static func conv1d(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        T: Int,
        inC: Int,
        outC: Int,
        kernel: Int,
        padding: Int,
        dilation: Int = 1,
        weights: UnsafePointer<Float>,
        bias: UnsafePointer<Float>,
        applyActivation: Bool
    ) {
        let span = (kernel - 1) * dilation
        let innerStart = min(T, max(0, padding))
        let innerEnd = max(innerStart, min(T, max(0, T + padding - span)))

        // 1. 左境界（0 <= t < innerStart）: 境界検査付き
        var t = 0
        while t < innerStart {
            let outRow = t * outC
            var c = 0
            while c < outC {
                var sum = bias[c]
                let wRow = c * kernel * inC
                var k = 0
                while k < kernel {
                    let inT = t + (k * dilation) - padding
                    if 0 <= inT && inT < T {
                        let inRow = inT * inC
                        let wOffset = wRow + (k * inC)
                        sum += VectorOperations.dotProduct(
                            a: input.advanced(by: inRow),
                            b: weights.advanced(by: wOffset),
                            count: inC
                        )
                    }
                    k += 1
                }
                if applyActivation {
                    sum = leakyRelu(sum)
                }
                output[outRow + c] = sum
                c += 1
            }
            t += 1
        }

        // 2. 内部領域（innerStart <= t < innerEnd）: 条件分岐完全排除
        if inC == 64 {
            switch (kernel, outC) {
            case (3, 64):
                let d0 = 0 - padding
                let d1 = dilation - padding
                let d2 = (2 * dilation) - padding
                while t < innerEnd {
                    let inK0 = SIMD64Float(from: input.advanced(by: (t + d0) * 64))
                    let inK1 = SIMD64Float(from: input.advanced(by: (t + d1) * 64))
                    let inK2 = SIMD64Float(from: input.advanced(by: (t + d2) * 64))
                    let outRow = t * 64
                    var c = 0
                    while c < 64 {
                        let wRow = c * 192
                        var sum = bias[c]
                        sum += inK0.dot(with: weights.advanced(by: wRow))
                        sum += inK1.dot(with: weights.advanced(by: wRow + 64))
                        sum += inK2.dot(with: weights.advanced(by: wRow + 128))
                        if applyActivation {
                            sum = leakyRelu(sum)
                        }
                        output[outRow + c] = sum
                        c += 1
                    }
                    t += 1
                }
            case (7, 64):
                let d0 = 0 - padding
                let d1 = dilation - padding
                let d2 = (2 * dilation) - padding
                let d3 = (3 * dilation) - padding
                let d4 = (4 * dilation) - padding
                let d5 = (5 * dilation) - padding
                let d6 = (6 * dilation) - padding
                while t < innerEnd {
                    let inK0 = SIMD64Float(from: input.advanced(by: (t + d0) * 64))
                    let inK1 = SIMD64Float(from: input.advanced(by: (t + d1) * 64))
                    let inK2 = SIMD64Float(from: input.advanced(by: (t + d2) * 64))
                    let inK3 = SIMD64Float(from: input.advanced(by: (t + d3) * 64))
                    let inK4 = SIMD64Float(from: input.advanced(by: (t + d4) * 64))
                    let inK5 = SIMD64Float(from: input.advanced(by: (t + d5) * 64))
                    let inK6 = SIMD64Float(from: input.advanced(by: (t + d6) * 64))
                    let outRow = t * 64
                    var c = 0
                    while c < 64 {
                        let wRow = c * 448
                        var sum = bias[c]
                        sum += inK0.dot(with: weights.advanced(by: wRow))
                        sum += inK1.dot(with: weights.advanced(by: wRow + 64))
                        sum += inK2.dot(with: weights.advanced(by: wRow + 128))
                        sum += inK3.dot(with: weights.advanced(by: wRow + 192))
                        sum += inK4.dot(with: weights.advanced(by: wRow + 256))
                        sum += inK5.dot(with: weights.advanced(by: wRow + 320))
                        sum += inK6.dot(with: weights.advanced(by: wRow + 384))
                        if applyActivation {
                            sum = leakyRelu(sum)
                        }
                        output[outRow + c] = sum
                        c += 1
                    }
                    t += 1
                }
            default:
                while t < innerEnd {
                    let outRow = t * outC
                    var c = 0
                    while c < outC {
                        var sum = bias[c]
                        let wRow = c * kernel * inC
                        var k = 0
                        while k < kernel {
                            let inT = t + (k * dilation) - padding
                            let inRow = inT * inC
                            let wOffset = wRow + (k * inC)
                            sum += VectorOperations.dotProduct64(
                                a: input.advanced(by: inRow),
                                b: weights.advanced(by: wOffset)
                            )
                            k += 1
                        }
                        if applyActivation {
                            sum = leakyRelu(sum)
                        }
                        output[outRow + c] = sum
                        c += 1
                    }
                    t += 1
                }
            }
        } else {
            while t < innerEnd {
                let outRow = t * outC
                var c = 0
                while c < outC {
                    var sum = bias[c]
                    let wRow = c * kernel * inC
                    var k = 0
                    while k < kernel {
                        let inT = t + (k * dilation) - padding
                        let inRow = inT * inC
                        let wOffset = wRow + (k * inC)
                        sum += VectorOperations.dotProduct(
                            a: input.advanced(by: inRow),
                            b: weights.advanced(by: wOffset),
                            count: inC
                        )
                        k += 1
                    }
                    if applyActivation {
                        sum = leakyRelu(sum)
                    }
                    output[outRow + c] = sum
                    c += 1
                }
                t += 1
            }
        }

        // 3. 右境界（innerEnd <= t < T）: 境界検査付き
        while t < T {
            let outRow = t * outC
            var c = 0
            while c < outC {
                var sum = bias[c]
                let wRow = c * kernel * inC
                var k = 0
                while k < kernel {
                    let inT = t + (k * dilation) - padding
                    if 0 <= inT && inT < T {
                        let inRow = inT * inC
                        let wOffset = wRow + (k * inC)
                        sum += VectorOperations.dotProduct(
                            a: input.advanced(by: inRow),
                            b: weights.advanced(by: wOffset),
                            count: inC
                        )
                    }
                    k += 1
                }
                if applyActivation {
                    sum = leakyRelu(sum)
                }
                output[outRow + c] = sum
                c += 1
            }
            t += 1
        }
    }

    /// 1D 転置畳み込み演算（階層的アップサンプリング）
    ///
    /// なぜ偶数カーネルかつ 2 * stride の線形補間型転置畳み込みを採用するか:
    /// 奇数カーネルや不整合なストライドで発生するチェッカーボード歪み（エイリアシングノイズ）を物理的に抑制し、
    /// 境界分離および dotProduct64 によりリアルタイム係数を大幅に削減するため。
    private static func convTransposed1d(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        inT: Int,
        stride: Int,
        inC: Int,
        outC: Int,
        weights: UnsafePointer<Float>,
        bias: UnsafePointer<Float>,
        applyActivation: Bool
    ) {
        let outT = inT * stride
        let padding = stride / 2
        let K = 2 * stride

        let innerStart = min(outT, max(0, stride - padding))
        let innerEnd = max(innerStart, min(outT, max(0, outT - padding)))

        var tau = 0
        // 1. 左境界
        while tau < innerStart {
            let q = (tau + padding) / stride
            let r = (tau + padding) % stride
            let t0 = q
            let k0 = r
            let t1 = q - 1
            let k1 = r + stride
            let outRow = tau * outC
            var c = 0
            while c < outC {
                var sum = bias[c]
                let wBase = c * K * inC
                if 0 <= t0 && t0 < inT {
                    sum += VectorOperations.dotProduct(a: input.advanced(by: t0 * inC), b: weights.advanced(by: wBase + (k0 * inC)), count: inC)
                }
                if 0 <= t1 && t1 < inT {
                    sum += VectorOperations.dotProduct(a: input.advanced(by: t1 * inC), b: weights.advanced(by: wBase + (k1 * inC)), count: inC)
                }
                if applyActivation { sum = leakyRelu(sum) }
                output[outRow + c] = sum
                c += 1
            }
            tau += 1
        }

        // 2. 内部領域（境界検査不要）
        switch (inC, outC) {
        case (64, 64):
            let kStride = K * 64
            while tau < innerEnd {
                let q = (tau + padding) / stride
                let r = (tau + padding) % stride
                let in0 = SIMD64Float(from: input.advanced(by: q * 64))
                let in1 = SIMD64Float(from: input.advanced(by: (q - 1) * 64))
                let k0Offset = r * 64
                let k1Offset = (r + stride) * 64
                let outRow = tau * 64
                var c = 0
                while c < 64 {
                    let wBase = c * kStride
                    var sum = bias[c]
                    sum += in0.dot(with: weights.advanced(by: wBase + k0Offset))
                    sum += in1.dot(with: weights.advanced(by: wBase + k1Offset))
                    if applyActivation { sum = leakyRelu(sum) }
                    output[outRow + c] = sum
                    c += 1
                }
                tau += 1
            }
        case (64, _):
            while tau < innerEnd {
                let q = (tau + padding) / stride
                let r = (tau + padding) % stride
                let t0 = q
                let k0 = r
                let t1 = q - 1
                let k1 = r + stride
                let inRow0 = t0 * inC
                let inRow1 = t1 * inC
                let outRow = tau * outC
                var c = 0
                while c < outC {
                    let wBase = c * K * inC
                    var sum = bias[c]
                    sum += VectorOperations.dotProduct64(a: input.advanced(by: inRow0), b: weights.advanced(by: wBase + (k0 * inC)))
                    sum += VectorOperations.dotProduct64(a: input.advanced(by: inRow1), b: weights.advanced(by: wBase + (k1 * inC)))
                    if applyActivation { sum = leakyRelu(sum) }
                    output[outRow + c] = sum
                    c += 1
                }
                tau += 1
            }
        default:
            while tau < innerEnd {
                let q = (tau + padding) / stride
                let r = (tau + padding) % stride
                let t0 = q
                let k0 = r
                let t1 = q - 1
                let k1 = r + stride
                let inRow0 = t0 * inC
                let inRow1 = t1 * inC
                let outRow = tau * outC
                var c = 0
                while c < outC {
                    let wBase = c * K * inC
                    var sum = bias[c]
                    sum += VectorOperations.dotProduct(a: input.advanced(by: inRow0), b: weights.advanced(by: wBase + (k0 * inC)), count: inC)
                    sum += VectorOperations.dotProduct(a: input.advanced(by: inRow1), b: weights.advanced(by: wBase + (k1 * inC)), count: inC)
                    if applyActivation { sum = leakyRelu(sum) }
                    output[outRow + c] = sum
                    c += 1
                }
                tau += 1
            }
        }

        // 3. 右境界
        while tau < outT {
            let q = (tau + padding) / stride
            let r = (tau + padding) % stride
            let t0 = q
            let k0 = r
            let t1 = q - 1
            let k1 = r + stride
            let outRow = tau * outC
            var c = 0
            while c < outC {
                var sum = bias[c]
                let wBase = c * K * inC
                if 0 <= t0 && t0 < inT {
                    sum += VectorOperations.dotProduct(a: input.advanced(by: t0 * inC), b: weights.advanced(by: wBase + (k0 * inC)), count: inC)
                }
                if 0 <= t1 && t1 < inT {
                    sum += VectorOperations.dotProduct(a: input.advanced(by: t1 * inC), b: weights.advanced(by: wBase + (k1 * inC)), count: inC)
                }
                if applyActivation { sum = leakyRelu(sum) }
                output[outRow + c] = sum
                c += 1
            }
            tau += 1
        }
    }

    /// Mel スペクトログラム系列および F0 輪郭から時間領域 16kHz PCM 波形を直接合成する
    public func synthesize(
        mel: [[Float]],
        f0Contour: [Float] = [],
        voicedFlags: [Float] = [],
        voice: VoiceProfile = .female
    ) -> [Float] {
        let totalFrames = mel.count
        if totalFrames <= 0 {
            return []
        }

        // なぜ 250 フレーム単位でチャンク分割推論を行うか:
        // 超長文合成時（数万フレーム）に内部テンソルバッファの過大確保を抑制し、
        // メモリ制約を確実に遵守しながら受容野境界を滑らかに接続するため。
        let maxChunkFrames = 250
        if totalFrames <= maxChunkFrames {
            return synthesizeChunk(
                mel: mel,
                f0Contour: f0Contour,
                voicedFlags: voicedFlags,
                voice: voice
            )
        }

        let hopSize = config.hopSize
        let totalSamples = totalFrames * hopSize
        var outputAudio = [Float](repeating: 0.0, count: totalSamples)

        let padFrames = 4
        var chunkStart = 0
        while chunkStart < totalFrames {
            let validStart = chunkStart
            let validEnd = min(totalFrames, chunkStart + maxChunkFrames)

            let padLeft = max(0, validStart - padFrames)
            let padRight = min(totalFrames, validEnd + padFrames)

            var chunkMel = [[Float]]()
            chunkMel.reserveCapacity(padRight - padLeft)
            var chunkF0 = [Float]()
            chunkF0.reserveCapacity(padRight - padLeft)
            var chunkVoiced = [Float]()
            chunkVoiced.reserveCapacity(padRight - padLeft)

            var f = padLeft
            while f < padRight {
                chunkMel.append(mel[f])
                var f0Val = voice.baseF0
                if f < f0Contour.count {
                    let fVal = f0Contour[f]
                    if 0.0 < fVal { f0Val = fVal }
                }
                chunkF0.append(f0Val)

                var vVal: Float = 1.0
                if f < voicedFlags.count {
                    vVal = voicedFlags[f]
                }
                chunkVoiced.append(vVal)
                f += 1
            }

            let chunkAudio = synthesizeChunk(
                mel: chunkMel,
                f0Contour: chunkF0,
                voicedFlags: chunkVoiced,
                voice: voice
            )

            let trimStartSamples = (validStart - padLeft) * hopSize
            let validCount = (validEnd - validStart) * hopSize
            let outStartSamples = validStart * hopSize

            var s = 0
            while s < validCount {
                let srcIdx = trimStartSamples + s
                let dstIdx = outStartSamples + s
                if srcIdx < chunkAudio.count && dstIdx < totalSamples {
                    outputAudio[dstIdx] = chunkAudio[srcIdx]
                }
                s += 1
            }

            chunkStart += maxChunkFrames
        }

        return outputAudio
    }

    /// 単一チャンクに対する現代的ニューラルボコーダー推論
    private func synthesizeChunk(
        mel: [[Float]],
        f0Contour: [Float] = [],
        voicedFlags: [Float] = [],
        voice: VoiceProfile = .female
    ) -> [Float] {
        let totalFrames = mel.count
        if totalFrames <= 0 {
            return []
        }

        let melCh = config.melChannels
        let hCh = config.hiddenChannels
        let inCh = melCh + 2 // Mel 64ch + F0 1ch + Voiced 1ch

        // 1. 入力特徴量テンソルの平坦化 [totalFrames * inCh]
        var inputFeats = [Float](repeating: 0.0, count: totalFrames * inCh)
        var t = 0
        while t < totalFrames {
            let rowStart = t * inCh
            let frameMel = mel[t]

            var c = 0
            let copyLimit = min(melCh, frameMel.count)
            while c < copyLimit {
                inputFeats[rowStart + c] = frameMel[c]
                c += 1
            }

            var vVal: Float = 1.0
            if t < voicedFlags.count {
                vVal = voicedFlags[t]
            }
            if vVal < 0.0 { vVal = 0.0 }
            if 1.0 < vVal { vVal = 1.0 }

            var normF0: Float = 0.0
            if 0.5 <= vVal {
                var f0Val: Float = voice.baseF0
                if t < f0Contour.count {
                    let f = f0Contour[t]
                    if 0.0 < f {
                        f0Val = f
                    }
                }
                var nF0 = f0Val / 500.0
                if nF0 < 0.0 { nF0 = 0.0 }
                if 1.0 < nF0 { nF0 = 1.0 }
                normF0 = nF0
            }

            inputFeats[rowStart + melCh] = normF0
            inputFeats[rowStart + melCh + 1] = vVal

            t += 1
        }

        let totalSamples = totalFrames * config.hopSize
        var outputAudio = [Float](repeating: 0.0, count: totalSamples)

        // 2. メモリバッファの確保（ゼロアロケーション）
        let lenPre = totalFrames * hCh
        if bufPre.count < lenPre { bufPre = [Float](repeating: 0.0, count: lenPre) }

        let T1 = totalFrames * 10
        let lenUp1 = T1 * hCh
        if bufUp1.count < lenUp1 { bufUp1 = [Float](repeating: 0.0, count: lenUp1) }
        if bufMid1.count < lenUp1 { bufMid1 = [Float](repeating: 0.0, count: lenUp1) }
        if bufR1.count < lenUp1 { bufR1 = [Float](repeating: 0.0, count: lenUp1) }
        if bufR2.count < lenUp1 { bufR2 = [Float](repeating: 0.0, count: lenUp1) }

        let T2 = T1 * 4
        let lenUp2 = T2 * hCh
        if bufUp2.count < lenUp2 { bufUp2 = [Float](repeating: 0.0, count: lenUp2) }
        if bufMid2.count < lenUp2 { bufMid2 = [Float](repeating: 0.0, count: lenUp2) }
        if bufR3.count < lenUp2 { bufR3 = [Float](repeating: 0.0, count: lenUp2) }
        if bufR4.count < lenUp2 { bufR4 = [Float](repeating: 0.0, count: lenUp2) }

        let T3 = T2 * 4 // totalSamples
        let lenUp3 = T3 * hCh
        if bufUp3.count < lenUp3 { bufUp3 = [Float](repeating: 0.0, count: lenUp3) }
        if bufMid3.count < lenUp3 { bufMid3 = [Float](repeating: 0.0, count: lenUp3) }
        if bufR5.count < lenUp3 { bufR5 = [Float](repeating: 0.0, count: lenUp3) }
        if bufR6.count < lenUp3 { bufR6 = [Float](repeating: 0.0, count: lenUp3) }

        inputFeats.withUnsafeBufferPointer { pIn in
            bufPre.withUnsafeMutableBufferPointer { pPre in
                bufUp1.withUnsafeMutableBufferPointer { pUp1 in
                    bufMid1.withUnsafeMutableBufferPointer { pMid1 in
                        bufR1.withUnsafeMutableBufferPointer { pR1 in
                            bufR2.withUnsafeMutableBufferPointer { pR2 in
                                bufUp2.withUnsafeMutableBufferPointer { pUp2 in
                                    bufMid2.withUnsafeMutableBufferPointer { pMid2 in
                                        bufR3.withUnsafeMutableBufferPointer { pR3 in
                                            bufR4.withUnsafeMutableBufferPointer { pR4 in
                                                bufUp3.withUnsafeMutableBufferPointer { pUp3 in
                                                    bufMid3.withUnsafeMutableBufferPointer { pMid3 in
                                                        bufR5.withUnsafeMutableBufferPointer { pR5 in
                                                            bufR6.withUnsafeMutableBufferPointer { pR6 in
                                                                outputAudio.withUnsafeMutableBufferPointer { pOut in
                                                                    let w = self.weights

                                                                    // 1. convPre: kernel 7, inCh -> hCh, LeakyReLU
                                                                    w.convPreWeight.withUnsafeBufferPointer { pWPre in
                                                                        w.convPreBias.withUnsafeBufferPointer { pBPre in
                                                                            Self.conv1d(
                                                                                input: pIn.baseAddress!,
                                                                                output: pPre.baseAddress!,
                                                                                T: totalFrames,
                                                                                inC: inCh,
                                                                                outC: hCh,
                                                                                kernel: 7,
                                                                                padding: 3,
                                                                                dilation: 1,
                                                                                weights: pWPre.baseAddress!,
                                                                                bias: pBPre.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }

                                                                    // 2. Stage 1: up1 (stride 10, kernel 20, padding 5, hCh -> hCh)
                                                                    w.up1Weight.withUnsafeBufferPointer { pWUp1 in
                                                                        w.up1Bias.withUnsafeBufferPointer { pBUp1 in
                                                                            Self.convTransposed1d(
                                                                                input: pPre.baseAddress!,
                                                                                output: pUp1.baseAddress!,
                                                                                inT: totalFrames,
                                                                                stride: 10,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                weights: pWUp1.baseAddress!,
                                                                                bias: pBUp1.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 1 ResBlock 1 (kernel 3, dilation 1, 3)
                                                                    w.res1Conv1Weight.withUnsafeBufferPointer { pWR1C1 in
                                                                        w.res1Conv1Bias.withUnsafeBufferPointer { pBR1C1 in
                                                                            Self.conv1d(
                                                                                input: pUp1.baseAddress!,
                                                                                output: pMid1.baseAddress!,
                                                                                T: T1,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 3,
                                                                                padding: 1,
                                                                                dilation: 1,
                                                                                weights: pWR1C1.baseAddress!,
                                                                                bias: pBR1C1.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }
                                                                    w.res1Conv2Weight.withUnsafeBufferPointer { pWR1C2 in
                                                                        w.res1Conv2Bias.withUnsafeBufferPointer { pBR1C2 in
                                                                            Self.conv1d(
                                                                                input: pMid1.baseAddress!,
                                                                                output: pR1.baseAddress!,
                                                                                T: T1,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 3,
                                                                                padding: 3,
                                                                                dilation: 3,
                                                                                weights: pWR1C2.baseAddress!,
                                                                                bias: pBR1C2.baseAddress!,
                                                                                applyActivation: false
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 1 ResBlock 2 (kernel 7, dilation 1, 3)
                                                                    w.res2Conv1Weight.withUnsafeBufferPointer { pWR2C1 in
                                                                        w.res2Conv1Bias.withUnsafeBufferPointer { pBR2C1 in
                                                                            Self.conv1d(
                                                                                input: pUp1.baseAddress!,
                                                                                output: pMid1.baseAddress!,
                                                                                T: T1,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 7,
                                                                                padding: 3,
                                                                                dilation: 1,
                                                                                weights: pWR2C1.baseAddress!,
                                                                                bias: pBR2C1.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }
                                                                    w.res2Conv2Weight.withUnsafeBufferPointer { pWR2C2 in
                                                                        w.res2Conv2Bias.withUnsafeBufferPointer { pBR2C2 in
                                                                            Self.conv1d(
                                                                                input: pMid1.baseAddress!,
                                                                                output: pR2.baseAddress!,
                                                                                T: T1,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 7,
                                                                                padding: 9,
                                                                                dilation: 3,
                                                                                weights: pWR2C2.baseAddress!,
                                                                                bias: pBR2C2.baseAddress!,
                                                                                applyActivation: false
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 1 残差結合: h1 = h1 + 0.5 * (r1 + r2)
                                                                    var idx = 0
                                                                    while idx < lenUp1 {
                                                                        pUp1[idx] = pUp1[idx] + (0.5 * (pR1[idx] + pR2[idx]))
                                                                        idx += 1
                                                                    }

                                                                    // 3. Stage 2: up2 (stride 4, kernel 8, padding 2, hCh -> hCh)
                                                                    w.up2Weight.withUnsafeBufferPointer { pWUp2 in
                                                                        w.up2Bias.withUnsafeBufferPointer { pBUp2 in
                                                                            Self.convTransposed1d(
                                                                                input: pUp1.baseAddress!,
                                                                                output: pUp2.baseAddress!,
                                                                                inT: T1,
                                                                                stride: 4,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                weights: pWUp2.baseAddress!,
                                                                                bias: pBUp2.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 2 ResBlock 1 (kernel 3, dilation 1, 3)
                                                                    w.res3Conv1Weight.withUnsafeBufferPointer { pWR3C1 in
                                                                        w.res3Conv1Bias.withUnsafeBufferPointer { pBR3C1 in
                                                                            Self.conv1d(
                                                                                input: pUp2.baseAddress!,
                                                                                output: pMid2.baseAddress!,
                                                                                T: T2,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 3,
                                                                                padding: 1,
                                                                                dilation: 1,
                                                                                weights: pWR3C1.baseAddress!,
                                                                                bias: pBR3C1.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }
                                                                    w.res3Conv2Weight.withUnsafeBufferPointer { pWR3C2 in
                                                                        w.res3Conv2Bias.withUnsafeBufferPointer { pBR3C2 in
                                                                            Self.conv1d(
                                                                                input: pMid2.baseAddress!,
                                                                                output: pR3.baseAddress!,
                                                                                T: T2,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 3,
                                                                                padding: 3,
                                                                                dilation: 3,
                                                                                weights: pWR3C2.baseAddress!,
                                                                                bias: pBR3C2.baseAddress!,
                                                                                applyActivation: false
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 2 ResBlock 2 (kernel 7, dilation 1, 3)
                                                                    w.res4Conv1Weight.withUnsafeBufferPointer { pWR4C1 in
                                                                        w.res4Conv1Bias.withUnsafeBufferPointer { pBR4C1 in
                                                                            Self.conv1d(
                                                                                input: pUp2.baseAddress!,
                                                                                output: pMid2.baseAddress!,
                                                                                T: T2,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 7,
                                                                                padding: 3,
                                                                                dilation: 1,
                                                                                weights: pWR4C1.baseAddress!,
                                                                                bias: pBR4C1.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }
                                                                    w.res4Conv2Weight.withUnsafeBufferPointer { pWR4C2 in
                                                                        w.res4Conv2Bias.withUnsafeBufferPointer { pBR4C2 in
                                                                            Self.conv1d(
                                                                                input: pMid2.baseAddress!,
                                                                                output: pR4.baseAddress!,
                                                                                T: T2,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 7,
                                                                                padding: 9,
                                                                                dilation: 3,
                                                                                weights: pWR4C2.baseAddress!,
                                                                                bias: pBR4C2.baseAddress!,
                                                                                applyActivation: false
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 2 残差結合: h2 = h2 + 0.5 * (r3 + r4)
                                                                    idx = 0
                                                                    while idx < lenUp2 {
                                                                        pUp2[idx] = pUp2[idx] + (0.5 * (pR3[idx] + pR4[idx]))
                                                                        idx += 1
                                                                    }

                                                                    // 4. Stage 3: up3 (stride 4, kernel 8, padding 2, hCh -> hCh)
                                                                    w.up3Weight.withUnsafeBufferPointer { pWUp3 in
                                                                        w.up3Bias.withUnsafeBufferPointer { pBUp3 in
                                                                            Self.convTransposed1d(
                                                                                input: pUp2.baseAddress!,
                                                                                output: pUp3.baseAddress!,
                                                                                inT: T2,
                                                                                stride: 4,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                weights: pWUp3.baseAddress!,
                                                                                bias: pBUp3.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 3 ResBlock 1 (kernel 3, dilation 1, 3)
                                                                    w.res5Conv1Weight.withUnsafeBufferPointer { pWR5C1 in
                                                                        w.res5Conv1Bias.withUnsafeBufferPointer { pBR5C1 in
                                                                            Self.conv1d(
                                                                                input: pUp3.baseAddress!,
                                                                                output: pMid3.baseAddress!,
                                                                                T: T3,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 3,
                                                                                padding: 1,
                                                                                dilation: 1,
                                                                                weights: pWR5C1.baseAddress!,
                                                                                bias: pBR5C1.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }
                                                                    w.res5Conv2Weight.withUnsafeBufferPointer { pWR5C2 in
                                                                        w.res5Conv2Bias.withUnsafeBufferPointer { pBR5C2 in
                                                                            Self.conv1d(
                                                                                input: pMid3.baseAddress!,
                                                                                output: pR5.baseAddress!,
                                                                                T: T3,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 3,
                                                                                padding: 3,
                                                                                dilation: 3,
                                                                                weights: pWR5C2.baseAddress!,
                                                                                bias: pBR5C2.baseAddress!,
                                                                                applyActivation: false
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 3 ResBlock 2 (kernel 7, dilation 1, 3)
                                                                    w.res6Conv1Weight.withUnsafeBufferPointer { pWR6C1 in
                                                                        w.res6Conv1Bias.withUnsafeBufferPointer { pBR6C1 in
                                                                            Self.conv1d(
                                                                                input: pUp3.baseAddress!,
                                                                                output: pMid3.baseAddress!,
                                                                                T: T3,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 7,
                                                                                padding: 3,
                                                                                dilation: 1,
                                                                                weights: pWR6C1.baseAddress!,
                                                                                bias: pBR6C1.baseAddress!,
                                                                                applyActivation: true
                                                                            )
                                                                        }
                                                                    }
                                                                    w.res6Conv2Weight.withUnsafeBufferPointer { pWR6C2 in
                                                                        w.res6Conv2Bias.withUnsafeBufferPointer { pBR6C2 in
                                                                            Self.conv1d(
                                                                                input: pMid3.baseAddress!,
                                                                                output: pR6.baseAddress!,
                                                                                T: T3,
                                                                                inC: hCh,
                                                                                outC: hCh,
                                                                                kernel: 7,
                                                                                padding: 9,
                                                                                dilation: 3,
                                                                                weights: pWR6C2.baseAddress!,
                                                                                bias: pBR6C2.baseAddress!,
                                                                                applyActivation: false
                                                                            )
                                                                        }
                                                                    }

                                                                    // MRF 3 残差結合: h3 = h3 + 0.5 * (r5 + r6)
                                                                    idx = 0
                                                                    while idx < lenUp3 {
                                                                        pUp3[idx] = pUp3[idx] + (0.5 * (pR5[idx] + pR6[idx]))
                                                                        idx += 1
                                                                    }

                                                                    // 5. 終段波形出力 1D 畳み込み層（純粋 HiFi-GAN MRF ニューラル波形生成）
                                                                    // なぜ正弦波やパルスの直接加算を完全撤廃するか:
                                                                    // 人工的な電子ビープ音・モデム音・発振音を根絶し、Mel 特徴量および F0 輪郭から
                                                                    // 畳み込みネットワークの受容野結合によって人間の自然な肉声波形を直接生成するため。
                                                                    w.convPostWeight.withUnsafeBufferPointer { pWPost in
                                                                        w.convPostBias.withUnsafeBufferPointer { pBPost in
                                                                            var s = 0
                                                                            while s < T3 {
                                                                                var sum = pBPost[0]
                                                                                var k = 0
                                                                                while k < 7 {
                                                                                    let inS = s + k - 3
                                                                                    if 0 <= inS && inS < T3 {
                                                                                        let inRow = inS * hCh
                                                                                        let wRow = k * hCh
                                                                                        sum += VectorOperations.dotProduct(
                                                                                            a: pUp3.baseAddress!.advanced(by: inRow),
                                                                                            b: pWPost.baseAddress!.advanced(by: wRow),
                                                                                            count: hCh
                                                                                        )
                                                                                    }
                                                                                    k += 1
                                                                                }
                                                                                pOut[s] = tanhf(sum)
                                                                                s += 1
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return outputAudio
    }

    /// 単一フレームの逐次波形合成（低遅延ストリーミング出力用）
    public func synthesizeFrame(
        melFrame: [Float],
        f0: Float = 220.0,
        voiced: Float = 1.0,
        voice: VoiceProfile = .female,
        dst: UnsafeMutablePointer<Float>
    ) {
        let pcm = synthesize(
            mel: [melFrame],
            f0Contour: [f0],
            voicedFlags: [voiced],
            voice: voice
        )
        let copyCount = min(config.hopSize, pcm.count)
        var i = 0
        while i < copyCount {
            dst[i] = pcm[i]
            i += 1
        }
    }
}

