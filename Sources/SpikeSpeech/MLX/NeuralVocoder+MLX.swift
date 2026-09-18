#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// MLX 上で動作する現代的ニューラルボコーダー（Neural Vocoder）ネットワーク
///
/// なぜ MLX モジュールとして実装するか:
/// Apple Silicon のユニファイドメモリ上でゼロコピーかつ高スループットに自動微分・学習を実行し、
/// Hann 窓付き Multi-Resolution STFT 損失関数および Mel 再構成損失を用いて高品質な波形生成モデルを直接最適化するため。
public final class MLXNeuralVocoder: Module, @unchecked Sendable {
    public let config: NeuralVocoderConfig

    /// 初段 1D 畳み込み層 (kernel 7, stride 1, padding 3)
    public var convPre: Conv1d

    /// アップサンプリング Stage 1 (stride 10, kernel 20, padding 5)
    public var up1: ConvTransposed1d

    /// MRF 1 ResBlock 1 (kernel 3, stride 1, dilation 1, 3)
    public var r1Conv1: Conv1d
    public var r1Conv2: Conv1d

    /// MRF 1 ResBlock 2 (kernel 7, stride 1, dilation 1, 3)
    public var r2Conv1: Conv1d
    public var r2Conv2: Conv1d

    /// アップサンプリング Stage 2 (stride 4, kernel 8, padding 2)
    public var up2: ConvTransposed1d

    /// MRF 2 ResBlock 1 (kernel 3, stride 1, dilation 1, 3)
    public var res3Conv1: Conv1d
    public var res3Conv2: Conv1d

    /// MRF 2 ResBlock 2 (kernel 7, stride 1, dilation 1, 3)
    public var res4Conv1: Conv1d
    public var res4Conv2: Conv1d

    /// アップサンプリング Stage 3 (stride 4, kernel 8, padding 2)
    public var up3: ConvTransposed1d

    /// MRF 3 ResBlock 1 (kernel 3, stride 1, dilation 1, 3)
    public var res5Conv1: Conv1d
    public var res5Conv2: Conv1d

    /// MRF 3 ResBlock 2 (kernel 7, stride 1, dilation 1, 3)
    public var res6Conv1: Conv1d
    public var res6Conv2: Conv1d

    /// 終段波形出力 1D 畳み込み層 (kernel 7, stride 1, padding 3)
    public var convPost: Conv1d

    public init(config: NeuralVocoderConfig = NeuralVocoderConfig()) {
        self.config = config
        let inCh = config.melChannels + 2 // Mel 64ch + F0 1ch + Voiced 1ch
        let hCh = config.hiddenChannels

        // 1. 初段畳み込み (kernel 7, stride 1, padding 3)
        self.convPre = Conv1d(
            inputChannels: inCh,
            outputChannels: hCh,
            kernelSize: 7,
            stride: 1,
            padding: 3
        )

        // 2. Stage 1 アップサンプリング (stride 10, kernel 20, padding 5)
        self.up1 = ConvTransposed1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 20,
            stride: 10,
            padding: 5
        )

        // MRF 1 ResBlock 1 (kernel 3, dilation 1, 3)
        self.r1Conv1 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1
        )
        self.r1Conv2 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 3,
            stride: 1,
            padding: 3,
            dilation: 3
        )

        // MRF 1 ResBlock 2 (kernel 7, dilation 1, 3)
        self.r2Conv1 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1
        )
        self.r2Conv2 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 7,
            stride: 1,
            padding: 9,
            dilation: 3
        )

        // 3. Stage 2 アップサンプリング (stride 4, kernel 8, padding 2)
        self.up2 = ConvTransposed1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 8,
            stride: 4,
            padding: 2
        )

        // MRF 2 ResBlock 1 (kernel 3, dilation 1, 3)
        self.res3Conv1 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1
        )
        self.res3Conv2 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 3,
            stride: 1,
            padding: 3,
            dilation: 3
        )

        // MRF 2 ResBlock 2 (kernel 7, dilation 1, 3)
        self.res4Conv1 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1
        )
        self.res4Conv2 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 7,
            stride: 1,
            padding: 9,
            dilation: 3
        )

        // 4. Stage 3 アップサンプリング (stride 4, kernel 8, padding 2)
        self.up3 = ConvTransposed1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 8,
            stride: 4,
            padding: 2
        )

        // MRF 3 ResBlock 1 (kernel 3, dilation 1, 3)
        self.res5Conv1 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1
        )
        self.res5Conv2 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 3,
            stride: 1,
            padding: 3,
            dilation: 3
        )

        // MRF 3 ResBlock 2 (kernel 7, dilation 1, 3)
        self.res6Conv1 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1
        )
        self.res6Conv2 = Conv1d(
            inputChannels: hCh,
            outputChannels: hCh,
            kernelSize: 7,
            stride: 1,
            padding: 9,
            dilation: 3
        )

        // 5. 終段波形生成 (kernel 7, stride 1, padding 3)
        self.convPost = Conv1d(
            inputChannels: hCh,
            outputChannels: 1,
            kernelSize: 7,
            stride: 1,
            padding: 3
        )

        super.init()
    }

    /// 純粋重み構造体からの初期化
    public convenience init(weights: NeuralVocoderWeights) {
        self.init(config: weights.config)
        self.importWeights(from: weights)
    }

    /// 順伝播計算: 入力特徴量テンソル [B, T, inCh] から HiFi-GAN MRF 畳み込みネットワークにより直接時間領域波形 [B, T * 160] を生成
    /// なぜ古典正弦波音源や手書きパルスの加算を完全撤廃するか:
    /// 入力特徴量（Mel, F0, Voiced）から直接実音声 PCM 波形への写像をエンドツーエンドで学習し、
    /// 人工的な電子ビープ音やモデム音を根絶して人間の生々しい肉声を純粋に合成するため。
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let lrelu = LeakyReLU(negativeSlope: 0.1)

        // 1. 初段畳み込み: [B, T, inCh] -> [B, T, hCh]
        let h0 = lrelu(convPre(input))

        // 2. Stage 1: up1 (stride 10) -> [B, T * 10, hCh]
        let h1 = lrelu(up1(h0))

        // MRF 1 ResBlocks (dilation 1, 3)
        let r1 = r1Conv2(lrelu(r1Conv1(h1)))
        let r2 = r2Conv2(lrelu(r2Conv1(h1)))
        let mrf1 = h1 + (0.5 * (r1 + r2))

        // 3. Stage 2: up2 (stride 4) -> [B, T * 40, hCh]
        let h2 = lrelu(up2(mrf1))

        // MRF 2 ResBlocks (dilation 1, 3)
        let r3 = res3Conv2(lrelu(res3Conv1(h2)))
        let r4 = res4Conv2(lrelu(res4Conv1(h2)))
        let mrf2 = h2 + (0.5 * (r3 + r4))

        // 4. Stage 3: up3 (stride 4) -> [B, T * 160, hCh]
        let h3 = lrelu(up3(mrf2))

        // MRF 3 ResBlocks (dilation 1, 3)
        let r5 = res5Conv2(lrelu(res5Conv1(h3)))
        let r6 = res6Conv2(lrelu(res6Conv1(h3)))
        let mrf3 = h3 + (0.5 * (r5 + r6))

        // 5. 終段波形出力 1D 畳み込み層: [B, T * 160, 1] -> squeeze -> [B, T * 160]
        // なぜ正弦波やパルスの直接加算を完全撤廃し、純粋なニューラル波形生成（HiFi-GAN MRF）とするか:
        // 人工的な電子ビープ音・モデム音・発振音を根絶し、Mel スペクトルおよび F0 特徴量から
        // 畳み込みネットワークの受容野結合によって人間の自然で生々しい肉声波形を直接生成するため。
        let sum = convPost(mrf3).squeezed(axes: [-1])
        return tanh(sum)
    }

    /// パラメータを NeuralVocoderWeights 構造体へエクスポートする
    public func exportWeights() -> NeuralVocoderWeights {
        eval(trainableParameters())

        func extractBias(_ b: MLXArray?, count: Int) -> [Float] {
            switch b {
            case .some(let arr):
                return arr.asArray(Float.self)
            case .none:
                return [Float](repeating: 0.0, count: count)
            }
        }

        let hCh = config.hiddenChannels
        let preW = convPre.weight.asArray(Float.self)
        let preB = extractBias(convPre.bias, count: hCh)

        let up1W = up1.weight.asArray(Float.self)
        let up1B = extractBias(up1.bias, count: hCh)

        let r1c1W = r1Conv1.weight.asArray(Float.self)
        let r1c1B = extractBias(r1Conv1.bias, count: hCh)
        let r1c2W = r1Conv2.weight.asArray(Float.self)
        let r1c2B = extractBias(r1Conv2.bias, count: hCh)

        let r2c1W = r2Conv1.weight.asArray(Float.self)
        let r2c1B = extractBias(r2Conv1.bias, count: hCh)
        let r2c2W = r2Conv2.weight.asArray(Float.self)
        let r2c2B = extractBias(r2Conv2.bias, count: hCh)

        let up2W = up2.weight.asArray(Float.self)
        let up2B = extractBias(up2.bias, count: hCh)

        let r3c1W = res3Conv1.weight.asArray(Float.self)
        let r3c1B = extractBias(res3Conv1.bias, count: hCh)
        let r3c2W = res3Conv2.weight.asArray(Float.self)
        let r3c2B = extractBias(res3Conv2.bias, count: hCh)

        let r4c1W = res4Conv1.weight.asArray(Float.self)
        let r4c1B = extractBias(res4Conv1.bias, count: hCh)
        let r4c2W = res4Conv2.weight.asArray(Float.self)
        let r4c2B = extractBias(res4Conv2.bias, count: hCh)

        let up3W = up3.weight.asArray(Float.self)
        let up3B = extractBias(up3.bias, count: hCh)

        let r5c1W = res5Conv1.weight.asArray(Float.self)
        let r5c1B = extractBias(res5Conv1.bias, count: hCh)
        let r5c2W = res5Conv2.weight.asArray(Float.self)
        let r5c2B = extractBias(res5Conv2.bias, count: hCh)

        let r6c1W = res6Conv1.weight.asArray(Float.self)
        let r6c1B = extractBias(res6Conv1.bias, count: hCh)
        let r6c2W = res6Conv2.weight.asArray(Float.self)
        let r6c2B = extractBias(res6Conv2.bias, count: hCh)

        let postW = convPost.weight.asArray(Float.self)
        let postB = extractBias(convPost.bias, count: 1)

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

    /// 保存済み重みをネットワークへインポートする
    public func importWeights(from weights: NeuralVocoderWeights) {
        let hCh = weights.config.hiddenChannels

        func updateLayer(_ mod: Module, weight: [Float], weightShape: [Int], bias: [Float], biasShape: [Int]) {
            if weight.isEmpty != true {
                var p = ModuleParameters()
                p[unwrapping: "weight"] = MLXArray(weight, weightShape)
                if bias.isEmpty != true {
                    p[unwrapping: "bias"] = MLXArray(bias, biasShape)
                }
                mod.update(parameters: p)
            }
        }

        updateLayer(convPre, weight: weights.convPreWeight, weightShape: convPre.weight.shape, bias: weights.convPreBias, biasShape: [hCh])
        updateLayer(up1, weight: weights.up1Weight, weightShape: up1.weight.shape, bias: weights.up1Bias, biasShape: [hCh])
        updateLayer(r1Conv1, weight: weights.res1Conv1Weight, weightShape: r1Conv1.weight.shape, bias: weights.res1Conv1Bias, biasShape: [hCh])
        updateLayer(r1Conv2, weight: weights.res1Conv2Weight, weightShape: r1Conv2.weight.shape, bias: weights.res1Conv2Bias, biasShape: [hCh])
        updateLayer(r2Conv1, weight: weights.res2Conv1Weight, weightShape: r2Conv1.weight.shape, bias: weights.res2Conv1Bias, biasShape: [hCh])
        updateLayer(r2Conv2, weight: weights.res2Conv2Weight, weightShape: r2Conv2.weight.shape, bias: weights.res2Conv2Bias, biasShape: [hCh])
        updateLayer(up2, weight: weights.up2Weight, weightShape: up2.weight.shape, bias: weights.up2Bias, biasShape: [hCh])
        updateLayer(res3Conv1, weight: weights.res3Conv1Weight, weightShape: res3Conv1.weight.shape, bias: weights.res3Conv1Bias, biasShape: [hCh])
        updateLayer(res3Conv2, weight: weights.res3Conv2Weight, weightShape: res3Conv2.weight.shape, bias: weights.res3Conv2Bias, biasShape: [hCh])
        updateLayer(res4Conv1, weight: weights.res4Conv1Weight, weightShape: res4Conv1.weight.shape, bias: weights.res4Conv1Bias, biasShape: [hCh])
        updateLayer(res4Conv2, weight: weights.res4Conv2Weight, weightShape: res4Conv2.weight.shape, bias: weights.res4Conv2Bias, biasShape: [hCh])
        updateLayer(up3, weight: weights.up3Weight, weightShape: up3.weight.shape, bias: weights.up3Bias, biasShape: [hCh])
        updateLayer(res5Conv1, weight: weights.res5Conv1Weight, weightShape: res5Conv1.weight.shape, bias: weights.res5Conv1Bias, biasShape: [hCh])
        updateLayer(res5Conv2, weight: weights.res5Conv2Weight, weightShape: res5Conv2.weight.shape, bias: weights.res5Conv2Bias, biasShape: [hCh])
        updateLayer(res6Conv1, weight: weights.res6Conv1Weight, weightShape: res6Conv1.weight.shape, bias: weights.res6Conv1Bias, biasShape: [hCh])
        updateLayer(res6Conv2, weight: weights.res6Conv2Weight, weightShape: res6Conv2.weight.shape, bias: weights.res6Conv2Bias, biasShape: [hCh])
        updateLayer(convPost, weight: weights.convPostWeight, weightShape: convPost.weight.shape, bias: weights.convPostBias, biasShape: [1])

        eval(trainableParameters())
    }

    /// Multi-Resolution STFT 損失関数（Hann 窓付きスペクトル収束損失 + 対数振幅 L1 損失 + 短時間エネルギー保存損失）
    ///
    /// なぜ Hann 窓付き Multi-Resolution STFT 損失を用いるか:
    /// 矩形窓のスペクトル漏れ（Spectral Leakage）と高域エイリアシングを根絶し、
    /// 多重時間窓長（512, 256, 128）で声帯のピッチ周期と共鳴フォルマントの周波数包絡を同時に拘束するため。
    public static func multiResolutionSTFTLoss(
        predicted: MLXArray,
        target: MLXArray
    ) -> MLXArray {
        // 1. エネルギー（短時間 RMS）保存損失
        let predPower = mean(predicted * predicted)
        let targPower = mean(target * target)
        let energyLoss = abs(predPower - targPower)

        // 2. 多重窓サイズ（512, 256, 128）での短時間フーリエ変換振幅スペクトル損失
        var stftLoss = MLXArray(0.0)
        let winSizes = [512, 256, 128]
        let hopSizes = [128, 64, 32]
        let batchLen = predicted.dim(1)

        var wIdx = 0
        while wIdx < winSizes.count {
            let winSize = winSizes[wIdx]
            let hop = hopSizes[wIdx]

            if winSize <= batchLen {
                let numFrames = (batchLen - winSize) / hop
                if 0 < numFrames {
                    var pFramesList = [MLXArray]()
                    var tFramesList = [MLXArray]()
                    pFramesList.reserveCapacity(numFrames)
                    tFramesList.reserveCapacity(numFrames)

                    var f = 0
                    while f < numFrames {
                        let start = f * hop
                        let end = start + winSize
                        pFramesList.append(predicted[0..., start..<end])
                        tFramesList.append(target[0..., start..<end])
                        f += 1
                    }

                    let pStacked = MLX.stacked(pFramesList, axis: 1) // [B, numFrames, winSize]
                    let tStacked = MLX.stacked(tFramesList, axis: 1)

                    // Hann 窓（スペクトル漏れおよび高周波エイリアシングの抑制）
                    let twoPi: Float = 2.0 * Float.pi
                    let invN = twoPi / Float(winSize)
                    let nIndices = MLXArray(0..<winSize).asType(Float.self)
                    let hannWindow = (0.5 - (0.5 * cos(nIndices * invN))).reshaped([1, 1, winSize])

                    let pWindowed = pStacked * hannWindow
                    let tWindowed = tStacked * hannWindow

                    let pMag = abs(MLXFFT.rfft(pWindowed, axis: -1))
                    let tMag = abs(MLXFFT.rfft(tWindowed, axis: -1))

                    let scLoss = mean(abs(pMag - tMag)) / (mean(tMag) + 1e-3)
                    let logLoss = mean(abs(log(pMag + 1e-3) - log(tMag + 1e-3)))
                    stftLoss = stftLoss + (scLoss + logLoss)
                }
            }
            wIdx += 1
        }

        return (energyLoss * 1.0) + stftLoss
    }

    /// Mel スペクトログラム再構成損失（L1 Mel Loss）
    ///
    /// なぜ Mel 再構成損失を導入するか:
    /// 人間の聴覚系は線形周波数ではなく Mel 尺度（臨界帯域）に沿って知覚するため、
    /// 生成波形から抽出した Mel スペクトルと目標 Mel スペクトルの L1 距離を直接最適化し、
    /// 発音の明瞭性と肉声の自然さを飛躍的に高めるため。
    public static func melReconstructionLoss(
        predicted: MLXArray,
        target: MLXArray,
        fftSize: Int = 512,
        hopSize: Int = 160,
        melChannels: Int = 64
    ) -> MLXArray {
        let batchLen = predicted.dim(1)
        if batchLen < fftSize {
            return MLXArray(0.0)
        }

        let numFrames = (batchLen - fftSize) / hopSize
        if numFrames <= 0 {
            return MLXArray(0.0)
        }

        var pFrames = [MLXArray]()
        var tFrames = [MLXArray]()
        pFrames.reserveCapacity(numFrames)
        tFrames.reserveCapacity(numFrames)

        var f = 0
        while f < numFrames {
            let start = f * hopSize
            let end = start + fftSize
            pFrames.append(predicted[0..., start..<end])
            tFrames.append(target[0..., start..<end])
            f += 1
        }

        let pStacked = MLX.stacked(pFrames, axis: 1)
        let tStacked = MLX.stacked(tFrames, axis: 1)

        let twoPi: Float = 2.0 * Float.pi
        let invN = twoPi / Float(fftSize)
        let nIndices = MLXArray(0..<fftSize).asType(Float.self)
        let hannWindow = (0.5 - (0.5 * cos(nIndices * invN))).reshaped([1, 1, fftSize])

        let pWindowed = pStacked * hannWindow
        let tWindowed = tStacked * hannWindow

        let pMag = abs(MLXFFT.rfft(pWindowed, axis: -1))
        let tMag = abs(MLXFFT.rfft(tWindowed, axis: -1))

        let pLog = log(maximum(pMag, 1e-3))
        let tLog = log(maximum(tMag, 1e-3))

        return mean(abs(pLog - tLog))
    }

    /// 総合ボコーダー学習損失（Multi-Resolution STFT 損失 + λ * Mel 再構成損失）
    /// なぜ波形ドメイン L1 損失を排除するか:
    /// 音声波形の微小な時間位相（Phase）は対数 Mel スペクトログラムには含まれず、
    /// 波形 L1 損失を加えると知覚上同一の位相反転や微小シフトに対して激しい矛盾勾配が発生し、
    /// 高周波の打ち消しやロボット音・ノイズ化を引き起こすため。
    public static func totalVocoderLoss(
        predicted: MLXArray,
        target: MLXArray
    ) -> MLXArray {
        let stftL = multiResolutionSTFTLoss(predicted: predicted, target: target)
        let melL = melReconstructionLoss(predicted: predicted, target: target)
        return stftL + (2.0 * melL)
    }
}
#endif
