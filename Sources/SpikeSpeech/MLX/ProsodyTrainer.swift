import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// MLX による音素継続時間予測モデル (Duration Predictor)
public final class MLXDurationPredictorModel: Module {
    public let inputDim: Int
    public let hiddenDim: Int
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "fc2") public var fc2: Linear

    public init(inputDim: Int = 72, hiddenDim: Int = 64) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self._fc1.wrappedValue = Linear(inputDim, hiddenDim)
        self._fc2.wrappedValue = Linear(hiddenDim, 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = leakyRelu(fc1(x), negativeSlope: 0.1)
        return fc2(h)
    }

    public func exportWeights() -> DurationPredictorWeights {
        eval(self.trainableParameters())
        let w1Arr = fc1.weight.asArray(Float.self)
        let b1Arr = fc1.bias?.asArray(Float.self) ?? [Float](repeating: 0.0, count: hiddenDim)
        let w2Arr = fc2.weight.asArray(Float.self)
        let b2Arr = fc2.bias?.asArray(Float.self) ?? [Float](repeating: 0.0, count: 1)

        return DurationPredictorWeights(
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            w1: w1Arr,
            b1: b1Arr,
            w2: w2Arr,
            b2: b2Arr
        )
    }

    public func importWeights(from weights: DurationPredictorWeights) {
        var p1 = ModuleParameters()
        p1[unwrapping: "weight"] = MLXArray(weights.w1, [weights.hiddenDim, weights.inputDim])
        p1[unwrapping: "bias"] = MLXArray(weights.b1, [weights.hiddenDim])
        self.fc1.update(parameters: p1)

        var p2 = ModuleParameters()
        p2[unwrapping: "weight"] = MLXArray(weights.w2, [1, weights.hiddenDim])
        p2[unwrapping: "bias"] = MLXArray(weights.b2, [1])
        self.fc2.update(parameters: p2)
    }
}

/// MLX による F0 輪郭系列予測モデル (F0 Predictor)
///
/// 時間方向の 1D Depthwise Conv (K=3) と残差接続を備え、
/// 有声フレームの対数周波数（log Hz）を直接予測する。
public final class MLXF0PredictorModel: Module {
    public let inputDim: Int
    public let hiddenDim: Int
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "wConv") public var wConv: MLXArray
    @ModuleInfo(key: "fc2") public var fc2: Linear

    public init(inputDim: Int = 76, hiddenDim: Int = 128) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self._fc1.wrappedValue = Linear(inputDim, hiddenDim)

        // デフォルト: K=5 [0.10, 0.20, 0.40, 0.20, 0.10] の滑らかな時間平滑化フィルタ
        // 受容野を 5 フレーム (50ms) に拡大し、前後のモーラ・音素コンテキストを捕捉する
        var defaultConv = [Float](repeating: 0.0, count: 5 * hiddenDim)
        var c = 0
        while c < hiddenDim {
            defaultConv[(0 * hiddenDim) + c] = 0.10
            defaultConv[(1 * hiddenDim) + c] = 0.20
            defaultConv[(2 * hiddenDim) + c] = 0.40
            defaultConv[(3 * hiddenDim) + c] = 0.20
            defaultConv[(4 * hiddenDim) + c] = 0.10
            c += 1
        }
        self._wConv.wrappedValue = MLXArray(defaultConv, [5, hiddenDim])

        // 出力層: 対数周波数を予測するため、バイアス初期値を JSUT 女性実測平均 log(235) ≈ 5.4596 に設定
        // なぜ w2 を全要素 0 ではなく微小な決定論的値で初期化するか:
        // w2 が全要素 0 だと初段 fc1 および wConv への誤差逆伝播勾配 (dLoss/dh = dLoss/dout * w2) が
        // 完全にゼロ消失し、初段特徴量表現が学習されない問題を根本解決するため。
        let outLinear = Linear(hiddenDim, 1)
        var p2 = ModuleParameters()
        var w2Init = [Float](repeating: 0.0, count: hiddenDim)
        var hIdx = 0
        while hIdx < hiddenDim {
            w2Init[hIdx] = 0.01 * sinf(Float(hIdx + 1) * 1.618)
            hIdx += 1
        }
        let b2Init = [Float(log(235.0))]
        p2[unwrapping: "weight"] = MLXArray(w2Init, [1, hiddenDim])
        p2[unwrapping: "bias"] = MLXArray(b2Init, [1])
        outLinear.update(parameters: p2)
        self._fc2.wrappedValue = outLinear
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        var expanded = false
        if input.ndim == 2 {
            input = expandedDimensions(input, axis: 0)
            expanded = true
        }
        let seqLen = input.shape[1]

        // 1. 初段線形射影 + LeakyReLU
        let h0 = leakyRelu(fc1(input), negativeSlope: 0.1)

        // 2. 時間畳み込み（1D Depthwise Conv, K=5, replicate padding 2 frames）
        let hLeft = h0[0..., 0..<1, 0...]
        let hPadL = concatenated([hLeft, hLeft], axis: 1)
        let hRight = h0[0..., (seqLen - 1)..<seqLen, 0...]
        let hPadR = concatenated([hRight, hRight], axis: 1)
        let hPadded = concatenated([hPadL, h0, hPadR], axis: 1)

        let w0 = wConv[0]
        let w1 = wConv[1]
        let w2 = wConv[2]
        let w3 = wConv[3]
        let w4 = wConv[4]

        let part0 = hPadded[0..., 0..<seqLen, 0...] * w0
        let part1 = hPadded[0..., 1..<(seqLen + 1), 0...] * w1
        let part2 = hPadded[0..., 2..<(seqLen + 2), 0...] * w2
        let part3 = hPadded[0..., 3..<(seqLen + 3), 0...] * w3
        let part4 = hPadded[0..., 4..<(seqLen + 4), 0...] * w4
        let zConv = (part0 + part1) + (part2 + part3) + part4

        // 3. 残差加算 + 非線形活性化
        let m = leakyRelu(h0 + zConv, negativeSlope: 0.1)

        // 4. 終段線形射影 -> 対数 Hz (log Hz)
        let out = fc2(m)

        if expanded {
            return squeezed(out, axis: 0)
        }
        return out
    }

    public func exportWeights() -> F0PredictorWeights {
        eval(self.trainableParameters())
        eval(wConv)
        let w1Arr = fc1.weight.asArray(Float.self)
        let b1Arr = fc1.bias?.asArray(Float.self) ?? [Float](repeating: 0.0, count: hiddenDim)
        let wConvArr = wConv.asArray(Float.self)
        let w2Arr = fc2.weight.asArray(Float.self)
        let b2Arr = fc2.bias?.asArray(Float.self) ?? [Float](repeating: 0.0, count: 1)

        return F0PredictorWeights(
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            w1: w1Arr,
            b1: b1Arr,
            wConv: wConvArr,
            w2: w2Arr,
            b2: b2Arr
        )
    }

    public func importWeights(from weights: F0PredictorWeights) {
        var p1 = ModuleParameters()
        p1[unwrapping: "weight"] = MLXArray(weights.w1, [weights.hiddenDim, weights.inputDim])
        p1[unwrapping: "bias"] = MLXArray(weights.b1, [weights.hiddenDim])
        self.fc1.update(parameters: p1)

        var pConv = ModuleParameters()
        let k = weights.wConv.count / max(1, weights.hiddenDim)
        switch k {
        case 5:
            pConv[unwrapping: "wConv"] = MLXArray(weights.wConv, [5, weights.hiddenDim])
        default:
            // K=3 からのマイグレーション対応
            var conv5 = [Float](repeating: 0.0, count: 5 * weights.hiddenDim)
            var c = 0
            while c < weights.hiddenDim {
                conv5[(0 * weights.hiddenDim) + c] = 0.10
                conv5[(1 * weights.hiddenDim) + c] = weights.wConv[(0 * weights.hiddenDim) + c]
                conv5[(2 * weights.hiddenDim) + c] = weights.wConv[(1 * weights.hiddenDim) + c]
                conv5[(3 * weights.hiddenDim) + c] = weights.wConv[(2 * weights.hiddenDim) + c]
                conv5[(4 * weights.hiddenDim) + c] = 0.10
                c += 1
            }
            pConv[unwrapping: "wConv"] = MLXArray(conv5, [5, weights.hiddenDim])
        }
        self.update(parameters: pConv)

        var p2 = ModuleParameters()
        p2[unwrapping: "weight"] = MLXArray(weights.w2, [1, weights.hiddenDim])
        p2[unwrapping: "bias"] = MLXArray(weights.b2, [1])
        self.fc2.update(parameters: p2)
    }
}

/// MLX 韻律学習トレーナー（Duration / F0）
public final class MLXProsodyTrainer {
    public let durationModel: MLXDurationPredictorModel
    public let f0Model: MLXF0PredictorModel
    public let durationOptimizer: Adam
    public let f0Optimizer: Adam

    public init(
        durationInputDim: Int = 72,
        durationHiddenDim: Int = 64,
        f0InputDim: Int = 76,
        f0HiddenDim: Int = 128,
        learningRate: Float = 0.005
    ) {
        self.durationModel = MLXDurationPredictorModel(inputDim: durationInputDim, hiddenDim: durationHiddenDim)
        self.f0Model = MLXF0PredictorModel(inputDim: f0InputDim, hiddenDim: f0HiddenDim)
        self.durationOptimizer = Adam(learningRate: 0.001)
        self.f0Optimizer = Adam(learningRate: learningRate)
    }

    /// 音素 duration バッチの 1 ステップ学習
    /// feat: [N, inDim], ruleDur: [N, 1], targetDur: [N, 1]
    public func trainDurationStep(
        features: [[Float]],
        ruleDurations: [Float],
        targetDurations: [Float]
    ) -> Float {
        let n = features.count
        if n <= 0 {
            return 0.0
        }
        let inD = durationModel.inputDim
        var flatFeat = [Float]()
        flatFeat.reserveCapacity(n * inD)
        var i = 0
        while i < n {
            flatFeat.append(contentsOf: features[i])
            i += 1
        }

        let featArr = MLXArray(flatFeat, [n, inD])
        let ruleArr = MLXArray(ruleDurations, [n, 1])
        let targetArr = MLXArray(targetDurations, [n, 1])

        let lg = valueAndGrad(model: durationModel) { (model: MLXDurationPredictorModel, arrays: [MLXArray]) -> [MLXArray] in
            let f = arrays[0]
            let r = arrays[1]
            let t = arrays[2]

            let offset = model(f)
            let clampedOffset = clip(offset, min: -1.5, max: 1.5)
            let predDur = r * exp(clampedOffset)
            let loss = mean(abs(predDur - t))
            return [loss]
        }

        let (lossVals, grads) = lg(durationModel, [featArr, ruleArr, targetArr])
        let lossVal = lossVals[0].item(Float.self)
        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 1.0)
        durationOptimizer.update(model: durationModel, gradients: clippedGrads)
        eval(durationModel.trainableParameters())

        return lossVal
    }

    /// F0 フレーム系列バッチの 1 ステップ学習
    /// features: [T, inDim], fujisakiF0: [T] (互換用、未使用), targetF0: [T], voicedMask: [T] (有声=1.0)
    public func trainF0Step(
        features: [[Float]],
        fujisakiF0: [Float] = [],
        targetF0: [Float],
        voicedMask: [Float]
    ) -> Float {
        let n = features.count
        if n <= 0 {
            return 0.0
        }
        let inD = f0Model.inputDim
        var flatFeat = [Float]()
        flatFeat.reserveCapacity(n * inD)
        var i = 0
        while i < n {
            flatFeat.append(contentsOf: features[i])
            i += 1
        }

        let featArr = MLXArray(flatFeat, [n, inD])
        let targetArr = MLXArray(targetF0, [n, 1])
        let maskArr = MLXArray(voicedMask, [n, 1])

        let lg = valueAndGrad(model: f0Model) { (model: MLXF0PredictorModel, arrays: [MLXArray]) -> [MLXArray] in
            let f = arrays[0]
            let t = arrays[1]
            let m = arrays[2]

            // 対数 Hz 直接予測 (log Hz)
            let logPred = model(f)
            let targetLog = log(maximum(t, 1.0))

            // なぜ対数周波数空間の Pseudo-Huber (Smooth L1) 損失にするか:
            // Hz 空間 L1 損失では勾配が exp(logPred) 倍に膨張して振動し微小誤差領域（< 20 Hz）で収束しないが、
            // 対数空間の滑らかな損失 (delta ≈ 0.05 ≈ 11 Hz) では微小誤差時に勾配がゼロへ収束し、目標 MAE < 20 Hz に確実に吸着するため。
            let diff = (logPred - targetLog) * m
            let smoothL1 = (sqrt(diff * diff + 0.0025) - 0.05) * m
            let validCount = sum(m) + 1.0e-5
            let loss = sum(smoothL1) / validCount
            return [loss]
        }

        let (lossVals, grads) = lg(f0Model, [featArr, targetArr, maskArr])
        eval(lossVals)
        // なぜ maxNorm を 10.0 に緩和するか:
        // 2.0 では 10,500 パラメータ全体で勾配が過剰に圧縮され、1 ステップあたりの更新量が微小となり
        // F0 MAE の収束が著しく遅延するため、適切なステップ幅を確保するため。
        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 10.0)
        f0Optimizer.update(model: f0Model, gradients: clippedGrads)
        eval(f0Model.trainableParameters())
        eval(f0Model.wConv)

        // 評価指標として実測有声 F0 MAE (Hz) を計算して返却
        let logPred = f0Model(featArr)
        let predF0 = exp(logPred)
        let hzDiff = abs(predF0 - targetArr) * maskArr
        let validCount = sum(maskArr) + 1.0e-5
        let maeHz = sum(hzDiff) / validCount
        eval(maeHz)

        return maeHz.item(Float.self)
    }

    /// 学習済み重みを統合 ProsodyWeights としてエクスポート
    public func exportWeights() -> ProsodyWeights {
        return ProsodyWeights(
            durationWeights: durationModel.exportWeights(),
            f0Weights: f0Model.exportWeights()
        )
    }

    /// 既存の重みをインポート
    public func importWeights(from weights: ProsodyWeights) {
        durationModel.importWeights(from: weights.durationWeights)
        f0Model.importWeights(from: weights.f0Weights)
    }
}
