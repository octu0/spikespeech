#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// MLX 上で動作する多層 SNN 音響モデルネットワーク
///
/// ユニファイドメモリ上でデータ転送を行わずに高スループットな自動微分を実行し、
/// 多層 SNN パラメータを一括管理する。
public final class MLXSpikingAcousticNetwork: Module, @unchecked Sendable {
    public let numLayers: Int
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let lifConfig: LIFConfig

    /// 層 0 入力重み
    public var wIn: MLXArray

    /// 層 0 再帰重み
    public var wRec: MLXArray

    /// 層 0 バイアス
    public var bH: MLXArray

    /// 層 1 以降の FF 結合重み
    public var wLayers: [MLXArray]

    /// 層 1 以降のバイアス
    public var bHLayers: [MLXArray]

    /// 層 1 以降の RMSNorm ゲイン
    public var gammaRMS: [MLXArray]

    /// リードアウト射影重み
    public var wOut: MLXArray

    /// リードアウトバイアス
    public var bOut: MLXArray

    /// 学習・獲得された語彙知識（単語表記、読み、品詞、アクセント核、コスト）
    /// なぜネットワーク内で保持するか:
    /// BPTT 学習時および重みエクスポート時に、獲得された語彙知識が消失・初期化されることを防ぐため。
    public var lexicon: [LexiconEntry]

    public init(
        numLayers: Int = 2,
        inputDim: Int = 128,
        maxHiddenDim: Int = 1024,
        outputDim: Int = 80,
        timeSteps: Int = 4,
        lifConfig: LIFConfig = LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.1),
        lexicon: [LexiconEntry] = []
    ) {
        let safeLayers = max(1, numLayers)
        self.numLayers = safeLayers
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.lifConfig = lifConfig
        self.lexicon = lexicon

        let scaleIn = sqrt(2.0 / Float(inputDim))
        let scaleRec = 0.1 / sqrt(Float(maxHiddenDim))
        let scaleOut = sqrt(2.0 / Float(maxHiddenDim))
        let scaleLayer = sqrt(2.0 / Float(maxHiddenDim))

        self.wIn = MLXRandom.uniform(low: -scaleIn, high: scaleIn, [inputDim, maxHiddenDim])
        self.wRec = MLXRandom.uniform(low: -scaleRec, high: scaleRec, [maxHiddenDim, maxHiddenDim])
        self.bH = MLXArray.zeros([maxHiddenDim])

        var wl: [MLXArray] = []
        var bl: [MLXArray] = []
        var gl: [MLXArray] = []
        var l = 1
        while l < safeLayers {
            wl.append(MLXRandom.uniform(low: -scaleLayer, high: scaleLayer, [maxHiddenDim, maxHiddenDim]))
            bl.append(MLXArray.zeros([maxHiddenDim]))
            gl.append(MLXArray.ones([maxHiddenDim]))
            l += 1
        }
        self.wLayers = wl
        self.bHLayers = bl
        self.gammaRMS = gl

        self.wOut = MLXRandom.uniform(low: -scaleOut, high: scaleOut, [maxHiddenDim, outputDim])
        self.bOut = MLXArray.zeros([outputDim])

        super.init()
    }

    /// 保存済み多層重みから構成を復元する
    public convenience init(weights: SpikingNetworkWeights) {
        self.init(
            numLayers: weights.numLayers,
            inputDim: weights.inputDim,
            maxHiddenDim: weights.maxHiddenDim,
            outputDim: weights.outputDim,
            timeSteps: weights.timeSteps,
            lifConfig: weights.lifConfig,
            lexicon: weights.lexicon
        )
        self.importWeights(from: weights)
    }

    /// 多層重み構造体からパラメータをインポートする。
    /// 行優先配列から MLX の行優先形状に適合させるため転置を適用する。
    public func importWeights(from weights: SpikingNetworkWeights) {
        self.lexicon = weights.lexicon
        let hSize = weights.maxHiddenDim
        self.wIn = MLXArray(weights.wIn, [hSize, weights.inputDim]).transposed()
        self.wRec = MLXArray(weights.wRec, [hSize, hSize]).transposed()
        self.bH = MLXArray(weights.bH, [hSize])
        self.wOut = MLXArray(weights.wOut, [weights.outputDim, hSize]).transposed()
        self.bOut = MLXArray(weights.bOut, [weights.outputDim])

        var arraysToEval: [MLXArray] = [self.wIn, self.wRec, self.bH, self.wOut, self.bOut]
        var l = 0
        while l < min(self.wLayers.count, weights.wLayers.count) {
            self.wLayers[l] = MLXArray(weights.wLayers[l], [hSize, hSize]).transposed()
            self.bHLayers[l] = MLXArray(weights.bHLayers[l], [hSize])
            self.gammaRMS[l] = MLXArray(weights.gammaRMS[l], [hSize])
            arraysToEval.append(self.wLayers[l])
            arraysToEval.append(self.bHLayers[l])
            arraysToEval.append(self.gammaRMS[l])
            l += 1
        }
        eval(arraysToEval)
    }

    /// 学習済みパラメータを純粋推論用の多層重み構造体へエクスポートする
    public func exportWeights() -> SpikingNetworkWeights {
        var arraysToEval: [MLXArray] = [self.wIn, self.wRec, self.bH, self.wOut, self.bOut]
        var l = 0
        while l < wLayers.count {
            arraysToEval.append(wLayers[l])
            arraysToEval.append(bHLayers[l])
            arraysToEval.append(gammaRMS[l])
            l += 1
        }
        eval(arraysToEval)

        var wl: [[Float]] = []
        var bl: [[Float]] = []
        var gl: [[Float]] = []
        l = 0
        while l < wLayers.count {
            wl.append(self.wLayers[l].transposed().asArray(Float.self))
            bl.append(self.bHLayers[l].asArray(Float.self))
            gl.append(self.gammaRMS[l].asArray(Float.self))
            l += 1
        }

        return SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: self.wIn.transposed().asArray(Float.self),
            wRec: self.wRec.transposed().asArray(Float.self),
            bH: self.bH.asArray(Float.self),
            wLayers: wl,
            bHLayers: bl,
            gammaRMS: gl,
            wOut: self.wOut.transposed().asArray(Float.self),
            bOut: self.bOut.asArray(Float.self),
            lexicon: self.lexicon
        )
    }

    /// 音響系列の多層 SNN 順伝播計算を実行する。
    /// 層0 の再帰結合と上位層の RMSNorm 電流および前層入力電流残差加算により、多層化時のスパイク減衰を防ぐ。
    public func forward(
        features: MLXArray,
        bpttWindow: Int = 16
    ) -> MLXArray {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hSize = maxHiddenDim

        let beta = lifConfig.beta
        let vTh = lifConfig.vTh
        let alpha = lifConfig.alpha
        let rho = lifConfig.rho
        let gamma = lifConfig.gamma
        let vMin = LIFNeuronEngine.vClampMin
        let vMax = LIFNeuronEngine.vClampMax
        let readoutK = LIFNeuronEngine.readoutClipInThresholdUnits

        // 直流入力電流の事前計算
        let currentSeq0 = matmul(features, self.wIn) + self.bH

        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hSize]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hSize]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hSize]), count: numLayers)

        var readoutList: [MLXArray] = []
        readoutList.reserveCapacity(seqLen)

        var t = 0
        while t < seqLen {
            let current0_t = currentSeq0[0..., t, 0...]
            var readoutSum = MLXArray.zeros([batchSize, hSize])

            if (t % bpttWindow) == 0 {
                var l = 0
                while l < numLayers {
                    v[l] = stopGradient(v[l])
                    s[l] = stopGradient(s[l])
                    a[l] = stopGradient(a[l])
                    l += 1
                }
            }

            var step = 0
            while step < timeSteps {
                var current = current0_t + matmul(stopGradient(s[0]), self.wRec)
                var l = 0
                while l < numLayers {
                    let isLast = (l + 1) == numLayers

                    if 0 < l {
                        let upperIdx = l - 1
                        let denseCur = matmul(s[l - 1], self.wLayers[upperIdx]) + self.bHLayers[upperIdx]
                        let meanSq = mean(denseCur * denseCur, axis: -1, keepDims: true)
                        let rms = sqrt(meanSq + 1e-5)
                        current = (denseCur / rms) * self.gammaRMS[upperIdx] + current
                    }

                    if isLast {
                        v[l] = clip((v[l] * beta) + current, min: vMin, max: vMax)
                    } else {
                        v[l] = clip(((v[l] * beta) * (1.0 - s[l])) + current, min: vMin, max: vMax)
                    }

                    a[l] = (a[l] * rho) + (s[l] * gamma)
                    let dynVTh = vTh + a[l]
                    s[l] = SurrogateGradients.fastSigmoidSTE(v: v[l], vTh: dynVTh, alpha: alpha)

                    if isLast {
                        readoutSum = readoutSum + clip(v[l] / vTh, min: -readoutK, max: readoutK)
                        let sHard = (dynVTh .<= v[l]).asType(.float32)
                        v[l] = clip(v[l] - (sHard * vTh), min: vMin, max: vMax)
                    }
                    l += 1
                }
                step += 1
            }

            readoutList.append(readoutSum / Float(timeSteps))
            t += 1
        }

        let stackedReadout = stacked(readoutList, axis: 1)
        return matmul(stackedReadout, self.wOut) + self.bOut
    }
}

/// SNN 音響モデル BPTT 学習トレーナー
///
/// Truncated BPTT と 32 アライメントを統合し、リソースリークを防ぎつつ最適化を行う。
public final class MLXAcousticBPTTTrainer: @unchecked Sendable {
    public let network: MLXSpikingAcousticNetwork
    public let optimizer: AdamW
    public let bpttWindow: Int

    public init(
        network: MLXSpikingAcousticNetwork,
        learningRate: Float = 0.003,
        bpttWindow: Int = 16,
        weightDecay: Float = 1.0e-4
    ) {
        self.network = network
        // なぜ AdamW を採用し weightDecay 1e-4 を指定するか:
        // オンライン学習における再帰重み・出力重みの過大成長と膜電位クランプ飽和を防ぎ、
        // コサイン減衰スケジューラと整合する正則化を decoupled に効かせるため。
        self.optimizer = AdamW(
            learningRate: learningRate,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weightDecay: weightDecay,
            biasCorrection: true
        )
        if bpttWindow <= 1 {
            self.bpttWindow = 1
        } else {
            self.bpttWindow = bpttWindow
        }
    }

    /// スケジューラからのエポックごとの学習率更新を反映する。
    /// なぜオプティマイザを再生成せず learningRate プロパティを直接更新するか:
    /// Adam の蓄積された一次・二次モーメントを破棄せず連続性を維持するため。
    public func setLearningRate(_ lr: Float) {
        var safe = lr
        if safe.isFinite != true || safe < 0.0 {
            safe = 0.0
        }
        optimizer.learningRate = safe
    }

    public func currentLearningRate() -> Float {
        return optimizer.learningRate
    }

    /// 各重み行列のフロベニウスノルムを算出し、学習中の重み肥大化・発散を診断する。
    public func weightNorms() -> (wIn: Float, wRec: Float, wOut: Float, wLayer0: Float) {
        eval(network.wIn, network.wRec, network.wOut)
        let nIn = sqrt(sum(network.wIn * network.wIn).item(Float.self))
        let nRec = sqrt(sum(network.wRec * network.wRec).item(Float.self))
        let nOut = sqrt(sum(network.wOut * network.wOut).item(Float.self))
        var nL0: Float = 0.0
        if 0 < network.wLayers.count {
            eval(network.wLayers[0])
            nL0 = sqrt(sum(network.wLayers[0] * network.wLayers[0]).item(Float.self))
        }
        return (wIn: nIn, wRec: nRec, wOut: nOut, wLayer0: nL0)
    }

    /// 系列長を 32 の倍数に切り上げる。
    /// メモリプールでのバッファ再利用を促進し、Metal バッファの増殖クラッシュを防止する。
    public static func alignTo32(seqLen: Int) -> Int {
        if seqLen <= 0 {
            return 32
        }
        let remainder = seqLen % 32
        if remainder == 0 {
            return seqLen
        }
        return seqLen + (32 - remainder)
    }

    /// ミニバッチ学習ステップを実行する。
    /// スペクトル L1 再構成損失とフレーム間差分損失の線形結合により、静的かつ動的なスペクトル精度を高める。
    @discardableResult
    public func trainBatch(
        features: MLXArray,
        targets: MLXArray,
        mask: MLXArray? = nil
    ) -> Float {
        let lg = valueAndGrad(model: network) { (model: MLXSpikingAcousticNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let fArr = arrays[0]
            let tArr = arrays[1]
            let mArr = arrays[2]

            let pred = model.forward(features: fArr, bpttWindow: self.bpttWindow)
            let l1Loss = AcousticLossFunctions.spectralL1Loss(
                predicted: pred,
                target: tArr,
                mask: mArr
            )
            let deltaLoss = AcousticLossFunctions.spectralDeltaLoss(
                predicted: pred,
                target: tArr,
                mask: mArr
            )
            let totalLoss = l1Loss + (deltaLoss * 0.5)
            return [totalLoss]
        }

        let maskArray = mask ?? MLXArray.ones([features.shape[0], features.shape[1]])
        let (lossVals, grads) = lg(network, [features, targets, maskArray])
        let lossVal = lossVals[0]
        var safeGrads = grads
        if let recG = safeGrads[unwrapping: "wRec"] {
            let recNorm = sqrt(sum(recG * recG))
            let scale = minimum(MLXArray(1.0), MLXArray(2.0) / (recNorm + 1e-6))
            safeGrads[unwrapping: "wRec"] = recG * scale
        }
        if let inG = safeGrads[unwrapping: "wIn"] {
            let inNorm = sqrt(sum(inG * inG))
            let scale = minimum(MLXArray(1.0), MLXArray(2.0) / (inNorm + 1e-6))
            safeGrads[unwrapping: "wIn"] = inG * scale
        }
        let (clippedGrads, _) = clipGradNorm(gradients: safeGrads, maxNorm: 5.0)

        optimizer.update(model: network, gradients: clippedGrads)
        eval(network, optimizer, lossVal)

        return lossVal.item(Float.self)
    }

    /// 各パラメータの生の勾配ノルムおよびクリップ前後の診断
    public func diagnoseGradNorms(
        features: MLXArray,
        targets: MLXArray,
        mask: MLXArray? = nil
    ) -> [String: Float] {
        let lg = valueAndGrad(model: network) { (model: MLXSpikingAcousticNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let fArr = arrays[0]
            let tArr = arrays[1]
            let mArr = arrays[2]

            let pred = model.forward(features: fArr, bpttWindow: self.bpttWindow)
            let l1Loss = AcousticLossFunctions.spectralL1Loss(
                predicted: pred,
                target: tArr,
                mask: mArr
            )
            let deltaLoss = AcousticLossFunctions.spectralDeltaLoss(
                predicted: pred,
                target: tArr,
                mask: mArr
            )
            let totalLoss = l1Loss + (deltaLoss * 0.5)
            return [totalLoss]
        }

        let maskArray = mask ?? MLXArray.ones([features.shape[0], features.shape[1]])
        let (lossVals, grads) = lg(network, [features, targets, maskArray])
        eval(lossVals[0])

        var result: [String: Float] = [:]
        for (k, g) in grads.flattened() {
            eval(g)
            let gSq = g * g
            let sumVal = sum(gSq).item(Float.self)
            result[k] = sqrt(sumVal)
        }
        return result
    }

    /// 単一発話の系列学習ヘルパー
    public func trainSequence(
        features: [[Float]],
        targets: [[Float]]
    ) -> Float {
        let rawLen = min(features.count, targets.count)
        if rawLen <= 0 {
            return 0.0
        }

        let alignedLen = Self.alignTo32(seqLen: rawLen)
        let inDim = network.inputDim
        let outDim = network.outputDim

        var flatFeat = [Float](repeating: 0.0, count: alignedLen * inDim)
        var flatTgt = [Float](repeating: 0.0, count: alignedLen * outDim)
        var flatMask = [Float](repeating: 0.0, count: alignedLen)

        var t = 0
        while t < rawLen {
            flatMask[t] = 1.0
            var i = 0
            while i < inDim {
                flatFeat[(t * inDim) + i] = features[t][i]
                i += 1
            }
            var c = 0
            while c < outDim {
                flatTgt[(t * outDim) + c] = targets[t][c]
                c += 1
            }
            t += 1
        }

        let fArr = MLXArray(flatFeat, [1, alignedLen, inDim])
        let tArr = MLXArray(flatTgt, [1, alignedLen, outDim])
        let mArr = MLXArray(flatMask, [1, alignedLen])

        return trainBatch(
            features: fArr,
            targets: tArr,
            mask: mArr
        )
    }
}
#endif
