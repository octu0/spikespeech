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

    /// 層 1 以降の時間畳み込み（フレーム間混合）重み [3, maxHiddenDim]
    public var wConv: [MLXArray]

    /// リードアウト射影重み
    public var wOut: MLXArray

    /// リードアウトバイアス
    public var bOut: MLXArray

    /// 学習・獲得された語彙知識（単語表記、読み、品詞、アクセント核、コスト）
    /// なぜネットワーク内で保持するか:
    /// BPTT 学習時および重みエクスポート時に、獲得された語彙知識が消失・初期化されることを防ぐため。
    public var lexicon: [LexiconEntry]

    public init(
        numLayers: Int = 4,
        inputDim: Int = AudioConfig.acousticInputDim,
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
        var wc: [MLXArray] = []
        var l = 1
        while l < safeLayers {
            wl.append(MLXRandom.uniform(low: -scaleLayer, high: scaleLayer, [maxHiddenDim, maxHiddenDim]))
            bl.append(MLXArray.zeros([maxHiddenDim]))
            gl.append(MLXArray.ones([maxHiddenDim]))

            // 時間畳み込み（長時間混合）重み初期化: K=5 [0.10, 0.20, 0.40, 0.20, 0.10] に微小ノイズ
            var convInit = [Float](repeating: 0.0, count: 5 * maxHiddenDim)
            var c = 0
            while c < maxHiddenDim {
                convInit[(0 * maxHiddenDim) + c] = 0.10
                convInit[(1 * maxHiddenDim) + c] = 0.20
                convInit[(2 * maxHiddenDim) + c] = 0.40
                convInit[(3 * maxHiddenDim) + c] = 0.20
                convInit[(4 * maxHiddenDim) + c] = 0.10
                c += 1
            }
            let baseConv = MLXArray(convInit, [5, maxHiddenDim])
            let noiseConv = MLXRandom.uniform(low: -0.01, high: 0.01, [5, maxHiddenDim])
            wc.append(baseConv + noiseConv)
            l += 1
        }
        self.wLayers = wl
        self.bHLayers = bl
        self.gammaRMS = gl
        self.wConv = wc

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
            if l < weights.wConv.count {
                let savedConv = weights.wConv[l]
                let kSize = savedConv.count / max(1, hSize)
                switch kSize {
                case 5:
                    self.wConv[l] = MLXArray(savedConv, [5, hSize])
                case 3:
                    // 既存 K=3 重みを K=5 (中央 3 タップ) に滑らかに引き継ぎ、完全ウォームスタート
                    var upgradedConv = [Float](repeating: 0.0, count: 5 * hSize)
                    var c = 0
                    while c < hSize {
                        upgradedConv[(0 * hSize) + c] = 0.05
                        upgradedConv[(1 * hSize) + c] = savedConv[(0 * hSize) + c]
                        upgradedConv[(2 * hSize) + c] = savedConv[(1 * hSize) + c]
                        upgradedConv[(3 * hSize) + c] = savedConv[(2 * hSize) + c]
                        upgradedConv[(4 * hSize) + c] = 0.05
                        c += 1
                    }
                    self.wConv[l] = MLXArray(upgradedConv, [5, hSize])
                default:
                    break
                }
                arraysToEval.append(self.wConv[l])
            }
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
            arraysToEval.append(wConv[l])
            l += 1
        }
        eval(arraysToEval)

        var wl: [[Float]] = []
        var bl: [[Float]] = []
        var gl: [[Float]] = []
        var wc: [[Float]] = []
        l = 0
        while l < wLayers.count {
            wl.append(self.wLayers[l].transposed().asArray(Float.self))
            bl.append(self.bHLayers[l].asArray(Float.self))
            gl.append(self.gammaRMS[l].asArray(Float.self))
            wc.append(self.wConv[l].asArray(Float.self))
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
            wConv: wc,
            wOut: self.wOut.transposed().asArray(Float.self),
            bOut: self.bOut.asArray(Float.self),
            lexicon: self.lexicon
        )
    }

    /// 音響系列の多層 SNN 順伝播計算を実行する。
    /// 層0 の再帰結合と上位ブロックのフレーム間時間畳み込み（1D Depthwise Conv）および RMSNorm 残差加算により、
    /// Mel スペクトログラムの横縞・平坦倍音化を根本から排し、滑らかなフォルマント軌跡を形成する。
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

        // ------------------------------------------------------------
        // ブロック 0: 層 0 再帰 LIF
        // ------------------------------------------------------------
        let currentSeq0 = matmul(features, self.wIn) + self.bH
        var v0 = MLXArray.zeros([batchSize, hSize])
        var s0 = MLXArray.zeros([batchSize, hSize])
        var a0 = MLXArray.zeros([batchSize, hSize])

        var layer0Readouts: [MLXArray] = []
        layer0Readouts.reserveCapacity(seqLen)

        var t = 0
        while t < seqLen {
            let current0_t = currentSeq0[0..., t, 0...]
            var readoutSum = MLXArray.zeros([batchSize, hSize])

            // 音素境界での層 0 膜電位・スパイク・適応変数リセット
            // なぜ音素境界でリセットを行うか:
            // 前の音素の再帰結合による約 300ms 周期の自励振動を次の音素へ持ち越すのを物理的に遮断するため。
            if AudioConfig.pulseChannel < inputDim {
                let pulseCh = AudioConfig.pulseChannel
                let pulse = features[0..., t, pulseCh..<(pulseCh + 1)]
                let resetFactor = 1.0 - pulse
                v0 = v0 * resetFactor
                s0 = s0 * resetFactor
                a0 = a0 * resetFactor
            }

            if (t % bpttWindow) == 0 {
                v0 = stopGradient(v0)
                s0 = stopGradient(s0)
                a0 = stopGradient(a0)
            }

            var step = 0
            while step < timeSteps {
                let current = current0_t + matmul(stopGradient(s0), self.wRec)
                v0 = clip(((v0 * beta) * (1.0 - s0)) + current, min: vMin, max: vMax)
                a0 = (a0 * rho) + (s0 * gamma)
                let dynVTh = vTh + a0
                s0 = SurrogateGradients.fastSigmoidSTE(v: v0, vTh: dynVTh, alpha: alpha)
                readoutSum = readoutSum + clip(v0 / vTh, min: -readoutK, max: readoutK)
                step += 1
            }

            layer0Readouts.append(readoutSum / Float(timeSteps))
            t += 1
        }

        var prevBlockH = stacked(layer0Readouts, axis: 1) // [batchSize, seqLen, hSize]

        // ------------------------------------------------------------
        // 上位ブロック 1 ..< numLayers
        // 時間畳み込み（Depthwise 1D Conv, K=3）＋ チャネル結合 ＋ RMSNorm ＋ 残差加算 ＋ LIF
        // ------------------------------------------------------------
        var blockIdx = 1
        while blockIdx < numLayers {
            let upperIdx = blockIdx - 1
            let isLast = (blockIdx + 1) == numLayers

            // 1. 時間畳み込み（K=5, Dilation d = 1 << upperIdx, replicate パディング左右 2d）
            let d = 1 << upperIdx

            let padLeft = prevBlockH[0..., 0..<1, 0...]
            var leftPads: [MLXArray] = []
            var p = 0
            while p < (2 * d) {
                leftPads.append(padLeft)
                p += 1
            }
            let hLeft = concatenated(leftPads, axis: 1)

            let padRight = prevBlockH[0..., (seqLen - 1)..<seqLen, 0...]
            var rightPads: [MLXArray] = []
            p = 0
            while p < (2 * d) {
                rightPads.append(padRight)
                p += 1
            }
            let hRight = concatenated(rightPads, axis: 1)
            let hPadded = concatenated([hLeft, prevBlockH, hRight], axis: 1) // [batchSize, seqLen + 4*d, hSize]

            let w0 = self.wConv[upperIdx][0]
            let w1 = self.wConv[upperIdx][1]
            let w2 = self.wConv[upperIdx][2]
            let w3 = self.wConv[upperIdx][3]
            let w4 = self.wConv[upperIdx][4]

            let part0 = hPadded[0..., 0..<seqLen, 0...] * w0
            let part1 = hPadded[0..., d..<(seqLen + d), 0...] * w1
            let part2 = hPadded[0..., (2 * d)..<(seqLen + (2 * d)), 0...] * w2
            let part3 = hPadded[0..., (3 * d)..<(seqLen + (3 * d)), 0...] * w3
            let part4 = hPadded[0..., (4 * d)..<(seqLen + (4 * d)), 0...] * w4
            let zConv = (((part0 + part1) + part2) + part3) + part4
            let mConv = zConv + prevBlockH // 残差接続

            // 2. チャネル間結合 (Pointwise Dense) ＋ RMSNorm ＋ 残差加算
            let denseCur = matmul(mConv, self.wLayers[upperIdx]) + self.bHLayers[upperIdx]
            let meanSq = mean(denseCur * denseCur, axis: -1, keepDims: true)
            let rms = sqrt(meanSq + 1e-5)
            let currentSeq = (denseCur / rms) * self.gammaRMS[upperIdx] + mConv

            // 3. 上位ブロック全時間軸一括 LIF 時間ステップ実行
            // 時間畳み込み（1D Depthwise Conv）により系列全体の時間相関が既に注入されているため、
            // フレームごとの反復・テンソル配列 append・stacked による Metal バッファ肥大化（499,000 リミット超過）を完全根絶し、
            // [batchSize, seqLen, hSize] の一括テンソル演算として 4 内部ステップを並列実行する。
            var vL = MLXArray.zeros([batchSize, seqLen, hSize])
            var sL = MLXArray.zeros([batchSize, seqLen, hSize])
            var aL = MLXArray.zeros([batchSize, seqLen, hSize])
            var readoutSum = MLXArray.zeros([batchSize, seqLen, hSize])

            var step = 0
            while step < timeSteps {
                if isLast {
                    vL = clip((vL * beta) + currentSeq, min: vMin, max: vMax)
                } else {
                    vL = clip(((vL * beta) * (1.0 - sL)) + currentSeq, min: vMin, max: vMax)
                }

                aL = (aL * rho) + (sL * gamma)
                let dynVTh = vTh + aL
                sL = SurrogateGradients.fastSigmoidSTE(v: vL, vTh: dynVTh, alpha: alpha)

                readoutSum = readoutSum + clip(vL / vTh, min: -readoutK, max: readoutK)
                if isLast {
                    let sHard = (dynVTh .<= vL).asType(.float32)
                    vL = clip(vL - (sHard * vTh), min: vMin, max: vMax)
                }
                step += 1
            }

            prevBlockH = readoutSum / Float(timeSteps)
            blockIdx += 1
        }

        // ------------------------------------------------------------
        // 最終線形射影: 最上位ブロック出力 -> Mel スペクトル
        // ------------------------------------------------------------
        return matmul(prevBlockH, self.wOut) + self.bOut
    }
}

/// SNN 音響モデル BPTT 学習トレーナー
///
/// Truncated BPTT と 32 アライメントを統合し、リソースリークを防ぎつつ最適化を行う。
public final class MLXAcousticBPTTTrainer: @unchecked Sendable {
    public let network: MLXSpikingAcousticNetwork
    public let optimizer: AdamW
    public let bpttWindow: Int
    private let lossAndGrad: (MLXSpikingAcousticNetwork, [MLXArray]) -> ([MLXArray], ModuleParameters)

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
        let resolvedBPTT: Int
        if bpttWindow <= 1 {
            resolvedBPTT = 1
        } else {
            resolvedBPTT = bpttWindow
        }
        self.bpttWindow = resolvedBPTT

        // なぜキャッシュ上限を 32MB に制限するか:
        // デフォルトでは RAM の 1.5 倍まで解放済み Metal バッファがアロケータのプールに保持され続け、
        // 異なる系列長の音声学習を数百ステップ回すとバッファ個数が 499,000 個を超過してクラッシュするため。
        Memory.cacheLimit = 32 * 1024 * 1024

        // なぜ valueAndGrad を init で一度だけ束縛するか:
        // 毎ミニバッチでのクロージャ生成とグラフ登録の反復による Metal 内部オブジェクト肥大化を防ぐため。
        self.lossAndGrad = valueAndGrad(model: network) { (model: MLXSpikingAcousticNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let fArr = arrays[0]
            let tArr = arrays[1]
            let mArr = arrays[2]

            let pred = model.forward(features: fArr, bpttWindow: resolvedBPTT)
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
        let maskArray = mask ?? MLXArray.ones([features.shape[0], features.shape[1]])
        let (lossVals, grads) = self.lossAndGrad(network, [features, targets, maskArray])
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
        Stream.gpu.synchronize()

        let lossResult = lossVal.item(Float.self)
        Memory.clearCache()
        return lossResult
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

        return autoreleasepool {
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
}
#endif
