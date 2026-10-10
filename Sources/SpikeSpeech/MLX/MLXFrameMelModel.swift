#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// SpikeVoice (arXiv:2408.00788) に基づくフレーム単位対数メル学習・推論モデル（MLX 実装）
public final class MLXFrameMelModel: Module, @unchecked Sendable {
    // 1. 音素エンコーダ
    public var embedCur: Embedding
    public var embedPrev: Embedding
    public var embedNext: Embedding
    public var encBIn: MLXArray
    /// 韻律位置特徴（prosodicFeatureCount 次元）の線形射影（バイアスなし、ゼロ初期化）
    public var encFeatProj: Linear
    public var encConv0: Conv1d
    public var encConv1: Conv1d
    public var encConv2: Conv1d
    public var encConv3: Conv1d

    // 2. 分散アダプタ (Variance Adaptor)
    public var durConv1: Conv1d
    public var durConv2: Conv1d

    public var f0Conv1: Conv1d
    public var f0Conv2: Conv1d

    public var energyConv1: Conv1d
    public var energyConv2: Conv1d

    // 3. メルデコーダ (Mel Decoder)
    public var decIn: Conv1d
    public var decConv0: Conv1d
    public var decConv1: Conv1d
    public var decConv2: Conv1d
    public var decConv3: Conv1d
    public var decOut: Conv1d

    // 4. PostNet
    public var postConv0: Conv1d
    public var postConv1: Conv1d
    public var postConv2: Conv1d
    public var postConv3: Conv1d
    public var postConv4: Conv1d

    // 5. メル残差 (Mel Residual: 2層時間畳み込み)
    public var melResidual: MLXMelResidual?

    public init(decoderKernel: Int) {
        let hiddenDim = 256
        let melDim = 64
        let vocabSize = 64
        let decPad = decoderKernel / 2

        // エンコーダ
        self.embedCur = Embedding(embeddingCount: vocabSize, dimensions: hiddenDim)
        self.embedPrev = Embedding(embeddingCount: vocabSize, dimensions: hiddenDim)
        self.embedNext = Embedding(embeddingCount: vocabSize, dimensions: hiddenDim)
        self.encBIn = MLXArray.zeros([hiddenDim])
        self.encFeatProj = Linear(weight: MLXArray.zeros([hiddenDim, FrameMelWeights.prosodicFeatureCount]), bias: nil)
        self.encConv0 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 3, stride: 1, padding: 1)
        self.encConv1 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 3, stride: 1, padding: 1)
        self.encConv2 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 3, stride: 1, padding: 1)
        self.encConv3 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 3, stride: 1, padding: 1)

        // 継続時間予測器
        self.durConv1 = Conv1d(inputChannels: hiddenDim, outputChannels: 128, kernelSize: 3, stride: 1, padding: 1)
        self.durConv2 = Conv1d(inputChannels: 128, outputChannels: 1, kernelSize: 1, stride: 1, padding: 0)

        // F0 予測器
        self.f0Conv1 = Conv1d(inputChannels: 257, outputChannels: 128, kernelSize: 5, stride: 1, padding: 2)
        self.f0Conv2 = Conv1d(inputChannels: 128, outputChannels: 1, kernelSize: 1, stride: 1, padding: 0)

        // エネルギー予測器
        self.energyConv1 = Conv1d(inputChannels: 257, outputChannels: 128, kernelSize: 5, stride: 1, padding: 2)
        self.energyConv2 = Conv1d(inputChannels: 128, outputChannels: 1, kernelSize: 1, stride: 1, padding: 0)

        // デコーダ
        self.decIn = Conv1d(inputChannels: 260, outputChannels: hiddenDim, kernelSize: 1, stride: 1, padding: 0)
        self.decConv0 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: decoderKernel, stride: 1, padding: decPad)
        self.decConv1 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: decoderKernel, stride: 1, padding: decPad)
        self.decConv2 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: decoderKernel, stride: 1, padding: decPad)
        self.decConv3 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: decoderKernel, stride: 1, padding: decPad)
        self.decOut = Conv1d(inputChannels: hiddenDim, outputChannels: melDim, kernelSize: 1, stride: 1, padding: 0)

        // PostNet
        self.postConv0 = Conv1d(inputChannels: melDim, outputChannels: hiddenDim, kernelSize: 5, stride: 1, padding: 2)
        self.postConv1 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 5, stride: 1, padding: 2)
        self.postConv2 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 5, stride: 1, padding: 2)
        self.postConv3 = Conv1d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 5, stride: 1, padding: 2)
        self.postConv4 = Conv1d(inputChannels: hiddenDim, outputChannels: melDim, kernelSize: 5, stride: 1, padding: 2)

        super.init()
    }

    public override convenience init() {
        self.init(decoderKernel: 17)
    }

    public convenience init(weights: FrameMelWeights) {
        let k = weights.decWConv[0].count / (256 * 256)
        self.init(decoderKernel: k)
        self.importWeights(from: weights)
    }

    /// 重み構造体からのインポート
    public func importWeights(from weights: FrameMelWeights) {
        func updateConv(_ conv: Conv1d, w: [Float], b: [Float], inC: Int, outC: Int, k: Int) {
            var p = ModuleParameters()
            p[unwrapping: "weight"] = MLXArray(w, [outC, k, inC])
            p[unwrapping: "bias"] = MLXArray(b, [outC])
            conv.update(parameters: p)
        }

        var pEmbCur = ModuleParameters()
        pEmbCur[unwrapping: "weight"] = MLXArray(weights.embedCur, [64, 256])
        embedCur.update(parameters: pEmbCur)

        var pEmbPrev = ModuleParameters()
        pEmbPrev[unwrapping: "weight"] = MLXArray(weights.embedPrev, [64, 256])
        embedPrev.update(parameters: pEmbPrev)

        var pEmbNext = ModuleParameters()
        pEmbNext[unwrapping: "weight"] = MLXArray(weights.embedNext, [64, 256])
        embedNext.update(parameters: pEmbNext)

        self.encBIn = MLXArray(weights.encBIn, [256])

        let featCount = FrameMelWeights.prosodicFeatureCount
        var pFeat = ModuleParameters()
        switch weights.encWFeat {
        case .some(let w) where w.count == 256 * featCount:
            pFeat[unwrapping: "weight"] = MLXArray(w, [256, featCount])
        default:
            pFeat[unwrapping: "weight"] = MLXArray.zeros([256, featCount])
        }
        encFeatProj.update(parameters: pFeat)

        updateConv(encConv0, w: weights.encWConv[0], b: weights.encBConv[0], inC: 256, outC: 256, k: 3)
        updateConv(encConv1, w: weights.encWConv[1], b: weights.encBConv[1], inC: 256, outC: 256, k: 3)
        updateConv(encConv2, w: weights.encWConv[2], b: weights.encBConv[2], inC: 256, outC: 256, k: 3)
        updateConv(encConv3, w: weights.encWConv[3], b: weights.encBConv[3], inC: 256, outC: 256, k: 3)

        updateConv(durConv1, w: weights.durW1, b: weights.durB1, inC: 256, outC: 128, k: 3)
        updateConv(durConv2, w: weights.durW2, b: weights.durB2, inC: 128, outC: 1, k: 1)

        updateConv(f0Conv1, w: weights.f0W1, b: weights.f0B1, inC: 257, outC: 128, k: 5)
        updateConv(f0Conv2, w: weights.f0W2, b: weights.f0B2, inC: 128, outC: 1, k: 1)

        updateConv(energyConv1, w: weights.energyW1, b: weights.energyB1, inC: 257, outC: 128, k: 5)
        updateConv(energyConv2, w: weights.energyW2, b: weights.energyB2, inC: 128, outC: 1, k: 1)

        let decK = weights.decWConv[0].count / (256 * 256)
        let decPad = decK / 2
        if decConv0.weight.shape[1] != decK {
            decConv0 = Conv1d(inputChannels: 256, outputChannels: 256, kernelSize: decK, stride: 1, padding: decPad)
            decConv1 = Conv1d(inputChannels: 256, outputChannels: 256, kernelSize: decK, stride: 1, padding: decPad)
            decConv2 = Conv1d(inputChannels: 256, outputChannels: 256, kernelSize: decK, stride: 1, padding: decPad)
            decConv3 = Conv1d(inputChannels: 256, outputChannels: 256, kernelSize: decK, stride: 1, padding: decPad)
        }

        updateConv(decIn, w: weights.decWIn, b: weights.decBIn, inC: 260, outC: 256, k: 1)
        updateConv(decConv0, w: weights.decWConv[0], b: weights.decBConv[0], inC: 256, outC: 256, k: decK)
        updateConv(decConv1, w: weights.decWConv[1], b: weights.decBConv[1], inC: 256, outC: 256, k: decK)
        updateConv(decConv2, w: weights.decWConv[2], b: weights.decBConv[2], inC: 256, outC: 256, k: decK)
        updateConv(decConv3, w: weights.decWConv[3], b: weights.decBConv[3], inC: 256, outC: 256, k: decK)
        updateConv(decOut, w: weights.decWOut, b: weights.decBOut, inC: 256, outC: 64, k: 1)

        updateConv(postConv0, w: weights.postWConv[0], b: weights.postBConv[0], inC: 64, outC: 256, k: 5)
        updateConv(postConv1, w: weights.postWConv[1], b: weights.postBConv[1], inC: 256, outC: 256, k: 5)
        updateConv(postConv2, w: weights.postWConv[2], b: weights.postBConv[2], inC: 256, outC: 256, k: 5)
        updateConv(postConv3, w: weights.postWConv[3], b: weights.postBConv[3], inC: 256, outC: 256, k: 5)
        updateConv(postConv4, w: weights.postWConv[4], b: weights.postBConv[4], inC: 256, outC: 64, k: 5)

        switch (weights.resW1, weights.resB1, weights.resW2, weights.resB2) {
        case (.some(let w1), .some(let b1), .some(let w2), .some(let b2)):
            self.melResidual = MLXMelResidual(w1: w1, b1: b1, w2: w2, b2: b2)
        default:
            self.melResidual = nil
        }

        eval(trainableParameters())
    }

    /// 純粋重み構造体へのエクスポート
    public func exportWeights() -> FrameMelWeights {
        func getArr(_ arr: MLXArray) -> [Float] {
            return arr.asArray(Float.self)
        }

        func getConv(_ conv: Conv1d) -> (w: [Float], b: [Float]) {
            let w = conv.weight.asArray(Float.self)
            let b: [Float]
            switch conv.bias {
            case .some(let bArr):
                b = bArr.asArray(Float.self)
            case .none:
                b = [Float](repeating: 0.0, count: conv.weight.shape[0])
            }
            return (w, b)
        }

        let ec0 = getConv(encConv0)
        let ec1 = getConv(encConv1)
        let ec2 = getConv(encConv2)
        let ec3 = getConv(encConv3)

        let d1 = getConv(durConv1)
        let d2 = getConv(durConv2)

        let f1 = getConv(f0Conv1)
        let f2 = getConv(f0Conv2)

        let e1 = getConv(energyConv1)
        let e2 = getConv(energyConv2)

        let di = getConv(decIn)
        let dc0 = getConv(decConv0)
        let dc1 = getConv(decConv1)
        let dc2 = getConv(decConv2)
        let dc3 = getConv(decConv3)
        let dOut = getConv(decOut)

        let pc0 = getConv(postConv0)
        let pc1 = getConv(postConv1)
        let pc2 = getConv(postConv2)
        let pc3 = getConv(postConv3)
        let pc4 = getConv(postConv4)

        let (rw1, rb1, rw2, rb2): ([Float]?, [Float]?, [Float]?, [Float]?)
        switch melResidual {
        case .some(let res):
            let exp = res.exportWeights()
            rw1 = exp.w1
            rb1 = exp.b1
            rw2 = exp.w2
            rb2 = exp.b2
        case .none:
            rw1 = nil
            rb1 = nil
            rw2 = nil
            rb2 = nil
        }

        return FrameMelWeights(
            embedCur: getArr(embedCur.weight),
            embedPrev: getArr(embedPrev.weight),
            embedNext: getArr(embedNext.weight),
            encBIn: getArr(encBIn),
            encWConv: [ec0.w, ec1.w, ec2.w, ec3.w],
            encBConv: [ec0.b, ec1.b, ec2.b, ec3.b],
            durW1: d1.w,
            durB1: d1.b,
            durW2: d2.w,
            durB2: d2.b,
            f0W1: f1.w,
            f0B1: f1.b,
            f0W2: f2.w,
            f0B2: f2.b,
            energyW1: e1.w,
            energyB1: e1.b,
            energyW2: e2.w,
            energyB2: e2.b,
            decWIn: di.w,
            decBIn: di.b,
            decWConv: [dc0.w, dc1.w, dc2.w, dc3.w],
            decBConv: [dc0.b, dc1.b, dc2.b, dc3.b],
            decWOut: dOut.w,
            decBOut: dOut.b,
            postWConv: [pc0.w, pc1.w, pc2.w, pc3.w, pc4.w],
            postBConv: [pc0.b, pc1.b, pc2.b, pc3.b, pc4.b],
            resW1: rw1,
            resB1: rb1,
            resW2: rw2,
            resB2: rb2,
            encWFeat: getArr(encFeatProj.weight)
        )
    }

    /// 音素エンコーダ順伝播: [1, P] -> [1, P, 256]
    /// - Parameter feat: 韻律位置特徴 [1, P, prosodicFeatureCount]（nil の場合は加算しない）
    public func forwardEncoder(cur: MLXArray, prev: MLXArray, next: MLXArray, feat: MLXArray? = nil) -> MLXArray {
        let lrelu = LeakyReLU(negativeSlope: 0.1)
        var h = embedCur(cur) + embedPrev(prev) + embedNext(next) + encBIn.reshaped([1, 1, 256])
        // 特徴が渡されない場合も音素 ID 列から算出して加算し、Pure Swift 推論（常に加算）と数値一致を保つ
        let effectiveFeat: MLXArray
        switch feat {
        case .some(let f):
            effectiveFeat = f
        case .none:
            let ids = cur.reshaped([-1]).asArray(Int32.self)
            let featCount = FrameMelWeights.prosodicFeatureCount
            let rows = FrameMelModel.prosodicFeatures(phoneIds: ids)
            var flat = [Float]()
            flat.reserveCapacity(ids.count * featCount)
            for row in rows {
                flat.append(contentsOf: row)
            }
            effectiveFeat = MLXArray(flat, [1, ids.count, featCount])
        }
        h = h + encFeatProj(effectiveFeat)
        h = h + lrelu(encConv0(h))
        h = h + lrelu(encConv1(h))
        h = h + lrelu(encConv2(h))
        h = h + lrelu(encConv3(h))
        return h
    }

    /// 継続時間予測: [1, P, 256] -> [1, P]
    public func forwardDuration(encStates: MLXArray) -> MLXArray {
        let lrelu = LeakyReLU(negativeSlope: 0.1)
        let h1 = lrelu(durConv1(encStates))
        let z = durConv2(h1)
        let zClamped = clip(z, min: -20.0, max: 20.0)
        let dur = log(1.0 + exp(zClamped)) + 1.0
        return dur.squeezed(axis: -1)
    }

    /// F0 予測: [1, T, 257] -> [1, T, 1]
    public func forwardF0(frameStates: MLXArray, voicedMask: MLXArray) -> MLXArray {
        let lrelu = LeakyReLU(negativeSlope: 0.1)
        let h1 = lrelu(f0Conv1(frameStates))
        let z = f0Conv2(h1)
        let sig = MLX.sigmoid(z)
        let masked = sig * voicedMask
        return masked
    }

    /// エネルギー予測: [1, T, 257] -> [1, T, 1]
    public func forwardEnergy(frameStates: MLXArray) -> MLXArray {
        let lrelu = LeakyReLU(negativeSlope: 0.1)
        let h1 = lrelu(energyConv1(frameStates))
        let z = energyConv2(h1)
        return MLX.sigmoid(z)
    }

    /// メルデコーダ + PostNet: [1, T, 260] -> (decMel: [1, T, 64], postMel: [1, T, 64])
    public func forwardDecoder(condition: MLXArray) -> (decMel: MLXArray, postMel: MLXArray) {
        let lrelu = LeakyReLU(negativeSlope: 0.1)
        var h = lrelu(decIn(condition))
        h = h + lrelu(decConv0(h))
        h = h + lrelu(decConv1(h))
        h = h + lrelu(decConv2(h))
        h = h + lrelu(decConv3(h))
        let decMel = decOut(h)

        // PostNet
        let p0 = tanh(postConv0(decMel))
        let p1 = tanh(postConv1(p0))
        let p2 = tanh(postConv2(p1))
        let p3 = tanh(postConv3(p2))
        let residual = postConv4(p3)

        let postMel = decMel + residual
        return (decMel: decMel, postMel: postMel)
    }

    /// メル残差順伝播（残差モジュールが存在する場合のみ）
    public func forwardResidual(condition: MLXArray) -> MLXArray? {
        return melResidual?.forward(condition: condition)
    }
}

/// フォルマント残差予測モデル（2層時間畳み込み MLX 実装）
///
/// 設計仕様（design_mel_residual.md）:
/// 残差は 2 層。入力はデコーダと同じ 260 次元。
/// 1 層目はカーネル 3、padding 1、260 から 256、Leaky ReLU。初期化スケールは sqrt(2 / (3 * 260))。
/// 2 層目はカーネル 1、padding 0、256 から 64 で、重みとバイアスはすべて 0 で始める。
public final class MLXMelResidual: Module, @unchecked Sendable {
    public var resConv1: Conv1d
    public var resConv2: Conv1d

    public override init() {
        let inDim = 260
        let hiddenDim = 256
        let melDim = 64
        self.resConv1 = Conv1d(inputChannels: inDim, outputChannels: hiddenDim, kernelSize: 3, stride: 1, padding: 1)
        self.resConv2 = Conv1d(inputChannels: hiddenDim, outputChannels: melDim, kernelSize: 1, stride: 1, padding: 0)
        super.init()
        self.initWeights()
    }

    public init(w1: [Float], b1: [Float], w2: [Float], b2: [Float]) {
        let inDim = 260
        let hiddenDim = 256
        let melDim = 64
        self.resConv1 = Conv1d(inputChannels: inDim, outputChannels: hiddenDim, kernelSize: 3, stride: 1, padding: 1)
        self.resConv2 = Conv1d(inputChannels: hiddenDim, outputChannels: melDim, kernelSize: 1, stride: 1, padding: 0)
        super.init()
        self.importWeights(w1: w1, b1: b1, w2: w2, b2: b2)
    }

    public func initWeights(seed: UInt64 = 2026) {
        let initW = FrameMelWeights.makeInitialResidualWeights(seed: seed)
        self.importWeights(w1: initW.resW1, b1: initW.resB1, w2: initW.resW2, b2: initW.resB2)
    }

    public func importWeights(w1: [Float], b1: [Float], w2: [Float], b2: [Float]) {
        var p1 = ModuleParameters()
        p1[unwrapping: "weight"] = MLXArray(w1, [256, 3, 260])
        p1[unwrapping: "bias"] = MLXArray(b1, [256])
        self.resConv1.update(parameters: p1)

        var p2 = ModuleParameters()
        p2[unwrapping: "weight"] = MLXArray(w2, [64, 1, 256])
        p2[unwrapping: "bias"] = MLXArray(b2, [64])
        self.resConv2.update(parameters: p2)
    }

    public func exportWeights() -> (w1: [Float], b1: [Float], w2: [Float], b2: [Float]) {
        let w1 = self.resConv1.weight.asArray(Float.self)
        let b1 = self.resConv1.bias?.asArray(Float.self) ?? [Float](repeating: 0.0, count: 256)
        let w2 = self.resConv2.weight.asArray(Float.self)
        let b2 = self.resConv2.bias?.asArray(Float.self) ?? [Float](repeating: 0.0, count: 64)
        return (w1, b1, w2, b2)
    }

    public func forward(condition: MLXArray) -> MLXArray {
        let lrelu = LeakyReLU(negativeSlope: 0.1)
        let h = lrelu(resConv1(condition))
        let res = resConv2(h)
        return res
    }
}
#endif
