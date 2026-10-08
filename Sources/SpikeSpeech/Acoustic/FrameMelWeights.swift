import Foundation

/// SpikeVoice (arXiv:2408.00788) に基づくフレーム単位対数メルモデルの重みパラメータ
///
/// 音素エンコーダ、分散アダプタ（継続時間・F0・エネルギー予測）、
/// 時間畳み込みメルデコーダ、および PostNet 残差予測器の全パラメータを保持・永続化する。
public struct FrameMelWeights: Sendable, Codable, Equatable {
    // 1. 音素エンコーダ (Phoneme Encoder)
    public let embedCur: [Float]       // [64 * 256]
    public let embedPrev: [Float]      // [64 * 256]
    public let embedNext: [Float]      // [64 * 256]
    public let encBIn: [Float]         // [256]
    public let encWConv: [[Float]]     // 4 layers of [256 * 3 * 256]
    public let encBConv: [[Float]]     // 4 layers of [256]

    // 2. 分散アダプタ (Variance Adaptor)
    // 継続時間予測器 (Duration Predictor)
    public let durW1: [Float]          // [128 * 3 * 256]
    public let durB1: [Float]          // [128]
    public let durW2: [Float]          // [1 * 1 * 128]
    public let durB2: [Float]          // [1]

    // F0 予測器 (F0 Predictor)
    public let f0W1: [Float]           // [128 * 5 * 257]
    public let f0B1: [Float]           // [128]
    public let f0W2: [Float]           // [1 * 1 * 128]
    public let f0B2: [Float]           // [1]

    // エネルギー予測器 (Energy Predictor)
    public let energyW1: [Float]       // [128 * 5 * 257]
    public let energyB1: [Float]       // [128]
    public let energyW2: [Float]       // [1 * 1 * 128]
    public let energyB2: [Float]       // [1]

    // 3. メルデコーダ (Mel Decoder)
    public let decWIn: [Float]         // [256 * 1 * 260]
    public let decBIn: [Float]         // [256]
    public let decWConv: [[Float]]     // 4 layers of [256 * 17 * 256]
    public let decBConv: [[Float]]     // 4 layers of [256]
    public let decWOut: [Float]        // [64 * 1 * 256]
    public let decBOut: [Float]        // [64]

    // 4. PostNet
    public let postWConv: [[Float]]    // 5 layers
    public let postBConv: [[Float]]    // 5 layers

    public init(
        embedCur: [Float],
        embedPrev: [Float],
        embedNext: [Float],
        encBIn: [Float],
        encWConv: [[Float]],
        encBConv: [[Float]],
        durW1: [Float],
        durB1: [Float],
        durW2: [Float],
        durB2: [Float],
        f0W1: [Float],
        f0B1: [Float],
        f0W2: [Float],
        f0B2: [Float],
        energyW1: [Float],
        energyB1: [Float],
        energyW2: [Float],
        energyB2: [Float],
        decWIn: [Float],
        decBIn: [Float],
        decWConv: [[Float]],
        decBConv: [[Float]],
        decWOut: [Float],
        decBOut: [Float],
        postWConv: [[Float]],
        postBConv: [[Float]]
    ) {
        self.embedCur = embedCur
        self.embedPrev = embedPrev
        self.embedNext = embedNext
        self.encBIn = encBIn
        self.encWConv = encWConv
        self.encBConv = encBConv
        self.durW1 = durW1
        self.durB1 = durB1
        self.durW2 = durW2
        self.durB2 = durB2
        self.f0W1 = f0W1
        self.f0B1 = f0B1
        self.f0W2 = f0W2
        self.f0B2 = f0B2
        self.energyW1 = energyW1
        self.energyB1 = energyB1
        self.energyW2 = energyW2
        self.energyB2 = energyB2
        self.decWIn = decWIn
        self.decBIn = decBIn
        self.decWConv = decWConv
        self.decBConv = decBConv
        self.decWOut = decWOut
        self.decBOut = decBOut
        self.postWConv = postWConv
        self.postBConv = postBConv
    }

    /// 決定論的疑似乱数による初期化重みの生成
    public static func randomWeights(
        seed: UInt64 = 2026,
        meanMel: [Float]? = nil
    ) -> FrameMelWeights {
        var rngState = seed
        func nextUniform(scale: Float) -> Float {
            rngState ^= rngState << 13
            rngState ^= rngState >> 7
            rngState ^= rngState << 17
            let u01 = Float(rngState & 0x00FFFFFF) / Float(0x01000000)
            return (u01 * 2.0 - 1.0) * scale
        }

        func makeArray(count: Int, scale: Float) -> [Float] {
            var arr = [Float](repeating: 0.0, count: count)
            var i = 0
            while i < count {
                arr[i] = nextUniform(scale: scale)
                i += 1
            }
            return arr
        }

        let hiddenDim = 256
        let melDim = 64
        let vocabSize = 64

        // 1. エンコーダ
        let embScale = sqrtf(2.0 / Float(hiddenDim))
        let embedCur = makeArray(count: vocabSize * hiddenDim, scale: embScale)
        let embedPrev = makeArray(count: vocabSize * hiddenDim, scale: embScale)
        let embedNext = makeArray(count: vocabSize * hiddenDim, scale: embScale)
        let encBIn = [Float](repeating: 0.0, count: hiddenDim)

        let encConvScale = sqrtf(2.0 / Float(3 * hiddenDim))
        var encWConv: [[Float]] = []
        var encBConv: [[Float]] = []
        var l = 0
        while l < 4 {
            encWConv.append(makeArray(count: hiddenDim * 3 * hiddenDim, scale: encConvScale))
            encBConv.append([Float](repeating: 0.0, count: hiddenDim))
            l += 1
        }

        // 2. 継続時間予測器 (256 -> 128 -> 1)
        let durScale1 = sqrtf(2.0 / Float(3 * hiddenDim))
        let durW1 = makeArray(count: 128 * 3 * hiddenDim, scale: durScale1)
        let durB1 = [Float](repeating: 0.0, count: 128)
        let durW2 = makeArray(count: 1 * 1 * 128, scale: 0.01)
        // 初期継続時間を約 8 フレーム (softplus(2.0) + 1.0 ≈ 8.1) に設定
        let durB2 = [Float](repeating: 2.0, count: 1)

        // F0 予測器 (257 -> 128 -> 1)
        let f0Scale1 = sqrtf(2.0 / Float(5 * 257))
        let f0W1 = makeArray(count: 128 * 5 * 257, scale: f0Scale1)
        let f0B1 = [Float](repeating: 0.0, count: 128)
        let f0W2 = makeArray(count: 1 * 1 * 128, scale: 0.01)
        // 初期 F0 を約 230 Hz (230/500 = 0.46, logit(0.46) ≈ -0.16) に設定
        let f0B2 = [Float](repeating: -0.16, count: 1)

        // エネルギー予測器 (257 -> 128 -> 1)
        let energyScale1 = sqrtf(2.0 / Float(5 * 257))
        let energyW1 = makeArray(count: 128 * 5 * 257, scale: energyScale1)
        let energyB1 = [Float](repeating: 0.0, count: 128)
        let energyW2 = makeArray(count: 1 * 1 * 128, scale: 0.01)
        // 初期エネルギーを約 0.50 (logit(0.5) = 0.0) に設定
        let energyB2 = [Float](repeating: 0.0, count: 1)

        // 3. メルデコーダ (260 -> 256 -> 64)
        let decInScale = sqrtf(2.0 / Float(260))
        let decWIn = makeArray(count: hiddenDim * 1 * 260, scale: decInScale)
        let decBIn = [Float](repeating: 0.0, count: hiddenDim)

        let decConvScale = sqrtf(2.0 / Float(17 * hiddenDim))
        var decWConv: [[Float]] = []
        var decBConv: [[Float]] = []
        l = 0
        while l < 4 {
            decWConv.append(makeArray(count: hiddenDim * 17 * hiddenDim, scale: decConvScale))
            decBConv.append([Float](repeating: 0.0, count: hiddenDim))
            l += 1
        }

        let decOutScale = sqrtf(2.0 / Float(hiddenDim))
        let decWOut = makeArray(count: melDim * 1 * hiddenDim, scale: decOutScale)
        var decBOut = [Float](repeating: 0.0, count: melDim)
        switch meanMel {
        case .some(let mm):
            var c = 0
            while c < melDim {
                if c < mm.count {
                    decBOut[c] = mm[c]
                }
                c += 1
            }
        case .none:
            break
        }

        // 4. PostNet
        var postWConv: [[Float]] = []
        var postBConv: [[Float]] = []

        // 層 0: 64 -> 256, kernel 5
        let postScale0 = sqrtf(2.0 / Float(5 * melDim))
        postWConv.append(makeArray(count: hiddenDim * 5 * melDim, scale: postScale0))
        postBConv.append([Float](repeating: 0.0, count: hiddenDim))

        // 層 1..3: 256 -> 256, kernel 5
        let postScaleMid = sqrtf(2.0 / Float(5 * hiddenDim))
        l = 1
        while l < 4 {
            postWConv.append(makeArray(count: hiddenDim * 5 * hiddenDim, scale: postScaleMid))
            postBConv.append([Float](repeating: 0.0, count: hiddenDim))
            l += 1
        }

        // 層 4: 256 -> 64, kernel 5 (残差出力は微小値で初期化)
        postWConv.append(makeArray(count: melDim * 5 * hiddenDim, scale: 0.001))
        postBConv.append([Float](repeating: 0.0, count: melDim))

        return FrameMelWeights(
            embedCur: embedCur,
            embedPrev: embedPrev,
            embedNext: embedNext,
            encBIn: encBIn,
            encWConv: encWConv,
            encBConv: encBConv,
            durW1: durW1,
            durB1: durB1,
            durW2: durW2,
            durB2: durB2,
            f0W1: f0W1,
            f0B1: f0B1,
            f0W2: f0W2,
            f0B2: f0B2,
            energyW1: energyW1,
            energyB1: energyB1,
            energyW2: energyW2,
            energyB2: energyB2,
            decWIn: decWIn,
            decBIn: decBIn,
            decWConv: decWConv,
            decBConv: decBConv,
            decWOut: decWOut,
            decBOut: decBOut,
            postWConv: postWConv,
            postBConv: postBConv
        )
    }
}
