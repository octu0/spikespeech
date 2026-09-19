import Foundation

/// 音素継続時間（Duration）予測器の重みパラメータ
public struct DurationPredictorWeights: Sendable, Codable, Equatable {
    public let inputDim: Int
    public let hiddenDim: Int
    public let w1: [Float] // [hiddenDim * inputDim]
    public let b1: [Float] // [hiddenDim]
    public let w2: [Float] // [1 * hiddenDim]
    public let b2: [Float] // [1]

    public init(
        inputDim: Int = 72,
        hiddenDim: Int = 64,
        w1: [Float],
        b1: [Float],
        w2: [Float],
        b2: [Float]
    ) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
    }

    /// 決定論的ランダム初期化重みを生成する
    public static func randomWeights(
        inputDim: Int = 72,
        hiddenDim: Int = 64,
        seed: UInt64 = 2026
    ) -> DurationPredictorWeights {
        var rngState = seed
        let scale1 = sqrtf(2.0 / Float(inputDim))
        let scale2: Float = 0.01 // 出力オフセット初期値をゼロ近傍にする

        func nextUniform(scale: Float) -> Float {
            rngState ^= rngState << 13
            rngState ^= rngState >> 7
            rngState ^= rngState << 17
            let u01 = Float(rngState & 0x00FFFFFF) / Float(0x01000000)
            return (u01 * 2.0 - 1.0) * scale
        }

        var w1 = [Float](repeating: 0.0, count: hiddenDim * inputDim)
        var i = 0
        while i < w1.count {
            w1[i] = nextUniform(scale: scale1)
            i += 1
        }
        let b1 = [Float](repeating: 0.0, count: hiddenDim)

        var w2 = [Float](repeating: 0.0, count: hiddenDim)
        i = 0
        while i < w2.count {
            w2[i] = nextUniform(scale: scale2)
            i += 1
        }
        let b2 = [Float](repeating: 0.0, count: 1)

        return DurationPredictorWeights(
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            w1: w1,
            b1: b1,
            w2: w2,
            b2: b2
        )
    }
}

/// 基本周波数（F0）系列予測器の重みパラメータ
///
/// 従来の藤崎掛け算 Prior を撤廃し、系列特徴量に対する 1D Depthwise Conv (K=3) と
/// 線形射影により、有声フレームの対数周波数（log Hz）を直接予測する。
public struct F0PredictorWeights: Sendable, Codable, Equatable {
    public let inputDim: Int
    public let hiddenDim: Int
    public let w1: [Float]       // [hiddenDim * inputDim]
    public let b1: [Float]       // [hiddenDim]
    public let wConv: [Float]    // [5 * hiddenDim] (Kernel size 5 Depthwise Conv) または [3 * hiddenDim]
    public let w2: [Float]       // [1 * hiddenDim]
    public let b2: [Float]       // [1]

    public init(
        inputDim: Int = 76,
        hiddenDim: Int = 64,
        w1: [Float],
        b1: [Float],
        wConv: [Float]? = nil,
        w2: [Float],
        b2: [Float]
    ) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.w1 = w1
        self.b1 = b1
        if let explicitConv = wConv, explicitConv.isEmpty != true {
            self.wConv = explicitConv
        } else {
            // デフォルト: K=5 [0.10, 0.20, 0.40, 0.20, 0.10] の平滑化フィルタ
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
            self.wConv = defaultConv
        }
        self.w2 = w2
        self.b2 = b2
    }

    enum CodingKeys: String, CodingKey {
        case inputDim
        case hiddenDim
        case w1
        case b1
        case wConv
        case w2
        case b2
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.inputDim = try container.decode(Int.self, forKey: .inputDim)
        self.hiddenDim = try container.decode(Int.self, forKey: .hiddenDim)
        self.w1 = try container.decode([Float].self, forKey: .w1)
        self.b1 = try container.decode([Float].self, forKey: .b1)
        if let decodedConv = try container.decodeIfPresent([Float].self, forKey: .wConv), decodedConv.isEmpty != true {
            self.wConv = decodedConv
        } else {
            var defaultConv = [Float](repeating: 0.0, count: 5 * self.hiddenDim)
            var c = 0
            while c < self.hiddenDim {
                defaultConv[(0 * self.hiddenDim) + c] = 0.10
                defaultConv[(1 * self.hiddenDim) + c] = 0.20
                defaultConv[(2 * self.hiddenDim) + c] = 0.40
                defaultConv[(3 * self.hiddenDim) + c] = 0.20
                defaultConv[(4 * self.hiddenDim) + c] = 0.10
                c += 1
            }
            self.wConv = defaultConv
        }
        self.w2 = try container.decode([Float].self, forKey: .w2)
        self.b2 = try container.decode([Float].self, forKey: .b2)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(inputDim, forKey: .inputDim)
        try container.encode(hiddenDim, forKey: .hiddenDim)
        try container.encode(w1, forKey: .w1)
        try container.encode(b1, forKey: .b1)
        try container.encode(wConv, forKey: .wConv)
        try container.encode(w2, forKey: .w2)
        try container.encode(b2, forKey: .b2)
    }

    /// 決定論的ランダム初期化重みを生成する
    public static func randomWeights(
        inputDim: Int = 76,
        hiddenDim: Int = 64,
        seed: UInt64 = 2027
    ) -> F0PredictorWeights {
        var rngState = seed
        let scale1 = sqrtf(2.0 / Float(inputDim))
        let scale2: Float = 0.08

        func nextUniform(scale: Float) -> Float {
            rngState ^= rngState << 13
            rngState ^= rngState >> 7
            rngState ^= rngState << 17
            let u01 = Float(rngState & 0x00FFFFFF) / Float(0x01000000)
            return (u01 * 2.0 - 1.0) * scale
        }

        var w1 = [Float](repeating: 0.0, count: hiddenDim * inputDim)
        var i = 0
        while i < w1.count {
            w1[i] = nextUniform(scale: scale1)
            i += 1
        }
        let b1 = [Float](repeating: 0.0, count: hiddenDim)

        var wConv = [Float](repeating: 0.0, count: 5 * hiddenDim)
        var c = 0
        while c < hiddenDim {
            wConv[(0 * hiddenDim) + c] = 0.10 + nextUniform(scale: 0.01)
            wConv[(1 * hiddenDim) + c] = 0.20 + nextUniform(scale: 0.01)
            wConv[(2 * hiddenDim) + c] = 0.40 + nextUniform(scale: 0.01)
            wConv[(3 * hiddenDim) + c] = 0.20 + nextUniform(scale: 0.01)
            wConv[(4 * hiddenDim) + c] = 0.10 + nextUniform(scale: 0.01)
            c += 1
        }

        var w2 = [Float](repeating: 0.0, count: hiddenDim)
        i = 0
        while i < w2.count {
            w2[i] = nextUniform(scale: scale2)
            i += 1
        }
        // なぜ b2 初期値を log(220) ≈ 5.3936 にするか:
        // 日本語女性話者（JSUT）の実音声平均基本周波数（~220Hz）に対数空間で初期アンカーし、
        // 学習開始直後から有声 F0 MAE が 20〜30Hz 近傍からスタートして速やかに < 20Hz へ収束できるようにするため。
        let b2 = [Float]([logf(220.0)])

        return F0PredictorWeights(
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            w1: w1,
            b1: b1,
            wConv: wConv,
            w2: w2,
            b2: b2
        )
    }
}

/// 韻律予測器（Duration & F0）統合重み構造体
public struct ProsodyWeights: Sendable, Codable, Equatable {
    public let durationWeights: DurationPredictorWeights
    public let f0Weights: F0PredictorWeights

    public init(
        durationWeights: DurationPredictorWeights,
        f0Weights: F0PredictorWeights
    ) {
        self.durationWeights = durationWeights
        self.f0Weights = f0Weights
    }

    /// 決定論的初期化重みを生成する
    public static func randomWeights() -> ProsodyWeights {
        return ProsodyWeights(
            durationWeights: DurationPredictorWeights.randomWeights(),
            f0Weights: F0PredictorWeights.randomWeights()
        )
    }
}

/// 韻律学習（Duration / F0）用サンプルデータ
public struct ProsodyTrainingSample: Sendable {
    public let durationFeatures: [[Float]] // 各音素の特徴量 [72]
    public let ruleDurations: [Float]      // 各音素の規則フレーム数
    public let targetDurations: [Float]    // 各音素の実音声アライメントフレーム数
    public let f0Features: [[Float]]       // 各フレームの特徴量 [76]
    public let fujisakiF0: [Float]         // 各フレームの藤崎規則 F0
    public let targetF0: [Float]           // 各フレームの PitchTracker 実測 F0
    public let voicedMask: [Float]         // 各フレームの有声フラグ (1.0 or 0.0)

    public init(
        durationFeatures: [[Float]],
        ruleDurations: [Float],
        targetDurations: [Float],
        f0Features: [[Float]],
        fujisakiF0: [Float],
        targetF0: [Float],
        voicedMask: [Float]
    ) {
        self.durationFeatures = durationFeatures
        self.ruleDurations = ruleDurations
        self.targetDurations = targetDurations
        self.f0Features = f0Features
        self.fujisakiF0 = fujisakiF0
        self.targetF0 = targetF0
        self.voicedMask = voicedMask
    }
}
