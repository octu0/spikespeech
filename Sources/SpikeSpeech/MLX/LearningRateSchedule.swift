import Foundation

/// エポック単位の線形ウォームアップおよびコサイン減衰（Cosine Annealing with Warmup）スケジューラ。
/// なぜこのスケジューラを採用するか:
/// SNN の学習初期における膜電位と発火ダイナミクスの急激な乱れを線形ウォームアップで抑え、
/// 谷底に近づくにつれて実効ステップサイズを滑らかに縮小して Adam の実効学習率膨張によるオーバーシュートを防ぐため。
public struct CosineWarmupSchedule: Sendable, Equatable {
    public let lrBase: Float
    public let lrMin: Float
    public let warmupEpochs: Int
    public let totalEpochs: Int

    public init(
        lrBase: Float = 0.003,
        lrMin: Float = 1.0e-5,
        warmupEpochs: Int = 2,
        totalEpochs: Int = 15
    ) {
        var base = lrBase
        if base.isFinite != true || base < 0.0 {
            base = 0.003
        }
        var minLR = lrMin
        if minLR.isFinite != true || minLR < 0.0 {
            minLR = 0.0
        }
        if base < minLR {
            minLR = base
        }
        self.lrBase = base
        self.lrMin = minLR
        self.warmupEpochs = max(1, warmupEpochs)
        self.totalEpochs = max(self.warmupEpochs + 1, totalEpochs)
    }

    /// 指定エポックにおける目標学習率を算出する。
    /// - Parameter epoch: 0-indexed のエポック番号（0 ..< totalEpochs）
    public func learningRate(epoch: Int) -> Float {
        let e = max(0, epoch)
        if e < warmupEpochs {
            // なぜ線形補間とするか: 学習開始直後のスパイク枯渇・急峻な勾配変動を回避し徐々に適応させるため
            let numer = Float(e + 1)
            let denom = Float(warmupEpochs)
            return lrMin + (lrBase - lrMin) * (numer / denom)
        }
        // なぜ総エポック以降を lrMin にクランプするか:
        // 追加エポックを回した場合でも極小学習率による微調整を継続可能にするため
        let cosineSpan = max(1, totalEpochs - warmupEpochs)
        // なぜ +1 を加えるか:
        // warmupEpochs の最終エポックでピーク（lrBase）に到達した直後から減衰を開始し、
        // 最終エポック（totalEpochs - 1）で正確に lrMin に到達させるため
        var progress = Float(e - warmupEpochs + 1) / Float(cosineSpan)
        if 1.0 < progress {
            progress = 1.0
        }
        let cosine = (1.0 + cosf(Float.pi * progress)) * 0.5
        return lrMin + (lrBase - lrMin) * cosine
    }
}

/// 損失悪化時の非常ブレーキ用 Plateau ガード。
/// なぜこのガードを設けるか:
/// 予期しない勾配爆発や発火レジームの崩壊が連続した場合に学習率を強制減衰させて発散を食い止めるため。
public struct PlateauGuard: Sendable {
    public let patience: Int
    public let factor: Float
    public let relThreshold: Float
    public private(set) var bestLoss: Float
    public private(set) var badEpochs: Int
    public private(set) var decayMultiplier: Float

    public init(patience: Int = 2, factor: Float = 0.5, relThreshold: Float = 0.005) {
        self.patience = max(1, patience)
        var f = factor
        if f.isFinite != true || f <= 0.0 || 1.0 <= f {
            f = 0.5
        }
        self.factor = f
        self.relThreshold = max(0.0, relThreshold)
        self.bestLoss = Float.greatestFiniteMagnitude
        self.badEpochs = 0
        self.decayMultiplier = 1.0
    }

    /// エポック平均損失を観測し、必要に応じて減衰乗数を更新する。
    @discardableResult
    public mutating func observe(epochLoss: Float) -> Float {
        if epochLoss.isFinite != true {
            return decayMultiplier
        }
        let threshold = bestLoss * (1.0 + relThreshold)
        if epochLoss < bestLoss {
            bestLoss = epochLoss
            badEpochs = 0
            return decayMultiplier
        }
        if threshold < epochLoss {
            badEpochs += 1
            if patience <= badEpochs {
                decayMultiplier *= factor
                if decayMultiplier < 1.0e-3 {
                    decayMultiplier = 1.0e-3
                }
                badEpochs = 0
            }
        } else {
            badEpochs = 0
        }
        return decayMultiplier
    }
}

/// スケジューラと Plateau ガードを合成した最終実効学習率を算出する。
public func resolvedLearningRate(
    schedule: CosineWarmupSchedule,
    epoch: Int,
    plateauMultiplier: Float
) -> Float {
    let cosine = schedule.learningRate(epoch: epoch)
    var lr = cosine * plateauMultiplier
    if lr < schedule.lrMin {
        lr = schedule.lrMin
    }
    return lr
}

/// 決定論的エポックシャッフルユーティリティ。
/// なぜ外部乱数に頼らず xorshift64 を使用するか:
/// プラットフォーム差分やシステムクロック依存を排除し、シード値とエポック数から完全に決定論的な並び替えを再現するため。
public enum TrainingShuffle {
    /// エポック番号に応じた個別シードを生成する。
    public static func mixSeed(baseSeed: UInt64, epoch: Int) -> UInt64 {
        var x = baseSeed &+ (UInt64(epoch) &* 0x9E3779B97F4A7C15)
        x ^= x >> 30
        x &*= 0xBF58476D1CE4E5B9
        x ^= x >> 27
        x &*= 0x94D049BB133111EB
        x ^= x >> 31
        return x
    }

    public static func nextUInt64(state: inout UInt64) -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }

    /// 決定論的な Fisher-Yates アルゴリズムによる配列の in-place シャッフル。
    public static func shuffleInPlace<T>(_ data: inout [T], seed: UInt64) {
        if data.count <= 1 {
            return
        }
        var state = seed
        if state == 0 {
            state = 0xA5A5A5A5A5A5A5A5
        }
        var i = data.count - 1
        while 0 < i {
            let r = nextUInt64(state: &state)
            let j = Int(r % UInt64(i + 1))
            if i != j {
                let tmp = data[i]
                data[i] = data[j]
                data[j] = tmp
            }
            i -= 1
        }
    }
}

/// エポックスナップショットおよびモデル重みファイルの安全な保存ユーティリティ。
/// なぜエポックごとに独立ファイルとして書き出すか:
/// 途中の最良エポックの重みが後続エポックの上書きで破壊されるのを防ぎ、いつでも任意のエポック重みを取り出せるようにするため。
public enum WeightCheckpoint {
    /// エポック番号（1-indexed）に対応するファイル名（例: "weights.ep01.json"）を生成する。
    public static func epochFileName(epochOneIndexed: Int) -> String {
        let n = max(1, epochOneIndexed)
        let padded = String(format: "%02d", n)
        return "weights.ep\(padded).json"
    }

    /// ディレクトリとファイル名を結合した URL を生成する。
    public static func resolvePath(directory: String, fileName: String) -> URL {
        let dir = URL(fileURLWithPath: directory, isDirectory: true)
        return dir.appendingPathComponent(fileName)
    }

    /// モデル重みを整形済み JSON として安全にアトミック保存する。
    public static func atomicWritePretty(_ weights: SpikingNetworkWeights, to url: URL) throws {
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(weights)
        let tmp = url.appendingPathExtension("tmp")
        try data.write(to: tmp, options: .atomic)
        if FileManager.default.fileExists(atPath: url.path) {
            let replaced = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
            if replaced == nil {
                // なぜフォールバック移動を行うか: APFS / HFS 間の移動や特殊環境下で replaceItemAt が nil を返した場合に備えるため
                try? FileManager.default.removeItem(at: url)
                try FileManager.default.moveItem(at: tmp, to: url)
            }
        } else {
            try FileManager.default.moveItem(at: tmp, to: url)
        }
    }
}
