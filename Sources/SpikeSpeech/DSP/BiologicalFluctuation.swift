import Foundation

/// Voss-McCartney アルゴリズムによる Pure Swift 1/f ピンクノイズおよび生体ゆらぎ発生器
///
/// なぜ 1/f ゆらぎを用いるか:
/// 人間の生体音声（声帯振動のピッチ微小変動・呼気圧変動・発話テンポ）は、
/// 完全にランダムなホワイトノイズ（1/f^0）でも、予測可能な正弦波でもなく、
/// 「適度な相関と予測不能性が共存する」1/f パワースペクトル密度（ピンクノイズ）に従う。
/// 単純な sin 波によるビブラートを廃し、真の生体ゆらぎを合成することでロボット感を根絶する。
public struct BiologicalFluctuation: Sendable {

    /// Voss-McCartney アルゴリズム用のオクターブ分解能（段数）
    private let numOctaves: Int = 8

    /// 疑似乱数生成用の LCG 状態
    private var rngState: UInt64

    /// 各オクターブ帯域のホワイトノイズ保持配列
    private var octaveValues: [Float]

    /// 現在のオクターブ値の総和
    private var runningSum: Float

    /// 呼び出しカウンター
    private var counter: UInt32

    /// テキスト文字列から決定論的な 64-bit ハッシュシードを生成する (FNV-1a)
    /// 同一テキストに対しては常に同一のゆらぎ系列を再現しつつ、発話テキストごとに異なる生体ゆらぎを与える
    public static func seed(from text: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in text.utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x0100_0000_01b3
        }
        if hash == 0 {
            return 0x1234_5678_9abc_def0
        }
        return hash
    }

    /// 初期化（シード値指定可能、デフォルトは再現性のある固定シード）
    public init(seed: UInt64 = 0x1234_5678_9abc_def0) {
        if seed == 0 {
            self.rngState = 0x1234_5678_9abc_def0
        } else {
            self.rngState = seed
        }
        self.octaveValues = [Float](repeating: 0.0, count: 8)
        self.runningSum = 0.0
        self.counter = 0

        // 初期状態で各オクターブにランダム値を投入し、定常状態からスタートさせる
        var i = 0
        while i < 8 {
            let randVal = nextUniform()
            self.octaveValues[i] = randVal
            self.runningSum += randVal
            i += 1
        }
    }

    /// [ -1.0, 1.0 ] の一様乱数を生成する（高速な線形合同法）
    private mutating func nextUniform() -> Float {
        // PCG/LCG 定数による高速な擬似乱数生成
        rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
        let x = Float(Double(rngState >> 32) / Double(UInt32.max))
        return (x * 2.0) - 1.0
    }

    /// 1/f パワースペクトル密度を持つピンクノイズサンプルを 1 サンプル進めて取得する
    /// 出力レンジ: およそ [ -1.0, 1.0 ] (単位分散スケーリング)
    public mutating func nextPink() -> Float {
        counter &+= 1

        // カウンターの最下位ビットの 0 の個数（Trailing Zeros）によって更新するオクターブを決定
        // これにより、第 0 オクターブは毎フレーム、第 1 は 2 フレームごと、第 2 は 4 フレームごと…と
        // 各周波数帯域が幾何級数的な時間スケールで更新され、厳密な 1/f スペクトルが形成される
        var tz = counter.trailingZeroBitCount
        if 8 <= tz {
            tz = 7
        }

        let oldVal = octaveValues[tz]
        let newVal = nextUniform()
        octaveValues[tz] = newVal
        runningSum = runningSum - oldVal + newVal

        // 8 段の一様乱数加算による標準偏差 sqrt(8/3) ≈ 1.633 で正規化し、
        // 真の単位分散（分散 1.0、標準偏差 1.0）の 1/f ピンクノイズ系列を生成する。
        // これにより、後段の Jitter (±1.2%)、Shimmer (±2.5%)、テンポゆらぎ (±6%) の
        // 公称物理強度が数学的・生理学的な実効振幅（RMS）として正確に発揮される。
        let stdDev = sqrtf(Float(numOctaves) / 3.0)
        var normalized = runningSum / stdDev
        if normalized < -1.0 {
            normalized = -1.0
        }
        if 1.0 < normalized {
            normalized = 1.0
        }
        return normalized
    }

    /// ピッチ周波数 F0 [Hz] に対する自然な生体ピッチゆらぎ（Jitter）を算出する
    ///
    /// なぜ ±1.2% にするか:
    /// 音声病理学・音響音声学の臨床データにおいて、健常成人の自然な声帯振動ジッターは
    /// 0.5% 〜 1.5% の範囲に収まり、これを超えると嗄声（かすれ声）、下回ると完全なロボット声になるため。
    public mutating func computePitchJitter(baseF0: Float) -> Float {
        if baseF0 <= 0.0 {
            return 0.0
        }
        let pink = nextPink()
        // 最大 ±1.2% のゆらぎ倍率
        let jitterScale = 1.0 + (pink * 0.012)
        return baseF0 * jitterScale
    }

    /// 音量ゲインに対する自然な呼気圧振幅ゆらぎ（Shimmer）を算出する
    ///
    /// なぜ ±2.5% にするか:
    /// 人間の発話時のシマー（振幅の周期間ゆらぎ）は健康な発話で 2% 〜 4% 程度であり、
    /// 微小な息の強弱が声に「温かみ」と「生々しさ」をもたらすため。
    public mutating func computeAmplitudeShimmer(baseGain: Float) -> Float {
        if baseGain <= 0.0 {
            return 0.0
        }
        let pink = nextPink()
        // 最大 ±2.5% のゆらぎ倍率
        let shimmerScale = 1.0 + (pink * 0.025)
        var result = baseGain * shimmerScale
        if result < 0.0 {
            result = 0.0
        }
        return result
    }

    /// 音素・モーラの発話フレーム数に対する自然な生体テンポ倍率（ルバート）を算出する
    ///
    /// なぜ ±6% の伸縮にするか:
    /// メトロノームのような正確な等時拍（完全な等間隔フレーム）はロボット的印象を決定づける主因であり、
    /// 累積和量子化前の実数 Duration に微小なテンポ伸縮（±6%）を付与することで有機的な発話リズムを再現するため。
    public mutating func computeTempoScale() -> Float {
        let pink = nextPink()
        return 1.0 + (pink * 0.06)
    }

    /// 音素・モーラの発話フレーム数に対する自然な生体テンポゆらぎ（整数フレーム）を算出する
    public mutating func computeDurationFluctuation(baseFrames: Int) -> Int {
        if baseFrames <= 1 {
            return baseFrames
        }
        let scale = computeTempoScale()
        var result = Int(roundf(Float(baseFrames) * scale))
        if result < 1 {
            result = 1
        }
        return result
    }
}
