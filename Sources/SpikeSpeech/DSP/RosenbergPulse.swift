import Foundation

/// 声門開口および閉口モデルに基づく有声音源励起パルス生成器
///
/// 単一インパルス列は高域成分が過多となり合成音声に耳障りな金属的クリック音を発生させるため、
/// 開口から滑らかな閉口に至る非対称な声門容積速度多項式をシミュレートし、自然な声帯振動の倍音スペクトルを生成する。
public final class RosenbergPulse: @unchecked Sendable {

    public let sampleRate: Float
    public let n1Ratio: Float // 開口時間比率
    public let n2Ratio: Float // 閉口時間比率
    private let openCloseSum: Float
    private let invN1: Float
    private let invN2: Float
    private let meanDcOffset: Float // 直流成分オフセット

    /// 連続発振器位相 [0.0, 1.0)
    private var phase: Float = 0.0

    /// 初期化
    ///
    /// 音声音響学における標準的な声門気流計測データに基づき、
    /// 急激な声帯閉鎖による明瞭な音響励起と滑らかな倍音減衰特性を両立する比率を設定する。
    public init(
        sampleRate: Float = 16000.0,
        n1Ratio: Float = 0.40,
        n2Ratio: Float = 0.16
    ) {
        self.sampleRate = sampleRate
        self.n1Ratio = n1Ratio
        self.n2Ratio = n2Ratio
        self.openCloseSum = n1Ratio + n2Ratio
        self.invN1 = 1.0 / n1Ratio
        self.invN2 = 1.0 / n2Ratio
        // 理論平均値による直流成分
        self.meanDcOffset = (0.5 * n1Ratio) + ((2.0 / 3.0) * n2Ratio)
    }

    /// 位相を初期状態にリセット
    ///
    /// 文の開始時や長時間のポーズ直後において過去の発振位相を持ち越さずに
    /// ゼロ位相から波形生成を開始する。
    public func reset() {
        phase = 0.0
    }

    /// 現在の位相値を取得
    public var currentPhase: Float {
        return phase
    }

    /// 指定位相における Rosenberg パルスの瞬時値を計算
    ///
    /// 3次多項式の開口期と2次多項式の閉口期の計算を分岐予測しやすい
    /// ネスト構造で展開し、関数呼び出しオーバーヘッドを削減する。
    @inline(__always)
    public func pulseValue(at p: Float) -> Float {
        if p < n1Ratio {
            // 開口期
            let tau = p * invN1
            return (tau * tau) * (3.0 - (2.0 * tau))
        } else {
            if p < openCloseSum {
                // 閉口期
                let tau = (p - n1Ratio) * invN2
                return 1.0 - (tau * tau)
            } else {
                // 閉鎖期
                return 0.0
            }
        }
    }

    /// 1 サンプル分のパルスを生成し、位相を進める
    ///
    /// 無声区間、非有限値、およびナイキスト周波数以上の周波数を検知した際に
    /// 発振器の更新を抑止し、無限ループやノイズ混入を防止する。
    @inline(__always)
    public func nextSample(f0: Float, removeDC: Bool = true) -> Float {
        if f0.isFinite != true || f0 <= 0.0 || (sampleRate * 0.5) <= f0 {
            return 0.0
        }

        let raw = pulseValue(at: phase)

        // 声門容積波形の直流成分が後段の全極フィルタに積分されベースラインドリフトを起こすのを防ぐ。
        let out: Float
        if removeDC {
            out = raw - meanDcOffset
        } else {
            out = raw
        }

        let phaseStep = f0 / sampleRate
        phase += phaseStep
        if 1.0 <= phase {
            phase = phase.truncatingRemainder(dividingBy: 1.0)
        }
        if phase.isFinite != true || phase < 0.0 {
            phase = 0.0
        }
        return out
    }

    /// 指定されたフレーム長分のパルス波形を一括生成
    ///
    /// ボコーダーのフレーム処理ループにおいて1サンプルごとのメソッド呼び出しと
    /// 中間配列確保を排除してキャッシュ局所性を高める。
    @inline(__always)
    public func generateFrame(
        f0: Float,
        count: Int,
        dst: UnsafeMutablePointer<Float>,
        removeDC: Bool = true
    ) {
        if f0.isFinite != true || f0 <= 0.0 || (sampleRate * 0.5) <= f0 {
            var i = 0
            while i < count {
                dst[i] = 0.0
                i += 1
            }
            return
        }

        let phaseStep = f0 / sampleRate
        var i = 0
        while i < count {
            let raw = pulseValue(at: phase)
            if removeDC {
                dst[i] = raw - meanDcOffset
            } else {
                dst[i] = raw
            }
            phase += phaseStep
            if 1.0 <= phase {
                phase = phase.truncatingRemainder(dividingBy: 1.0)
            }
            if phase.isFinite != true || phase < 0.0 {
                phase = 0.0
            }
            i += 1
        }
    }
}
