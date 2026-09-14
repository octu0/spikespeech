#if canImport(MLX)
import Foundation
import MLX

/// Straight-Through Estimator (STE) スタイルの Fast Sigmoid 代理勾配
///
/// 順伝播時は厳密な 0.0 または 1.0 の離散ステップ（Heaviside 関数）で動作させて
/// SNN のスパイクダイナミクスを厳密に成立させつつ、逆伝播時のみ滑らかな Fast Sigmoid の
/// 導関数を流すことで勾配消失を防ぎ、エンドツーエンドの BPTT 学習を可能にする。
public enum SurrogateGradients {

    /// Fast Sigmoid 代理勾配によるスパイク生成ステップ
    ///
    /// MLX の自動微分グラフにおいて、順伝播の値は差分相殺により sHard となり、
    /// 逆伝播では stopGradient により第 1 項の勾配が 0 となって第 2 項の sSurrogate の勾配のみが流れる。
    @inline(__always)
    public static func fastSigmoidSTE(
        v: MLXArray,
        vTh: MLXArray,
        alpha: Float = 2.0
    ) -> MLXArray {
        let vRel = (v - vTh) * alpha
        let sSurrogate = Float(0.5) * ((vRel / (Float(1.0) + abs(vRel))) + Float(1.0))
        let sHard = (vTh .<= v).asType(.float32)
        return stopGradient(sHard - sSurrogate) + sSurrogate
    }
}
#endif
