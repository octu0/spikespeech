#if canImport(MLX)
import Foundation
import MLX

/// SNN 音響モデル学習用のスペクトル回帰損失関数
///
/// 音声合成の音響モデルは離散分類ではなく連続スペクトルの多次元回帰問題であるため、
/// 周波数ビンごとの幾何学的距離を安定して最小化する。
public enum AcousticLossFunctions {

    /// L1 スペクトル再構成損失
    ///
    /// L2 損失と比較して外れ値に対する勾配が穏やかであり、スペクトルのフォルマント
    /// ピークや高周波帯域の微細構造を過度につぶさずに鮮明に再構成する。
    public static func spectralL1Loss(
        predicted: MLXArray,
        target: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let diff = abs(predicted - target)

        if let m = mask {
            let mExpanded: MLXArray
            if m.ndim < diff.ndim {
                mExpanded = expandedDimensions(m, axis: -1)
            } else {
                mExpanded = m
            }
            let maskedDiff = diff * mExpanded
            let featureDim = Float(predicted.shape[predicted.ndim - 1])
            let totalValid = sum(mExpanded) * featureDim
            return sum(maskedDiff) / (totalValid + 1e-5)
        }

        return mean(diff)
    }

    /// フレーム間差分スペクトル損失
    ///
    /// 静的スペクトルの絶対値だけでなく時間方向の動的変化をターゲットに一致させ、
    /// フレーム間の遷移における不連続性やクリックノイズを抑制する。
    public static func spectralDeltaLoss(
        predicted: MLXArray,
        target: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let pDelta = predicted[0..., 1..., 0...] - predicted[0..., ..<(-1), 0...]
        let tDelta = target[0..., 1..., 0...] - target[0..., ..<(-1), 0...]

        var deltaMask: MLXArray? = nil
        if let m = mask {
            deltaMask = m[0..., 1...]
        }

        return spectralL1Loss(predicted: pDelta, target: tDelta, mask: deltaMask)
    }
}
#endif
