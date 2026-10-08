#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// メル残差モデル専用の学習損失コンテナ
public struct MelResidualLosses: Sendable {
    public let totalLoss: Float
    public let melL1: Float
    public let deltaMelL1: Float

    public init(totalLoss: Float, melL1: Float, deltaMelL1: Float) {
        self.totalLoss = totalLoss
        self.melL1 = melL1
        self.deltaMelL1 = deltaMelL1
    }
}

/// 凍結エポック 19 モデルを土台としたメル残差専用トレーナー（MLX 実装）
///
/// 設計仕様（design_mel_residual.md）:
/// 1. エポック 19 の全パラメータ（エンコーダ、継続時間、F0、エネルギー、デコーダ、PostNet）を完全凍結。
/// 2. オプティマイザには残差モジュール（MLXMelResidual）のパラメータのみを登録。
/// 3. 損失は「残差を足したメルの L1」と「母音 ID 5,6,7,8,9 の内側だけのフレーム差 L1」の二つだけ。
public final class MLXMelResidualTrainer: @unchecked Sendable {
    public let frozenModel: MLXFrameMelModel
    public let residualModel: MLXMelResidual
    public let optimizer: Adam
    public var deltaLossWeight: Float
    public private(set) var lastLosses: MelResidualLosses

    public init(
        frozenModel: MLXFrameMelModel,
        residualModel: MLXMelResidual,
        learningRate: Float = 0.001,
        deltaLossWeight: Float = 1.0
    ) {
        self.frozenModel = frozenModel
        self.residualModel = residualModel
        self.optimizer = Adam(learningRate: learningRate)
        self.deltaLossWeight = deltaLossWeight
        self.lastLosses = MelResidualLosses(totalLoss: 0.0, melL1: 0.0, deltaMelL1: 0.0)
    }

    public func setLearningRate(_ lr: Float) {
        optimizer.learningRate = lr
    }

    public func currentLearningRate() -> Float {
        return optimizer.learningRate
    }

    /// 1 発話サンプルの残差学習ステップ
    public func trainSample(
        phoneIds: [Int32],
        targetDurations: [Int],
        targetMel: [[Float]],
        targetF0: [Float],
        targetEnergy: [Float]
    ) -> MelResidualLosses {
        let phoneCount = phoneIds.count
        let totalFrames = targetMel.count
        if phoneCount < 3 || totalFrames <= 1 {
            return MelResidualLosses(totalLoss: 0.0, melL1: 0.0, deltaMelL1: 0.0)
        }

        // 1. トライフォン文脈（prev, cur, next）の作成
        var curP = [Int32](repeating: 0, count: phoneCount)
        var prevP = [Int32](repeating: 0, count: phoneCount)
        var nextP = [Int32](repeating: 0, count: phoneCount)
        var p = 0
        while p < phoneCount {
            curP[p] = phoneIds[p]
            if 0 < p {
                prevP[p] = phoneIds[p - 1]
            } else {
                prevP[p] = Int32(PhonemeVocabulary.silId)
            }
            if (p + 1) < phoneCount {
                nextP[p] = phoneIds[p + 1]
            } else {
                nextP[p] = Int32(PhonemeVocabulary.silId)
            }
            p += 1
        }

        // 2. 長さ調節器用インデックスおよび音素内位置 (phonePos) の算出
        var gatherIndices = [Int32](repeating: 0, count: totalFrames)
        var pos = [Float](repeating: 0.0, count: totalFrames)
        var curF = 0
        p = 0
        while p < phoneCount {
            let dur = targetDurations[p]
            let maxF = Float(max(1, dur - 1))
            var f = 0
            while f < dur {
                let fIdx = curF + f
                if fIdx < totalFrames {
                    gatherIndices[fIdx] = Int32(p)
                    pos[fIdx] = Float(f) / maxF
                }
                f += 1
            }
            curF += dur
            p += 1
        }

        // 3. 目標 Delta F0 の算出
        var deltaF0 = [Float](repeating: 0.0, count: totalFrames)
        var t = 0
        while t < totalFrames {
            if 0 < t {
                let curF0 = targetF0[t]
                let prevF0 = targetF0[t - 1]
                if 0.0 < curF0 && 0.0 < prevF0 {
                    let d = (curF0 - prevF0) / 0.1
                    deltaF0[t] = max(-1.0, min(1.0, d))
                }
            }
            t += 1
        }

        // 4. 目標対数メルスペクトルのフラット化
        var targetMelFlat = [Float]()
        targetMelFlat.reserveCapacity(totalFrames * 64)
        t = 0
        while t < totalFrames {
            targetMelFlat.append(contentsOf: targetMel[t])
            t += 1
        }

        let curArr = MLXArray(curP, [1, phoneCount])
        let prevArr = MLXArray(prevP, [1, phoneCount])
        let nextArr = MLXArray(nextP, [1, phoneCount])
        let gatherArr = MLXArray(gatherIndices)
        let posArr = MLXArray(pos, [totalFrames, 1])
        let targetF0Arr = MLXArray(targetF0, [totalFrames, 1])
        let deltaF0Arr = MLXArray(deltaF0, [totalFrames, 1])
        let targetEnergyArr = MLXArray(targetEnergy, [totalFrames, 1])
        let targetMelArr = MLXArray(targetMelFlat, [1, totalFrames, 64])

        // 5. 母音 (ID 5, 6, 7, 8, 9) の内側フレーム差マスク (totalFrames - 1)
        // 区間の先頭と末尾のフレームは内側に入れない。t-1 が別の音素なら入れない。
        var deltaMask = [Float](repeating: 0.0, count: totalFrames - 1)
        var curFOffset = 0
        p = 0
        while p < phoneCount {
            let pid = Int(phoneIds[p])
            let dur = targetDurations[p]
            var isVowel = false
            switch pid {
            case 5, 6, 7, 8, 9:
                isVowel = true
            default:
                break
            }
            if isVowel && 3 < dur {
                var f = 2
                while f <= (dur - 2) {
                    let tf = curFOffset + f
                    if tf < totalFrames {
                        deltaMask[tf - 1] = 1.0
                    }
                    f += 1
                }
            }
            curFOffset += dur
            p += 1
        }
        let deltaMaskArr = MLXArray(deltaMask, [1, totalFrames - 1, 1])

        // 6. 凍結エポック 19 モデルによる PostNet 後メルの事前算出（勾配追跡なし）
        let encStates = frozenModel.forwardEncoder(cur: curArr, prev: prevArr, next: nextArr)
        let frameEnc = take(encStates.squeezed(axis: 0), gatherArr, axis: 0)
        let decCondition = concatenated([frameEnc, targetF0Arr, deltaF0Arr, targetEnergyArr, posArr], axis: -1).expandedDimensions(axis: 0)
        let (_, basePostMel) = frozenModel.forwardDecoder(condition: decCondition)
        let detachedBasePostMel = stopGradient(basePostMel)
        let detachedCondition = stopGradient(decCondition)

        // 7. 残差モジュールのみの勾配計算とオプティマイザ更新
        let dWeight = self.deltaLossWeight
        let lg = valueAndGrad(model: residualModel) { (resMod: MLXMelResidual, _) -> [MLXArray] in
            let resMel = resMod.forward(condition: detachedCondition)
            let mel = detachedBasePostMel + resMel

            // 損失 1: 残差を足したメルと教師対数メルの L1
            let melL1 = mean(abs(mel - targetMelArr))

            // 損失 2: 母音 ID 5, 6, 7, 8, 9 の内側だけのフレーム差 L1
            let predDelta = mel[0..., 1..<totalFrames, 0...] - mel[0..., 0..<(totalFrames - 1), 0...]
            let targetDelta = targetMelArr[0..., 1..<totalFrames, 0...] - targetMelArr[0..., 0..<(totalFrames - 1), 0...]
            let deltaDiff = abs(predDelta - targetDelta)
            let maskedDelta = deltaDiff * deltaMaskArr
            let maskSum = sum(deltaMaskArr)
            let deltaLoss = sum(maskedDelta) / maximum(maskSum * 64.0, MLXArray(1.0))

            let totalLoss = melL1 + (deltaLoss * MLXArray(dWeight))
            return [totalLoss, melL1, deltaLoss]
        }

        let (lossVals, grads) = lg(residualModel, [])
        optimizer.update(model: residualModel, gradients: grads)
        eval(residualModel, optimizer)

        let losses = MelResidualLosses(
            totalLoss: lossVals[0].item(Float.self),
            melL1: lossVals[1].item(Float.self),
            deltaMelL1: lossVals[2].item(Float.self)
        )
        self.lastLosses = losses
        return losses
    }
}
#endif
