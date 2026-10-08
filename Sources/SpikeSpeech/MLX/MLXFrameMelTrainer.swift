#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// FrameMel モデルの 5 損失値コンテナ
public struct FrameMelLosses: Sendable {
    public let totalLoss: Float
    public let decMelL1: Float
    public let postMelL1: Float
    public let voicedF0MSE: Float
    public let energyMSE: Float
    public let durMSE: Float

    public init(
        totalLoss: Float,
        decMelL1: Float,
        postMelL1: Float,
        voicedF0MSE: Float,
        energyMSE: Float,
        durMSE: Float
    ) {
        self.totalLoss = totalLoss
        self.decMelL1 = decMelL1
        self.postMelL1 = postMelL1
        self.voicedF0MSE = voicedF0MSE
        self.energyMSE = energyMSE
        self.durMSE = durMSE
    }
}

/// SpikeVoice に基づくフレーム単位対数メルモデルの MLX 学習トレーナー
///
/// なぜ 1D 時間畳み込み + 長さ調節器で学習するか:
/// 再帰型（CfC/RNN）の逐次状態遷移に起因する長大母音内でのメル停滞（低重心・高停止割合）を排除し、
/// 前後コンテキスト（kernel 17 / 5）を直接受容野とするフィードフォワード畳み込みにより
/// 高速かつ安定したメルスペクトルと自然な F0・エネルギー輪郭を同時に学習するため。
public final class MLXFrameMelTrainer: @unchecked Sendable {
    public let model: MLXFrameMelModel
    public let optimizer: Adam
    public private(set) var lastLosses: FrameMelLosses

    public init(
        model: MLXFrameMelModel = MLXFrameMelModel(),
        learningRate: Float = 0.001
    ) {
        self.model = model
        self.optimizer = Adam(learningRate: learningRate)
        self.lastLosses = FrameMelLosses(
            totalLoss: 0.0,
            decMelL1: 0.0,
            postMelL1: 0.0,
            voicedF0MSE: 0.0,
            energyMSE: 0.0,
            durMSE: 0.0
        )
    }

    public func setLearningRate(_ lr: Float) {
        optimizer.learningRate = lr
    }

    public func currentLearningRate() -> Float {
        return optimizer.learningRate
    }

    public func exportWeights() -> FrameMelWeights {
        return model.exportWeights()
    }

    /// 単一発話サンプルの勾配計算およびパラメータ更新（5つの損失を最適化）
    public func trainSample(
        phoneIds: [Int32],
        targetDurations: [Int],
        targetMel: [[Float]],
        targetF0: [Float],
        targetEnergy: [Float]
    ) -> FrameMelLosses {
        let phoneCount = phoneIds.count
        let totalFrames = targetMel.count
        if phoneCount < 3 || totalFrames <= 0 {
            return FrameMelLosses(
                totalLoss: 0.0,
                decMelL1: 0.0,
                postMelL1: 0.0,
                voicedF0MSE: 0.0,
                energyMSE: 0.0,
                durMSE: 0.0
            )
        }

        // トライフォン文脈（prev, cur, next）の作成
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

        // 長さ調節器用インデックスおよび音素内位置 (phonePos) の算出
        var gatherIndices = [Int32](repeating: 0, count: totalFrames)
        var pos = [Float](repeating: 0.0, count: totalFrames)
        var voicedFlags = [Float](repeating: 0.0, count: totalFrames)
        var curF = 0
        p = 0
        while p < phoneCount {
            let dur = targetDurations[p]
            let maxF = Float(max(1, dur - 1))
            let pid = Int(phoneIds[p])
            let isV = PhonemeVocabulary.isVoicedPhone(phoneId: pid)
            let vVal: Float
            switch isV {
            case true: vVal = 1.0
            case false: vVal = 0.0
            }

            var f = 0
            while f < dur {
                let fIdx = curF + f
                if fIdx < totalFrames {
                    gatherIndices[fIdx] = Int32(p)
                    pos[fIdx] = Float(f) / maxF
                    voicedFlags[fIdx] = vVal
                }
                f += 1
            }
            curF += dur
            p += 1
        }

        // 目標 Delta F0 の算出
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

        // 目標対数メルスペクトルのフラット化
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
        let targetDurArr = MLXArray(targetDurations.map { Float($0) }, [1, phoneCount])
        let gatherArr = MLXArray(gatherIndices)
        let posArr = MLXArray(pos, [totalFrames, 1])
        let voicedArr = MLXArray(voicedFlags, [1, totalFrames, 1])
        let targetF0Arr = MLXArray(targetF0, [totalFrames, 1])
        let targetF0Batch = MLXArray(targetF0, [1, totalFrames, 1])
        let deltaF0Arr = MLXArray(deltaF0, [totalFrames, 1])
        let targetEnergyArr = MLXArray(targetEnergy, [totalFrames, 1])
        let targetEnergyBatch = MLXArray(targetEnergy, [1, totalFrames, 1])
        let targetMelArr = MLXArray(targetMelFlat, [1, totalFrames, 64])

        let lg = valueAndGrad(model: model) { (m: MLXFrameMelModel, _) -> [MLXArray] in
            // 1. 音素エンコーダ
            let encStates = m.forwardEncoder(cur: curArr, prev: prevArr, next: nextArr)

            // 2. 継続時間予測
            let predDur = m.forwardDuration(encStates: encStates)
            let durLoss = mean(square(predDur - targetDurArr))

            // 3. 長さ調節器（音素隠れ状態のフレーム展開）
            let encSqueezed = encStates.squeezed(axis: 0)
            let hRep = take(encSqueezed, gatherArr, axis: 0)
            let frameStates = concatenated([hRep, posArr], axis: -1).expandedDimensions(axis: 0)

            // 4. F0 予測
            let predF0 = m.forwardF0(frameStates: frameStates, voicedMask: voicedArr)
            let f0Diff = (predF0 - targetF0Batch) * voicedArr
            let voicedSum = sum(voicedArr)
            let f0Loss = sum(square(f0Diff)) / maximum(voicedSum, MLXArray(1.0))

            // 5. エネルギー予測
            let predEnergy = m.forwardEnergy(frameStates: frameStates)
            let energyLoss = mean(square(predEnergy - targetEnergyBatch))

            // 6. メルデコーダ + PostNet（260次元条件付け）
            let condition = concatenated([hRep, targetF0Arr, deltaF0Arr, targetEnergyArr, posArr], axis: -1).expandedDimensions(axis: 0)
            let (decMel, postMel) = m.forwardDecoder(condition: condition)

            let decLoss = mean(abs(decMel - targetMelArr))
            let postLoss = mean(abs(postMel - targetMelArr))

            let totalLoss = decLoss + postLoss + f0Loss + energyLoss + durLoss
            return [totalLoss, decLoss, postLoss, f0Loss, energyLoss, durLoss]
        }

        let (lossVals, grads) = lg(model, [])
        eval(lossVals)
        let totalL = lossVals[0].item(Float.self)
        if totalL.isFinite != true {
            return self.lastLosses
        }
        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 1.0)
        optimizer.update(model: model, gradients: clippedGrads)
        eval(model, optimizer)

        let losses = FrameMelLosses(
            totalLoss: totalL,
            decMelL1: lossVals[1].item(Float.self),
            postMelL1: lossVals[2].item(Float.self),
            voicedF0MSE: lossVals[3].item(Float.self),
            energyMSE: lossVals[4].item(Float.self),
            durMSE: lossVals[5].item(Float.self)
        )
        self.lastLosses = losses
        return losses
    }
}
#endif
