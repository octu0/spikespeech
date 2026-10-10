#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// FrameMel モデルの損失値コンテナ（5つの基本損失 + PostNet後メルフレーム差損失）
public struct FrameMelLosses: Sendable {
    public let totalLoss: Float
    public let decMelL1: Float
    public let postMelL1: Float
    public let voicedF0MSE: Float
    public let energyMSE: Float
    public let durMSE: Float
    public let deltaMelL1: Float

    public init(
        totalLoss: Float,
        decMelL1: Float,
        postMelL1: Float,
        voicedF0MSE: Float,
        energyMSE: Float,
        durMSE: Float,
        deltaMelL1: Float = 0.0
    ) {
        self.totalLoss = totalLoss
        self.decMelL1 = decMelL1
        self.postMelL1 = postMelL1
        self.voicedF0MSE = voicedF0MSE
        self.energyMSE = energyMSE
        self.durMSE = durMSE
        self.deltaMelL1 = deltaMelL1
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
    public var deltaLossWeight: Float
    /// 全フレームの PostNet 後メルフレーム差 L1 に掛ける係数
    /// なぜ母音内部だけでなく全フレームのフレーム差を学習するか:
    /// メル L1 回帰は時間方向に平滑化しやすく、合成音では隣接フレームの包絡がほぼ同一になる割合が
    /// 教師音声の 2〜3 倍（0.55〜0.77 対 0.25）に達していた。音素境界を含む全フレームで
    /// 時間変化そのものを教師に合わせ、フォルマント遷移の鈍りを抑えるため。
    public var allFrameDeltaWeight: Float
    /// デコーダ条件付けに教師 F0/エネルギーではなく自モデルの予測値を用いる確率（scheduled sampling）
    /// なぜ予測値で条件付けする学習を混ぜるか:
    /// 学習時は教師の F0/エネルギー、推論時は予測器の出力（教師より滑らか）が入るため、
    /// 推論時にデコーダが見たことのない滑らかな条件を受けてメルがさらに平滑化していた。
    /// 予測値（勾配は遮断）でも教師メルを再現するよう学習させ、学習と推論の条件を近づける。
    public var conditionSamplingProbability: Float
    /// 大域分散（各メルチャネルの時間方向標準偏差）一致損失の係数
    public var globalVarianceWeight: Float
    private var rngState: UInt64
    public private(set) var lastLosses: FrameMelLosses

    public init(
        model: MLXFrameMelModel = MLXFrameMelModel(),
        learningRate: Float = 0.001,
        deltaLossWeight: Float = 1.0,
        allFrameDeltaWeight: Float = 1.0,
        conditionSamplingProbability: Float = 0.5,
        globalVarianceWeight: Float = 1.0,
        seed: UInt64 = 2026
    ) {
        self.model = model
        self.optimizer = Adam(learningRate: learningRate)
        self.deltaLossWeight = deltaLossWeight
        self.allFrameDeltaWeight = allFrameDeltaWeight
        self.conditionSamplingProbability = conditionSamplingProbability
        self.globalVarianceWeight = globalVarianceWeight
        self.rngState = seed | 1
        self.lastLosses = FrameMelLosses(
            totalLoss: 0.0,
            decMelL1: 0.0,
            postMelL1: 0.0,
            voicedF0MSE: 0.0,
            energyMSE: 0.0,
            durMSE: 0.0,
            deltaMelL1: 0.0
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
        return runSample(
            phoneIds: phoneIds,
            targetDurations: targetDurations,
            targetMel: targetMel,
            targetF0: targetF0,
            targetEnergy: targetEnergy,
            update: true
        )
    }

    /// 単一発話サンプルの損失のみを算出する（パラメータ更新なし・検証用）
    /// なぜ学習と同じ損失関数で検証するか:
    /// 保持検証データの損失でエポックを選定し、特定文の経験則指標に依存しない採否判定を行うため。
    public func evaluateSample(
        phoneIds: [Int32],
        targetDurations: [Int],
        targetMel: [[Float]],
        targetF0: [Float],
        targetEnergy: [Float]
    ) -> FrameMelLosses {
        return runSample(
            phoneIds: phoneIds,
            targetDurations: targetDurations,
            targetMel: targetMel,
            targetF0: targetF0,
            targetEnergy: targetEnergy,
            update: false
        )
    }

    /// 決定論的な xorshift64 乱数（0.0 ..< 1.0）
    private func nextUniform() -> Float {
        var x = rngState
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        rngState = x
        return Float(x >> 40) / Float(1 << 24)
    }

    private func runSample(
        phoneIds: [Int32],
        targetDurations: [Int],
        targetMel: [[Float]],
        targetF0: [Float],
        targetEnergy: [Float],
        update: Bool
    ) -> FrameMelLosses {
        let phoneCount = phoneIds.count
        let totalFrames = targetMel.count
        if phoneCount < 3 || totalFrames <= 1 {
            return FrameMelLosses(
                totalLoss: 0.0,
                decMelL1: 0.0,
                postMelL1: 0.0,
                voicedF0MSE: 0.0,
                energyMSE: 0.0,
                durMSE: 0.0,
                deltaMelL1: 0.0
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

        // 母音 (ID 5, 6, 7, 8, 9) の内側フレーム差マスク (totalFrames - 1)
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
                    let t = curFOffset + f
                    if t < totalFrames {
                        deltaMask[t - 1] = 1.0
                    }
                    f += 1
                }
            }
            curFOffset += dur
            p += 1
        }
        let deltaMaskArr = MLXArray(deltaMask, [1, totalFrames - 1, 1])

        let dWeight = self.deltaLossWeight
        let allDeltaWeight = self.allFrameDeltaWeight
        let gvWeight = self.globalVarianceWeight
        var useSampledCondition = false
        if update && 0.0 < conditionSamplingProbability {
            if nextUniform() < conditionSamplingProbability {
                useSampledCondition = true
            }
        }
        let lossFn: (MLXFrameMelModel) -> [MLXArray] = { (m: MLXFrameMelModel) -> [MLXArray] in
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
            // 教師 F0/エネルギー、または自モデルの予測値（勾配遮断）で条件付けする
            let condF0: MLXArray
            let condDeltaF0: MLXArray
            let condEnergy: MLXArray
            switch useSampledCondition {
            case true:
                let sampledF0 = stopGradient(predF0.squeezed(axis: 0)) // [T, 1]
                let sampledEnergy = stopGradient(predEnergy.squeezed(axis: 0)) // [T, 1]
                let prevF0 = concatenated([MLXArray.zeros([1, 1]), sampledF0[0..<(totalFrames - 1), 0...]], axis: 0)
                let bothVoiced = (sampledF0 .> MLXArray(Float(0.0))).asType(Float.self) * (prevF0 .> MLXArray(Float(0.0))).asType(Float.self)
                let rawDelta = clip((sampledF0 - prevF0) / MLXArray(Float(0.1)), min: -1.0, max: 1.0)
                condF0 = sampledF0
                condDeltaF0 = rawDelta * bothVoiced
                condEnergy = sampledEnergy
            case false:
                condF0 = targetF0Arr
                condDeltaF0 = deltaF0Arr
                condEnergy = targetEnergyArr
            }
            let condition = concatenated([hRep, condF0, condDeltaF0, condEnergy, posArr], axis: -1).expandedDimensions(axis: 0)
            let (decMel, postMel) = m.forwardDecoder(condition: condition)

            let decLoss = mean(abs(decMel - targetMelArr))
            let postLoss = mean(abs(postMel - targetMelArr))

            // 7. PostNet 後対数メルフレーム差損失 (母音内部のみ)
            let predDelta = postMel[0..., 1..<totalFrames, 0...] - postMel[0..., 0..<(totalFrames - 1), 0...]
            let targetDelta = targetMelArr[0..., 1..<totalFrames, 0...] - targetMelArr[0..., 0..<(totalFrames - 1), 0...]
            let deltaDiff = abs(predDelta - targetDelta)
            let maskedDelta = deltaDiff * deltaMaskArr
            let maskSum = sum(deltaMaskArr)
            let deltaLoss = sum(maskedDelta) / maximum(maskSum * 64.0, MLXArray(1.0))

            // 8. 全フレームの PostNet 後メルフレーム差 L1（音素境界を含む時間変化の一致）
            let allFrameDeltaLoss = mean(deltaDiff)

            // 9. 大域分散 (Global Variance) 損失: 各メルチャネルの時間方向標準偏差を教師に一致させる
            // なぜ分散を直接合わせるか:
            // L1 回帰は各フレームで中央値へ寄るため、発話全体で見るとチャネルごとの時間変動幅が縮み、
            // 音量の抑揚が平坦で高域の細部が消えた「こもった・平板な」メルになる。
            // 同一文の比較で合成音の RMS 変動幅は教師の約半分（10 dB 対 18 dB）だった。
            // 標準偏差の一致はこの圧縮を直接罰し、GAN なしで平滑化を緩和する古典的手法（Toda & Tokuda 2007）。
            let predStd = sqrt(variance(postMel, axis: 1) + MLXArray(Float(1e-4)))
            let targStd = sqrt(variance(targetMelArr, axis: 1) + MLXArray(Float(1e-4)))
            let gvLoss = mean(abs(predStd - targStd))

            // 10. チャネル別時間平均の一致（スペクトル傾斜の偏り抑制）
            // なぜ平均も合わせるか: 100 発話で予測メルは教師より低域が +0.7、高域が -0.7 偏り、
            // こもった低い声に聞こえていた。L1 は各フレームの中央値へ寄るため帯域ごとの平均が系統的にずれる。
            let channelMeanLoss = mean(abs(mean(postMel, axis: 1) - mean(targetMelArr, axis: 1)))

            // 11. 有声フレームの F0 標準偏差一致（抑揚幅の圧縮抑制）
            let voicedCount = maximum(sum(voicedArr), MLXArray(Float(1.0)))
            let predF0Mean = sum(predF0 * voicedArr) / voicedCount
            let targF0Mean = sum(targetF0Batch * voicedArr) / voicedCount
            let predF0Std = sqrt(sum(square((predF0 - predF0Mean) * voicedArr)) / voicedCount + MLXArray(Float(1e-6)))
            let targF0Std = sqrt(sum(square((targetF0Batch - targF0Mean) * voicedArr)) / voicedCount + MLXArray(Float(1e-6)))
            let f0StdLoss = abs(predF0Std - targF0Std)

            let totalLoss = decLoss + postLoss + f0Loss + energyLoss + durLoss + (deltaLoss * MLXArray(dWeight)) + (allFrameDeltaLoss * MLXArray(allDeltaWeight)) + ((gvLoss + channelMeanLoss) * MLXArray(gvWeight)) + (f0StdLoss * MLXArray(Float(1.0)))
            return [totalLoss, decLoss, postLoss, f0Loss, energyLoss, durLoss, allFrameDeltaLoss]
        }

        let lossVals: [MLXArray]
        switch update {
        case true:
            let lg = valueAndGrad(model: model) { (m: MLXFrameMelModel, _) -> [MLXArray] in
                return lossFn(m)
            }
            let (vals, grads) = lg(model, [])
            eval(vals)
            let totalL = vals[0].item(Float.self)
            if totalL.isFinite != true {
                return self.lastLosses
            }
            let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 1.0)
            optimizer.update(model: model, gradients: clippedGrads)
            eval(model, optimizer)
            lossVals = vals
        case false:
            let vals = lossFn(model)
            eval(vals)
            lossVals = vals
        }
        let totalL = lossVals[0].item(Float.self)
        if totalL.isFinite != true {
            return self.lastLosses
        }

        let losses = FrameMelLosses(
            totalLoss: totalL,
            decMelL1: lossVals[1].item(Float.self),
            postMelL1: lossVals[2].item(Float.self),
            voicedF0MSE: lossVals[3].item(Float.self),
            energyMSE: lossVals[4].item(Float.self),
            durMSE: lossVals[5].item(Float.self),
            deltaMelL1: lossVals[6].item(Float.self)
        )
        self.lastLosses = losses
        return losses
    }
}
#endif
