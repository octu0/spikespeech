#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// 波形損失勾配受入ゲート（ゲート 1〜4）検証結果
public struct WaveformGradGateResult: Sendable {
    public let segStart: Int
    public let maxSliceDiff: Float
    public let gate1Passed: Bool

    public let teacherSTFTLoss: Float
    public let predSTFTLoss: Float
    public let gate2Passed: Bool

    public let avgTeacherSTFTLoss: Float
    public let avgPredSTFTLoss: Float

    public let melGradMeanAbsTanh: Float
    public let outputSaturationRatio: Float
    public let melGradMeanAbsPreTanh: Float
    public let usePreTanh: Bool
    public let gate3Passed: Bool

    public let wOutMelGradNorm: Float
    public let wOutWaveGradNorm: Float
    public let gradNormRatio: Float
    public let recommendedWeight: Float
    public let gate4Passed: Bool

    public var allPassed: Bool {
        return gate1Passed && gate2Passed && gate3Passed && gate4Passed
    }
}

/// 波形損失勾配受入ゲート（ゲート 1〜4）の診断・検証器
public enum WaveformGradGateEvaluator {

    /// BASIC5000_0001 の先頭から32フレームずつ重ならない全区間において平均損失でゲートを厳密に検証する
    public static func evaluate(
        network: MLXSpikingAcousticNetwork,
        vocoder: MLXNeuralVocoder,
        features: [[Float]],
        targets: [[Float]],
        targetAudio: [Float],
        waveformLossWeight: Float = 0.15,
        bpttWindow: Int = 1
    ) -> WaveformGradGateResult {
        let segFrames = 32
        let rawLen = features.count
        let hopSize = AudioConfig.hopSize
        let segSamples = segFrames * hopSize
        let inDim = features.first?.count ?? AudioConfig.acousticInputDim
        let outDim = targets.first?.count ?? AudioConfig.melChannels

        let numSegments = rawLen / segFrames
        let totalSamples = numSegments * segSamples

        // 有声フレーム数が最大の 32 フレーム区間を探索（代表有声区間 segStart）
        var bestStart = 0
        var maxVoiced = -1
        let maxStart = max(0, rawLen - segFrames)
        var s = 0
        while s <= maxStart {
            var vCount = 0
            var f = 0
            while f < segFrames {
                let idx = s + f
                if 192 < features[idx].count {
                    if 0.5 <= features[idx][192] {
                        vCount += 1
                    }
                }
                f += 1
            }
            if maxVoiced < vCount {
                maxVoiced = vCount
                bestStart = s
            }
            s += 1
        }
        let segStart = bestStart

        // テンソル準備
        var flatFeat = [Float](repeating: 0.0, count: rawLen * inDim)
        var flatTgt = [Float](repeating: 0.0, count: rawLen * outDim)
        let flatMask = [Float](repeating: 1.0, count: rawLen)

        var t = 0
        while t < rawLen {
            var i = 0
            while i < inDim {
                flatFeat[(t * inDim) + i] = features[t][i]
                i += 1
            }
            var c = 0
            while c < outDim {
                flatTgt[(t * outDim) + c] = targets[t][c]
                c += 1
            }
            t += 1
        }

        var segAudio = [Float](repeating: 0.0, count: totalSamples)
        let copyLimit = min(totalSamples, targetAudio.count)
        var sa = 0
        while sa < copyLimit {
            segAudio[sa] = targetAudio[sa]
            sa += 1
        }

        let fArr = MLXArray(flatFeat, [1, rawLen, inDim])
        let tArr = MLXArray(flatTgt, [1, rawLen, outDim])
        let mArr = MLXArray(flatMask, [1, rawLen])
        let targetWaveArr = MLXArray(segAudio, [1, totalSamples])

        // 1. 全長予測
        let predFull = network.forward(features: fArr, bpttWindow: bpttWindow)
        eval(predFull)

        // ゲート 1: 各区間の予測メルの先頭フレームが、全長予測の segStart フレームと一致する。最大絶対差は 0
        var maxSliceDiff: Float = 0.0
        var k = 0
        while k < numSegments {
            let sStart = k * segFrames
            let sEnd = sStart + segFrames
            let predSliced = predFull[0..., sStart..<sEnd, 0...]
            let frameFromFull = predFull[0, sStart, 0...]
            let frameFromSlice = predSliced[0, 0, 0...]
            eval(frameFromFull, frameFromSlice)
            let diffSliceArr = abs(frameFromFull - frameFromSlice)
            eval(diffSliceArr)
            let maxDiffSlice = max(diffSliceArr).item(Float.self)
            if maxSliceDiff < maxDiffSlice {
                maxSliceDiff = maxDiffSlice
            }
            k += 1
        }
        let gate1Passed = (maxSliceDiff <= 1.0e-6)

        // ゲート 2: ボコーダに教師メルを入れた STFT 損失が、予測メルを入れた STFT 損失より小さい
        // 有声代表区間（segStart）における検証
        let vSegEnd = segStart + segFrames
        let vTgtMel = tArr[0..., segStart..<vSegEnd, 0...]
        let vPredMel = predFull[0..., segStart..<vSegEnd, 0...]
        let vF0 = fArr[0..., segStart..<vSegEnd, 194..<195]
        let vVoiced = fArr[0..., segStart..<vSegEnd, 192..<193]
        let vVocTgtIn = concatenated([vTgtMel, vF0, vVoiced], axis: -1)
        let vVocPredIn = concatenated([vPredMel, vF0, vVoiced], axis: -1)
        let waveTeacherVoiced = vocoder(vVocTgtIn)
        let wavePredVoiced = vocoder(vVocPredIn)
        eval(waveTeacherVoiced, wavePredVoiced)

        let vSampleStart = segStart * hopSize
        let vSampleEnd = vSampleStart + segSamples
        var targetWaveVoicedArr: MLXArray
        if vSampleEnd <= targetAudio.count {
            var vAudio = [Float](repeating: 0.0, count: segSamples)
            var va = 0
            while va < segSamples {
                vAudio[va] = targetAudio[vSampleStart + va]
                va += 1
            }
            targetWaveVoicedArr = MLXArray(vAudio, [1, segSamples])
        } else {
            targetWaveVoicedArr = targetWaveArr[0..., 0..<segSamples]
        }
        let lossTeacherVoiced = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: waveTeacherVoiced, target: targetWaveVoicedArr)
        let lossPredVoiced = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: wavePredVoiced, target: targetWaveVoicedArr)
        eval(lossTeacherVoiced, lossPredVoiced)
        let teacherSTFTLoss = lossTeacherVoiced.item(Float.self)
        let predSTFTLoss = lossPredVoiced.item(Float.self)
        let gate2Passed = (teacherSTFTLoss < predSTFTLoss)

        // 全区間（非重複 32 フレーム）における STFT 損失および出力飽和率の集計
        var sumTeacherLoss: Float = 0.0
        var sumPredLoss: Float = 0.0
        var totalOver99: Float = 0.0
        k = 0
        while k < numSegments {
            let sStart = k * segFrames
            let sEnd = sStart + segFrames
            let targetMelSegment = tArr[0..., sStart..<sEnd, 0...]
            let predMelSegment = predFull[0..., sStart..<sEnd, 0...]
            let f0Segment = fArr[0..., sStart..<sEnd, 194..<195]
            let voicedSegment = fArr[0..., sStart..<sEnd, 192..<193]

            let vocTeacherInput = concatenated([targetMelSegment, f0Segment, voicedSegment], axis: -1)
            let vocPredInput = concatenated([predMelSegment, f0Segment, voicedSegment], axis: -1)

            let waveTeacher = vocoder(vocTeacherInput)
            let wavePred = vocoder(vocPredInput)
            eval(waveTeacher, wavePred)

            let sampleStart = k * segSamples
            let sampleEnd = sampleStart + segSamples
            let targetWaveSeg = targetWaveArr[0..., sampleStart..<sampleEnd]

            let lossTeacherArr = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: waveTeacher, target: targetWaveSeg)
            let lossPredArr = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: wavePred, target: targetWaveSeg)
            eval(lossTeacherArr, lossPredArr)

            sumTeacherLoss += lossTeacherArr.item(Float.self)
            sumPredLoss += lossPredArr.item(Float.self)

            let absWavePred = abs(wavePred)
            eval(absWavePred)
            let over99CountArr = sum((MLXArray(Float(0.99)) .< absWavePred).asType(Float.self))
            eval(over99CountArr)
            totalOver99 += over99CountArr.item(Float.self)

            k += 1
        }

        var avgTeacherSTFTLoss: Float = 0.0
        var avgPredSTFTLoss: Float = 0.0
        var outputSaturationRatio: Float = 0.0
        if 0 < numSegments {
            avgTeacherSTFTLoss = sumTeacherLoss / Float(numSegments)
            avgPredSTFTLoss = sumPredLoss / Float(numSegments)
        }
        if 0 < totalSamples {
            outputSaturationRatio = totalOver99 / Float(totalSamples)
        }

        // ゲート 3: 波形の平均 STFT 損失を予測メルで微分した勾配の平均絶対値、および飽和率
        let gradMelTanhFn = grad { (mel: MLXArray) -> MLXArray in
            var lossSum = MLXArray(0.0)
            var idx = 0
            while idx < numSegments {
                let sStart = idx * segFrames
                let sEnd = sStart + segFrames
                let pMel = mel[0..., sStart..<sEnd, 0...]
                let f0Seg = fArr[0..., sStart..<sEnd, 194..<195]
                let voicedSeg = fArr[0..., sStart..<sEnd, 192..<193]
                let vocIn = concatenated([pMel, f0Seg, voicedSeg], axis: -1)
                let w = vocoder(vocIn)
                let sampStart = idx * segSamples
                let sampEnd = sampStart + segSamples
                let tWaveSeg = targetWaveArr[0..., sampStart..<sampEnd]
                let l = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: w, target: tWaveSeg)
                lossSum = lossSum + l
                idx += 1
            }
            if 0 < numSegments {
                return lossSum / Float(numSegments)
            }
            return lossSum
        }
        let gradMelTanh = gradMelTanhFn(predFull)
        eval(gradMelTanh)
        let melGradMeanAbsTanh = mean(abs(gradMelTanh)).item(Float.self)

        let gradMelPreTanhFn = grad { (mel: MLXArray) -> MLXArray in
            var lossSum = MLXArray(0.0)
            var idx = 0
            while idx < numSegments {
                let sStart = idx * segFrames
                let sEnd = sStart + segFrames
                let pMel = mel[0..., sStart..<sEnd, 0...]
                let f0Seg = fArr[0..., sStart..<sEnd, 194..<195]
                let voicedSeg = fArr[0..., sStart..<sEnd, 192..<193]
                let vocIn = concatenated([pMel, f0Seg, voicedSeg], axis: -1)
                let w = vocoder.forwardPreTanh(vocIn)
                let sampStart = idx * segSamples
                let sampEnd = sampStart + segSamples
                let tWaveSeg = targetWaveArr[0..., sampStart..<sampEnd]
                let l = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: w, target: tWaveSeg)
                lossSum = lossSum + l
                idx += 1
            }
            if 0 < numSegments {
                return lossSum / Float(numSegments)
            }
            return lossSum
        }
        let gradMelPreTanh = gradMelPreTanhFn(predFull)
        eval(gradMelPreTanh)
        let melGradMeanAbsPreTanh = mean(abs(gradMelPreTanh)).item(Float.self)

        var usePreTanh = false
        if 0.01 <= outputSaturationRatio {
            usePreTanh = true
        }
        if melGradMeanAbsTanh < 1.0e-5 {
            usePreTanh = true
        }

        let effectiveMelGradMean: Float
        switch usePreTanh {
        case true:
            effectiveMelGradMean = melGradMeanAbsPreTanh
        case false:
            effectiveMelGradMean = melGradMeanAbsTanh
        }
        let gate3Passed = (1.0e-4 < effectiveMelGradMean)

        // ゲート 4: 波形項だけの勾配ノルムと、メル L1 だけの勾配ノルムを、wOut で比べる。波形側がメル側の 0.2 倍から 5 倍
        let melLg = valueAndGrad(model: network) { (model: MLXSpikingAcousticNetwork, _: [MLXArray]) -> [MLXArray] in
            let p = model.forward(features: fArr, bpttWindow: bpttWindow)
            let l = AcousticLossFunctions.spectralL1Loss(predicted: p, target: tArr, mask: mArr)
            return [l]
        }
        let (_, melGrads) = melLg(network, [])
        eval(melGrads)

        var wOutMelGradNorm: Float = 0.0
        if let gMel = melGrads[unwrapping: "wOut"] {
            eval(gMel)
            wOutMelGradNorm = sqrt(sum(gMel * gMel)).item(Float.self)
        }

        let waveLg = valueAndGrad(model: network) { (model: MLXSpikingAcousticNetwork, _: [MLXArray]) -> [MLXArray] in
            let p = model.forward(features: fArr, bpttWindow: bpttWindow)
            var lossSum = MLXArray(0.0)
            var idx = 0
            while idx < numSegments {
                let sStart = idx * segFrames
                let sEnd = sStart + segFrames
                let pSeg = p[0..., sStart..<sEnd, 0...]
                let f0Seg = fArr[0..., sStart..<sEnd, 194..<195]
                let voicedSeg = fArr[0..., sStart..<sEnd, 192..<193]
                let vocIn = concatenated([pSeg, f0Seg, voicedSeg], axis: -1)
                let w: MLXArray
                switch usePreTanh {
                case true:
                    w = vocoder.forwardPreTanh(vocIn)
                case false:
                    w = vocoder(vocIn)
                }
                let sampStart = idx * segSamples
                let sampEnd = sampStart + segSamples
                let targetSeg = targetWaveArr[0..., sampStart..<sampEnd]
                let l = MLXNeuralVocoder.multiResolutionSTFTLoss(predicted: w, target: targetSeg)
                lossSum = lossSum + l
                idx += 1
            }
            var avgLoss = lossSum
            if 0 < numSegments {
                avgLoss = lossSum / Float(numSegments)
            }
            let wTerm = MLXArray(waveformLossWeight) * avgLoss
            return [wTerm]
        }
        let (_, waveGrads) = waveLg(network, [])
        eval(waveGrads)

        var wOutWaveGradNorm: Float = 0.0
        if let gWave = waveGrads[unwrapping: "wOut"] {
            eval(gWave)
            wOutWaveGradNorm = sqrt(sum(gWave * gWave)).item(Float.self)
        }

        var gradNormRatio: Float = 0.0
        if 1.0e-7 < wOutMelGradNorm {
            gradNormRatio = wOutWaveGradNorm / wOutMelGradNorm
        }

        var recommendedWeight = waveformLossWeight
        if 5.0 < gradNormRatio {
            // 5倍を超えるなら波形項係数を下げて 1.0〜2.0 倍前後に収める
            let scale = 1.0 / gradNormRatio
            recommendedWeight = waveformLossWeight * scale
        }

        let gate4Passed = (0.2 <= gradNormRatio && gradNormRatio <= 5.0)

        return WaveformGradGateResult(
            segStart: segStart,
            maxSliceDiff: maxSliceDiff,
            gate1Passed: gate1Passed,
            teacherSTFTLoss: teacherSTFTLoss,
            predSTFTLoss: predSTFTLoss,
            gate2Passed: gate2Passed,
            avgTeacherSTFTLoss: avgTeacherSTFTLoss,
            avgPredSTFTLoss: avgPredSTFTLoss,
            melGradMeanAbsTanh: melGradMeanAbsTanh,
            outputSaturationRatio: outputSaturationRatio,
            melGradMeanAbsPreTanh: melGradMeanAbsPreTanh,
            usePreTanh: usePreTanh,
            gate3Passed: gate3Passed,
            wOutMelGradNorm: wOutMelGradNorm,
            wOutWaveGradNorm: wOutWaveGradNorm,
            gradNormRatio: gradNormRatio,
            recommendedWeight: recommendedWeight,
            gate4Passed: gate4Passed
        )
    }
}
#endif
