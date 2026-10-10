import Foundation

/// SpikeVoice (arXiv:2408.00788) に基づくフレーム単位対数メル推論モデル（Pure Swift 実装）
///
/// 音素エンコーダ、分散アダプタ（継続時間・F0・エネルギー予測）、
/// 長さ調節、4層時間畳み込みメルデコーダ、および 5層 PostNet 残差予測器により、
/// 母音内部で動的に変化する対数メルスペクトルと基本周波数を生成する。
public final class FrameMelModel: @unchecked Sendable {
    public let weights: FrameMelWeights

    public init(weights: FrameMelWeights) {
        self.weights = weights
    }

    public enum Activation {
        case none
        case leakyRelu
        case tanh
        case sigmoid
        case softplusPlusOne
    }

    /// 高速 1D 畳み込み演算（境界領域と内部領域を分離して分岐を排除）
    @inline(__always)
    public static func conv1d(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        T: Int,
        inC: Int,
        outC: Int,
        kernel: Int,
        padding: Int,
        weights: UnsafePointer<Float>,
        bias: UnsafePointer<Float>,
        activation: Activation
    ) {
        let span = kernel - 1
        let innerStart = min(T, max(0, padding))
        let innerEnd = max(innerStart, min(T, max(0, T + padding - span)))

        // 1. 左境界（0 <= t < innerStart）: 境界検査付き
        var t = 0
        while t < innerStart {
            let outRow = t * outC
            var c = 0
            while c < outC {
                var sum = bias[c]
                let wRow = c * kernel * inC
                var k = 0
                while k < kernel {
                    let inT = t + k - padding
                    if 0 <= inT && inT < T {
                        let inRow = inT * inC
                        let wOffset = wRow + (k * inC)
                        sum += VectorOperations.dotProduct(
                            a: input.advanced(by: inRow),
                            b: weights.advanced(by: wOffset),
                            count: inC
                        )
                    }
                    k += 1
                }
                switch activation {
                case .none:
                    break
                case .leakyRelu:
                    if sum < 0.0 {
                        sum = sum * 0.1
                    }
                case .tanh:
                    sum = tanhf(sum)
                case .sigmoid:
                    sum = 1.0 / (1.0 + expf(-sum))
                case .softplusPlusOne:
                    if sum < 20.0 {
                        sum = log1pf(expf(sum)) + 1.0
                    } else {
                        sum = sum + 1.0
                    }
                }
                output[outRow + c] = sum
                c += 1
            }
            t += 1
        }

        // 2. 内部領域（innerStart <= t < innerEnd）: 境界検査なし
        while t < innerEnd {
            let outRow = t * outC
            var c = 0
            while c < outC {
                var sum = bias[c]
                let wRow = c * kernel * inC
                var k = 0
                while k < kernel {
                    let inT = t + k - padding
                    let inRow = inT * inC
                    let wOffset = wRow + (k * inC)
                    sum += VectorOperations.dotProduct(
                        a: input.advanced(by: inRow),
                        b: weights.advanced(by: wOffset),
                        count: inC
                    )
                    k += 1
                }
                switch activation {
                case .none:
                    break
                case .leakyRelu:
                    if sum < 0.0 {
                        sum = sum * 0.1
                    }
                case .tanh:
                    sum = tanhf(sum)
                case .sigmoid:
                    sum = 1.0 / (1.0 + expf(-sum))
                case .softplusPlusOne:
                    if sum < 20.0 {
                        sum = log1pf(expf(sum)) + 1.0
                    } else {
                        sum = sum + 1.0
                    }
                }
                output[outRow + c] = sum
                c += 1
            }
            t += 1
        }

        // 3. 右境界（innerEnd <= t < T）: 境界検査付き
        while t < T {
            let outRow = t * outC
            var c = 0
            while c < outC {
                var sum = bias[c]
                let wRow = c * kernel * inC
                var k = 0
                while k < kernel {
                    let inT = t + k - padding
                    if 0 <= inT && inT < T {
                        let inRow = inT * inC
                        let wOffset = wRow + (k * inC)
                        sum += VectorOperations.dotProduct(
                            a: input.advanced(by: inRow),
                            b: weights.advanced(by: wOffset),
                            count: inC
                        )
                    }
                    k += 1
                }
                switch activation {
                case .none:
                    break
                case .leakyRelu:
                    if sum < 0.0 {
                        sum = sum * 0.1
                    }
                case .tanh:
                    sum = tanhf(sum)
                case .sigmoid:
                    sum = 1.0 / (1.0 + expf(-sum))
                case .softplusPlusOne:
                    if sum < 20.0 {
                        sum = log1pf(expf(sum)) + 1.0
                    } else {
                        sum = sum + 1.0
                    }
                }
                output[outRow + c] = sum
                c += 1
            }
            t += 1
        }
    }

    /// 音素 ID 列だけから決まる韻律位置特徴を算出する（音素ごとに prosodicFeatureCount 次元）
    ///
    /// 学習時と推論時で同じ入力（音素 ID 列）から同じ値が得られるため、追加のアノテーションを要しない。
    /// - 0: 発話内の音素位置 (0..1)
    /// - 1: 発話内のモーラ位置 (0..1)
    /// - 2: ポーズ区切り句内のモーラ位置 (0..1)
    /// - 3: 句のモーラ数 / 20 (0..1 にクリップ)
    /// - 4: 句の通し番号 / 句数 (0..1)
    /// - 5: モーラ高低 (1=高) ※ accent が与えられた場合
    /// - 6: アクセント句内のモーラ位置 (0..1) ※ accent が与えられた場合
    /// - 7: アクセント核 (1=核) ※ accent が与えられた場合
    public static func prosodicFeatures(phoneIds: [Int32], accent: [[Float]] = []) -> [[Float]] {
        let featCount = FrameMelWeights.prosodicFeatureCount
        let n = phoneIds.count
        if n <= 0 {
            return []
        }
        func isBoundary(_ pid: Int) -> Bool {
            return pid == PhonemeVocabulary.silId || pid == PhonemeVocabulary.pauId
        }
        // 句分割（境界音素は直前の句に属させず、単独で前句終端として扱う）
        var segmentIndex = [Int](repeating: 0, count: n)
        var segmentCount = 0
        var inSegment = false
        var i = 0
        while i < n {
            let pid = Int(phoneIds[i])
            if isBoundary(pid) {
                if inSegment {
                    segmentCount += 1
                    inSegment = false
                }
                segmentIndex[i] = max(0, segmentCount - 1)
            } else {
                if inSegment != true {
                    inSegment = true
                }
                segmentIndex[i] = segmentCount
            }
            i += 1
        }
        if inSegment {
            segmentCount += 1
        }
        let totalSegments = max(1, segmentCount)
        // モーラ数（発話全体・句ごと）
        var totalMoras = 0
        var segMoraCount = [Int](repeating: 0, count: totalSegments)
        i = 0
        while i < n {
            let pid = Int(phoneIds[i])
            if PhonemeVocabulary.isVowelOrSpecialMora(phoneId: pid) {
                totalMoras += 1
                let sIdx = min(totalSegments - 1, segmentIndex[i])
                segMoraCount[sIdx] += 1
            }
            i += 1
        }
        var feats = [[Float]](repeating: [Float](repeating: 0.0, count: featCount), count: n)
        var moraSoFar = 0
        var segMoraSoFar = 0
        var currentSeg = -1
        i = 0
        while i < n {
            let pid = Int(phoneIds[i])
            let sIdx = min(totalSegments - 1, segmentIndex[i])
            if sIdx != currentSeg {
                currentSeg = sIdx
                segMoraSoFar = 0
            }
            let segMoras = max(1, segMoraCount[sIdx])
            feats[i][0] = Float(i) / Float(max(1, n - 1))
            feats[i][1] = Float(moraSoFar) / Float(max(1, totalMoras))
            feats[i][2] = Float(segMoraSoFar) / Float(segMoras)
            feats[i][3] = min(1.0, Float(segMoras) / 20.0)
            feats[i][4] = Float(sIdx) / Float(max(1, totalSegments - 1 == 0 ? 1 : totalSegments - 1))
            if accent.count == n && 3 <= accent[i].count {
                feats[i][5] = accent[i][0]
                feats[i][6] = accent[i][1]
                feats[i][7] = accent[i][2]
            }
            if PhonemeVocabulary.isVowelOrSpecialMora(phoneId: pid) {
                moraSoFar += 1
                segMoraSoFar += 1
            }
            i += 1
        }
        return feats
    }

    /// 音素系列からエンコーダ特徴量を生成する（音素単位、4層時間畳み込み）
    public func encodePhonemes(phoneIds: [Int32], accent: [[Float]] = []) -> [[Float]] {
        let phoneCount = phoneIds.count
        if phoneCount <= 0 {
            return []
        }
        let hiddenDim = 256
        let vocabSize = 64
        let featCount = FrameMelWeights.prosodicFeatureCount
        let feats = Self.prosodicFeatures(phoneIds: phoneIds, accent: accent)
        var featW: [Float]? = nil
        if let w = weights.encWFeat, w.count == hiddenDim * featCount {
            featW = w
        }

        // 埋め込みベクトルの加算
        var encStates = [Float](repeating: 0.0, count: phoneCount * hiddenDim)
        var p = 0
        while p < phoneCount {
            var curPid = Int(phoneIds[p])
            if curPid < 0 || vocabSize <= curPid {
                curPid = PhonemeVocabulary.unkId
            }
            var prevPid = PhonemeVocabulary.silId
            if 0 < p {
                prevPid = Int(phoneIds[p - 1])
                if prevPid < 0 || vocabSize <= prevPid {
                    prevPid = PhonemeVocabulary.unkId
                }
            }
            var nextPid = PhonemeVocabulary.silId
            if (p + 1) < phoneCount {
                nextPid = Int(phoneIds[p + 1])
                if nextPid < 0 || vocabSize <= nextPid {
                    nextPid = PhonemeVocabulary.unkId
                }
            }

            let pRow = p * hiddenDim
            let curRow = curPid * hiddenDim
            let prevRow = prevPid * hiddenDim
            let nextRow = nextPid * hiddenDim

            var h = 0
            while h < hiddenDim {
                var v = weights.embedCur[curRow + h] + weights.embedPrev[prevRow + h] + weights.embedNext[nextRow + h] + weights.encBIn[h]
                if let fw = featW {
                    let wRow = h * featCount
                    var k = 0
                    while k < featCount {
                        v += fw[wRow + k] * feats[p][k]
                        k += 1
                    }
                }
                encStates[pRow + h] = v
                h += 1
            }
            p += 1
        }

        // 4層の時間畳み込み（kernel 3, padding 1, residual + leakyRelu）
        var layerBuf = [Float](repeating: 0.0, count: phoneCount * hiddenDim)
        var l = 0
        while l < 4 {
            let wConv = weights.encWConv[l]
            let bConv = weights.encBConv[l]
            encStates.withUnsafeBufferPointer { pIn in
                layerBuf.withUnsafeMutableBufferPointer { pOut in
                    wConv.withUnsafeBufferPointer { pW in
                        bConv.withUnsafeBufferPointer { pB in
                            Self.conv1d(
                                input: pIn.baseAddress!,
                                output: pOut.baseAddress!,
                                T: phoneCount,
                                inC: hiddenDim,
                                outC: hiddenDim,
                                kernel: 3,
                                padding: 1,
                                weights: pW.baseAddress!,
                                bias: pB.baseAddress!,
                                activation: .leakyRelu
                            )
                        }
                    }
                }
            }

            // 残差加算
            var i = 0
            while i < encStates.count {
                encStates[i] = encStates[i] + layerBuf[i]
                i += 1
            }
            l += 1
        }

        var result = [[Float]](repeating: [Float](repeating: 0.0, count: hiddenDim), count: phoneCount)
        p = 0
        while p < phoneCount {
            let pRow = p * hiddenDim
            var h = 0
            while h < hiddenDim {
                result[p][h] = encStates[pRow + h]
                h += 1
            }
            p += 1
        }
        return result
    }

    /// 音素単位特徴量から音素継続時間を予測する
    public func predictDurations(encStates: [[Float]]) -> [Float] {
        let phoneCount = encStates.count
        if phoneCount <= 0 {
            return []
        }
        let hiddenDim = 256
        var flatIn = [Float](repeating: 0.0, count: phoneCount * hiddenDim)
        var p = 0
        while p < phoneCount {
            var h = 0
            while h < hiddenDim {
                flatIn[(p * hiddenDim) + h] = encStates[p][h]
                h += 1
            }
            p += 1
        }

        var h1Buf = [Float](repeating: 0.0, count: phoneCount * 128)
        flatIn.withUnsafeBufferPointer { pIn in
            h1Buf.withUnsafeMutableBufferPointer { pOut in
                weights.durW1.withUnsafeBufferPointer { pW in
                    weights.durB1.withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: phoneCount,
                            inC: hiddenDim,
                            outC: 128,
                            kernel: 3,
                            padding: 1,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .leakyRelu
                        )
                    }
                }
            }
        }

        var outBuf = [Float](repeating: 0.0, count: phoneCount)
        h1Buf.withUnsafeBufferPointer { pIn in
            outBuf.withUnsafeMutableBufferPointer { pOut in
                weights.durW2.withUnsafeBufferPointer { pW in
                    weights.durB2.withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: phoneCount,
                            inC: 128,
                            outC: 1,
                            kernel: 1,
                            padding: 0,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .softplusPlusOne
                        )
                    }
                }
            }
        }
        return outBuf
    }

    /// フレーム特徴量から F0（正規化値 0..1）およびエネルギー（0..1）を予測する
    public func predictF0AndEnergy(
        frameStates: [Float],
        totalFrames: Int,
        voicedFlags: [Float]
    ) -> (f0: [Float], energy: [Float]) {
        if totalFrames <= 0 {
            return ([], [])
        }
        let inDim = 257

        // F0 予測器
        var f0H1 = [Float](repeating: 0.0, count: totalFrames * 128)
        frameStates.withUnsafeBufferPointer { pIn in
            f0H1.withUnsafeMutableBufferPointer { pOut in
                weights.f0W1.withUnsafeBufferPointer { pW in
                    weights.f0B1.withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: totalFrames,
                            inC: inDim,
                            outC: 128,
                            kernel: 5,
                            padding: 2,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .leakyRelu
                        )
                    }
                }
            }
        }

        var f0Out = [Float](repeating: 0.0, count: totalFrames)
        f0H1.withUnsafeBufferPointer { pIn in
            f0Out.withUnsafeMutableBufferPointer { pOut in
                weights.f0W2.withUnsafeBufferPointer { pW in
                    weights.f0B2.withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: totalFrames,
                            inC: 128,
                            outC: 1,
                            kernel: 1,
                            padding: 0,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .sigmoid
                        )
                    }
                }
            }
        }

        // 有声マスクの適用（無声フレームは厳密に 0）
        var t = 0
        while t < totalFrames {
            var vMask: Float = 0.0
            if t < voicedFlags.count {
                vMask = voicedFlags[t]
            }
            if vMask < 0.5 {
                f0Out[t] = 0.0
            }
            t += 1
        }

        // エネルギー予測器
        var energyH1 = [Float](repeating: 0.0, count: totalFrames * 128)
        frameStates.withUnsafeBufferPointer { pIn in
            energyH1.withUnsafeMutableBufferPointer { pOut in
                weights.energyW1.withUnsafeBufferPointer { pW in
                    weights.energyB1.withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: totalFrames,
                            inC: inDim,
                            outC: 128,
                            kernel: 5,
                            padding: 2,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .leakyRelu
                        )
                    }
                }
            }
        }

        var energyOut = [Float](repeating: 0.0, count: totalFrames)
        energyH1.withUnsafeBufferPointer { pIn in
            energyOut.withUnsafeMutableBufferPointer { pOut in
                weights.energyW2.withUnsafeBufferPointer { pW in
                    weights.energyB2.withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: totalFrames,
                            inC: 128,
                            outC: 1,
                            kernel: 1,
                            padding: 0,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .sigmoid
                        )
                    }
                }
            }
        }

        return (f0: f0Out, energy: energyOut)
    }

    /// メルデコーダおよび PostNet 残差処理を実行する
    public func decodeMel(
        decoderCondition: [Float],
        totalFrames: Int
    ) -> (decMel: [[Float]], postMel: [[Float]]) {
        if totalFrames <= 0 {
            return ([], [])
        }
        let inDim = 260
        let hiddenDim = 256
        let melDim = 64

        // 入力射影: 260 -> 256
        var hStates = [Float](repeating: 0.0, count: totalFrames * hiddenDim)
        decoderCondition.withUnsafeBufferPointer { pIn in
            hStates.withUnsafeMutableBufferPointer { pOut in
                weights.decWIn.withUnsafeBufferPointer { pW in
                    weights.decBIn.withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: totalFrames,
                            inC: inDim,
                            outC: hiddenDim,
                            kernel: 1,
                            padding: 0,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .leakyRelu
                        )
                    }
                }
            }
        }

        // 4層の時間畳み込み (kernel 17, padding 8 または kernel 5, padding 2)
        var layerBuf = [Float](repeating: 0.0, count: totalFrames * hiddenDim)
        var l = 0
        while l < 4 {
            let wConv = weights.decWConv[l]
            let bConv = weights.decBConv[l]
            let decKernel = wConv.count / (hiddenDim * hiddenDim)
            let decPad = decKernel / 2
            hStates.withUnsafeBufferPointer { pIn in
                layerBuf.withUnsafeMutableBufferPointer { pOut in
                    wConv.withUnsafeBufferPointer { pW in
                        bConv.withUnsafeBufferPointer { pB in
                            Self.conv1d(
                                input: pIn.baseAddress!,
                                output: pOut.baseAddress!,
                                T: totalFrames,
                                inC: hiddenDim,
                                outC: hiddenDim,
                                kernel: decKernel,
                                padding: decPad,
                                weights: pW.baseAddress!,
                                bias: pB.baseAddress!,
                                activation: .leakyRelu
                            )
                        }
                    }
                }
            }

            // 残差加算
            var i = 0
            while i < hStates.count {
                hStates[i] = hStates[i] + layerBuf[i]
                i += 1
            }
            l += 1
        }

        // 出力射影: 256 -> 64
        var flatDecMel = [Float](repeating: 0.0, count: totalFrames * melDim)
        hStates.withUnsafeBufferPointer { pIn in
            flatDecMel.withUnsafeMutableBufferPointer { pOut in
                weights.decWOut.withUnsafeBufferPointer { pW in
                    weights.decBOut.withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: totalFrames,
                            inC: hiddenDim,
                            outC: melDim,
                            kernel: 1,
                            padding: 0,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .none
                        )
                    }
                }
            }
        }

        // PostNet: 5層の kernel 5 時間畳み込み
        let postIn = flatDecMel
        var postHidden = [Float](repeating: 0.0, count: totalFrames * hiddenDim)

        // 層 0: 64 -> 256
        postIn.withUnsafeBufferPointer { pIn in
            postHidden.withUnsafeMutableBufferPointer { pOut in
                weights.postWConv[0].withUnsafeBufferPointer { pW in
                    weights.postBConv[0].withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: totalFrames,
                            inC: melDim,
                            outC: hiddenDim,
                            kernel: 5,
                            padding: 2,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .tanh
                        )
                    }
                }
            }
        }

        // 層 1..3: 256 -> 256
        var postNextHidden = [Float](repeating: 0.0, count: totalFrames * hiddenDim)
        l = 1
        while l < 4 {
            postHidden.withUnsafeBufferPointer { pIn in
                postNextHidden.withUnsafeMutableBufferPointer { pOut in
                    weights.postWConv[l].withUnsafeBufferPointer { pW in
                        weights.postBConv[l].withUnsafeBufferPointer { pB in
                            Self.conv1d(
                                input: pIn.baseAddress!,
                                output: pOut.baseAddress!,
                                T: totalFrames,
                                inC: hiddenDim,
                                outC: hiddenDim,
                                kernel: 5,
                                padding: 2,
                                weights: pW.baseAddress!,
                                bias: pB.baseAddress!,
                                activation: .tanh
                            )
                        }
                    }
                }
            }
            postHidden = postNextHidden
            l += 1
        }

        // 層 4: 256 -> 64
        var flatResidual = [Float](repeating: 0.0, count: totalFrames * melDim)
        postHidden.withUnsafeBufferPointer { pIn in
            flatResidual.withUnsafeMutableBufferPointer { pOut in
                weights.postWConv[4].withUnsafeBufferPointer { pW in
                    weights.postBConv[4].withUnsafeBufferPointer { pB in
                        Self.conv1d(
                            input: pIn.baseAddress!,
                            output: pOut.baseAddress!,
                            T: totalFrames,
                            inC: hiddenDim,
                            outC: melDim,
                            kernel: 5,
                            padding: 2,
                            weights: pW.baseAddress!,
                            bias: pB.baseAddress!,
                            activation: .none
                        )
                    }
                }
            }
        }

        var decMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melDim), count: totalFrames)
        var postMelSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melDim), count: totalFrames)
        var t = 0
        while t < totalFrames {
            let row = t * melDim
            var c = 0
            while c < melDim {
                let dVal = flatDecMel[row + c]
                let rVal = flatResidual[row + c]
                decMelSeq[t][c] = dVal
                postMelSeq[t][c] = dVal + rVal
                c += 1
            }
            t += 1
        }

        return (decMel: decMelSeq, postMel: postMelSeq)
    }

    /// 言語特徴量から音声合成用対数メルスペクトルおよび F0/有声度を完全推論する
    public func synthesizeMelAndF0(
        linguisticFeatures: LinguisticFeatures,
        meanFramesPerMora: Float? = nil,
        f0Scale: Float = 1.0,
        durationScale: Float = 1.0
    ) -> (mel: [[Float]], f0Contour: [Float], voicedFlags: [Float], energyContour: [Float], durations: [Int32]) {
        let phoneIds = linguisticFeatures.phoneIds
        let phoneCount = phoneIds.count
        if phoneCount < 3 {
            return ([], [], [], [], [])
        }

        // 1. 音素エンコーダ（アクセント特徴付き）
        let encStates = encodePhonemes(phoneIds: phoneIds, accent: linguisticFeatures.phoneAccent)

        // 2. 継続時間予測
        let rawDurs = predictDurations(encStates: encStates)

        // 3. 長さ調節
        // なぜ meanFramesPerMora 未指定時は予測継続時間をそのまま使うか:
        // 継続時間予測器は文脈（句末の伸び、短い発話のゆっくりさ）を学習しているが、
        // 固定のモーラ速度へ総和を強制すると比率しか残らず、学習した発話速度が捨てられるため。
        // 明示的にモーラ速度が与えられた場合のみ、その速度へ総和を合わせる。
        let leadSil = Int(linguisticFeatures.durations[0])
        let trailSil = Int(linguisticFeatures.durations[phoneCount - 1])
        let bodyPhoneCount = phoneCount - 2

        var moraCount = 0
        var pIdx = 0
        while pIdx < bodyPhoneCount {
            let pid = Int(phoneIds[1 + pIdx])
            if PhonemeVocabulary.isVowelOrSpecialMora(phoneId: pid) {
                moraCount += 1
            }
            pIdx += 1
        }
        if moraCount <= 0 {
            moraCount = max(1, bodyPhoneCount / 2)
        }

        var predBodySum: Float = 0.0
        var bI = 0
        while bI < bodyPhoneCount {
            predBodySum += rawDurs[1 + bI]
            bI += 1
        }
        var safeDurationScale = durationScale
        if safeDurationScale.isFinite != true || safeDurationScale <= 0.0 {
            safeDurationScale = 1.0
        }
        let targetSpeechFrames: Int
        switch meanFramesPerMora {
        case .some(let rate):
            targetSpeechFrames = max(bodyPhoneCount, Int(roundf(rate * Float(moraCount))))
        case .none:
            targetSpeechFrames = max(bodyPhoneCount, Int(roundf(predBodySum * safeDurationScale)))
        }

        // 予測継続時間の比率を維持した拡大縮小
        let scale: Float
        if 0.001 < predBodySum {
            scale = Float(targetSpeechFrames) / predBodySum
        } else {
            scale = 1.0
        }

        var scaledBodyDurs: [Float] = []
        bI = 0
        while bI < bodyPhoneCount {
            scaledBodyDurs.append(max(1.0, rawDurs[1 + bI] * scale))
            bI += 1
        }

        // 累積和量子化によるフレーム割り当て
        var bodyIntDurs = [Int](repeating: 1, count: bodyPhoneCount)
        var cumTarget: Float = 0.0
        var cumAssigned: Int = 0
        bI = 0
        while bI < bodyPhoneCount {
            cumTarget += scaledBodyDurs[bI]
            let roundedTarget = Int(roundf(cumTarget))
            let dur = max(1, roundedTarget - cumAssigned)
            bodyIntDurs[bI] = dur
            cumAssigned += dur
            bI += 1
        }

        // 合計フレーム数を厳密に targetSpeechFrames と一致させる
        let diff = targetSpeechFrames - cumAssigned
        if diff != 0 && bodyIntDurs.isEmpty != true {
            let lastIdx = bodyIntDurs.count - 1
            bodyIntDurs[lastIdx] = max(1, bodyIntDurs[lastIdx] + diff)
        }

        var finalDurs = [Int](repeating: 0, count: phoneCount)
        finalDurs[0] = leadSil
        bI = 0
        while bI < bodyPhoneCount {
            finalDurs[1 + bI] = bodyIntDurs[bI]
            bI += 1
        }
        finalDurs[phoneCount - 1] = trailSil

        var totalFrames = 0
        var fI = 0
        while fI < phoneCount {
            totalFrames += finalDurs[fI]
            fI += 1
        }

        // 4. フレーム展開および音素内位置 (phonePos) 付与
        let hiddenDim = 256
        var frameStates = [Float](repeating: 0.0, count: totalFrames * 257)
        var voicedFlags = [Float](repeating: 0.0, count: totalFrames)

        var curFrame = 0
        pIdx = 0
        while pIdx < phoneCount {
            let pid = Int(phoneIds[pIdx])
            let dur = finalDurs[pIdx]
            let isV = PhonemeVocabulary.isVoicedPhone(phoneId: pid)
            let vVal: Float
            switch isV {
            case true: vVal = 1.0
            case false: vVal = 0.0
            }

            let maxF = Float(max(1, dur - 1))
            var f = 0
            while f < dur {
                let frameIdx = curFrame + f
                if totalFrames <= frameIdx { break }
                let row = frameIdx * 257
                var h = 0
                while h < hiddenDim {
                    frameStates[row + h] = encStates[pIdx][h]
                    h += 1
                }
                let pos = Float(f) / maxF
                frameStates[row + hiddenDim] = pos
                voicedFlags[frameIdx] = vVal
                f += 1
            }
            curFrame += dur
            pIdx += 1
        }

        // 5. F0 およびエネルギーの予測
        let (rawPredF0Norm, predEnergy) = predictF0AndEnergy(
            frameStates: frameStates,
            totalFrames: totalFrames,
            voicedFlags: voicedFlags
        )
        var predF0Norm = rawPredF0Norm
        if f0Scale != 1.0 {
            var f = 0
            while f < totalFrames {
                predF0Norm[f] = rawPredF0Norm[f] * f0Scale
                f += 1
            }
        }

        // 6. デコーダ条件付けベクトルの構築 (260 次元)
        // [H_frame (256), F0 (1), deltaF0 (1), Energy (1), pos (1)]
        var decCondition = [Float](repeating: 0.0, count: totalFrames * 260)
        var t = 0
        while t < totalFrames {
            let srcRow = t * 257
            let dstRow = t * 260
            var h = 0
            while h < hiddenDim {
                decCondition[dstRow + h] = frameStates[srcRow + h]
                h += 1
            }
            let f0Norm = predF0Norm[t]
            decCondition[dstRow + 256] = f0Norm

            var deltaF0: Float = 0.0
            if 0 < t {
                let prevF0 = predF0Norm[t - 1]
                if 0.0 < f0Norm && 0.0 < prevF0 {
                    let d = (f0Norm - prevF0) / 0.1
                    deltaF0 = max(-1.0, min(1.0, d))
                }
            }
            decCondition[dstRow + 257] = deltaF0
            decCondition[dstRow + 258] = predEnergy[t]
            decCondition[dstRow + 259] = frameStates[srcRow + hiddenDim] // pos
            t += 1
        }

        // 7. デコーダ推論 + PostNet
        let (_, basePostMel) = decodeMel(decoderCondition: decCondition, totalFrames: totalFrames)

        // 7.1 残差加算 (残差重みがある場合)
        var finalPostMel = basePostMel
        if let resFlat = computeMelResidual(condition: decCondition, totalFrames: totalFrames) {
            var t = 0
            while t < totalFrames {
                let row = t * 64
                var c = 0
                while c < 64 {
                    finalPostMel[t][c] = basePostMel[t][c] + resFlat[row + c]
                    c += 1
                }
                t += 1
            }
        }

        // 8. ボコーダ用 F0 (Hz) の復元
        var f0Hz = [Float](repeating: 0.0, count: totalFrames)
        t = 0
        while t < totalFrames {
            let norm = predF0Norm[t]
            if 0.0 < norm && 0.5 <= voicedFlags[t] {
                f0Hz[t] = norm * 500.0
            } else {
                f0Hz[t] = 0.0
            }
            t += 1
        }

        return (mel: finalPostMel, f0Contour: f0Hz, voicedFlags: voicedFlags, energyContour: predEnergy, durations: finalDurs.map { Int32($0) })
    }

    /// 2層時間畳み込みによるメル残差の計算（Pure Swift 実装）
    /// 入力: decoderCondition [totalFrames * 260]
    /// 出力: flatResidual [totalFrames * 64]
    public func computeMelResidual(condition: [Float], totalFrames: Int) -> [Float]? {
        let inDim = 260
        let hiddenDim = 256
        let melDim = 64
        switch (weights.resW1, weights.resB1, weights.resW2, weights.resB2) {
        case (.some(let w1), .some(let b1), .some(let w2), .some(let b2)):
            if w1.count != (hiddenDim * 3 * inDim) || b1.count != hiddenDim || w2.count != (melDim * 1 * hiddenDim) || b2.count != melDim {
                return nil
            }
            var hBuf = [Float](repeating: 0.0, count: totalFrames * hiddenDim)
            condition.withUnsafeBufferPointer { pIn in
                hBuf.withUnsafeMutableBufferPointer { pOut in
                    w1.withUnsafeBufferPointer { pW in
                        b1.withUnsafeBufferPointer { pB in
                            Self.conv1d(
                                input: pIn.baseAddress!,
                                output: pOut.baseAddress!,
                                T: totalFrames,
                                inC: inDim,
                                outC: hiddenDim,
                                kernel: 3,
                                padding: 1,
                                weights: pW.baseAddress!,
                                bias: pB.baseAddress!,
                                activation: .leakyRelu
                            )
                        }
                    }
                }
            }

            var outBuf = [Float](repeating: 0.0, count: totalFrames * melDim)
            hBuf.withUnsafeBufferPointer { pIn in
                outBuf.withUnsafeMutableBufferPointer { pOut in
                    w2.withUnsafeBufferPointer { pW in
                        b2.withUnsafeBufferPointer { pB in
                            Self.conv1d(
                                input: pIn.baseAddress!,
                                output: pOut.baseAddress!,
                                T: totalFrames,
                                inC: hiddenDim,
                                outC: melDim,
                                kernel: 1,
                                padding: 0,
                                weights: pW.baseAddress!,
                                bias: pB.baseAddress!,
                                activation: .none
                            )
                        }
                    }
                }
            }
            return outBuf
        default:
            return nil
        }
    }
}
