import Foundation

/// 推論時ホットパス用のゼロアロケーション作業バッファ
///
/// フレーム単位および内部時間ステップの推論ループ内において、配列の新規生成や
/// メモリの再割り当てを完全に排除し、決定論的な低レイテンシ推論を保証する。
public final class AcousticWorkspace: @unchecked Sendable {
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let numLayers: Int

    /// 入力電流キャッシュ
    public var inputCurrents: [Float]

    /// 時間ステップ内の結合電流バッファ
    public var stepCurrents: [Float]

    /// 前層入力電流の残差加算用バッファ
    public var stepCurrentsPrev: [Float]

    /// 層 0 発火ニューロンのインデックス記録配列
    public var activeSpikes: [Int]

    /// 上位層発火ニューロンのインデックス記録配列
    public var activeLayerSpikes: [Int]

    /// リードアウト有効発火ニューロンのインデックス記録配列
    public var activeReadoutIndices: [Int]

    /// リードアウト有効発火ニューロンの時間平均レート記録配列
    public var activeRates: [Float]

    /// 最終リードアウト層のアナログ膜電位積算バッファ
    public var readoutSums: [Float]

    /// フレーム出力特徴量の一時格納バッファ
    public var outputFeatures: [Float]

    /// 各層のニューロン状態
    public var layerStates: [LIFState]

    /// 高速バルクゼロクリア用の参照テンプレートバッファ
    private let zeroFloats: [Float]

    public init(maxHiddenDim: Int = 1024, outputDim: Int = 80, numLayers: Int = 2) {
        let safeLayers = max(1, numLayers)
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.numLayers = safeLayers

        self.inputCurrents = [Float](repeating: 0.0, count: maxHiddenDim)
        self.stepCurrents = [Float](repeating: 0.0, count: maxHiddenDim)
        self.stepCurrentsPrev = [Float](repeating: 0.0, count: maxHiddenDim)
        self.activeSpikes = [Int](repeating: 0, count: maxHiddenDim)
        self.activeLayerSpikes = [Int](repeating: 0, count: maxHiddenDim)
        self.activeReadoutIndices = [Int](repeating: 0, count: maxHiddenDim)
        self.activeRates = [Float](repeating: 0.0, count: maxHiddenDim)
        self.readoutSums = [Float](repeating: 0.0, count: maxHiddenDim)
        self.outputFeatures = [Float](repeating: 0.0, count: outputDim)
        self.zeroFloats = [Float](repeating: 0.0, count: maxHiddenDim)

        var states: [LIFState] = []
        var l = 0
        while l < safeLayers {
            states.append(LIFState(size: maxHiddenDim))
            l += 1
        }
        self.layerStates = states
    }

    /// 発話間の境界でニューロン状態および積算バッファを初期化する際、ヒープ割り当てを発生させずにポインタ直接操作でバルククリアする。
    @inline(__always)
    public func reset() {
        var l = 0
        while l < numLayers {
            layerStates[l].reset()
            l += 1
        }

        let hDim = maxHiddenDim
        inputCurrents.withUnsafeMutableBufferPointer { dst in
            zeroFloats.withUnsafeBufferPointer { src in
                dst.baseAddress!.update(from: src.baseAddress!, count: hDim)
            }
        }
        stepCurrents.withUnsafeMutableBufferPointer { dst in
            zeroFloats.withUnsafeBufferPointer { src in
                dst.baseAddress!.update(from: src.baseAddress!, count: hDim)
            }
        }
        stepCurrentsPrev.withUnsafeMutableBufferPointer { dst in
            zeroFloats.withUnsafeBufferPointer { src in
                dst.baseAddress!.update(from: src.baseAddress!, count: hDim)
            }
        }
        readoutSums.withUnsafeMutableBufferPointer { dst in
            zeroFloats.withUnsafeBufferPointer { src in
                dst.baseAddress!.update(from: src.baseAddress!, count: hDim)
            }
        }

        let outDim = outputDim
        outputFeatures.withUnsafeMutableBufferPointer { dst in
            zeroFloats.withUnsafeBufferPointer { src in
                dst.baseAddress!.update(from: src.baseAddress!, count: outDim)
            }
        }
    }

    /// 時間ステップの開始時に結合電流バッファを高速にゼロクリアする。
    @inline(__always)
    public func clearStepCurrents() {
        let count = maxHiddenDim
        stepCurrents.withUnsafeMutableBufferPointer { dst in
            zeroFloats.withUnsafeBufferPointer { src in
                dst.baseAddress!.update(from: src.baseAddress!, count: count)
            }
        }
    }

    /// フレーム先頭で内部時間ステップにわたるアナログ膜電位の積算をゼロから開始する。
    @inline(__always)
    public func clearReadoutSums() {
        let count = maxHiddenDim
        readoutSums.withUnsafeMutableBufferPointer { dst in
            zeroFloats.withUnsafeBufferPointer { src in
                dst.baseAddress!.update(from: src.baseAddress!, count: count)
            }
        }
    }

    /// フレーム間で膜電位が過大に飽和して発火状態が固定化（ロック）するのを防ぐため、
    /// 直前フレームの膜電位を減衰リークさせて音素遷移への感度を回復する。
    @inline(__always)
    public func leakMembranes(decay: Float = 0.2) {
        var l = 0
        while l < numLayers {
            let st = layerStates[l]
            let hMax = maxHiddenDim
            let limit = hMax - (hMax % 8)
            let decayVec = SIMD8<Float>(repeating: decay)
            st.v.withUnsafeMutableBufferPointer { vBuf in
                let vPtr = vBuf.baseAddress!
                var n = 0
                while n < limit {
                    let v = SIMD8<Float>(
                        vPtr[n + 0], vPtr[n + 1], vPtr[n + 2], vPtr[n + 3],
                        vPtr[n + 4], vPtr[n + 5], vPtr[n + 6], vPtr[n + 7]
                    )
                    let decayed = v * decayVec
                    vPtr[n + 0] = decayed[0]
                    vPtr[n + 1] = decayed[1]
                    vPtr[n + 2] = decayed[2]
                    vPtr[n + 3] = decayed[3]
                    vPtr[n + 4] = decayed[4]
                    vPtr[n + 5] = decayed[5]
                    vPtr[n + 6] = decayed[6]
                    vPtr[n + 7] = decayed[7]
                    n += 8
                }
                while n < hMax {
                    vPtr[n] *= decay
                    n += 1
                }
            }
            st.s.withUnsafeMutableBufferPointer { sBuf in
                zeroFloats.withUnsafeBufferPointer { zBuf in
                    sBuf.baseAddress!.update(from: zBuf.baseAddress!, count: hMax)
                }
            }
            l += 1
        }
    }
}
