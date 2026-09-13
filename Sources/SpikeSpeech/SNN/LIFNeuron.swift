import Foundation

/// LIF (Leaky Integrate-and-Fire) ニューロン設定パラメータ
///
/// SNN 音響モデル内の隠れ層およびリードアウト層で共通の膜電位ダイナミクスと
/// 適応閾値定数を一元管理し、推論と学習で完全に同一の挙動を保証する。
public struct LIFConfig: Sendable, Equatable, Codable {
    public let beta: Float      // 膜電位減衰率 (0.0 < beta < 1.0)
    public let vTh: Float       // 基本発火閾値 (通常 1.0)
    public let vReset: Float    // リセット基準電位 (0.0)
    public let alpha: Float     // Fast Sigmoid 代理勾配の鋭さパラメータ (通常 2.0)
    public let rho: Float       // 適応閾値減衰率 (通常 0.85)
    public let gamma: Float     // 発火時閾値上昇幅 (0.0 で固定閾値, 0.0 < gamma で ALIF)

    public init(
        beta: Float = 0.8,
        vTh: Float = 1.0,
        vReset: Float = 0.0,
        alpha: Float = 2.0,
        rho: Float = 0.85,
        gamma: Float = 0.0
    ) {
        self.beta = beta
        self.vTh = vTh
        self.vReset = vReset
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
    }
}

/// 膜電位 `v`、スパイク `s`、適応閾値 `a` の状態バッファ
///
/// 連続したヒープメモリ領域を保証し、推論時のポインタアクセスおよび SIMD8
/// レジスタロードにおけるキャッシュ効率を最大化する。
public final class LIFState: @unchecked Sendable {
    public var v: ContiguousArray<Float>
    public var s: ContiguousArray<Float>
    public var a: ContiguousArray<Float>
    public let size: Int

    public init(size: Int) {
        self.size = size
        self.v = ContiguousArray<Float>(repeating: 0.0, count: size)
        self.s = ContiguousArray<Float>(repeating: 0.0, count: size)
        self.a = ContiguousArray<Float>(repeating: 0.0, count: size)
    }

    /// フレームや発話の境界で状態を初期化する際、新たな配列確保を排して
    /// メモリ再割り当てコストを発生させないインプレースリセット。
    @inline(__always)
    public func reset() {
        var i = 0
        while i < size {
            v[i] = 0.0
            s[i] = 0.0
            a[i] = 0.0
            i += 1
        }
    }
}

/// LIF / ALIF 膜電位計算エンジン (スカラーおよび SIMD8 最適化)
///
/// 状態を持たない静的インライン展開可能関数群としてまとめ、呼び出しオーバーヘッドを
/// ゼロにしつつ、隠れ層のハードリセットとリードアウト層の減算リセットを明示的に分離する。
public enum LIFNeuronEngine {
    /// 膜電位のクランプ範囲 (学習側 MLX の clip と厳密に同期)
    public static let vClampMin: Float = -20.0
    public static let vClampMax: Float = 20.0

    /// リードアウト層における閾値単位クリップ幅 ([-1.0, 1.0])
    public static let readoutClipInThresholdUnits: Float = 1.0

    /// 閾値未満のアナログ連続値を線形層に渡す際、極端な飽和電圧が後段の音響スペクトルに
    /// 異常なノイズを混入させるのを防ぐ目的で膜電位をクリップする。
    @inline(__always)
    public static func scaleReadout(_ v: Float, vTh: Float) -> Float {
        let k = readoutClipInThresholdUnits
        var scaled: Float = 0.0
        if vTh != 0.0 {
            scaled = v / vTh
        }
        if scaled < -k {
            return -k
        }
        if k < scaled {
            return k
        }
        return scaled
    }

    /// 再帰結合による膜電位の発散やオーバーフローを未然に防ぎ、浮動小数点の安定性を保つ。
    @inline(__always)
    public static func clampMembrane(_ v: Float) -> Float {
        if v < vClampMin {
            return vClampMin
        }
        if vClampMax < v {
            return vClampMax
        }
        return v
    }

    /// 1 ニューロンのスカラー更新ステップ (固定閾値 LIF, ハードリセット)
    ///
    /// 発火したニューロンの膜電位をゼロ電位に戻すことで、過剰な連続発火を抑制し疎なスパイク表現を促進する。
    @inline(__always)
    public static func stepScalar(
        config: LIFConfig,
        vPrev: Float,
        sPrev: Float,
        inputCurrent: Float
    ) -> (vNext: Float, sNext: Float) {
        let vDecayed = config.beta * vPrev * (1.0 - sPrev)
        let vNext = clampMembrane(vDecayed + inputCurrent)
        var sNext: Float = 0.0
        if config.vTh <= vNext {
            sNext = 1.0
        }
        return (vNext: vNext, sNext: sNext)
    }

    /// 1 ニューロンの適応型スカラー更新ステップ (ALIF, ハードリセット)
    ///
    /// 同一の強い入力が持続した際の発火頻度飽和を防ぎ、時間文脈に応じた適応特性を獲得させる。
    @inline(__always)
    public static func stepScalarAdaptive(
        config: LIFConfig,
        vPrev: Float,
        sPrev: Float,
        aPrev: Float,
        inputCurrent: Float
    ) -> (vNext: Float, sNext: Float, aNext: Float) {
        let vDecayed = config.beta * vPrev * (1.0 - sPrev)
        let vNext = clampMembrane(vDecayed + inputCurrent)
        let aNext = (config.rho * aPrev) + (config.gamma * sPrev)
        let dynVTh = config.vTh + aNext
        var sNext: Float = 0.0
        if dynVTh <= vNext {
            sNext = 1.0
        }
        return (vNext: vNext, sNext: sNext, aNext: aNext)
    }

    /// 隠れ層の ALIF を SIMD8 で一括並列更新 (ハードリセット)
    ///
    /// 条件分岐命令による CPU 分岐予測失敗を排除し、8 レーンの膜電位および
    /// スパイク発火判定を単一サイクルでベクトル実行する。
    @inline(__always)
    public static func stepAdaptiveSIMD8(
        config: LIFConfig,
        vPtr: UnsafeMutablePointer<Float>,
        sPtr: UnsafeMutablePointer<Float>,
        aPtr: UnsafeMutablePointer<Float>,
        curPtr: UnsafePointer<Float>,
        count: Int
    ) {
        let limit = count - (count % 8)
        let betaVec = SIMD8<Float>(repeating: config.beta)
        let oneVec = SIMD8<Float>(repeating: 1.0)
        let rhoVec = SIMD8<Float>(repeating: config.rho)
        let gammaVec = SIMD8<Float>(repeating: config.gamma)
        let vThVec = SIMD8<Float>(repeating: config.vTh)
        let lowVec = SIMD8<Float>(repeating: vClampMin)
        let highVec = SIMD8<Float>(repeating: vClampMax)
        let zeroVec = SIMD8<Float>(repeating: 0.0)

        var i = 0
        while i < limit {
            let vPrev = SIMD8<Float>(
                vPtr[i + 0], vPtr[i + 1], vPtr[i + 2], vPtr[i + 3],
                vPtr[i + 4], vPtr[i + 5], vPtr[i + 6], vPtr[i + 7]
            )
            let sPrev = SIMD8<Float>(
                sPtr[i + 0], sPtr[i + 1], sPtr[i + 2], sPtr[i + 3],
                sPtr[i + 4], sPtr[i + 5], sPtr[i + 6], sPtr[i + 7]
            )
            let aPrev = SIMD8<Float>(
                aPtr[i + 0], aPtr[i + 1], aPtr[i + 2], aPtr[i + 3],
                aPtr[i + 4], aPtr[i + 5], aPtr[i + 6], aPtr[i + 7]
            )
            let inCur = SIMD8<Float>(
                curPtr[i + 0], curPtr[i + 1], curPtr[i + 2], curPtr[i + 3],
                curPtr[i + 4], curPtr[i + 5], curPtr[i + 6], curPtr[i + 7]
            )

            let vDecayed = betaVec * vPrev * (oneVec - sPrev)
            let vRaw = vDecayed + inCur
            var vNext = vRaw.replacing(with: lowVec, where: vRaw .< lowVec)
            vNext = vNext.replacing(with: highVec, where: highVec .< vNext)
            let aNext = (rhoVec * aPrev) + (gammaVec * sPrev)
            let dynVTh = vThVec + aNext
            let sNext = zeroVec.replacing(with: oneVec, where: dynVTh .<= vNext)

            vPtr[i + 0] = vNext[0]
            vPtr[i + 1] = vNext[1]
            vPtr[i + 2] = vNext[2]
            vPtr[i + 3] = vNext[3]
            vPtr[i + 4] = vNext[4]
            vPtr[i + 5] = vNext[5]
            vPtr[i + 6] = vNext[6]
            vPtr[i + 7] = vNext[7]

            sPtr[i + 0] = sNext[0]
            sPtr[i + 1] = sNext[1]
            sPtr[i + 2] = sNext[2]
            sPtr[i + 3] = sNext[3]
            sPtr[i + 4] = sNext[4]
            sPtr[i + 5] = sNext[5]
            sPtr[i + 6] = sNext[6]
            sPtr[i + 7] = sNext[7]

            aPtr[i + 0] = aNext[0]
            aPtr[i + 1] = aNext[1]
            aPtr[i + 2] = aNext[2]
            aPtr[i + 3] = aNext[3]
            aPtr[i + 4] = aNext[4]
            aPtr[i + 5] = aNext[5]
            aPtr[i + 6] = aNext[6]
            aPtr[i + 7] = aNext[7]
            i += 8
        }

        // 8 の倍数以外の任意のニューロン数構成においても境界を正確に更新する。
        while i < count {
            let res = stepScalarAdaptive(
                config: config,
                vPrev: vPtr[i],
                sPrev: sPtr[i],
                aPrev: aPtr[i],
                inputCurrent: curPtr[i]
            )
            vPtr[i] = res.vNext
            sPtr[i] = res.sNext
            aPtr[i] = res.aNext
            i += 1
        }
    }

    /// 最終リードアウト層の 1 ニューロンスカラー更新ステップ (減算リセット)
    ///
    /// 閾値を超えた分の膜電位エネルギーを次ステップへ引き継ぎつつ、
    /// 膜電位のアナログ値を積算することで連続音響スペクトルの微細情報を保持する。
    @inline(__always)
    public static func stepReadoutScalarAdaptive(
        config: LIFConfig,
        vPrev: Float,
        sPrev: Float,
        aPrev: Float,
        inputCurrent: Float
    ) -> (vNext: Float, sNext: Float, aNext: Float, readout: Float) {
        let vIntegrated = clampMembrane(config.beta * vPrev + inputCurrent)
        let aNext = (config.rho * aPrev) + (config.gamma * sPrev)
        let dynVTh = config.vTh + aNext
        var sNext: Float = 0.0
        if dynVTh <= vIntegrated {
            sNext = 1.0
        }
        let readout = scaleReadout(vIntegrated, vTh: config.vTh)
        let vNext = clampMembrane(vIntegrated - (sNext * config.vTh))
        return (vNext: vNext, sNext: sNext, aNext: aNext, readout: readout)
    }

    /// 最終リードアウト層の SIMD8 一括並列更新 (減算リセット ＋ アナログ膜電位積算)
    ///
    /// 内部時間ステップ全体での膜電位積算をメモリロード・ストアと同時に行い、
    /// リードアウト計算のレイテンシを極小化する。
    @inline(__always)
    public static func stepReadoutAdaptiveSIMD8(
        config: LIFConfig,
        vPtr: UnsafeMutablePointer<Float>,
        sPtr: UnsafeMutablePointer<Float>,
        aPtr: UnsafeMutablePointer<Float>,
        curPtr: UnsafePointer<Float>,
        readoutSumPtr: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        let limit = count - (count % 8)
        let betaVec = SIMD8<Float>(repeating: config.beta)
        let rhoVec = SIMD8<Float>(repeating: config.rho)
        let gammaVec = SIMD8<Float>(repeating: config.gamma)
        let vThVec = SIMD8<Float>(repeating: config.vTh)
        let lowVec = SIMD8<Float>(repeating: vClampMin)
        let highVec = SIMD8<Float>(repeating: vClampMax)
        let zeroVec = SIMD8<Float>(repeating: 0.0)
        let oneVec = SIMD8<Float>(repeating: 1.0)
        let k = readoutClipInThresholdUnits
        let negKVec = SIMD8<Float>(repeating: -k)
        let posKVec = SIMD8<Float>(repeating: k)
        var invThVec = SIMD8<Float>(repeating: 0.0)
        if config.vTh != 0.0 {
            invThVec = SIMD8<Float>(repeating: 1.0 / config.vTh)
        }

        var i = 0
        while i < limit {
            let vPrev = SIMD8<Float>(
                vPtr[i + 0], vPtr[i + 1], vPtr[i + 2], vPtr[i + 3],
                vPtr[i + 4], vPtr[i + 5], vPtr[i + 6], vPtr[i + 7]
            )
            let sPrev = SIMD8<Float>(
                sPtr[i + 0], sPtr[i + 1], sPtr[i + 2], sPtr[i + 3],
                sPtr[i + 4], sPtr[i + 5], sPtr[i + 6], sPtr[i + 7]
            )
            let aPrev = SIMD8<Float>(
                aPtr[i + 0], aPtr[i + 1], aPtr[i + 2], aPtr[i + 3],
                aPtr[i + 4], aPtr[i + 5], aPtr[i + 6], aPtr[i + 7]
            )
            let inCur = SIMD8<Float>(
                curPtr[i + 0], curPtr[i + 1], curPtr[i + 2], curPtr[i + 3],
                curPtr[i + 4], curPtr[i + 5], curPtr[i + 6], curPtr[i + 7]
            )

            let vRaw = (betaVec * vPrev) + inCur
            var vIntegrated = vRaw.replacing(with: lowVec, where: vRaw .< lowVec)
            vIntegrated = vIntegrated.replacing(with: highVec, where: highVec .< vIntegrated)
            let aNext = (rhoVec * aPrev) + (gammaVec * sPrev)
            let dynVTh = vThVec + aNext
            let sNext = zeroVec.replacing(with: oneVec, where: dynVTh .<= vIntegrated)

            var scaled = vIntegrated * invThVec
            scaled = scaled.replacing(with: negKVec, where: scaled .< negKVec)
            scaled = scaled.replacing(with: posKVec, where: posKVec .< scaled)
            let sumPrev = SIMD8<Float>(
                readoutSumPtr[i + 0], readoutSumPtr[i + 1], readoutSumPtr[i + 2], readoutSumPtr[i + 3],
                readoutSumPtr[i + 4], readoutSumPtr[i + 5], readoutSumPtr[i + 6], readoutSumPtr[i + 7]
            )
            let sumNext = sumPrev + scaled

            let vSub = vIntegrated - (sNext * vThVec)
            var vNext = vSub.replacing(with: lowVec, where: vSub .< lowVec)
            vNext = vNext.replacing(with: highVec, where: highVec .< vNext)

            vPtr[i + 0] = vNext[0]
            vPtr[i + 1] = vNext[1]
            vPtr[i + 2] = vNext[2]
            vPtr[i + 3] = vNext[3]
            vPtr[i + 4] = vNext[4]
            vPtr[i + 5] = vNext[5]
            vPtr[i + 6] = vNext[6]
            vPtr[i + 7] = vNext[7]

            sPtr[i + 0] = sNext[0]
            sPtr[i + 1] = sNext[1]
            sPtr[i + 2] = sNext[2]
            sPtr[i + 3] = sNext[3]
            sPtr[i + 4] = sNext[4]
            sPtr[i + 5] = sNext[5]
            sPtr[i + 6] = sNext[6]
            sPtr[i + 7] = sNext[7]

            aPtr[i + 0] = aNext[0]
            aPtr[i + 1] = aNext[1]
            aPtr[i + 2] = aNext[2]
            aPtr[i + 3] = aNext[3]
            aPtr[i + 4] = aNext[4]
            aPtr[i + 5] = aNext[5]
            aPtr[i + 6] = aNext[6]
            aPtr[i + 7] = aNext[7]

            readoutSumPtr[i + 0] = sumNext[0]
            readoutSumPtr[i + 1] = sumNext[1]
            readoutSumPtr[i + 2] = sumNext[2]
            readoutSumPtr[i + 3] = sumNext[3]
            readoutSumPtr[i + 4] = sumNext[4]
            readoutSumPtr[i + 5] = sumNext[5]
            readoutSumPtr[i + 6] = sumNext[6]
            readoutSumPtr[i + 7] = sumNext[7]
            i += 8
        }

        while i < count {
            let res = stepReadoutScalarAdaptive(
                config: config,
                vPrev: vPtr[i],
                sPrev: sPtr[i],
                aPrev: aPtr[i],
                inputCurrent: curPtr[i]
            )
            vPtr[i] = res.vNext
            sPtr[i] = res.sNext
            aPtr[i] = res.aNext
            readoutSumPtr[i] += res.readout
            i += 1
        }
    }
}
