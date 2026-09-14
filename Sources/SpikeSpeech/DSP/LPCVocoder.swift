import Foundation

/// ボコーダー用音響パラメータフレーム
///
/// SNN音響モデルまたはMelToLPC変換器から供給される1フレーム分の
/// LPC多項式係数、音響ゲイン、基本周波数F0、および有声度を一括して保持する。
public struct AcousticFrame: Sendable, Equatable {
    public let lpcCoefficients: [Float] // 16次 LPC 係数
    public let gain: Float              // フレームエネルギー / RMS ゲイン
    public let pitchF0: Float           // 基本周波数 [Hz]、無声音は 0.0
    public let voiced: Float            // 有声度 [0.0, 1.0]

    public init(
        lpcCoefficients: [Float],
        gain: Float,
        pitchF0: Float,
        voiced: Float
    ) {
        self.lpcCoefficients = lpcCoefficients
        self.gain = gain
        self.pitchF0 = pitchF0
        self.voiced = voiced
    }
}

/// Source-Filter LPC 合成ボコーダーエンジン
///
/// 声帯振動と気流雑音を声道共鳴で畳み込む物理音響構造を模倣し、
/// 1サンプルあたりわずか16回の積和演算という超高速性と定数メモリフットプリントで明瞭な音声を合成する。
public final class LPCVocoder: @unchecked Sendable {

    public let sampleRate: Float
    public let frameSize: Int       // 160サンプル
    public let lpcOrder: Int        // 16
    public let deEmphasisCoeff: Float // 0.97
    public private(set) var glottal: GlottalSource

    private let rosenbergPulse: RosenbergPulse
    private var rngState: UInt64 = 88172645463325252

    // 内部状態 (ゼロアロケーション保持)
    private var prevLpcCoeffs: [Float]
    private var currLpcCoeffs: [Float]
    private var prevGain: Float = 0.0
    private var currGain: Float = 0.0
    private var filterMemory: [Float] // 過去 P サンプル
    private var deEmphasisState: Float = 0.0
    /// 唇放射微分のための直前声門波サンプル
    private var prevVoicedPulse: Float = 0.0
    /// スペクトル傾斜フィルタ用直前パルスサンプル
    private var prevTiltedPulse: Float = 0.0
    /// 無声音摩擦気流の高域放射整形（低域濁りヒスノイズ排除）のための直前乱数サンプル
    private var prevUnvoicedNoise: Float = 0.0
    /// フレーム間 F0 連続補間のための直前周波数
    private var prevPitchF0: Float = 0.0
    /// フレーム間有声度連続補間のための直前有声度
    private var prevVoicedRatio: Float = 0.0

    /// 初期化
    ///
    /// フレーム合成ごとの動的メモリ再割り当てを排除し、
    /// リアルタイムストリーミング合成時の処理遅延やレイテンシスパイクを根絶する。
    public init(
        sampleRate: Float = 16000.0,
        frameSize: Int = AudioConfig.hopSize, // 160
        lpcOrder: Int = AudioConfig.lpcOrder, // 16
        deEmphasisCoeff: Float = 0.95,
        glottal: GlottalSource = GlottalSource()
    ) {
        self.sampleRate = sampleRate
        self.frameSize = frameSize
        self.lpcOrder = lpcOrder
        self.deEmphasisCoeff = deEmphasisCoeff
        self.glottal = glottal

        self.rosenbergPulse = RosenbergPulse(sampleRate: sampleRate)
        self.rosenbergPulse.apply(glottal: glottal)
        self.prevLpcCoeffs = [Float](repeating: 0.0, count: lpcOrder)
        self.currLpcCoeffs = [Float](repeating: 0.0, count: lpcOrder)
        self.filterMemory = [Float](repeating: 0.0, count: lpcOrder)
    }

    /// 声帯生理パラメータ（GlottalSource）の動的適用
    ///
    /// 声門パルス開口率・閉鎖形状を更新し、息漏れやスペクトル傾斜を即時反映する。
    public func apply(glottal: GlottalSource) {
        self.glottal = glottal
        rosenbergPulse.apply(glottal: glottal)
    }

    /// ボコーダーの内部状態をリセット
    ///
    /// 新しい発話文の合成開始時に過去の残響やフィルタメモリを完全に消去し、
    /// 先頭フレームでの異音や位相引きずりを防止する。
    public func reset() {
        rosenbergPulse.reset()
        rngState = 88172645463325252

        var i = 0
        while i < lpcOrder {
            prevLpcCoeffs[i] = 0.0
            currLpcCoeffs[i] = 0.0
            filterMemory[i] = 0.0
            i += 1
        }
        prevGain = 0.0
        currGain = 0.0
        deEmphasisState = 0.0
        prevVoicedPulse = 0.0
        prevTiltedPulse = 0.0
        prevUnvoicedNoise = 0.0
        prevPitchF0 = 0.0
        prevVoicedRatio = 0.0
    }

    /// 高速 Xorshift64 疑似乱数生成器
    ///
    /// システムコール負荷を排除し、ホットループ内で3回のビット演算のみで無声音ノイズを高速生成する。
    @inline(__always)
    private func nextRandomFloat() -> Float {
        rngState ^= (rngState << 13)
        rngState ^= (rngState >> 7)
        rngState ^= (rngState << 17)
        let u32 = UInt32(truncatingIfNeeded: rngState)
        // [0, UInt32.max] を [-1.0, 1.0] に線形スケーリング
        return (Float(u32) * (2.0 / 4294967295.0)) - 1.0
    }

    /// 1 フレームの音声を逐次合成
    ///
    /// フレーム境界でLPC係数やゲインが急変した際にフィルタ伝達関数が不連続となり
    /// クリックノイズが発生する現象を抑制するため、サンプル単位で線形補間を行う。
    @inline(__always)
    public func synthesizeFrame(
        frame: AcousticFrame,
        dst: UnsafeMutablePointer<Float>
    ) {
        // 現フレーム係数・パラメータの取り込みと包括的 NaN / Inf 検査
        var hasInvalidParam = false
        if frame.gain.isFinite != true {
            hasInvalidParam = true
            currGain = 0.0
        } else {
            currGain = frame.gain
        }

        if frame.pitchF0.isFinite != true {
            hasInvalidParam = true
        }
        if frame.voiced.isFinite != true {
            hasInvalidParam = true
        }

        var k = 0
        let coeffCount = frame.lpcCoefficients.count
        while k < lpcOrder {
            if k < coeffCount {
                let c = frame.lpcCoefficients[k]
                if c.isFinite != true {
                    hasInvalidParam = true
                    currLpcCoeffs[k] = 0.0
                } else {
                    currLpcCoeffs[k] = c
                }
            } else {
                currLpcCoeffs[k] = 0.0
            }
            k += 1
        }

        // 音響モデルの勾配発散等により非有限値が入力された際に過去の残響や発振状態を速やかに断ち切り、後続サンプルへの例外値伝播を防止する。
        if hasInvalidParam {
            resetMemory()
            deEmphasisState = 0.0
            prevGain = 0.0
            currGain = 0.0
            var i = 0
            while i < frameSize {
                dst[i] = 0.0
                i += 1
            }
            return
        }

        // 最初のフレーム等で前フレームが未設定の場合は現フレーム値で初期化
        var isInitial = true
        k = 0
        while k < lpcOrder {
            if prevLpcCoeffs[k] != 0.0 {
                isInitial = false
                break
            }
            k += 1
        }
        if isInitial {
            prevLpcCoeffs.withUnsafeMutableBufferPointer { prevBuf in
                currLpcCoeffs.withUnsafeBufferPointer { curBuf in
                    prevBuf.baseAddress!.update(from: curBuf.baseAddress!, count: lpcOrder)
                }
            }
            prevGain = currGain
            prevPitchF0 = frame.pitchF0
            prevVoicedRatio = frame.voiced
        }

        let f0 = frame.pitchF0
        var effectiveVoiced = frame.voiced
        // 無声子音区間において有声音励起が漏洩し濁音化するのを防ぐため、F0がゼロ以下の場合は有声度をゼロとする。
        if f0 <= 0.0 {
            effectiveVoiced = 0.0
        }
        if effectiveVoiced < 0.0 {
            effectiveVoiced = 0.0
        }
        if 1.0 < effectiveVoiced {
            effectiveVoiced = 1.0
        }
        let unvoicedWeight = 1.0 - effectiveVoiced

        if lpcOrder == 16 {
            synthesizeFrameOrder16(
                dst: dst,
                effectiveVoiced: effectiveVoiced,
                unvoicedWeight: unvoicedWeight,
                f0: f0
            )
        } else {
            synthesizeFrameGeneral(
                dst: dst,
                effectiveVoiced: effectiveVoiced,
                unvoicedWeight: unvoicedWeight,
                f0: f0
            )
        }

        // 次フレームへの係数およびゲインのスライド
        prevLpcCoeffs.withUnsafeMutableBufferPointer { prevBuf in
            currLpcCoeffs.withUnsafeBufferPointer { curBuf in
                prevBuf.baseAddress!.update(from: curBuf.baseAddress!, count: lpcOrder)
            }
        }
        // 次フレームでの線形補間において前フレームの確定ゲインを正確に引き継ぎ、
        // 無音フレームへの遷移時に過去のゲインが残留して未補正ノイズを放出し続ける不具合を根絶する。
        prevGain = currGain
        prevPitchF0 = f0
        prevVoicedRatio = effectiveVoiced

        // 連続無音フレームにおいて過去の残響やディエンファシス積分テールを即時フラッシュし、ポーズ区間でのヒスノイズ漏洩を根絶
        if currGain <= 1e-6 && prevGain <= 1e-6 {
            resetMemory()
            deEmphasisState = 0.0
            prevVoicedPulse = 0.0
            prevUnvoicedNoise = 0.0
        }
    }

    /// 16次全極 AR 声道フィルタの SIMD8 高速合成 (Hot Path)
    @inline(__always)
    private func synthesizeFrameOrder16(
        dst: UnsafeMutablePointer<Float>,
        effectiveVoiced: Float,
        unvoicedWeight: Float,
        f0: Float
    ) {
        let invN = 1.0 / Float(frameSize)
        let p0 = SIMD8<Float>(
            prevLpcCoeffs[0], prevLpcCoeffs[1], prevLpcCoeffs[2], prevLpcCoeffs[3],
            prevLpcCoeffs[4], prevLpcCoeffs[5], prevLpcCoeffs[6], prevLpcCoeffs[7]
        )
        let p1 = SIMD8<Float>(
            prevLpcCoeffs[8], prevLpcCoeffs[9], prevLpcCoeffs[10], prevLpcCoeffs[11],
            prevLpcCoeffs[12], prevLpcCoeffs[13], prevLpcCoeffs[14], prevLpcCoeffs[15]
        )
        let c0 = SIMD8<Float>(
            currLpcCoeffs[0], currLpcCoeffs[1], currLpcCoeffs[2], currLpcCoeffs[3],
            currLpcCoeffs[4], currLpcCoeffs[5], currLpcCoeffs[6], currLpcCoeffs[7]
        )
        let c1 = SIMD8<Float>(
            currLpcCoeffs[8], currLpcCoeffs[9], currLpcCoeffs[10], currLpcCoeffs[11],
            currLpcCoeffs[12], currLpcCoeffs[13], currLpcCoeffs[14], currLpcCoeffs[15]
        )

        var n = 0
        while n < frameSize {
            let lambda = Float(n) * invN
            let oneMinusLambda = 1.0 - lambda

            // 1. 補間ゲインと励起信号の生成
            let g = (oneMinusLambda * prevGain) + (lambda * currGain)

            // F0 のサンプル単位連続補間（急峻な低周波スライドや10ms段差クリックを防止）
            let sampleF0: Float
            switch true {
            case prevPitchF0 <= 0.0:
                sampleF0 = f0
            case f0 <= 0.0:
                sampleF0 = prevPitchF0
            default:
                sampleF0 = (oneMinusLambda * prevPitchF0) + (lambda * f0)
            }

            // 有声度・無声度のサンプル単位連続補間
            // 前フレームが無声（破裂音等）で現フレームが有声（母音等）に遷移する境界において、
            // 有声度を160サンプルかけて徐々に補間すると母音の立ち上がりで激しい無声乱数ノイズが注入される。
            // したがって現フレームが有声（0.5 <= effectiveVoiced）の場合は即座に有声音励起とし、無声ノイズを完全遮断する。
            let sampleVoiced: Float
            let sampleUnvoiced: Float
            switch true {
            case prevVoicedRatio <= 0.0 && effectiveVoiced <= 0.0:
                sampleVoiced = 0.0
                sampleUnvoiced = 1.0
            case 0.5 <= effectiveVoiced && prevVoicedRatio < 0.5:
                // 無声から有声へのアタック境界: 母音開始部でのホワイトノイズ混入を根絶するため即座に有声化
                sampleVoiced = effectiveVoiced
                sampleUnvoiced = 0.0
            case effectiveVoiced < 0.5 && 0.5 <= prevVoicedRatio:
                // 有声から無声への減衰境界: 直前の有声波形をスムーズにフェードアウト
                sampleVoiced = oneMinusLambda * prevVoicedRatio
                sampleUnvoiced = 1.0 - sampleVoiced
            default:
                sampleVoiced = (oneMinusLambda * prevVoicedRatio) + (lambda * effectiveVoiced)
                sampleUnvoiced = 1.0 - sampleVoiced
            }

            let rawPulse = rosenbergPulse.nextSample(f0: sampleF0, removeDC: false)
            // 声門容積速度波形に対して口唇放射微分(+6dB/oct)を適用し、声門気流微分波形を再現して母音フォルマント倍音を豊かに励振する。
            // 口唇放射微分フィルタ (1 - 0.98 z^-1) により直流成分は完全に除去され、有声音開始時のステップ不連続（ポップノイズ）を根絶する。
            let lipRadiationCoeff: Float = 0.98
            let radiatedPulse = rawPulse - (lipRadiationCoeff * prevVoicedPulse)
            prevVoicedPulse = rawPulse

            // スペクトル傾斜（Spectral Tilt）の物理制御
            // 負の値（女性や小児の丸い声色）: 1極低域通過フィルタ（リーク積分器）で高域倍音を減衰
            // 正の値（男性や重低音の引き締まったエッジ感）: 1極高域強調フィルタで急峻な声帯閉鎖の高調波を増強
            let tiltedPulse: Float
            let tilt = glottal.spectralTilt
            if tilt < 0.0 {
                var tiltAlpha = -tilt * 0.05
                if 0.35 < tiltAlpha {
                    tiltAlpha = 0.35
                }
                tiltedPulse = ((1.0 - tiltAlpha) * radiatedPulse) + (tiltAlpha * prevTiltedPulse)
            } else {
                if 0.0 < tilt {
                    var k = tilt * 0.15
                    if 0.40 < k {
                        k = 0.40
                    }
                    tiltedPulse = radiatedPulse + (k * (radiatedPulse - prevTiltedPulse))
                } else {
                    tiltedPulse = radiatedPulse
                }
            }
            prevTiltedPulse = tiltedPulse

            // 無声子音区間での低域濁りヒスノイズを防止し、ディエンファシス積分器 (1 / (1 - 0.95 z^-1)) の直流利得発散を相殺するための高域放射微分整形 (1 - 0.95 z^-1)。
            // 無声励起が過大になると母音のフォルマント共鳴を妨げ耳障りなホワイトノイズが知覚されるため、自然な子音アタックが得られる0.18に調整する。
            let rawNoise = nextRandomFloat()
            let shapedNoise = rawNoise - (0.95 * prevUnvoicedNoise)
            prevUnvoicedNoise = rawNoise
            let unvoicedExcitation = shapedNoise * 0.18

            // 有声励起への高周波息漏れ気流ノイズ（Aspiration）の動的混合
            // 無声子音用の微弱スケール（0.18）に依存せず、有声パルス（振幅~0.85）と聴覚的に釣り合うゲイン（0.40）で
            // 高域通過整形ノイズを混合し、子供や女性の息の多い声（Breathy voice）を声帯励起レベルで明瞭に再現する。
            let asp = glottal.aspirationMix
            let excitationScale: Float = 0.85
            let voicedPulsePart = (1.0 - (asp * 0.5)) * (tiltedPulse * excitationScale)
            let aspirationNoise = shapedNoise * 0.40
            let voicedAspPart = asp * aspirationNoise
            let voicedExcitation = voicedPulsePart + voicedAspPart

            let excitation = (sampleVoiced * voicedExcitation) + (sampleUnvoiced * unvoicedExcitation)
            let inputSignal = excitation * g

            // 2. 16次全極 AR 声道フィルタの実行 (SIMD8並列)
            let omlVec = SIMD8<Float>(repeating: oneMinusLambda)
            let lVec = SIMD8<Float>(repeating: lambda)
            let a0 = (omlVec * p0) + (lVec * c0)
            let a1 = (omlVec * p1) + (lVec * c1)
            let m0 = SIMD8<Float>(
                filterMemory[0], filterMemory[1], filterMemory[2], filterMemory[3],
                filterMemory[4], filterMemory[5], filterMemory[6], filterMemory[7]
            )
            let m1 = SIMD8<Float>(
                filterMemory[8], filterMemory[9], filterMemory[10], filterMemory[11],
                filterMemory[12], filterMemory[13], filterMemory[14], filterMemory[15]
            )
            let dot0 = a0 * m0
            let dot1 = a1 * m1
            let arSum = (dot0[0] + dot0[1] + dot0[2] + dot0[3] + dot0[4] + dot0[5] + dot0[6] + dot0[7]) +
                        (dot1[0] + dot1[1] + dot1[2] + dot1[3] + dot1[4] + dot1[5] + dot1[6] + dot1[7])
            var s = inputSignal + arSum

            // 不安定な係数補間により共鳴が急激に増大するのを検知し、状態を初期化して過大バーストノイズを未然に防ぐ。
            if s.isFinite != true {
                s = 0.0
                resetMemory()
                deEmphasisState = 0.0
            } else {
                if 1e4 < abs(s) {
                    if s < 0.0 {
                        s = -10.0
                    } else {
                        s = 10.0
                    }
                    resetMemory()
                    deEmphasisState = 0.0
                }
            }

            // 3. フィルタメモリの展開更新 (ループ不使用)
            filterMemory[15] = filterMemory[14]
            filterMemory[14] = filterMemory[13]
            filterMemory[13] = filterMemory[12]
            filterMemory[12] = filterMemory[11]
            filterMemory[11] = filterMemory[10]
            filterMemory[10] = filterMemory[9]
            filterMemory[9] = filterMemory[8]
            filterMemory[8] = filterMemory[7]
            filterMemory[7] = filterMemory[6]
            filterMemory[6] = filterMemory[5]
            filterMemory[5] = filterMemory[4]
            filterMemory[4] = filterMemory[3]
            filterMemory[3] = filterMemory[2]
            filterMemory[2] = filterMemory[1]
            filterMemory[1] = filterMemory[0]
            filterMemory[0] = s

            // 4. ディエンファシスフィルタ
            var y = s + (deEmphasisCoeff * deEmphasisState)
            if y.isFinite != true {
                y = 0.0
                deEmphasisState = 0.0
            } else {
                deEmphasisState = y
                if abs(deEmphasisState) < 1e-7 {
                    deEmphasisState = 0.0
                }
            }

            // 5. Soft Limiter によるサチュレーション歪み防止
            let absY = abs(y)
            let outSample: Float
            if absY <= 0.8 {
                outSample = y
            } else {
                let excess = absY - 0.8
                let compressed = 0.8 + (0.2 * tanh(excess * 5.0))
                if y < 0.0 {
                    outSample = -compressed
                } else {
                    outSample = compressed
                }
            }

            dst[n] = outSample
            n += 1
        }
    }

    /// 汎用次数の全極 AR 声道フィルタ合成
    @inline(__always)
    private func synthesizeFrameGeneral(
        dst: UnsafeMutablePointer<Float>,
        effectiveVoiced: Float,
        unvoicedWeight: Float,
        f0: Float
    ) {
        let invN = 1.0 / Float(frameSize)
        var n = 0

        while n < frameSize {
            let lambda = Float(n) * invN
            let oneMinusLambda = 1.0 - lambda

            // 1. 補間ゲインと励起信号の生成
            let g = (oneMinusLambda * prevGain) + (lambda * currGain)

            // F0 のサンプル単位連続補間（急峻な低周波スライドや10ms段差クリックを防止）
            let sampleF0: Float
            switch true {
            case prevPitchF0 <= 0.0:
                sampleF0 = f0
            case f0 <= 0.0:
                sampleF0 = prevPitchF0
            default:
                sampleF0 = (oneMinusLambda * prevPitchF0) + (lambda * f0)
            }

            // 有声度・無声度のサンプル単位連続補間
            // 前フレームが無声（破裂音等）で現フレームが有声（母音等）に遷移する境界において、
            // 有声度を160サンプルかけて徐々に補間すると母音の立ち上がりで激しい無声乱数ノイズが注入される。
            // したがって現フレームが有声（0.5 <= effectiveVoiced）の場合は即座に有声音励起とし、無声ノイズを完全遮断する。
            let sampleVoiced: Float
            let sampleUnvoiced: Float
            switch true {
            case prevVoicedRatio <= 0.0 && effectiveVoiced <= 0.0:
                sampleVoiced = 0.0
                sampleUnvoiced = 1.0
            case 0.5 <= effectiveVoiced && prevVoicedRatio < 0.5:
                // 無声から有声へのアタック境界: 母音開始部でのホワイトノイズ混入を根絶するため即座に有声化
                sampleVoiced = effectiveVoiced
                sampleUnvoiced = 0.0
            case effectiveVoiced < 0.5 && 0.5 <= prevVoicedRatio:
                // 有声から無声への減衰境界: 直前の有声波形をスムーズにフェードアウト
                sampleVoiced = oneMinusLambda * prevVoicedRatio
                sampleUnvoiced = 1.0 - sampleVoiced
            default:
                sampleVoiced = (oneMinusLambda * prevVoicedRatio) + (lambda * effectiveVoiced)
                sampleUnvoiced = 1.0 - sampleVoiced
            }

            let rawPulse = rosenbergPulse.nextSample(f0: sampleF0, removeDC: false)
            // 声門容積速度波形に対して口唇放射微分(+6dB/oct)を適用し、声門気流微分波形を再現して母音フォルマント倍音を豊かに励振する。
            // 口唇放射微分フィルタ (1 - 0.98 z^-1) により直流成分は完全に除去され、有声音開始時のステップ不連続（ポップノイズ）を根絶する。
            let lipRadiationCoeff: Float = 0.98
            let radiatedPulse = rawPulse - (lipRadiationCoeff * prevVoicedPulse)
            prevVoicedPulse = rawPulse

            // スペクトル傾斜（Spectral Tilt）の物理制御
            // 負の値（女性や小児の丸い声色）: 1極低域通過フィルタ（リーク積分器）で高域倍音を減衰
            // 正の値（男性や重低音の引き締まったエッジ感）: 1極高域強調フィルタで急峻な声帯閉鎖の高調波を増強
            let tiltedPulse: Float
            let tilt = glottal.spectralTilt
            if tilt < 0.0 {
                var tiltAlpha = -tilt * 0.05
                if 0.35 < tiltAlpha {
                    tiltAlpha = 0.35
                }
                tiltedPulse = ((1.0 - tiltAlpha) * radiatedPulse) + (tiltAlpha * prevTiltedPulse)
            } else {
                if 0.0 < tilt {
                    var k = tilt * 0.15
                    if 0.40 < k {
                        k = 0.40
                    }
                    tiltedPulse = radiatedPulse + (k * (radiatedPulse - prevTiltedPulse))
                } else {
                    tiltedPulse = radiatedPulse
                }
            }
            prevTiltedPulse = tiltedPulse

            // 無声子音区間での低域濁りヒスノイズを防止し、ディエンファシス積分器 (1 / (1 - 0.95 z^-1)) の直流利得発散を相殺するための高域放射微分整形 (1 - 0.95 z^-1)。
            // 無声励起が過大になると母音のフォルマント共鳴を妨げ耳障りなホワイトノイズが知覚されるため、自然な子音アタックが得られる0.18に調整する。
            let rawNoise = nextRandomFloat()
            let shapedNoise = rawNoise - (0.95 * prevUnvoicedNoise)
            prevUnvoicedNoise = rawNoise
            let unvoicedExcitation = shapedNoise * 0.18

            // 有声励起への高周波息漏れ気流ノイズ（Aspiration）の動的混合
            // 無声子音用の微弱スケール（0.18）に依存せず、有声パルス（振幅~0.85）と聴覚的に釣り合うゲイン（0.40）で
            // 高域通過整形ノイズを混合し、子供や女性の息の多い声（Breathy voice）を声帯励起レベルで明瞭に再現する。
            let asp = glottal.aspirationMix
            let excitationScale: Float = 0.85
            let voicedPulsePart = (1.0 - (asp * 0.5)) * (tiltedPulse * excitationScale)
            let aspirationNoise = shapedNoise * 0.40
            let voicedAspPart = asp * aspirationNoise
            let voicedExcitation = voicedPulsePart + voicedAspPart

            let excitation = (sampleVoiced * voicedExcitation) + (sampleUnvoiced * unvoicedExcitation)
            let inputSignal = excitation * g

            // 2. 全極 AR 声道フィルタの実行
            var arSum: Float = 0.0
            var i = 0
            while i < lpcOrder {
                let interpA = (oneMinusLambda * prevLpcCoeffs[i]) + (lambda * currLpcCoeffs[i])
                arSum += interpA * filterMemory[i]
                i += 1
            }
            var s = inputSignal + arSum

            // 不安定な係数補間により共鳴が急激に増大するのを検知し、状態を初期化して過大バーストノイズを未然に防ぐ。
            if s.isFinite != true {
                s = 0.0
                resetMemory()
                deEmphasisState = 0.0
            } else {
                if 1e4 < abs(s) {
                    if s < 0.0 {
                        s = -10.0
                    } else {
                        s = 10.0
                    }
                    resetMemory()
                    deEmphasisState = 0.0
                }
            }

            // 3. フィルタメモリの更新
            var shiftIdx = lpcOrder - 1
            while 0 < shiftIdx {
                filterMemory[shiftIdx] = filterMemory[shiftIdx - 1]
                shiftIdx -= 1
            }
            filterMemory[0] = s

            // 4. ディエンファシスフィルタ
            var y = s + (deEmphasisCoeff * deEmphasisState)
            if y.isFinite != true {
                y = 0.0
                deEmphasisState = 0.0
            } else {
                deEmphasisState = y
                if abs(deEmphasisState) < 1e-7 {
                    deEmphasisState = 0.0
                }
            }

            // 5. Soft Limiter によるサチュレーション歪み防止
            let absY = abs(y)
            let outSample: Float
            if absY <= 0.8 {
                outSample = y
            } else {
                let excess = absY - 0.8
                let compressed = 0.8 + (0.2 * tanh(excess * 5.0))
                if y < 0.0 {
                    outSample = -compressed
                } else {
                    outSample = compressed
                }
            }

            dst[n] = outSample
            n += 1
        }
    }

    /// フィルタメモリのクリア
    ///
    /// 例外値や過大発振検知時において発散した残響のみを速やかに初期化し、次サンプルの計算を正常に継続させる。
    private func resetMemory() {
        var i = 0
        while i < lpcOrder {
            filterMemory[i] = 0.0
            i += 1
        }
        prevVoicedPulse = 0.0
    }

    /// 複数フレームの一括波形合成
    ///
    /// テスト実行やオフライン音声合成において、フレーム配列から連続PCMバッファを単一呼び出しで安全に生成する。
    @discardableResult
    public func synthesize(frames: [AcousticFrame]) -> [Float] {
        if frames.isEmpty {
            return []
        }
        let totalSamples = frames.count * frameSize
        var output = [Float](repeating: 0.0, count: totalSamples)

        output.withUnsafeMutableBufferPointer { outBuf in
            let basePtr = outBuf.baseAddress!
            var f = 0
            while f < frames.count {
                let dstPtr = basePtr.advanced(by: f * frameSize)
                synthesizeFrame(frame: frames[f], dst: dstPtr)
                f += 1
            }
        }
        return output
    }
}
