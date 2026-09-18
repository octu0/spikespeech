import Foundation

/// カスケード共鳴器ボコーダー用音響パラメータフレーム
///
/// 音響音声学（Fant音響管理論およびKlatt共鳴モデル）に基づき、
/// 4共鳴極フォルマント設定、RMS音響ゲイン、基本周波数F0、および有声度を一括保持する。
public struct ResonatorFrame: Sendable, Equatable {
    public var formants: FormantConfig
    public var gain: Float
    public var pitchF0: Float
    public var voiced: Float

    public init(
        formants: FormantConfig,
        gain: Float,
        pitchF0: Float,
        voiced: Float
    ) {
        self.formants = formants
        self.gain = gain
        self.pitchF0 = pitchF0
        self.voiced = voiced
    }
}

/// 64音素ごとの生理学的標準フォルマント周波数および帯域幅テーブル
///
/// 音響音声学における日本人標準話者の調音位置（舌高・舌前後・円唇性）に基づき、
/// 各音素および後続母音ローカスに対応する物理共鳴周波数を決定論的に提供する。
public enum PhonemeFormantTable: Sendable {

    /// 音素 ID および後続母音から標準ホルマント設定を取得
    public static func formantConfig(for phoneId: Int, nextVowelId: Int? = nil) -> FormantConfig {
        switch phoneId {
        case 5: // /a/: 低舌・後舌母音
            // なぜ高次帯域幅 (B3/B4) を適正化するか: 声道壁粘膜の音響損失と頭蓋組織吸収を忠実に反映し、鋭すぎるQ共鳴（ボコーダー特有の金属的リンギング・鼻声感）を解消するため
            return FormantConfig(f1: 800.0, b1: 110.0, f2: 1300.0, b2: 140.0, f3: 2600.0, b3: 220.0, f4: 3500.0, b4: 300.0)
        case 6: // /i/: 高舌・前舌母音
            return FormantConfig(f1: 300.0, b1: 90.0, f2: 2300.0, b2: 140.0, f3: 3000.0, b3: 240.0, f4: 3800.0, b4: 320.0)
        case 7: // /u/: 高舌・後舌円唇母音
            return FormantConfig(f1: 350.0, b1: 95.0, f2: 1200.0, b2: 135.0, f3: 2500.0, b3: 220.0, f4: 3600.0, b4: 300.0)
        case 8: // /e/: 中舌・前舌母音
            return FormantConfig(f1: 500.0, b1: 100.0, f2: 1900.0, b2: 130.0, f3: 2700.0, b3: 220.0, f4: 3700.0, b4: 300.0)
        case 9: // /o/: 中舌・後舌円唇母音
            return FormantConfig(f1: 500.0, b1: 100.0, f2: 900.0, b2: 120.0, f3: 2600.0, b3: 220.0, f4: 3500.0, b4: 300.0)
        case 13: // /n/: 歯茎鼻音
            return FormantConfig(f1: 280.0, b1: 130.0, f2: 1500.0, b2: 190.0, f3: 2500.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        case 15: // /m/: 両唇鼻音
            return FormantConfig(f1: 280.0, b1: 130.0, f2: 1000.0, b2: 190.0, f3: 2300.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        case 24: // /N/ (ん): 口蓋垂鼻音・音節鼻音
            return FormantConfig(f1: 250.0, b1: 150.0, f2: 1200.0, b2: 200.0, f3: 2400.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        case 16: // /y/: 硬口蓋半母音
            return FormantConfig(f1: 300.0, b1: 95.0, f2: 2200.0, b2: 135.0, f3: 2900.0, b3: 220.0, f4: 3600.0, b4: 300.0)
        case 17: // /r/: 歯茎弾き音
            return FormantConfig(f1: 400.0, b1: 120.0, f2: 1350.0, b2: 150.0, f3: 2100.0, b3: 220.0, f4: 3300.0, b4: 300.0)
        case 18: // /w/: 両唇軟口蓋半母音
            return FormantConfig(f1: 320.0, b1: 100.0, f2: 800.0, b2: 130.0, f3: 2400.0, b3: 220.0, f4: 3500.0, b4: 300.0)
        case 19: // /g/: 軟口蓋有声破裂音
            return FormantConfig(f1: 320.0, b1: 130.0, f2: 1900.0, b2: 180.0, f3: 2500.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        case 20: // /z/: 歯茎有声摩擦音
            return FormantConfig(f1: 350.0, b1: 150.0, f2: 1600.0, b2: 200.0, f3: 2800.0, b3: 250.0, f4: 3800.0, b4: 320.0)
        case 21: // /d/: 歯茎有声破裂音
            return FormantConfig(f1: 300.0, b1: 130.0, f2: 1700.0, b2: 180.0, f3: 2600.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        case 22: // /b/: 両唇有声破裂音
            return FormantConfig(f1: 300.0, b1: 130.0, f2: 1000.0, b2: 180.0, f3: 2400.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        case 36: // /j/: 歯茎硬口蓋有声破擦音
            return FormantConfig(f1: 320.0, b1: 130.0, f2: 2100.0, b2: 180.0, f3: 2800.0, b3: 240.0, f4: 3700.0, b4: 300.0)
        case 10: // /k/: 軟口蓋無声破裂音（後続母音に応じたF2ローカス追従）
            var kF2: Float = 1900.0
            switch nextVowelId {
            case .some(5): kF2 = 1600.0 // ka
            case .some(6): kF2 = 2300.0 // ki
            case .some(7): kF2 = 1400.0 // ku
            case .some(8): kF2 = 2000.0 // ke
            case .some(9): kF2 = 1200.0 // ko
            default: break
            }
            return FormantConfig(f1: 350.0, b1: 150.0, f2: kF2, b2: 200.0, f3: 2600.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        case 11: // /s/: 歯茎無声摩擦音
            return FormantConfig(f1: 300.0, b1: 250.0, f2: 1600.0, b2: 310.0, f3: 4500.0, b3: 375.0, f4: 6200.0, b4: 500.0)
        case 12: // /t/: 歯茎無声破裂音
            return FormantConfig(f1: 300.0, b1: 150.0, f2: 1700.0, b2: 200.0, f3: 2800.0, b3: 250.0, f4: 3700.0, b4: 320.0)
        case 14: // /h/: 声門摩擦音（後続母音のフォルマント形状をそのまま通過）
            switch nextVowelId {
            case .some(let nv):
                return formantConfig(for: nv)
            case .none:
                return FormantConfig(f1: 500.0, b1: 130.0, f2: 1500.0, b2: 190.0, f3: 2500.0, b3: 250.0, f4: 3500.0, b4: 320.0)
            }
        case 23: // /p/: 両唇無声破裂音
            return FormantConfig(f1: 300.0, b1: 150.0, f2: 1000.0, b2: 200.0, f3: 2400.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        case 27: // /sh/: 歯茎硬口蓋無声摩擦音
            return FormantConfig(f1: 300.0, b1: 225.0, f2: 2000.0, b2: 275.0, f3: 3200.0, b3: 310.0, f4: 4600.0, b4: 440.0)
        case 28: // /ch/: 歯茎硬口蓋無声破擦音
            return FormantConfig(f1: 300.0, b1: 150.0, f2: 2100.0, b2: 225.0, f3: 3100.0, b3: 275.0, f4: 4400.0, b4: 375.0)
        case 29: // /ts/: 歯茎無声破擦音
            return FormantConfig(f1: 300.0, b1: 190.0, f2: 1650.0, b2: 250.0, f3: 4200.0, b3: 350.0, f4: 5800.0, b4: 440.0)
        case 30: // /ky/: 軟口蓋口蓋化無声破裂音
            return FormantConfig(f1: 300.0, b1: 130.0, f2: 2200.0, b2: 180.0, f3: 2900.0, b3: 240.0, f4: 3600.0, b4: 300.0)
        case 31: // /ny/: 歯茎口蓋化鼻音
            return FormantConfig(f1: 280.0, b1: 130.0, f2: 2100.0, b2: 180.0, f3: 2800.0, b3: 240.0, f4: 3600.0, b4: 300.0)
        case 32: // /hy/: 硬口蓋無声摩擦音
            return FormantConfig(f1: 300.0, b1: 150.0, f2: 2200.0, b2: 200.0, f3: 3000.0, b3: 250.0, f4: 3800.0, b4: 320.0)
        case 33: // /my/: 両唇口蓋化鼻音
            return FormantConfig(f1: 280.0, b1: 130.0, f2: 1800.0, b2: 180.0, f3: 2600.0, b3: 240.0, f4: 3500.0, b4: 300.0)
        case 34: // /ry/: 歯茎口蓋化弾き音
            return FormantConfig(f1: 350.0, b1: 110.0, f2: 1950.0, b2: 150.0, f3: 2600.0, b3: 220.0, f4: 3500.0, b4: 300.0)
        case 35: // /gy/: 軟口蓋口蓋化有声破裂音
            return FormantConfig(f1: 300.0, b1: 120.0, f2: 2200.0, b2: 170.0, f3: 2800.0, b3: 240.0, f4: 3600.0, b4: 300.0)
        case 37: // /by/: 両唇口蓋化有声破裂音
            return FormantConfig(f1: 300.0, b1: 120.0, f2: 1800.0, b2: 170.0, f3: 2600.0, b3: 240.0, f4: 3500.0, b4: 300.0)
        case 38: // /py/: 両唇口蓋化無声破裂音
            return FormantConfig(f1: 300.0, b1: 130.0, f2: 1800.0, b2: 180.0, f3: 2600.0, b3: 240.0, f4: 3500.0, b4: 300.0)
        default:
            // 中立声道共鳴設定（広めの帯域幅で開放的な共鳴を維持）
            return FormantConfig(f1: 500.0, b1: 130.0, f2: 1500.0, b2: 190.0, f3: 2500.0, b3: 250.0, f4: 3500.0, b4: 320.0)
        }
    }
}

/// 4段カスケード2次IIR共鳴器（Klatt音響管モデル）音声合成ボコーダー
///
/// 人間の声道音響管モデルに基づき、声帯振動波形および気流雑音を
/// 直列配置した4つの2次IIRデジタル共鳴器（F1〜F4）に通過させることで、
/// 機械音・ブザー音を完全に根絶し、滑らかで自然な人間音声波形を合成する。
public final class CascadeResonatorVocoder: @unchecked Sendable {

    public let sampleRate: Float
    public let hopSize: Int
    public private(set) var glottal: GlottalSource
    public private(set) var tract: VocalTract

    private let rosenbergPulse: RosenbergPulse
    private var rngState: UInt64 = 88172645463325252
    private var bioFluctuation: BiologicalFluctuation
    /// 声門周期ごとの生体ピッチゆらぎ倍率（周期同期ジッター: cycle-synchronous jitter）
    private var cycleJitter: Float = 0.0
    /// 声門周期ごとの生体振幅ゆらぎ倍率（周期同期シマー: cycle-synchronous shimmer）
    private var cycleShimmer: Float = 1.0
    /// 直前サンプルの声門パルス発振位相（周期ラップ検知用）
    private var lastPhase: Float = 0.0

    // 4段共鳴器の遅延状態メモリ（直前2サンプルの出力を保持）
    private var y1_1: Float = 0.0
    private var y1_2: Float = 0.0
    private var y2_1: Float = 0.0
    private var y2_2: Float = 0.0
    private var y3_1: Float = 0.0
    private var y3_2: Float = 0.0
    private var y4_1: Float = 0.0
    private var y4_2: Float = 0.0

    // 高域放射補正および DC カットフィルタ状態メモリ
    private var radY1: Float = 0.0
    private var hpX1: Float = 0.0
    private var hpY1: Float = 0.0
    private var unvoicedHpX1: Float = 0.0

    // 直前フレームの保持（フレーム間パラメータ線形補間用）
    private var prevFrame: ResonatorFrame?

    /// 初期化
    public init(
        sampleRate: Float = 16000.0,
        hopSize: Int = AudioConfig.hopSize,
        glottal: GlottalSource = GlottalSource(),
        tract: VocalTract = VocalTract()
    ) {
        self.sampleRate = sampleRate
        self.hopSize = hopSize
        self.glottal = glottal
        self.tract = tract

        self.rosenbergPulse = RosenbergPulse(sampleRate: sampleRate)
        self.rosenbergPulse.apply(glottal: glottal)
        self.prevFrame = nil
        self.bioFluctuation = BiologicalFluctuation(seed: 88172645463325252)
        self.cycleJitter = 0.0
        self.cycleShimmer = 1.0
        self.lastPhase = 0.0
        self.radY1 = 0.0
        self.hpX1 = 0.0
        self.hpY1 = 0.0
        self.unvoicedHpX1 = 0.0
    }

    /// 声帯生理パラメータの動的適用
    public func apply(glottal: GlottalSource) {
        self.glottal = glottal
        rosenbergPulse.apply(glottal: glottal)
    }

    /// 声道幾何パラメータ（VTLN）の動的適用
    public func apply(tract: VocalTract) {
        self.tract = tract
    }

    /// ボコーダー内部フィルタ状態およびパルス位相のリセット
    public func reset() {
        rosenbergPulse.reset()
        rngState = 88172645463325252
        bioFluctuation = BiologicalFluctuation(seed: 88172645463325252)
        cycleJitter = 0.0
        cycleShimmer = 1.0
        lastPhase = 0.0
        y1_1 = 0.0
        y1_2 = 0.0
        y2_1 = 0.0
        y2_2 = 0.0
        y3_1 = 0.0
        y3_2 = 0.0
        y4_1 = 0.0
        y4_2 = 0.0
        radY1 = 0.0
        hpX1 = 0.0
        hpY1 = 0.0
        unvoicedHpX1 = 0.0
        prevFrame = nil
    }

    /// 高速 Xorshift64 疑似乱数生成器
    @inline(__always)
    private func nextRandomFloat() -> Float {
        rngState ^= (rngState << 13)
        rngState ^= (rngState >> 7)
        rngState ^= (rngState << 17)
        let u = UInt32(truncatingIfNeeded: rngState)
        return (Float(u) * (2.0 / 4294967295.0)) - 1.0
    }

    /// 2次 IIR 共鳴器の1ステップ実行
    ///
    /// なぜ共鳴周波数における正確なピーク利得 1.0 (0dB) 正規化を行うか:
    /// 2次 IIR 共鳴器の1ステップ実行
    ///
    /// Klatt カスケード音響管モデルに準拠し、直流 z = 1 (ω = 0) における
    /// 各共鳴器の伝達関数ゲインを 1.0 (0dB: b0 = 1 - a1 - a2) に正規化する。
    /// 4段直列接続時の総直流ゲイン 1.0 を保ちつつ、各極周波数において自然な
    /// フォルマント共鳴ピーク（Q値に応じたエネルギー）を形成する。
    @inline(__always)
    private func stepResonator(inVal: Float, freq: Float, bw: Float, y1: inout Float, y2: inout Float) -> Float {
        let r = expf(-Float.pi * bw / sampleRate)
        let theta = 2.0 * Float.pi * freq / sampleRate
        let a1 = 2.0 * r * cosf(theta)
        let a2 = -(r * r)
        let b0 = 1.0 - a1 - a2
        let y0 = (b0 * inVal) + (a1 * y1) + (a2 * y2)
        y2 = y1
        y1 = y0
        return y0
    }

    /// 1フレーム分の音声を逐次合成（リアルタイムストリーミング対応）
    @inline(__always)
    public func synthesizeFrame(frame: ResonatorFrame, dst: UnsafeMutablePointer<Float>) {
        var p = prevFrame
        switch p {
        case .none:
            p = frame
            prevFrame = frame
        case .some:
            break
        }
        let prev = p!

        let invHop = 1.0 / Float(hopSize)
        let vtl = tract.lengthScale
        let bwScale = tract.bandwidthScale

        let targetF1 = frame.formants.f1 * vtl
        let targetB1 = frame.formants.b1 * bwScale
        let prevF1 = prev.formants.f1 * vtl
        let prevB1 = prev.formants.b1 * bwScale

        let targetF2 = frame.formants.f2 * vtl
        let targetB2 = frame.formants.b2 * bwScale
        let prevF2 = prev.formants.f2 * vtl
        let prevB2 = prev.formants.b2 * bwScale

        let targetF3 = frame.formants.f3 * vtl
        let targetB3 = frame.formants.b3 * bwScale
        let prevF3 = prev.formants.f3 * vtl
        let prevB3 = prev.formants.b3 * bwScale

        let targetF4 = frame.formants.f4 * vtl
        let targetB4 = frame.formants.b4 * bwScale
        let prevF4 = prev.formants.f4 * vtl
        let prevB4 = prev.formants.b4 * bwScale

        var s = 0
        while s < hopSize {
            let frac = Float(s) * invHop
            let oneMinusFrac = 1.0 - frac

            let f1 = (oneMinusFrac * prevF1) + (frac * targetF1)
            let b1 = (oneMinusFrac * prevB1) + (frac * targetB1)
            let f2 = (oneMinusFrac * prevF2) + (frac * targetF2)
            let b2 = (oneMinusFrac * prevB2) + (frac * targetB2)
            let f3 = (oneMinusFrac * prevF3) + (frac * targetF3)
            let b3 = (oneMinusFrac * prevB3) + (frac * targetB3)
            let f4 = (oneMinusFrac * prevF4) + (frac * targetF4)
            let b4 = (oneMinusFrac * prevB4) + (frac * targetB4)

            let gain = (oneMinusFrac * prev.gain) + (frac * frame.gain)
            let voiced = (oneMinusFrac * prev.voiced) + (frac * frame.voiced)

            var curF0: Float = 0.0
            switch true {
            case prev.pitchF0 <= 0.0 && frame.pitchF0 <= 0.0:
                curF0 = 0.0
            case prev.pitchF0 <= 0.0:
                curF0 = frame.pitchF0
            case frame.pitchF0 <= 0.0:
                curF0 = prev.pitchF0
            default:
                curF0 = (oneMinusFrac * prev.pitchF0) + (frac * frame.pitchF0)
            }
            if curF0 <= 0.0 {
                curF0 = 220.0
            }

            // 完全無音フレームにおける過去の共鳴テール即時フラッシュ
            if gain <= 1e-4 {
                y1_1 = 0.0
                y1_2 = 0.0
                y2_1 = 0.0
                y2_2 = 0.0
                y3_1 = 0.0
                y3_2 = 0.0
                y4_1 = 0.0
                y4_2 = 0.0
                radY1 = 0.0
                hpX1 = 0.0
                hpY1 = 0.0
                unvoicedHpX1 = 0.0
                dst[s] = 0.0
                s += 1
                continue
            }

            // 声帯音源励起信号の生成（声門周期同期ピッチジッター 1.2% + 周期同期生体シマー ±5.0% + 呼気息漏れ気流ピンクノイズ）
            // なぜ声門周期同期（Cycle-Synchronous）ジッター＆シマーにするか:
            // 毎サンプル周波数を急変させると 8kHz 帯域に及ぶ広帯域 FM 位相ノイズ（電気的ザラつき・ブザー音）
            // が重畳されるため。声門パルス周期境界（位相ラップ時）にのみジッターとシマーを更新することで、
            // 単一周期内の倍音位相整合性を保ちながら、健常成人の生体特有の有機的な肉声ゆらぎ（1.2% ジッター、5.0% シマー）を再現する。
            let curPhase = rosenbergPulse.currentPhase
            if curPhase < lastPhase {
                let pinkJitter = bioFluctuation.nextPink()
                cycleJitter = pinkJitter * 0.012
                let whiteShimmer = nextRandomFloat() * 0.035
                let pinkShimmer = bioFluctuation.nextPink() * 0.025
                cycleShimmer = 1.0 + whiteShimmer + pinkShimmer
            }
            lastPhase = curPhase

            let jitterF0 = curF0 * (1.0 + cycleJitter)
            let rawGlottalPulse = rosenbergPulse.nextDerivativeSample(f0: jitterF0)
            let glottalPulse = rawGlottalPulse * cycleShimmer
            let rawNoise = nextRandomFloat()
            let pinkBreath = bioFluctuation.nextPink()
            let breathNoise = (pinkBreath * 0.70) + (rawNoise * 0.30)
            let asp = max(0.04, glottal.aspirationMix)
            // 声門周期同期息漏れ（開口期に同期して呼気乱流が最大化する生理現象の再現）
            let aspMod = 0.5 * (1.0 - cosf(2.0 * Float.pi * curPhase))
            let glottalAspiration = breathNoise * aspMod * asp * 0.80

            let voicedPulsePart = (glottalPulse * 1.80) + glottalAspiration
            let voicedSample = voicedPulsePart * gain * voiced

            // なぜ無声子音信号に高域差分摩擦成分を含めるか:
            // 日本語の摩擦音 /s/, /sh/ および破裂音 /t/, /k/ は 4kHz〜8kHz に集中する高域乱流音響エネルギーを有するため。
            // 高域差分フィルタにより超高域の抜け感を付与し、子音の立ち上がりと明瞭度を確立する。
            let unvoicedNoise = (rawNoise * 0.80) + (pinkBreath * 0.20)
            let unvoicedHigh = unvoicedNoise - (0.60 * unvoicedHpX1)
            unvoicedHpX1 = unvoicedNoise
            let unvoicedSample = unvoicedHigh * gain * (1.0 - voiced) * 0.85

            // 4段カスケード IIR 声道共鳴フィルタリング (F1 -> F2 -> F3 -> F4)
            // 有声音（声帯音源）は F1 から全 4 共鳴管を通過
            let s1 = stepResonator(inVal: voicedSample, freq: f1, bw: b1, y1: &y1_1, y2: &y1_2)
            let s2 = stepResonator(inVal: s1, freq: f2, bw: b2, y1: &y2_1, y2: &y2_2)
            let s3 = stepResonator(inVal: s2, freq: f3, bw: b3, y1: &y3_1, y2: &y3_2)

            // なぜ無声子音を高域共鳴極 F4 のみへ注入するか:
            // F3 と F4 の両段へ直列通過させると、高域（4.5kHz〜6.2kHz）でのカスケード極二重乗算により
            // ゲインが 1000 倍以上に暴走して耳障りなデジタル破綻・過大スパイクを発生させるため。
            // 口腔前方の単一共鳴腔モデル（Klatt Parallel Branch 準拠）として F4 共鳴器へ注入する。
            let in4 = s3 + (unvoicedSample * 0.15)
            let s4 = stepResonator(inVal: in4, freq: f4, bw: b4, y1: &y4_1, y2: &y4_2)

            // 75Hz ハイパス DC カットフィルタ
            // 音源パルス（nextDerivativeSample）はすでに口唇微分（+6dB/oct）を含んでいるため、
            // 追加の口唇微分差分（2重微分）を排し、自然な母音低域フォルマントエネルギーを保持しつつ直流成分のみを遮断する。
            let rHp: Float = 0.971
            let sHp = s4 - hpX1 + (rHp * hpY1)
            hpX1 = s4
            hpY1 = sHp

            let scaledSample = sHp * 0.075

            // デジタル飽和クリッピングを防止するソフトニーコンプレッサー
            var finalSample = scaledSample
            let absVal = abs(finalSample)
            if 0.85 < absVal {
                let excess = absVal - 0.85
                let compressed = 0.85 + (0.15 * tanhf(excess * 5.0))
                if finalSample < 0.0 {
                    finalSample = -compressed
                } else {
                    finalSample = compressed
                }
            }
            dst[s] = finalSample
            s += 1
        }

        prevFrame = frame
    }

    /// 複数フレームの一括波形合成
    public func synthesize(frames: [ResonatorFrame]) -> [Float] {
        if frames.isEmpty {
            return []
        }
        let totalSamples = frames.count * hopSize
        var output = [Float](repeating: 0.0, count: totalSamples)
        output.withUnsafeMutableBufferPointer { outBuf in
            let basePtr = outBuf.baseAddress!
            var f = 0
            while f < frames.count {
                let offsetPtr = basePtr + (f * hopSize)
                synthesizeFrame(frame: frames[f], dst: offsetPtr)
                f += 1
            }
        }
        return output
    }
}
