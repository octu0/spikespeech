import Foundation

/// 音声処理および音響モデルの基本サンプリング定数
///
/// 言語処理フロントエンドの10ミリ秒フレーム周期、SNN音響モデル、およびDSPボコーダー間で
/// 時間解像度とフレーム幅の不整合を防ぎ、一貫した基準を提供する。
public struct AudioConfig: Sendable {
    public static let sampleRate: Int = 16000
    public static let hopSize: Int = 160      // 10ミリ秒周期、16kHzで160サンプル
    public static let frameSize: Int = 320    // 20ミリ秒長、16kHzで320サンプル
    public static let lpcOrder: Int = 16
    public static let melChannels: Int = 64
}

/// ホルマント共鳴周波数および帯域幅設定 (Klatt型カスケード音響管共鳴モデル)
///
/// 音響音声学（Fant音響管理論）に基づく声道伝達特性の4共鳴極 (F1〜F4) および
/// 共鳴尖鋭度を司る帯域幅 (B1〜B4) を Hz 単位で物理的に規定する。
public struct FormantConfig: Sendable, Equatable, Codable {
    public var f1: Float
    public var b1: Float
    public var f2: Float
    public var b2: Float
    public var f3: Float
    public var b3: Float
    public var f4: Float
    public var b4: Float

    public init(
        f1: Float, b1: Float,
        f2: Float, b2: Float,
        f3: Float, b3: Float,
        f4: Float, b4: Float
    ) {
        self.f1 = f1
        self.b1 = b1
        self.f2 = f2
        self.b2 = b2
        self.f3 = f3
        self.b3 = b3
        self.f4 = f4
        self.b4 = b4
    }
}


/// 形態素解析における品詞分類
///
/// Viterbi探索の連接コスト行列におけるインデックス参照を高速化し、
/// 助詞置換やアクセント句境界の判定を分岐テーブルで即座に完了させる。
public enum PartOfSpeech: UInt8, Sendable, CaseIterable, Codable {
    case noun          = 0 // 名詞
    case verb          = 1 // 動詞
    case adjective     = 2 // 形容詞
    case adverb        = 3 // 副詞
    case particle      = 4 // 助詞
    case auxiliaryVerb = 5 // 助動詞
    case prefix        = 6 // 接頭辞
    case suffix        = 7 // 接尾辞
    case conjunction   = 8 // 接続詞
    case interjection  = 9 // 感嘆詞
    case symbol        = 10 // 記号
    case unknown       = 11 // 未知語
}

/// 音素の音響生理学的カテゴリ
///
/// 有声と無声の判定によるF0生成、標準的な発話継続時間の割り当て、
/// およびSNN音響モデルでの発音特性制御を正確に分岐させる。
public enum PhonemeCategory: UInt8, Sendable {
    case vowel         = 0 // 母音
    case consonant     = 1 // 子音
    case contracted    = 2 // 拗音および拡張子音
    case geminate      = 3 // 促音
    case nasalSyllable = 4 // 撥音
    case prolonged     = 5 // 長音
    case pause         = 6 // ポーズおよび無音
}

/// 東京方言アクセントの高低トーン
///
/// 日本語ピッチアクセントの基本律である1拍目と2拍目の高低反転や核直後の下降を
/// モーラ単位で決定論的に保持し、F0輪郭生成のステップ入力とする。
public enum AccentTone: UInt8, Sendable {
    case low  = 0
    case high = 1
}

/// 音素トークン構造体
///
/// 音素ID、音素記号、生理学的カテゴリ、および10ミリ秒単位のフレーム継続時間を
/// 密結合で保持し、長さ調整機構での音響フレーム展開を確実にする。
public struct PhonemeToken: Sendable, Equatable {
    public let id: Int
    public let symbol: String
    public let category: PhonemeCategory
    public var durationFrames: Int

    public init(id: Int, symbol: String, category: PhonemeCategory, durationFrames: Int = 0) {
        self.id = id
        self.symbol = symbol
        self.category = category
        self.durationFrames = durationFrames
    }
}

/// モーラ構造体
///
/// 日本語のリズムの等時性とピッチ変化は音素単体ではなくモーラ単位で生起するため、
/// ピッチトーンやアクセント核情報をモーラに紐付けて管理する。
public struct MoraToken: Sendable, Equatable {
    public let text: String
    public var phonemes: [PhonemeToken]
    public var tone: AccentTone
    public var isAccentKernel: Bool
    public var pitchF0: Float

    public init(
        text: String,
        phonemes: [PhonemeToken],
        tone: AccentTone = .low,
        isAccentKernel: Bool = false,
        pitchF0: Float = 0.0
    ) {
        self.text = text
        self.phonemes = phonemes
        self.tone = tone
        self.isAccentKernel = isAccentKernel
        self.pitchF0 = pitchF0
    }

    /// モーラ全体の総継続フレーム数
    public var totalDurationFrames: Int {
        var sum = 0
        var i = 0
        while i < phonemes.count {
            sum += phonemes[i].durationFrames
            i += 1
        }
        return sum
    }
}

/// アクセント句構造体
///
/// 東京方言において1つのアクセント句内に高音部が1箇所しか存在しない韻律的制約に基づき、
/// フレーズ境界ごとにピッチの立ち上がり成分を計算する。
public struct AccentPhrase: Sendable, Equatable {
    public var moras: [MoraToken]
    public var pauseAfter: Bool
    public var pauseDurationFrames: Int
    /// 疑問文または問いかけ上昇調フラグ
    public var isQuestion: Bool

    public init(
        moras: [MoraToken],
        pauseAfter: Bool = false,
        pauseDurationFrames: Int = 0,
        isQuestion: Bool = false
    ) {
        self.moras = moras
        self.pauseAfter = pauseAfter
        self.pauseDurationFrames = pauseDurationFrames
        self.isQuestion = isQuestion
    }
}

/// SNN 音響モデルへの入力となる言語および韻律特徴量
///
/// 言語処理フロントエンドとSNN音響モデル間のインターフェース契約を満たし、
/// 音素ID列、継続時間フレーム数、目標F0輪郭、有声マスクを一括して供給する。
public struct LinguisticFeatures: Sendable, Equatable {
    public let phoneIds: [Int32]
    public let durations: [Int32]
    public let f0Contour: [Float]
    public let voicedFlags: [Float]
    public let energyContour: [Float]
    public let totalFrames: Int

    public init(
        phoneIds: [Int32],
        durations: [Int32],
        f0Contour: [Float],
        voicedFlags: [Float],
        energyContour: [Float] = [],
        totalFrames: Int
    ) {
        self.phoneIds = phoneIds
        self.durations = durations
        self.f0Contour = f0Contour
        self.voicedFlags = voicedFlags
        self.energyContour = energyContour
        self.totalFrames = totalFrames
    }
}

/// 声帯音源（Source）の生理音響パラメータ
///
/// 声門容積速度波形（Rosenbergパルス）の開口率、閉鎖急峻度、および呼気息漏れ比率を
/// 物理的に制御し、ピッチ（F0）とは独立した声帯個性を決定する。
public struct GlottalSource: Sendable, Codable, Equatable {
    /// 声門開口時間比率 (Open Quotient: n1 + n2)
    /// 男性: ~0.42 (引き締まった声帯振動), 女性: ~0.55, 子供: ~0.62, 重低音: ~0.38
    public var openQuotient: Float

    /// 声門閉口急峻度比率 (Return Quotient: n2 / (n1 + n2))
    /// 男性: ~0.10 (急峻な声門閉鎖により高次倍音を豊かに励振), 女性: ~0.16, 子供: ~0.18, 重低音: ~0.08
    public var returnQuotient: Float

    /// 有声励起に混入する高周波息漏れ気流ノイズ比率 [0.0, 1.0]
    /// 子供や女性の息の多い声（Breathy voice）を再現する
    public var aspirationMix: Float

    /// パルススペクトル傾斜 (dB/oct)
    /// 負の値は高域を減衰させて丸い声質とし、正の値はエッジの効いた声を再現する
    public var spectralTilt: Float

    public init(
        openQuotient: Float = 0.55,
        returnQuotient: Float = 0.16,
        aspirationMix: Float = 0.04,
        spectralTilt: Float = -2.0
    ) {
        // 非有限値 (NaN / Inf) や音響学的に破綻する極端な値によるゼロ除算・位相反転を防止するため生理範囲にサニタイズ
        var oq = openQuotient
        if oq.isFinite != true {
            oq = 0.55
        }
        if oq < 0.20 {
            oq = 0.20
        }
        if 0.80 < oq {
            oq = 0.80
        }

        var rq = returnQuotient
        if rq.isFinite != true {
            rq = 0.16
        }
        if rq < 0.05 {
            rq = 0.05
        }
        if 0.40 < rq {
            rq = 0.40
        }

        var asp = aspirationMix
        if asp.isFinite != true {
            asp = 0.04
        }
        if asp < 0.0 {
            asp = 0.0
        }
        if 1.0 < asp {
            asp = 1.0
        }

        var tilt = spectralTilt
        if tilt.isFinite != true {
            tilt = -2.0
        }
        if tilt < -6.0 {
            tilt = -6.0
        }
        if 6.0 < tilt {
            tilt = 6.0
        }

        self.openQuotient = oq
        self.returnQuotient = rq
        self.aspirationMix = asp
        self.spectralTilt = tilt
    }
}

/// 声道伝達（Filter）の物理共鳴パラメータ
///
/// 音響音声学における声道長スケーリング（VTLN）およびフォルマント共鳴帯域幅を
/// Hz 領域で直接制御し、母音の音韻同一性（F2/F1幾何）を保ったまま声道サイズを伸縮する。
public struct VocalTract: Sendable, Codable, Equatable {
    /// 周波数倍率としての声道スケーリング比率 (長い声道ほど小さくフォルマント低域共鳴)
    /// 男性: ~0.85 (成人男性、長い声道), 女性: 1.00 (成人女性基準), 子供: ~1.18 (小児、短い声道), 重低音: ~0.80
    public var lengthScale: Float

    /// フォルマント共鳴帯域幅スケーリング比率 (Q値制御)
    /// 1.0 未満で共鳴ピークが鋭敏化（男性的）、1.0 超で共鳴が平滑化
    public var bandwidthScale: Float

    public init(
        lengthScale: Float = 1.0,
        bandwidthScale: Float = 1.0
    ) {
        // 非有限値や極端な周波数偏向によるフォルマント破綻・0割NaNを防止するためサニタイズ
        var ls = lengthScale
        if ls.isFinite != true {
            ls = 1.0
        }
        if ls < 0.50 {
            ls = 0.50
        }
        if 1.60 < ls {
            ls = 1.60
        }

        var bw = bandwidthScale
        if bw.isFinite != true {
            bw = 1.0
        }
        if bw < 0.20 {
            bw = 0.20
        }
        if 3.00 < bw {
            bw = 3.00
        }

        self.lengthScale = ls
        self.bandwidthScale = bw
    }
}

/// 物理音響 Source-Filter モデルに基づく話者プロファイル
///
/// 発音や方言・アクセントの言語処理とは完全に分離し、
/// 声帯音源（GlottalSource）と声道共鳴（VocalTract）および話者基音周波数を独立して指定する。
public struct VoiceProfile: Sendable, Codable, Equatable {
    public let name: String
    /// 話者の絶対基音周波数 [Hz]
    /// 句成分やアクセントなどの相対抑揚はこの絶対基音に乗算される
    public var baseF0: Float
    /// 声帯音源励起パラメータ
    public var glottal: GlottalSource
    /// 声道長・共鳴パラメータ
    public var tract: VocalTract
    /// 全体音響エネルギースケーリング
    public var energyScale: Float

    public init(
        name: String,
        baseF0: Float = 220.0,
        glottal: GlottalSource = GlottalSource(),
        tract: VocalTract = VocalTract(),
        energyScale: Float = 1.0
    ) {
        var safeF0 = baseF0
        if safeF0.isFinite != true {
            safeF0 = 220.0
        }
        if safeF0 < 50.0 {
            safeF0 = 50.0
        }
        if 800.0 < safeF0 {
            safeF0 = 800.0
        }

        var safeEnergy = energyScale
        if safeEnergy.isFinite != true {
            safeEnergy = 1.0
        }
        if safeEnergy < 0.1 {
            safeEnergy = 0.1
        }
        if 5.0 < safeEnergy {
            safeEnergy = 5.0
        }

        self.name = name
        self.baseF0 = safeF0
        self.glottal = glottal
        self.tract = tract
        self.energyScale = safeEnergy
    }

    /// 標準的な女性声プロファイル（JSUT基準: baseF0 220Hz, OQ 0.55, lengthScale 1.00）
    public static let female = VoiceProfile(
        name: "female",
        baseF0: 220.0,
        glottal: GlottalSource(openQuotient: 0.55, returnQuotient: 0.16, aspirationMix: 0.04, spectralTilt: -2.0),
        tract: VocalTract(lengthScale: 1.00, bandwidthScale: 1.00),
        energyScale: 1.00
    )

    /// 成人男性声プロファイル（baseF0 120Hz, 締まった声帯 OQ 0.42, 急峻閉鎖 RQ 0.10, 長い声道 lengthScale 0.85）
    public static let male = VoiceProfile(
        name: "male",
        baseF0: 120.0,
        glottal: GlottalSource(openQuotient: 0.42, returnQuotient: 0.10, aspirationMix: 0.01, spectralTilt: 0.0),
        tract: VocalTract(lengthScale: 0.85, bandwidthScale: 0.90),
        energyScale: 1.05
    )

    /// 中性的な声プロファイル（baseF0 170Hz, 中庸な声帯・声道設定）
    public static let neutral = VoiceProfile(
        name: "neutral",
        baseF0: 170.0,
        glottal: GlottalSource(openQuotient: 0.48, returnQuotient: 0.13, aspirationMix: 0.03, spectralTilt: -1.0),
        tract: VocalTract(lengthScale: 0.93, bandwidthScale: 0.95),
        energyScale: 1.00
    )

    /// 子供・高音ボイスプロファイル（baseF0 300Hz, 息漏れ aspiration 0.10, 短い声道 lengthScale 1.18）
    public static let child = VoiceProfile(
        name: "child",
        baseF0: 300.0,
        glottal: GlottalSource(openQuotient: 0.62, returnQuotient: 0.18, aspirationMix: 0.10, spectralTilt: -4.0),
        tract: VocalTract(lengthScale: 1.18, bandwidthScale: 1.10),
        energyScale: 0.95
    )

    /// 重低音男性ボイスプロファイル（超低域 baseF0 95Hz, 強く締まった声帯 OQ 0.38, 極めて長い声道 lengthScale 0.80）
    public static let deepMale = VoiceProfile(
        name: "deepMale",
        baseF0: 95.0,
        glottal: GlottalSource(openQuotient: 0.38, returnQuotient: 0.08, aspirationMix: 0.00, spectralTilt: 1.0),
        tract: VocalTract(lengthScale: 0.80, bandwidthScale: 0.85),
        energyScale: 1.10
    )

    /// 既定プロファイル
    public static let `default` = female

    /// プリセット名から対応するプロファイルを取得
    public static func preset(named name: String) -> VoiceProfile {
        let lower = name.lowercased()
        switch lower {
        case "female", "woman", "jsut":
            return .female
        case "male", "man":
            return .male
        case "neutral":
            return .neutral
        case "child", "kid":
            return .child
        case "deep", "deepmale":
            return .deepMale
        default:
            return .female
        }
    }
}

