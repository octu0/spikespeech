import Foundation

/// 音声処理および音響モデルの基本サンプリング定数
///
/// 言語処理フロントエンドの10ミリ秒フレーム周期、SNN音響モデル、およびニューラルボコーダー間で
/// 時間解像度とフレーム幅の不整合を防ぎ、一貫した基準を提供する。
public struct AudioConfig: Sendable {
    public static let sampleRate: Int = 16000
    public static let hopSize: Int = 160      // 10ミリ秒周期、16kHzで160サンプル
    public static let frameSize: Int = 320    // 20ミリ秒長、16kHzで320サンプル
    public static let melChannels: Int = 64
    public static let acousticInputDim: Int = 256
    public static let pulseChannel: Int = 199
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

/// ボコーダーの話者条件付け特徴量（Speaker Conditioning）
///
/// 将来の参照音声からの話者埋め込み（Speaker Embedding）を受け取るための拡張点。
/// Wave 1 ではゼロベクトルで固定され、将来の話者適応・声色差し替えのためのインターフェースを定義する。
public struct SpeakerConditioning: Sendable, Codable, Equatable {
    public static let defaultDimension: Int = 128
    public var embedding: [Float]

    public var dimension: Int {
        embedding.count
    }

    public init(embedding: [Float] = [Float](repeating: 0.0, count: defaultDimension)) {
        self.embedding = embedding
    }

    public static let zero = SpeakerConditioning()
}

/// 言語 F0 および話者基音周波数に基づく話者プロファイル
///
/// 発音や方言・アクセントの言語処理と連携し、
/// 話者の基本周波数（baseF0）および全体音響エネルギースケーリングを指定する。
public struct VoiceProfile: Sendable, Codable, Equatable {
    public let name: String
    /// 話者の絶対基音周波数 [Hz]
    /// 句成分やアクセントなどの相対抑揚はこの絶対基音に乗算される
    public var baseF0: Float
    /// 全体音響エネルギースケーリング
    public var energyScale: Float

    public init(
        name: String,
        baseF0: Float = 220.0,
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
        if safeEnergy < 0.0 {
            safeEnergy = 0.0
        }
        if 5.0 < safeEnergy {
            safeEnergy = 5.0
        }

        self.name = name
        self.baseF0 = safeF0
        self.energyScale = safeEnergy
    }

    /// 標準的な女性声プロファイル（baseF0 220Hz）
    public static let female = VoiceProfile(
        name: "female",
        baseF0: 220.0,
        energyScale: 1.00
    )

    /// 成人男性声プロファイル（baseF0 120Hz）
    public static let male = VoiceProfile(
        name: "male",
        baseF0: 120.0,
        energyScale: 1.00
    )

    /// 中性的な声プロファイル（baseF0 170Hz）
    public static let neutral = VoiceProfile(
        name: "neutral",
        baseF0: 170.0,
        energyScale: 1.00
    )

    /// 子供・高音ボイスプロファイル（baseF0 300Hz）
    public static let child = VoiceProfile(
        name: "child",
        baseF0: 300.0,
        energyScale: 1.00
    )

    /// 重低音男性ボイスプロファイル（超低域 baseF0 95Hz）
    public static let deepMale = VoiceProfile(
        name: "deepMale",
        baseF0: 95.0,
        energyScale: 1.00
    )

    /// 既定プロファイル
    public static let `default` = female

    /// プリセット名から対応するプロファイルを取得
    public static func preset(named name: String) -> VoiceProfile {
        let lower = name.lowercased()
        switch lower {
        case "female", "woman":
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

