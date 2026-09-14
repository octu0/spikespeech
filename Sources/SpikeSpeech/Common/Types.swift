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

/// 形態素解析における品詞分類
///
/// Viterbi探索の連接コスト行列におけるインデックス参照を高速化し、
/// 助詞置換やアクセント句境界の判定を分岐テーブルで即座に完了させる。
public enum PartOfSpeech: UInt8, Sendable, CaseIterable {
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

/// 声の特徴量および話者プロファイル
///
/// 発音やアクセントの言語処理とは独立して、特定の話者（男性、女性、任意の話者埋め込み）の
/// 声質・ピッチ・フォルマント特性を SNN 音響モデルおよびボコーダーへ注入する。
public struct VoiceProfile: Sendable, Codable, Equatable {
    public let name: String
    public let pitchScale: Float
    public let pitchShift: Float
    public let formantScale: Float
    public let energyScale: Float
    public let speakerEmbedding: [Float]

    public init(
        name: String,
        pitchScale: Float = 1.0,
        pitchShift: Float = 0.0,
        formantScale: Float = 1.0,
        energyScale: Float = 1.0,
        speakerEmbedding: [Float] = []
    ) {
        self.name = name
        self.pitchScale = pitchScale
        self.pitchShift = pitchShift
        self.formantScale = formantScale
        self.energyScale = energyScale
        self.speakerEmbedding = speakerEmbedding
    }

    /// 標準的な女性声プロファイル（基本ピッチ 1.0）
    public static let female = VoiceProfile(
        name: "female",
        pitchScale: 1.0,
        pitchShift: 0.0,
        formantScale: 1.0,
        energyScale: 1.0,
        speakerEmbedding: [
            1.0, 0.0, 0.0, 0.0,
            0.5, 0.2, 0.8, 0.1,
            0.0, 0.0, 0.5, 0.5,
            0.1, 0.9, 0.3, 0.7
        ]
    )

    /// 標準的な男性声プロファイル（低域ピッチ ~0.65、男性的な声道共鳴 0.88）
    public static let male = VoiceProfile(
        name: "male",
        pitchScale: 0.65,
        pitchShift: -40.0,
        formantScale: 0.88,
        energyScale: 1.05,
        speakerEmbedding: [
            0.0, 1.0, 0.0, 0.0,
            -0.5, -0.2, -0.8, -0.1,
            0.5, 0.5, 0.0, 0.0,
            -0.1, -0.9, -0.3, -0.7
        ]
    )

    /// 中性的な声プロファイル
    public static let neutral = VoiceProfile(
        name: "neutral",
        pitchScale: 0.85,
        pitchShift: -15.0,
        formantScale: 0.95,
        energyScale: 1.0,
        speakerEmbedding: [
            0.5, 0.5, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.2, 0.2, 0.2, 0.2,
            0.0, 0.0, 0.0, 0.0
        ]
    )

    /// 子供・高音ボイスプロファイル（高域ピッチ、短い声道共鳴 1.15）
    /// 声優切り替えの多様性を高め、アニメ風・小児風の音声を可能にする
    public static let child = VoiceProfile(
        name: "child",
        pitchScale: 1.35,
        pitchShift: 50.0,
        formantScale: 1.15,
        energyScale: 0.95,
        speakerEmbedding: [
            0.8, -0.4, 0.6, 0.2,
            0.7, 0.5, 0.1, -0.3,
            0.2, -0.1, 0.8, 0.4,
            0.3, 0.6, 0.2, 0.5
        ]
    )

    /// 重低音男性ボイスプロファイル（超低域ピッチ、極めて長い声道共鳴 0.82）
    /// ナレーションや低音男性キャラクター向けに明瞭な音響差分を付与する
    public static let deepMale = VoiceProfile(
        name: "deepMale",
        pitchScale: 0.52,
        pitchShift: -60.0,
        formantScale: 0.82,
        energyScale: 1.10,
        speakerEmbedding: [
            -0.6, 0.9, -0.4, 0.2,
            -0.8, -0.5, -0.9, 0.0,
            0.7, 0.4, -0.3, -0.5,
            -0.4, -0.8, -0.6, -0.9
        ]
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

