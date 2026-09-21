import Foundation

/// 個々の音素の Forced Alignment 結果
public struct PhonemeAlignment: Codable, Sendable {
    public let symbol: String
    public let phoneId: Int32
    public let durationFrames: Int

    public init(symbol: String, phoneId: Int32, durationFrames: Int) {
        self.symbol = symbol
        self.phoneId = phoneId
        self.durationFrames = durationFrames
    }
}

/// 単一発話の Forced Alignment 結果
public struct UtteranceAlignment: Codable, Sendable {
    public let utteranceId: String
    public let leadSilenceFrames: Int
    public let trailSilenceFrames: Int
    public let totalSpeechFrames: Int
    public let phonemes: [PhonemeAlignment]

    public init(
        utteranceId: String,
        leadSilenceFrames: Int,
        trailSilenceFrames: Int,
        totalSpeechFrames: Int,
        phonemes: [PhonemeAlignment]
    ) {
        self.utteranceId = utteranceId
        self.leadSilenceFrames = leadSilenceFrames
        self.trailSilenceFrames = trailSilenceFrames
        self.totalSpeechFrames = totalSpeechFrames
        self.phonemes = phonemes
    }
}

/// Forced Alignment 記録の永続化および統計集計器
public final class AlignmentStore: Sendable {
    public init() {}

    /// アライメントファイル (JSON) を読み込み、発話 ID をキーとする辞書として返す
    public static func load(from path: String) throws -> [String: UtteranceAlignment] {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let list = try decoder.decode([UtteranceAlignment].self, from: data)
        var dict: [String: UtteranceAlignment] = [:]
        var i = 0
        while i < list.count {
            let item = list[i]
            dict[item.utteranceId] = item
            i += 1
        }
        return dict
    }

    /// アライメント結果の配列を JSON ファイルとして保存する
    public static func save(_ alignments: [UtteranceAlignment], to path: String) throws {
        let url = URL(fileURLWithPath: path)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        let data = try encoder.encode(alignments)
        try data.write(to: url, options: .atomic)
    }

    /// 全アライメントデータから音素 ID ごとの平均フレーム数を集計する
    /// なぜ学習アライメント統計を推論正本とするか:
    /// 16.0 モーラ固定や等時間割りを完全撤廃し、実音声データで観測された
    /// 各音素の実際の継続時間（母音、子音、促音、撥音）の平均値を推論に適用することで、
    /// 学習と推論で完全に同一の音素時間スケールを実現するため。
    public static func computeAverageDurations(from alignments: [UtteranceAlignment]) -> [Int32: Float] {
        var durationSums: [Int32: Float] = [:]
        var counts: [Int32: Float] = [:]

        var u = 0
        while u < alignments.count {
            let utt = alignments[u]
            var p = 0
            while p < utt.phonemes.count {
                let ph = utt.phonemes[p]
                let pid = ph.phoneId
                let curSum = durationSums[pid] ?? 0.0
                let curCount = counts[pid] ?? 0.0
                durationSums[pid] = curSum + Float(ph.durationFrames)
                counts[pid] = curCount + 1.0
                p += 1
            }
            u += 1
        }

        var averages: [Int32: Float] = [:]
        for (pid, sum) in durationSums {
            let cnt = counts[pid] ?? 1.0
            if 0.0 < cnt {
                var avg = sum / cnt
                if avg < 1.0 {
                    avg = 1.0
                }
                averages[pid] = avg
            }
        }
        return averages
    }
}
