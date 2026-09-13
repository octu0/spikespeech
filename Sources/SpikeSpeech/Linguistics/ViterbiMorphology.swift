import Foundation

/// 形態素解析辞書エントリ
///
/// クラスによる参照オーバーヘッドを排し、
/// ソート済み連続配列としてメモリ上に配置してキャッシュ効率を高める。
public struct LexiconEntry: Sendable, Equatable {
    public let surface: String
    public let reading: String
    public let pos: PartOfSpeech
    public let accentKernel: Int16
    public let cost: Int16

    public init(
        surface: String,
        reading: String,
        pos: PartOfSpeech,
        accentKernel: Int16 = 0,
        cost: Int16 = 100
    ) {
        self.surface = surface
        self.reading = reading
        self.pos = pos
        self.accentKernel = accentKernel
        self.cost = cost
    }
}

/// 形態素解析結果トークン
///
/// 後続の助詞置換判定および東京方言アクセント高低トーン付与に必要な属性を保持する。
public struct Morpheme: Sendable, Equatable {
    public let surface: String
    public let reading: String
    public let pos: PartOfSpeech
    public let accentKernel: Int16

    public init(surface: String, reading: String, pos: PartOfSpeech, accentKernel: Int16) {
        self.surface = surface
        self.reading = reading
        self.pos = pos
        self.accentKernel = accentKernel
    }
}

/// ラティス展開ノード
///
/// 動的計画法により各ノードに到達する最小コスト経路を線形時間で決定し、
/// バックポインタで最短パスを復元する。
public struct LatticeNode: Sendable {
    public let id: Int
    public let startPos: Int
    public let length: Int
    public let surface: String
    public let reading: String
    public let pos: PartOfSpeech
    public let accentKernel: Int16
    public let nodeCost: Int16
    public var cumulativeCost: Int
    public var bestPrevId: Int

    public init(
        id: Int,
        startPos: Int,
        length: Int,
        surface: String,
        reading: String,
        pos: PartOfSpeech,
        accentKernel: Int16,
        nodeCost: Int16,
        cumulativeCost: Int = Int.max / 2,
        bestPrevId: Int = -1
    ) {
        self.id = id
        self.startPos = startPos
        self.length = length
        self.surface = surface
        self.reading = reading
        self.pos = pos
        self.accentKernel = accentKernel
        self.nodeCost = nodeCost
        self.cumulativeCost = cumulativeCost
        self.bestPrevId = bestPrevId
    }
}

/// Pure Swift 二分探索ラティス構築および品詞間遷移 Viterbi 最短経路形態素解析器
///
/// 外部環境への依存を排除し完全な並行安全性を満たしつつ、
/// 音声合成特化の語彙とアクセント核を直接統合した高精度な形態素分割を行う。
public final class ViterbiMorphology: Sendable {
    private let sortedLexicon: [LexiconEntry]
    private let transitionMatrix: [[Int16]]

    public init() {
        let entries = Self.buildDefaultLexicon()
        // 文字列の共通接頭辞検索を二分探索で実行するため、辞書を表記の昇順で整列する。
        self.sortedLexicon = entries.sorted { a, b in
            a.surface < b.surface
        }
        self.transitionMatrix = Self.buildTransitionMatrix()
    }

    /// 品詞間接続コストを取得する
    ///
    /// 名詞の後に助詞が続くような自然な結合に低コストを与え、
    /// 記号の後に助詞が直結するような文法的不整合を高コストで排除する。
    public func transitionCost(from: PartOfSpeech, to: PartOfSpeech) -> Int16 {
        let r = Int(from.rawValue)
        let c = Int(to.rawValue)
        return transitionMatrix[r][c]
    }

    /// ソート済み語彙配列から指定の表層文字列に完全一致するエントリ群を二分探索する
    ///
    /// 辞書エントリが表記昇順で整列されているため、二分探索で先頭位置を特定し、
    /// 同一表記の異品詞エントリを含めて漏れなく連続区間から抽出する。
    public func findEntries(surface: String) -> [LexiconEntry] {
        var low = 0
        var high = sortedLexicon.count
        while low < high {
            let mid = low + (high - low) / 2
            if sortedLexicon[mid].surface < surface {
                low = mid + 1
            } else {
                high = mid
            }
        }
        var results: [LexiconEntry] = []
        var idx = low
        while idx < sortedLexicon.count && sortedLexicon[idx].surface == surface {
            results.append(sortedLexicon[idx])
            idx += 1
        }
        return results
    }

    /// テキストの指定開始位置から共通接頭辞検索を行う
    ///
    /// 最大語長までの各部分文字列に対して二分探索を適用し、確実に全一致エントリを抽出する。
    public func commonPrefixSearch(in chars: [Character], startOffset: Int) -> [LexiconEntry] {
        let count = chars.count
        if count <= startOffset {
            return []
        }

        var matches: [LexiconEntry] = []
        var maxLen = 10
        let remaining = count - startOffset
        if remaining < maxLen {
            maxLen = remaining
        }

        var len = 1
        while len <= maxLen {
            let sub = String(chars[startOffset ..< (startOffset + len)])
            let found = findEntries(surface: sub)
            matches.append(contentsOf: found)
            len += 1
        }

        return matches
    }

    /// テキストを形態素解析し、最適形態素系列を抽出する
    ///
    /// 可能な単語分割経路の数は入力長に対して増大するため、
    /// 各文字境界での最小累積コストノードのみを更新して多項式時間で最適解を導出する。
    public func tokenize(_ text: String) -> [Morpheme] {
        if text.isEmpty {
            return []
        }

        let chars = Array(text)
        let charCount = chars.count

        // 各文字位置で始まるノードのリストを管理 (0..charCount)
        var nodesStartingAt: [[LatticeNode]] = Array(repeating: [], count: charCount + 1)
        var nextNodeId = 1

        // 1. ラティスノードの構築
        var pos = 0
        while pos < charCount {
            let matches = commonPrefixSearch(in: chars, startOffset: pos)

            var added = false
            for entry in matches {
                let len = entry.surface.count
                if 0 < len && pos + len <= charCount {
                    let node = LatticeNode(
                        id: nextNodeId,
                        startPos: pos,
                        length: len,
                        surface: entry.surface,
                        reading: entry.reading,
                        pos: entry.pos,
                        accentKernel: entry.accentKernel,
                        nodeCost: entry.cost
                    )
                    nodesStartingAt[pos].append(node)
                    nextNodeId += 1
                    added = true
                }
            }

            // 辞書に該当がない場合の未知語ノード生成
            if added != true {
                let char = chars[pos]
                let (unknownPos, unknownReading, unknownCost) = determineUnknownCharProperties(char)
                let singleStr = String(char)
                let node = LatticeNode(
                    id: nextNodeId,
                    startPos: pos,
                    length: 1,
                    surface: singleStr,
                    reading: unknownReading,
                    pos: unknownPos,
                    accentKernel: 0,
                    nodeCost: unknownCost
                )
                nodesStartingAt[pos].append(node)
                nextNodeId += 1
            }

            pos += 1
        }

        // 仮想的な BOS (文頭) ノード
        let bosNode = LatticeNode(
            id: 0,
            startPos: -1,
            length: 0,
            surface: "<BOS>",
            reading: "",
            pos: .symbol,
            accentKernel: 0,
            nodeCost: 0,
            cumulativeCost: 0,
            bestPrevId: -1
        )

        // 2. Viterbi 前向きコスト積算
        // 位置 0 のノード群は BOS から接続
        var p0Idx = 0
        while p0Idx < nodesStartingAt[0].count {
            let nodeCost = Int(nodesStartingAt[0][p0Idx].nodeCost)
            let trans = Int(transitionCost(from: bosNode.pos, to: nodesStartingAt[0][p0Idx].pos))
            nodesStartingAt[0][p0Idx].cumulativeCost = nodeCost + trans
            nodesStartingAt[0][p0Idx].bestPrevId = 0
            p0Idx += 1
        }

        // 全位置の DP 更新
        var allNodesMap: [Int: LatticeNode] = [0: bosNode]
        var startPos = 0
        while startPos < charCount {
            var nIdx = 0
            while nIdx < nodesStartingAt[startPos].count {
                let currentNode = nodesStartingAt[startPos][nIdx]
                allNodesMap[currentNode.id] = currentNode

                let endPos = startPos + currentNode.length
                if endPos <= charCount {
                    // endPos で始まる後続ノード群への遷移を更新
                    var succIdx = 0
                    while succIdx < nodesStartingAt[endPos].count {
                        let succ = nodesStartingAt[endPos][succIdx]
                        let trans = Int(transitionCost(from: currentNode.pos, to: succ.pos))
                        let totalCost = currentNode.cumulativeCost + Int(succ.nodeCost) + trans
                        if totalCost < succ.cumulativeCost {
                            nodesStartingAt[endPos][succIdx].cumulativeCost = totalCost
                            nodesStartingAt[endPos][succIdx].bestPrevId = currentNode.id
                        }
                        succIdx += 1
                    }
                }
                nIdx += 1
            }
            startPos += 1
        }

        // 3. 文末 (EOS) に接続する最適ノードの選定
        var bestLastNode: LatticeNode? = nil
        var minEosCost = Int.max

        // 末尾文字に達した全ノードを走査
        var sPos = 0
        while sPos <= charCount {
            var nIdx = 0
            while nIdx < nodesStartingAt[sPos].count {
                let node = nodesStartingAt[sPos][nIdx]
                allNodesMap[node.id] = node
                if node.startPos + node.length == charCount {
                    let trans = Int(transitionCost(from: node.pos, to: .symbol))
                    let costWithEos = node.cumulativeCost + trans
                    if costWithEos < minEosCost {
                        minEosCost = costWithEos
                        bestLastNode = node
                    }
                }
                nIdx += 1
            }
            sPos += 1
        }

        guard var curr = bestLastNode else {
            return []
        }

        // 4. バックトラックによる最適形態素系列の復元
        var reversedMorphemes: [Morpheme] = []
        while 0 < curr.id {
            reversedMorphemes.append(Morpheme(
                surface: curr.surface,
                reading: curr.reading,
                pos: curr.pos,
                accentKernel: curr.accentKernel
            ))

            let prevId = curr.bestPrevId
            if prevId <= 0 {
                break
            }
            guard let prevNode = allNodesMap[prevId] else {
                break
            }
            curr = prevNode
        }

        return reversedMorphemes.reversed()
    }

    /// 未知文字の文字種に応じた品詞・読み・コストを決定する
    ///
    /// 辞書未登録の文字が出現しても解析を途絶させず、適切なフォールバック読みを付与する。
    private func determineUnknownCharProperties(_ char: Character) -> (PartOfSpeech, String, Int16) {
        let str = String(char)
        let scalar = char.unicodeScalars.first!
        let val = scalar.value

        switch val {
        case 0x3041...0x3096: // ひらがな
            return (.noun, str, 200)
        case 0x30A1...0x30FA: // カタカナ
            if let hiraScalar = UnicodeScalar(val - 0x60) {
                return (.noun, String(Character(hiraScalar)), 200)
            }
            return (.noun, str, 200)
        case 0x4E00...0x9FFF, 0x3400...0x4DBF: // 漢字
            return (.noun, str, 400)
        case 0x30...0x39, 0xFF10...0xFF19: // 数字
            return (.noun, str, 150)
        case 0x41...0x5A, 0x61...0x7A, 0xFF21...0xFF3A, 0xFF41...0xFF5A: // 英字
            return (.noun, str, 300)
        default:
            return (.symbol, str, 250)
        }
    }

    /// 品詞間接続コスト行列を構築する
    ///
    /// 固定サイズで行列を事前展開し、探索ループ内での条件分岐を排除して定数時間で接続コストを評価する。
    private static func buildTransitionMatrix() -> [[Int16]] {
        let count = PartOfSpeech.allCases.count
        var matrix = Array(repeating: Array(repeating: Int16(50), count: count), count: count)

        // 名詞 -> 助詞 (強い結合)
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 0
        // 名詞 -> 助動詞 (だ、です)
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 10
        // 名詞 -> 接尾辞
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.suffix.rawValue)] = 0
        // 名詞 -> 動詞
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.verb.rawValue)] = 30
        // 名詞 -> 記号
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.symbol.rawValue)] = 10

        // 動詞 -> 助動詞 (た、ます)
        matrix[Int(PartOfSpeech.verb.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 0
        // 動詞 -> 助詞 (て、に)
        matrix[Int(PartOfSpeech.verb.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 10
        // 動詞 -> 記号
        matrix[Int(PartOfSpeech.verb.rawValue)][Int(PartOfSpeech.symbol.rawValue)] = 10

        // 形容詞 -> 助動詞 / 助詞
        matrix[Int(PartOfSpeech.adjective.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 0
        matrix[Int(PartOfSpeech.adjective.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 10

        // 接頭辞 -> 名詞
        matrix[Int(PartOfSpeech.prefix.rawValue)][Int(PartOfSpeech.noun.rawValue)] = 0

        // 助詞 -> 助詞 (の、は、など)
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 20
        // 助詞 -> 名詞
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.noun.rawValue)] = 10
        // 助詞 -> 動詞
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.verb.rawValue)] = 10
        // 助詞 -> 記号
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.symbol.rawValue)] = 0

        // 助動詞 -> 助詞 (ですね、でしたか)
        matrix[Int(PartOfSpeech.auxiliaryVerb.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 10
        // 助動詞 -> 助動詞 (でした)
        matrix[Int(PartOfSpeech.auxiliaryVerb.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 10
        // 助動詞 -> 記号
        matrix[Int(PartOfSpeech.auxiliaryVerb.rawValue)][Int(PartOfSpeech.symbol.rawValue)] = 0

        // 記号 -> 助詞 (文法的不自然)
        matrix[Int(PartOfSpeech.symbol.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 500
        // 記号 -> 名詞 / 動詞 (文頭や句点後の新文開始)
        matrix[Int(PartOfSpeech.symbol.rawValue)][Int(PartOfSpeech.noun.rawValue)] = 10
        matrix[Int(PartOfSpeech.symbol.rawValue)][Int(PartOfSpeech.verb.rawValue)] = 20

        return matrix
    }

    /// 日本語 TTS 基本語彙テーブルを構築する
    ///
    /// 外部辞書ファイルへの依存を排除し、正書法テキストの形態素解析、読み、アクセント核を高速に提供する。
    private static func buildDefaultLexicon() -> [LexiconEntry] {
        var list: [LexiconEntry] = []

        // 助詞 (アクセント核 0: 平板結合)
        let particles: [(String, String, Int16)] = [
            ("は", "わ", 0), ("が", "が", 0), ("を", "お", 0), ("に", "に", 0), ("へ", "え", 0),
            ("で", "で", 0), ("と", "と", 0), ("も", "も", 0), ("の", "の", 0), ("から", "から", 0),
            ("まで", "まで", 0), ("より", "より", 0), ("か", "か", 0), ("ね", "ね", 0), ("よ", "よ", 0),
            ("な", "な", 0), ("わ", "わ", 0), ("ぞ", "ぞ", 0), ("ぜ", "ぜ", 0), ("など", "など", 0),
            ("だけ", "だけ", 0), ("ほど", "ほど", 0), ("ばかり", "ばかり", 0), ("でも", "でも", 0),
        ]
        for (s, r, k) in particles {
            list.append(LexiconEntry(surface: s, reading: r, pos: .particle, accentKernel: k, cost: 5))
        }

        // 助動詞
        let auxVerbs: [(String, String, Int16)] = [
            ("だ", "だ", 0), ("です", "です", 1), ("た", "た", 0), ("ます", "ます", 1),
            ("ない", "ない", 1), ("たい", "たい", 1), ("らしい", "らしい", 2),
            ("でした", "でした", 1), ("ません", "ません", 2), ("ました", "ました", 2),
        ]
        for (s, r, k) in auxVerbs {
            list.append(LexiconEntry(surface: s, reading: r, pos: .auxiliaryVerb, accentKernel: k, cost: 5))
        }

        // 指示詞・代名詞
        let pronouns: [(String, String, Int16)] = [
            ("これ", "これ", 0), ("それ", "それ", 0), ("あれ", "あれ", 0), ("どれ", "どれ", 1),
            ("ここ", "ここ", 0), ("そこ", "そこ", 0), ("あそこ", "あそこ", 0), ("どこ", "どこ", 1),
            ("私", "わたし", 0), ("わたし", "わたし", 0), ("僕", "ぼく", 1), ("ぼく", "ぼく", 1),
            ("彼", "かれ", 1), ("彼女", "かのじょ", 1), ("誰", "だれ", 1), ("何", "なに", 1),
        ]
        for (s, r, k) in pronouns {
            list.append(LexiconEntry(surface: s, reading: r, pos: .noun, accentKernel: k, cost: 10))
        }

        // 基本名詞 (単語固有のアクセント核)
        let nouns: [(String, String, Int16)] = [
            ("水", "みず", 0), ("花", "はな", 2), ("本", "ほん", 1), ("先生", "せんせー", 3),
            ("東京", "とーきょー", 0), ("日本", "にほん", 2), ("靴", "くつ", 2), ("映画", "えーが", 0),
            ("猫", "ねこ", 1), ("ねこ", "ねこ", 1), ("桜", "さくら", 0), ("さくら", "さくら", 0),
            ("卵", "たまご", 2), ("たまご", "たまご", 2), ("学校", "がっこー", 0), ("雨", "あめ", 1),
            ("人", "ひと", 2), ("年", "ねん", 1), ("月", "つき", 2), ("日", "ひ", 0),
            ("時", "じ", 1), ("分", "ふん", 1), ("秒", "びょう", 1), ("円", "えん", 1),
            ("個", "こ", 1), ("冊", "さつ", 1), ("匹", "ひき", 1), ("枚", "まい", 1),
            ("車", "くるま", 0), ("電車", "でんしゃ", 0), ("駅", "えき", 1), ("家", "いえ", 2),
            ("部屋", "へや", 2), ("今日", "きょう", 1), ("明日", "あした", 3), ("昨日", "きのう", 2),
            ("母", "はは", 1), ("父", "ちち", 2), ("友達", "ともだち", 0), ("子供", "こども", 0),
            ("朝", "あさ", 1), ("昼", "ひる", 2), ("夜", "よる", 1), ("天気", "てんき", 1),
            ("声", "こえ", 1), ("音", "おと", 2), ("言葉", "ことば", 3), ("時間", "じかん", 0),
            ("世界", "せかい", 1), ("山", "やま", 2), ("川", "かわ", 2), ("海", "うみ", 1),
            ("空", "そら", 1), ("犬", "いぬ", 2), ("鳥", "とり", 0), ("魚", "さかな", 0),
        ]
        for (s, r, k) in nouns {
            list.append(LexiconEntry(surface: s, reading: r, pos: .noun, accentKernel: k, cost: 20))
        }

        // 基本動詞 (終止形および語幹)
        let verbs: [(String, String, Int16)] = [
            ("思う", "おもう", 2), ("思い", "おもい", 2),
            ("買う", "かう", 0), ("買い", "かい", 0),
            ("会う", "あう", 1), ("会い", "あい", 1),
            ("追う", "おう", 0), ("追い", "おい", 0),
            ("行く", "いく", 0), ("行き", "いき", 0),
            ("来る", "くる", 1), ("来", "き", 1),
            ("する", "する", 0), ("し", "し", 0),
            ("見る", "みる", 1), ("見", "み", 1),
            ("聞く", "きく", 0), ("聞き", "きき", 0),
            ("読む", "よむ", 1), ("読み", "よみ", 1),
            ("書く", "かく", 1), ("書き", "かき", 1),
            ("話す", "はなす", 2), ("話し", "はなし", 2),
            ("走る", "はしる", 2), ("走り", "はしり", 2),
            ("歩く", "あるく", 2), ("歩き", "あるき", 2),
            ("食べる", "たべる", 2), ("食べ", "たべ", 2),
            ("飲む", "のむ", 1), ("飲み", "のみ", 1),
            ("愛す", "あいす", 1), ("愛する", "あいする", 3),
        ]
        for (s, r, k) in verbs {
            list.append(LexiconEntry(surface: s, reading: r, pos: .verb, accentKernel: k, cost: 20))
        }

        // 挨拶・副詞・接続詞
        let phrases: [(String, String, PartOfSpeech, Int16)] = [
            ("こんにちは", "こんにちは", .interjection, 0),
            ("こんばんは", "こんばんは", .interjection, 0),
            ("おはよう", "おはよー", .interjection, 2),
            ("ありがとう", "ありがとー", .interjection, 2),
            ("さようなら", "さよーなら", .interjection, 3),
            ("はい", "はい", .interjection, 1),
            ("いいえ", "いいえ", .interjection, 3),
            ("そして", "そして", .conjunction, 0),
            ("しかし", "しかし", .conjunction, 2),
            ("また", "また", .conjunction, 0),
            ("とても", "とても", .adverb, 0),
            ("いつも", "いつも", .adverb, 1),
            ("少し", "すこし", .adverb, 2),
        ]
        for (s, r, pos, k) in phrases {
            list.append(LexiconEntry(surface: s, reading: r, pos: pos, accentKernel: k, cost: 15))
        }

        // 句読点・記号
        let symbols: [(String, String)] = [
            ("、", "<pau>"), ("。", "<sil>"), ("！", "<sil>"), ("？", "<sil>"),
            ("・", "<pau>"), ("…", "<pau>"), ("〜", "ー"), ("～", "ー"),
        ]
        for (s, r) in symbols {
            list.append(LexiconEntry(surface: s, reading: r, pos: .symbol, accentKernel: 0, cost: 5))
        }

        return list
    }
}
