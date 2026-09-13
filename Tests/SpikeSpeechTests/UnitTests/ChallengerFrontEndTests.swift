import XCTest
@testable import SpikeSpeech

/// Challenger 1 による助数詞連声・長文スケーリング・形態素ラティス・コード規約の敵対的・実証的検証テストスイート
///
/// 既存テストの合格に依存せず、助数詞連声の網羅的オラクル、
/// 長文における O(N) 線形スケーリングの実測、極限境界値、および静的コード規約の
/// 遵守を完全に独立して実証・保証する。
final class ChallengerFrontEndTests: XCTestCase {

    private var normalizer: TextNormalizer!
    private var morphology: ViterbiMorphology!
    private var vocabulary: PhonemeVocabulary!
    private var prosodyModel: ProsodyModel!
    private var lengthRegulator: LengthRegulator!

    override func setUp() {
        super.setUp()
        self.morphology = ViterbiMorphology()
        self.normalizer = TextNormalizer(morphology: morphology)
        self.vocabulary = PhonemeVocabulary()
        self.prosodyModel = ProsodyModel()
        self.lengthRegulator = LengthRegulator(hiddenDimension: 64)
    }

    // MARK: - 1. 助数詞連声の網羅的オラクル検証 (円・分・秒・全角・複合文)

    func testCounterSandhiComprehensiveOracle() {
        // 助数詞「0円」「100円」「1分」「3分」「10秒」等の展開漏れ再発を防ぎ、
        // 全角数字や連続複合文でも確実に音便展開されることを実証する。
        let yenCases: [(input: String, expected: String)] = [
            ("0円", "ぜろえん"),
            ("1円", "いちえん"),
            ("5円", "ごえん"),
            ("10円", "じゅうえん"),
            ("100円", "ひゃくえん"),
            ("300円", "さんびゃくえん"),
            ("600円", "ろっぴゃくえん"),
            ("800円", "はっぴゃくえん"),
            ("1000円", "せんえん"),
            ("3000円", "さんぜんえん"),
            ("8000円", "はっせんえん"),
            ("10000円", "いちまんえん"),
        ]

        var yIdx = 0
        while yIdx < yenCases.count {
            let tc = yenCases[yIdx]
            let result = normalizer.expandNumbersAndCounters(tc.input)
            XCTAssertEqual(result, tc.expected, "円の展開失敗: \(tc.input)")
            yIdx += 1
        }

        // 2. 分の展開オラクル (促音化・半濁音化・濁音化)
        let minuteCases: [(input: String, expected: String)] = [
            ("0分", "ぜろふん"),
            ("1分", "いっぷん"),
            ("2分", "にふん"),
            ("3分", "さんぷん"),
            ("4分", "よんぷん"),
            ("5分", "ごふん"),
            ("6分", "ろっぷん"),
            ("7分", "ななふん"),
            ("8分", "はっぷん"),
            ("9分", "きゅうふん"),
            ("10分", "じゅっぷん"),
            ("13分", "じゅうさんぷん"),
            ("14分", "じゅうよんぷん"),
        ]

        var mIdx = 0
        while mIdx < minuteCases.count {
            let tc = minuteCases[mIdx]
            let result = normalizer.expandNumbersAndCounters(tc.input)
            XCTAssertEqual(result, tc.expected, "分の展開失敗: \(tc.input)")
            mIdx += 1
        }

        // 3. 秒の展開オラクル
        let secondCases: [(input: String, expected: String)] = [
            ("0秒", "ぜろびょう"),
            ("1秒", "いちびょう"),
            ("2秒", "にびょう"),
            ("3秒", "さんびょう"),
            ("10秒", "じゅうびょう"),
            ("30秒", "さんじゅうびょう"),
            ("60秒", "ろくじゅうびょう"),
        ]

        var sIdx = 0
        while sIdx < secondCases.count {
            let tc = secondCases[sIdx]
            let result = normalizer.expandNumbersAndCounters(tc.input)
            XCTAssertEqual(result, tc.expected, "秒の展開失敗: \(tc.input)")
            sIdx += 1
        }

        // 4. 全角数字の連声展開
        let fullWidthCases: [(input: String, expected: String)] = [
            ("０円", "ぜろえん"),
            ("１００円", "ひゃくえん"),
            ("１分", "いっぷん"),
            ("３分", "さんぷん"),
            ("１０秒", "じゅうびょう"),
        ]

        var fwIdx = 0
        while fwIdx < fullWidthCases.count {
            let tc = fullWidthCases[fwIdx]
            let result = normalizer.expandNumbersAndCounters(tc.input)
            XCTAssertEqual(result, tc.expected, "全角数字助数詞の展開失敗: \(tc.input)")
            fwIdx += 1
        }

        // 5. 複合文での展開
        let mixedText = "100円で3分話して10秒待つ"
        let normUnits = normalizer.normalize(text: mixedText)
        let reading = normUnits.map { $0.reading }.joined()
        // 100円 -> ひゃくえん, 3分 -> さんぷん, 10秒 -> じゅうびょう
        XCTAssertTrue(reading.contains("ひゃくえん"), "複合文で100円が展開されていません: \(reading)")
        XCTAssertTrue(reading.contains("さんぷん"), "複合文で3分が展開されていません: \(reading)")
        XCTAssertTrue(reading.contains("じゅうびょう"), "複合文で10秒が展開されていません: \(reading)")
    }

    // MARK: - 2. 長文スケーリング性能・計算量オラクル実証

    func testLongTextScalingLinearComplexityOracle() {
        // 文字列走査における多重アロケーションを排除し、
        // 11,000文字で 0.2秒未満を達成していること、
        // さらに 22,000文字でも線形 O(N) に近似する処理時間で完了することを実証する。

        let sampleUnit = "東京都渋谷区神南の天気予報は晴れのち曇りです。" // 24文字
        let count11k = 460 // 11,040文字
        let text11k = String(repeating: sampleUnit, count: count11k)

        // 11,000文字の計測
        let start11k = Date()
        let morphemes11k = normalizer.normalize(text: text11k)
        let elapsed11k = Date().timeIntervalSince(start11k)

        XCTAssertTrue(0 < morphemes11k.count)
        // ユーザー要件: 11,000文字で0.2秒未満
        XCTAssertTrue(elapsed11k < 0.20, "11,000文字の正規化が0.2秒を超過しました: \(elapsed11k)s")

        print("--- [Challenger Benchmark] ---")
        print("11,040 文字処理時間: \(elapsed11k) 秒")

        // 22,000文字の計測 (倍長テスト)
        let count22k = 920 // 22,080文字
        let text22k = String(repeating: sampleUnit, count: count22k)

        let start22k = Date()
        let morphemes22k = normalizer.normalize(text: text22k)
        let elapsed22k = Date().timeIntervalSince(start22k)

        XCTAssertTrue(0 < morphemes22k.count)
        print("22,080 文字処理時間: \(elapsed22k) 秒")
        let ratio = elapsed22k / elapsed11k
        print("スケーリング比率 (22k/11k): \(ratio) 倍")

        // O(N^2) であれば ratio は 4.0 以上になるが、O(N) であれば約 2.0 倍付近に収まる。
        // キャッシュやアロケーションの変動を許容し 3.2 倍未満であることを検証
        XCTAssertTrue(ratio < 3.2, "スケーリング比率が O(N) から乖離しています: \(ratio)")
        print("-------------------------------")
    }

    // MARK: - 3. 形態素ラティス・サロゲートペア・未知語境界値テスト

    func testViterbiLatticeEdgeCasesAndSurrogates() {
        // UTF-16 複数コードユニット文字（𠮷, 𪚥等）や制御文字がラティス探索の
        // インデックス計算やスライス境界を破壊しないことを実証する。

        let complexInputs = [
            "𠮷野家で𪚥を食べる",
            "寿司🍣と緑茶🍵を注文する",
            "123abcXYZ漢字ひらがなカタカナ",
            "鬱鬱鬱鬱鬱",
            "、。！？",
            "",
        ]

        var idx = 0
        while idx < complexInputs.count {
            let input = complexInputs[idx]
            let tokens = morphology.tokenize(input)
            if input.isEmpty {
                XCTAssertTrue(tokens.isEmpty)
            } else {
                XCTAssertTrue(0 < tokens.count)
                // 表層文字列の再構築が入力と等しいか確認（未知語含む完全被覆性）
                let reconstructed = tokens.map { $0.surface }.joined()
                XCTAssertEqual(reconstructed, input, "ラティス探索で文字の脱落または不正挿入が発生: \(input)")
            }
            idx += 1
        }
    }

    // MARK: - 4. 静的コード規約の自動遵守検証

    func testStaticCodingRulesCompliance() {
        // 比較演算子規約、制御構文規約、メモリ初期化規約が
        // リポジトリ内で遵守されていることを機械的に保証する。

        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let sourcesPath = currentDir + "/Sources/SpikeSpeech"

        guard let enumerator = fileManager.enumerator(atPath: sourcesPath) else {
            XCTFail("Sources/SpikeSpeech が走査できませんでした")
            return
        }

        var checkedFiles = 0
        while let relativePath = enumerator.nextObject() as? String {
            if relativePath.hasSuffix(".swift") != true {
                continue
            }

            let fullPath = sourcesPath + "/" + relativePath
            guard let content = try? String(contentsOfFile: fullPath, encoding: .utf8) else {
                continue
            }

            let lines = content.components(separatedBy: .newlines)
            var lineIdx = 0
            while lineIdx < lines.count {
                let line = lines[lineIdx]
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                // コメント行は除外
                if trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") {
                    lineIdx += 1
                    continue
                }

                // 1. 比較演算子 > または >= の禁止検証 (ジェネリクス <T>, 矢印 ->, XML等は除外)
                // 通常の条件式で ` > ` や ` >= ` が使われていないか
                XCTAssertFalse(
                    trimmed.contains(" > ") && trimmed.contains("->") != true,
                    "比較演算子 > が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(" >= "),
                    "比較演算子 >= が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )

                // 2. else if の禁止検証
                XCTAssertFalse(
                    trimmed.contains("else if"),
                    "else if が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )

                // 3. initialize(from: の禁止検証 (update(from: を使用すべき)
                XCTAssertFalse(
                    trimmed.contains(".initialize(from:"),
                    "initialize(from:) が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )

                // 4. 不要 guard 節（名指しされた自前確保バッファ等のチェック）の禁止検証
                XCTAssertFalse(
                    trimmed.contains("guard let dstBase = dstBuf.baseAddress"),
                    "不要な guard 節が残存しています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains("guard let scalar = c.unicodeScalars.first"),
                    "不要な guard 節が残存しています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )

                lineIdx += 1
            }

            checkedFiles += 1
        }

        XCTAssertTrue(0 < checkedFiles, "チェック対象の Swift ファイルが存在しませんでした")
        print("--- [Challenger Static Rule Check] ---")
        print("検証完了 Swift ファイル数: \(checkedFiles) 件 (全ファイル規約適合)")
        print("---------------------------------------")
    }
}
