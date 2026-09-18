import Foundation

/// 多層 SNN 音響モデル重み構造体
///
/// 単一隠れ層の動的スライスに起因する表現力の制約を解消し、層0の再帰結合ダイナミクスで
/// 音素間の時間的遷移コンテキストを保持しつつ、上位層のフィードフォワード結合と RMSNorm
/// および電流残差加算によって高次スペクトルの階層的特徴表現を安定して獲得する。
public struct SpikingNetworkWeights: Sendable, Codable, Equatable {
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let lifConfig: LIFConfig

    /// 層 0 再帰 LIF 層パラメータ
    public let wIn: [Float]

    /// 層 0 再帰結合重み
    public let wRec: [Float]

    /// 層 0 バイアス
    public let bH: [Float]

    /// 層 1 以降の FF 結合重み
    public let wLayers: [[Float]]

    /// 層 1 以降のバイアス
    public let bHLayers: [[Float]]

    /// 層 1 以降の RMSNorm ゲイン
    public let gammaRMS: [[Float]]

    /// リードアウト射影重み
    public let wOut: [Float]

    /// リードアウト出力バイアス
    public let bOut: [Float]

    /// 学習・獲得された語彙知識（単語表記、読み、品詞、アクセント核、コスト）
    /// なぜ重みとともに記録するか:
    /// ソースコード内に辞書データをハードコードすることを排し、教師データから学習した
    /// 語彙知識をモデルの音響重みと一体化して永続化・更新可能にするため。
    public let lexicon: [LexiconEntry]

    /// 総層数
    public var numLayers: Int {
        return 1 + wLayers.count
    }

    public init(
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        timeSteps: Int,
        lifConfig: LIFConfig,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wLayers: [[Float]] = [],
        bHLayers: [[Float]] = [],
        gammaRMS: [[Float]] = [],
        wOut: [Float],
        bOut: [Float],
        lexicon: [LexiconEntry] = []
    ) {
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.lifConfig = lifConfig
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wLayers = wLayers
        self.bHLayers = bHLayers
        self.gammaRMS = gammaRMS
        self.wOut = wOut
        self.bOut = bOut
        self.lexicon = lexicon
    }

    private enum CodingKeys: String, CodingKey {
        case inputDim, maxHiddenDim, outputDim, timeSteps, lifConfig
        case wIn, wRec, bH, wLayers, bHLayers, gammaRMS, wOut, bOut
        case lexicon
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.inputDim = try container.decode(Int.self, forKey: .inputDim)
        self.maxHiddenDim = try container.decode(Int.self, forKey: .maxHiddenDim)
        self.outputDim = try container.decode(Int.self, forKey: .outputDim)
        self.timeSteps = try container.decode(Int.self, forKey: .timeSteps)
        self.lifConfig = try container.decode(LIFConfig.self, forKey: .lifConfig)
        self.wIn = try container.decode([Float].self, forKey: .wIn)
        self.wRec = try container.decode([Float].self, forKey: .wRec)
        self.bH = try container.decode([Float].self, forKey: .bH)
        self.wLayers = try container.decode([[Float]].self, forKey: .wLayers)
        self.bHLayers = try container.decode([[Float]].self, forKey: .bHLayers)
        self.gammaRMS = try container.decode([[Float]].self, forKey: .gammaRMS)
        self.wOut = try container.decode([Float].self, forKey: .wOut)
        self.bOut = try container.decode([Float].self, forKey: .bOut)
        // なぜ decodeIfPresent を用いるか:
        // 既存の重みファイルに lexicon フィールドが含まれていない場合でも後方互換性を保ち、安全に空配列で初期化するため。
        switch try container.decodeIfPresent([LexiconEntry].self, forKey: .lexicon) {
        case .some(let lex):
            self.lexicon = lex
        case .none:
            self.lexicon = []
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(inputDim, forKey: .inputDim)
        try container.encode(maxHiddenDim, forKey: .maxHiddenDim)
        try container.encode(outputDim, forKey: .outputDim)
        try container.encode(timeSteps, forKey: .timeSteps)
        try container.encode(lifConfig, forKey: .lifConfig)
        try container.encode(wIn, forKey: .wIn)
        try container.encode(wRec, forKey: .wRec)
        try container.encode(bH, forKey: .bH)
        try container.encode(wLayers, forKey: .wLayers)
        try container.encode(bHLayers, forKey: .bHLayers)
        try container.encode(gammaRMS, forKey: .gammaRMS)
        try container.encode(wOut, forKey: .wOut)
        try container.encode(bOut, forKey: .bOut)
        try container.encode(lexicon, forKey: .lexicon)
    }

    /// 語彙知識を付与した新しい重みインスタンスを生成する
    /// なぜ不変構造体のコピーとして返すか:
    /// 並行安全性（Sendable）を維持しながら、学習済み語彙知識を動的に重みへ結合するため。
    public func withLexicon(_ newLexicon: [LexiconEntry]) -> SpikingNetworkWeights {
        return SpikingNetworkWeights(
            inputDim: self.inputDim,
            maxHiddenDim: self.maxHiddenDim,
            outputDim: self.outputDim,
            timeSteps: self.timeSteps,
            lifConfig: self.lifConfig,
            wIn: self.wIn,
            wRec: self.wRec,
            bH: self.bH,
            wLayers: self.wLayers,
            bHLayers: self.bHLayers,
            gammaRMS: self.gammaRMS,
            wOut: self.wOut,
            bOut: self.bOut,
            lexicon: newLexicon
        )
    }

    /// 出力層バイアスを置換した新しい重みインスタンスを生成する
    /// なぜ実音声平均対数 Mel スペクトルで初期化するか:
    /// SNN の出力基準値を実音声エネルギーレベルに一致させ、絶対対数 Mel への直接回帰を加速するため。
    public func withBOut(_ newBOut: [Float]) -> SpikingNetworkWeights {
        return SpikingNetworkWeights(
            inputDim: self.inputDim,
            maxHiddenDim: self.maxHiddenDim,
            outputDim: self.outputDim,
            timeSteps: self.timeSteps,
            lifConfig: self.lifConfig,
            wIn: self.wIn,
            wRec: self.wRec,
            bH: self.bH,
            wLayers: self.wLayers,
            bHLayers: self.bHLayers,
            gammaRMS: self.gammaRMS,
            wOut: self.wOut,
            bOut: newBOut,
            lexicon: self.lexicon
        )
    }

    /// 発火ニューロンから各出力ニューロンへの流出結合重みをメモリ上で連続配置に変換し、推論時の SIMD8 ロードにおけるキャッシュミスを根絶する。
    public func makeWRecT() -> [Float] {
        let hSize = maxHiddenDim
        var recT = [Float](repeating: 0.0, count: hSize * hSize)
        wRec.withUnsafeBufferPointer { srcBuf in
            recT.withUnsafeMutableBufferPointer { dstBuf in
                let src = srcBuf.baseAddress!
                let dst = dstBuf.baseAddress!
                var n = 0
                while n < hSize {
                    let rowOffset = n * hSize
                    var j = 0
                    while j < hSize {
                        dst[(j * hSize) + n] = src[rowOffset + j]
                        j += 1
                    }
                    n += 1
                }
            }
        }
        return recT
    }

    /// 前層の発火ニューロンから次層ニューロンへの重みを連続配置にし、上位層の結合電流計算をイベント駆動 SIMD8 疎加算で完結させる。
    public func makeWLayersT() -> [[Float]] {
        let hSize = maxHiddenDim
        var layersT: [[Float]] = []
        layersT.reserveCapacity(wLayers.count)

        var l = 0
        while l < wLayers.count {
            var layerT = [Float](repeating: 0.0, count: hSize * hSize)
            wLayers[l].withUnsafeBufferPointer { srcBuf in
                layerT.withUnsafeMutableBufferPointer { dstBuf in
                    let src = srcBuf.baseAddress!
                    let dst = dstBuf.baseAddress!
                    var n = 0
                    while n < hSize {
                        let rowOffset = n * hSize
                        var j = 0
                        while j < hSize {
                            dst[(j * hSize) + n] = src[rowOffset + j]
                            j += 1
                        }
                        n += 1
                    }
                }
            }
            layersT.append(layerT)
            l += 1
        }
        return layersT
    }

    /// 最終層の有効発火ニューロンの全出力チャンネルへの結合重みをメモリ上で連続化し、SIMD8 疎射影による高スループットなスペクトル再構成を行う。
    public func makeWOutT() -> [Float] {
        let hSize = maxHiddenDim
        var outT = [Float](repeating: 0.0, count: hSize * outputDim)
        wOut.withUnsafeBufferPointer { srcBuf in
            outT.withUnsafeMutableBufferPointer { dstBuf in
                let src = srcBuf.baseAddress!
                let dst = dstBuf.baseAddress!
                var c = 0
                while c < outputDim {
                    let rowOffset = c * hSize
                    var k = 0
                    while k < hSize {
                        dst[(k * outputDim) + c] = src[rowOffset + k]
                        k += 1
                    }
                    c += 1
                }
            }
        }
        return outT
    }

    /// 外部事前学習重みファイルが存在しない環境でも、スケーリング則に準拠した安定した多層重みを決定論的に初期化し、再現性のある推論および学習を可能にする。
    public static func randomWeights(
        inputDim: Int = 128,
        maxHiddenDim: Int = 1024,
        outputDim: Int = 80,
        timeSteps: Int = 4,
        numLayers: Int = 2,
        lifConfig: LIFConfig = LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.1),
        seed: UInt64 = 42,
        lexicon: [LexiconEntry] = []
    ) -> SpikingNetworkWeights {
        return standardInit(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            numLayers: numLayers,
            lifConfig: lifConfig,
            seed: seed,
            lexicon: lexicon
        )
    }

    /// 標準多層初期化メソッド
    public static func standardInit(
        inputDim: Int = 128,
        maxHiddenDim: Int = 1024,
        outputDim: Int = 80,
        timeSteps: Int = 4,
        numLayers: Int = 2,
        lifConfig: LIFConfig = LIFConfig(beta: 0.8, vTh: 1.0, alpha: 2.0, rho: 0.85, gamma: 0.1),
        seed: UInt64 = 42,
        lexicon: [LexiconEntry] = []
    ) -> SpikingNetworkWeights {
        var rngState = seed
        let scaleIn = sqrt(2.0 / Float(inputDim))
        let scaleRec = 0.1 / sqrt(Float(maxHiddenDim))
        let scaleLayer = sqrt(2.0 / Float(maxHiddenDim))
        let scaleOut = sqrt(2.0 / Float(maxHiddenDim))

        func nextUniform(scale: Float) -> Float {
            rngState ^= rngState << 13
            rngState ^= rngState >> 7
            rngState ^= rngState << 17
            let u01 = Float(rngState & 0x00FFFFFF) / Float(0x01000000)
            return (u01 * 2.0 - 1.0) * scale
        }

        var wIn = [Float](repeating: 0.0, count: maxHiddenDim * inputDim)
        var i = 0
        while i < wIn.count {
            wIn[i] = nextUniform(scale: scaleIn)
            i += 1
        }

        var wRec = [Float](repeating: 0.0, count: maxHiddenDim * maxHiddenDim)
        i = 0
        while i < wRec.count {
            wRec[i] = nextUniform(scale: scaleRec)
            i += 1
        }

        let bH = [Float](repeating: 0.0, count: maxHiddenDim)

        var wLayers: [[Float]] = []
        var bHLayers: [[Float]] = []
        var gammaRMS: [[Float]] = []

        let safeLayers = max(1, numLayers)
        var l = 1
        while l < safeLayers {
            var wLayer = [Float](repeating: 0.0, count: maxHiddenDim * maxHiddenDim)
            var j = 0
            while j < wLayer.count {
                wLayer[j] = nextUniform(scale: scaleLayer)
                j += 1
            }
            wLayers.append(wLayer)
            bHLayers.append([Float](repeating: 0.0, count: maxHiddenDim))
            gammaRMS.append([Float](repeating: 1.0, count: maxHiddenDim))
            l += 1
        }

        var wOut = [Float](repeating: 0.0, count: outputDim * maxHiddenDim)
        i = 0
        while i < wOut.count {
            wOut[i] = nextUniform(scale: scaleOut)
            i += 1
        }

        let bOut = [Float](repeating: 0.0, count: outputDim)

        return SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: wIn,
            wRec: wRec,
            bH: bH,
            wLayers: wLayers,
            bHLayers: bHLayers,
            gammaRMS: gammaRMS,
            wOut: wOut,
            bOut: bOut,
            lexicon: lexicon
        )
    }

    /// JSON 保存
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// JSON 復元
    public static func load(from url: URL) throws -> SpikingNetworkWeights {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(SpikingNetworkWeights.self, from: data)
    }
}
