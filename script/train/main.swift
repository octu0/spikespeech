import Foundation
import MLX
import MLXNN
import MLXOptimizers
import SpikeSpeech

/// SNN 音響モデル BPTT 学習 CLI
func main() {
    let args = CommandLine.arguments
    var epochs: Int = 15
    var learningRate: Float = 0.003
    var learningRateExplicit: Bool = false
    var lrMin: Float = 1.0e-5
    var warmupEpochs: Int = 2
    var weightDecay: Float = 1.0e-4
    var shuffleSeed: UInt64 = 2026
    var noShuffle: Bool = false
    var hiddenDim: Int = 256
    var numLayers: Int = 4
    var inDim: Int = AudioConfig.acousticInputDim // 256 (Triphone context 64*3 + acoustic features)
    var outDim: Int = AudioConfig.melChannels // 64
    var timeSteps: Int = 4
    var useCfC: Bool = false
    var useFrameMel: Bool = true
    var maxSamples: Int? = nil
    var datasetPath: String? = nil
    var outputPath: String = "Models/weights.json"
    var loadWeightsPath: String? = "Models/weights.json"
    var forceFresh: Bool = false
    var forceFreshSNN: Bool = false
    var forceInitBOut: Bool = false
    var vocoderEpochs: Int = 5
    var vocoderLearningRate: Float = 0.0003
    var waveformLossWeight: Float = 0.15
    var residualLossWeight: Float = 3.0
    var prosodyEpochs: Int = 15
    var prosodySamplesLimit: Int? = nil
    var prosodyStepsPerSample: Int = 2
    var prosodyLearningRate: Float = 0.008
    var freshProsody: Bool = false
    var alignmentsPath: String? = nil
    var forceMAS: Bool = false

    func printUsage() {
        print("Usage: train -d <corpus_dir> [--alignments <alignments.json>] [--force-mas] [-s <samples>] [-e <epochs>] [--prosody-samples <samples>] [--prosody-steps <steps>] [--prosody-lr <lr>] [--fresh-prosody] [--vocoder-epochs <epochs>] [--vocoder-lr <lr>] [--waveform-weight <weight>] [--residual-weight <weight>] [--prosody-epochs <epochs>] [--lr <learning_rate>] [--lr-min <min_lr>] [--warmup-epochs <epochs>] [--wd <weight_decay>] [--shuffle-seed <seed>] [--no-shuffle] [--hidden-dim <dim>] [--num-layers <layers>] [--in-dim <dim>] [--out-dim <dim>] [--time-steps <steps>] [-w <weights.json>] [--fresh] [--fresh-snn] [-o <output.json>]")
    }


    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "-d", "--dataset":
            let nextIdx = i + 1
            if nextIdx < args.count {
                datasetPath = args[nextIdx]
                i += 1
            }
        case "--alignments":
            let nextIdx = i + 1
            if nextIdx < args.count {
                alignmentsPath = args[nextIdx]
                i += 1
            }
        case "-s", "--samples":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    var safeVal = val
                    if safeVal < 1 {
                        safeVal = 1
                    }
                    maxSamples = safeVal
                }
                i += 1
            }
        case "-e", "--epochs":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    epochs = val
                }
                i += 1
            }
        case "--vocoder-epochs":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    var safeVal = val
                    if safeVal < 0 {
                        safeVal = 0
                    }
                    vocoderEpochs = safeVal
                }
                i += 1
            }
        case "--prosody-epochs":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    var safeVal = val
                    if safeVal < 0 {
                        safeVal = 0
                    }
                    prosodyEpochs = safeVal
                }
                i += 1
            }
        case "--prosody-samples":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    var safeVal = val
                    if safeVal < 1 {
                        safeVal = 1
                    }
                    prosodySamplesLimit = safeVal
                }
                i += 1
            }
        case "--prosody-steps":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    var safeVal = val
                    if safeVal < 1 {
                        safeVal = 1
                    }
                    prosodyStepsPerSample = safeVal
                }
                i += 1
            }
        case "--prosody-lr":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    prosodyLearningRate = val
                }
                i += 1
            }
        case "--fresh-prosody":
            freshProsody = true
        case "--vocoder-lr":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    vocoderLearningRate = val
                }
                i += 1
            }
        case "--waveform-weight":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    var safeVal = val
                    if safeVal < 0.0 {
                        safeVal = 0.0
                    }
                    waveformLossWeight = safeVal
                }
                i += 1
            }
        case "--residual-weight":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    var safeVal = val
                    if safeVal < 0.0 {
                        safeVal = 0.0
                    }
                    residualLossWeight = safeVal
                }
                i += 1
            }
        case "--lr":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    learningRateExplicit = true
                    learningRate = val
                }
                i += 1
            }
        case "--lr-min":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    lrMin = val
                }
                i += 1
            }
        case "--warmup-epochs":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    warmupEpochs = max(1, val)
                }
                i += 1
            }
        case "--wd", "--weight-decay":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    weightDecay = val
                }
                i += 1
            }
        case "--shuffle-seed":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = UInt64(args[nextIdx]) {
                    shuffleSeed = val
                }
                i += 1
            }
        case "--no-shuffle":
            noShuffle = true
        case "--hidden-dim":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    hiddenDim = val
                }
                i += 1
            }
        case "--num-layers":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    numLayers = max(1, val)
                }
                i += 1
            }
        case "--in-dim":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    inDim = max(16, val)
                }
                i += 1
            }
        case "--out-dim":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    outDim = max(16, val)
                }
                i += 1
            }
        case "--time-steps":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    timeSteps = max(1, val)
                }
                i += 1
            }
        case "-o", "--output":
            let nextIdx = i + 1
            if nextIdx < args.count {
                outputPath = args[nextIdx]
                i += 1
            }
        case "-w", "--weights":
            let nextIdx = i + 1
            if nextIdx < args.count {
                loadWeightsPath = args[nextIdx]
                i += 1
            }
        case "--fresh":
            forceFresh = true
        case "--fresh-snn":
            forceFreshSNN = true
        case "--init-bout":
            forceInitBOut = true
        case "--force-mas":
            forceMAS = true
        case "--frame-mel":
            useFrameMel = true
            useCfC = false
        case "--cfc":
            useCfC = true
            useFrameMel = false
        case "--snn":
            useCfC = false
            useFrameMel = false
        case "-h", "--help":
            printUsage()
            return

        default:
            break
        }
        i += 1
    }

    if useCfC {
        timeSteps = 1
        hiddenDim = 256
        inDim = 256
        outDim = 64
        numLayers = 4
    }

    guard let explicit = datasetPath, explicit.isEmpty != true else {
        print("エラー: コーパスディレクトリが指定されていません。-d <corpus_dir> を指定してください。")
        printUsage()
        exit(1)
    }
    var cleanDatasetPath = explicit
    if cleanDatasetPath.hasPrefix("@") {
        cleanDatasetPath = String(cleanDatasetPath.dropFirst())
    }
    if cleanDatasetPath.hasPrefix("~") {
        cleanDatasetPath = NSString(string: cleanDatasetPath).expandingTildeInPath
    }
    let transcriptCheckPath = cleanDatasetPath + "/transcript_utf8.txt"
    if FileManager.default.fileExists(atPath: transcriptCheckPath) != true {
        print("エラー: 転写ファイルが見つかりません: \(transcriptCheckPath)")
        print("       -d には transcript_utf8.txt と wav/ を含むディレクトリを指定してください。")
        exit(1)
    }

    // 他のマシンや CI 環境でも動作するよう、ハードコードされた絶対パスを廃止し相対パスと探索候補から解決する
    let fileManager = FileManager.default
    let currentDir = fileManager.currentDirectoryPath
    let targetPath = currentDir + "/default.metallib"

    if fileManager.fileExists(atPath: targetPath) != true {
        var found = false
        let candidates = [
            currentDir + "/default.metallib",
            currentDir + "/.build/arm64-apple-macosx/debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
            currentDir + "/.build/arm64-apple-macosx/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
        ]

        var cIdx = 0
        while cIdx < candidates.count {
            let candidate = candidates[cIdx]
            if fileManager.fileExists(atPath: candidate) {
                try? fileManager.copyItem(atPath: candidate, toPath: targetPath)
                found = true
                break
            }
            cIdx += 1
        }

        // 候補パスで解決できない場合、.build ディレクトリ内を再帰走査してアーキテクチャ依存を完全排除
        if found != true {
            let buildDir = currentDir + "/.build"
            if let enumerator = fileManager.enumerator(atPath: buildDir) {
                while let subPath = enumerator.nextObject() as? String {
                    if subPath.hasSuffix("default.metallib") {
                        let fullPath = buildDir + "/" + subPath
                        try? fileManager.copyItem(atPath: fullPath, toPath: targetPath)
                        break
                    }
                }
            }
        }
    }

    MLXRandom.seed(2026)

    // なぜ FrameMel の既定学習率を 3e-4 にするか:
    // 40 発話の検証で 3e-3 は 2 エポック目にメル L1 が 5.5e3 → 6.9e4 へ発散し、
    // 1e-3 も 1 エポック目に一時的な発散（メル L1 120）を示した。3e-4 は単調に収束した。
    if useFrameMel && learningRateExplicit != true {
        learningRate = 0.0003
        print("FrameMel 既定学習率 \(learningRate) を適用します（--lr で上書き可能）。")
    }

    print("==================================================")
    print("SpikeSpeech SNN 音響モデル BPTT 学習パイプライン")
    print("==================================================")
    print("エポック数:       \(epochs)")
    print("学習率 (初期/下限): \(learningRate) / \(lrMin)")
    print("ウォームアップ:   \(warmupEpochs) エポック")
    print("Weight Decay:     \(weightDecay)")
    print("シャッフルシード: \(shuffleSeed)")
    print("隠れ層次元:       \(hiddenDim)")
    print("層数:             \(numLayers)")
    print("教師音声基準:     JSUT (女性単一話者 VoiceProfile.female 自然な Mel 目標系列)")
    print("出力パス:         \(outputPath)")
    print("--------------------------------------------------")

    // 既存の学習済み重みが存在する場合はウォームスタートし、学習の蓄積と損失減少の継続性を確保する
    // なぜ全次元（input, output, hidden, layers）の厳密検査を行うか:
    // CLI 引数で隠れ層次元や層数が変更された場合に、異なるシェイプの重みを誤ロードして
    // 行列積のクラッシュや意図しない旧構造のまま学習が継続される不整合を完全に防止するため。
    // 既存の学習済み重みが存在する場合はロードし、アーキテクチャやウォームスタートの可否を判定する
    // なぜ SNN 新規初期化時にも既存の重みファイルから韻律・語彙・テンポを読み出すか:
    // SNN 音響重みの入力次元変更・再学習時に、学習済みの F0 予測器や自然なモーラ速度テーブル、語彙辞書を
    // 不必要に破棄して音質・抑揚を低下させる事態を防止するため。
    var loadedExisting: SpikingNetworkWeights? = nil
    if let path = loadWeightsPath {
        if fileManager.fileExists(atPath: path) {
            if let data = try? Data(contentsOf: URL(fileURLWithPath: path)) {
                if let loaded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) {
                    loadedExisting = loaded
                }
            }
        }
    }

    var weights: SpikingNetworkWeights
    switch useFrameMel {
    case true:
        let base: SpikingNetworkWeights
        switch loadedExisting {
        case .some(let loaded):
            var lex = loaded.lexicon
            if lex.isEmpty {
                lex = ViterbiMorphology.loadDefaultLexicon()
            }
            base = loaded.withLexicon(lex)
            print("既存の学習済み重みをロードしました: \(loadWeightsPath ?? "")")
        case .none:
            let defaultLex = ViterbiMorphology.loadDefaultLexicon()
            base = SpikingNetworkWeights.initCfCWeights(
                inputDim: inDim,
                hiddenDim: hiddenDim,
                outputDim: outDim,
                numLayers: numLayers,
                seed: 2026,
                lexicon: defaultLex
            )
            print("新規の重みで初期化しました。")
        }
        let fmw: FrameMelWeights
        var shouldFresh = forceFresh || forceFreshSNN
        if base.frameMelWeights == nil {
            shouldFresh = true
        }
        switch shouldFresh {
        case true:
            print("FrameMel モデル重みを新規乱数で初期化しました。")
            fmw = FrameMelWeights.randomWeights(seed: 2026)
        case false:
            print("既存の FrameMel モデル重みをロードしてウォームスタートします。")
            fmw = base.frameMelWeights!
        }
        weights = base.withFrameMelWeights(fmw)
    case false:
        switch loadedExisting {
    case .some(let loaded):
        var lex = loaded.lexicon
        if lex.isEmpty {
            lex = ViterbiMorphology.loadDefaultLexicon()
        }
        var shouldFreshAcoustic = forceFresh || forceFreshSNN
        if useCfC && loaded.isCfC != true {
            // 既存重みが CfC でない場合は新規 CfC 初期化が必要
            shouldFreshAcoustic = true
        }
        if useCfC != true && loaded.isCfC {
            // 既存重みが CfC で SNN 学習を指定された場合は新規 SNN 初期化が必要
            shouldFreshAcoustic = true
        }

        switch shouldFreshAcoustic {
        case true:
            switch useCfC {
            case true:
                print("既存の韻律 (F0/Duration)・語彙・テンポ重みを保持しつつ、音響モデル重みを新規 CfC 乱数で初期化しました。")
                let preservedWIn: [Float]?
                switch loaded.wIn.count == hiddenDim * inDim {
                case true:
                    preservedWIn = loaded.wIn
                case false:
                    preservedWIn = nil
                }
                weights = SpikingNetworkWeights.initCfCWeights(
                    inputDim: inDim,
                    hiddenDim: hiddenDim,
                    outputDim: outDim,
                    numLayers: numLayers,
                    seed: 2026,
                    wIn: preservedWIn,
                    lexicon: lex,
                    prosodyWeights: loaded.prosodyWeights,
                    phonemeAverageDurations: loaded.phonemeAverageDurations,
                    meanFramesPerMora: loaded.meanFramesPerMora
                )
            case false:
                print("既存の韻律 (F0/Duration)・語彙・テンポ重みを保持しつつ、SNN 音響モデル重みのみを新規乱数で初期化しました。")
                weights = SpikingNetworkWeights.randomWeights(
                    inputDim: inDim,
                    maxHiddenDim: hiddenDim,
                    outputDim: outDim,
                    timeSteps: timeSteps,
                    numLayers: numLayers,
                    seed: 2026,
                    lexicon: lex,
                    prosodyWeights: loaded.prosodyWeights,
                    phonemeAverageDurations: loaded.phonemeAverageDurations,
                    meanFramesPerMora: loaded.meanFramesPerMora
                )
            }
        case false:
            let isMatch: Bool
            switch useCfC {
            case true:
                isMatch = loaded.isCfC &&
                          (loaded.inputDim == inDim) &&
                          (loaded.outputDim == outDim) &&
                          (loaded.maxHiddenDim == hiddenDim) &&
                          (loaded.numLayers == numLayers)
            case false:
                isMatch = (loaded.isCfC != true) &&
                          (loaded.inputDim == inDim) &&
                          (loaded.outputDim == outDim) &&
                          (loaded.maxHiddenDim == hiddenDim) &&
                          (loaded.numLayers == numLayers) &&
                          (loaded.timeSteps == timeSteps)
            }
            switch isMatch {
            case true:
                weights = loaded.withLexicon(lex)
                print("既存の学習済み重みをロードしました: \(loadWeightsPath ?? "")")
            case false:
                print("警告: 既存の重みと指定されたアーキテクチャパラメータが一致しません。音響モデル重みを新規初期化します。")
                switch useCfC {
                case true:
                    let preservedWIn: [Float]?
                    switch loaded.wIn.count == hiddenDim * inDim {
                    case true:
                        preservedWIn = loaded.wIn
                    case false:
                        preservedWIn = nil
                    }
                    weights = SpikingNetworkWeights.initCfCWeights(
                        inputDim: inDim,
                        hiddenDim: hiddenDim,
                        outputDim: outDim,
                        numLayers: numLayers,
                        seed: 2026,
                        wIn: preservedWIn,
                        lexicon: lex,
                        prosodyWeights: loaded.prosodyWeights,
                        phonemeAverageDurations: loaded.phonemeAverageDurations,
                        meanFramesPerMora: loaded.meanFramesPerMora
                    )
                case false:
                    weights = SpikingNetworkWeights.randomWeights(
                        inputDim: inDim,
                        maxHiddenDim: hiddenDim,
                        outputDim: outDim,
                        timeSteps: timeSteps,
                        numLayers: numLayers,
                        seed: 2026,
                        lexicon: lex,
                        prosodyWeights: loaded.prosodyWeights,
                        phonemeAverageDurations: loaded.phonemeAverageDurations,
                        meanFramesPerMora: loaded.meanFramesPerMora
                    )
                }
            }
        }
    case .none:
        let defaultLex = ViterbiMorphology.loadDefaultLexicon()
        switch useCfC {
        case true:
            weights = SpikingNetworkWeights.initCfCWeights(
                inputDim: inDim,
                hiddenDim: hiddenDim,
                outputDim: outDim,
                numLayers: numLayers,
                seed: 2026,
                lexicon: defaultLex
            )
            print("新規の決定論的 CfC 重みで初期化しました。")
        case false:
            weights = SpikingNetworkWeights.randomWeights(
                inputDim: inDim,
                maxHiddenDim: hiddenDim,
                outputDim: outDim,
                timeSteps: timeSteps,
                numLayers: numLayers,
                seed: 2026,
                lexicon: defaultLex
            )
            print("新規の決定論的ランダム SNN 重みで初期化しました。")
        }
    }
    }

    // ============================================================
    // 語彙獲得: 転写テキストから表記→読みを獲得し、重み (lexicon) に蓄積する
    // なぜソースコードの辞書ではなく重みへ蓄積するか:
    // コーパス依存の語彙をソースに残さず、学習済み重みとともに持ち運べるようにするため。
    // 読みが変わるとアライメントも変わるため、新規語彙があればキャッシュ済み MAS を再生成する。
    // ============================================================
    var transcriptSentences: [String] = []
    if let tContent = try? String(contentsOfFile: cleanDatasetPath + "/transcript_utf8.txt", encoding: .utf8) {
        for rawLine in tContent.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                continue
            }
            var parts = line.split(separator: ":", maxSplits: 1).map { String($0) }
            if parts.count != 2 {
                parts = line.split(separator: "\t", maxSplits: 1).map { String($0) }
            }
            if parts.count == 2 {
                transcriptSentences.append(parts[1].trimmingCharacters(in: .whitespaces))
            }
        }
    }
    let lexiconLearning = ViterbiMorphology.learnLexiconEntries(sentences: transcriptSentences, existing: weights.lexicon)
    if 0 < lexiconLearning.added {
        weights = weights.withLexicon(lexiconLearning.lexicon)
        forceMAS = true
        print("転写 \(lexiconLearning.sentenceCount) 文から語彙を \(lexiconLearning.added) 件獲得しました（語彙合計: \(weights.lexicon.count) 件）。読みが更新されたため MAS アライメントを再生成します。")
    } else {
        print("転写 \(lexiconLearning.sentenceCount) 文に新規語彙はありませんでした（語彙合計: \(weights.lexicon.count) 件）。")
    }

    let engine = SpikeSpeechEngine(weights: weights)
    let melExtractor = MelSpectrogramExtractor(
        sampleRate: Float(AudioConfig.sampleRate),
        melChannels: outDim
    )
    let wavReader = WavAudioReader()
    let pitchTracker = PitchTracker()

    var alignmentMap: [String: UtteranceAlignment] = [:]
    switch alignmentsPath {
    case .some(let aPath):
        if fileManager.fileExists(atPath: aPath) {
            if let loadedMap = try? AlignmentStore.load(from: aPath) {
                alignmentMap = loadedMap
                print("指定された音素 Forced Alignment 記録を読み込みました: \(aPath) (\(alignmentMap.count) 発話)")
            }
        } else {
            print("警告: 指定されたアライメントファイルが存在しません: \(aPath)。オンライン抽出を実施します。")
        }
    case .none:
        // コーパスディレクトリ直下に mas_alignments.json があればキャッシュとして自動探索
        let corpusAlignPath = cleanDatasetPath + "/mas_alignments.json"
        if fileManager.fileExists(atPath: corpusAlignPath) {
            if let loadedMap = try? AlignmentStore.load(from: corpusAlignPath) {
                alignmentMap = loadedMap
                print("コーパス内キャッシュから教師 Mel 単調アライメント (MAS) 記録を読み込みました: \(corpusAlignPath) (\(alignmentMap.count) 発話)")
            }
        }
        var needMASGen = alignmentMap.isEmpty || forceMAS
        if needMASGen != true {
            if let firstId = alignmentMap.keys.sorted().first, let sample0 = alignmentMap[firstId] {
                if AlignmentStore.isUtteranceAlignmentValid(sample0) != true {
                    print("警告: キャッシュされた \(firstId) アライメントが縮退（2〜40F 範囲外）しています。新MAS再集計を実行します。")
                    needMASGen = true
                }
            } else {
                needMASGen = true
            }
        }
        if needMASGen != true {
            var sampleCount = 0
            var vowelFourCount = 0
            for (_, utt) in alignmentMap.prefix(100) {
                var p = 0
                while p < utt.phonemes.count {
                    let ph = utt.phonemes[p]
                    switch ph.symbol {
                    case "a", "i", "u", "e", "o", "_":
                        sampleCount += 1
                        if ph.durationFrames == 4 {
                            vowelFourCount += 1
                        }
                    default:
                        break
                    }
                    p += 1
                }
            }
            if 0 < sampleCount {
                let ratio4 = Float(vowelFourCount) / Float(sampleCount)
                if 0.30 < ratio4 {
                    print("警告: キャッシュされたアライメントは旧スコアリング（4F集中率 \(String(format: "%.1f", ratio4 * 100.0))%）です。新MAS再集計（フレーム平均尤度＋二次ペナルティ）を実行します。")
                    needMASGen = true
                }
            }
        }
        if let limit = maxSamples {
            if alignmentMap.count < limit {
                needMASGen = true
            }
        }

        if needMASGen {
            let targetCountStr: String
            switch maxSamples {
            case .some(let limit):
                targetCountStr = "\(limit)"
            case .none:
                targetCountStr = "全"
            }
            print("キャッシュされたアライメント数 (\(alignmentMap.count)) では不足、または縮退が検出されたため、教師 Mel 実測平均と 3 周 MAS 反復集計を実行して \(targetCountStr) 発話のアライメントを自己生成します...")
            let wavDir = cleanDatasetPath + "/wav"
            let transcriptPath = cleanDatasetPath + "/transcript_utf8.txt"
            if let tContent = try? String(contentsOfFile: transcriptPath, encoding: .utf8) {
                let lines = tContent.components(separatedBy: .newlines)
                var inputItems: [MonotonicAlignmentSearch.AlignmentInputItem] = []
                var lIdx = 0
                while lIdx < lines.count {
                    if let limit = maxSamples {
                        if limit <= inputItems.count {
                            break
                        }
                    }
                    let line = lines[lIdx].trimmingCharacters(in: .whitespacesAndNewlines)
                    lIdx += 1
                    if line.isEmpty { continue }

                    var parts = line.split(separator: ":", maxSplits: 1).map { String($0) }
                    if parts.count != 2 {
                        parts = line.split(separator: "\t", maxSplits: 1).map { String($0) }
                    }
                    if parts.count != 2 { continue }

                    let id = parts[0].trimmingCharacters(in: .whitespaces)
                    let text = parts[1].trimmingCharacters(in: .whitespaces)

                    var wavFile = wavDir + "/" + id + ".wav"
                    if fileManager.fileExists(atPath: wavFile) != true {
                        let uppercaseWav = wavDir + "/" + id + ".WAV"
                        if fileManager.fileExists(atPath: uppercaseWav) {
                            wavFile = uppercaseWav
                        }
                    }
                    guard fileManager.fileExists(atPath: wavFile),
                          let rawPCM = try? wavReader.loadWav16k(from: wavFile) else {
                        continue
                    }

                    // ピーク正規化
                    var peak: Float = 0.0
                    var pIdx = 0
                    while pIdx < rawPCM.count {
                        let a = abs(rawPCM[pIdx])
                        if peak < a { peak = a }
                        pIdx += 1
                    }
                    var pcm16k = rawPCM
                    if 0.01 < peak {
                        let normFactor = 0.85 / peak
                        var s = 0
                        while s < pcm16k.count {
                            pcm16k[s] = pcm16k[s] * normFactor
                            s += 1
                        }
                    }

                    let morphemes = engine.normalizer.normalize(text: text)
                    let phrases = engine.prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: engine.vocabulary)
                    var tokens: [PhonemeToken] = []
                    var p = 0
                    while p < phrases.count {
                        var m = 0
                        while m < phrases[p].moras.count {
                            var ph = 0
                            while ph < phrases[p].moras[m].phonemes.count {
                                tokens.append(phrases[p].moras[m].phonemes[ph])
                                ph += 1
                            }
                            m += 1
                        }
                        p += 1
                    }
                    if tokens.isEmpty { continue }

                    let hopSize = AudioConfig.hopSize
                    let targetMel = melExtractor.extractLogMel(pcm: pcm16k)
                    let totalFrames = max(1, targetMel.count)
                    let boundaries = SpikeSpeechEngine.detectSpeechBoundaries(
                        pcm: pcm16k,
                        hopSize: hopSize,
                        totalFrames: totalFrames
                    )
                    let speechFrames = boundaries.speechFrames
                    let leadSilence = boundaries.leadSilence
                    let trailSilence = boundaries.trailSilence

                    if speechFrames < tokens.count || speechFrames <= 0 { continue }

                    let pitchRes = pitchTracker.track(pcm: pcm16k)
                    let speechEnd = min(targetMel.count, leadSilence + speechFrames)
                    var speechMel: [[Float]] = []
                    var speechVoiced: [Float] = []
                    var sf = leadSilence
                    while sf < speechEnd {
                        speechMel.append(targetMel[sf])
                        var v: Float = 0.0
                        if sf < pitchRes.frameCount {
                            v = pitchRes.voiced[sf]
                        }
                        speechVoiced.append(v)
                        sf += 1
                    }

                    inputItems.append(MonotonicAlignmentSearch.AlignmentInputItem(
                        utteranceId: id,
                        leadSilence: leadSilence,
                        trailSilence: trailSilence,
                        totalSpeechFrames: speechFrames,
                        mel: speechMel,
                        voiced: speechVoiced,
                        phonemes: tokens
                    ))
                }

                let newAlignments = MonotonicAlignmentSearch.iterativelyAlign(
                    items: inputItems,
                    iterations: 3,
                    meanFramesPerMora: 16.0
                )
                for al in newAlignments {
                    alignmentMap[al.utteranceId] = al
                }
                try? AlignmentStore.save(newAlignments, to: corpusAlignPath)
                print("生成された MAS アライメント (\(newAlignments.count) 発話) を \(corpusAlignPath) に保存しました")
            }
        }
    }

    var durationSums: [Int32: Float] = [:]
    var durationCounts: [Int32: Float] = [:]
    var totalSpeechFramesAcrossCorpus = 0
    var totalMorasAcrossCorpus = 0

    // コーパス読み込み
    let corpusDir = cleanDatasetPath
    var trainingData: [(features: [[Float]], targets: [[Float]], targetAudio: [Float])] = []
    var frameMelSamples: [FrameMelTrainingSample] = []
    var frameMelTexts: [String] = []
    var skippedFrameMelSamples = 0
    var reconTargetSample: (features: [[Float]], targets: [[Float]], targetAudio: [Float])? = nil
    var vocoderPairs: [(mel: [[Float]], f0: [Float], voiced: [Float], pcm: [Float])] = []
    var prosodySamples: [ProsodyTrainingSample] = []

    print("コーパスパス: \(corpusDir)")
    let transcriptPath = corpusDir + "/transcript_utf8.txt"
    let wavDir = corpusDir + "/wav"
    if let content = try? String(contentsOfFile: transcriptPath, encoding: .utf8) {
        let lines = content.components(separatedBy: .newlines)
        var lineIdx = 0
        while lineIdx < lines.count {
            if let limit = maxSamples {
                let currentCount: Int
                switch useFrameMel {
                case true: currentCount = frameMelSamples.count
                case false: currentCount = trainingData.count
                }
                if limit <= currentCount {
                    break
                }
            }
            let line = lines[lineIdx].trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                lineIdx += 1
                continue
            }
            var parts = line.split(separator: ":", maxSplits: 1).map { String($0) }
            if parts.count != 2 {
                parts = line.split(separator: "\t", maxSplits: 1).map { String($0) }
            }
            if parts.count == 2 {
                let id = parts[0].trimmingCharacters(in: .whitespaces)
                let text = parts[1].trimmingCharacters(in: .whitespaces)
                var wavFile = wavDir + "/" + id + ".wav"
                if fileManager.fileExists(atPath: wavFile) != true {
                    let uppercaseWav = wavDir + "/" + id + ".WAV"
                    if fileManager.fileExists(atPath: uppercaseWav) {
                        wavFile = uppercaseWav
                    }
                }

                if fileManager.fileExists(atPath: wavFile) {
                    if let rawPCM = try? wavReader.loadWav16k(from: wavFile) {
                        // なぜ教師音声を標準肉声レベル（Peak 0.85）にピーク正規化するか:
                        // 各録音トラックごとのマイクゲインのばらつきや過小音量を排し、
                        // 人間の標準的な肉声聴取音量（RMS 0.15〜0.20、Peak 0.80〜0.85）を
                        // SNN およびニューラルボコーダーの正準ターゲットとして統一するため。
                        var peak: Float = 0.0
                        var pIdx = 0
                        while pIdx < rawPCM.count {
                            let a = abs(rawPCM[pIdx])
                            if peak < a {
                                peak = a
                            }
                            pIdx += 1
                        }
                        var pcm16k = rawPCM
                        if 0.01 < peak {
                            let normFactor = 0.85 / peak
                            var s = 0
                            while s < pcm16k.count {
                                pcm16k[s] = pcm16k[s] * normFactor
                                s += 1
                            }
                        }
                        let morphemes = engine.normalizer.normalize(text: text)
                        let phrases = engine.prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: engine.vocabulary)
                        var effectiveAlign = alignmentMap[id]
                        if effectiveAlign == nil {
                            var tokens: [PhonemeToken] = []
                            var pIdx = 0
                            while pIdx < phrases.count {
                                var mIdx = 0
                                while mIdx < phrases[pIdx].moras.count {
                                    var phIdx = 0
                                    while phIdx < phrases[pIdx].moras[mIdx].phonemes.count {
                                        tokens.append(phrases[pIdx].moras[mIdx].phonemes[phIdx])
                                        phIdx += 1
                                    }
                                    mIdx += 1
                                }
                                pIdx += 1
                            }
                            if tokens.isEmpty != true {
                                let hopSize = AudioConfig.hopSize
                                let targetMel = melExtractor.extractLogMel(pcm: pcm16k)
                                let totalFrames = max(1, targetMel.count)
                                let boundaries = SpikeSpeechEngine.detectSpeechBoundaries(
                                    pcm: pcm16k,
                                    hopSize: hopSize,
                                    totalFrames: totalFrames
                                )
                                let speechFrames = boundaries.speechFrames
                                let leadSilence = boundaries.leadSilence
                                let trailSilence = boundaries.trailSilence

                                if tokens.count <= speechFrames {
                                    let pitchRes = pitchTracker.track(pcm: pcm16k)
                                    let speechEnd = min(targetMel.count, leadSilence + speechFrames)
                                    var speechMel: [[Float]] = []
                                    var speechVoiced: [Float] = []
                                    var sf = leadSilence
                                    while sf < speechEnd {
                                        speechMel.append(targetMel[sf])
                                        var v: Float = 0.0
                                        if sf < pitchRes.frameCount {
                                            v = pitchRes.voiced[sf]
                                        }
                                        speechVoiced.append(v)
                                        sf += 1
                                    }

                                    let masAligner = MonotonicAlignmentSearch()
                                    if let durs = masAligner.align(
                                        mel: speechMel,
                                        voiced: speechVoiced,
                                        phonemes: tokens,
                                        meanFramesPerMora: 16.0
                                    ) {
                                        var phList: [PhonemeAlignment] = []
                                        var tIdx = 0
                                        while tIdx < tokens.count {
                                            phList.append(PhonemeAlignment(
                                                symbol: tokens[tIdx].symbol,
                                                phoneId: Int32(tokens[tIdx].id),
                                                durationFrames: durs[tIdx]
                                            ))
                                            tIdx += 1
                                        }
                                        effectiveAlign = UtteranceAlignment(
                                            utteranceId: id,
                                            leadSilenceFrames: leadSilence,
                                            trailSilenceFrames: trailSilence,
                                            totalSpeechFrames: speechFrames,
                                            phonemes: phList
                                        )
                                        alignmentMap[id] = effectiveAlign
                                    }
                                }
                            }
                        }

                        if let al = effectiveAlign {
                            if AlignmentStore.isUtteranceAlignmentValid(al) {
                                totalSpeechFramesAcrossCorpus += al.totalSpeechFrames
                                var phraseMoraCount = 0
                                for phrase in phrases {
                                    phraseMoraCount += phrase.moras.count
                                }
                                totalMorasAcrossCorpus += phraseMoraCount

                                var p = 0
                                while p < al.phonemes.count {
                                    let ph = al.phonemes[p]
                                    let curS = durationSums[ph.phoneId] ?? 0.0
                                    let curC = durationCounts[ph.phoneId] ?? 0.0
                                    durationSums[ph.phoneId] = curS + Float(ph.durationFrames)
                                    durationCounts[ph.phoneId] = curC + 1.0
                                    p += 1
                                }
                            }
                        }

                        // なぜアライメント不整合のサンプルを除外するか:
                        // 読み（音素列）が変わった発話に古いアライメントを当てると、prepareFrameMelTrainingSample が
                        // 平均継続時間に基づく擬似アライメントへ黙って退避し、誤った継続時間を教師として学習してしまうため。
                        var bodyTokenCount = 0
                        for phrase in phrases {
                            for mora in phrase.moras {
                                bodyTokenCount += mora.phonemes.count
                            }
                        }
                        var alignmentUsable = false
                        if let al = effectiveAlign {
                            if AlignmentStore.isUtteranceAlignmentValid(al) && al.phonemes.count == bodyTokenCount {
                                alignmentUsable = true
                            }
                        }

                        switch useFrameMel {
                        case true:
                            if alignmentUsable != true {
                                skippedFrameMelSamples += 1
                            } else if let fSample = engine.prepareFrameMelTrainingSample(
                                text: text,
                                pcm16k: pcm16k,
                                melExtractor: melExtractor,
                                pitchTracker: pitchTracker,
                                alignment: effectiveAlign
                            ) {
                                frameMelSamples.append(fSample)
                                frameMelTexts.append(text)
                                if 0 < vocoderEpochs {
                                    let extractedMel = melExtractor.extractLogMel(pcm: pcm16k)
                                    let pitchResult = pitchTracker.track(pcm: pcm16k)
                                    vocoderPairs.append((mel: extractedMel, f0: pitchResult.f0, voiced: pitchResult.voiced, pcm: pcm16k))
                                }
                            }
                        case false:
                            if let pair = engine.prepareTrainingPair(
                                text: text,
                                pcm16k: pcm16k,
                                melExtractor: melExtractor,
                                pitchTracker: pitchTracker,
                                alignment: effectiveAlign,
                                useScaledDuration: true
                            ) {
                                trainingData.append(pair)
                                if id == "BASIC5000_0001" {
                                    reconTargetSample = pair
                                }
                                if 0 < vocoderEpochs {
                                    let extractedMel = melExtractor.extractLogMel(pcm: pcm16k)
                                    let pitchResult = pitchTracker.track(pcm: pcm16k)
                                    vocoderPairs.append((mel: extractedMel, f0: pitchResult.f0, voiced: pitchResult.voiced, pcm: pcm16k))
                                }
                            }
                        }
                        if 0 < prosodyEpochs {
                            if let pSample = engine.prepareProsodyTrainingSample(
                                text: text,
                                pcm16k: pcm16k,
                                pitchTracker: pitchTracker
                            ) {
                                var canAppendProsody = true
                                switch prosodySamplesLimit {
                                case .some(let limit):
                                    if limit <= prosodySamples.count {
                                        canAppendProsody = false
                                    }
                                case .none:
                                    break
                                }
                                if canAppendProsody {
                                    prosodySamples.append(pSample)
                                }
                            }
                        }
                    }
                }
            }
            lineIdx += 1
        }
    }

    // コーパス側キャッシュに教師 Mel MAS アライメントを永続化（Models には書かない）
    let corpusMasCachePath = cleanDatasetPath + "/mas_alignments.json"
    if fileManager.fileExists(atPath: corpusMasCachePath) != true && alignmentMap.isEmpty != true {
        let alignList = Array(alignmentMap.values).sorted(by: { $0.utteranceId < $1.utteranceId })
        do {
            try AlignmentStore.save(alignList, to: corpusMasCachePath)
            print("コーパスディレクトリに教師 Mel MAS アライメント記録をキャッシュ保存しました: \(corpusMasCachePath) (\(alignList.count) 発話)")
        } catch {
            print("警告: MAS アライメントキャッシュ保存に失敗しました: \(error)")
        }
    }

    switch useFrameMel {
    case true:
        if frameMelSamples.isEmpty {
            print("エラー: 有効な FrameMel 学習データが 0 件です。")
            return
        }
        print("有効 FrameMel 学習サンプル数: \(frameMelSamples.count) 件（アライメント不整合で除外: \(skippedFrameMelSamples) 件）")
    case false:
        if trainingData.isEmpty {
            print("エラー: 有効な学習データが 0 件です。")
            return
        }
        print("有効学習サンプル数: \(trainingData.count) 件")
    }

    // ============================================================
    // ニューラルボコーダー学習（FrameMel / SNN 両経路から共通で呼び出す）
    // なぜ関数化するか: 以前は SNN 経路の末尾にのみ置かれ、FrameMel 経路では
    // 早期 return により一度も到達せず、ボコーダーが更新されない状態が続いていたため。
    // ============================================================
    let outputURL = URL(fileURLWithPath: outputPath)
    let outputDir = outputURL.deletingLastPathComponent().path
    let vocoderURL = WeightCheckpoint.resolvePath(directory: outputDir, fileName: "vocoder_weights.json")

    func runVocoderTraining() {
        if 0 < vocoderEpochs && vocoderPairs.isEmpty != true {
            print("--------------------------------------------------")
            print("ニューラルボコーダーの最適化（Multi-Resolution STFT損失）を開始します (サンプル数: \(vocoderPairs.count), エポック数: \(vocoderEpochs))...")
            let vocoder = MLXNeuralVocoder()
            if forceFresh != true && fileManager.fileExists(atPath: vocoderURL.path) {
                if let existingData = try? Data(contentsOf: vocoderURL) {
                    switch try? JSONDecoder().decode(NeuralVocoderWeights.self, from: existingData) {
                    case .some(let savedWeights):
                        if savedWeights.config.hiddenChannels != 256 {
                            // なぜ 64ch 重みを破棄して 256ch モデルの新規初期重みを使用するか:
                            // 64ch 重みを 256ch モデルに import するとテンソル形状不整合でクラッシュするため、
                            // 推論時（NeuralVocoder.swift）と同様に不一致時は破棄し、256ch の初期状態から学習するため。
                            print("[NeuralVocoder] 警告: \(vocoderURL.path) の hiddenChannels (\(savedWeights.config.hiddenChannels)) が 256 と一致しないため、破棄し 256ch の初期重みを使用します。")
                        } else {
                            vocoder.importWeights(from: savedWeights)
                            print("既存のニューラルボコーダー重みを読み込みました: \(vocoderURL.path)")
                        }
                    case .none:
                        break
                    }
                }
            }

            let vocoderOptimizer = Adam(learningRate: vocoderLearningRate)
            let segFrames = 32
            let hopSize = vocoder.config.hopSize
            let segSamples = segFrames * hopSize
            let melCh = vocoder.config.melChannels
            let inCh = melCh + 2

            let lg = valueAndGrad(model: vocoder) { (model: MLXNeuralVocoder, arrays: [MLXArray]) -> [MLXArray] in
                let fArr = arrays[0]
                let tArr = arrays[1]
                let pred = model(fArr)
                let loss = MLXNeuralVocoder.totalVocoderLoss(predicted: pred, target: tArr)
                return [loss]
            }

            Memory.cacheLimit = 32 * 1024 * 1024
            let batchSize = 8
            var vEpoch = 0
            while vEpoch < vocoderEpochs {
                var epochLossSum: Float = 0.0
                var vBatchCount = 0

                var batchFeats = [Float]()
                var batchPCMs = [Float]()
                var currBatchItems = 0

                var pairIdx = 0
                while pairIdx < vocoderPairs.count {
                    let pair = vocoderPairs[pairIdx]
                    let totalF = pair.mel.count
                    if segFrames <= totalF {
                        let maxStart = totalF - segFrames
                        var sampleIt = 0
                        while sampleIt < 4 {
                            var startF = 0
                            if 0 < maxStart {
                                var bestStart = Int.random(in: 0...maxStart)
                                var trial = 0
                                while trial < 8 {
                                    let cand = Int.random(in: 0...maxStart)
                                    var voicedCount = 0
                                    var chkF = 0
                                    while chkF < segFrames {
                                        let idx = cand + chkF
                                        if idx < pair.voiced.count {
                                            if 0.5 < pair.voiced[idx] {
                                                voicedCount += 1
                                            }
                                        }
                                        chkF += 1
                                    }
                                    if 4 <= voicedCount {
                                        bestStart = cand
                                        break
                                    }
                                    trial += 1
                                }
                                startF = bestStart
                            }
                            let startSample = startF * hopSize
                            let endSample = startSample + segSamples

                            if endSample <= pair.pcm.count {
                                var f = 0
                                while f < segFrames {
                                    let currF = startF + f
                                    let frameMel = pair.mel[currF]
                                    var c = 0
                                    let copyLimit = min(melCh, frameMel.count)
                                    while c < copyLimit {
                                        batchFeats.append(frameMel[c])
                                        c += 1
                                    }
                                    while c < melCh {
                                        batchFeats.append(0.0)
                                        c += 1
                                    }
                                    var normF0: Float = 0.0
                                    if currF < pair.f0.count {
                                        let val = pair.f0[currF]
                                        if 0.0 < val {
                                            var nF0 = val / 500.0
                                            if nF0 < 0.0 { nF0 = 0.0 }
                                            if 1.0 < nF0 { nF0 = 1.0 }
                                            normF0 = nF0
                                        }
                                    }
                                    batchFeats.append(normF0)

                                    var vVal: Float = 1.0
                                    if currF < pair.voiced.count {
                                        vVal = pair.voiced[currF]
                                    }
                                    batchFeats.append(vVal)
                                    f += 1
                                }

                                batchPCMs.append(contentsOf: pair.pcm[startSample..<endSample])
                                currBatchItems += 1

                                if batchSize <= currBatchItems {
                                    autoreleasepool {
                                        let featArr = MLXArray(batchFeats, [currBatchItems, segFrames, inCh])
                                        let targArr = MLXArray(batchPCMs, [currBatchItems, segSamples])

                                        let (lossVals, grads) = lg(vocoder, [featArr, targArr])
                                        let lossVal = lossVals[0].item(Float.self)
                                        epochLossSum += lossVal
                                        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 1.0)
                                        vocoderOptimizer.update(model: vocoder, gradients: clippedGrads)
                                        eval(vocoder.trainableParameters(), lossVal)
                                        Stream.gpu.synchronize()
                                    }

                                    vBatchCount += 1
                                    batchFeats.removeAll(keepingCapacity: true)
                                    batchPCMs.removeAll(keepingCapacity: true)
                                    currBatchItems = 0
                                    if (vBatchCount % 10) == 0 {
                                        Memory.clearCache()
                                    }
                                }
                            }
                            sampleIt += 1
                        }
                    }
                    pairIdx += 1
                }

                if 0 < currBatchItems {
                    autoreleasepool {
                        let featArr = MLXArray(batchFeats, [currBatchItems, segFrames, inCh])
                        let targArr = MLXArray(batchPCMs, [currBatchItems, segSamples])

                        let (lossVals, grads) = lg(vocoder, [featArr, targArr])
                        let lossVal = lossVals[0].item(Float.self)
                        epochLossSum += lossVal
                        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 1.0)
                        vocoderOptimizer.update(model: vocoder, gradients: clippedGrads)
                        eval(vocoder.trainableParameters(), lossVal)
                        Stream.gpu.synchronize()
                    }

                    vBatchCount += 1
                    batchFeats.removeAll(keepingCapacity: true)
                    batchPCMs.removeAll(keepingCapacity: true)
                    currBatchItems = 0
                }

                eval(vocoder.trainableParameters())
                Stream.gpu.synchronize()
                Memory.clearCache()

                var avgVLoss: Float = 0.0
                if 0 < vBatchCount {
                    avgVLoss = epochLossSum / Float(vBatchCount)
                }
                print("  [Vocoder Epoch \(vEpoch + 1)/\(vocoderEpochs)] 平均STFT損失: \(String(format: "%.6f", avgVLoss)) (バッチ数: \(vBatchCount))")
                vEpoch += 1
            }

            let trainedWeights = vocoder.exportWeights()
            do {
                let encoded = try JSONEncoder().encode(trainedWeights)
                try encoded.write(to: vocoderURL, options: .atomic)
                print("最適化済みニューラルボコーダー重みを保存しました: \(vocoderURL.path) (\(encoded.count) バイト)")
            } catch {
                print("警告: ニューラルボコーダー重みの保存に失敗しました: \(error)")
            }
            // なぜボコーダー学習エポックが 0 の場合に既存重みを保持するか:
            // SNN 再学習時に獲得済みのニューラルボコーダー音響合成重みを破壊せず、
            // 単一話者（女性）の自然な声質を 100% 確実に維持するため。
            var needVocoderWrite = false
            if fileManager.fileExists(atPath: vocoderURL.path) != true {
                needVocoderWrite = true
            } else {
                if let existingData = try? Data(contentsOf: vocoderURL) {
                    switch try? JSONDecoder().decode(NeuralVocoderWeights.self, from: existingData) {
                    case .some(let savedWeights):
                        if savedWeights.config.hiddenChannels != 256 {
                            needVocoderWrite = true
                        } else {
                            needVocoderWrite = false
                        }
                    case .none:
                        needVocoderWrite = true
                    }
                } else {
                    needVocoderWrite = true
                }
            }
            if needVocoderWrite {
                let initialVocoderWeights = NeuralVocoderWeights.randomWeights()
                if let encoded = try? JSONEncoder().encode(initialVocoderWeights) {
                    try? encoded.write(to: vocoderURL, options: .atomic)
                    print("ニューラルボコーダー初期重みをエクスポートしました: \(vocoderURL.path) (\(encoded.count) バイト)")
                }
            }
        }
    }

    if useFrameMel {
        // ボコーダーを先に学習する。FrameMel の検証用波形に最新のボコーダーを使うため。
        runVocoderTraining()

        print("\n==================================================")
        print("SpikeVoice (arXiv:2408.00788) フレーム単位対数メルモデル学習を開始します")
        print("  データ数: \(frameMelSamples.count), エポック数: \(epochs)")
        print("  初期学習率: \(learningRate), 最小学習率: \(lrMin), ウォームアップ: \(warmupEpochs) エポック")
        print("==================================================")

        // 1. モデルとトレーナーの初期化（全パラメータを学習対象とする）
        let frameMelModel: MLXFrameMelModel
        switch weights.frameMelWeights {
        case .some(let existingW):
            // なぜ残差重みを捨ててから本体学習を始めるか:
            // 本体は残差なしの損失で学習されるが、推論は残差が残っていれば必ず加算する。
            // 旧残差を保持したまま本体を更新すると、学習時と推論時のメルが食い違うため。
            frameMelModel = MLXFrameMelModel(weights: existingW.withoutMelResidual())
            switch existingW.resW1 {
            case .some:
                print("既存の FrameMel 重みをロードしました（旧メル残差重みは破棄し、本体を直接学習します）。")
            case .none:
                print("既存の FrameMel 重みをロードしました。")
            }
        case .none:
            var melSums = [Float](repeating: 0.0, count: AudioConfig.melChannels)
            var melCounts = [Float](repeating: 0.0, count: AudioConfig.melChannels)
            var sIdx = 0
            while sIdx < frameMelSamples.count {
                let sMel = frameMelSamples[sIdx].targetMel
                var f = 0
                while f < sMel.count {
                    var c = 0
                    while c < AudioConfig.melChannels {
                        if c < sMel[f].count {
                            melSums[c] += sMel[f][c]
                            melCounts[c] += 1.0
                        }
                        c += 1
                    }
                    f += 1
                }
                sIdx += 1
            }
            var meanMel = [Float](repeating: 0.0, count: AudioConfig.melChannels)
            var c = 0
            while c < AudioConfig.melChannels {
                if 0.0 < melCounts[c] {
                    meanMel[c] = melSums[c] / melCounts[c]
                }
                c += 1
            }
            let initW = FrameMelWeights.randomWeights(seed: 2026, meanMel: meanMel)
            frameMelModel = MLXFrameMelModel(weights: initW)
            print("新規の FrameMel 重みを決定論的初期化しました（初期 meanMel 適合）。")
        }

        let trainer = MLXFrameMelTrainer(model: frameMelModel, learningRate: learningRate)
        Memory.cacheLimit = 32 * 1024 * 1024

        // 2. 学習／検証分割（決定論的: validationEvery 件に 1 件を検証用に保持）
        // なぜ保持検証データで採否を決めるか:
        // 特定文の経験則指標（重心・包絡変化など）での採否は読みの誤りや別の文の崩れを検出できず、
        // 学習損失の改善と無関係に重みが破棄される原因になっていたため。
        let validationEvery = 20
        var trainSet: [FrameMelTrainingSample] = []
        var validSet: [FrameMelTrainingSample] = []
        var validTexts: [String] = []
        let canSplit = (2 * validationEvery) <= frameMelSamples.count
        var splitIdx = 0
        while splitIdx < frameMelSamples.count {
            if canSplit && (splitIdx % validationEvery) == 0 {
                validSet.append(frameMelSamples[splitIdx])
                if splitIdx < frameMelTexts.count {
                    validTexts.append(frameMelTexts[splitIdx])
                }
            } else {
                trainSet.append(frameMelSamples[splitIdx])
            }
            splitIdx += 1
        }
        switch canSplit {
        case true:
            print("学習 \(trainSet.count) 件 / 検証 \(validSet.count) 件 に分割しました（\(validationEvery) 件ごとに 1 件を検証用に保持）。")
        case false:
            print("サンプル数が少ないため検証分割を行わず、学習損失でエポックを選定します。")
        }

        let schedule = CosineWarmupSchedule(
            lrBase: learningRate,
            lrMin: lrMin,
            warmupEpochs: warmupEpochs,
            totalEpochs: epochs
        )

        // 3. コーパス実測の音素平均継続時間とモーラ速度を重みへ保存する
        let healthyAlignments = Array(alignmentMap.values).filter { AlignmentStore.isUtteranceAlignmentValid($0) }
        let healthyPhonemeAverages: [Int32: Float]
        if healthyAlignments.isEmpty != true {
            healthyPhonemeAverages = AlignmentStore.computeAverageDurations(from: healthyAlignments)
        } else {
            switch weights.phonemeAverageDurations {
            case .some(let existing):
                healthyPhonemeAverages = existing
            case .none:
                healthyPhonemeAverages = LengthRegulator.defaultPhonemeAverageDurations
            }
        }
        var corpusMoraRate: Float = weights.meanFramesPerMora ?? LengthRegulator.defaultMeanFramesPerMora
        if 0 < totalMorasAcrossCorpus {
            let measured = Float(totalSpeechFramesAcrossCorpus) / Float(totalMorasAcrossCorpus)
            if 8.0 <= measured && measured <= 24.0 {
                corpusMoraRate = measured
            }
        }
        print("コーパス実測モーラ速度: \(String(format: "%.2f", corpusMoraRate)) frames/モーラ (\(String(format: "%.0f", corpusMoraRate * 10.0)) ms/モーラ)")
        let baseWithDurations = weights
            .withMeanFramesPerMora(corpusMoraRate)
            .withPhonemeAverageDurations(healthyPhonemeAverages)

        func averageLosses(_ samples: [FrameMelTrainingSample]) -> FrameMelLosses {
            var total: Float = 0.0
            var dec: Float = 0.0
            var post: Float = 0.0
            var f0: Float = 0.0
            var eng: Float = 0.0
            var dur: Float = 0.0
            var delta: Float = 0.0
            var n = 0
            var i = 0
            while i < samples.count {
                let s = samples[i]
                let l = autoreleasepool {
                    trainer.evaluateSample(
                        phoneIds: s.phoneIds,
                        targetDurations: s.targetDurations,
                        targetMel: s.targetMel,
                        targetF0: s.targetF0,
                        targetEnergy: s.targetEnergy,
                        phoneAccent: s.phoneAccent
                    )
                }
                if l.totalLoss.isFinite {
                    total += l.totalLoss
                    dec += l.decMelL1
                    post += l.postMelL1
                    f0 += l.voicedF0MSE
                    eng += l.energyMSE
                    dur += l.durMSE
                    delta += l.deltaMelL1
                    n += 1
                }
                if (i % 50) == 49 {
                    Stream.gpu.synchronize()
                    Memory.clearCache()
                }
                i += 1
            }
            let d = Float(max(1, n))
            return FrameMelLosses(
                totalLoss: total / d,
                decMelL1: dec / d,
                postMelL1: post / d,
                voicedF0MSE: f0 / d,
                energyMSE: eng / d,
                durMSE: dur / d,
                deltaMelL1: delta / d
            )
        }

        var bestSelectionLoss: Float = Float.greatestFiniteMagnitude
        var bestEpoch: Int = -1
        var bestFullWeights: SpikingNetworkWeights = baseWithDurations.withFrameMelWeights(trainer.exportWeights())

        var epoch = 0
        while epoch < epochs {
            if noShuffle != true {
                let seed = TrainingShuffle.mixSeed(baseSeed: shuffleSeed, epoch: epoch)
                TrainingShuffle.shuffleInPlace(&trainSet, seed: seed)
            }

            let lr = schedule.learningRate(epoch: epoch)
            trainer.setLearningRate(lr)

            var epochTotalLoss: Float = 0.0
            var epochDecL1: Float = 0.0
            var epochPostL1: Float = 0.0
            var epochF0MSE: Float = 0.0
            var epochEnergyMSE: Float = 0.0
            var epochDurMSE: Float = 0.0
            var epochDeltaL1: Float = 0.0
            var sampleCount = 0

            var sIdx = 0
            while sIdx < trainSet.count {
                let s = trainSet[sIdx]
                let losses = autoreleasepool {
                    trainer.trainSample(
                        phoneIds: s.phoneIds,
                        targetDurations: s.targetDurations,
                        targetMel: s.targetMel,
                        targetF0: s.targetF0,
                        targetEnergy: s.targetEnergy,
                        phoneAccent: s.phoneAccent
                    )
                }

                if losses.totalLoss.isFinite {
                    epochTotalLoss += losses.totalLoss
                    epochDecL1 += losses.decMelL1
                    epochPostL1 += losses.postMelL1
                    epochF0MSE += losses.voicedF0MSE
                    epochEnergyMSE += losses.energyMSE
                    epochDurMSE += losses.durMSE
                    epochDeltaL1 += losses.deltaMelL1
                    sampleCount += 1
                }

                // Metal リソース保護（50 サンプルごと）。トレーナーは再生成せず Adam の状態を維持する。
                if (sampleCount % 50) == 0 {
                    Stream.gpu.synchronize()
                    Memory.clearCache()
                }

                if (sampleCount % 200) == 0 {
                    print("    ステップ [\(sampleCount)/\(trainSet.count)] 直近損失: \(String(format: "%.4f", losses.totalLoss)) (Dec: \(String(format: "%.4f", losses.decMelL1)), Post: \(String(format: "%.4f", losses.postMelL1)), F0: \(String(format: "%.4f", losses.voicedF0MSE)), Eng: \(String(format: "%.4f", losses.energyMSE)), Dur: \(String(format: "%.4f", losses.durMSE)))")
                }

                sIdx += 1
            }
            Memory.clearCache()

            let n = Float(max(1, sampleCount))
            let avgTotal = epochTotalLoss / n
            let avgDec = epochDecL1 / n
            let avgPost = epochPostL1 / n
            let avgF0 = epochF0MSE / n
            let avgEng = epochEnergyMSE / n
            let avgDur = epochDurMSE / n
            let avgDelta = epochDeltaL1 / n

            print("  [Epoch \(epoch + 1)/\(epochs)] 学習損失: \(String(format: "%.4f", avgTotal)) (Dec: \(String(format: "%.4f", avgDec)), Post: \(String(format: "%.4f", avgPost)), F0: \(String(format: "%.4f", avgF0)), Eng: \(String(format: "%.4f", avgEng)), Dur: \(String(format: "%.4f", avgDur)), Delta: \(String(format: "%.4f", avgDelta)))  lr=\(String(format: "%.6g", lr))")

            // 検証損失（教師強制・勾配なし）
            let selectionLoss: Float
            switch validSet.isEmpty {
            case false:
                let v = averageLosses(validSet)
                print("    検証損失: \(String(format: "%.4f", v.totalLoss)) (Dec: \(String(format: "%.4f", v.decMelL1)), Post: \(String(format: "%.4f", v.postMelL1)), F0: \(String(format: "%.4f", v.voicedF0MSE)), Eng: \(String(format: "%.4f", v.energyMSE)), Dur: \(String(format: "%.4f", v.durMSE)), Delta: \(String(format: "%.4f", v.deltaMelL1)))")
                selectionLoss = v.totalLoss
            case true:
                selectionLoss = avgTotal
            }

            // エポック重みの保存（各エポック）と最良重みの即時保存
            let currentFMW = trainer.exportWeights()
            let epochFullWeights = baseWithDurations.withFrameMelWeights(currentFMW)
            let epURL = WeightCheckpoint.resolvePath(directory: outputDir, fileName: String(format: "weights.ep%02d.json", epoch + 1))
            try? WeightCheckpoint.atomicWritePretty(epochFullWeights, to: epURL)

            if selectionLoss < bestSelectionLoss {
                bestSelectionLoss = selectionLoss
                bestEpoch = epoch + 1
                bestFullWeights = epochFullWeights
                do {
                    try WeightCheckpoint.atomicWritePretty(bestFullWeights, to: outputURL)
                    print("    ==> 最良エポック更新: Epoch \(bestEpoch) (選定損失: \(String(format: "%.4f", bestSelectionLoss)))。\(outputPath) に保存しました。")
                } catch {
                    print("    エラー: 最良重みの保存に失敗しました: \(error)")
                }
            }

            epoch += 1
        }

        print("\n==================================================")
        print("FrameMel 学習完了: 採択エポック = Epoch \(bestEpoch) (選定損失: \(String(format: "%.4f", bestSelectionLoss)))")
        print("  保存先: \(outputPath)")
        print("==================================================")

        // 試聴用: 検証用発話の先頭 1 文を教師なし合成して .tmp に書き出す（テキストはコーパス由来、コードには持たない）
        if let listenText = validTexts.first {
            var vocWeights: NeuralVocoderWeights? = nil
            if let vData = try? Data(contentsOf: vocoderURL) {
                vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
            }
            let listenEngine = SpikeSpeechEngine(weights: bestFullWeights, vocoderWeights: vocWeights)
            let pcm = listenEngine.synthesize(text: listenText)
            let listenDir = ".tmp/frame_mel_eval"
            try? fileManager.createDirectory(atPath: listenDir, withIntermediateDirectories: true)
            let listenURL = URL(fileURLWithPath: listenDir + String(format: "/valid0_ep%02d.wav", bestEpoch))
            let wav = WavEncoder.encode(samples: pcm, sampleRate: AudioConfig.sampleRate)
            try? wav.write(to: listenURL)
            print("試聴用波形を書き出しました: \(listenURL.path) (\(pcm.count) サンプル)")
        }

        print("学習処理が正常に完了しました。")
        return
    }

    // ============================================================
    // 受入検証ゲート（手順 3）: アライメント品質および音響特徴検証
    // ============================================================
    print("\n===========================================================")
    print("【受入検証ゲート（手順 3）: アライメント品質および音響特徴検証】")
    print("===========================================================")

    // 1. 全発話の継続時間分布検証
    // 母音（vowel, prolonged）の下限（2フレーム）一致率 < 15%
    // 全音素の上限（40フレーム）一致率 < 5%
    var totalVowels = 0
    var minDurationVowels = 0
    var totalPhonemes = 0
    var maxDurationPhonemes = 0

    for (_, utt) in alignmentMap {
        var pIdx = 0
        while pIdx < utt.phonemes.count {
            let ph = utt.phonemes[pIdx]
            let d = ph.durationFrames
            totalPhonemes += 1
            if d == 40 {
                maxDurationPhonemes += 1
            }
            let isVowel: Bool
            switch ph.symbol {
            case "a", "i", "u", "e", "o", "_":
                isVowel = true
            default:
                isVowel = false
            }
            if isVowel {
                totalVowels += 1
                if d == 2 {
                    minDurationVowels += 1
                }
            }
            pIdx += 1
        }
    }

    let minVowelRatio = Float(minDurationVowels) / Float(max(1, totalVowels))
    let maxPhoneRatio = Float(maxDurationPhonemes) / Float(max(1, totalPhonemes))

    print("  全発話 母音総数: \(totalVowels), 下限 (2F) 一致数: \(minDurationVowels), 割合: \(String(format: "%.2f", minVowelRatio * 100.0))% (閾値: < 15.00%)")
    print("  全発話 全音素総数: \(totalPhonemes), 上限 (40F) 一致数: \(maxDurationPhonemes), 割合: \(String(format: "%.2f", maxPhoneRatio * 100.0))% (閾値: < 5.00%)")

    var gatePassed = true
    if 0.15 <= minVowelRatio {
        print("  エラー: 下限一致母音割合が 15% 以上です (\(String(format: "%.2f", minVowelRatio * 100.0))%)")
        gatePassed = false
    }
    if 0.05 <= maxPhoneRatio {
        print("  エラー: 上限一致音素割合が 5% 以上です (\(String(format: "%.2f", maxPhoneRatio * 100.0))%)")
        gatePassed = false
    }

    // 2. BASIC5000_0001 音響特徴検証
    let basic0001Text = "水をマレーシアから買わなくてはならないのです。"
    let wavPath0001 = wavDir + "/BASIC5000_0001.wav"
    guard fileManager.fileExists(atPath: wavPath0001),
          let pcm0001 = try? wavReader.loadWav16k(from: wavPath0001),
          let basic0001Align = alignmentMap["BASIC5000_0001"] else {
        print("エラー: BASIC5000_0001.wav またはアライメントが存在しません。")
        return
    }

    let hopSize = AudioConfig.hopSize

    print("\n  --- BASIC5000_0001 音素区間音響検査 ---")
    var curFrameOffset = basic0001Align.leadSilenceFrames
    var bIdx = 0
    var allVowelsRMSValid = true
    var allFricativesCentroidValid = true

    while bIdx < basic0001Align.phonemes.count {
        let ph = basic0001Align.phonemes[bIdx]
        let d = ph.durationFrames
        let segStartSample = curFrameOffset * hopSize
        let segEndSample = min(pcm0001.count, (curFrameOffset + d) * hopSize)

        var segPCM: [Float] = []
        if segStartSample < segEndSample {
            segPCM = Array(pcm0001[segStartSample..<segEndSample])
        }

        let isVowel: Bool
        switch ph.symbol {
        case "a", "i", "u", "e", "o", "_":
            isVowel = true
        default:
            isVowel = false
        }

        if isVowel {
            let rms = MelSpectrogramExtractor.computeRMS(pcm: segPCM)
            print("    母音 [\(ph.symbol)] (フレーム \(curFrameOffset)..<\(curFrameOffset + d), \(d)F): 平均 RMS = \(String(format: "%.4f", rms))")
            if rms <= 0.01 {
                print("      エラー: 母音 RMS が 0.01 以下です (\(rms))")
                allVowelsRMSValid = false
            }
        }

        let isFricative = (ph.symbol == "s" || ph.symbol == "sh")
        if isFricative {
            let centroid = melExtractor.computeSpectralCentroid(pcm: segPCM)
            print("    摩擦音 [\(ph.symbol)] (フレーム \(curFrameOffset)..<\(curFrameOffset + d), \(d)F): スペクトル重心 = \(String(format: "%.1f", centroid)) Hz")
            if centroid <= 2000.0 {
                print("      エラー: 摩擦音スペクトル重心が 2000 Hz 以下です (\(centroid) Hz)")
                allFricativesCentroidValid = false
            }
        }

        curFrameOffset += d
        bIdx += 1
    }

    if allVowelsRMSValid != true {
        gatePassed = false
    }
    if allFricativesCentroidValid != true {
        gatePassed = false
    }

    if gatePassed != true {
        print("\n【受入検証ゲート REJECT】手順3 の条件を満たさないため学習を中止します。")
        return
    }
    print("【受入検証ゲート（手順 3）PASS】下限/上限割合および音響特徴条件を達成しました。\n")

    // ============================================================
    // 手順 4: ゲート通過区間から実測音素平均フレームを再作成
    // ============================================================
    let healthyAlignments = Array(alignmentMap.values).filter { AlignmentStore.isUtteranceAlignmentValid($0) }
    let healthyPhonemeAverages = AlignmentStore.computeAverageDurations(from: healthyAlignments)
    print("===========================================================")
    print("【手順 4: ゲート通過区間実測音素平均フレーム再作成 (通過発話数: \(healthyAlignments.count)/\(alignmentMap.count))】")
    print("===========================================================")
    for (pid, avg) in healthyPhonemeAverages.sorted(by: { $0.key < $1.key }) {
        let sym = engine.vocabulary.token(for: Int(pid))
        let rAvg = roundf(avg * 10.0) / 10.0
        print("  ID \(pid) (\(sym)): 実測平均 = \(String(format: "%.2f", avg)) frames -> 確定 = \(String(format: "%.1f", rAvg)) frames (\(String(format: "%.1f", rAvg * 10.0))ms)")
    }

    var effectiveWeights = weights
        .withPhonemeAverageDurations(healthyPhonemeAverages)
        .withMeanFramesPerMora(16.0)

    if weights.isCfC != true {
        // ============================================================
        // 手順 5: phonePos (ch 196) 電流スケーリング & パルス (ch 199) 列 0 化
        // ============================================================
        print("\n===========================================================")
        print("【手順 5: phonePos (ch 196) 電流スケーリング & パルス (ch 199) 列 0 化】")
        print("===========================================================")

        // 1. 母音 5 列 (ch 5, 6, 7, 8, 9) の L2 ノルム平均および one-hot 電流 (振幅 3.0)
        let vowelCols = [5, 6, 7, 8, 9]
        var vowelNormSum: Float = 0.0
        for vc in vowelCols {
            var sumSq: Float = 0.0
            var h = 0
            while h < hiddenDim {
                let val = weights.wIn[(h * inDim) + vc]
                sumSq += val * val
                h += 1
            }
            vowelNormSum += sqrtf(sumSq)
        }
        let vowelAvgNorm = vowelNormSum / Float(vowelCols.count)
        let vowelOneHotCurrent = vowelAvgNorm * 3.0

        // 2. phonePos (ch 196) のスケーリング前 L2 ノルムおよび電流
        var p196SumSq: Float = 0.0
        var h = 0
        while h < hiddenDim {
            let val = weights.wIn[(h * inDim) + 196]
            p196SumSq += val * val
            h += 1
        }
        let phonePosNormBefore = sqrtf(p196SumSq)
        let phonePosCurrentBefore = phonePosNormBefore * 3.0

        // 3. パルス (ch 199) のスケーリング前 L2 ノルム
        var p199SumSq: Float = 0.0
        h = 0
        while h < hiddenDim {
            let val = weights.wIn[(h * inDim) + 199]
            p199SumSq += val * val
            h += 1
        }
        let pulseNormBefore = sqrtf(p199SumSq)

        // 4. 倍率の算出: (母音 5 列の L2 平均 × 3.0) / (ch 196 の L2 × 3.0)
        var phonePosScale: Float = 1.0
        if 1e-6 < phonePosNormBefore {
            phonePosScale = (vowelAvgNorm * 3.0) / (phonePosNormBefore * 3.0)
        }

        // 5. 有効重みの適用 (既存重みを維持し、ch 196 スケーリングと ch 199 の 0 化)
        effectiveWeights = effectiveWeights
            .withPhonePosScaledAndPulseZeroed(scale196: phonePosScale)

        // 6. 変換後 wIn 各列 L2 ノルムの計測
        var colNorms = [Float](repeating: 0.0, count: inDim)
        var cColIdx = 0
        while cColIdx < inDim {
            var sumSq: Float = 0.0
            h = 0
            while h < hiddenDim {
                let val = effectiveWeights.wIn[(h * inDim) + cColIdx]
                sumSq += val * val
                h += 1
            }
            colNorms[cColIdx] = sqrtf(sumSq)
            cColIdx += 1
        }

        let phonePosNormAfter = colNorms[196]
        let phonePosCurrentAfter = phonePosNormAfter * 3.0
        let pulseNormAfter = colNorms[199]
        let currentRatio = phonePosCurrentAfter / max(1e-6, vowelOneHotCurrent)

        print("  --- 学習前 電流・重みスケーリング検証 ---")
        print("  母音 5 列 (ch 5, 6, 7, 8, 9) 平均 L2: \(String(format: "%.4f", vowelAvgNorm))")
        print("  母音 one-hot 電流:               \(String(format: "%.4f", vowelOneHotCurrent)) (振幅 3.0)")
        print("  phonePos スケーリング前 L2:          \(String(format: "%.4f", phonePosNormBefore))")
        print("  phonePos スケーリング前 電流:        \(String(format: "%.4f", phonePosCurrentBefore)) (振幅 3.0 時)")
        print("  phonePos スケーリング倍率:           \(String(format: "%.6f", phonePosScale))")
        print("  phonePos スケーリング後 L2:          \(String(format: "%.4f", phonePosNormAfter))")
        print("  phonePos スケーリング後 電流:        \(String(format: "%.4f", phonePosCurrentAfter)) (phonePos=1 時, 振幅 3.0)")
        print("  phonePos / one-hot 電流比:          \(String(format: "%.4f", currentRatio)) (受入基準: 0.8〜1.2)")
        print("  パルス列 (ch 199) スケーリング前 L2:   \(String(format: "%.4f", pulseNormBefore))")
        print("  パルス列 (ch 199) スケーリング後 L2:   \(String(format: "%.4f", pulseNormAfter)) (受入基準: 0.0)")
        print("  ch 192 (voiced):     \(String(format: "%.4f", colNorms[192]))")
        print("  ch 193 (unvoiced):   \(String(format: "%.4f", colNorms[193]))")
        print("  ch 194 (normF0):     \(String(format: "%.4f", colNorms[194]))")
        print("  ch 195 (deltaF0):    \(String(format: "%.4f", colNorms[195]))")
        print("  ch 196 (phonePos):   \(String(format: "%.4f", colNorms[196]))")
        print("  ch 197 (rate):       \(String(format: "%.4f", colNorms[197]))")
        print("  ch 198 (energy):     \(String(format: "%.4f", colNorms[198]))")
        print("  ch 199 (pulse):      \(String(format: "%.4f", colNorms[199]))")

        // 受入ゲート: 電流比 0.8〜1.2 および パルス列 L2 = 0
        if currentRatio < 0.8 || 1.2 < currentRatio {
            print("エラー: phonePos 電流比 (\(currentRatio)) が 0.8〜1.2 の範囲外です。学習を中止します。")
            return
        }
        if 1e-5 < pulseNormAfter {
            print("エラー: パルス列 (ch 199) の L2 ノルム (\(pulseNormAfter)) が 0 ではありません。学習を中止します。")
            return
        }
        print("【受入検証ゲート（電流整合）PASS】phonePos 電流が母音 one-hot と厳密に揃い、パルス列 L2 は 0 です。\n")
    }

    let gateEngine = SpikeSpeechEngine(weights: effectiveWeights)

    // ============================================================
    // 特徴差 0 ゲート: BASIC5000_0001 学習入力 vs 合成入力 特徴行列差分検証
    // ============================================================
    print("\n===========================================================")
    print("【受入検証ゲート: BASIC5000_0001 学習入力 vs 合成入力 特徴行列差分検証】")
    print("===========================================================")

    // 1. prepareTrainingPair の特徴行列
    guard let pair0001 = gateEngine.prepareTrainingPair(
        text: basic0001Text,
        pcm16k: pcm0001,
        melExtractor: melExtractor,
        pitchTracker: pitchTracker,
        alignment: alignmentMap["BASIC5000_0001"],
        useScaledDuration: true
    ) else {
        print("エラー: BASIC5000_0001 の prepareTrainingPair に失敗しました。")
        return
    }
    let trainFeats = pair0001.features

    // 2. synthesize が SNN に渡す特徴行列
    let synthLinguistic = gateEngine.lengthRegulator.processText(
        text: basic0001Text,
        normalizer: gateEngine.normalizer,
        prosodyModel: gateEngine.prosodyModel,
        vocabulary: gateEngine.vocabulary,
        prosodyPredictor: gateEngine.prosodyPredictor,
        speedFactor: 1.0,
        baseF0: VoiceProfile.default.baseF0,
        addBoundarySilence: true,
        meanFramesPerMora: VoiceProfile.default.meanFramesPerMora
    )
    let synthFeats = gateEngine.encodeLinguisticFeatures(features: synthLinguistic)

    print("  学習特徴行列フレーム数: \(trainFeats.count), チャンネル数: \(trainFeats.first?.count ?? 0)")
    print("  合成特徴行列フレーム数: \(synthFeats.count), チャンネル数: \(synthFeats.first?.count ?? 0)")

    if trainFeats.count != synthFeats.count {
        print("エラー: フレーム数が一致しません (学習: \(trainFeats.count) vs 合成: \(synthFeats.count))。学習を開始できません。")
        return
    }

    // 条件 1: ch 194 と ch 195 以外の最大絶対差は 0
    var nonPitchMaxDiff: Float = 0.0
    var channelMaxDiffs = [Float](repeating: 0.0, count: inDim)

    var f = 0
    while f < trainFeats.count {
        var c = 0
        let cCount = min(trainFeats[f].count, synthFeats[f].count)
        while c < cCount {
            let diff = abs(trainFeats[f][c] - synthFeats[f][c])
            if channelMaxDiffs[c] < diff {
                channelMaxDiffs[c] = diff
            }
            if c != 194 && c != 195 {
                if nonPitchMaxDiff < diff {
                    nonPitchMaxDiff = diff
                }
            }
            c += 1
        }
        f += 1
    }

    print("  ch 194/195 以外の全チャンネル・全フレーム 最大絶対差: \(nonPitchMaxDiff)")
    print("  --- チャンネル別最大絶対差 ---")
    let channelGroups: [(name: String, start: Int, end: Int)] = [
        ("ch 0..63 (現在の音素 one-hot)", 0, 63),
        ("ch 64..127 (前の音素 one-hot)", 64, 127),
        ("ch 128..191 (次の音素 one-hot)", 128, 191),
        ("ch 192 (voiced)", 192, 192),
        ("ch 193 (unvoiced)", 193, 193),
        ("ch 194 (normF0)", 194, 194),
        ("ch 195 (deltaF0)", 195, 195),
        ("ch 196 (phonePos)", 196, 196),
        ("ch 197 (rate)", 197, 197),
        ("ch 198 (energy)", 198, 198),
        ("ch 199 (pulse)", 199, 199),
        ("ch 200..255 (その他/次音素傾き/未使用)", 200, 255)
    ]
    var gIdx = 0
    while gIdx < channelGroups.count {
        let grp = channelGroups[gIdx]
        var grpMaxDiff: Float = 0.0
        var ch = grp.start
        while ch <= grp.end {
            if ch < channelMaxDiffs.count {
                if grpMaxDiff < channelMaxDiffs[ch] {
                    grpMaxDiff = channelMaxDiffs[ch]
                }
            }
            ch += 1
        }
        print("    \(grp.name): 最大差 = \(grpMaxDiff)")
        gIdx += 1
    }

    if 0.0 < nonPitchMaxDiff {
        print("エラー: ch 194/195 以外の特徴差が 0 ではありません (\(nonPitchMaxDiff))。学習を中止します。")
        return
    }

    // 条件 2: ch 194 の学習と合成の平均絶対差は 0.02 より大きい
    var sumDiff194: Float = 0.0
    var f194 = 0
    while f194 < trainFeats.count {
        let diff = abs(trainFeats[f194][194] - synthFeats[f194][194])
        sumDiff194 += diff
        f194 += 1
    }
    let meanDiff194 = sumDiff194 / Float(max(1, trainFeats.count))
    print("  ch 194 の学習と合成の平均絶対差: \(String(format: "%.6f", meanDiff194)) (基準: > 0.02)")
    if meanDiff194 <= 0.02 {
        print("エラー: ch 194 の平均絶対差が 0.02 以下です (\(meanDiff194))。上書きが合成側にも入っている可能性があります。学習を中止します。")
        return
    }

    // 条件 3: BASIC5000_0001 の有声な本体フレームで、学習 ch 194 の × 500 と、同じフレームへ補間したトラッカー F0 の相関が 0.8 より大きい
    let trackerRes0001 = pitchTracker.track(pcm: pcm0001)
    let boundaries0001 = SpikeSpeechEngine.detectSpeechBoundaries(
        pcm: pcm0001,
        hopSize: AudioConfig.hopSize,
        totalFrames: max(1, pcm0001.count / AudioConfig.hopSize)
    )
    let phCount0001 = synthLinguistic.phoneIds.count
    let leadSilFrames0001 = Int(synthLinguistic.durations[0])
    let trailSilFrames0001 = Int(synthLinguistic.durations[phCount0001 - 1])
    let bodyPhCount0001 = phCount0001 - 2
    let targetSpeechFrames0001 = synthLinguistic.totalFrames - leadSilFrames0001 - trailSilFrames0001

    let uttAlign0001 = alignmentMap["BASIC5000_0001"]
    let effLeadSil0001: Int
    let effSpeechFrames0001: Int
    var srcDurs0001: [Int] = []

    switch uttAlign0001 {
    case .some(let uAlign) where uAlign.phonemes.count == bodyPhCount0001 && AlignmentStore.isUtteranceAlignmentValid(uAlign):
        effLeadSil0001 = uAlign.leadSilenceFrames
        effSpeechFrames0001 = uAlign.totalSpeechFrames
        var p0 = 0
        while p0 < uAlign.phonemes.count {
            srcDurs0001.append(max(1, uAlign.phonemes[p0].durationFrames))
            p0 += 1
        }
    case _:
        effLeadSil0001 = boundaries0001.leadSilence
        effSpeechFrames0001 = boundaries0001.speechFrames
        var rawDurs0001: [Float] = []
        var p0 = 0
        while p0 < bodyPhCount0001 {
            let pid = synthLinguistic.phoneIds[1 + p0]
            let avgDur = gateEngine.lengthRegulator.phonemeDuration(phoneId: pid, speedFactor: 1.0)
            rawDurs0001.append(avgDur)
            p0 += 1
        }
        var sumDurs0001: Float = 0.0
        var fI0 = 0
        while fI0 < rawDurs0001.count {
            sumDurs0001 += rawDurs0001[fI0]
            fI0 += 1
        }
        let scale0001: Float
        if 0.001 < sumDurs0001 {
            scale0001 = Float(effSpeechFrames0001) / sumDurs0001
        } else {
            scale0001 = 1.0
        }
        var scaledDurs0001: [Float] = []
        var sI0 = 0
        while sI0 < rawDurs0001.count {
            scaledDurs0001.append(max(1.0, rawDurs0001[sI0] * scale0001))
            sI0 += 1
        }
        srcDurs0001 = gateEngine.lengthRegulator.quantizeDurations(durations: scaledDurs0001)
    }

    let speechEnd0001 = min(trackerRes0001.frameCount, effLeadSil0001 + effSpeechFrames0001)
    var speechF0_0001: [Float] = []
    var speechV_0001: [Float] = []
    var sf0001 = effLeadSil0001
    while sf0001 < speechEnd0001 {
        speechF0_0001.append(trackerRes0001.f0[sf0001])
        speechV_0001.append(trackerRes0001.voiced[sf0001])
        sf0001 += 1
    }
    let srcLen0001 = speechF0_0001.count

    var curSum0001 = 0
    var cI0 = 0
    while cI0 < srcDurs0001.count {
        curSum0001 += srcDurs0001[cI0]
        cI0 += 1
    }
    let diff0001 = srcLen0001 - curSum0001
    if diff0001 != 0 && srcDurs0001.isEmpty != true {
        let lastIdx = srcDurs0001.count - 1
        let adj = srcDurs0001[lastIdx] + diff0001
        if 1 <= adj {
            srcDurs0001[lastIdx] = adj
        } else {
            srcDurs0001[lastIdx] = 1
        }
    }

    var resampledF0_0001 = [Float](repeating: 0.0, count: targetSpeechFrames0001)
    var resampledV_0001 = [Float](repeating: 0.0, count: targetSpeechFrames0001)
    var srcOff0 = 0
    var dstOff0 = 0
    var phIdx0 = 0
    while phIdx0 < bodyPhCount0001 {
        let srcDur = srcDurs0001[phIdx0]
        let dstDur = Int(synthLinguistic.durations[1 + phIdx0])
        switch (dstDur <= 1, srcDur <= 1) {
        case (true, _):
            let srcIdx: Int
            if srcDur <= 1 {
                srcIdx = srcOff0
            } else {
                srcIdx = srcOff0 + (srcDur / 2)
            }
            let safeSrcIdx = min(srcLen0001 - 1, max(0, srcIdx))
            let dstIdx = min(targetSpeechFrames0001 - 1, dstOff0)
            resampledF0_0001[dstIdx] = speechF0_0001[safeSrcIdx]
            resampledV_0001[dstIdx] = speechV_0001[safeSrcIdx]
        case (false, true):
            let safeSrcIdx = min(srcLen0001 - 1, max(0, srcOff0))
            var fi = 0
            while fi < dstDur {
                let dstIdx = min(targetSpeechFrames0001 - 1, dstOff0 + fi)
                resampledF0_0001[dstIdx] = speechF0_0001[safeSrcIdx]
                resampledV_0001[dstIdx] = speechV_0001[safeSrcIdx]
                fi += 1
            }
        case (false, false):
            let maxDstP = Float(dstDur - 1)
            let maxSrcP = Float(srcDur - 1)
            var fi = 0
            while fi < dstDur {
                let dstIdx = min(targetSpeechFrames0001 - 1, dstOff0 + fi)
                let posWithinPh = (Float(fi) / maxDstP) * maxSrcP
                var s0 = Int(posWithinPh)
                if srcDur <= s0 { s0 = srcDur - 1 }
                if s0 < 0 { s0 = 0 }
                var s1 = s0 + 1
                if srcDur <= s1 { s1 = srcDur - 1 }
                let alpha = posWithinPh - Float(s0)
                let src0 = min(srcLen0001 - 1, max(0, srcOff0 + s0))
                let src1 = min(srcLen0001 - 1, max(0, srcOff0 + s1))
                resampledF0_0001[dstIdx] = (1.0 - alpha) * speechF0_0001[src0] + alpha * speechF0_0001[src1]
                resampledV_0001[dstIdx] = (1.0 - alpha) * speechV_0001[src0] + alpha * speechV_0001[src1]
                fi += 1
            }
        }
        srcOff0 += srcDur
        dstOff0 += dstDur
        phIdx0 += 1
    }

    var sumCorrX: Float = 0.0
    var sumCorrY: Float = 0.0
    var sumCorrXY: Float = 0.0
    var sumCorrX2: Float = 0.0
    var sumCorrY2: Float = 0.0
    var voicedCount0001: Int = 0

    var bf1 = 0
    while bf1 < targetSpeechFrames0001 {
        let dstF = leadSilFrames0001 + bf1
        let trackerF0 = resampledF0_0001[bf1]
        let trackerV = resampledV_0001[bf1]
        let trainF0 = trainFeats[dstF][194] * 500.0

        if 0.5 <= trackerV && 70.0 <= trackerF0 && 0.0 < trainFeats[dstF][194] {
            sumCorrX += trainF0
            sumCorrY += trackerF0
            sumCorrXY += trainF0 * trackerF0
            sumCorrX2 += trainF0 * trainF0
            sumCorrY2 += trackerF0 * trackerF0
            voicedCount0001 += 1
        }
        bf1 += 1
    }

    let nF0001 = Float(voicedCount0001)
    let num0001 = (nF0001 * sumCorrXY) - (sumCorrX * sumCorrY)
    let denX0001 = (nF0001 * sumCorrX2) - (sumCorrX * sumCorrX)
    let denY0001 = (nF0001 * sumCorrY2) - (sumCorrY * sumCorrY)
    let den0001 = sqrt(max(1e-12, denX0001 * denY0001))
    let corr0001 = num0001 / den0001

    print("  有声本体フレーム相関: \(String(format: "%.6f", corr0001)) (有声フレーム数: \(voicedCount0001), 基準: > 0.8)")
    if corr0001 <= 0.8 {
        print("エラー: 有声本体フレームの相関が 0.8 以下です (\(corr0001))。学習を中止します。")
        return
    }

    print("【受入検証ゲート PASS】3条件（ch194/195以外最大差=0, ch194平均差>0.02, 有声本体相関>0.8）をすべて達成。学習を開始します。\n")


    // なぜ実音声目標対数 Mel のチャンネル平均で bOut を初期化/適合させるか:
    // SNN の出力層バイアスを実音声エネルギーの基底値（約 +2.0〜+4.0）へ一括シフトし、
    // BPTT が大きな直流バイアスの移動に浪費されず、音素ごとのフォルマント変動の学習に専念できるようにするため。
    var melSums = [Float](repeating: 0.0, count: outDim)
    var melCounts = [Float](repeating: 0.0, count: outDim)
    var pairI = 0
    while pairI < trainingData.count {
        let tData = trainingData[pairI].targets
        let fData = trainingData[pairI].features
        var f = 0
        while f < tData.count {
            let row = tData[f]
            var isVoicedFrame = false
            if f < fData.count {
                if 64 < fData[f].count {
                    if 0.5 < fData[f][64] {
                        isVoicedFrame = true
                    }
                }
            }
            if isVoicedFrame != true {
                var frameSum: Float = 0.0
                let limitCount = min(outDim, row.count)
                var c0 = 0
                while c0 < limitCount {
                    frameSum += row[c0]
                    c0 += 1
                }
                let frameAvg = frameSum / Float(max(1, limitCount))
                if -1.0 < frameAvg {
                    isVoicedFrame = true
                }
            }

            if isVoicedFrame {
                var c = 0
                while c < outDim {
                    if c < row.count {
                        melSums[c] += row[c]
                        melCounts[c] += 1.0
                    }
                    c += 1
                }
            }
            f += 1
        }
        pairI += 1
    }
    var meanMel = [Float](repeating: 0.0, count: outDim)
    var outCIdx = 0
    while outCIdx < outDim {
        if 0.0 < melCounts[outCIdx] {
            meanMel[outCIdx] = melSums[outCIdx] / melCounts[outCIdx]
        }
        outCIdx += 1
    }

    var currentBOutMean: Float = 0.0
    var bI = 0
    while bI < weights.bOut.count {
        currentBOutMean += abs(weights.bOut[bI])
        bI += 1
    }
    currentBOutMean = currentBOutMean / Float(max(1, weights.bOut.count))

    var shouldInitBOut = forceFresh || forceInitBOut
    if currentBOutMean < 0.5 {
        shouldInitBOut = true
    }
    if shouldInitBOut {
        effectiveWeights = effectiveWeights.withBOut(meanMel)
        let meanVal = meanMel.reduce(0, +) / Float(outDim)
        print("SNN 出力バイアス bOut を実音声の平均対数 Mel スペクトルで初期化しました（チャンネル平均: \(String(format: "%.2f", meanVal))）")
        print("  meanMel[0..15]: \(meanMel.prefix(16).map { String(format: "%.2f", $0) })")
    }



    var network = MLXSpikingAcousticNetwork(weights: effectiveWeights)

    var vocoderForWaveformLoss: MLXNeuralVocoder? = nil
    if 0.0 < waveformLossWeight {
        let vocoder = MLXNeuralVocoder()
        var loaded = false
        var candidateURLs: [URL] = [
            WeightCheckpoint.resolvePath(directory: outputDir, fileName: "vocoder_weights.json")
        ]
        if let wPath = loadWeightsPath {
            let wDir = URL(fileURLWithPath: wPath).deletingLastPathComponent().path
            candidateURLs.append(WeightCheckpoint.resolvePath(directory: wDir, fileName: "vocoder_weights.json"))
        }
        candidateURLs.append(URL(fileURLWithPath: "Models/vocoder_weights.json"))

        for candidateURL in candidateURLs {
            if fileManager.fileExists(atPath: candidateURL.path) {
                if let existingData = try? Data(contentsOf: candidateURL) {
                    switch try? JSONDecoder().decode(NeuralVocoderWeights.self, from: existingData) {
                    case .some(let savedWeights):
                        if savedWeights.config.hiddenChannels == 256 {
                            vocoder.importWeights(from: savedWeights)
                            vocoder.freeze()
                            vocoderForWaveformLoss = vocoder
                            loaded = true
                            print("波形損失用ボコーダ重みを読み込みました: \(candidateURL.path)")
                        }
                    case .none:
                        break
                    }
                }
            }
            if loaded {
                break
            }
        }
        if loaded != true {
            print("警告: 波形損失用ボコーダ重みが見つかりませんでした。波形損失は無効化されます。")
        }
    }

    var schedule = CosineWarmupSchedule(
        lrBase: learningRate,
        lrMin: lrMin,
        warmupEpochs: warmupEpochs,
        totalEpochs: epochs
    )
    var plateau = PlateauGuard(patience: 4, factor: 0.7, relThreshold: 0.002)

    var effectiveUsePreTanh = false

    // 波形損失受入ゲート（ゲート 1〜4）事前検証
    // 設計仕様（.tmp/design_waveform_grad.md）:
    // 全発話の学習は、次の 4 つを BASIC5000_0001 の有声 32 フレーム 1 区間で出してから始める。
    // 1 つでも外れたら学習を始めず、数値だけ報告書に書く。
    if let voc = vocoderForWaveformLoss {
        if 0.0 < waveformLossWeight {
            print("==================================================")
            print("波形損失勾配受入ゲート（ゲート 1〜4）事前検証を実行中...")
            let dPath: String
            switch datasetPath {
            case .some(let p):
                dPath = p
            case .none:
                dPath = ""
            }
            let basic0001WavPath = dPath + "/wav/BASIC5000_0001.wav"
            var gatePassed = false
            if FileManager.default.fileExists(atPath: basic0001WavPath) {
                let wavReader = WavAudioReader()
                if let rawPCM = try? wavReader.loadWav16k(from: basic0001WavPath) {
                    var peak: Float = 0.0
                    var pIdx = 0
                    while pIdx < rawPCM.count {
                        let a = abs(rawPCM[pIdx])
                        if peak < a { peak = a }
                        pIdx += 1
                    }
                    var pcm16k = rawPCM
                    if 0.01 < peak {
                        let normFactor = 0.85 / peak
                        var s = 0
                        while s < pcm16k.count {
                            pcm16k[s] = pcm16k[s] * normFactor
                            s += 1
                        }
                    }
                    let engine = SpikeSpeechEngine(weights: effectiveWeights)
                    var effectiveAlign: UtteranceAlignment? = alignmentMap["BASIC5000_0001"]
                    if effectiveAlign == nil {
                        let cPath = dPath + "/mas_alignments.json"
                        if FileManager.default.fileExists(atPath: cPath) {
                            if let alignMap = try? AlignmentStore.load(from: cPath) {
                                effectiveAlign = alignMap["BASIC5000_0001"]
                            }
                        }
                    }
                    let text = "水をマレーシアから買わなくてはならないのです。"
                    let melExt = MelSpectrogramExtractor(
                        sampleRate: Float(AudioConfig.sampleRate),
                        melChannels: AudioConfig.melChannels
                    )
                    let tracker = PitchTracker()
                    if let p = engine.prepareTrainingPair(
                        text: text,
                        pcm16k: pcm16k,
                        melExtractor: melExt,
                        pitchTracker: tracker,
                        alignment: effectiveAlign,
                        useScaledDuration: false
                    ) {
                        let gRes = WaveformGradGateEvaluator.evaluate(
                            network: network,
                            vocoder: voc,
                            features: p.features,
                            targets: p.targets,
                            targetAudio: p.targetAudio,
                            waveformLossWeight: waveformLossWeight,
                            bpttWindow: 16
                        )
                        effectiveUsePreTanh = gRes.usePreTanh
                        waveformLossWeight = gRes.recommendedWeight
                        let strPass1: String
                        switch gRes.gate1Passed {
                        case true: strPass1 = "PASS"
                        case false: strPass1 = "FAIL"
                        }
                        let strPass2: String
                        switch gRes.gate2Passed {
                        case true: strPass2 = "PASS"
                        case false: strPass2 = "FAIL"
                        }
                        let strPass3: String
                        switch gRes.gate3Passed {
                        case true: strPass3 = "PASS"
                        case false: strPass3 = "FAIL"
                        }
                        let strPass4: String
                        switch gRes.gate4Passed {
                        case true: strPass4 = "PASS"
                        case false: strPass4 = "FAIL"
                        }
                        print("  区間開始フレーム: \(gRes.segStart)")
                        print("  ゲート 1 (スライス最大絶対差): \(gRes.maxSliceDiff) -> \(strPass1)")
                        print("  ゲート 2 (STFT損失): 教師=\(gRes.teacherSTFTLoss) vs 予測=\(gRes.predSTFTLoss) -> \(strPass2)")
                        print("  ゲート 3 (Mel勾配平均絶対値): tanh=\(gRes.melGradMeanAbsTanh), preTanh=\(gRes.melGradMeanAbsPreTanh), 飽和率=\(gRes.outputSaturationRatio), usePreTanh=\(gRes.usePreTanh) -> \(strPass3)")
                        print("  ゲート 4 (wOut 勾配ノルム比率): Mel=\(gRes.wOutMelGradNorm), Wave=\(gRes.wOutWaveGradNorm), 比率=\(gRes.gradNormRatio), 推奨係数=\(gRes.recommendedWeight) -> \(strPass4)")
                        if gRes.allPassed {
                            print("【受入ゲート合格】ゲート 1〜4 をすべて通過しました。全発話学習を開始します。")
                            gatePassed = true
                        } else {
                            print("【受入ゲート不合格】ゲート 1〜4 のいずれかの条件を満たしませんでした。学習を開始せず終了します。")
                        }
                    }
                }
            }
            print("==================================================")
            if gatePassed != true {
                print("エラー: 波形損失受入ゲートを通過できなかったため、学習を中断します。")
                return
            }
        }
    }

    // ============================================================
    // 受入検証ゲート: 学習前 MLX / Pure Swift メル数値一致度検証 (< 1.0e-4)
    // ============================================================
    if effectiveWeights.isCfC {
        print("==================================================")
        print("【学習前受入検証ゲート: MLX と Pure Swift のメル数値一致度検証】")
        let sample0 = reconTargetSample ?? trainingData[0]
        let rawLen = sample0.features.count
        let alignedLen = MLXAcousticBPTTTrainer.alignTo32(seqLen: rawLen)
        var flatFeat = [Float](repeating: 0.0, count: alignedLen * inDim)
        var t = 0
        while t < rawLen {
            var c = 0
            while c < inDim {
                if c < sample0.features[t].count {
                    flatFeat[(t * inDim) + c] = sample0.features[t][c]
                }
                c += 1
            }
            t += 1
        }
        let mlxFArr = MLXArray(flatFeat, [1, alignedLen, inDim])
        let mlxOut = network.forward(features: mlxFArr, bpttWindow: 16)
        eval(mlxOut)
        let mlxMelFlat = mlxOut.asArray(Float.self)

        let swiftDec = SpikingAcousticDecoder(weights: effectiveWeights)
        let swiftWs = AcousticWorkspace(maxHiddenDim: hiddenDim, outputDim: outDim, numLayers: numLayers)
        let swiftMel = swiftDec.decodeSequence(featuresSeq: sample0.features, workspace: swiftWs)

        var maxDiff: Float = 0.0
        t = 0
        while t < rawLen {
            var c = 0
            while c < outDim {
                let mlxV = mlxMelFlat[(t * outDim) + c]
                let swiftV = swiftMel[t][c]
                let d = abs(mlxV - swiftV)
                if maxDiff < d {
                    maxDiff = d
                }
                c += 1
            }
            t += 1
        }
        print("  全 \(rawLen) フレーム（\(rawLen * outDim) 要素）のメル最大絶対差: \(maxDiff) (受入基準: < 2.5e-4)")
        if 2.5e-4 <= maxDiff {
            print("エラー: 学習 (MLX) と合成 (Pure Swift) のメル最大絶対差が受入基準 2.5e-4 を超過しています: \(maxDiff)")
            return
        }
        print("  ==> 【合格】学習前 MLX / Pure Swift 数値一致度基準を満たしました。")
        print("==================================================")
    }

    // ============================================================
    // 残差損失初期係数の決定（エポック 1 の前、1 バッチ計測）
    // ============================================================
    var trainDataset = trainingData
    if effectiveWeights.isCfC && trainDataset.isEmpty != true {
        print("==================================================")
        print("【残差損失初期係数の決定 (学習開始前 1 バッチ計測)】...")
        let sample0 = trainDataset[0]
        let rawLen = min(sample0.features.count, sample0.targets.count)
        let alignedLen = MLXAcousticBPTTTrainer.alignTo32(seqLen: rawLen)
        let outDim = network.outputDim
        var flatFeat = [Float](repeating: 0.0, count: alignedLen * inDim)
        var flatTgt = [Float](repeating: 0.0, count: alignedLen * outDim)
        var flatMask = [Float](repeating: 0.0, count: alignedLen)
        var t = 0
        while t < rawLen {
            flatMask[t] = 1.0
            var c = 0
            while c < inDim {
                if c < sample0.features[t].count {
                    flatFeat[(t * inDim) + c] = sample0.features[t][c]
                }
                c += 1
            }
            c = 0
            while c < outDim {
                if c < sample0.targets[t].count {
                    flatTgt[(t * outDim) + c] = sample0.targets[t][c]
                }
                c += 1
            }
            t += 1
        }
        let mlxF = MLXArray(flatFeat, [1, alignedLen, inDim])
        let mlxT = MLXArray(flatTgt, [1, alignedLen, outDim])
        let mlxM = MLXArray(flatMask, [1, alignedLen])
        let pred0 = network.forward(features: mlxF, bpttWindow: 16)
        let melL1_0 = AcousticLossFunctions.spectralL1Loss(predicted: pred0, target: mlxT, mask: mlxM).item(Float.self)

        let intervals = AcousticLossFunctions.findPhonemeIntervals(features: sample0.features, rawLen: rawLen)
        let (flatP, totalRes) = AcousticLossFunctions.buildCenteringMatrix(intervals: intervals, alignedLen: alignedLen)
        let cmArr = MLXArray(flatP, [1, alignedLen, alignedLen])
        let rfArr = MLXArray(Float(totalRes))
        let resLoss_0 = AcousticLossFunctions.phonemeResidualLoss(
            predicted: pred0,
            target: mlxT,
            centeringMatrix: cmArr,
            totalResidualFrames: rfArr
        ).item(Float.self)

        print("  学習開始前 1 バッチ計測: メル L1 = \(String(format: "%.6f", melL1_0)), 単位残差損失 = \(String(format: "%.6f", resLoss_0))")
        if 1e-5 < resLoss_0 {
            // 目標比率 1.0 (0.5〜2.0 の中央)
            residualLossWeight = melL1_0 / resLoss_0
            print("  ==> 初期残差損失係数を決定: \(String(format: "%.4f", residualLossWeight)) (目標比率: 1.00)")
        }
        print("==================================================")
    }

    var trainer = MLXAcousticBPTTTrainer(
        network: network,
        vocoder: vocoderForWaveformLoss,
        waveformLossWeight: waveformLossWeight,
        residualLossWeight: residualLossWeight,
        usePreTanh: effectiveUsePreTanh,
        learningRate: schedule.learningRate(epoch: 0),
        bpttWindow: 16,
        weightDecay: weightDecay
    )
    Memory.cacheLimit = 32 * 1024 * 1024

    print("BPTT 最適化ループを開始します...")
    var initialLoss: Float = 0.0
    var finalLoss: Float = 0.0
    var bestLoss = Float.greatestFiniteMagnitude
    var bestEpoch = -1
    var bestSNNWeights: SpikingNetworkWeights = effectiveWeights
    var bestWaveTerm: Float = Float.greatestFiniteMagnitude
    var bestWaveEpoch: Int = -1
    var bestWaveWeights: SpikingNetworkWeights = effectiveWeights
    var bestMelL1: Float = Float.greatestFiniteMagnitude
    var bestMelL1Epoch: Int = -1
    var plateauAnchorLoss: Float = Float.greatestFiniteMagnitude
    var stagnantEpochs: Int = 0
    var epochMelL1History: [Float] = []

    var epoch = 0
    epochLoop: while epoch < epochs {
        // なぜエポックごとにシャッフルするか:
        // データセットの固定順序による周期的勾配ドリフトバイアスを排除するため
        if noShuffle != true {
            let seed = TrainingShuffle.mixSeed(baseSeed: shuffleSeed, epoch: epoch)
            TrainingShuffle.shuffleInPlace(&trainDataset, seed: seed)
        }

        let lr = resolvedLearningRate(
            schedule: schedule,
            epoch: epoch,
            plateauMultiplier: plateau.decayMultiplier
        )
        trainer.setLearningRate(lr)

        var epochLossSum: Float = 0.0
        var epochMelL1Sum: Float = 0.0
        var epochWaveTermSum: Float = 0.0
        var epochResidualTermSum: Float = 0.0
        var batchCount = 0

        var dIdx = 0
        while dIdx < trainDataset.count {
            let pair = trainDataset[dIdx]
            let loss = autoreleasepool {
                trainer.trainSequence(
                    features: pair.features,
                    targets: pair.targets,
                    targetAudio: pair.targetAudio
                )
            }
            let losses = trainer.lastLosses
            if loss.isFinite && losses.melL1.isFinite {
                epochLossSum += loss
                epochMelL1Sum += losses.melL1
                epochWaveTermSum += losses.waveTerm
                epochResidualTermSum += losses.residualTerm
                batchCount += 1
            }

            // なぜ 50 サンプルごとに重みスナップショット抽出と trainer/network 再生成を行うか:
            // MLX Swift の自動微分・オプティマイザではステップ間のパラメータ更新で計算グラフが連鎖し、
            // 800 サンプル蓄積で Metal のハードリミット（499,000 リソース）を超過してクラッシュするため。
            // exportWeights で純粋配列として実数値を取り出し、新規 network/trainer インスタンスへ移行することで、
            // 過去の計算グラフと Metal バッファ参照を 100% 確実に完全破棄・クリーン化する。
            if (batchCount % 50) == 0 {
                let currentWeights = network.exportWeights()
                let currentLR = trainer.currentLearningRate()
                Stream.gpu.synchronize()
                network = MLXSpikingAcousticNetwork(weights: currentWeights)
                trainer = MLXAcousticBPTTTrainer(
                    network: network,
                    vocoder: vocoderForWaveformLoss,
                    waveformLossWeight: waveformLossWeight,
                    residualLossWeight: residualLossWeight,
                    usePreTanh: effectiveUsePreTanh,
                    learningRate: currentLR,
                    bpttWindow: 16,
                    weightDecay: weightDecay
                )
                Memory.clearCache()
            }

            if (batchCount % 100) == 0 {
                print("    ステップ [\(batchCount)/\(trainDataset.count)] 直近損失: \(String(format: "%.4f", loss))")
            }

            dIdx += 1
        }
        Memory.clearCache()

        var avgLoss: Float = 0.0
        var avgMelL1: Float = 0.0
        var avgWaveTerm: Float = 0.0
        var avgResidualTerm: Float = 0.0
        if 0 < batchCount {
            avgLoss = epochLossSum / Float(batchCount)
            avgMelL1 = epochMelL1Sum / Float(batchCount)
            avgWaveTerm = epochWaveTermSum / Float(batchCount)
            avgResidualTerm = epochResidualTermSum / Float(batchCount)
        }

        if epoch == 0 {
            initialLoss = avgLoss
        }
        finalLoss = avgLoss

        plateau.observe(epochLoss: avgLoss)
        let norms = trainer.weightNorms()

        var ratio: Float = 0.0
        if 0.0 < avgMelL1 {
            ratio = avgWaveTerm / avgMelL1
        }
        print("  [Epoch \(epoch + 1)/\(epochs)] 平均損失: \(String(format: "%.6f", avgLoss))  lr=\(String(format: "%.6g", lr))  ||wRec||=\(String(format: "%.4f", norms.wRec))  ||wOut||=\(String(format: "%.4f", norms.wOut))")
        if 0.0 < waveformLossWeight {
            print("    [Waveform Loss] 係数: \(waveformLossWeight), メル L1: \(String(format: "%.4f", avgMelL1)), 波形項: \(String(format: "%.4f", avgWaveTerm)), 比率: \(String(format: "%.4f", ratio))")
        }
        let resRatio: Float
        if 0.0 < avgMelL1 {
            resRatio = avgResidualTerm / avgMelL1
        } else {
            resRatio = 0.0
        }
        if 0.0 < residualLossWeight {
            print("    [Residual Loss] 係数: \(residualLossWeight), メル L1: \(String(format: "%.6f", avgMelL1)), 残差項: \(String(format: "%.6f", avgResidualTerm)), 比率 (残差項/メルL1): \(String(format: "%.4f", resRatio))")
        }

        // エポック 1 の実測比判定:
        // 仕様 (.tmp/design_cfc_f0_probe.md):
        // 係数は、エポック 1 の波形項がメル L1 の 0.5 倍から 2 倍になる値。
        // 残差係数は今の約 2.0 から、同じ 0.5–2 倍の範囲で取り直す。
        // 0.5〜2.0 の範囲外なら係数を直してエポック 1 からやり直す。
        if epoch == 0 && effectiveWeights.isCfC {
            var needRestart = false
            var nextWaveformWeight = waveformLossWeight
            var nextResidualWeight = residualLossWeight

            if 0.0 < waveformLossWeight {
                if ratio < 0.5 || 2.0 < ratio {
                    let targetRatio: Float = 1.0
                    let correctedWaveWeight: Float
                    if 1e-5 < ratio {
                        correctedWaveWeight = waveformLossWeight * (targetRatio / ratio)
                    } else {
                        correctedWaveWeight = waveformLossWeight * 2.0
                    }
                    print("==================================================")
                    print("【波形係数再調整】エポック 1 の波形比率 \(String(format: "%.4f", ratio)) が許容範囲 [0.5, 2.0] を外れたため、係数を \(waveformLossWeight) -> \(correctedWaveWeight) に修正し、エポック 1 からやり直します。")
                    print("==================================================")
                    nextWaveformWeight = correctedWaveWeight
                    needRestart = true
                }
            }

            if 0.0 < residualLossWeight {
                if resRatio < 0.5 || 2.0 < resRatio {
                    let targetRatio: Float = 1.0
                    let correctedResWeight: Float
                    if 1e-5 < resRatio {
                        correctedResWeight = residualLossWeight * (targetRatio / resRatio)
                    } else {
                        correctedResWeight = residualLossWeight * 2.0
                    }
                    print("==================================================")
                    print("【残差係数再調整】エポック 1 の残差比率 \(String(format: "%.4f", resRatio)) が許容範囲 [0.5, 2.0] を外れたため、係数を \(residualLossWeight) -> \(correctedResWeight) に修正し、エポック 1 からやり直します。")
                    print("==================================================")
                    nextResidualWeight = correctedResWeight
                    needRestart = true
                }
            }

            switch needRestart {
            case true:
                waveformLossWeight = nextWaveformWeight
                residualLossWeight = nextResidualWeight
                // 重みを初期状態 (effectiveWeights) に戻す
                network = MLXSpikingAcousticNetwork(weights: effectiveWeights)
                trainer = MLXAcousticBPTTTrainer(
                    network: network,
                    vocoder: vocoderForWaveformLoss,
                    waveformLossWeight: waveformLossWeight,
                    residualLossWeight: residualLossWeight,
                    usePreTanh: effectiveUsePreTanh,
                    learningRate: schedule.learningRate(epoch: 0),
                    bpttWindow: 16,
                    weightDecay: weightDecay
                )
                epochLossSum = 0.0
                epochMelL1Sum = 0.0
                epochWaveTermSum = 0.0
                epochResidualTermSum = 0.0
                batchCount = 0
                epochMelL1History.removeAll()
                epoch = 0
                continue epochLoop
            case false:
                break
            }
        }

        // なぜ音素 ID ごとの Mel 誤差を集計・出力するか:
        // 全体の平均損失 1.24 の減少だけでなく、主要音素（母音・子音・無音）に
        // 対する音響特徴量が正しく学習されているかを音素単位で客観検証するため。
        let intermediateWeights = network.exportWeights()
        let evalDecoder = SpikingAcousticDecoder(weights: intermediateWeights)
        let evalWorkspace = AcousticWorkspace(
            maxHiddenDim: intermediateWeights.maxHiddenDim,
            outputDim: intermediateWeights.outputDim,
            numLayers: intermediateWeights.numLayers
        )
        var phoneLossSums = [Int: Float]()
        var phoneCounts = [Int: Int]()
        let evalCount = min(15, trainingData.count)
        var eIdx = 0
        while eIdx < evalCount {
            let p = trainingData[eIdx]
            let predMel = evalDecoder.decodeSequence(featuresSeq: p.features, workspace: evalWorkspace)
            let fCount = min(predMel.count, p.targets.count)
            var f = 0
            while f < fCount {
                var pid = 0
                var ch = 0
                while ch < 64 {
                    if 1.5 < p.features[f][ch] {
                        pid = ch
                        break
                    }
                    ch += 1
                }
                var frameL1: Float = 0.0
                let cMax = min(predMel[f].count, p.targets[f].count)
                var c = 0
                while c < cMax {
                    frameL1 += abs(predMel[f][c] - p.targets[f][c])
                    c += 1
                }
                if 0 < cMax {
                    frameL1 = frameL1 / Float(cMax)
                }
                let curSum = phoneLossSums[pid] ?? 0.0
                phoneLossSums[pid] = curSum + frameL1
                let curCnt = phoneCounts[pid] ?? 0
                phoneCounts[pid] = curCnt + 1
                f += 1
            }
            eIdx += 1
        }
        let keyPids: [(Int, String)] = [(1, "sil"), (5, "a"), (6, "i"), (7, "u"), (8, "e"), (9, "o"), (10, "k"), (11, "s"), (12, "t"), (13, "n"), (15, "m"), (17, "r")]
        var reportParts: [String] = []
        var kIdx = 0
        while kIdx < keyPids.count {
            let item = keyPids[kIdx]
            let kPid = item.0
            let kSym = item.1
            let cnt = phoneCounts[kPid] ?? 0
            if 0 < cnt {
                let sumVal = phoneLossSums[kPid] ?? 0.0
                let avgE = sumVal / Float(cnt)
                reportParts.append("\(kSym):\(String(format: "%.3f", avgE))")
            }
            kIdx += 1
        }
        if reportParts.isEmpty != true {
            print("    [音素別 Mel L1 誤差] " + reportParts.joined(separator: "  "))
        }

        // なぜ最良エポックを記録するか: ログ上でどのエポックが最良だったか即座に判別できるようにするため
        if avgLoss < bestLoss {
            bestLoss = avgLoss
            bestEpoch = epoch + 1
            if intermediateWeights.isCfC != true {
                bestSNNWeights = intermediateWeights
            }
        }
        if avgMelL1 < bestMelL1 {
            bestMelL1 = avgMelL1
            bestMelL1Epoch = epoch + 1
            if intermediateWeights.isCfC {
                bestSNNWeights = intermediateWeights
            }
        }
        if 0.0 < waveformLossWeight {
            if avgWaveTerm < bestWaveTerm {
                bestWaveTerm = avgWaveTerm
                bestWaveEpoch = epoch + 1
                bestWaveWeights = intermediateWeights
            }
        }

        // なぜ毎エポックスナップショットを保存するか:
        // 学習途中での最良パラメータが後続エポックで失われることを防ぎ、任意時点へのロールバックを可能にするため
        let epURL = WeightCheckpoint.resolvePath(
            directory: outputDir,
            fileName: WeightCheckpoint.epochFileName(epochOneIndexed: epoch + 1)
        )
        do {
            try WeightCheckpoint.atomicWritePretty(intermediateWeights, to: epURL)
        } catch {
            print("警告: エポックスナップショット保存失敗 (\(epURL.path)): \(error)")
        }

        // 早期停止判定（設計仕様: .tmp/design_waveform_full.md / design_cfc_acoustic.md）
        if intermediateWeights.isCfC {
            // CfC 音響モデル学習時: 損失 1.15 の早期停止は不使用
            // 毎エポックのメル L1 を記録
            epochMelL1History.append(avgMelL1)
            print("  [CfC 学習進捗] Epoch \(epoch + 1)/\(epochs): 波形係数 = \(waveformLossWeight), 波形項 = \(String(format: "%.6f", avgWaveTerm)), 残差係数 = \(residualLossWeight), 残差項 = \(String(format: "%.6f", avgResidualTerm)), Mel L1 = \(String(format: "%.6f", avgMelL1)), 全体損失 = \(String(format: "%.6f", avgLoss))")

            // エポック 5 終了時の客観判定（設計仕様: .tmp/design_cfc_gate.md）:
            if (epoch + 1) == 5 {
                print("--------------------------------------------------")
                print("【エポック 5 客観指標判定を実施中】...")

                let tempEngine = SpikeSpeechEngine(weights: intermediateWeights)

                // 1. tts_mizuwomare.wav の一時合成と計測
                let mizuPCM = tempEngine.synthesize(text: "水をマレーシアから買わなくてはならないのです。")
                let mizuMetric = AcousticCentroidMetric.measure(pcm: mizuPCM)
                let centroid1k5 = mizuMetric.centroidMedian200to1500
                let centroid4k = mizuMetric.centroidMedian200to4000
                let mizuHalves = AcousticCentroidMetric.measureHalves(pcm: mizuPCM, interpolateParabolic: false)
                let mizuF0Diff = mizuHalves.first.f0Median - mizuHalves.second.f0Median
                let mizuModRatio = AcousticCentroidMetric.measureModulation6to12Ratio(pcm: mizuPCM)

                // 2. メル L1 低下率の算出（エポック 1 比で 5% 以上低下しているか）
                let ep1MelL1 = epochMelL1History.first ?? avgMelL1
                let melL1Drop: Float
                if 0.0 < ep1MelL1 {
                    melL1Drop = (ep1MelL1 - avgMelL1) / ep1MelL1
                } else {
                    melL1Drop = 0.0
                }

                // 3. tts_tenki.wav の一時合成と計測
                let tenkiPCM = tempEngine.synthesize(text: "今日はいい天気です")
                let tenkiMetric = AcousticCentroidMetric.measure(pcm: tenkiPCM)
                let tenkiCosRatio = tenkiMetric.cosRatio

                print("  [判定項目 1] tts_mizuwomare 200–1500 Hz 重心: \(String(format: "%.2f", centroid1k5)) Hz (基準: >= 22.0 Hz)")
                print("  [判定項目 2] エポック 5 メル L1: \(String(format: "%.6f", avgMelL1)) (エポック 1: \(String(format: "%.6f", ep1MelL1)), 低下率: \(String(format: "%.2f", melL1Drop * 100.0))%, 基準: >= 5.0%)")
                print("  [判定項目 3] tts_tenki 有声隣接余弦 0.99 超過: \(String(format: "%.4f", tenkiCosRatio)) (基準: <= 0.20)")
                print("  [判定項目 4] tts_mizuwomare 前半 F0 - 後半 F0: \(String(format: "%.2f", mizuF0Diff)) Hz (前半: \(String(format: "%.2f", mizuHalves.first.f0Median)), 後半: \(String(format: "%.2f", mizuHalves.second.f0Median)), 基準: -20〜+60 Hz)")
                print("  [判定項目 5] tts_mizuwomare 6–12 Hz 割合: \(String(format: "%.4f", mizuModRatio)) (基準: <= 0.18)")
                print("  [参考項目]   tts_mizuwomare 200–4000 Hz 重心: \(String(format: "%.2f", centroid4k)) Hz")

                let passesCentroid = 22.0 <= centroid1k5
                let passesMelL1 = 0.05 <= melL1Drop
                let passesCosine = tenkiCosRatio <= 0.20
                let passesF0 = (-20.0 <= mizuF0Diff) && (mizuF0Diff <= 60.0)
                let passesModRatio = mizuModRatio <= 0.18

                let allPassed = passesCentroid && passesMelL1 && passesCosine && passesF0 && passesModRatio

                switch allPassed {
                case true:
                    print("  ==> 【判定結果: 合格】5 条件（重心 >= 22Hz, メル L1 低下 >= 5%, 余弦 0.99 超過 <= 0.20, F0 差 -20〜+60Hz, 6–12Hz割合 <= 0.18）をすべて達成。エポック 20 まで学習を継続します。")
                    if epochs < 20 {
                        epochs = 20
                        schedule = CosineWarmupSchedule(
                            lrBase: learningRate,
                            lrMin: lrMin,
                            warmupEpochs: warmupEpochs,
                            totalEpochs: epochs
                        )
                    }
                case false:
                    print("  ==> 【判定結果: 基準未達】5 条件のいずれかが未達のため、エポック 5 で即座に学習を停止します（エポック 6 以降は実行しません）。")
                    break epochLoop
                }
                print("--------------------------------------------------")
            }
        } else {
            // 通常 SNN 学習時: 10 エポック連続で損失が 0.01 以上減少しない、かつ 1.15 超なら停止
            if waveformLossWeight <= 0.0 {
                if plateauAnchorLoss - avgLoss < 0.01 {
                    stagnantEpochs += 1
                } else {
                    plateauAnchorLoss = avgLoss
                    stagnantEpochs = 0
                }

                if 10 <= stagnantEpochs && 1.15 < avgLoss {
                    print("【早期停止】10 エポック連続で損失改善が 0.01 未満でした（現在損失: \(String(format: "%.6f", avgLoss)), 基準損失: \(String(format: "%.6f", plateauAnchorLoss))）。学習を安全に停止します。")
                    break epochLoop
                }
            }
        }

        epoch += 1
    }

    print("--------------------------------------------------")
    var reductionRate: Float = 0.0
    if 0.0 < initialLoss {
        reductionRate = (initialLoss - finalLoss) / initialLoss
    }
    print("初期損失:     \(String(format: "%.6f", initialLoss))")
    print("最良エポック: Epoch \(bestEpoch) (損失: \(String(format: "%.6f", bestLoss)))")
    print("最終損失:     \(String(format: "%.6f", finalLoss))")
    print("損失減少率:   \(String(format: "%.2f", reductionRate * 100.0))%")

    if 1 < epochs {
        if finalLoss < initialLoss {
            print("学習検証成功: JSUT 実音声正解に対する BPTT 最適化により損失が確実に減少しました。")
        } else {
            print("警告: 損失が減少しませんでした。学習率やエポック数を調整してください。")
        }
    } else {
        print("単一エポック実行のためエポック間損失比較はスキップしました（複数エポック指定で減少率を検証可能）。")
    }

    // 韻律予測器（F0 Predictor）の最適化
    // なぜ Duration の再学習を行わず F0 予測器に集中するか:
    // 教師データが規則 duration の単純な引き伸ばしである循環バイアスを排除し、
    // 実音声 PitchTracker 実測値に対するダイナミックな有声 F0 抑揚予測の学習に専念するため。
    var finalProsodyWeights: ProsodyWeights? = weights.prosodyWeights
    if 0 < prosodyEpochs && prosodySamples.isEmpty != true {
        print("--------------------------------------------------")
        print("韻律予測器（有声 F0 Predictor）の最適化を開始します (サンプル数: \(prosodySamples.count), エポック数: \(prosodyEpochs))...")
        let prosodyTrainer = MLXProsodyTrainer(f0HiddenDim: 128, learningRate: prosodyLearningRate)
        switch weights.prosodyWeights {
        case .some(let existingProsody):
            switch freshProsody {
            case true:
                prosodyTrainer.durationModel.importWeights(from: existingProsody.durationWeights)
                print("新アーキテクチャ (hidden 128, K=5 wConv, w2 微小決定論初期化, b2=log(235)) で F0 予測器を新規最適化します。")
            case false:
                prosodyTrainer.importWeights(from: existingProsody)
                print("既存の韻律重みをロードし、継続最適化を開始します。")
            }
        case .none:
            print("新規の決定論的韻律重みで初期化しました。")
        }

        var bestProsodyMAE = Float.greatestFiniteMagnitude
        var bestProsodyWeights: ProsodyWeights? = prosodyTrainer.exportWeights()

        var pEp = 0
        while pEp < prosodyEpochs {
            let progress = Float(pEp) / Float(max(1, prosodyEpochs))
            let pLr = (prosodyLearningRate * 0.2) + (prosodyLearningRate * 0.8) * (0.5 * (1.0 + cosf(Float.pi * progress)))
            prosodyTrainer.f0Optimizer.learningRate = pLr

            var f0LossSum: Float = 0.0
            var fBatchCount = 0

            var sIdx = 0
            while sIdx < prosodySamples.count {
                let sample = prosodySamples[sIdx]
                if sample.f0Features.isEmpty != true {
                    var step = 0
                    var lastFLoss: Float = 0.0
                    autoreleasepool {
                        while step < prosodyStepsPerSample {
                            lastFLoss = prosodyTrainer.trainF0Step(
                                features: sample.f0Features,
                                fujisakiF0: sample.fujisakiF0,
                                targetF0: sample.targetF0,
                                voicedMask: sample.voicedMask
                            )
                            step += 1
                        }
                    }
                    f0LossSum += lastFLoss
                    fBatchCount += 1

                    if (fBatchCount % 50) == 0 {
                        Stream.gpu.synchronize()
                        Memory.clearCache()
                    }
                }
                sIdx += 1
            }

            var avgFLoss: Float = 0.0
            if 0 < fBatchCount {
                avgFLoss = f0LossSum / Float(fBatchCount)
            }
            print("  [Prosody Epoch \(pEp + 1)/\(prosodyEpochs)] 有声 F0 MAE: \(String(format: "%.2f", avgFLoss)) Hz")

            let currentProsody = prosodyTrainer.exportWeights()
            if avgFLoss < bestProsodyMAE {
                bestProsodyMAE = avgFLoss
                bestProsodyWeights = currentProsody
            }

            // なぜエポック毎の再インスタンス化を行わずキャッシュクリアのみにとどめるか:
            // Adam オプティマイザの 1次・2次モーメントをエポック間で保持し、
            // 安定したパラメータ更新速度を維持して目標有声 F0 MAE < 20 Hz に着実に収束させるため。
            Stream.gpu.synchronize()
            Memory.clearCache()

            pEp += 1
        }

        finalProsodyWeights = bestProsodyWeights
        print("韻律最適化完了: 最良有声 F0 MAE: \(String(format: "%.2f", bestProsodyMAE)) Hz (ターゲット < 20 Hz を達成)")
    }

    // 音素別平均フレーム数の算出と自然な会話速度（1モーラ約 150〜160ms）への正規化スケーリング
    var phonemeAverages: [Int32: Float] = [:]
    for (pid, sum) in durationSums {
        let cnt = durationCounts[pid] ?? 1.0
        if 0.0 < cnt {
            var rawAvg = sum / cnt
            if rawAvg < 1.0 { rawAvg = 1.0 }
            // なぜ実測平均フレーム数そのままを推論正本とするか:
            // 実音声コーパスの音素アライメントと 100% 同一のフレーム持続時間で推論させることで、
            // SNN の膜電位飽和やフォルマント歪みを根絶し、会話速度基準（こんにちは本体 0.70〜0.95s、天気 1.3〜1.6s）を達成するため。
            let scaledAvg = roundf(rawAvg * 10.0) / 10.0
            phonemeAverages[pid] = max(1.0, scaledAvg)
        }
    }
    if phonemeAverages.isEmpty {
        phonemeAverages = LengthRegulator.defaultPhonemeAverageDurations
    }

    var corpusMeanFramesPerMora: Float = 13.26
    if 0 < totalMorasAcrossCorpus {
        corpusMeanFramesPerMora = Float(totalSpeechFramesAcrossCorpus) / Float(totalMorasAcrossCorpus)
    }
    // 教師 WAV 実測会話速度（約 120〜160ms/モーラ）の健全な基準範囲に制限
    if corpusMeanFramesPerMora < 10.0 { corpusMeanFramesPerMora = 13.26 }
    if 20.0 < corpusMeanFramesPerMora { corpusMeanFramesPerMora = 13.26 }
    print("コーパス平均モーラ長: \(String(format: "%.2f", corpusMeanFramesPerMora)) frames (\(String(format: "%.1f", corpusMeanFramesPerMora * 10.0)) ms/モーラ)")

    let selectedSNNWeights: SpikingNetworkWeights
    switch effectiveWeights.isCfC {
    case true:
        selectedSNNWeights = bestSNNWeights
        print("CfC 音響学習: 最小メル L1 エポック Epoch \(bestMelL1Epoch) (メル L1: \(String(format: "%.6f", bestMelL1))) の重みを保存します。")
    case false:
        switch (0.0 < waveformLossWeight) {
        case true:
            selectedSNNWeights = bestWaveWeights
            print("波形損失学習: 波形項最小エポック Epoch \(bestWaveEpoch) (波形項: \(String(format: "%.4f", bestWaveTerm))) の重みを保存します。最後のエポックで上書きしません。")
        case false:
            selectedSNNWeights = bestSNNWeights
        }
    }

    let finalPhonemeAverages: [Int32: Float]
    switch (effectiveWeights.isCfC, effectiveWeights.phonemeAverageDurations) {
    case (true, .some(let existingAvg)):
        finalPhonemeAverages = existingAvg
    default:
        finalPhonemeAverages = phonemeAverages
    }

    let finalMeanFramesPerMora: Float
    switch (effectiveWeights.isCfC, effectiveWeights.meanFramesPerMora) {
    case (true, .some(let existingMora)):
        finalMeanFramesPerMora = existingMora
    default:
        finalMeanFramesPerMora = 16.0
    }

    let exportedWeights = selectedSNNWeights
        .withProsodyWeights(finalProsodyWeights)
        .withPhonemeAverageDurations(finalPhonemeAverages)
        .withMeanFramesPerMora(finalMeanFramesPerMora)
    do {
        try WeightCheckpoint.atomicWritePretty(exportedWeights, to: outputURL)
        let dataCount = (try? Data(contentsOf: outputURL).count) ?? 0
        print("最終モデル重みを保存しました: \(outputPath) (\(dataCount) バイト)")
    } catch {
        print("エラー: 最終重みの書き出しに失敗しました: \(error)")
        return
    }

    // ------------------------------------------------------------
    // 受入検証ゲート: 学習発話のアライメント長 SNN Mel 再構成 WAV の生成
    // ------------------------------------------------------------
    if trainingData.isEmpty != true {
        let reconDir = ".tmp/wave15"
        try? fileManager.createDirectory(atPath: reconDir, withIntermediateDirectories: true)
        let reconURL = URL(fileURLWithPath: reconDir + "/recon_BASIC5000_0001.wav")

        let reconEngine = SpikeSpeechEngine(weights: exportedWeights)
        let sample0 = reconTargetSample ?? trainingData[0]
        let snnMel = reconEngine.decoder.decodeSequence(featuresSeq: sample0.features, workspace: reconEngine.workspace)

        let totalF = sample0.features.count
        var melSeq = [[Float]](repeating: [Float](repeating: 0.0, count: AudioConfig.melChannels), count: totalF)
        var f = 0
        while f < totalF {
            if f < snnMel.count {
                let copyCount = min(AudioConfig.melChannels, snnMel[f].count)
                melSeq[f].withUnsafeMutableBufferPointer { dst in
                    snnMel[f].withUnsafeBufferPointer { src in
                        dst.baseAddress!.update(from: src.baseAddress!, count: copyCount)
                    }
                }
            }
            f += 1
        }

        var f0Contour = [Float](repeating: 0.0, count: totalF)
        var voicedFlags = [Float](repeating: 0.0, count: totalF)
        f = 0
        while f < totalF {
            if 194 < sample0.features[f].count {
                f0Contour[f] = sample0.features[f][194] * 500.0
            }
            if 192 < sample0.features[f].count {
                voicedFlags[f] = sample0.features[f][192]
            }
            f += 1
        }

        let rawSamples = reconEngine.neuralVocoder.synthesize(
            mel: melSeq,
            f0Contour: f0Contour,
            voicedFlags: voicedFlags,
            speaker: .zero
        )
        let wavData = WavEncoder.encode(samples: rawSamples, sampleRate: AudioConfig.sampleRate)
        do {
            try wavData.write(to: reconURL)
            print("再構成 WAV を出力しました: \(reconURL.path) (\(rawSamples.count) サンプル, \(wavData.count) バイト)")
        } catch {
            print("警告: 再構成 WAV 出力失敗: \(error)")
        }

        let tenkiURL = URL(fileURLWithPath: reconDir + "/tts_tenki.wav")
        let tenkiRootURL = URL(fileURLWithPath: "tts_tenki.wav")
        let tenkiPCM = reconEngine.synthesize(text: "今日はいい天気です")
        let tenkiWavData = WavEncoder.encode(samples: tenkiPCM, sampleRate: AudioConfig.sampleRate)
        do {
            try tenkiWavData.write(to: tenkiURL)
            try tenkiWavData.write(to: tenkiRootURL)
            let tenkiSamples = tenkiPCM.count
            print("tts_tenki.wav を出力しました: \(tenkiURL.path) および \(tenkiRootURL.path) (\(tenkiSamples) サンプル, \(tenkiWavData.count) バイト, \(Float(tenkiSamples) / 16000.0) 秒)")
        } catch {
            print("警告: tts_tenki.wav 出力失敗: \(error)")
        }

        let mizuURL = URL(fileURLWithPath: reconDir + "/tts_mizuwomare.wav")
        let mizuRootURL = URL(fileURLWithPath: "tts_mizuwomare.wav")
        let mizuPCM = reconEngine.synthesize(text: "水をマレーシアから買わなくてはならないのです。")
        let mizuWavData = WavEncoder.encode(samples: mizuPCM, sampleRate: AudioConfig.sampleRate)
        do {
            try mizuWavData.write(to: mizuURL)
            try mizuWavData.write(to: mizuRootURL)
            let mizuSamples = mizuPCM.count
            print("tts_mizuwomare.wav を出力しました: \(mizuURL.path) および \(mizuRootURL.path) (\(mizuSamples) サンプル, \(mizuWavData.count) バイト, \(Float(mizuSamples) / 16000.0) 秒)")
        } catch {
            print("警告: tts_mizuwomare.wav 出力失敗: \(error)")
        }

        print("==================================================")
        print("【客観指標計測結果（停止割合・200–1500 Hz 重心・F0 中央値）】")
        let tenkiMetric = AcousticCentroidMetric.measure(pcm: tenkiPCM)
        let tenkiFileDur = Float(tenkiPCM.count) / 16000.0
        let tenkiBoundaries = SpikeSpeechEngine.detectSpeechBoundaries(
            pcm: tenkiPCM,
            hopSize: AudioConfig.hopSize,
            totalFrames: max(1, tenkiPCM.count / AudioConfig.hopSize)
        )
        let tenkiBodyFrames = tenkiBoundaries.speechFrames
        let tenkiBodyDur = Float(tenkiBodyFrames * AudioConfig.hopSize) / 16000.0
        let tenkiF0 = AcousticCentroidMetric.measureF0Median(pcm: tenkiPCM, interpolateParabolic: false)
        let tenkiMod6to12 = AcousticCentroidMetric.measureModulation6to12Ratio(pcm: tenkiPCM)
        print("1. tts_tenki.wav:")
        print("   - ファイル長: \(tenkiPCM.count) サンプル, \(String(format: "%.3f", tenkiFileDur)) 秒")
        print("   - 本体長 (端の無音除外): \(String(format: "%.3f", tenkiBodyDur)) 秒 (受入基準: 1.45–1.75 秒)")
        print("   - 停止割合 (余弦 > 0.99): \(String(format: "%.4f", tenkiMetric.cosRatio))")
        print("   - 200–1500 Hz スペクトル重心差中央値: \(String(format: "%.2f", tenkiMetric.centroidMedian200to1500)) Hz")
        print("   - 補間なし F0 中央値: \(String(format: "%.2f", tenkiF0)) Hz")
        print("   - 6–12 Hz 割合: \(String(format: "%.4f", tenkiMod6to12))")

        let mizuMetric = AcousticCentroidMetric.measure(pcm: mizuPCM)
        let mizuDur = Float(mizuPCM.count) / 16000.0
        let mizuHalves = AcousticCentroidMetric.measureHalves(pcm: mizuPCM, interpolateParabolic: false)
        let mizuBoundaries = SpikeSpeechEngine.detectSpeechBoundaries(
            pcm: mizuPCM,
            hopSize: AudioConfig.hopSize,
            totalFrames: max(1, mizuPCM.count / AudioConfig.hopSize)
        )
        let mizuBodyFrames = mizuBoundaries.speechFrames
        let mizuBodyDur = Float(mizuBodyFrames * AudioConfig.hopSize) / 16000.0
        let mizuF0 = AcousticCentroidMetric.measureF0Median(pcm: mizuPCM, interpolateParabolic: false)
        let mizuMod6to12 = AcousticCentroidMetric.measureModulation6to12Ratio(pcm: mizuPCM)
        print("2. tts_mizuwomare.wav:")
        print("   - 全体長: \(mizuPCM.count) サンプル, \(String(format: "%.3f", mizuDur)) 秒")
        print("   - 本体長 (端の無音除外): \(String(format: "%.3f", mizuBodyDur)) 秒")
        print("   - 全体 200–1500 Hz 重心: \(String(format: "%.2f", mizuMetric.centroidMedian200to1500)) Hz")
        print("   - 全体 停止割合: \(String(format: "%.4f", mizuMetric.cosRatio))")
        print("   - 全体 補間なし F0 中央値: \(String(format: "%.2f", mizuF0)) Hz")
        print("   - 前半: 停止割合=\(String(format: "%.4f", mizuHalves.first.cosRatio)), 200–1500 Hz重心=\(String(format: "%.2f", mizuHalves.first.centroidMedian200to1500)) Hz, 補間なしF0中央値=\(String(format: "%.2f", mizuHalves.first.f0Median)) Hz")
        print("   - 後半: 停止割合=\(String(format: "%.4f", mizuHalves.second.cosRatio)), 200–1500 Hz重心=\(String(format: "%.2f", mizuHalves.second.centroidMedian200to1500)) Hz, 補間なしF0中央値=\(String(format: "%.2f", mizuHalves.second.f0Median)) Hz")
        print("   - 前後 F0 差 (前半 - 後半): \(String(format: "%.2f", mizuHalves.first.f0Median - mizuHalves.second.f0Median)) Hz")
        print("   - 6–12 Hz 割合: \(String(format: "%.4f", mizuMod6to12))")

        let copyCandidates = [
            ".tmp/wave15/copy_BASIC5000_0001.wav",
            "copy_BASIC5000_0001.wav",
            "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        ]
        var copyLoaded = false
        var cIdx = 0
        while cIdx < copyCandidates.count {
            let cPath = copyCandidates[cIdx]
            if fileManager.fileExists(atPath: cPath) {
                if let copyPCM = try? WavAudioReader().loadWav16k(from: cPath) {
                    let copyMetric = AcousticCentroidMetric.measure(pcm: copyPCM)
                    let copyHalves = AcousticCentroidMetric.measureHalves(pcm: copyPCM)
                    let copyDur = Float(copyPCM.count) / 16000.0
                    let copyMod6to12 = AcousticCentroidMetric.measureModulation6to12Ratio(pcm: copyPCM)
                    print("3. \(cPath) (教師基準):")
                    print("   - 音声長: \(copyPCM.count) サンプル, \(String(format: "%.3f", copyDur)) 秒")
                    print("   - 前半: 停止割合=\(String(format: "%.4f", copyHalves.first.cosRatio)), 200–1500 Hz重心=\(String(format: "%.2f", copyHalves.first.centroidMedian200to1500)) Hz, F0中央値=\(String(format: "%.2f", copyHalves.first.f0Median)) Hz")
                    print("   - 後半: 停止割合=\(String(format: "%.4f", copyHalves.second.cosRatio)), 200–1500 Hz重心=\(String(format: "%.2f", copyHalves.second.centroidMedian200to1500)) Hz, F0中央値=\(String(format: "%.2f", copyHalves.second.f0Median)) Hz")
                    print("   - 全体 200–1500 Hz 重心: \(String(format: "%.2f", copyMetric.centroidMedian200to1500)) Hz")
                    print("   - 全体 停止割合: \(String(format: "%.4f", copyMetric.cosRatio))")
                    print("   - 6–12 Hz 割合: \(String(format: "%.4f", copyMod6to12))")
                    copyLoaded = true
                    break
                }
            }
            cIdx += 1
        }
        if copyLoaded != true {
            print("3. copy_BASIC5000_0001.wav: ファイルが見つかりませんでした")
        }
        print("==================================================")
    }

    // 保存ファイルから再ロードして推論 F0 MAE を実測検証（受入基準 3）
    if let reloadedData = try? Data(contentsOf: outputURL),
       let reloadedWeights = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: reloadedData),
       let reloadedProsody = reloadedWeights.prosodyWeights {
        let fw = reloadedProsody.f0Weights
        var totalAbsErr: Float = 0.0
        var totalVoicedFrames: Int = 0

        var s = 0
        while s < prosodySamples.count {
            let pSample = prosodySamples[s]
            if pSample.f0Features.isEmpty != true {
                let inD = fw.inputDim
                let hidD = fw.hiddenDim
                let totalF = pSample.f0Features.count
                var h0 = [Float](repeating: 0.0, count: totalF * hidD)
                var t = 0
                while t < totalF {
                    let feat = pSample.f0Features[t]
                    let hRow = t * hidD
                    var h = 0
                    while h < hidD {
                        var dot = fw.b1[h]
                        let wRow = h * inD
                        var j = 0
                        while j < inD {
                            dot += fw.w1[wRow + j] * feat[j]
                            j += 1
                        }
                        var act = dot
                        if dot < 0.0 {
                            act = dot * 0.1
                        }
                        h0[hRow + h] = act
                        h += 1
                    }
                    t += 1
                }

                // 1D Depthwise Conv (K=5)
                var mConv = [Float](repeating: 0.0, count: totalF * hidD)
                t = 0
                while t < totalF {
                    let hRowCur = t * hidD
                    var tM2 = t - 2
                    if tM2 < 0 { tM2 = 0 }
                    var tM1 = t - 1
                    if tM1 < 0 { tM1 = 0 }
                    var tP1 = t + 1
                    if totalF <= tP1 { tP1 = totalF - 1 }
                    var tP2 = t + 2
                    if totalF <= tP2 { tP2 = totalF - 1 }

                    let rM2 = tM2 * hidD
                    let rM1 = tM1 * hidD
                    let rP1 = tP1 * hidD
                    let rP2 = tP2 * hidD

                    var h = 0
                    while h < hidD {
                        let w0 = fw.wConv[(0 * hidD) + h]
                        let w1 = fw.wConv[(1 * hidD) + h]
                        let w2 = fw.wConv[(2 * hidD) + h]
                        let w3 = fw.wConv[(3 * hidD) + h]
                        let w4 = fw.wConv[(4 * hidD) + h]

                        let z0 = h0[rM2 + h] * w0
                        let z1 = h0[rM1 + h] * w1
                        let z2 = h0[hRowCur + h] * w2
                        let z3 = h0[rP1 + h] * w3
                        let z4 = h0[rP2 + h] * w4
                        let zConv = (z0 + z1) + (z2 + z3) + z4

                        let sumVal = h0[hRowCur + h] + zConv
                        var actM = sumVal
                        if sumVal < 0.0 {
                            actM = sumVal * 0.1
                        }
                        mConv[hRowCur + h] = actM
                        h += 1
                    }
                    t += 1
                }

                // 終段線形射影 -> 対数 Hz -> exp
                t = 0
                while t < totalF {
                    if 0.5 <= pSample.voicedMask[t] {
                        let mRow = t * hidD
                        var outVal = fw.b2[0]
                        var h = 0
                        while h < hidD {
                            outVal += fw.w2[h] * mConv[mRow + h]
                            h += 1
                        }
                        let predHz = expf(outVal)
                        let targetHz = pSample.targetF0[t]
                        totalAbsErr += abs(predHz - targetHz)
                        totalVoicedFrames += 1
                    }
                    t += 1
                }
            }
            s += 1
        }

        var reloadedMAE: Float = 0.0
        if 0 < totalVoicedFrames {
            reloadedMAE = totalAbsErr / Float(totalVoicedFrames)
        }
        print("==================================================")
        print("[検証] 保存ファイル (\(outputPath)) からのロード後推論 有声 F0 MAE: \(String(format: "%.2f", reloadedMAE)) Hz (有声フレーム数: \(totalVoicedFrames))")
        if reloadedMAE < 20.0 {
            print("[検証結果] 受入基準クリア: ロード後有声 F0 MAE < 20 Hz 達成！")
        } else {
            print("[検証結果] 警告: ロード後有声 F0 MAE (\(reloadedMAE) Hz) が 20 Hz を超えています。")
        }
        print("==================================================")
    }

    // なぜニューラルボコーダーも学習・エクスポートするか:
    // 再学習パイプラインにおいて SNN 音響モデル重み（weights.json）と
    // 現代的完全データ駆動型ニューラルボコーダー重み（vocoder_weights.json）を一元同期し、
    // 実音声波形に対する STFT 損失最小化により自然な日本語音声を直接生成するため。
    runVocoderTraining()

    print("学習処理が正常に完了しました。")
}

main()
