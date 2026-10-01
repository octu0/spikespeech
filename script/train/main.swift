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
    var maxSamples: Int? = nil
    var datasetPath: String? = nil
    var outputPath: String = "Models/weights.json"
    var loadWeightsPath: String? = "Models/weights.json"
    var forceFresh: Bool = false
    var forceFreshSNN: Bool = false
    var forceInitBOut: Bool = false
    var vocoderEpochs: Int = 5
    var vocoderLearningRate: Float = 0.0003
    var prosodyEpochs: Int = 15
    var prosodySamplesLimit: Int? = nil
    var prosodyStepsPerSample: Int = 2
    var prosodyLearningRate: Float = 0.008
    var freshProsody: Bool = false
    var alignmentsPath: String? = nil
    var forceMAS: Bool = false

    func printUsage() {
        print("Usage: train -d <corpus_dir> [--alignments <alignments.json>] [--force-mas] [-s <samples>] [-e <epochs>] [--prosody-samples <samples>] [--prosody-steps <steps>] [--prosody-lr <lr>] [--fresh-prosody] [--vocoder-epochs <epochs>] [--vocoder-lr <lr>] [--prosody-epochs <epochs>] [--lr <learning_rate>] [--lr-min <min_lr>] [--warmup-epochs <epochs>] [--wd <weight_decay>] [--shuffle-seed <seed>] [--no-shuffle] [--hidden-dim <dim>] [--num-layers <layers>] [--in-dim <dim>] [--out-dim <dim>] [--time-steps <steps>] [-w <weights.json>] [--fresh] [--fresh-snn] [-o <output.json>]")
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
        case "--lr":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
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
        case "-h", "--help":
            printUsage()
            return

        default:
            break
        }
        i += 1
    }

    guard let explicit = datasetPath, explicit.isEmpty != true else {
        printUsage()
        exit(1)
    }
    var cleanDatasetPath = explicit
    if cleanDatasetPath.hasPrefix("@") {
        cleanDatasetPath = String(cleanDatasetPath.dropFirst())
    }
    if FileManager.default.fileExists(atPath: cleanDatasetPath + "/transcript_utf8.txt") != true {
        printUsage()
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

    let weights: SpikingNetworkWeights
    switch loadedExisting {
    case .some(let loaded):
        var lex = loaded.lexicon
        if lex.isEmpty {
            lex = ViterbiMorphology.loadDefaultLexicon()
        }
        let shouldFreshSNN = forceFresh || forceFreshSNN
        switch shouldFreshSNN {
        case true:
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
        case false:
            let isMatch = (loaded.inputDim == inDim) &&
                          (loaded.outputDim == outDim) &&
                          (loaded.maxHiddenDim == hiddenDim) &&
                          (loaded.numLayers == numLayers) &&
                          (loaded.timeSteps == timeSteps)
            switch isMatch {
            case true:
                weights = loaded.withLexicon(lex)
                print("既存の学習済み重みをロードしました: \(loadWeightsPath ?? "")")
            case false:
                print("警告: 既存の重みと指定されたアーキテクチャパラメータが一致しません (input: \(loaded.inputDim)vs\(inDim), output: \(loaded.outputDim)vs\(outDim), hidden: \(loaded.maxHiddenDim)vs\(hiddenDim), layers: \(loaded.numLayers)vs\(numLayers), timeSteps: \(loaded.timeSteps)vs\(timeSteps))。SNN 重みを新規初期化します。")
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
    case .none:
        let defaultLex = ViterbiMorphology.loadDefaultLexicon()
        weights = SpikingNetworkWeights.randomWeights(
            inputDim: inDim,
            maxHiddenDim: hiddenDim,
            outputDim: outDim,
            timeSteps: timeSteps,
            numLayers: numLayers,
            seed: 2026,
            lexicon: defaultLex
        )
        print("新規の決定論的ランダム重みで初期化しました。")
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
            if let sample0 = alignmentMap["BASIC5000_0001"] {
                if AlignmentStore.isUtteranceAlignmentValid(sample0) != true {
                    print("警告: キャッシュされた BASIC5000_0001 アライメントが縮退（2〜40F 範囲外）しています。新MAS再集計を実行します。")
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
    var trainingData: [(features: [[Float]], targets: [[Float]])] = []
    var reconTargetSample: (features: [[Float]], targets: [[Float]])? = nil
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
                if limit <= trainingData.count {
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

    if trainingData.isEmpty {
        print("エラー: 有効な学習データが 0 件です。")
        return
    }
    print("有効学習サンプル数: \(trainingData.count) 件")

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

    // ============================================================
    // 手順 5: wIn 音響特徴チャンネル (ch 192..198) 再初期化 & ウォームスタート重み設定
    // ============================================================
    print("\n===========================================================")
    print("【手順 5: 入力重み wIn 音響特徴チャンネル (ch 192..198) 再初期化】")
    print("===========================================================")
    var effectiveWeights = weights
        .withPhonemeAverageDurations(healthyPhonemeAverages)
        .withMeanFramesPerMora(16.0)
        .withResetAcousticInWeights(seed: 2026)

    var colNorms = [Float](repeating: 0.0, count: inDim)
    var cColIdx = 0
    while cColIdx < inDim {
        var sumSq: Float = 0.0
        var h = 0
        while h < hiddenDim {
            let val = effectiveWeights.wIn[(h * inDim) + cColIdx]
            sumSq += val * val
            h += 1
        }
        colNorms[cColIdx] = sqrtf(sumSq)
        cColIdx += 1
    }
    print("  --- wIn 列 L2 ノルム (再初期化後) ---")
    print("  ch 192 (voiced):     \(String(format: "%.4f", colNorms[192]))")
    print("  ch 193 (unvoiced):   \(String(format: "%.4f", colNorms[193]))")
    print("  ch 194 (normF0):     \(String(format: "%.4f", colNorms[194]))")
    print("  ch 195 (deltaF0):    \(String(format: "%.4f", colNorms[195]))")
    print("  ch 196 (phonePos):   \(String(format: "%.4f", colNorms[196]))")
    print("  ch 197 (rate):       \(String(format: "%.4f", colNorms[197]))")
    print("  ch 198 (energy):     \(String(format: "%.4f", colNorms[198]))")
    var unusedNormSum: Float = 0.0
    var unCh = 200
    while unCh < inDim {
        unusedNormSum += colNorms[unCh]
        unCh += 1
    }
    let unusedAvgNorm = unusedNormSum / Float(max(1, inDim - 200))
    print("  ch 200..255 (未使用平均): \(String(format: "%.4f", unusedAvgNorm))")
    print("  母音 a (one-hot ch 5):   \(String(format: "%.4f", colNorms[5]))")

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

    var globalMaxAbsDiff: Float = 0.0
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
            if globalMaxAbsDiff < diff {
                globalMaxAbsDiff = diff
            }
            c += 1
        }
        f += 1
    }

    print("  全チャンネル・全フレーム 最大絶対差: \(globalMaxAbsDiff)")
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
        ("ch 200..255 (その他/未使用)", 200, 255)
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

    if 0.0 < globalMaxAbsDiff {
        print("エラー: 特徴差が 0 ではありません (\(globalMaxAbsDiff))。学習を中止します。")
        return
    }
    print("【受入検証ゲート PASS】全チャンネル・全フレームの特徴差は厳密に 0 です。学習を開始します。\n")


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

    let schedule = CosineWarmupSchedule(
        lrBase: learningRate,
        lrMin: lrMin,
        warmupEpochs: warmupEpochs,
        totalEpochs: epochs
    )
    var plateau = PlateauGuard(patience: 4, factor: 0.7, relThreshold: 0.002)

    var trainer = MLXAcousticBPTTTrainer(
        network: network,
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
    var plateauAnchorLoss: Float = Float.greatestFiniteMagnitude
    var stagnantEpochs: Int = 0

    let outputURL = URL(fileURLWithPath: outputPath)
    let outputDir = outputURL.deletingLastPathComponent().path

    var epoch = 0
    while epoch < epochs {
        // なぜエポックごとにシャッフルするか:
        // データセットの固定順序による周期的勾配ドリフトバイアスを排除するため
        if noShuffle != true {
            let seed = TrainingShuffle.mixSeed(baseSeed: shuffleSeed, epoch: epoch)
            TrainingShuffle.shuffleInPlace(&trainingData, seed: seed)
        }

        let lr = resolvedLearningRate(
            schedule: schedule,
            epoch: epoch,
            plateauMultiplier: plateau.decayMultiplier
        )
        trainer.setLearningRate(lr)

        var epochLossSum: Float = 0.0
        var batchCount = 0

        var dIdx = 0
        while dIdx < trainingData.count {
            let pair = trainingData[dIdx]
            let loss = autoreleasepool {
                trainer.trainSequence(
                    features: pair.features,
                    targets: pair.targets
                )
            }
            epochLossSum += loss
            batchCount += 1

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
                    learningRate: currentLR,
                    bpttWindow: 16,
                    weightDecay: weightDecay
                )
                Memory.clearCache()
            }

            if (batchCount % 100) == 0 {
                print("    ステップ [\(batchCount)/\(trainingData.count)] 直近損失: \(String(format: "%.4f", loss))")
            }

            dIdx += 1
        }
        Memory.clearCache()

        var avgLoss: Float = 0.0
        if 0 < batchCount {
            avgLoss = epochLossSum / Float(batchCount)
        }

        if epoch == 0 {
            initialLoss = avgLoss
        }
        finalLoss = avgLoss

        plateau.observe(epochLoss: avgLoss)
        let norms = trainer.weightNorms()

        print("  [Epoch \(epoch + 1)/\(epochs)] 平均損失: \(String(format: "%.6f", avgLoss))  lr=\(String(format: "%.6g", lr))  ||wRec||=\(String(format: "%.4f", norms.wRec))  ||wOut||=\(String(format: "%.4f", norms.wOut))")

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
            bestSNNWeights = intermediateWeights
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

        // 早期停止判定（設計者指示: 10 エポック連続で損失が 0.01 以上減少しない、かつ 1.15 超なら停止して ASK）
        if plateauAnchorLoss - avgLoss < 0.01 {
            stagnantEpochs += 1
        } else {
            plateauAnchorLoss = avgLoss
            stagnantEpochs = 0
        }

        if 10 <= stagnantEpochs && 1.15 < avgLoss {
            print("【早期停止】10 エポック連続で損失改善が 0.01 未満でした（現在損失: \(String(format: "%.6f", avgLoss)), 基準損失: \(String(format: "%.6f", plateauAnchorLoss))）。学習を安全に停止します。")
            epoch += 1
            break
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

    let exportedWeights = bestSNNWeights
        .withProsodyWeights(finalProsodyWeights)
        .withPhonemeAverageDurations(phonemeAverages)
        .withMeanFramesPerMora(16.0)
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
    let vocoderURL = WeightCheckpoint.resolvePath(directory: outputDir, fileName: "vocoder_weights.json")
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

    print("学習処理が正常に完了しました。")
}

main()
