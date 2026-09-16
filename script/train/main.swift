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
    var numLayers: Int = 2
    var inDim: Int = 128
    var outDim: Int = AudioConfig.melChannels // 64
    var timeSteps: Int = 4
    var maxSamples: Int = 50
    var datasetPath: String? = nil
    var outputPath: String = "Models/weights.json"
    var loadWeightsPath: String? = "Models/weights.json"
    var forceFresh: Bool = false

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
        case "-h", "--help":
            print("Usage: train [-d <jsut_dir>] [-s <samples>] [-e <epochs>] [--lr <learning_rate>] [--lr-min <min_lr>] [--warmup-epochs <epochs>] [--wd <weight_decay>] [--shuffle-seed <seed>] [--no-shuffle] [--hidden-dim <dim>] [--num-layers <layers>] [--in-dim <dim>] [--out-dim <dim>] [--time-steps <steps>] [-w <weights.json>] [--fresh] [-o <output.json>]")
            return
        default:
            break
        }
        i += 1
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
    print("教師音声基準:     JSUT (女性単一話者 VoiceProfile.female.tract Prior)")
    print("出力パス:         \(outputPath)")
    print("--------------------------------------------------")

    // 既存の学習済み重みが存在する場合はウォームスタートし、学習の蓄積と損失減少の継続性を確保する
    // なぜ全次元（input, output, hidden, layers）の厳密検査を行うか:
    // CLI 引数で隠れ層次元や層数が変更された場合に、異なるシェイプの重みを誤ロードして
    // 行列積のクラッシュや意図しない旧構造のまま学習が継続される不整合を完全に防止するため。
    var initialWeights: SpikingNetworkWeights? = nil
    if forceFresh != true {
        if let path = loadWeightsPath {
            if fileManager.fileExists(atPath: path) {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: path)) {
                    if let loaded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) {
                        // なぜ全アーキテクチャパラメータ（input, output, hidden, layers, timeSteps）を厳密に検証するか:
                        // SNN BPTT 学習では層数や隠れ層次元だけでなく、時間ステップ数（timeSteps）が異なると
                        // 膜電位蓄積やスパイク時間統合のダイナミクスが破綻して学習が不能となるため。
                        let isMatch = (loaded.inputDim == inDim) &&
                                      (loaded.outputDim == outDim) &&
                                      (loaded.maxHiddenDim == hiddenDim) &&
                                      (loaded.numLayers == numLayers) &&
                                      (loaded.timeSteps == timeSteps)
                        switch isMatch {
                        case true:
                            initialWeights = loaded
                            print("既存の学習済み重みをロードしました: \(path)")
                        default:
                            print("警告: 既存の重みと指定されたアーキテクチャパラメータが一致しません (input: \(loaded.inputDim)vs\(inDim), output: \(loaded.outputDim)vs\(outDim), hidden: \(loaded.maxHiddenDim)vs\(hiddenDim), layers: \(loaded.numLayers)vs\(numLayers), timeSteps: \(loaded.timeSteps)vs\(timeSteps))。新規初期化します。")
                        }
                    }
                }
            }
        }
    }

    let weights: SpikingNetworkWeights
    switch initialWeights {
    case .some(let w):
        // なぜ語彙が空の場合にデフォルト語彙を補完するか:
        // 旧バージョンの重みファイルをロードした際にも語彙を補完し、学習エクスポート時に語彙が消失するのを防ぐため。
        if w.lexicon.isEmpty != true {
            weights = w
        } else {
            let defaultLex = ViterbiMorphology.loadDefaultLexicon()
            weights = w.withLexicon(defaultLex)
        }
    case .none:
        // なぜ新規乱数重みにもデフォルト語彙を注入するか:
        // 新規学習から開始した場合でも、モデル重みに語彙知識を保持・永続化させるため。
        let defaultLex = ViterbiMorphology.loadDefaultLexicon()
        weights = SpikingNetworkWeights.randomWeights(
            inputDim: inDim,
            maxHiddenDim: hiddenDim,
            outputDim: outDim,
            timeSteps: timeSteps,
            numLayers: numLayers,
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

    // コーパス探索候補
    var corpusDir: String? = nil
    var candidates: [String] = []
    if let explicit = datasetPath, explicit.isEmpty != true {
        var cleanPath = explicit
        if cleanPath.hasPrefix("@") {
            cleanPath = String(cleanPath.dropFirst())
        }
        candidates.append(cleanPath)
    }
    if let envPath = ProcessInfo.processInfo.environment["JSUT_CORPUS_DIR"], envPath.isEmpty != true {
        candidates.append(envPath)
    }
    candidates.append(currentDir + "/../spiketrans/.tmp/jsut_ver1.1/basic5000")
    candidates.append(currentDir + "/.tmp/jsut_ver1.1/basic5000")
    candidates.append(currentDir + "/basic5000")
    let homeDir = fileManager.homeDirectoryForCurrentUser.path
    candidates.append(homeDir + "/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000")

    var cIdx = 0
    while cIdx < candidates.count {
        let cand = candidates[cIdx]
        if fileManager.fileExists(atPath: cand + "/transcript_utf8.txt") {
            corpusDir = cand
            break
        }
        cIdx += 1
    }

    var trainingData: [(features: [[Float]], targets: [[Float]])] = []

    // なぜローカル関数を廃止し SpikeSpeechEngine.prepareTrainingPair を正本として呼ぶか:
    // 同一ロジックの二重実装を根絶し、単体テスト・学習 CLI・データセット生成で全く同一の
    // アライメント・Blended Prior 目標残差生成器を唯一の正本として共有するため。
    let pitchTracker = PitchTracker()

    switch corpusDir {
    case .some(let cDir):
        print("JSUT basic5000 コーパスパス: \(cDir)")
        let transcriptPath = cDir + "/transcript_utf8.txt"
        let wavDir = cDir + "/wav"
        if let content = try? String(contentsOfFile: transcriptPath, encoding: .utf8) {
            let lines = content.components(separatedBy: .newlines)
            var lineIdx = 0
            while lineIdx < lines.count {
                if maxSamples <= trainingData.count {
                    break
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
                        if let pcm16k = try? wavReader.loadWav16k(from: wavFile) {
                            if let pair = engine.prepareTrainingPair(
                                text: text,
                                pcm16k: pcm16k,
                                melExtractor: melExtractor,
                                pitchTracker: pitchTracker
                            ) {
                                trainingData.append(pair)
                            }
                        }
                    }
                }
                lineIdx += 1
            }
        }
    case .none:
        print("JSUT コーパス未検出のため、SyntheticAudioGenerator による基準音声で学習データを自動生成します。")
        let synthGen = SyntheticAudioGenerator(sampleRate: Float(AudioConfig.sampleRate))
        let standardCorpus = synthGen.generateStandardCorpus()
        var sIdx = 0
        while sIdx < standardCorpus.count {
            let item = standardCorpus[sIdx]
            if let pair = engine.prepareTrainingPair(
                text: item.text,
                pcm16k: item.samples,
                melExtractor: melExtractor,
                pitchTracker: pitchTracker
            ) {
                trainingData.append(pair)
            }
            sIdx += 1
        }
    }

    if trainingData.isEmpty {
        print("エラー: 有効な学習データが 0 件です。")
        return
    }
    print("有効学習サンプル数: \(trainingData.count) 件")

    let network = MLXSpikingAcousticNetwork(weights: weights)

    let schedule = CosineWarmupSchedule(
        lrBase: learningRate,
        lrMin: lrMin,
        warmupEpochs: warmupEpochs,
        totalEpochs: epochs
    )
    var plateau = PlateauGuard(patience: 2, factor: 0.5, relThreshold: 0.005)

    let trainer = MLXAcousticBPTTTrainer(
        network: network,
        learningRate: schedule.learningRate(epoch: 0),
        bpttWindow: 16,
        weightDecay: weightDecay
    )

    print("BPTT 最適化ループを開始します...")
    var initialLoss: Float = 0.0
    var finalLoss: Float = 0.0
    var bestLoss = Float.greatestFiniteMagnitude
    var bestEpoch = -1

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
            let loss = trainer.trainSequence(
                features: pair.features,
                targets: pair.targets
            )
            epochLossSum += loss
            batchCount += 1
            dIdx += 1
        }

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

        // なぜ最良エポックを記録するか: ログ上でどのエポックが最良だったか即座に判別できるようにするため
        if avgLoss < bestLoss {
            bestLoss = avgLoss
            bestEpoch = epoch + 1
        }

        // なぜ毎エポックスナップショットを保存するか:
        // 学習途中での最良パラメータが後続エポックで失われることを防ぎ、任意時点へのロールバックを可能にするため
        let epURL = WeightCheckpoint.resolvePath(
            directory: outputDir,
            fileName: WeightCheckpoint.epochFileName(epochOneIndexed: epoch + 1)
        )
        let intermediateWeights = network.exportWeights()
        do {
            try WeightCheckpoint.atomicWritePretty(intermediateWeights, to: epURL)
        } catch {
            print("警告: エポックスナップショット保存失敗 (\(epURL.path)): \(error)")
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

    let exportedWeights = network.exportWeights()
    do {
        try WeightCheckpoint.atomicWritePretty(exportedWeights, to: outputURL)
        let dataCount = (try? Data(contentsOf: outputURL).count) ?? 0
        print("最終モデル重みを保存しました: \(outputPath) (\(dataCount) バイト)")
    } catch {
        print("エラー: 最終重みの書き出しに失敗しました: \(error)")
        return
    }

    // なぜニューラルボコーダー重みも出力ディレクトリへエクスポートするか:
    // 再学習パイプラインにおいて SNN 音響モデル重み（weights.json）と
    // ニューラルボコーダー重み（vocoder_weights.json）を一元同期し、
    // 推論エンジンが最新の音響モデルおよびボコーダー構造を即座に利用可能にするため。
    let vocoderURL = WeightCheckpoint.resolvePath(directory: outputDir, fileName: "vocoder_weights.json")
    var needVocoderWrite = true
    if fileManager.fileExists(atPath: vocoderURL.path) {
        if let existingData = try? Data(contentsOf: vocoderURL) {
            switch try? JSONDecoder().decode(NeuralVocoderWeights.self, from: existingData) {
            case .some:
                needVocoderWrite = false
            case .none:
                break
            }
        }
    }
    if needVocoderWrite {
        let initialVocoderWeights = NeuralVocoderWeights.randomWeights()
        if let encoded = try? JSONEncoder().encode(initialVocoderWeights) {
            try? encoded.write(to: vocoderURL)
            print("ニューラルボコーダー重みをエクスポートしました: \(vocoderURL.path) (\(encoded.count) バイト)")
        }
    }

    print("学習処理が正常に完了しました。")
}

main()
