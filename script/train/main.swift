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
    var forceInitBOut: Bool = false
    var vocoderEpochs: Int = 5
    var vocoderLearningRate: Float = 0.0003

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
        case "--init-bout":
            forceInitBOut = true
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
    var vocoderPairs: [(mel: [[Float]], f0: [Float], voiced: [Float], pcm: [Float])] = []

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
                            if let pair = engine.prepareTrainingPair(
                                text: text,
                                pcm16k: pcm16k,
                                melExtractor: melExtractor,
                                pitchTracker: pitchTracker
                            ) {
                                trainingData.append(pair)
                                let extractedMel = melExtractor.extractLogMel(pcm: pcm16k)
                                let pitchResult = pitchTracker.track(pcm: pcm16k)
                                vocoderPairs.append((mel: extractedMel, f0: pitchResult.f0, voiced: pitchResult.voiced, pcm: pcm16k))
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
            var peak: Float = 0.0
            var pIdx = 0
            while pIdx < item.samples.count {
                let a = abs(item.samples[pIdx])
                if peak < a {
                    peak = a
                }
                pIdx += 1
            }
            var pcm16k = item.samples
            if 0.01 < peak {
                let normFactor = 0.85 / peak
                var s = 0
                while s < pcm16k.count {
                    pcm16k[s] = pcm16k[s] * normFactor
                    s += 1
                }
            }
            if let pair = engine.prepareTrainingPair(
                text: item.text,
                pcm16k: pcm16k,
                melExtractor: melExtractor,
                pitchTracker: pitchTracker
            ) {
                trainingData.append(pair)
                let extractedMel = melExtractor.extractLogMel(pcm: pcm16k)
                let pitchResult = pitchTracker.track(pcm: pcm16k)
                vocoderPairs.append((mel: extractedMel, f0: pitchResult.f0, voiced: pitchResult.voiced, pcm: pcm16k))
            }
            sIdx += 1
        }
    }

    if trainingData.isEmpty {
        print("エラー: 有効な学習データが 0 件です。")
        return
    }
    print("有効学習サンプル数: \(trainingData.count) 件")

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

    var effectiveWeights = weights
    var shouldInitBOut = forceFresh || forceInitBOut
    if currentBOutMean < 0.5 {
        shouldInitBOut = true
    }
    if shouldInitBOut {
        effectiveWeights = weights.withBOut(meanMel)
        let meanVal = meanMel.reduce(0, +) / Float(outDim)
        print("SNN 出力バイアス bOut を実音声の平均対数 Mel スペクトルで初期化しました（チャンネル平均: \(String(format: "%.2f", meanVal))）")
        print("  meanMel[0..15]: \(meanMel.prefix(16).map { String(format: "%.2f", $0) })")
    }

    let network = MLXSpikingAcousticNetwork(weights: effectiveWeights)

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
                    vocoder.importWeights(from: savedWeights)
                    print("既存のニューラルボコーダー重みを読み込みました: \(vocoderURL.path)")
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
                                let featArr = MLXArray(batchFeats, [currBatchItems, segFrames, inCh])
                                let targArr = MLXArray(batchPCMs, [currBatchItems, segSamples])

                                let (lossVals, grads) = lg(vocoder, [featArr, targArr])
                                let lossVal = lossVals[0].item(Float.self)
                                epochLossSum += lossVal
                                let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 1.0)
                                vocoderOptimizer.update(model: vocoder, gradients: clippedGrads)

                                vBatchCount += 1
                                batchFeats.removeAll(keepingCapacity: true)
                                batchPCMs.removeAll(keepingCapacity: true)
                                currBatchItems = 0
                            }
                        }
                        sampleIt += 1
                    }
                }
                pairIdx += 1
            }

            if 0 < currBatchItems {
                let featArr = MLXArray(batchFeats, [currBatchItems, segFrames, inCh])
                let targArr = MLXArray(batchPCMs, [currBatchItems, segSamples])

                let (lossVals, grads) = lg(vocoder, [featArr, targArr])
                let lossVal = lossVals[0].item(Float.self)
                epochLossSum += lossVal
                let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 1.0)
                vocoderOptimizer.update(model: vocoder, gradients: clippedGrads)

                vBatchCount += 1
                batchFeats.removeAll(keepingCapacity: true)
                batchPCMs.removeAll(keepingCapacity: true)
                currBatchItems = 0
            }

            eval(vocoder.trainableParameters())

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
    } else {
        // なぜ forceFresh 時または破損時にニューラルボコーダー初期重みを再書き出しするか:
        // 旧アーキテクチャで 100Hz 周期共鳴に過学習したボコーダー重みをクリーンな
        // He 初期化＋Fant/Rosenberg 音源励起適合重みへ明示的にリセット可能にするため。
        var needVocoderWrite = forceFresh
        if needVocoderWrite != true && fileManager.fileExists(atPath: vocoderURL.path) {
            if let existingData = try? Data(contentsOf: vocoderURL) {
                switch try? JSONDecoder().decode(NeuralVocoderWeights.self, from: existingData) {
                case .some:
                    needVocoderWrite = false
                case .none:
                    needVocoderWrite = true
                }
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
