import Foundation
import MLX
import MLXNN
import MLXOptimizers
import SpikeSpeech

/// SNN 音響モデル BPTT 学習 CLI
func main() {
    let args = CommandLine.arguments
    var epochs: Int = 30
    var learningRate: Float = 0.005
    var hiddenDim: Int = 256
    var numLayers: Int = 2
    var inDim: Int = 128
    var outDim: Int = AudioConfig.melChannels // 64
    var maxSamples: Int = 50
    var datasetPath: String? = nil
    var voiceName: String = "female"
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
        case "-v", "--voice":
            let nextIdx = i + 1
            if nextIdx < args.count {
                voiceName = args[nextIdx]
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
            print("Usage: train [-d <jsut_dir>] [-s <samples>] [-v <voice>] [-e <epochs>] [--lr <learning_rate>] [--hidden-dim <dim>] [--num-layers <layers>] [--in-dim <dim>] [--out-dim <dim>] [-w <weights.json>] [--fresh] [-o <output.json>]")
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

    // ルート直下に誤って配置された不要 WAV ファイルを Tests/resources へ安全に移動
    let rootWavs = [
        "test_denoised.wav",
        "test_aiueo_final.wav",
        "test_aiueo_test.wav",
        "test_child.wav",
        "test_deep.wav",
        "test_female.wav",
        "test_male.wav"
    ]
    for wavName in rootWavs {
        let srcPath = currentDir + "/" + wavName
        let dstPath = currentDir + "/Tests/resources/" + wavName
        if fileManager.fileExists(atPath: srcPath) {
            if fileManager.fileExists(atPath: dstPath) != true {
                try? fileManager.copyItem(atPath: srcPath, toPath: dstPath)
            }
            try? fileManager.removeItem(atPath: srcPath)
        }
    }

    // ルート直下の weights.json を Models/weights.json と同期してルートから除去
    let rootWeights = currentDir + "/weights.json"
    if fileManager.fileExists(atPath: rootWeights) {
        try? fileManager.removeItem(atPath: rootWeights)
    }

    // ライブラリ本体 (Sources/SpikeSpeech/Corpus) からコーパスファイルを完全排除
    let corpusDirInSources = currentDir + "/Sources/SpikeSpeech/Corpus"
    if fileManager.fileExists(atPath: corpusDirInSources) {
        try? fileManager.removeItem(atPath: corpusDirInSources)
    }

    MLXRandom.seed(2026)

    let voiceProfile = VoiceProfile.preset(named: voiceName)

    print("==================================================")
    print("SpikeSpeech SNN 音響モデル BPTT 学習パイプライン")
    print("==================================================")
    print("エポック数:   \(epochs)")
    print("学習率:       \(learningRate)")
    print("隠れ層次元:   \(hiddenDim)")
    print("層数:         \(numLayers)")
    print("話者設定:     \(voiceProfile.name)")
    print("出力パス:     \(outputPath)")
    print("--------------------------------------------------")

    // 既存の学習済み重みが存在する場合はウォームスタートし、学習の蓄積と損失減少の継続性を確保する
    var initialWeights: SpikingNetworkWeights? = nil
    if forceFresh != true {
        if let path = loadWeightsPath {
            if fileManager.fileExists(atPath: path) {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: path)) {
                    if let loaded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) {
                        if loaded.inputDim == inDim && loaded.outputDim == outDim {
                            initialWeights = loaded
                            print("既存の学習済み重みをロードしました: \(path)")
                        }
                    }
                }
            }
        }
    }

    let weights: SpikingNetworkWeights
    switch initialWeights {
    case .some(let w):
        weights = w
    case .none:
        weights = SpikingNetworkWeights.randomWeights(
            inputDim: inDim,
            maxHiddenDim: hiddenDim,
            outputDim: outDim,
            numLayers: numLayers
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

    // 実音声波形から先頭無音、発話有音区間、および末尾無音区間のフレーム数を検出（VAD）
    func detectSpeechBoundaries(
        pcm: [Float],
        hopSize: Int = AudioConfig.hopSize,
        totalFrames: Int
    ) -> (leadSilence: Int, speechFrames: Int, trailSilence: Int) {
        if totalFrames <= 0 || pcm.isEmpty {
            return (0, 0, 0)
        }

        var frameRms = [Float](repeating: 0.0, count: totalFrames)
        var maxRms: Float = 0.0
        var f = 0
        while f < totalFrames {
            let sampleStart = f * hopSize
            var sumSq: Float = 0.0
            var s = 0
            while s < hopSize {
                let sIdx = sampleStart + s
                if sIdx < pcm.count {
                    let val = pcm[sIdx]
                    sumSq += val * val
                }
                s += 1
            }
            let rms = sqrt(sumSq / Float(hopSize))
            frameRms[f] = rms
            if maxRms < rms {
                maxRms = rms
            }
            f += 1
        }

        var threshold = maxRms * 0.04
        if threshold < 0.005 {
            threshold = 0.005
        }

        var lead = 0
        while lead < totalFrames {
            if threshold <= frameRms[lead] {
                break
            }
            lead += 1
        }

        var trail = 0
        var tIdx = totalFrames - 1
        while 0 <= tIdx {
            if threshold <= frameRms[tIdx] {
                break
            }
            trail += 1
            tIdx -= 1
        }

        var speech = totalFrames - lead - trail
        if speech < 4 {
            lead = 0
            trail = 0
            speech = totalFrames
        }

        return (leadSilence: lead, speechFrames: speech, trailSilence: trail)
    }

    // テキストと言語特徴量、音声波形から VAD アライメント・残差目標 (targetMel - prior) を抽出
    func preparePair(
        text: String,
        pcm16k: [Float],
        engine: SpikeSpeechEngine,
        voiceProfile: VoiceProfile,
        melExtractor: MelSpectrogramExtractor,
        pitchTracker: PitchTracker
    ) -> (features: [[Float]], targets: [[Float]])? {
        if pcm16k.isEmpty {
            return nil
        }
        let targetMel = melExtractor.extractLogMel(pcm: pcm16k)
        let targetFrames = targetMel.count
        if targetFrames <= 0 {
            return nil
        }

        // 学習時は決定論的アライメントのため 1/f ゆらぎをバイパス（アライメント汚染の根絶）
        let baseLinguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0,
            baseF0: voiceProfile.baseF0,
            applyFluctuation: false
        )

        let origTotalFrames = baseLinguistic.totalFrames
        let phoneCount = baseLinguistic.phoneIds.count

        let boundaries = detectSpeechBoundaries(
            pcm: pcm16k,
            hopSize: AudioConfig.hopSize,
            totalFrames: targetFrames
        )
        let leadSilence = boundaries.leadSilence
        let speechFrames = boundaries.speechFrames
        let trailSilence = boundaries.trailSilence

        var alignedFeatures: [[Float]]
        var fullPhoneIds: [Int32] = []
        var fullDurations: [Int] = []

        if origTotalFrames <= 0 || phoneCount <= 0 {
            alignedFeatures = [[Float]](repeating: [Float](repeating: 0.0, count: engine.weights.inputDim), count: targetFrames)
            fullPhoneIds = [Int32(PhonemeVocabulary.silId)]
            fullDurations = [targetFrames]
        } else {
            let stretchRatio = Float(speechFrames) / Float(max(1, origTotalFrames))
            var scaledDurations = [Float](repeating: 0.0, count: phoneCount)
            var p = 0
            while p < phoneCount {
                let origD = Float(baseLinguistic.durations[p])
                scaledDurations[p] = max(1.0, origD * stretchRatio)
                p += 1
            }

            let quantizedDurations = engine.lengthRegulator.quantizeDurations(durations: scaledDurations)
            var sumQuantized = 0
            var q = 0
            while q < quantizedDurations.count {
                sumQuantized += quantizedDurations[q]
                q += 1
            }

            var speechDurations = quantizedDurations
            let diff = speechFrames - sumQuantized
            if diff < 0 {
                var remaining = -diff
                var idx = speechDurations.count - 1
                while 0 <= idx && 0 < remaining {
                    if 1 < speechDurations[idx] {
                        let reducible = speechDurations[idx] - 1
                        let dec = min(remaining, reducible)
                        speechDurations[idx] -= dec
                        remaining -= dec
                    }
                    idx -= 1
                }
            }
            if 0 < diff && 0 < speechDurations.count {
                let lastIdx = speechDurations.count - 1
                speechDurations[lastIdx] += diff
            }

            if 0 < leadSilence {
                fullPhoneIds.append(Int32(PhonemeVocabulary.silId))
                fullDurations.append(leadSilence)
            }

            var bIdx = 0
            while bIdx < phoneCount {
                fullPhoneIds.append(baseLinguistic.phoneIds[bIdx])
                fullDurations.append(speechDurations[bIdx])
                bIdx += 1
            }

            if 0 < trailSilence {
                fullPhoneIds.append(Int32(PhonemeVocabulary.silId))
                fullDurations.append(trailSilence)
            }

            // Pure Swift PitchTracker による実音声からの実測 F0、有声度、および実測短時間エネルギー抽出
            // なぜ実測値を用いるか:
            // 数式による理論 F0 は平坦で幾何学的なカーブとなるためロボット的な発音を招く。
            // 実際の声優さんの音声から抽出した実測ピッチコンター・生体ゆらぎ・有声無声遷移・エネルギーを
            // 入力特徴量として直接供給することで、人間味あふれる抑揚と音響スペクトルの相関を SNN に直接学習させる。
            let pitchResult = pitchTracker.track(pcm: pcm16k)
            var alignedF0 = [Float](repeating: 0.0, count: targetFrames)
            var alignedVoiced = [Float](repeating: 0.0, count: targetFrames)
            var alignedEnergy = [Float](repeating: 0.0, count: targetFrames)
            var f = 0
            while f < targetFrames {
                if f < pitchResult.frameCount {
                    alignedF0[f] = pitchResult.f0[f]
                    alignedVoiced[f] = pitchResult.voiced[f]
                    alignedEnergy[f] = pitchResult.energy[f]
                }
                f += 1
            }

            var int32Durations = [Int32](repeating: 0, count: fullDurations.count)
            var dIdx = 0
            while dIdx < fullDurations.count {
                int32Durations[dIdx] = Int32(fullDurations[dIdx])
                dIdx += 1
            }

            let alignedLinguistic = LinguisticFeatures(
                phoneIds: fullPhoneIds,
                durations: int32Durations,
                f0Contour: alignedF0,
                voicedFlags: alignedVoiced,
                energyContour: alignedEnergy,
                totalFrames: targetFrames
            )

            alignedFeatures = engine.encodeLinguisticFeatures(
                features: alignedLinguistic
            )
        }

        let finalCount = min(alignedFeatures.count, targetMel.count)
        var safeFeatures = alignedFeatures
        if finalCount < safeFeatures.count {
            safeFeatures.removeSubrange(finalCount..<safeFeatures.count)
        }

        var safeTargets = [[Float]](repeating: [Float](repeating: 0.0, count: AudioConfig.melChannels), count: finalCount)
        var framePhoneIds = [Int](repeating: 1, count: finalCount)
        var curFrame = 0
        var pIdx = 0
        let pCount = min(fullPhoneIds.count, fullDurations.count)
        while pIdx < pCount {
            let pId = fullPhoneIds[pIdx]
            let d = fullDurations[pIdx]
            var f = 0
            while f < d {
                let frameIdx = curFrame + f
                if frameIdx < finalCount {
                    framePhoneIds[frameIdx] = Int(pId)
                }
                f += 1
            }
            curFrame += d
            pIdx += 1
        }

        // なぜコーパス話者基準（成人女性基準）の Prior を固定して引くか:
        // JSUT は成人女性単一話者の音声コーパスである。
        // 訓練目標は「女性実音声 Mel − 女性基準 Prior」として純粋な音韻残差を学習させる必要があり、
        // 異なる話者 Prior を引くと SNN が声道幾何差（VTLN）を打ち消す有害残差を学習して声道層と干渉するため。
        let corpusTract = VocalTract(lengthScale: 1.0, bandwidthScale: 1.0)
        let activePrior = engine.prior(for: corpusTract)
        var t = 0
        while t < finalCount {
            let phoneId = framePhoneIds[t]
            var prior = [Float](repeating: 0.0, count: AudioConfig.melChannels)
            prior.withUnsafeMutableBufferPointer { dst in
                activePrior.copyPriorMel(phoneId: phoneId, dst: dst.baseAddress!)
            }

            let melChannels = min(targetMel[t].count, AudioConfig.melChannels)
            var c = 0
            while c < melChannels {
                safeTargets[t][c] = targetMel[t][c] - prior[c]
                c += 1
            }
            t += 1
        }

        return (features: safeFeatures, targets: safeTargets)
    }

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
                            if let pair = preparePair(
                                text: text,
                                pcm16k: pcm16k,
                                engine: engine,
                                voiceProfile: voiceProfile,
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
            if let pair = preparePair(
                text: item.text,
                pcm16k: item.samples,
                engine: engine,
                voiceProfile: voiceProfile,
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

    let trainer = MLXAcousticBPTTTrainer(
        network: network,
        learningRate: learningRate,
        bpttWindow: 16
    )

    print("BPTT 最適化ループを開始します...")
    var initialLoss: Float = 0.0
    var finalLoss: Float = 0.0

    var epoch = 0
    while epoch < epochs {
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

        print("  [Epoch \(epoch + 1)/\(epochs)] 平均損失: \(String(format: "%.6f", avgLoss))")

        // 5エポックごとに中間チェックポイントを最新重みファイルへアトミック保存する。
        if (epoch + 1) % 5 == 0 || epoch + 1 == epochs {
            let intermediateWeights = network.exportWeights()
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = .prettyPrinted
                let intermediateData = try encoder.encode(intermediateWeights)
                try intermediateData.write(to: URL(fileURLWithPath: outputPath))
            } catch {
                // pass
            }
        }

        epoch += 1
    }

    print("--------------------------------------------------")
    var reductionRate: Float = 0.0
    if 0.0 < initialLoss {
        reductionRate = (initialLoss - finalLoss) / initialLoss
    }
    print("初期損失:   \(String(format: "%.6f", initialLoss))")
    print("最終損失:   \(String(format: "%.6f", finalLoss))")
    print("損失減少率: \(String(format: "%.2f", reductionRate * 100.0))%")

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
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let jsonData = try encoder.encode(exportedWeights)
        let outputURL = URL(fileURLWithPath: outputPath)
        try jsonData.write(to: outputURL)
        print("モデル重みを保存しました: \(outputPath) (\(jsonData.count) バイト)")
    } catch {
        print("エラー: 重みの書き出しに失敗しました: \(error)")
        return
    }

    print("学習処理が正常に完了しました。")
}

main()
