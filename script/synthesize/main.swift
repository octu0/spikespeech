import Foundation
import SpikeSpeech

/// 日本語テキスト音声合成 CLI
func main() {
    let args = CommandLine.arguments
    var text: String = ""
    // プロジェクトルートへの WAV ファイル直接配置禁止規約に基づき、既定出力先を Tests/resources ディレクトリに隔離
    var outputPath: String = "Tests/resources/output.wav"
    var weightsPath: String? = nil
    var speed: Float = 1.0
    var pitch: Float = 1.0
    var voiceName: String = "female"
    var benchmark: Bool = false
    var probeF0: Bool = false
    var copyInputPath: String? = nil
    var ablateInputPath: String? = nil

    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "--probe-f0":
            probeF0 = true
        case "--copy":
            let nextIdx = i + 1
            if nextIdx < args.count {
                copyInputPath = args[nextIdx]
                i += 1
            }
        case "--ablate":
            let nextIdx = i + 1
            if nextIdx < args.count {
                ablateInputPath = args[nextIdx]
                i += 1
            }
        case "-t", "--text":
            let nextIdx = i + 1
            if nextIdx < args.count {
                text = args[nextIdx]
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
                weightsPath = args[nextIdx]
                i += 1
            }
        case "-v", "--voice":
            let nextIdx = i + 1
            if nextIdx < args.count {
                voiceName = args[nextIdx]
                i += 1
            }
        case "--speed":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    var safeVal = val
                    if safeVal.isFinite != true {
                        safeVal = 1.0
                    }
                    if safeVal < 0.1 {
                        safeVal = 0.1
                    }
                    if 10.0 < safeVal {
                        safeVal = 10.0
                    }
                    speed = safeVal
                }
                i += 1
            }
        case "--pitch":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Float(args[nextIdx]) {
                    pitch = val
                }
                i += 1
            }
        case "--benchmark":
            benchmark = true
        case "-h", "--help":
            print("Usage: synthesize [-t <text> | --copy <input.wav> | --probe-f0] [-o <output.wav>] [-w <weights.json>] [-v <voice>] [--speed 1.0] [--pitch 1.0] [--benchmark]")
            return
        default:
            if text.isEmpty {
                if arg.hasPrefix("-") != true {
                    text = arg
                }
            }
        }
        i += 1
    }

    // なぜニューラルボコーダー重みを自動検出・ロードするか:
    // SNN 音響モデル重みと対になる最新の学習済みニューラルボコーダー重みを自動ロードし、
    // 追加オプションなしで最高品位な肉声波形合成および Copy-synthesis を実行可能にするため。
    var vocWeights: NeuralVocoderWeights? = nil
    let defaultVocoderPath = "Models/vocoder_weights.json"
    var vocPath = defaultVocoderPath
    if let explicitWeights = weightsPath {
        if explicitWeights.contains("vocoder") {
            vocPath = explicitWeights
        }
    }
    if FileManager.default.fileExists(atPath: vocPath) {
        if let data = try? Data(contentsOf: URL(fileURLWithPath: vocPath)) {
            let loaded = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: data)
            switch loaded {
            case .some(let vw):
                vocWeights = vw
                print("ニューラルボコーダー重みを読み込みました: \(vocPath)")
            case .none:
                break
            }
        }
    }

    if let copyInput = copyInputPath {
        // Copy-synthesis モード: 教師 PCM -> Mel/F0/voiced -> NeuralVocoder -> WAV
        let wavReader = WavAudioReader()
        let rawPCM: [Float]
        do {
            rawPCM = try wavReader.loadWav16k(from: copyInput)
        } catch {
            print("エラー: 教師 WAV の読み込みに失敗しました (\(copyInput)): \(error)")
            return
        }

        // ピーク正規化
        var peak: Float = 0.0
        var pIdx = 0
        while pIdx < rawPCM.count {
            let a = abs(rawPCM[pIdx])
            if peak < a {
                peak = a
            }
            pIdx += 1
        }
        var rawSumSq: Double = 0.0
        var rawIdx = 0
        while rawIdx < rawPCM.count {
            let v = rawPCM[rawIdx]
            rawSumSq += Double(v * v)
            rawIdx += 1
        }
        var rawRms: Float = 0.0
        if 0 < rawPCM.count {
            rawRms = Float(sqrt(rawSumSq / Double(rawPCM.count)))
        }
        let rawDuration = Float(rawPCM.count) / Float(AudioConfig.sampleRate)
        print("教師音声統計: サンプル数=\(rawPCM.count), 時間=\(String(format: "%.2f", rawDuration))秒, 最大絶対振幅=\(String(format: "%.4f", peak)), RMS=\(String(format: "%.4f", rawRms))")

        var pcm16k = rawPCM
        if 0.01 < peak {
            let normFactor = 0.85 / peak
            var s = 0
            while s < pcm16k.count {
                pcm16k[s] = pcm16k[s] * normFactor
                s += 1
            }
        }

        let melExtractor = MelSpectrogramExtractor(
            sampleRate: Float(AudioConfig.sampleRate),
            melChannels: AudioConfig.melChannels
        )
        let pitchTracker = PitchTracker()
        let mel = melExtractor.extractLogMel(pcm: pcm16k)
        let pitchResult = pitchTracker.track(pcm: pcm16k)

        let vocoder = NeuralVocoder(weights: vocWeights)
        print("Copy-synthesis を開始します: 教師=\(copyInput), 出力=\(outputPath)")
        let startTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let outPCM = vocoder.synthesize(
            mel: mel,
            f0Contour: pitchResult.f0,
            voicedFlags: pitchResult.voiced
        )
        let endTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let elapsedSec = Double(endTime - startTime) / 1_000_000_000.0

        let wavData = WavEncoder.encode(samples: outPCM)
        let outputURL = URL(fileURLWithPath: outputPath)
        let outputDir = outputURL.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: outputDir.path) != true {
            try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        }
        do {
            try wavData.write(to: outputURL)
            print("WAV 音声ファイルを出力しました: \(outputPath) (\(wavData.count) バイト)")
            let sCount = outPCM.count
            var maxAbs: Float = 0.0
            var sumSq: Double = 0.0
            var s = 0
            while s < sCount {
                let v = outPCM[s]
                let absV = abs(v)
                if maxAbs < absV {
                    maxAbs = absV
                }
                sumSq += Double(v * v)
                s += 1
            }
            var rms: Float = 0.0
            if 0 < sCount {
                rms = Float(sqrt(sumSq / Double(sCount)))
            }
            let durationSec = Float(sCount) / Float(AudioConfig.sampleRate)
            print("波形統計: サンプル数=\(sCount), 時間=\(String(format: "%.2f", durationSec))秒 (処理時間=\(String(format: "%.3f", elapsedSec))秒), 最大絶対振幅=\(String(format: "%.4f", maxAbs)), RMS=\(String(format: "%.4f", rms))")
        } catch {
            print("エラー: WAV ファイルの書き込みに失敗しました: \(error)")
            return
        }
        return
    }

    if ablateInputPath != nil && text.isEmpty {
        text = "水をマレーシアから買わなくてはならないのです"
    }

    if probeF0 && text.isEmpty {
        text = "水をマレーシアから買わなくてはならないのです。"
    }

    if text.isEmpty {
        print("エラー: 入力テキストが指定されていません。-t \"日本語テキスト\" を指定してください。")
        print("ヘルプ表示: synthesize --help")
        return
    }

    let weights: SpikingNetworkWeights
    switch weightsPath {
    case .some(let path):
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: data)
            print("重みファイルを読み込みました: \(path)")
        } catch {
            print("警告: 重みファイルの読み込みに失敗しました (\(error))。デフォルト乱数重みを使用します。")
            weights = SpikingNetworkWeights.randomWeights()
        }
    case .none:
        let defaultWeights = "Models/weights.json"
        if FileManager.default.fileExists(atPath: defaultWeights) {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: defaultWeights))
                weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: data)
                print("学習済み重みファイルを自動検出・読み込みました: \(defaultWeights)")
            } catch {
                print("警告: デフォルト重みファイルの読み込みに失敗しました (\(error))。乱数重みを使用します。")
                weights = SpikingNetworkWeights.randomWeights()
            }
        } else {
            weights = SpikingNetworkWeights.randomWeights()
        }
    }

    let voiceProfile = VoiceProfile.preset(named: voiceName)
    let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

    if probeF0 {
        print("==================================================")
        print("【プローブ A / B 計測開始 (design_cfc_f0_probe.md)】")
        print("==================================================")
        let probeText = "水をマレーシアから買わなくてはならないのです。"
        let teacherWavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"

        // 1. プローブ A: synthesize の特徴、F0、有声、ボコーダ
        let samplesA = engine.synthesize(text: probeText)
        let dataA = WavEncoder.encode(samples: samplesA, sampleRate: Int(engine.sampleRate))
        let pathA = ".tmp/wave15/probe_prosody_f0.wav"
        let urlA = URL(fileURLWithPath: pathA)
        let dirA = urlA.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: dirA.path) != true {
            try? FileManager.default.createDirectory(at: dirA, withIntermediateDirectories: true)
        }
        try? dataA.write(to: urlA)
        let metricA = AcousticCentroidMetric.measure(pcm: samplesA)
        let centroid1k5A = metricA.centroidMedian200to1500
        print("プローブ A 波形出力完了: \(pathA) (サンプル数: \(samplesA.count))")
        print("  - プローブ A 200–1500 Hz 重心差中央値: \(String(format: "%.2f", centroid1k5A)) Hz")
        print("  - プローブ A 停止割合 (余弦 > 0.99):   \(String(format: "%.4f", metricA.cosRatio))")

        // 2. プローブ B: A と同じ特徴行列の ch 194 と ch 195 だけを、prepareTrainingPair と同じ補間で教師の PitchTracker F0 に替える。
        //    ボコーダへ渡す F0 と有声は A の prosody のまま。音響モデルの入力だけが変わる。
        let wavReader = WavAudioReader()
        guard let rawPCM = try? wavReader.loadWav16k(from: teacherWavPath) else {
            print("エラー: 教師 WAV の読み込みに失敗しました: \(teacherWavPath)")
            return
        }

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

        let melExtractor = MelSpectrogramExtractor(
            sampleRate: Float(AudioConfig.sampleRate),
            melChannels: AudioConfig.melChannels
        )
        let pitchTracker = PitchTracker()

        var effectiveAlign: UtteranceAlignment? = nil
        let corpusAlignPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/mas_alignments.json"
        if FileManager.default.fileExists(atPath: corpusAlignPath) {
            if let alignMap = try? AlignmentStore.load(from: corpusAlignPath) {
                effectiveAlign = alignMap["BASIC5000_0001"]
            }
        }

        guard let pair = engine.prepareTrainingPair(
            text: probeText,
            pcm16k: pcm16k,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker,
            alignment: effectiveAlign,
            useScaledDuration: true
        ) else {
            print("エラー: prepareTrainingPair に失敗しました")
            return
        }

        engine.workspace.reset()
        engine.neuralVocoder.reset()

        let linguisticFeatures = engine.lengthRegulator.processText(
            text: probeText,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.default.baseF0,
            addBoundarySilence: true,
            meanFramesPerMora: VoiceProfile.default.meanFramesPerMora
        )
        let totalFrames = linguisticFeatures.totalFrames
        let featA = engine.encodeLinguisticFeatures(features: linguisticFeatures)

        var featB = featA
        var f = 0
        while f < totalFrames {
            if f < pair.features.count {
                featB[f][194] = pair.features[f][194]
                featB[f][195] = pair.features[f][195]
            }
            f += 1
        }

        let snnAcousticSeqB = engine.decoder.decodeSequence(
            featuresSeq: featB,
            workspace: engine.workspace
        )

        let melChannels = AudioConfig.melChannels
        var melSeqB = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalFrames)
        var t = 0
        while t < totalFrames {
            if t < snnAcousticSeqB.count {
                let outDim = snnAcousticSeqB[t].count
                let copyCount = min(melChannels, outDim)
                melSeqB[t].withUnsafeMutableBufferPointer { melDst in
                    snnAcousticSeqB[t].withUnsafeBufferPointer { acSrc in
                        melDst.baseAddress!.update(from: acSrc.baseAddress!, count: copyCount)
                    }
                }
            }
            t += 1
        }

        var vocoderF0A = [Float](repeating: 0.0, count: totalFrames)
        var vocoderVoicedA = [Float](repeating: 0.0, count: totalFrames)
        var vf = 0
        while vf < totalFrames {
            if 194 < featA[vf].count {
                vocoderF0A[vf] = featA[vf][194] * 500.0
            }
            if 192 < featA[vf].count {
                vocoderVoicedA[vf] = featA[vf][192]
            }
            vf += 1
        }

        var rawSamplesB = engine.neuralVocoder.synthesize(
            mel: melSeqB,
            f0Contour: vocoderF0A,
            voicedFlags: vocoderVoicedA,
            speaker: .zero
        )

        let silenceMask = engine.computeFrameSilenceMask(
            linguisticFeatures: linguisticFeatures,
            totalFrames: totalFrames
        )
        let frameSize = AudioConfig.hopSize
        var fIdx = 0
        while fIdx < totalFrames {
            let isSilence = silenceMask[fIdx]
            switch isSilence {
            case true:
                var prevIsSilence = true
                if 0 < fIdx {
                    prevIsSilence = silenceMask[fIdx - 1]
                }
                let startSample = fIdx * frameSize
                let endSample = min(rawSamplesB.count, startSample + frameSize)
                switch prevIsSilence {
                case false:
                    let invN = 1.0 / Float(frameSize)
                    var s = startSample
                    while s < endSample {
                        let sampleOffset = s - startSample
                        let fade = 1.0 - (Float(sampleOffset) * invN)
                        rawSamplesB[s] = rawSamplesB[s] * fade
                        s += 1
                    }
                case true:
                    var s = startSample
                    while s < endSample {
                        rawSamplesB[s] = 0.0
                        s += 1
                    }
                }
            case false:
                break
            }
            fIdx += 1
        }

        let targetPeak: Float = 0.85
        var currentPeakB: Float = 0.0
        var pIdxB = 0
        while pIdxB < rawSamplesB.count {
            let absVal = abs(rawSamplesB[pIdxB])
            if currentPeakB < absVal {
                currentPeakB = absVal
            }
            pIdxB += 1
        }
        if 0.01 < currentPeakB {
            var normScale = targetPeak / currentPeakB
            if 6.0 < normScale {
                normScale = 6.0
            }
            var s = 0
            while s < rawSamplesB.count {
                var scaled = rawSamplesB[s] * normScale
                if targetPeak < scaled {
                    scaled = targetPeak
                }
                if scaled < -targetPeak {
                    scaled = -targetPeak
                }
                rawSamplesB[s] = scaled
                s += 1
            }
        }

        let fadeLen = 160
        if (fadeLen * 2) <= rawSamplesB.count {
            let invFade: Float = 1.0 / Float(fadeLen)
            var s = 0
            while s < fadeLen {
                let factor = Float(s) * invFade
                rawSamplesB[s] = rawSamplesB[s] * factor
                s += 1
            }
            let endOffset = rawSamplesB.count - fadeLen
            s = 0
            while s < fadeLen {
                let factor = Float(fadeLen - 1 - s) * invFade
                rawSamplesB[endOffset + s] = rawSamplesB[endOffset + s] * factor
                s += 1
            }
        }

        let dataB = WavEncoder.encode(samples: rawSamplesB, sampleRate: Int(engine.sampleRate))
        let pathB = ".tmp/wave15/probe_tracker_f0.wav"
        let urlB = URL(fileURLWithPath: pathB)
        try? dataB.write(to: urlB)
        let metricB = AcousticCentroidMetric.measure(pcm: rawSamplesB)
        let centroid1k5B = metricB.centroidMedian200to1500
        print("プローブ B 波形出力完了: \(pathB) (サンプル数: \(rawSamplesB.count))")
        print("  - プローブ B 200–1500 Hz 重心差中央値: \(String(format: "%.2f", centroid1k5B)) Hz")
        print("  - プローブ B 停止割合 (余弦 > 0.99):   \(String(format: "%.4f", metricB.cosRatio))")

        print("--------------------------------------------------")
        print("【プローブ計測結果と分岐判定】")
        print("  プローブ A (synthesize prosody F0): \(String(format: "%.2f", centroid1k5A)) Hz")
        print("  プローブ B (decoder tracker F0):    \(String(format: "%.2f", centroid1k5B)) Hz")

        if 22.0 <= centroid1k5B && centroid1k5A < 22.0 {
            print("  ==> 【分岐判定: 韻律分岐（ステップ 2）】プローブ B >= 22Hz かつ プローブ A < 22Hz。音響 CfC を凍結し、韻律の F0 予測器のみを 5 エポック学習します。")
        } else {
            print("  ==> 【分岐判定: 波形分岐（ステップ 3）】プローブ B < 22Hz。動きが教師 F0 を入れたメルからもボコーダへ出ていないため、韻律は回さず音響モデルに波形損失を足して 5 エポック学習します。")
        }
        print("--------------------------------------------------")
        return
    }

    if let ablateTeacherPath = ablateInputPath {
        // Ablation 4本切り分けモード
        print("=== Ablation 4本切り分けモードを開始します ===")
        print("教師 WAV: \(ablateTeacherPath)")
        print("入力テキスト: 「\(text)」")

        let wavReader = WavAudioReader()
        let rawPCM: [Float]
        do {
            rawPCM = try wavReader.loadWav16k(from: ablateTeacherPath)
        } catch {
            print("エラー: 教師 WAV の読み込みに失敗しました (\(ablateTeacherPath)): \(error)")
            return
        }

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

        let melExtractor = MelSpectrogramExtractor(
            sampleRate: Float(AudioConfig.sampleRate),
            melChannels: AudioConfig.melChannels
        )
        let pitchTracker = PitchTracker()
        let teacherMel = melExtractor.extractLogMel(pcm: pcm16k)
        let teacherPitch = pitchTracker.track(pcm: pcm16k)
        let tTeacherFrames = teacherMel.count

        // テキストからの言語・韻律・SNN 音響特徴量抽出
        engine.workspace.reset()
        engine.neuralVocoder.reset()

        let effectiveBaseF0 = voiceProfile.baseF0 * pitch
        let linguisticFeatures = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: speed,
            baseF0: effectiveBaseF0,
            addBoundarySilence: true
        )
        let totalPredFrames = linguisticFeatures.totalFrames
        let inputSeq = engine.encodeLinguisticFeatures(features: linguisticFeatures)
        let snnAcousticSeq = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)
        let melChannels = AudioConfig.melChannels
        var predMel = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: totalPredFrames)
        var t = 0
        while t < totalPredFrames {
            if t < snnAcousticSeq.count {
                let outDim = snnAcousticSeq[t].count
                let copyCount = min(melChannels, outDim)
                var c = 0
                while c < copyCount {
                    predMel[t][c] = snnAcousticSeq[t][c]
                    c += 1
                }
            }
            t += 1
        }
        let predF0 = linguisticFeatures.f0Contour
        let predVoiced = linguisticFeatures.voicedFlags

        // リサンプリング関数（線形補間）
        func resampleContour(source: [Float], targetCount: Int) -> [Float] {
            if targetCount <= 0 || source.isEmpty {
                return []
            }
            if source.count == 1 || targetCount == 1 {
                return [Float](repeating: source[0], count: targetCount)
            }
            var result = [Float](repeating: 0.0, count: targetCount)
            let srcMax = Float(source.count - 1)
            let tgtMax = Float(targetCount - 1)
            var i = 0
            while i < targetCount {
                let relPos = (Float(i) / tgtMax) * srcMax
                let idx0 = Int(relPos)
                let idx1 = min(source.count - 1, idx0 + 1)
                let frac = relPos - Float(idx0)
                result[i] = (source[idx0] * (1.0 - frac)) + (source[idx1] * frac)
                i += 1
            }
            return result
        }

        func resampleVoiced(source: [Float], targetCount: Int) -> [Float] {
            let res = resampleContour(source: source, targetCount: targetCount)
            var binaryVoiced = [Float](repeating: 0.0, count: targetCount)
            var i = 0
            while i < targetCount {
                if 0.5 <= res[i] {
                    binaryVoiced[i] = 1.0
                }
                i += 1
            }
            return binaryVoiced
        }

        let vocoder = NeuralVocoder(weights: vocWeights)

        func saveWav(samples: [Float], path: String, label: String) {
            let wavData = WavEncoder.encode(samples: samples)
            let url = URL(fileURLWithPath: path)
            let dir = url.deletingLastPathComponent()
            if FileManager.default.fileExists(atPath: dir.path) != true {
                try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            }
            try? wavData.write(to: url)
            var maxAbs: Float = 0.0
            var sumSq: Double = 0.0
            var s = 0
            while s < samples.count {
                let v = samples[s]
                let a = abs(v)
                if maxAbs < a { maxAbs = a }
                sumSq += Double(v * v)
                s += 1
            }
            var rms: Float = 0.0
            if 0 < samples.count {
                rms = Float(sqrt(sumSq / Double(samples.count)))
            }
            let dur = Float(samples.count) / Float(AudioConfig.sampleRate)
            print("[\(label)] 出力: \(path) (サンプル数=\(samples.count), 時間=\(String(format: "%.2f", dur))秒, 最大振幅=\(String(format: "%.4f", maxAbs)), RMS=\(String(format: "%.4f", rms)))")
        }

        // 1. ablate_copy.wav: 教師 Mel + 教師 F0
        vocoder.reset()
        let copySamples = vocoder.synthesize(
            mel: teacherMel,
            f0Contour: teacherPitch.f0,
            voicedFlags: teacherPitch.voiced
        )
        saveWav(samples: copySamples, path: ".tmp/wave15/ablate_copy.wav", label: "Ablation 1: 教師 Mel + 教師 F0")

        // 2. ablate_teacherMel_predF0.wav: 教師 Mel + 予測 F0 (長さを教師 Mel にリサンプル)
        vocoder.reset()
        let predF0OnTeacher = resampleContour(source: predF0, targetCount: tTeacherFrames)
        let predVoicedOnTeacher = resampleVoiced(source: predVoiced, targetCount: tTeacherFrames)
        let teacherMelPredF0Samples = vocoder.synthesize(
            mel: teacherMel,
            f0Contour: predF0OnTeacher,
            voicedFlags: predVoicedOnTeacher
        )
        saveWav(samples: teacherMelPredF0Samples, path: ".tmp/wave15/ablate_teacherMel_predF0.wav", label: "Ablation 2: 教師 Mel + 予測 F0")

        // 3. ablate_predMel_teacherF0.wav: 予測 Mel + 教師 F0 (長さを予測 Mel にリサンプル)
        vocoder.reset()
        let teacherF0OnPred = resampleContour(source: teacherPitch.f0, targetCount: totalPredFrames)
        let teacherVoicedOnPred = resampleVoiced(source: teacherPitch.voiced, targetCount: totalPredFrames)
        let predMelTeacherF0Samples = vocoder.synthesize(
            mel: predMel,
            f0Contour: teacherF0OnPred,
            voicedFlags: teacherVoicedOnPred
        )
        saveWav(samples: predMelTeacherF0Samples, path: ".tmp/wave15/ablate_predMel_teacherF0.wav", label: "Ablation 3: 予測 Mel + 教師 F0")

        // 4. ablate_tts.wav: 予測 Mel + 予測 F0 (通常 TTS)
        vocoder.reset()
        let ttsSamples = vocoder.synthesize(
            mel: predMel,
            f0Contour: predF0,
            voicedFlags: predVoiced
        )
        saveWav(samples: ttsSamples, path: ".tmp/wave15/ablate_tts.wav", label: "Ablation 4: 予測 Mel + 予測 F0")

        print("=== Ablation 4本切り分けの生成が完了しました ===")
        return
    }

    print("音声合成を開始します: 「\(text)」 (話者: \(voiceProfile.name), 速度: \(speed), ピッチ: \(pitch))")

    let dbgLinguistic = engine.lengthRegulator.processText(
        text: text,
        normalizer: engine.normalizer,
        prosodyModel: engine.prosodyModel,
        vocabulary: engine.vocabulary,
        prosodyPredictor: engine.prosodyPredictor,
        speedFactor: speed,
        baseF0: voiceProfile.baseF0 * pitch,
        addBoundarySilence: true
    )
    print("=== 音素別継続時間分析 (通常合成) ===")
    var dbgP2 = 0
    while dbgP2 < dbgLinguistic.phoneIds.count {
        let pid = Int(dbgLinguistic.phoneIds[dbgP2])
        let dur = dbgLinguistic.durations[dbgP2]
        let sym = engine.vocabulary.token(for: pid)
        print("[\(dbgP2)] ID:\(pid) (\(sym)): \(dur) frames (\(dur * 10) ms)")
        dbgP2 += 1
    }
    print("合計フレーム数: \(dbgLinguistic.totalFrames) (\(dbgLinguistic.totalFrames * 10) ms)")
    print("=====================================")

    let startTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

    let wavData = engine.synthesizeWav(
        text: text,
        voice: voiceProfile,
        speed: speed,
        pitch: pitch
    )

    let endTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let elapsedSec = Double(endTime - startTime) / 1_000_000_000.0

    do {
        let outputURL = URL(fileURLWithPath: outputPath)
        let outputDir = outputURL.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: outputDir.path) != true {
            try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        }
        try wavData.write(to: outputURL)
        print("WAV 音声ファイルを出力しました: \(outputPath) (\(wavData.count) バイト)")
        let pcmBytes = wavData.subdata(in: 44..<wavData.count)
        var maxAbs: Float = 0.0
        var sumSq: Double = 0.0
        var clipCount: Int = 0
        let sCount = pcmBytes.count / 2
        pcmBytes.withUnsafeBytes { rawPtr in
            let ptr16 = rawPtr.bindMemory(to: Int16.self)
            var s = 0
            while s < sCount {
                let v = Float(ptr16[s]) / 32767.0
                let absV = abs(v)
                if maxAbs < absV {
                    maxAbs = absV
                }
                if 0.95 <= absV {
                    clipCount += 1
                }
                sumSq += Double(v * v)
                s += 1
            }
        }
        var rms: Float = 0.0
        if 0 < sCount {
            rms = Float(sqrt(sumSq / Double(sCount)))
        }
        print("波形統計: サンプル数=\(sCount), 最大絶対振幅=\(String(format: "%.4f", maxAbs)), RMS=\(String(format: "%.4f", rms)), クリップ率=\(String(format: "%.2f", Float(clipCount) / Float(max(1, sCount)) * 100.0))%")

        print("--- 音素区間別実測 RMS / dB ---")
        var phStartSample = 0
        var curPhIdx = 0
        while curPhIdx < dbgLinguistic.phoneIds.count {
            let pid = Int(dbgLinguistic.phoneIds[curPhIdx])
            let durFrames = Int(dbgLinguistic.durations[curPhIdx])
            let phLenSamples = durFrames * 160
            let phEndSample = min(sCount, phStartSample + phLenSamples)
            let sym = engine.vocabulary.token(for: pid)

            var phSumSq: Double = 0.0
            var sampleCount = 0
            pcmBytes.withUnsafeBytes { rawPtr in
                let ptr16 = rawPtr.bindMemory(to: Int16.self)
                var s = phStartSample
                while s < phEndSample {
                    let v = Float(ptr16[s]) / 32767.0
                    phSumSq += Double(v * v)
                    sampleCount += 1
                    s += 1
                }
            }
            var phRms: Float = 0.0
            var phDb: Float = -99.0
            if 0 < sampleCount {
                phRms = Float(sqrt(phSumSq / Double(sampleCount)))
                if 0.00001 < phRms {
                    phDb = 20.0 * log10f(phRms)
                }
            }
            let startSec = Float(phStartSample) / 16000.0
            let endSec = Float(phEndSample) / 16000.0
            print("[\(curPhIdx)] \(sym) (\(String(format: "%.3f", startSec))s - \(String(format: "%.3f", endSec))s, \(durFrames * 10)ms): RMS=\(String(format: "%.4f", phRms)), dB=\(String(format: "%.1f", phDb)) dB")

            phStartSample = phEndSample
            curPhIdx += 1
        }
        print("-------------------------------")
        if 20060 < sCount {
            var sampleStr = ""
            pcmBytes.withUnsafeBytes { rawPtr in
                let ptr16 = rawPtr.bindMemory(to: Int16.self)
                var k = 20000
                while k < 20060 {
                    let v = Float(ptr16[k]) / 32767.0
                    sampleStr += String(format: "%.3f, ", v)
                    k += 1
                }
            }
            print("中間部サンプル (20000-20059): [\(sampleStr)]")
        }
    } catch {
        print("エラー: WAV ファイルの書き込みに失敗しました: \(error)")
        return
    }

    if benchmark {
        let pcmByteSize = max(0, wavData.count - 44)
        let totalSamples = pcmByteSize / 2
        let audioDurationSec = Double(totalSamples) / Double(AudioConfig.sampleRate)
        var rtf: Double = 0.0
        if 0.0 < audioDurationSec {
            rtf = elapsedSec / audioDurationSec
        }

        print("--- [Benchmark Statistics] ---")
        print("合成処理時間: \(String(format: "%.4f", elapsedSec)) 秒")
        print("音声実時間:   \(String(format: "%.4f", audioDurationSec)) 秒")
        print("Real-Time Factor (RTF): \(String(format: "%.4f", rtf))")
        print("------------------------------")
    }
}

main()
