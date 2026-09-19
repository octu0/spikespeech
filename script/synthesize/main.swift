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
    var copyInputPath: String? = nil
    var ablateInputPath: String? = nil

    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
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
            print("Usage: synthesize [-t <text> | --copy <input.wav>] [-o <output.wav>] [-w <weights.json>] [-v <voice>] [--speed 1.0] [--pitch 1.0] [--benchmark]")
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
