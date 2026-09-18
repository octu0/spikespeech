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
