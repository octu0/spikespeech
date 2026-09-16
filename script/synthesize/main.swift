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

    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
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
            print("Usage: synthesize -t <text> [-o <output.wav>] [-w <weights.json>] [-v <voice>] [--speed 1.0] [--pitch 1.0] [--benchmark]")
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

    // なぜニューラルボコーダー重みを自動検出するか:
    // SNN 音響モデル重みと対になる最新の学習済みニューラルボコーダー重みを自動ロードし、
    // 追加オプションなしで最高品位な肉声波形合成を実行可能にするため。
    var vocWeights: NeuralVocoderWeights? = nil
    let defaultVocoderPath = "Models/vocoder_weights.json"
    if FileManager.default.fileExists(atPath: defaultVocoderPath) {
        if let data = try? Data(contentsOf: URL(fileURLWithPath: defaultVocoderPath)) {
            let loaded = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: data)
            switch loaded {
            case .some(let vw):
                vocWeights = vw
                print("学習済みニューラルボコーダー重みを自動検出・読み込みました: \(defaultVocoderPath)")
            case .none:
                break
            }
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
        try wavData.write(to: outputURL)
        print("WAV 音声ファイルを出力しました: \(outputPath) (\(wavData.count) バイト)")
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
