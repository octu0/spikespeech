import Darwin
import Foundation
import SpikeSpeech

/// プロセスの常駐メモリサイズをバイト単位で取得する。
func getResidentMemoryBytes() -> UInt64 {
    var taskInfo = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if kerr == KERN_SUCCESS {
        return taskInfo.resident_size
    }
    return 0
}

/// SpikeSpeech 総合パフォーマンスベンチマーク CLI
func main() {
    let args = CommandLine.arguments
    var iterations: Int = 50
    var text: String = "こんにちは、世界。スパイクニューラルネットワークによる超低遅延音声合成です。"
    var mode: String = "all"

    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "-n", "--iterations":
            let nextIdx = i + 1
            if nextIdx < args.count {
                if let val = Int(args[nextIdx]) {
                    var safeVal = val
                    if safeVal < 1 {
                        safeVal = 1
                    }
                    iterations = safeVal
                }
                i += 1
            }
        case "-t", "--text":
            let nextIdx = i + 1
            if nextIdx < args.count {
                text = args[nextIdx]
                i += 1
            }
        case "-m", "--mode":
            let nextIdx = i + 1
            if nextIdx < args.count {
                mode = args[nextIdx]
                i += 1
            }
        case "-h", "--help":
            print("Usage: benchmark [-n <iterations>] [-t <text>] [-m all|linguistics|snn|vocoder|e2e]")
            return
        default:
            break
        }
        i += 1
    }

    if iterations < 1 {
        iterations = 1
    }

    print("==========================================================")
    print("SpikeSpeech パフォーマンスベンチマークスイート")
    print("==========================================================")
    print("対象テキスト: 「\(text)」 (\(text.count) 文字)")
    print("反復測定回数: \(iterations) 回")
    print("計測モード:   \(mode)")
    let initialRSS = getResidentMemoryBytes()
    print("初期常駐メモリ (RSS): \(String(format: "%.2f", Double(initialRSS) / (1024.0 * 1024.0))) MB")
    print("----------------------------------------------------------")

    // 1. 言語処理フロントエンド (Linguistics) の計測
    if mode == "all" || mode == "linguistics" {
        print("1. 言語処理フロントエンド (Linguistics)")
        let normalizer = TextNormalizer()
        let prosody = ProsodyModel()
        let vocab = PhonemeVocabulary()
        let regulator = LengthRegulator()

        regulator.processText(text: text, normalizer: normalizer, prosodyModel: prosody, vocabulary: vocab)

        let tStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var iter = 0
        var totalChars = 0
        var totalFramesGenerated = 0
        while iter < iterations {
            let res = regulator.processText(text: text, normalizer: normalizer, prosodyModel: prosody, vocabulary: vocab)
            totalChars += text.count
            totalFramesGenerated += res.totalFrames
            iter += 1
        }
        let tEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let elapsed = Double(tEnd - tStart) / 1_000_000_000.0
        let charsPerSec = Double(totalChars) / elapsed
        let avgMs = (elapsed / Double(iterations)) * 1000.0

        print("   平均処理時間:     \(String(format: "%.3f", avgMs)) ms / 発話")
        print("   スループット:     \(String(format: "%.1f", charsPerSec)) 文字/秒")
        print("   平均生成フレーム: \(totalFramesGenerated / iterations) フレーム (10ms/フレーム)")
        print("----------------------------------------------------------")
    }

    // 2. 多層 SNN 音響デコーダー推論の計測
    if mode == "all" || mode == "snn" {
        print("2. 多層 SNN 音響デコーダー推論")
        let maxHidden = 1024
        let inDim = 128
        let outDim = 80
        let testFrameCount = 100
        let dummySeq = [[Float]](repeating: [Float](repeating: 0.35, count: inDim), count: testFrameCount)

        let layerConfigs = [1, 2, 3]
        var lIdx = 0
        while lIdx < layerConfigs.count {
            let numLayers = layerConfigs[lIdx]
            let weights = SpikingNetworkWeights.randomWeights(
                inputDim: inDim,
                maxHiddenDim: maxHidden,
                outputDim: outDim,
                timeSteps: 4,
                numLayers: numLayers
            )
            let decoder = SpikingAcousticDecoder(weights: weights)
            let workspace = AcousticWorkspace(maxHiddenDim: maxHidden, outputDim: outDim, numLayers: numLayers)

            decoder.decodeSequence(featuresSeq: dummySeq, workspace: workspace)

            let tStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var iter = 0
            while iter < iterations {
                decoder.decodeSequence(featuresSeq: dummySeq, workspace: workspace)
                iter += 1
            }
            let tEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let elapsed = Double(tEnd - tStart) / 1_000_000_000.0
            let totalFrames = testFrameCount * iterations
            let msPerFrame = (elapsed / Double(totalFrames)) * 1000.0
            let framesPerSec = Double(totalFrames) / elapsed

            print("   [層数 \(numLayers)]: \(String(format: "%.4f", msPerFrame)) ms/フレーム (\(String(format: "%.1f", framesPerSec)) FPS)")
            lIdx += 1
        }
        print("----------------------------------------------------------")
    }

    // 3. ニューラルボコーダー (NeuralVocoder) の計測
    if mode == "all" || mode == "vocoder" {
        print("3. ニューラルボコーダー (NeuralVocoder) 波形合成")
        let vocoder = NeuralVocoder()

        let testFrameCount = 100
        let dummyMel = [[Float]](repeating: [Float](repeating: 0.2, count: AudioConfig.melChannels), count: testFrameCount)
        let dummyF0 = [Float](repeating: 130.0, count: testFrameCount)
        let dummyVoiced = [Float](repeating: 1.0, count: testFrameCount)

        vocoder.synthesize(mel: dummyMel, f0Contour: dummyF0, voicedFlags: dummyVoiced)

        let tStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var iter = 0
        while iter < iterations {
            vocoder.synthesize(mel: dummyMel, f0Contour: dummyF0, voicedFlags: dummyVoiced)
            iter += 1
        }
        let tEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let elapsed = Double(tEnd - tStart) / 1_000_000_000.0
        let audioTotalSec = Double(testFrameCount * iterations) * 0.01
        let rtf = elapsed / audioTotalSec

        print("   合成処理時間:   \(String(format: "%.4f", elapsed)) 秒")
        print("   音声実時間:     \(String(format: "%.4f", audioTotalSec)) 秒")
        print("   ボコーダー RTF: \(String(format: "%.6f", rtf))")
        print("----------------------------------------------------------")
    }

    // 4. End-to-End SpikeSpeechEngine の計測
    if mode == "all" || mode == "e2e" {
        print("4. End-to-End SpikeSpeechEngine 音声合成パイプライン")
        let engine = SpikeSpeechEngine()

        engine.synthesizeWav(text: text)

        let tStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var iter = 0
        var totalAudioSamples = 0
        while iter < iterations {
            let wavData = engine.synthesizeWav(text: text)
            totalAudioSamples += max(0, (wavData.count - 44) / 2)
            iter += 1
        }
        let tEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let elapsed = Double(tEnd - tStart) / 1_000_000_000.0
        let totalAudioDuration = Double(totalAudioSamples) / Double(AudioConfig.sampleRate)
        let e2eRTF = elapsed / totalAudioDuration
        let currentRSS = getResidentMemoryBytes()
        let rssMB = Double(currentRSS) / (1024.0 * 1024.0)

        print("   E2E 合計処理時間:   \(String(format: "%.4f", elapsed)) 秒")
        print("   生成音声総実時間:   \(String(format: "%.4f", totalAudioDuration)) 秒")
        print("   E2E Real-Time Factor: \(String(format: "%.4f", e2eRTF))")
        print("   E2E スループット比:   リアルタイムの \(String(format: "%.1f", 1.0 / e2eRTF)) 倍速")
        print("   終了時常駐メモリ (RSS): \(String(format: "%.2f", rssMB)) MB")
        print("----------------------------------------------------------")
    }

    print("ベンチマーク計測が完了しました。")
}

main()
