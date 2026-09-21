import Foundation
import SpikeSpeech

/// 日本語音声コーパス音響特徴 Forced Alignment CLI
func main() {
    let args = CommandLine.arguments
    var datasetPath: String? = nil
    var outputPath: String? = nil
    var maxSamples: Int = 1000

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
        case "-o", "--output":
            let nextIdx = i + 1
            if nextIdx < args.count {
                outputPath = args[nextIdx]
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
        case "-h", "--help":
            print("Usage: align -d <corpus_dir> [-o <output.json>] [-s <samples>]")
            return
        default:
            break
        }
        i += 1
    }

    guard let corpusDir = datasetPath else {
        print("エラー: コーパスディレクトリ (-d) を指定してください。")
        return
    }

    let corpusURL = URL(fileURLWithPath: corpusDir)
    let transcriptURL = corpusURL.appendingPathComponent("transcript_utf8.txt")
    let wavDirURL = corpusURL.appendingPathComponent("wav")

    guard let transcriptContent = try? String(contentsOf: transcriptURL, encoding: .utf8) else {
        print("エラー: トランスクリプトが読み込めません: \(transcriptURL.path)")
        return
    }

    let finalOutputPath: String
    switch outputPath {
    case .some(let p):
        finalOutputPath = p
    case .none:
        finalOutputPath = corpusURL.appendingPathComponent("alignments.json").path
    }

    print("==================================================")
    print("Swift 音響特徴 Forced Alignment を開始します")
    print("  コーパス: \(corpusURL.path)")
    print("  出力先: \(finalOutputPath)")
    print("  最大発話数: \(maxSamples)")
    print("==================================================")

    let normalizer = TextNormalizer()
    let vocabulary = PhonemeVocabulary()
    let prosodyModel = ProsodyModel()
    let melExtractor = MelSpectrogramExtractor(
        sampleRate: Float(AudioConfig.sampleRate),
        melChannels: AudioConfig.melChannels
    )
    let pitchTracker = PitchTracker()
    let aligner = AcousticForcedAligner()
    let wavReader = WavAudioReader()

    let lines = transcriptContent.components(separatedBy: .newlines)
    var alignments: [UtteranceAlignment] = []

    var lineIdx = 0
    while lineIdx < lines.count {
        if maxSamples <= alignments.count {
            break
        }

        let line = lines[lineIdx].trimmingCharacters(in: .whitespacesAndNewlines)
        if line.isEmpty {
            lineIdx += 1
            continue
        }

        let parts = line.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: true)
        if parts.count != 2 {
            lineIdx += 1
            continue
        }

        let uttId = String(parts[0])
        let text = String(parts[1])

        let wavURL = wavDirURL.appendingPathComponent("\(uttId).wav")
        guard let pcm = try? wavReader.loadWav16k(from: wavURL.path) else {
            lineIdx += 1
            continue
        }

        let totalFrames = pcm.count / AudioConfig.hopSize
        if totalFrames <= 0 {
            lineIdx += 1
            continue
        }

        // VAD による発話区間境界の検出
        let boundaries = SpikeSpeechEngine.detectSpeechBoundaries(
            pcm: pcm,
            hopSize: AudioConfig.hopSize,
            totalFrames: totalFrames
        )
        let leadSilence = boundaries.leadSilence
        let speechFrames = boundaries.speechFrames
        let trailSilence = boundaries.trailSilence

        // テキストから音素系列を抽出
        let morphemes = normalizer.normalize(text: text)
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
        if phrases.isEmpty {
            lineIdx += 1
            continue
        }

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

        if tokens.isEmpty || speechFrames <= 0 || speechFrames < tokens.count {
            lineIdx += 1
            continue
        }

        // 音響特徴抽出
        let features = aligner.extractFeatures(
            pcm: pcm,
            hopSize: AudioConfig.hopSize,
            startFrame: leadSilence,
            frameCount: speechFrames,
            pitchTracker: pitchTracker,
            melExtractor: melExtractor
        )

        // DP Forced Alignment
        guard let durations = aligner.align(features: features, phonemes: tokens) else {
            lineIdx += 1
            continue
        }

        if durations.count != tokens.count {
            lineIdx += 1
            continue
        }

        var phoneAlignments: [PhonemeAlignment] = []
        var tIdx = 0
        while tIdx < tokens.count {
            phoneAlignments.append(PhonemeAlignment(
                symbol: tokens[tIdx].symbol,
                phoneId: Int32(tokens[tIdx].id),
                durationFrames: durations[tIdx]
            ))
            tIdx += 1
        }

        let uttAlign = UtteranceAlignment(
            utteranceId: uttId,
            leadSilenceFrames: leadSilence,
            trailSilenceFrames: trailSilence,
            totalSpeechFrames: speechFrames,
            phonemes: phoneAlignments
        )

        alignments.append(uttAlign)

        if (alignments.count % 100) == 0 {
            print("  アライメント進捗: [\(alignments.count)/\(maxSamples)] 発話完了")
        }

        lineIdx += 1
    }

    print("==================================================")
    print("アライメント完了: \(alignments.count) 発話のアライメントを生成しました")

    if alignments.isEmpty {
        print("エラー: アライメントされた発話数が 0 件です。")
        return
    }

    do {
        let outURL = URL(fileURLWithPath: finalOutputPath)
        let outDir = outURL.deletingLastPathComponent().path
        if FileManager.default.fileExists(atPath: outDir) != true {
            try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        }
        try AlignmentStore.save(alignments, to: finalOutputPath)
        print("アライメント結果を保存しました: \(finalOutputPath)")
    } catch {
        print("エラー: アライメント保存に失敗しました: \(error)")
        return
    }

    // 音素別平均フレーム数の算出と表示
    let averages = AlignmentStore.computeAverageDurations(from: alignments)
    print("音素 ID 別平均継続時間 (推論正本):")
    for (pid, avg) in averages.sorted(by: { $0.key < $1.key }) {
        let sym = vocabulary.token(for: Int(pid))
        print("  [\(pid)] \(sym): \(String(format: "%.1f", avg)) frames (\(String(format: "%.0f", avg * 10.0)) ms)")
    }
    print("==================================================")
}

main()
