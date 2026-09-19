import XCTest
@testable import SpikeSpeech

final class AblationAnalysisTests: XCTestCase {
    func testAnalyzeAblationFourWavs() throws {
        let wavReader = WavAudioReader()
        let melExtractor = MelSpectrogramExtractor(
            sampleRate: Float(AudioConfig.sampleRate),
            melChannels: AudioConfig.melChannels
        )
        let pitchTracker = PitchTracker()

        let paths = [
            ("1. Copy (Teacher Mel + Teacher F0)", ".tmp/wave15/ablate_copy.wav"),
            ("2. Teacher Mel + Pred F0", ".tmp/wave15/ablate_teacherMel_predF0.wav"),
            ("3. Pred Mel + Teacher F0", ".tmp/wave15/ablate_predMel_teacherF0.wav"),
            ("4. TTS (Pred Mel + Pred F0)", ".tmp/wave15/ablate_tts.wav")
        ]

        print("\n=======================================================")
        print("Ablation 4本 客観音響特性分析 (Ablation Analysis)")
        print("=======================================================")

        var pIdx = 0
        while pIdx < paths.count {
            let (label, path) = paths[pIdx]
            guard FileManager.default.fileExists(atPath: path) else {
                print("[\(label)] ファイルが存在しません: \(path)")
                pIdx += 1
                continue
            }

            let pcm = try wavReader.loadWav16k(from: path)
            let mel = melExtractor.extractLogMel(pcm: pcm)
            let pitch = pitchTracker.track(pcm: pcm)

            // F0 統計
            var voicedF0s: [Float] = []
            var f = 0
            while f < pitch.frameCount {
                if 0.5 <= pitch.voiced[f] && 50.0 <= pitch.f0[f] && pitch.f0[f] <= 500.0 {
                    voicedF0s.append(pitch.f0[f])
                }
                f += 1
            }

            var meanF0: Float = 0.0
            var stdF0: Float = 0.0
            var minF0: Float = 0.0
            var maxF0: Float = 0.0
            if voicedF0s.isEmpty != true {
                let sumF = voicedF0s.reduce(0, +)
                meanF0 = sumF / Float(voicedF0s.count)
                let sumSq = voicedF0s.reduce(0) { $0 + powf($1 - meanF0, 2) }
                stdF0 = sqrtf(sumSq / Float(voicedF0s.count))
                minF0 = voicedF0s.min() ?? 0.0
                maxF0 = voicedF0s.max() ?? 0.0
            }

            // Mel フォルマント時間動態度 (フレーム間スペクトル差分平均: Spectral Flux)
            // 横縞（定常平坦）なら flux はほぼ 0 に近くなる。動的フォルマントなら大きな値になる。
            var spectralFluxSum: Float = 0.0
            var fluxCount = 0
            var t = 1
            while t < mel.count {
                var frameDiff: Float = 0.0
                var c = 0
                while c < AudioConfig.melChannels {
                    let d = mel[t][c] - mel[t - 1][c]
                    frameDiff += d * d
                    c += 1
                }
                spectralFluxSum += sqrtf(frameDiff)
                fluxCount += 1
                t += 1
            }
            var meanSpectralFlux: Float = 0.0
            if 0 < fluxCount {
                meanSpectralFlux = spectralFluxSum / Float(fluxCount)
            }

            // チャネル間分散（スペクトルの凹凸・フォルマントピーク度）
            var channelVarSum: Float = 0.0
            t = 0
            while t < mel.count {
                let m = mel[t].reduce(0, +) / Float(AudioConfig.melChannels)
                let v = mel[t].reduce(0) { $0 + powf($1 - m, 2) } / Float(AudioConfig.melChannels)
                channelVarSum += v
                t += 1
            }
            var meanChannelVar: Float = 0.0
            if 0 < mel.count {
                meanChannelVar = channelVarSum / Float(mel.count)
            }

            print("[\(label)]")
            print("  時間: \(String(format: "%.2f", Float(pcm.count) / 16000.0))s (\(pcm.count) samples, \(mel.count) frames)")
            print("  有声 F0: 平均=\(String(format: "%.1f", meanF0)) Hz, 標準偏差=\(String(format: "%.1f", stdF0)) Hz, min=\(String(format: "%.1f", minF0)) Hz, max=\(String(format: "%.1f", maxF0)) Hz")
            print("  スペクトル動態度 (Spectral Flux): \(String(format: "%.4f", meanSpectralFlux)) (時間変化の激しさ)")
            print("  フォルマント凹凸度 (Channel Variance): \(String(format: "%.4f", meanChannelVar)) (平坦度/横縞の逆指標)")
            print("-------------------------------------------------------")

            pIdx += 1
        }
    }

    func testCheckTTSF0Distribution() throws {
        let weightsPath = "Models/weights.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: weightsPath)),
              let decoded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) else {
            XCTFail("Models/weights.json を読み込めません")
            return
        }

        let predictor = ProsodyPredictor(weights: decoded.prosodyWeights)
        print("[Loaded F0 Weights] wConv count: \(decoded.prosodyWeights?.f0Weights.wConv.count ?? 0), b2: \(decoded.prosodyWeights?.f0Weights.b2 ?? [])")
        let normalizer = TextNormalizer(morphology: ViterbiMorphology())
        let vocabulary = PhonemeVocabulary()
        let prosodyModel = ProsodyModel()
        let lengthRegulator = LengthRegulator()

        let sentences = ["こんにちは", "今日はいい天気です"]
        for sentence in sentences {
            let morphemes = normalizer.normalize(text: sentence)
            let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
            let durations = predictor.predictDurations(
                phrases: phrases,
                vocabulary: vocabulary,
                lengthRegulator: lengthRegulator,
                speedFactor: 1.0,
                applyFluctuation: false
            )
            let f0Result = predictor.predictF0Contour(
                phrases: phrases,
                vocabulary: vocabulary,
                baseF0: 220.0,
                prosodyModel: prosodyModel,
                durations: durations,
                applyFluctuation: false
            )

            var voicedF0s: [Float] = []
            var f = 0
            while f < f0Result.totalFrames {
                if 0.5 <= f0Result.voicedFlags[f] {
                    voicedF0s.append(f0Result.f0Contour[f])
                }
                f += 1
            }
            let meanF0 = voicedF0s.reduce(0, +) / Float(max(1, voicedF0s.count))
            let minF0 = voicedF0s.min() ?? 0.0
            let maxF0 = voicedF0s.max() ?? 0.0
            print("[TTS F0 Check] 「\(sentence)」 有声フレーム数: \(voicedF0s.count), 平均 F0: \(String(format: "%.1f", meanF0)) Hz, min: \(String(format: "%.1f", minF0)) Hz, max: \(String(format: "%.1f", maxF0)) Hz")
        }
    }

    /// 受入基準 5: .tmp/wave15_spec/tts_tenki.png と copy_BASIC5000_0001.png を保存
    func testGenerateWave15SpectrogramPNGs() throws {
        let targets = [
            (".tmp/wave15/tts_tenki.wav", ".tmp/wave15_spec/tts_tenki.png"),
            (".tmp/wave15/copy_BASIC5000_0001.wav", ".tmp/wave15_spec/copy_BASIC5000_0001.png"),
            (".tmp/wave15/tts_konnichiwa.wav", ".tmp/wave15_spec/tts_konnichiwa.png"),
            (".tmp/wave15/tts_aiueo.wav", ".tmp/wave15_spec/tts_aiueo.png")
        ]

        var idx = 0
        while idx < targets.count {
            let (wavPath, pngPath) = targets[idx]
            if FileManager.default.fileExists(atPath: wavPath) {
                try SpectrogramRenderer.renderWavToPNG(wavPath: wavPath, outputPath: pngPath, scale: 2)
                let size = (try? Data(contentsOf: URL(fileURLWithPath: pngPath)).count) ?? 0
                print("[Spectrogram] 生成完了: \(pngPath) (\(size) バイト)")
                XCTAssertTrue(0 < size, "スペクトログラム PNG が空です: \(pngPath)")
            }
            idx += 1
        }
    }

    /// 教師 Mel と予測 Mel の詳細音響統計（平均、分散、min, max, 周波数帯域別）の比較分析
    func testCompareTeacherMelVsPredMel() throws {
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else {
            print("BASIC5000_0001.wav が存在しません")
            return
        }

        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)
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
        let teacherMel = melExtractor.extractLogMel(pcm: pcm16k)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        let engine = SpikeSpeechEngine(weights: weights)

        let text = "水をマレーシアから買わなくてはならないのです。"
        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true
        )
        let inputSeq = engine.encodeLinguisticFeatures(features: linguistic)
        let snnOut = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)

        func stats(mel: [[Float]], name: String) {
            var minV: Float = 999.0
            var maxV: Float = -999.0
            var sumV: Float = 0.0
            var count = 0
            var lowBandSum: Float = 0.0
            var midBandSum: Float = 0.0
            var highBandSum: Float = 0.0

            var t = 0
            while t < mel.count {
                var c = 0
                while c < AudioConfig.melChannels {
                    let v = mel[t][c]
                    if v < minV { minV = v }
                    if maxV < v { maxV = v }
                    sumV += v
                    count += 1
                    if c < 16 {
                        lowBandSum += v
                    } else {
                        switch c < 40 {
                        case true:
                            midBandSum += v
                        case false:
                            highBandSum += v
                        }
                    }
                    c += 1
                }
                t += 1
            }

            let meanV = sumV / Float(max(1, count))
            let lowMean = lowBandSum / Float(max(1, mel.count * 16))
            let midMean = midBandSum / Float(max(1, mel.count * 24))
            let highMean = highBandSum / Float(max(1, mel.count * 24))

            var sumSq: Float = 0.0
            t = 0
            while t < mel.count {
                var c = 0
                while c < AudioConfig.melChannels {
                    let diff = mel[t][c] - meanV
                    sumSq += diff * diff
                    c += 1
                }
                t += 1
            }
            let stdV = sqrtf(sumSq / Float(max(1, count)))

            print("\n[\(name)]")
            print("  フレーム数: \(mel.count)")
            print("  全体統計: 平均=\(String(format: "%.3f", meanV)), 標準偏差=\(String(format: "%.3f", stdV)), min=\(String(format: "%.3f", minV)), max=\(String(format: "%.3f", maxV))")
            print("  帯域別平均: 低域(0-15)=\(String(format: "%.3f", lowMean)), 中域(16-39)=\(String(format: "%.3f", midMean)), 高域(40-63)=\(String(format: "%.3f", highMean))")
        }

        stats(mel: teacherMel, name: "教師 Mel (BASIC5000_0001)")
        stats(mel: snnOut, name: "SNN 予測 Mel (テキスト: 水をマレーシアから...)")
    }

    /// 完全アライメントされた入力特徴量（319フレーム）からSNNが生成したMelによるボコーダ合成
    func testSynthesizeAlignedSNNMel() throws {
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else { return }

        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)
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

        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)
        let pitchTracker = PitchTracker()
        let pitchResult = pitchTracker.track(pcm: pcm16k)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        let engine = SpikeSpeechEngine(weights: weights)

        let text = "水をマレーシアから買わなくてはならないのです。"
        guard let pair = engine.prepareTrainingPair(
            text: text,
            pcm16k: pcm16k,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker
        ) else {
            XCTFail("prepareTrainingPair に失敗しました")
            return
        }

        engine.workspace.reset()
        let snnMel = engine.decoder.decodeSequence(featuresSeq: pair.features, workspace: engine.workspace)
        print("[Aligned SNN Mel] フレーム数: \(snnMel.count), 教師フレーム数: \(pitchResult.frameCount)")

        var vocWeights: NeuralVocoderWeights? = nil
        if let vData = try? Data(contentsOf: URL(fileURLWithPath: "Models/vocoder_weights.json")) {
            vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }
        let vocoder = NeuralVocoder(weights: vocWeights)
        let samples = vocoder.synthesize(
            mel: snnMel,
            f0Contour: pitchResult.f0,
            voicedFlags: pitchResult.voiced
        )

        let wavData = WavEncoder.encode(samples: samples)
        let outPath = ".tmp/wave15/test_aligned_snn.wav"
        try wavData.write(to: URL(fileURLWithPath: outPath))
        print("[Aligned SNN WAV] 出力完了: \(outPath) (\(samples.count) samples, \(Float(samples.count)/16000.0)s)")
    }

    /// こんにちはの音素・Duration・無音マスク・F0の詳細調査
    func testDebugKonnichiwaAlignment() throws {
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        let engine = SpikeSpeechEngine(weights: weights)

        let linguistic = engine.lengthRegulator.processText(
            text: "こんにちは",
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true
        )

        print("\n[Konnichiwa Debug]")
        print("Total frames: \(linguistic.totalFrames)")
        var p = 0
        while p < linguistic.phoneIds.count {
            let pid = Int(linguistic.phoneIds[p])
            let sym = engine.vocabulary.token(for: pid)
            let dur = linguistic.durations[p]
            print("  Phone[\(p)]: id=\(pid) (\(sym)), dur=\(dur)")
            p += 1
        }

        let silenceMask = engine.computeFrameSilenceMask(linguisticFeatures: linguistic, totalFrames: linguistic.totalFrames)
        let stopBurstMask = engine.computeStopBurstMask(linguisticFeatures: linguistic, totalFrames: linguistic.totalFrames)
        var silCount = 0
        var burstCount = 0
        var f = 0
        while f < linguistic.totalFrames {
            if silenceMask[f] { silCount += 1 }
            if stopBurstMask[f] { burstCount += 1 }
            f += 1
        }
        print("  Silence frames: \(silCount) / \(linguistic.totalFrames), Burst frames: \(burstCount)")
    }

    /// 後処理（マスキングやスケーリング）なしの素の TTS 合成テスト
    func testRawTTSSynthesis() throws {
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        var vocWeights: NeuralVocoderWeights? = nil
        if let vData = try? Data(contentsOf: URL(fileURLWithPath: "Models/vocoder_weights.json")) {
            vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }
        let vocoder = NeuralVocoder(weights: vocWeights)
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

        let sentences = [
            ("konnichiwa", "こんにちは"),
            ("tenki", "今日はいい天気です"),
            ("aiueo", "あいうえお")
        ]

        for (name, text) in sentences {
            engine.workspace.reset()
            vocoder.reset()

            let linguistic = engine.lengthRegulator.processText(
                text: text,
                normalizer: engine.normalizer,
                prosodyModel: engine.prosodyModel,
                vocabulary: engine.vocabulary,
                prosodyPredictor: engine.prosodyPredictor,
                speedFactor: 1.0,
                baseF0: VoiceProfile.female.baseF0,
                addBoundarySilence: true
            )
            let inputSeq = engine.encodeLinguisticFeatures(features: linguistic)
            let snnMel = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)

            let rawSamples = vocoder.synthesize(
                mel: snnMel,
                f0Contour: linguistic.f0Contour,
                voicedFlags: linguistic.voicedFlags
            )

            let wavData = WavEncoder.encode(samples: rawSamples)
            let outPath = ".tmp/wave15/raw_tts_\(name).wav"
            try wavData.write(to: URL(fileURLWithPath: outPath))
            print("[Raw TTS] \(name): \(outPath) (\(rawSamples.count) samples, \(Float(rawSamples.count)/16000.0)s)")
        }
    }

    /// 教師入力特徴量とTTS入力特徴量の比較分析
    func testCompareLinguisticFeatures() throws {
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else { return }

        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)
        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)
        let pitchTracker = PitchTracker()

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        let engine = SpikeSpeechEngine(weights: weights)

        let text = "水をマレーシアから買わなくてはならないのです。"
        guard let pair = engine.prepareTrainingPair(
            text: text,
            pcm16k: rawPCM,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker
        ) else { return }

        let ttsLinguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true
        )
        let ttsFeatures = engine.encodeLinguisticFeatures(features: ttsLinguistic)

        func printChannelStats(feats: [[Float]], label: String) {
            print("\n[\(label)] フレーム数: \(feats.count)")
            var ch = 64
            while ch <= 70 {
                var sumV: Float = 0.0
                var minV: Float = 999.0
                var maxV: Float = -999.0
                var t = 0
                while t < feats.count {
                    let v = feats[t][ch]
                    sumV += v
                    if v < minV { minV = v }
                    if maxV < v { maxV = v }
                    t += 1
                }
                let meanV = sumV / Float(max(1, feats.count))
                print("  ch[\(ch)]: 平均=\(String(format: "%.3f", meanV)), min=\(String(format: "%.3f", minV)), max=\(String(format: "%.3f", maxV))")
                ch += 1
            }
        }

        printChannelStats(feats: pair.features, label: "教師入力特徴量 (pair.features)")
        printChannelStats(feats: ttsFeatures, label: "TTS入力特徴量 (ttsFeatures)")

        print("\n[実音声 (pair) の音素 Duration 一覧]")
        var curP = -1
        var curDur = 0
        var pIdx = 0
        while pIdx < pair.features.count {
            var ph = -1
            var c = 0
            while c < 64 {
                if 1.0 < pair.features[pIdx][c] { ph = c }
                c += 1
            }
            if ph == curP {
                curDur += 1
            } else {
                if 0 <= curP {
                    let sym = engine.vocabulary.token(for: curP)
                    print("  phone: \(sym) (\(curP)) -> dur: \(curDur) frames (\(curDur * 10)ms)")
                }
                curP = ph
                curDur = 1
            }
            pIdx += 1
        }
        if 0 <= curP {
            let sym = engine.vocabulary.token(for: curP)
            print("  phone: \(sym) (\(curP)) -> dur: \(curDur) frames (\(curDur * 10)ms)")
        }
    }

    /// 滑らかなエネルギー輪郭（ベル型窓）を適用した TTS 合成の音質検証
    func testSmoothEnergyTTSSynthesis() throws {
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        var vocWeights: NeuralVocoderWeights? = nil
        if let vData = try? Data(contentsOf: URL(fileURLWithPath: "Models/vocoder_weights.json")) {
            vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }
        let vocoder = NeuralVocoder(weights: vocWeights)
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

        let sentences = [
            ("konnichiwa", "こんにちは"),
            ("tenki", "今日はいい天気です"),
            ("mizuwomare", "水をマレーシアから買わなくてはならないのです。")
        ]

        for (name, text) in sentences {
            engine.workspace.reset()
            vocoder.reset()

            let linguistic = engine.lengthRegulator.processText(
                text: text,
                normalizer: engine.normalizer,
                prosodyModel: engine.prosodyModel,
                vocabulary: engine.vocabulary,
                prosodyPredictor: engine.prosodyPredictor,
                speedFactor: 1.0,
                baseF0: VoiceProfile.female.baseF0,
                addBoundarySilence: true
            )

            // 音素ごとに中央ピークの滑らかなエネルギー曲線を構築
            var smoothEnergy = [Float](repeating: 0.0, count: linguistic.totalFrames)
            var curF = 0
            var p = 0
            while p < linguistic.phoneIds.count {
                let pid = Int(linguistic.phoneIds[p])
                let dur = Int(linguistic.durations[p])
                let peakEnergy: Float
                switch true {
                case engine.vocabulary.isPauseOrSilence(id: pid):
                    peakEnergy = 0.0
                case engine.vocabulary.isUnvoicedStop(id: pid):
                    peakEnergy = 0.02
                case engine.vocabulary.isUnvoicedFricative(id: pid):
                    peakEnergy = 0.18
                case engine.vocabulary.isAffricate(id: pid):
                    peakEnergy = 0.15
                case engine.vocabulary.isVoicedStop(id: pid):
                    peakEnergy = 0.25
                default:
                    let sym = engine.vocabulary.token(for: pid)
                    if engine.vocabulary.isVoiced(symbol: sym) {
                        switch sym {
                        case "a", "i", "u", "e", "o", "N", "_":
                            peakEnergy = 0.70
                        default:
                            peakEnergy = 0.35
                        }
                    } else {
                        peakEnergy = 0.10
                    }
                }

                var f = 0
                while f < dur {
                    let frameIdx = curF + f
                    if frameIdx < linguistic.totalFrames {
                        if peakEnergy <= 0.05 || dur <= 2 {
                            smoothEnergy[frameIdx] = peakEnergy
                        } else {
                            // 半正弦波窓: sin(pi * (f + 0.5) / dur)
                            let phase = (Float(f) + 0.5) / Float(dur)
                            let window = sinf(Float.pi * phase)
                            // 最低ベースライン（peakEnergy の 25%）+ 窓による起伏（75%）
                            smoothEnergy[frameIdx] = peakEnergy * (0.25 + (0.75 * window))
                        }
                    }
                    f += 1
                }
                curF += dur
                p += 1
            }

            let smoothLinguistic = LinguisticFeatures(
                phoneIds: linguistic.phoneIds,
                durations: linguistic.durations,
                f0Contour: linguistic.f0Contour,
                voicedFlags: linguistic.voicedFlags,
                energyContour: smoothEnergy,
                totalFrames: linguistic.totalFrames
            )

            let inputSeq = engine.encodeLinguisticFeatures(features: smoothLinguistic)
            let snnMel = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)

            let rawSamples = vocoder.synthesize(
                mel: snnMel,
                f0Contour: linguistic.f0Contour,
                voicedFlags: linguistic.voicedFlags
            )

            let wavData = WavEncoder.encode(samples: rawSamples)
            let outPath = ".tmp/wave15/smooth_tts_\(name).wav"
            try wavData.write(to: URL(fileURLWithPath: outPath))
            print("[Smooth TTS] \(name): \(outPath) (\(rawSamples.count) samples, \(Float(rawSamples.count)/16000.0)s)")
        }
    }

    /// SNN Mel / F0 / Duration アライメントの精密切り分け
    func testPreciseFactorDisentanglement() throws {
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else { return }

        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)
        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)
        let pitchTracker = PitchTracker()
        let pitchResult = pitchTracker.track(pcm: rawPCM)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        var vocWeights: NeuralVocoderWeights? = nil
        if let vData = try? Data(contentsOf: URL(fileURLWithPath: "Models/vocoder_weights.json")) {
            vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }
        let vocoder = NeuralVocoder(weights: vocWeights)
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

        let text = "水をマレーシアから買わなくてはならないのです。"
        guard let pair = engine.prepareTrainingPair(
            text: text,
            pcm16k: rawPCM,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker
        ) else { return }

        // ケース X: 完全アライメント SNN Mel (319 frames) + 予測 F0 (319 frames にリサンプル)
        engine.workspace.reset()
        let alignedSnnMel = engine.decoder.decodeSequence(featuresSeq: pair.features, workspace: engine.workspace)

        // 予測 F0 を 319 フレームにスケール
        let ttsLinguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true
        )

        // 319フレームの F0予測器出力を取得（speedFactor を 459/319 にして 319フレームで直接生成）
        let targetSpeed = Float(ttsLinguistic.totalFrames) / Float(alignedSnnMel.count)
        let matchedLinguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: targetSpeed,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true
        )

        print("[Disentangle] alignedSnnMel: \(alignedSnnMel.count) frames, matchedLinguistic: \(matchedLinguistic.totalFrames) frames")

        // X: aligned SNN Mel + matched Pred F0
        vocoder.reset()
        let countX = min(alignedSnnMel.count, matchedLinguistic.totalFrames)
        let samplesX = vocoder.synthesize(
            mel: Array(alignedSnnMel.prefix(countX)),
            f0Contour: Array(matchedLinguistic.f0Contour.prefix(countX)),
            voicedFlags: Array(matchedLinguistic.voicedFlags.prefix(countX))
        )
        try WavEncoder.encode(samples: samplesX).write(to: URL(fileURLWithPath: ".tmp/wave15/test_alignedMel_predF0.wav"))
        print("[Disentangle] 出力完了: .tmp/wave15/test_alignedMel_predF0.wav")
    }

    /// こんにちはのSNN Melの詳細検査
    func testInspectKonnichiwaMel() throws {
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        let engine = SpikeSpeechEngine(weights: weights)

        let linguistic = engine.lengthRegulator.processText(
            text: "こんにちは",
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true
        )
        let inputSeq = engine.encodeLinguisticFeatures(features: linguistic)
        let snnMel = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)

        print("\n[Konnichiwa Mel Inspection] totalFrames: \(snnMel.count)")
        var f = 0
        while f < snnMel.count {
            if f % 10 == 0 || f == snnMel.count - 1 {
                let mean = snnMel[f].reduce(0, +) / 64.0
                let minV = snnMel[f].min() ?? 0.0
                let maxV = snnMel[f].max() ?? 0.0
                let f0 = linguistic.f0Contour[f]
                let voiced = linguistic.voicedFlags[f]
                print("  frame[\(f)]: mean=\(String(format: "%.2f", mean)), min=\(String(format: "%.2f", minV)), max=\(String(format: "%.2f", maxV)), F0=\(String(format: "%.1f", f0)), voiced=\(String(format: "%.1f", voiced))")
            }
            f += 1
        }
    }

    /// Duration と F0 のどちらがロボット音の真犯人かを完全に特定する実験
    func testPinpointDifference() throws {
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else { return }

        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)
        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)
        let pitchTracker = PitchTracker()
        let pitchResult = pitchTracker.track(pcm: rawPCM)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        var vocWeights: NeuralVocoderWeights? = nil
        if let vData = try? Data(contentsOf: URL(fileURLWithPath: "Models/vocoder_weights.json")) {
            vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }
        let vocoder = NeuralVocoder(weights: vocWeights)
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

        let text = "水をマレーシアから買わなくてはならないのです。"

        // 実験 1: 319フレーム（実音声と同じ速度 1.44倍速）で、SNN も F0予測器も全て予測で合成
        engine.workspace.reset()
        vocoder.reset()
        let targetSpeed: Float = 459.0 / 319.0 // 1.4388
        let fastLinguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: targetSpeed,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true
        )
        let fastInputSeq = engine.encodeLinguisticFeatures(features: fastLinguistic)
        let fastSnnMel = engine.decoder.decodeSequence(featuresSeq: fastInputSeq, workspace: engine.workspace)
        let fastSamples = vocoder.synthesize(
            mel: fastSnnMel,
            f0Contour: fastLinguistic.f0Contour,
            voicedFlags: fastLinguistic.voicedFlags
        )
        try WavEncoder.encode(samples: fastSamples).write(to: URL(fileURLWithPath: ".tmp/wave15/test_fast_tts_319.wav"))
        print("[Pinpoint 1] test_fast_tts_319.wav 出力完了 (サンプル数: \(fastSamples.count), \(Float(fastSamples.count)/16000.0)s)")

        // 実験 2: 319フレームで、SNN Mel は純粋 TTS 予測 Mel、F0 だけ教師 F0
        vocoder.reset()
        let minC = min(fastSnnMel.count, pitchResult.frameCount)
        let samplesMelOnly = vocoder.synthesize(
            mel: Array(fastSnnMel.prefix(minC)),
            f0Contour: Array(pitchResult.f0.prefix(minC)),
            voicedFlags: Array(pitchResult.voiced.prefix(minC))
        )
        try WavEncoder.encode(samples: samplesMelOnly).write(to: URL(fileURLWithPath: ".tmp/wave15/test_fast_predMel_teacherF0.wav"))
        print("[Pinpoint 2] test_fast_predMel_teacherF0.wav 出力完了 (サンプル数: \(samplesMelOnly.count), \(Float(samplesMelOnly.count)/16000.0)s)")
    }

    /// pair.features (肉声) と fastInputSeq (ロボット) の完全 diff 検査
    func testDiffFeatures() throws {
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else { return }

        let wavReader = WavAudioReader()
        let rawPCM = try wavReader.loadWav16k(from: wavPath)
        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)
        let pitchTracker = PitchTracker()

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        let engine = SpikeSpeechEngine(weights: weights)

        let text = "水をマレーシアから買わなくてはならないのです。"
        guard let pair = engine.prepareTrainingPair(
            text: text,
            pcm16k: rawPCM,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker
        ) else { return }

        let targetSpeed: Float = 459.0 / 319.0
        let fastLinguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: targetSpeed,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true
        )
        let fastInputSeq = engine.encodeLinguisticFeatures(features: fastLinguistic)

        print("\n=======================================================")
        print("特徴量 完全 diff 検査: pair.features vs fastInputSeq")
        print("  pair.features frames: \(pair.features.count)")
        print("  fastInputSeq frames: \(fastInputSeq.count)")
        print("=======================================================")

        // 各音素の出現タイミングを比較
        // どのフレームでどの音素 (ch 0..63) が 3.0 になっているか
        var t = 0
        let minT = min(pair.features.count, fastInputSeq.count)
        while t < min(minT, 100) {
            var pPhone = -1
            var fPhone = -1
            var c = 0
            while c < 64 {
                if 1.0 < pair.features[t][c] { pPhone = c }
                if 1.0 < fastInputSeq[t][c] { fPhone = c }
                c += 1
            }
            if pPhone != fPhone || t % 10 == 0 {
                let pSym: String
                switch 0 <= pPhone {
                case true:
                    pSym = engine.vocabulary.token(for: pPhone)
                case false:
                    pSym = "none"
                }
                let fSym: String
                switch 0 <= fPhone {
                case true:
                    fSym = engine.vocabulary.token(for: fPhone)
                case false:
                    fSym = "none"
                }
                print("  t=\(t): pair=\(pSym)(\(pPhone)), fast=\(fSym)(\(fPhone)) | F0: pair=\(String(format: "%.1f", pair.features[t][66]*500)), fast=\(String(format: "%.1f", fastInputSeq[t][66]*500)) | voiced: pair=\(pair.features[t][64]), fast=\(fastInputSeq[t][64])")
            }

            t += 1
        }
    }

    /// 自然な実音声音素プロファイルによる TTS 合成検証
    func testNaturalDurationTTSSynthesis() throws {
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        var vocWeights: NeuralVocoderWeights? = nil
        if let vData = try? Data(contentsOf: URL(fileURLWithPath: "Models/vocoder_weights.json")) {
            vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }
        let vocoder = NeuralVocoder(weights: vocWeights)
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

        let sentences = [
            ("konnichiwa", "こんにちは"),
            ("tenki", "今日はいい天気です"),
            ("mizuwomare", "水をマレーシアから買わなくてはならないのです。")
        ]

        for (name, text) in sentences {
            engine.workspace.reset()
            vocoder.reset()

            let morphemes = engine.normalizer.normalize(text: text)
            let phrases = engine.prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: engine.vocabulary)

            var rawFloatDurations: [Float] = []
            for phrase in phrases {
                for mora in phrase.moras {
                    for phoneme in mora.phonemes {
                        let sym = phoneme.symbol
                        let cat = phoneme.category
                        let baseFrames: Float
                        switch cat {
                        case .vowel:
                            switch sym {
                            case "i", "u":
                                baseFrames = 6.0
                            default:
                                baseFrames = 7.5
                            }
                        case .consonant:
                            switch sym {
                            case "s", "sh", "h", "z", "j":
                                baseFrames = 4.5
                            case "k", "t", "p", "g", "d", "b":
                                baseFrames = 3.5
                            default:
                                baseFrames = 4.0
                            }
                        case .contracted:
                            baseFrames = 4.0
                        case .geminate:
                            baseFrames = 8.0
                        case .nasalSyllable:
                            baseFrames = 7.0
                        case .prolonged:
                            baseFrames = 7.5
                        case .pause:
                            baseFrames = 15.0
                        }
                        rawFloatDurations.append(baseFrames)
                    }
                }
                if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                    rawFloatDurations.append(12.0)
                }
            }

            let quantizedDurations = engine.lengthRegulator.quantizeDurations(durations: rawFloatDurations)

            var phoneIds: [Int32] = []
            var durations: [Int32] = []
            var qIdx = 0
            for phrase in phrases {
                for mora in phrase.moras {
                    for phoneme in mora.phonemes {
                        phoneIds.append(Int32(phoneme.id))
                        let d: Int
                        switch qIdx < quantizedDurations.count {
                        case true:
                            d = quantizedDurations[qIdx]
                        case false:
                            d = 6
                        }
                        durations.append(Int32(d))
                        qIdx += 1
                    }
                }
                if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                    phoneIds.append(Int32(PhonemeVocabulary.pauId))
                    let d: Int
                    switch qIdx < quantizedDurations.count {
                    case true:
                        d = quantizedDurations[qIdx]
                    case false:
                        d = 12
                    }
                    durations.append(Int32(d))
                    qIdx += 1
                }
            }

            let f0Result = engine.prosodyPredictor.predictF0Contour(
                phrases: phrases,
                vocabulary: engine.vocabulary,
                baseF0: VoiceProfile.female.baseF0,
                prosodyModel: engine.prosodyModel,
                durations: quantizedDurations,
                applyFluctuation: false
            )

            let leadSil = 20
            let trailSil = 25

            var fullPhoneIds: [Int32] = [Int32(PhonemeVocabulary.silId)]
            fullPhoneIds.append(contentsOf: phoneIds)
            fullPhoneIds.append(Int32(PhonemeVocabulary.silId))

            var fullDurations: [Int32] = [Int32(leadSil)]
            fullDurations.append(contentsOf: durations)
            fullDurations.append(Int32(trailSil))

            let totalFrames = f0Result.totalFrames + leadSil + trailSil

            var fullF0 = [Float](repeating: 0.0, count: leadSil)
            fullF0.append(contentsOf: f0Result.f0Contour)
            fullF0.append(contentsOf: [Float](repeating: 0.0, count: trailSil))

            var fullVoiced = [Float](repeating: 0.0, count: leadSil)
            fullVoiced.append(contentsOf: f0Result.voicedFlags)
            fullVoiced.append(contentsOf: [Float](repeating: 0.0, count: trailSil))

            var fullEnergy = [Float](repeating: 0.0, count: totalFrames)
            var curF = leadSil
            var p = 0
            while p < phoneIds.count {
                let pid = Int(phoneIds[p])
                let dur = Int(durations[p])
                let peakEnergy: Float
                switch true {
                case engine.vocabulary.isPauseOrSilence(id: pid):
                    peakEnergy = 0.0
                case engine.vocabulary.isUnvoicedStop(id: pid):
                    peakEnergy = 0.02
                case engine.vocabulary.isUnvoicedFricative(id: pid):
                    peakEnergy = 0.18
                case engine.vocabulary.isAffricate(id: pid):
                    peakEnergy = 0.15
                case engine.vocabulary.isVoicedStop(id: pid):
                    peakEnergy = 0.25
                default:
                    let sym = engine.vocabulary.token(for: pid)
                    if engine.vocabulary.isVoiced(symbol: sym) {
                        switch sym {
                        case "a", "i", "u", "e", "o", "N", "_":
                            peakEnergy = 0.70
                        default:
                            peakEnergy = 0.35
                        }
                    } else {
                        peakEnergy = 0.10
                    }
                }

                var f = 0
                while f < dur {
                    let frameIdx = curF + f
                    if frameIdx < totalFrames {
                        switch (peakEnergy <= 0.05, dur <= 2) {
                        case (true, _), (_, true):
                            fullEnergy[frameIdx] = peakEnergy
                        default:
                            let phase = (Float(f) + 0.5) / Float(dur)
                            let window = sinf(Float.pi * phase)
                            fullEnergy[frameIdx] = peakEnergy * (0.25 + (0.75 * window))
                        }
                    }
                    f += 1
                }
                curF += dur
                p += 1
            }

            let naturalLinguistic = LinguisticFeatures(
                phoneIds: fullPhoneIds,
                durations: fullDurations,
                f0Contour: fullF0,
                voicedFlags: fullVoiced,
                energyContour: fullEnergy,
                totalFrames: totalFrames
            )

            let inputSeq = engine.encodeLinguisticFeatures(features: naturalLinguistic)
            let snnMel = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)

            let rawSamples = vocoder.synthesize(
                mel: snnMel,
                f0Contour: naturalLinguistic.f0Contour,
                voicedFlags: naturalLinguistic.voicedFlags
            )

            let wavData = WavEncoder.encode(samples: rawSamples)
            let outPath = ".tmp/wave15/natural_tts_\(name).wav"
            try wavData.write(to: URL(fileURLWithPath: outPath))
            print("[Natural TTS] \(name): \(outPath) (\(rawSamples.count) samples, \(Float(rawSamples.count)/16000.0)s)")
        }
    }

    /// 音響試聴監査用: 全対象音声の物理・音響特徴（サンプル長、有声率、F0分布、ZCR、フォルマント動態）を詳細計測
    func testAcousticAudit() throws {
        let files = [
            (".tmp/wave15/copy_BASIC5000_0001.wav", "Copy-synth (教師基準)"),
            (".tmp/wave15/ablate_copy.wav", "Ablation 1: 教師Mel + 教師F0"),
            (".tmp/wave15/ablate_teacherMel_predF0.wav", "Ablation 2: 教師Mel + 予測F0"),
            (".tmp/wave15/ablate_predMel_teacherF0.wav", "Ablation 3: 予測Mel + 教師F0"),
            (".tmp/wave15/ablate_tts.wav", "Ablation 4: 予測Mel + 予測F0"),
            (".tmp/wave15/tts_konnichiwa.wav", "TTS: こんにちは"),
            (".tmp/wave15/tts_tenki.wav", "TTS: 今日はいい天気です"),
            (".tmp/wave15/tts_aiueo.wav", "TTS: あいうえお"),
            (".tmp/wave15/tts_mizuwomare.wav", "TTS: 水をマレーシアから買わなくてはならないのです")
        ]

        let reader = WavAudioReader()
        let tracker = PitchTracker()

        var fIdx = 0
        while fIdx < files.count {
            let (path, label) = files[fIdx]
            guard let pcm = try? reader.loadWav16k(from: path) else {
                print("[\(label)] 読込失敗: \(path)")
                fIdx += 1
                continue
            }
            let pitch = tracker.track(pcm: pcm)
            var voicedF0: [Float] = []
            var p = 0
            while p < pitch.f0.count {
                if 0.5 <= pitch.voiced[p] {
                    voicedF0.append(pitch.f0[p])
                }
                p += 1
            }

            var sumF0: Float = 0.0
            var minF0: Float = 1000.0
            var maxF0: Float = 0.0
            var vIdx = 0
            while vIdx < voicedF0.count {
                let f = voicedF0[vIdx]
                sumF0 += f
                if f < minF0 { minF0 = f }
                if maxF0 < f { maxF0 = f }
                vIdx += 1
            }
            var meanF0: Float = 0.0
            if 0 < voicedF0.count {
                meanF0 = sumF0 / Float(voicedF0.count)
            } else {
                minF0 = 0.0
            }

            let dur = Float(pcm.count) / 16000.0

            // ゼロ交差率 (ZCR)
            var zcrCount = 0
            var s = 1
            while s < pcm.count {
                let curr = pcm[s]
                let prev = pcm[s - 1]
                switch (0.0 <= curr && prev < 0.0, curr < 0.0 && 0.0 <= prev) {
                case (true, _), (_, true):
                    zcrCount += 1
                default:
                    break
                }
                s += 1
            }
            let zcr = Float(zcrCount) / Float(max(1, pcm.count))

            print("AUDIT_RESULT: label=\(label) | path=\(path) | samples=\(pcm.count) | dur=\(String(format: "%.2f", dur))s | voicedRatio=\(String(format: "%.1f", Float(voicedF0.count) / Float(max(1, pitch.f0.count)) * 100.0))% | meanF0=\(String(format: "%.1f", meanF0))Hz (min=\(String(format: "%.1f", minF0)), max=\(String(format: "%.1f", maxF0)), delta=\(String(format: "%.1f", maxF0 - minF0))Hz) | ZCR=\(String(format: "%.4f", zcr))")

            fIdx += 1
        }
    }
}











