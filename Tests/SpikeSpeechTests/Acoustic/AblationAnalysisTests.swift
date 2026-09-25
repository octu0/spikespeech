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
            (".tmp/wave15/recon_BASIC5000_0001.wav", ".tmp/wave15_spec/recon_BASIC5000_0001.png"),
            (".tmp/wave15/tts_konnichiwa.wav", ".tmp/wave15_spec/tts_konnichiwa.png"),
            (".tmp/wave15/tts_mizuwomare.wav", ".tmp/wave15_spec/tts_mizuwomare.png")
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

        // 各音素の最初のフレームにおける特徴量ベクトル (ch 0..127) を比較
        var pIdx = 0
        var curPPhone = -1
        while pIdx < pair.features.count {
            var ph = -1
            var c = 0
            while c < 64 {
                if 1.0 < pair.features[pIdx][c] { ph = c }
                c += 1
            }
            if ph != curPPhone && 0 <= ph {
                curPPhone = ph
                let pSym = engine.vocabulary.token(for: ph)

                // fastInputSeq で同じ音素のフレームを探す
                var fIdx = 0
                var foundF = -1
                while fIdx < fastInputSeq.count {
                    if 1.0 < fastInputSeq[fIdx][ph] {
                        foundF = fIdx
                        break
                    }
                    fIdx += 1
                }

                if 0 <= foundF {
                    print("\n[Phone \(pSym) (id=\(ph))] pair frame \(pIdx) vs fast frame \(foundF)")
                    var diffChannels: [String] = []
                    var ch = 0
                    while ch < 128 {
                        let pVal = pair.features[pIdx][ch]
                        let fVal = fastInputSeq[foundF][ch]
                        let diff = abs(pVal - fVal)
                        if 0.05 < diff {
                            diffChannels.append("ch\(ch): pair=\(String(format: "%.3f", pVal)) vs fast=\(String(format: "%.3f", fVal))")
                        }
                        ch += 1
                    }
                    print("  差分チャンネル数: \(diffChannels.count) / 128")
                    for d in diffChannels.prefix(15) {
                        print("    \(d)")
                    }
                }
            }
            pIdx += 1
        }
    }

    /// ブザー音の真因を特定する 3 条件ピンポイント比較実験
    func testPinpointCauseOfBuzzer() throws {
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

        // 条件 1: pair.features の ch64..70（韻律・エネルギー）を、音素定義に基づく推論時ルール値に差し替えた特徴量
        var featCond1 = pair.features
        var t = 0
        while t < featCond1.count {
            var pid = -1
            var c = 0
            while c < 64 {
                if 1.0 < featCond1[t][c] { pid = c }
                c += 1
            }
            let isVowel = engine.vocabulary.isVoiced(symbol: engine.vocabulary.token(for: pid))
            if isVowel {
                featCond1[t][64] = 1.0 // voiced
                featCond1[t][65] = 0.0 // unvoiced
                featCond1[t][70] = 0.70 // energy
            } else {
                featCond1[t][64] = 0.0
                featCond1[t][65] = 1.0
                featCond1[t][70] = 0.10
            }
            t += 1
        }

        engine.workspace.reset()
        vocoder.reset()
        let mel1 = engine.decoder.decodeSequence(featuresSeq: featCond1, workspace: engine.workspace)
        let wav1 = vocoder.synthesize(mel: mel1, f0Contour: pitchResult.f0, voicedFlags: pitchResult.voiced)
        try WavEncoder.encode(samples: wav1).write(to: URL(fileURLWithPath: ".tmp/wave15/test_pinpoint_cond1_ruleProsody.wav"))
        print("[Pinpoint Cond 1] ruleProsody 出力完了")

        // 条件 2: 音素 duration は推論時の Duration だが、エネルギーは実測の平均に合わせたもの
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

        engine.workspace.reset()
        vocoder.reset()
        let mel2 = engine.decoder.decodeSequence(featuresSeq: ttsFeatures, workspace: engine.workspace)
        let wav2 = vocoder.synthesize(mel: mel2, f0Contour: ttsLinguistic.f0Contour, voicedFlags: ttsLinguistic.voicedFlags)
        try WavEncoder.encode(samples: wav2).write(to: URL(fileURLWithPath: ".tmp/wave15/test_pinpoint_cond2_ttsStandard.wav"))
        print("[Pinpoint Cond 2] ttsStandard 出力完了")

        // 条件 3: 実音声の音素 duration 列をそのまま推論エンジンに与え、推論 F0/voiced/energy で合成
        let alignedDurs: [Int32] = [27, 3, 3, 2, 3, 3, 11, 13, 2, 12, 4, 2, 9, 3, 13, 10, 16, 5, 14, 3, 2, 10, 7, 4, 14, 3, 14, 3, 3, 4, 7, 3, 3, 3, 2, 3, 3, 6, 3, 13, 3, 2, 3, 43]
        var cond3PhoneIds: [Int32] = [Int32(PhonemeVocabulary.silId)]
        var p = 0
        while p < ttsLinguistic.phoneIds.count {
            let pid = ttsLinguistic.phoneIds[p]
            if pid != PhonemeVocabulary.silId && pid != PhonemeVocabulary.pauId {
                cond3PhoneIds.append(pid)
            }
            p += 1
        }
        cond3PhoneIds.append(Int32(PhonemeVocabulary.silId))

        if cond3PhoneIds.count == alignedDurs.count {
            var cond3TotalFrames = 0
            var dIdx = 0
            while dIdx < alignedDurs.count {
                cond3TotalFrames += Int(alignedDurs[dIdx])
                dIdx += 1
            }

            // 推論 F0 を cond3TotalFrames にリサンプル
            var cond3F0 = [Float](repeating: 0.0, count: cond3TotalFrames)
            var cond3Voiced = [Float](repeating: 0.0, count: cond3TotalFrames)
            var cond3Energy = [Float](repeating: 0.0, count: cond3TotalFrames)

            var curF = 0
            var ph = 0
            while ph < cond3PhoneIds.count {
                let pid = Int(cond3PhoneIds[ph])
                let dur = Int(alignedDurs[ph])
                let isVowel = engine.vocabulary.isVoiced(symbol: engine.vocabulary.token(for: pid))
                let peakE: Float
                let vFlag: Float
                switch isVowel {
                case true:
                    peakE = 0.70
                    vFlag = 1.0
                case false:
                    peakE = 0.10
                    vFlag = 0.0
                }

                var f = 0
                while f < dur {
                    let frameIdx = curF + f
                    if frameIdx < cond3TotalFrames {
                        cond3Voiced[frameIdx] = vFlag
                        cond3Energy[frameIdx] = peakE
                        // F0 はピッチ予測器から
                        let ratio = Float(frameIdx) / Float(max(1, cond3TotalFrames))
                        let srcF = Int(ratio * Float(ttsLinguistic.f0Contour.count))
                        if srcF < ttsLinguistic.f0Contour.count {
                            cond3F0[frameIdx] = ttsLinguistic.f0Contour[srcF]
                        }
                    }
                    f += 1
                }
                curF += dur
                ph += 1
            }

            let cond3Linguistic = LinguisticFeatures(
                phoneIds: cond3PhoneIds,
                durations: alignedDurs,
                f0Contour: cond3F0,
                voicedFlags: cond3Voiced,
                energyContour: cond3Energy,
                totalFrames: cond3TotalFrames
            )
            let cond3Features = engine.encodeLinguisticFeatures(features: cond3Linguistic)

            engine.workspace.reset()
            vocoder.reset()
            let mel3 = engine.decoder.decodeSequence(featuresSeq: cond3Features, workspace: engine.workspace)
            let wav3 = vocoder.synthesize(mel: mel3, f0Contour: cond3F0, voicedFlags: cond3Voiced)
            try WavEncoder.encode(samples: wav3).write(to: URL(fileURLWithPath: ".tmp/wave15/test_pinpoint_cond3_alignedDurs_predProsody.wav"))
            print("[Pinpoint Cond 3] alignedDurs + predProsody 出力完了 (\(wav3.count) samples)")
        }

        // 自然なモーラ比率での「こんにちは」合成実験
        let konDurs: [Int32] = [6, 4, 12, 12, 4, 11, 5, 11, 4, 14, 8]
        let konPhones: [Int32] = [
            Int32(PhonemeVocabulary.silId),
            10, // k
            9,  // o
            24, // N
            13, // n
            6,  // i
            28, // ch
            6,  // i
            18, // w
            5,  // a
            Int32(PhonemeVocabulary.silId)
        ]
        let konTotal = konDurs.reduce(0, +)
        var konF0 = [Float](repeating: 0.0, count: Int(konTotal))
        var konVoiced = [Float](repeating: 0.0, count: Int(konTotal))
        var konEnergy = [Float](repeating: 0.0, count: Int(konTotal))

        // F0 は藤崎モデルまたは基本ピッチ 220Hz
        var kCurF = 0
        var kP = 0
        while kP < konPhones.count {
            let pid = Int(konPhones[kP])
            let dur = Int(konDurs[kP])
            let isV = engine.vocabulary.isVoiced(symbol: engine.vocabulary.token(for: pid))
            let pE: Float
            let vF: Float
            switch isV {
            case true:
                pE = 0.70
                vF = 1.0
            case false:
                pE = 0.10
                vF = 0.0
            }
            var f = 0
            while f < dur {
                let fIdx = kCurF + f
                if fIdx < Int(konTotal) {
                    konVoiced[fIdx] = vF
                    konEnergy[fIdx] = pE
                    if 0.5 <= vF {
                        // 自然な「こ(低)ん(高)に(高)ち(低)は(低)」アクセント
                        let prog = Float(fIdx) / Float(konTotal)
                        if prog < 0.25 {
                            konF0[fIdx] = 210.0
                        } else {
                            switch prog < 0.65 {
                            case true:
                                konF0[fIdx] = 245.0
                            case false:
                                konF0[fIdx] = 205.0
                            }
                        }
                    }
                }
                f += 1
            }
            kCurF += dur
            kP += 1
        }

        let konLing = LinguisticFeatures(
            phoneIds: konPhones,
            durations: konDurs,
            f0Contour: konF0,
            voicedFlags: konVoiced,
            energyContour: konEnergy,
            totalFrames: Int(konTotal)
        )
        let konFeat = engine.encodeLinguisticFeatures(features: konLing)
        engine.workspace.reset()
        vocoder.reset()
        let konMel = engine.decoder.decodeSequence(featuresSeq: konFeat, workspace: engine.workspace)
        let konWav = vocoder.synthesize(mel: konMel, f0Contour: konF0, voicedFlags: konVoiced)
        try WavEncoder.encode(samples: konWav).write(to: URL(fileURLWithPath: ".tmp/wave15/test_natural_ratio_konnichiwa.wav"))
        print("[Natural Ratio Konnichiwa] 出力完了 (\(konWav.count) samples, \(Float(konWav.count)/16000.0)s)")

        // 自然なモーラ比率での「今日はいい天気です」合成実験
        // きょう(ky,o,_) は(w,a) いい(i,_) てんき(t,e,N,k,i) です(d,e,s,u)
        let tenkiDurs: [Int32] = [
            6,  // sil
            4, 8, 10, // ky, o, _
            4, 11,    // w, a
            9, 10,    // i, _
            4, 10, 11, 4, 11, // t, e, N, k, i
            4, 11, 4, 9,      // d, e, s, u
            8   // sil
        ]
        let tenkiPhones: [Int32] = [
            Int32(PhonemeVocabulary.silId),
            30, 9, 26,  // ky, o, _
            18, 5,      // w, a
            6, 26,      // i, _
            12, 8, 24, 10, 6, // t, e, N, k, i
            21, 8, 11, 7,     // d, e, s, u
            Int32(PhonemeVocabulary.silId)
        ]
        let tenkiTotal = tenkiDurs.reduce(0, +)
        var tenkiF0 = [Float](repeating: 0.0, count: Int(tenkiTotal))
        var tenkiVoiced = [Float](repeating: 0.0, count: Int(tenkiTotal))
        var tenkiEnergy = [Float](repeating: 0.0, count: Int(tenkiTotal))

        var tCurF = 0
        var tP = 0
        while tP < tenkiPhones.count {
            let pid = Int(tenkiPhones[tP])
            let dur = Int(tenkiDurs[tP])
            let isV = engine.vocabulary.isVoiced(symbol: engine.vocabulary.token(for: pid))
            let pE: Float
            let vF: Float
            switch isV {
            case true:
                pE = 0.70
                vF = 1.0
            case false:
                pE = 0.10
                vF = 0.0
            }
            var f = 0
            while f < dur {
                let fIdx = tCurF + f
                if fIdx < Int(tenkiTotal) {
                    tenkiVoiced[fIdx] = vF
                    tenkiEnergy[fIdx] = pE
                    if 0.5 <= vF {
                        let prog = Float(fIdx) / Float(tenkiTotal)
                        if prog < 0.20 {
                            tenkiF0[fIdx] = 230.0 // きょう
                        } else {
                            switch prog < 0.35 {
                            case true:
                                tenkiF0[fIdx] = 210.0 // は
                            case false:
                                switch prog < 0.60 {
                                case true:
                                    tenkiF0[fIdx] = 245.0 // いい
                                case false:
                                    switch prog < 0.85 {
                                    case true:
                                        tenkiF0[fIdx] = 235.0 // てんき
                                    case false:
                                        tenkiF0[fIdx] = 200.0 // です
                                    }
                                }
                            }
                        }
                    }
                }
                f += 1
            }
            tCurF += dur
            tP += 1
        }

        let tenkiLing = LinguisticFeatures(
            phoneIds: tenkiPhones,
            durations: tenkiDurs,
            f0Contour: tenkiF0,
            voicedFlags: tenkiVoiced,
            energyContour: tenkiEnergy,
            totalFrames: Int(tenkiTotal)
        )
        let tenkiFeat = engine.encodeLinguisticFeatures(features: tenkiLing)
        engine.workspace.reset()
        vocoder.reset()
        let tenkiMel = engine.decoder.decodeSequence(featuresSeq: tenkiFeat, workspace: engine.workspace)
        let tenkiWav = vocoder.synthesize(mel: tenkiMel, f0Contour: tenkiF0, voicedFlags: tenkiVoiced)
        try WavEncoder.encode(samples: tenkiWav).write(to: URL(fileURLWithPath: ".tmp/wave15/test_natural_ratio_tenki.wav"))
        print("[Natural Ratio Tenki] 出力完了 (\(tenkiWav.count) samples, \(Float(tenkiWav.count)/16000.0)s)")
    }

    /// Models/weights.json 内の音素平均フレームテーブルを自然な音素比率プロファイルに更新して永続化する
    func testUpdateWeightsWithHealthyDurations() throws {
        let weightsURL = URL(fileURLWithPath: "Models/weights.json")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else { return }

        let data = try Data(contentsOf: weightsURL)
        let loaded = try JSONDecoder().decode(SpikingNetworkWeights.self, from: data)

        // 健全な実測音素プロファイル
        let updated = loaded.withPhonemeAverageDurations(LengthRegulator.defaultPhonemeAverageDurations)
        try WeightCheckpoint.atomicWritePretty(updated, to: weightsURL)
        print("[Weights Updated] Models/weights.json に自然な音素平均フレームテーブルを永続化しました。")

        // エンジン経由で標準 synthesize を実行し、.tmp/wave15/ に診断音声を出力
        let engine = SpikeSpeechEngine(weights: updated)

        let konWavData = engine.synthesizeWav(text: "こんにちは")
        try konWavData.write(to: URL(fileURLWithPath: ".tmp/wave15/tts_konnichiwa.wav"))
        print("[TTS Output] tts_konnichiwa.wav 保存完了 (\(konWavData.count) bytes)")

        let tenkiWavData = engine.synthesizeWav(text: "今日はいい天気です")
        try tenkiWavData.write(to: URL(fileURLWithPath: ".tmp/wave15/tts_tenki.wav"))
        print("[TTS Output] tts_tenki.wav 保存完了 (\(tenkiWavData.count) bytes)")

        let mizuWavData = engine.synthesizeWav(text: "水をマレーシアから買わなくてはならないのです。")
        try mizuWavData.write(to: URL(fileURLWithPath: ".tmp/wave15/tts_mizuwomare.wav"))
        print("[TTS Output] tts_mizuwomare.wav 保存完了 (\(mizuWavData.count) bytes)")
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
            (".tmp/wave15/recon_BASIC5000_0001.wav", "Recon (SNN予測Mel + ボコーダ)"),
            (".tmp/wave15/tts_konnichiwa.wav", "TTS: こんにちは"),
            (".tmp/wave15/tts_tenki.wav", "TTS: 今日はいい天気です"),
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

        // Copy-synth と Recon の Mel 比較
        let melExtractor = MelSpectrogramExtractor(
            sampleRate: Float(AudioConfig.sampleRate),
            melChannels: AudioConfig.melChannels
        )
        if let copyPCM = try? reader.loadWav16k(from: ".tmp/wave15/copy_BASIC5000_0001.wav"),
           let reconPCM = try? reader.loadWav16k(from: ".tmp/wave15/recon_BASIC5000_0001.wav") {
            let copyMel = melExtractor.extractLogMel(pcm: copyPCM)
            let reconMel = melExtractor.extractLogMel(pcm: reconPCM)
            let frames = min(copyMel.count, reconMel.count)
            var totalL1: Float = 0.0
            var totalMSE: Float = 0.0
            var lowBandL1: Float = 0.0
            var midBandL1: Float = 0.0
            var highBandL1: Float = 0.0
            var t = 0
            while t < frames {
                var c = 0
                while c < AudioConfig.melChannels {
                    let diff = abs(copyMel[t][c] - reconMel[t][c])
                    totalL1 += diff
                    totalMSE += diff * diff
                    switch c {
                    case 0..<16:
                        lowBandL1 += diff
                    case 16..<40:
                        midBandL1 += diff
                    default:
                        highBandL1 += diff
                    }
                    c += 1
                }
                t += 1
            }
            let totalElements = Float(frames * AudioConfig.melChannels)
            let meanL1 = totalL1 / totalElements
            let meanMSE = totalMSE / totalElements
            let meanLowL1 = lowBandL1 / Float(frames * 16)
            let meanMidL1 = midBandL1 / Float(frames * 24)
            let meanHighL1 = highBandL1 / Float(frames * 24)
            print("MEL_COMPARE: frames=\(frames) | meanL1=\(String(format: "%.4f", meanL1)) | RMSE=\(String(format: "%.4f", sqrtf(meanMSE))) | lowL1(0-15)=\(String(format: "%.4f", meanLowL1)) | midL1(16-39)=\(String(format: "%.4f", meanMidL1)) | highL1(40-63)=\(String(format: "%.4f", meanHighL1))")
        }
    }

    /// コーパス実音声から音素別アライメント平均フレーム数を精密集計
    func testCalculateCorpusPhonemeAverages() throws {
        let corpusDir = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000"
        let corpusMasCachePath = "\(corpusDir)/mas_alignments.json"
        let vocabulary = PhonemeVocabulary()

        if FileManager.default.fileExists(atPath: corpusMasCachePath) {
            let alignDict = try AlignmentStore.load(from: corpusMasCachePath)
            let alignList = Array(alignDict.values)
            let averages = AlignmentStore.computeAverageDurations(from: alignList)
            print("\n=======================================================")
            print("コーパス実測音素平均フレーム数 (キャッシュ読み込み: \(alignList.count) 発話)")
            print("=======================================================")
            for (pid, avg) in averages.sorted(by: { $0.key < $1.key }) {
                let sym = vocabulary.token(for: Int(pid))
                print("  ID \(pid) (\(sym)): 平均=\(String(format: "%.2f", avg)) frames (\(String(format: "%.1f", avg * 10.0))ms)")
            }
            return
        }

        let wavDir = "\(corpusDir)/wav"
        let transcriptPath = "\(corpusDir)/transcript_utf8.txt"
        guard FileManager.default.fileExists(atPath: transcriptPath) else { return }

        let content = try String(contentsOfFile: transcriptPath, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)

        let wavReader = WavAudioReader()
        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)
        let pitchTracker = PitchTracker()
        let masAligner = MonotonicAlignmentSearch()

        let normalizer = TextNormalizer(morphology: ViterbiMorphology())
        let prosodyModel = ProsodyModel()

        var durationSums: [Int32: Float] = [:]
        var durationCounts: [Int32: Float] = [:]
        var totalSpeechFrames = 0
        var totalMoras = 0

        var processedCount = 0
        var lIdx = 0
        while lIdx < lines.count && processedCount < 50 {
            let line = lines[lIdx].trimmingCharacters(in: .whitespacesAndNewlines)
            lIdx += 1
            if line.isEmpty { continue }

            let parts = line.components(separatedBy: ":")
            if parts.count < 2 { continue }
            let id = parts[0]
            let text = parts[1]

            let wavPath = "\(wavDir)/\(id).wav"
            guard FileManager.default.fileExists(atPath: wavPath),
                  let pcm16k = try? wavReader.loadWav16k(from: wavPath) else {
                continue
            }

            let morphemes = normalizer.normalize(text: text)
            let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
            var tokens: [PhonemeToken] = []
            for phrase in phrases {
                for mora in phrase.moras {
                    for ph in mora.phonemes {
                        tokens.append(ph)
                    }
                }
            }
            if tokens.isEmpty { continue }

            let hopSize = AudioConfig.hopSize
            let totalFrames = max(1, pcm16k.count / hopSize)
            let boundaries = SpikeSpeechEngine.detectSpeechBoundaries(
                pcm: pcm16k,
                hopSize: hopSize,
                totalFrames: totalFrames
            )
            if boundaries.speechFrames <= 0 || boundaries.speechFrames < tokens.count {
                continue
            }

            let targetMel = melExtractor.extractLogMel(pcm: pcm16k)
            let pitchRes = pitchTracker.track(pcm: pcm16k)
            let speechEnd = min(targetMel.count, boundaries.leadSilence + boundaries.speechFrames)
            var speechMel: [[Float]] = []
            var speechVoiced: [Float] = []
            var sf = boundaries.leadSilence
            while sf < speechEnd {
                speechMel.append(targetMel[sf])
                var v: Float = 0.0
                if sf < pitchRes.frameCount {
                    v = pitchRes.voiced[sf]
                }
                speechVoiced.append(v)
                sf += 1
            }

            guard let durs = masAligner.align(
                mel: speechMel,
                voiced: speechVoiced,
                phonemes: tokens,
                meanFramesPerMora: 16.0
            ) else {
                continue
            }

            totalSpeechFrames += boundaries.speechFrames
            var phraseMoras = 0
            for phrase in phrases {
                phraseMoras += phrase.moras.count
            }
            totalMoras += phraseMoras

            var t = 0
            while t < tokens.count {
                let pid = Int32(tokens[t].id)
                let dur = Float(durs[t])
                let curS = durationSums[pid] ?? 0.0
                let curC = durationCounts[pid] ?? 0.0
                durationSums[pid] = curS + dur
                durationCounts[pid] = curC + 1.0
                t += 1
            }

            processedCount += 1
        }

        let meanFramesPerMora = Float(totalSpeechFrames) / Float(max(1, totalMoras))
        print("\n=======================================================")
        print("コーパス実測音素平均フレーム数 (集計サンプル数: \(processedCount))")
        print("  合計発話フレーム: \(totalSpeechFrames), 合計モーラ数: \(totalMoras)")
        print("  meanFramesPerMora 実測平均: \(String(format: "%.2f", meanFramesPerMora)) frames (\(String(format: "%.1f", meanFramesPerMora * 10.0)) ms/モーラ)")
        print("=======================================================")
        var averages: [Int32: Float] = [:]
        for (pid, sum) in durationSums.sorted(by: { $0.key < $1.key }) {
            let count = durationCounts[pid] ?? 1.0
            let avg = sum / count
            let sym = vocabulary.token(for: Int(pid))
            print("  ID \(pid) (\(sym)): 平均=\(String(format: "%.2f", avg)) frames (\(String(format: "%.1f", avg * 10.0))ms, N=\(Int(count)))")
            averages[pid] = roundf(avg * 10.0) / 10.0
        }
    }

    /// 全 5,000 発話の MAS アライメントを集計し、mas_alignments.json および Models/weights.json を更新する
    func testExtractFullCorpusMASAndExportDurations() throws {
        let corpusDir = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000"
        let corpusMasCachePath = "\(corpusDir)/mas_alignments.json"
        let wavDir = "\(corpusDir)/wav"
        let transcriptPath = "\(corpusDir)/transcript_utf8.txt"
        guard FileManager.default.fileExists(atPath: transcriptPath) else {
            XCTFail("transcript_utf8.txt が存在しません")
            return
        }

        let content = try String(contentsOfFile: transcriptPath, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)

        let wavReader = WavAudioReader()
        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: AudioConfig.melChannels)
        let pitchTracker = PitchTracker()
        let normalizer = TextNormalizer(morphology: ViterbiMorphology())
        let vocabulary = PhonemeVocabulary()
        let prosodyModel = ProsodyModel()

        var inputItems: [MonotonicAlignmentSearch.AlignmentInputItem] = []
        var lIdx = 0
        while lIdx < lines.count {
            let line = lines[lIdx].trimmingCharacters(in: .whitespacesAndNewlines)
            lIdx += 1
            if line.isEmpty { continue }

            var parts = line.split(separator: ":", maxSplits: 1).map { String($0) }
            if parts.count != 2 {
                parts = line.split(separator: "\t", maxSplits: 1).map { String($0) }
            }
            if parts.count != 2 { continue }

            let id = parts[0].trimmingCharacters(in: .whitespaces)
            let text = parts[1].trimmingCharacters(in: .whitespaces)

            var wavFile = "\(wavDir)/\(id).wav"
            if FileManager.default.fileExists(atPath: wavFile) != true {
                let uppercaseWav = "\(wavDir)/\(id).WAV"
                if FileManager.default.fileExists(atPath: uppercaseWav) {
                    wavFile = uppercaseWav
                }
            }
            guard FileManager.default.fileExists(atPath: wavFile),
                  let rawPCM = try? wavReader.loadWav16k(from: wavFile) else {
                continue
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

            let morphemes = normalizer.normalize(text: text)
            let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
            var tokens: [PhonemeToken] = []
            for phrase in phrases {
                for mora in phrase.moras {
                    for ph in mora.phonemes {
                        tokens.append(ph)
                    }
                }
            }
            if tokens.isEmpty { continue }

            let hopSize = AudioConfig.hopSize
            let targetMel = melExtractor.extractLogMel(pcm: pcm16k)
            let totalFrames = max(1, targetMel.count)
            let boundaries = SpikeSpeechEngine.detectSpeechBoundaries(
                pcm: pcm16k,
                hopSize: hopSize,
                totalFrames: totalFrames
            )
            let speechFrames = boundaries.speechFrames
            let leadSilence = boundaries.leadSilence
            let trailSilence = boundaries.trailSilence

            if speechFrames < tokens.count || speechFrames <= 0 { continue }

            let pitchRes = pitchTracker.track(pcm: pcm16k)
            let speechEnd = min(targetMel.count, leadSilence + speechFrames)
            var speechMel: [[Float]] = []
            var speechVoiced: [Float] = []
            var sf = leadSilence
            while sf < speechEnd {
                speechMel.append(targetMel[sf])
                var v: Float = 0.0
                if sf < pitchRes.frameCount {
                    v = pitchRes.voiced[sf]
                }
                speechVoiced.append(v)
                sf += 1
            }

            inputItems.append(MonotonicAlignmentSearch.AlignmentInputItem(
                utteranceId: id,
                leadSilence: leadSilence,
                trailSilence: trailSilence,
                totalSpeechFrames: speechFrames,
                mel: speechMel,
                voiced: speechVoiced,
                phonemes: tokens
            ))

            if (inputItems.count % 1000) == 0 {
                print("入力アイテム準備進行中: \(inputItems.count) / 5000 件")
            }
        }

        print("全発話データ準備完了: \(inputItems.count) 件。MAS 反復自己収束 (iterativelyAlign, 3 iterations) を開始します...")
        let newAlignments = MonotonicAlignmentSearch.iterativelyAlign(
            items: inputItems,
            iterations: 3,
            meanFramesPerMora: 16.0
        )
        print("MAS 反復自己収束完了: \(newAlignments.count) 件確定")

        try AlignmentStore.save(newAlignments, to: corpusMasCachePath)
        print("mas_alignments.json を更新保存しました: \(corpusMasCachePath) (\(newAlignments.count) 件)")

        let averages = AlignmentStore.computeAverageDurations(from: newAlignments)
        var roundedAverages: [Int32: Float] = [:]
        print("\n=======================================================")
        print("全 5,000 発話 MAS 反復自己収束実測音素平均フレーム数 (集計発話数: \(newAlignments.count))")
        print("=======================================================")
        for (pid, avg) in averages.sorted(by: { $0.key < $1.key }) {
            let sym = vocabulary.token(for: Int(pid))
            let rAvg = roundf(avg * 10.0) / 10.0
            let finalAvg = max(1.0, rAvg)
            roundedAverages[pid] = finalAvg
            print("  ID \(pid) (\(sym)): 平均=\(String(format: "%.2f", avg)) frames -> 確定=\(String(format: "%.1f", finalAvg)) frames (\(String(format: "%.1f", finalAvg * 10.0))ms)")
        }

        // Models/weights.json をロードし、phonemeAverageDurations を更新して保存
        let weightsURL = URL(fileURLWithPath: "Models/weights.json")
        let weightsData = try Data(contentsOf: weightsURL)
        let loadedWeights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        let updatedWeights = loadedWeights.withPhonemeAverageDurations(roundedAverages)
        try WeightCheckpoint.atomicWritePretty(updatedWeights, to: weightsURL)
        print("Models/weights.json の phonemeAverageDurations を MAS 反復収束値で更新保存しました！")
    }
}
