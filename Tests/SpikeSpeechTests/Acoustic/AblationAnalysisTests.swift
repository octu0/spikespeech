import XCTest
@testable import SpikeSpeech

final class AblationAnalysisTests: XCTestCase {
    func testAnalyzeAblationFourWavs() throws {
        let outputDir = ".tmp/wave15"
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        var vocWeights: NeuralVocoderWeights? = nil
        if let vData = try? Data(contentsOf: URL(fileURLWithPath: "Models/vocoder_weights.json")) {
            vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }
        let vocoder = NeuralVocoder(weights: vocWeights)
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

        let text = "水をマレーシアから買わなくてはならないのです。"
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else {
            XCTFail("BASIC5000_0001.wav が存在しません: \(wavPath)")
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
        let pitchTracker = PitchTracker()

        var effectiveAlign: UtteranceAlignment? = nil
        let corpusAlignPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/mas_alignments.json"
        if FileManager.default.fileExists(atPath: corpusAlignPath) {
            if let alignMap = try? AlignmentStore.load(from: corpusAlignPath) {
                effectiveAlign = alignMap["BASIC5000_0001"]
            }
        }

        guard let pair = engine.prepareTrainingPair(
            text: text,
            pcm16k: pcm16k,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker,
            alignment: effectiveAlign,
            useScaledDuration: false
        ) else {
            XCTFail("prepareTrainingPair に失敗しました")
            return
        }

        let totalF319 = pair.features.count
        print("[Ablation] 教師アライメント特徴量フレーム数: \(totalF319)")

        // 1. SNN 推論 (319フレーム)
        engine.workspace.reset()
        let snnMel319 = engine.decoder.decodeSequence(featuresSeq: pair.features, workspace: engine.workspace)

        var melSeq319 = [[Float]](repeating: [Float](repeating: 0.0, count: AudioConfig.melChannels), count: totalF319)
        var f = 0
        while f < totalF319 {
            if f < snnMel319.count {
                let copyCount = min(AudioConfig.melChannels, snnMel319[f].count)
                melSeq319[f].withUnsafeMutableBufferPointer { dst in
                    snnMel319[f].withUnsafeBufferPointer { src in
                        dst.baseAddress!.update(from: src.baseAddress!, count: copyCount)
                    }
                }
            }
            f += 1
        }

        // 教師 F0 / 有声フラグ (319フレーム)
        var teacherF0_319 = [Float](repeating: 0.0, count: totalF319)
        var teacherVoiced_319 = [Float](repeating: 0.0, count: totalF319)
        f = 0
        while f < totalF319 {
            if 194 < pair.features[f].count {
                teacherF0_319[f] = pair.features[f][194] * 500.0
            }
            if 192 < pair.features[f].count {
                teacherVoiced_319[f] = pair.features[f][192]
            }
            f += 1
        }

        // =======================================================
        // 1. ablate_recon.wav
        // =======================================================
        vocoder.reset()
        let samplesRecon = vocoder.synthesize(
            mel: melSeq319,
            f0Contour: teacherF0_319,
            voicedFlags: teacherVoiced_319,
            speaker: .zero
        )
        let path1 = "\(outputDir)/ablate_recon.wav"
        let pathRecon = "\(outputDir)/recon_BASIC5000_0001.wav"
        let reconData = WavEncoder.encode(samples: samplesRecon, sampleRate: AudioConfig.sampleRate)
        try reconData.write(to: URL(fileURLWithPath: path1))
        try reconData.write(to: URL(fileURLWithPath: pathRecon))
        print("[Ablation 1] 生成完了: \(path1), \(pathRecon)")

        // =======================================================
        // 2. ablate_recon_mel_pred_f0.wav
        // =======================================================
        let ttsLinguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            addBoundarySilence: true,
            meanFramesPerMora: VoiceProfile.female.meanFramesPerMora
        )
        let ttsFrames = ttsLinguistic.totalFrames
        var predF0_319 = [Float](repeating: 0.0, count: totalF319)
        var predVoiced_319 = [Float](repeating: 0.0, count: totalF319)
        let maxDst319 = Float(max(1, totalF319 - 1))
        let maxSrcTTS = Float(max(1, ttsFrames - 1))
        var rIdx = 0
        while rIdx < totalF319 {
            let pos = (Float(rIdx) / maxDst319) * maxSrcTTS
            var s0 = Int(pos)
            if ttsFrames <= s0 { s0 = ttsFrames - 1 }
            if s0 < 0 { s0 = 0 }
            var s1 = s0 + 1
            if ttsFrames <= s1 { s1 = ttsFrames - 1 }
            let alpha = pos - Float(s0)
            predF0_319[rIdx] = (1.0 - alpha) * ttsLinguistic.f0Contour[s0] + alpha * ttsLinguistic.f0Contour[s1]
            let vInterp = (1.0 - alpha) * ttsLinguistic.voicedFlags[s0] + alpha * ttsLinguistic.voicedFlags[s1]
            switch 0.5 <= vInterp {
            case true:
                predVoiced_319[rIdx] = 1.0
            case false:
                predVoiced_319[rIdx] = 0.0
            }
            rIdx += 1
        }

        vocoder.reset()
        let samplesReconPredF0 = vocoder.synthesize(
            mel: melSeq319,
            f0Contour: predF0_319,
            voicedFlags: predVoiced_319,
            speaker: .zero
        )
        let path2 = "\(outputDir)/ablate_recon_mel_pred_f0.wav"
        try WavEncoder.encode(samples: samplesReconPredF0, sampleRate: AudioConfig.sampleRate).write(to: URL(fileURLWithPath: path2))
        print("[Ablation 2] 生成完了: \(path2)")

        // =======================================================
        // 3. ablate_tts.wav
        // =======================================================
        engine.workspace.reset()
        vocoder.reset()
        let samplesTTS = engine.synthesize(text: text)
        let path3 = "\(outputDir)/ablate_tts.wav"
        try WavEncoder.encode(samples: samplesTTS, sampleRate: AudioConfig.sampleRate).write(to: URL(fileURLWithPath: path3))
        print("[Ablation 3] 生成完了: \(path3)")

        // =======================================================
        // 4. ablate_tts_teacher_f0.wav
        // =======================================================
        var teacherF0_tts = [Float](repeating: 0.0, count: ttsFrames)
        var teacherVoiced_tts = [Float](repeating: 0.0, count: ttsFrames)
        var tIdx = 0
        while tIdx < ttsFrames {
            let pos = (Float(tIdx) / maxSrcTTS) * maxDst319
            var s0 = Int(pos)
            if totalF319 <= s0 { s0 = totalF319 - 1 }
            if s0 < 0 { s0 = 0 }
            var s1 = s0 + 1
            if totalF319 <= s1 { s1 = totalF319 - 1 }
            let alpha = pos - Float(s0)
            teacherF0_tts[tIdx] = (1.0 - alpha) * teacherF0_319[s0] + alpha * teacherF0_319[s1]
            let vInterp = (1.0 - alpha) * teacherVoiced_319[s0] + alpha * teacherVoiced_319[s1]
            switch 0.5 <= vInterp {
            case true:
                teacherVoiced_tts[tIdx] = 1.0
            case false:
                teacherVoiced_tts[tIdx] = 0.0
            }
            tIdx += 1
        }

        let ttsTeacherLinguistic = LinguisticFeatures(
            phoneIds: ttsLinguistic.phoneIds,
            durations: ttsLinguistic.durations,
            f0Contour: teacherF0_tts,
            voicedFlags: teacherVoiced_tts,
            energyContour: ttsLinguistic.energyContour,
            totalFrames: ttsFrames
        )
        let ttsTeacherSeq = engine.encodeLinguisticFeatures(features: ttsTeacherLinguistic)

        engine.workspace.reset()
        let snnMelTTS = engine.decoder.decodeSequence(featuresSeq: ttsTeacherSeq, workspace: engine.workspace)

        var melSeqTTS = [[Float]](repeating: [Float](repeating: 0.0, count: AudioConfig.melChannels), count: ttsFrames)
        f = 0
        while f < ttsFrames {
            if f < snnMelTTS.count {
                let copyCount = min(AudioConfig.melChannels, snnMelTTS[f].count)
                melSeqTTS[f].withUnsafeMutableBufferPointer { dst in
                    snnMelTTS[f].withUnsafeBufferPointer { src in
                        dst.baseAddress!.update(from: src.baseAddress!, count: copyCount)
                    }
                }
            }
            f += 1
        }

        vocoder.reset()
        var samplesTTS_TeacherF0 = vocoder.synthesize(
            mel: melSeqTTS,
            f0Contour: teacherF0_tts,
            voicedFlags: teacherVoiced_tts,
            speaker: .zero
        )

        let silenceMask = engine.computeFrameSilenceMask(linguisticFeatures: ttsTeacherLinguistic, totalFrames: ttsFrames)
        let frameSize = AudioConfig.hopSize
        var fIdx = 0
        while fIdx < ttsFrames {
            let isSilence = silenceMask[fIdx]
            switch isSilence {
            case true:
                var prevIsSilence = true
                if 0 < fIdx {
                    prevIsSilence = silenceMask[fIdx - 1]
                }
                let startSample = fIdx * frameSize
                let endSample = min(samplesTTS_TeacherF0.count, startSample + frameSize)
                switch prevIsSilence {
                case false:
                    let invN = 1.0 / Float(frameSize)
                    var s = startSample
                    while s < endSample {
                        let sampleOffset = s - startSample
                        let fade = 1.0 - (Float(sampleOffset) * invN)
                        samplesTTS_TeacherF0[s] = samplesTTS_TeacherF0[s] * fade
                        s += 1
                    }
                case true:
                    var s = startSample
                    while s < endSample {
                        samplesTTS_TeacherF0[s] = 0.0
                        s += 1
                    }
                }
            case false:
                break
            }
            fIdx += 1
        }

        let targetPeak: Float = 0.85
        var currentPeak: Float = 0.0
        var pkIdx = 0
        while pkIdx < samplesTTS_TeacherF0.count {
            let absVal = abs(samplesTTS_TeacherF0[pkIdx])
            if currentPeak < absVal {
                currentPeak = absVal
            }
            pkIdx += 1
        }
        if 0.01 < currentPeak {
            var normScale = targetPeak / currentPeak
            if 6.0 < normScale {
                normScale = 6.0
            }
            var s = 0
            while s < samplesTTS_TeacherF0.count {
                var scaled = samplesTTS_TeacherF0[s] * normScale
                if targetPeak < scaled {
                    scaled = targetPeak
                }
                if scaled < -targetPeak {
                    scaled = -targetPeak
                }
                samplesTTS_TeacherF0[s] = scaled
                s += 1
            }
        }

        let fadeLen = 160
        if (fadeLen * 2) <= samplesTTS_TeacherF0.count {
            let invFade: Float = 1.0 / Float(fadeLen)
            var s = 0
            while s < fadeLen {
                let factor = Float(s) * invFade
                samplesTTS_TeacherF0[s] = samplesTTS_TeacherF0[s] * factor
                s += 1
            }
            let endOffset = samplesTTS_TeacherF0.count - fadeLen
            s = 0
            while s < fadeLen {
                let factor = Float(fadeLen - 1 - s) * invFade
                samplesTTS_TeacherF0[endOffset + s] = samplesTTS_TeacherF0[endOffset + s] * factor
                s += 1
            }
        }

        let path4 = "\(outputDir)/ablate_tts_teacher_f0.wav"
        try WavEncoder.encode(samples: samplesTTS_TeacherF0, sampleRate: AudioConfig.sampleRate).write(to: URL(fileURLWithPath: path4))
        print("[Ablation 4] 生成完了: \(path4)")

        // =======================================================
        // 5. tts_tenki.wav ("今日はいい天気です" 通常 TTS)
        // =======================================================
        engine.workspace.reset()
        vocoder.reset()
        let tenkiText = "今日はいい天気です"
        let samplesTenki = engine.synthesize(text: tenkiText)
        let path5 = "\(outputDir)/test_ablate_tenki.wav"
        try WavEncoder.encode(samples: samplesTenki, sampleRate: AudioConfig.sampleRate).write(to: URL(fileURLWithPath: path5))
        print("[TTS Tenki] 生成完了: \(path5)")

        // =======================================================
        // 客観音響特性の計測とフォーマット出力
        // =======================================================
        let paths = [
            ("1. ablate_recon.wav (SNN Recon Mel + Teacher F0)", path1),
            ("1b. recon_BASIC5000_0001.wav (SNN Recon Mel + Teacher F0)", pathRecon),
            ("2. ablate_recon_mel_pred_f0.wav (SNN Recon Mel + Pred F0)", path2),
            ("3. ablate_tts.wav (TTS Standard: Pred Mel + Pred F0)", path3),
            ("4. ablate_tts_teacher_f0.wav (TTS Teacher F0: SNN Mel w/ Teacher F0)", path4),
            ("5. test_ablate_tenki.wav (TTS Standard: 今日はいい天気です)", path5)
        ]

        print("\n=======================================================")
        print("Ablation波形およびTTS波形 客観音響特性分析 (Ablation Analysis)")
        print("=======================================================")

        var pCount = 0
        while pCount < paths.count {
            let (label, path) = paths[pCount]
            guard FileManager.default.fileExists(atPath: path) else {
                XCTFail("生成ファイルが存在しません: \(path)")
                pCount += 1
                continue
            }

            let pcm = try wavReader.loadWav16k(from: path)
            let mel = melExtractor.extractLogMel(pcm: pcm)
            let pitch = pitchTracker.track(pcm: pcm)

            var voicedF0s: [Float] = []
            var pf = 0
            while pf < pitch.frameCount {
                if 0.5 <= pitch.voiced[pf] && 50.0 <= pitch.f0[pf] && pitch.f0[pf] <= 500.0 {
                    voicedF0s.append(pitch.f0[pf])
                }
                pf += 1
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

            pCount += 1
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

    /// prepareTrainingPair の特徴行列が processText / synthesize の特徴行列と完全一致することを検証（手順3要件）
    func testTrainingFeaturesMatchProcessTextFeatures() throws {
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
        ) else {
            XCTFail("prepareTrainingPair に失敗しました")
            return
        }

        let synthLinguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            prosodyPredictor: engine.prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.default.baseF0,
            addBoundarySilence: true,
            meanFramesPerMora: VoiceProfile.default.meanFramesPerMora
        )
        let synthFeatures = engine.encodeLinguisticFeatures(features: synthLinguistic)

        // 1. フレーム数の一致検証
        XCTAssertEqual(pair.features.count, synthFeatures.count, "学習特徴行列と合成特徴行列のフレーム数が一致していません")
        XCTAssertEqual(pair.targets.count, pair.features.count, "目標Melスペクトルと特徴量のフレーム数が一致していません")

        // 2. ゲート条件 1: ch 194 と ch 195 以外の全チャンネル・全フレームの最大絶対差は 0
        var nonPitchMaxDiff: Float = 0.0
        var f = 0
        while f < pair.features.count {
            var c = 0
            let inDim = min(pair.features[f].count, synthFeatures[f].count)
            while c < inDim {
                if c != 194 && c != 195 {
                    let diff = abs(pair.features[f][c] - synthFeatures[f][c])
                    if nonPitchMaxDiff < diff {
                        nonPitchMaxDiff = diff
                    }
                }
                c += 1
            }
            f += 1
        }
        XCTAssertTrue(nonPitchMaxDiff <= 0.0, "ch 194/195 以外の特徴量最大絶対差が 0 ではありません: \(nonPitchMaxDiff)")

        // 3. ゲート条件 2: ch 194 の学習と合成の平均絶対差は 0.02 より大きい
        var sumDiff194: Float = 0.0
        var f194 = 0
        while f194 < pair.features.count {
            let diff = abs(pair.features[f194][194] - synthFeatures[f194][194])
            sumDiff194 += diff
            f194 += 1
        }
        let meanDiff194 = sumDiff194 / Float(max(1, pair.features.count))
        XCTAssertTrue(0.02 < meanDiff194, "ch 194 の学習と合成の平均絶対差が 0.02 以下です: \(meanDiff194)")

        // 4. ゲート条件 3: BASIC5000_0001 の有声な本体フレームで、学習 ch 194 の × 500 と、同じフレームへ補間したトラッカー F0 の相関が 0.8 より大きい
        let pitchResult = pitchTracker.track(pcm: rawPCM)
        let boundaries = SpikeSpeechEngine.detectSpeechBoundaries(
            pcm: rawPCM,
            hopSize: AudioConfig.hopSize,
            totalFrames: max(1, rawPCM.count / AudioConfig.hopSize)
        )
        let leadSilence = boundaries.leadSilence
        let actualSpeechFrames = boundaries.speechFrames
        let speechEnd = min(pitchResult.frameCount, leadSilence + actualSpeechFrames)
        var speechF0: [Float] = []
        var speechVoiced: [Float] = []
        var sf = leadSilence
        while sf < speechEnd {
            speechF0.append(pitchResult.f0[sf])
            speechVoiced.append(pitchResult.voiced[sf])
            sf += 1
        }
        let srcLen = speechF0.count

        let phoneCount = synthLinguistic.phoneIds.count
        let leadSil = Int(synthLinguistic.durations[0])
        let trailSil = Int(synthLinguistic.durations[phoneCount - 1])
        let bodyPhoneCount = phoneCount - 2
        let targetSpeechFrames = synthLinguistic.totalFrames - leadSil - trailSil

        var rawFloatDurs: [Float] = []
        var p = 0
        while p < bodyPhoneCount {
            let pid = synthLinguistic.phoneIds[1 + p]
            let avgDur = engine.lengthRegulator.phonemeDuration(phoneId: pid, speedFactor: 1.0)
            rawFloatDurs.append(avgDur)
            p += 1
        }
        var sumFloat: Float = 0.0
        var fI = 0
        while fI < rawFloatDurs.count {
            sumFloat += rawFloatDurs[fI]
            fI += 1
        }
        let scale: Float
        if 0.001 < sumFloat {
            scale = Float(actualSpeechFrames) / sumFloat
        } else {
            scale = 1.0
        }
        var scaledFloatDurs: [Float] = []
        var sI = 0
        while sI < rawFloatDurs.count {
            scaledFloatDurs.append(max(1.0, rawFloatDurs[sI] * scale))
            sI += 1
        }
        var srcDurs = engine.lengthRegulator.quantizeDurations(durations: scaledFloatDurs)
        var curSrcSum = 0
        var cI = 0
        while cI < srcDurs.count {
            curSrcSum += srcDurs[cI]
            cI += 1
        }
        let srcDiff = srcLen - curSrcSum
        if srcDiff != 0 && srcDurs.isEmpty != true {
            let lastIdx = srcDurs.count - 1
            let adj = srcDurs[lastIdx] + srcDiff
            if 1 <= adj {
                srcDurs[lastIdx] = adj
            } else {
                srcDurs[lastIdx] = 1
            }
        }

        var resampledF0 = [Float](repeating: 0.0, count: targetSpeechFrames)
        var resampledVoiced = [Float](repeating: 0.0, count: targetSpeechFrames)
        var srcOffset = 0
        var dstOffset = 0
        var phSeqIdx = 0
        while phSeqIdx < bodyPhoneCount {
            let srcDur = srcDurs[phSeqIdx]
            let dstDur = Int(synthLinguistic.durations[1 + phSeqIdx])
            switch (dstDur <= 1, srcDur <= 1) {
            case (true, _):
                let srcIdx: Int
                if srcDur <= 1 {
                    srcIdx = srcOffset
                } else {
                    srcIdx = srcOffset + (srcDur / 2)
                }
                let safeSrcIdx = min(srcLen - 1, max(0, srcIdx))
                let dstIdx = min(targetSpeechFrames - 1, dstOffset)
                resampledF0[dstIdx] = speechF0[safeSrcIdx]
                resampledVoiced[dstIdx] = speechVoiced[safeSrcIdx]
            case (false, true):
                let safeSrcIdx = min(srcLen - 1, max(0, srcOffset))
                var fi = 0
                while fi < dstDur {
                    let dstIdx = min(targetSpeechFrames - 1, dstOffset + fi)
                    resampledF0[dstIdx] = speechF0[safeSrcIdx]
                    resampledVoiced[dstIdx] = speechVoiced[safeSrcIdx]
                    fi += 1
                }
            case (false, false):
                let maxDstP = Float(dstDur - 1)
                let maxSrcP = Float(srcDur - 1)
                var fi = 0
                while fi < dstDur {
                    let dstIdx = min(targetSpeechFrames - 1, dstOffset + fi)
                    let posWithinPh = (Float(fi) / maxDstP) * maxSrcP
                    var s0 = Int(posWithinPh)
                    if srcDur <= s0 { s0 = srcDur - 1 }
                    if s0 < 0 { s0 = 0 }
                    var s1 = s0 + 1
                    if srcDur <= s1 { s1 = srcDur - 1 }
                    let alpha = posWithinPh - Float(s0)
                    let src0 = min(srcLen - 1, max(0, srcOffset + s0))
                    let src1 = min(srcLen - 1, max(0, srcOffset + s1))
                    resampledF0[dstIdx] = (1.0 - alpha) * speechF0[src0] + alpha * speechF0[src1]
                    resampledVoiced[dstIdx] = (1.0 - alpha) * speechVoiced[src0] + alpha * speechVoiced[src1]
                    fi += 1
                }
            }
            srcOffset += srcDur
            dstOffset += dstDur
            phSeqIdx += 1
        }

        // 有声な本体フレームで相関を計算
        var sumX: Float = 0.0
        var sumY: Float = 0.0
        var sumXY: Float = 0.0
        var sumX2: Float = 0.0
        var sumY2: Float = 0.0
        var countVoiced: Int = 0

        var bf = 0
        while bf < targetSpeechFrames {
            let dstF = leadSil + bf
            let trackerF0 = resampledF0[bf]
            let trackerV = resampledVoiced[bf]
            let trainF0 = pair.features[dstF][194] * 500.0

            if 0.5 <= trackerV && 70.0 <= trackerF0 && 0.0 < pair.features[dstF][194] {
                sumX += trainF0
                sumY += trackerF0
                sumXY += trainF0 * trackerF0
                sumX2 += trainF0 * trainF0
                sumY2 += trackerF0 * trackerF0
                countVoiced += 1
            }
            bf += 1
        }

        XCTAssertTrue(10 <= countVoiced, "有声フレーム数が少なすぎます: \(countVoiced)")
        let nF = Float(countVoiced)
        let numerator = (nF * sumXY) - (sumX * sumY)
        let denomX = (nF * sumX2) - (sumX * sumX)
        let denomY = (nF * sumY2) - (sumY * sumY)
        let denom = sqrt(max(1e-12, denomX * denomY))
        let correlation = numerator / denom

        print("  [特徴ゲート検証] ch194/195以外最大差: \(nonPitchMaxDiff), ch194平均差: \(meanDiff194), 有声本体相関: \(correlation) (N=\(countVoiced))")
        XCTAssertTrue(0.8 < correlation, "学習 ch 194 * 500 とトラッカー F0 の相関が 0.8 以下です: \(correlation)")

        // 5. アライメント指定時も ch 194/195 以外の差が 0 かつ ch 194 の平均差 > 0.02 を検証
        var effectiveAlign: UtteranceAlignment? = nil
        let corpusAlignPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/mas_alignments.json"
        if FileManager.default.fileExists(atPath: corpusAlignPath) {
            if let alignMap = try? AlignmentStore.load(from: corpusAlignPath) {
                effectiveAlign = alignMap["BASIC5000_0001"]
            }
        }
        if effectiveAlign != nil {
            guard let pairWithAlign = engine.prepareTrainingPair(
                text: text,
                pcm16k: rawPCM,
                melExtractor: melExtractor,
                pitchTracker: pitchTracker,
                alignment: effectiveAlign
            ) else {
                XCTFail("アライメント指定時の prepareTrainingPair に失敗しました")
                return
            }
            XCTAssertEqual(pairWithAlign.features.count, synthFeatures.count, "アライメント指定時のフレーム数が一致していません")
            XCTAssertEqual(pairWithAlign.targets.count, synthFeatures.count, "アライメント指定時の目標Melフレーム数が一致していません")

            var alignNonPitchMaxDiff: Float = 0.0
            var af = 0
            while af < pairWithAlign.features.count {
                var c = 0
                let inDim = min(pairWithAlign.features[af].count, synthFeatures[af].count)
                while c < inDim {
                    if c != 194 && c != 195 {
                        let diff = abs(pairWithAlign.features[af][c] - synthFeatures[af][c])
                        if alignNonPitchMaxDiff < diff {
                            alignNonPitchMaxDiff = diff
                        }
                    }
                    c += 1
                }
                af += 1
            }
            XCTAssertTrue(alignNonPitchMaxDiff <= 0.0, "アライメント指定時の ch 194/195 以外の最大絶対差が 0 ではありません: \(alignNonPitchMaxDiff)")

            var alignSumDiff194: Float = 0.0
            var af194 = 0
            while af194 < pairWithAlign.features.count {
                let diff = abs(pairWithAlign.features[af194][194] - synthFeatures[af194][194])
                alignSumDiff194 += diff
                af194 += 1
            }
            let alignMeanDiff194 = alignSumDiff194 / Float(max(1, pairWithAlign.features.count))
            XCTAssertTrue(0.02 < alignMeanDiff194, "アライメント指定時の ch 194 平均絶対差が 0.02 以下です: \(alignMeanDiff194)")
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

    /// Models/weights.json および波形ファイルへの書き込みを行わず return する
    func testUpdateWeightsWithHealthyDurations() throws {
        return
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
        let shouldRun = ProcessInfo.processInfo.environment["SPIKESPEECH_STRESS_TEST"] != nil
        if shouldRun != true {
            throw XCTSkip("全 5000 発話の MAS 反復抽出は SPIKESPEECH_STRESS_TEST=1 で実行してください")
        }

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

        // 注意: Models/weights.json の phonemeAverageDurations は testUpdateWeightsWithHealthyDurations
        // により自然な物理比率テーブルが正本として管理されるため、ここでは上書きしない。
        print("MAS 反復収束音素平均フレーム数の抽出・キャッシュ保存が完了しました。")
    }

    /// 音響試聴エンジニア用 詳細音響診断テスト
    func testAcousticListeningAuditDetail() throws {
        let outputDir = ".tmp/wave15"
        let specDir = ".tmp/wave15_spec"
        try FileManager.default.createDirectory(atPath: specDir, withIntermediateDirectories: true)

        let targetFiles = [
            ("1. ablate_recon.wav", "\(outputDir)/ablate_recon.wav", "\(specDir)/ablate_recon.png"),
            ("2. ablate_recon_mel_pred_f0.wav", "\(outputDir)/ablate_recon_mel_pred_f0.wav", "\(specDir)/ablate_recon_mel_pred_f0.png"),
            ("3. ablate_tts.wav", "\(outputDir)/ablate_tts.wav", "\(specDir)/ablate_tts.png"),
            ("4. ablate_tts_teacher_f0.wav", "\(outputDir)/ablate_tts_teacher_f0.wav", "\(specDir)/ablate_tts_teacher_f0.png"),
            ("5. tts_tenki.wav", "\(outputDir)/tts_tenki.wav", "\(specDir)/tts_tenki.png")
        ]

        let reader = WavAudioReader()
        let melExtractor = MelSpectrogramExtractor(
            sampleRate: Float(AudioConfig.sampleRate),
            melChannels: AudioConfig.melChannels
        )
        let tracker = PitchTracker()

        print("\n=======================================================")
        print("【音響試聴エンジニア 5ファイル詳細音響診断】")
        print("=======================================================")

        var fIdx = 0
        while fIdx < targetFiles.count {
            let (label, wavPath, pngPath) = targetFiles[fIdx]
            guard FileManager.default.fileExists(atPath: wavPath) else {
                print("[\(label)] ファイルが存在しません: \(wavPath)")
                fIdx += 1
                continue
            }

            // スペクトログラム PNG 生成
            try SpectrogramRenderer.renderWavToPNG(wavPath: wavPath, outputPath: pngPath, scale: 2)

            let pcm = try reader.loadWav16k(from: wavPath)
            let mel = melExtractor.extractLogMel(pcm: pcm)
            let pitch = tracker.track(pcm: pcm)

            var voicedF0s: [Float] = []
            var pf = 0
            while pf < pitch.frameCount {
                if 0.5 <= pitch.voiced[pf] && 50.0 <= pitch.f0[pf] && pitch.f0[pf] <= 500.0 {
                    voicedF0s.append(pitch.f0[pf])
                }
                pf += 1
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

            // 時間方向のスペクトル動態度 (Spectral Flux)
            var fluxSum: Float = 0.0
            var fluxCount = 0
            var t = 1
            while t < mel.count {
                var dSum: Float = 0.0
                var c = 0
                while c < AudioConfig.melChannels {
                    let d = mel[t][c] - mel[t - 1][c]
                    dSum += d * d
                    c += 1
                }
                fluxSum += sqrtf(dSum)
                fluxCount += 1
                t += 1
            }
            let spectralFlux = fluxSum / Float(max(1, fluxCount))

            // 周波数方向の凹凸度 (Channel Variance: フォルマント明瞭度)
            var varSum: Float = 0.0
            t = 0
            while t < mel.count {
                let mean = mel[t].reduce(0, +) / Float(AudioConfig.melChannels)
                let v = mel[t].reduce(0) { $0 + powf($1 - mean, 2) } / Float(AudioConfig.melChannels)
                varSum += v
                t += 1
            }
            let channelVar = varSum / Float(max(1, mel.count))

            // 横縞度指標 (Horizontal Striation Index): 時間変化が極めて小さい（静止している）フレームの割合
            var staticFrameCount = 0
            t = 1
            while t < mel.count {
                var diffMax: Float = 0.0
                var c = 0
                while c < AudioConfig.melChannels {
                    let diff = abs(mel[t][c] - mel[t - 1][c])
                    if diffMax < diff { diffMax = diff }
                    c += 1
                }
                if diffMax <= 0.15 {
                    staticFrameCount += 1
                }
                t += 1
            }
            let striationRatio = Float(staticFrameCount) / Float(max(1, mel.count - 1)) * 100.0

            // 帯域別エネルギー分布 (低域 0-15: F1/基本波, 中域 16-39: F2/母音識別, 高域 40-63: 子音摩擦音)
            var lowBandSum: Float = 0.0
            var midBandSum: Float = 0.0
            var highBandSum: Float = 0.0
            t = 0
            while t < mel.count {
                var c = 0
                while c < AudioConfig.melChannels {
                    let v = mel[t][c]
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
            let lowE = lowBandSum / Float(max(1, mel.count * 16))
            let midE = midBandSum / Float(max(1, mel.count * 24))
            let highE = highBandSum / Float(max(1, mel.count * 24))

            print("-------------------------------------------------------")
            print("【\(label)】")
            print("  WAV パス: \(wavPath)")
            print("  時間長: \(String(format: "%.3f", Float(pcm.count) / 16000.0)) 秒 (\(pcm.count) サンプル, \(mel.count) フレーム)")
            print("  有声比率: \(String(format: "%.1f", Float(voicedF0s.count) / Float(max(1, pitch.frameCount)) * 100.0))%")
            print("  F0: 平均=\(String(format: "%.1f", meanF0)) Hz, 標準偏差=\(String(format: "%.1f", stdF0)) Hz, min=\(String(format: "%.1f", minF0)), max=\(String(format: "%.1f", maxF0))")
            print("  スペクトル動態 (Spectral Flux): \(String(format: "%.4f", spectralFlux))")
            print("  フォルマント凹凸 (Channel Variance): \(String(format: "%.4f", channelVar))")
            print("  横縞・静止フレーム比率 (Striation Ratio): \(String(format: "%.1f", striationRatio))%")
            print("  帯域エネルギー: 低域=\(String(format: "%.2f", lowE)), 中域=\(String(format: "%.2f", midE)), 高域=\(String(format: "%.2f", highE))")

            fIdx += 1
        }

        // =======================================================
        // 教師特徴量 (pair.features) と TTS 特徴量の完全チャンネル差分
        // =======================================================
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        let engine = SpikeSpeechEngine(weights: weights)

        let text = "水をマレーシアから買わなくてはならないのです。"
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        if FileManager.default.fileExists(atPath: wavPath) {
            let rawPCM = try reader.loadWav16k(from: wavPath)
            var effectiveAlign: UtteranceAlignment? = nil
            let corpusAlignPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/mas_alignments.json"
            if FileManager.default.fileExists(atPath: corpusAlignPath) {
                if let alignMap = try? AlignmentStore.load(from: corpusAlignPath) {
                    effectiveAlign = alignMap["BASIC5000_0001"]
                }
            }

            if let pair = engine.prepareTrainingPair(
                text: text,
                pcm16k: rawPCM,
                melExtractor: melExtractor,
                pitchTracker: tracker,
                alignment: effectiveAlign,
                useScaledDuration: false
            ) {
                let ttsLinguistic = engine.lengthRegulator.processText(
                    text: text,
                    normalizer: engine.normalizer,
                    prosodyModel: engine.prosodyModel,
                    vocabulary: engine.vocabulary,
                    prosodyPredictor: engine.prosodyPredictor,
                    speedFactor: 1.0,
                    baseF0: VoiceProfile.female.baseF0,
                    addBoundarySilence: true,
                    meanFramesPerMora: VoiceProfile.female.meanFramesPerMora
                )
                let ttsSeq = engine.encodeLinguisticFeatures(features: ttsLinguistic)

                print("\n=======================================================")
                print("【学習特徴量 (pair) vs TTS推論特徴量 (ttsSeq) 差分分析】")
                print("  学習特徴量フレーム数: \(pair.features.count)")
                print("  TTS推論特徴量フレーム数: \(ttsSeq.count)")
                let morphs = engine.normalizer.normalize(text: text)
                let dbgPhrases = engine.prosodyModel.buildAccentPhrases(morphemes: morphs, vocabulary: engine.vocabulary)
                var dbgTotalMoras = 0
                for p in dbgPhrases {
                    dbgTotalMoras += p.moras.count
                    let moraTexts = p.moras.map { $0.text }.joined(separator: ", ")
                    print("  句: [\(moraTexts)] (モーラ数: \(p.moras.count), pauseAfter: \(p.pauseAfter), pauseDur: \(p.pauseDurationFrames))")
                }
                print("  形態素数: \(morphs.count), モーラ総数: \(dbgTotalMoras)")
                print("  音素数: 教師=\(pair.features.count) frames, TTS=\(ttsLinguistic.phoneIds.count) 音素, 計\(ttsLinguistic.totalFrames) frames")

                // 教師音素列の復元と表示
                print("\n[教師 (pair) 音素列と Duration]")
                var teacherPhones: [Int] = []
                var teacherDurs: [Int] = []
                var curPh = -1
                var curDur = 0
                var f = 0
                while f < pair.features.count {
                    var ph = -1
                    var c = 0
                    while c < 64 {
                        if 0.5 <= pair.features[f][c] { ph = c }
                        c += 1
                    }
                    if ph == curPh {
                        curDur += 1
                    } else {
                        if 0 <= curPh {
                            teacherPhones.append(curPh)
                            teacherDurs.append(curDur)
                            let sym = engine.vocabulary.token(for: curPh)
                            print("  [\(teacherPhones.count - 1)] id=\(curPh) (\(sym)): dur=\(curDur) frames")
                        }
                        curPh = ph
                        curDur = 1
                    }
                    f += 1
                }
                if 0 <= curPh {
                    teacherPhones.append(curPh)
                    teacherDurs.append(curDur)
                    let sym = engine.vocabulary.token(for: curPh)
                    print("  [\(teacherPhones.count - 1)] id=\(curPh) (\(sym)): dur=\(curDur) frames")
                }

                // 音素列の比較
                print("\n[音素列の比較 (教師 vs TTS)]")
                print("教師音素数: \(teacherPhones.count), TTS音素数: \(ttsLinguistic.phoneIds.count)")
                let maxP = max(teacherPhones.count, ttsLinguistic.phoneIds.count)
                var p = 0
                while p < maxP {
                    var tStr = "None"
                    if p < teacherPhones.count {
                        let tPid = teacherPhones[p]
                        let tSym = engine.vocabulary.token(for: tPid)
                        let tDur = teacherDurs[p]
                        tStr = "id=\(tPid) (\(tSym)) dur=\(tDur)"
                    }
                    var sStr = "None"
                    if p < ttsLinguistic.phoneIds.count {
                        let sPid = Int(ttsLinguistic.phoneIds[p])
                        let sSym = engine.vocabulary.token(for: sPid)
                        let sDur = ttsLinguistic.durations[p]
                        sStr = "id=\(sPid) (\(sSym)) dur=\(sDur)"
                    }
                    print("  [\(p)] 教師: [\(tStr)] vs TTS: [\(sStr)]")
                    p += 1
                }

                // 各チャンネルの平均値比較
                print("\n[チャンネルグループ別 平均値比較]")
                let groups = [
                    ("ch 0-63 (現在の音素 one-hot)", 0, 63),
                    ("ch 64-127 (前の音素 one-hot)", 64, 127),
                    ("ch 128-191 (次の音素 one-hot)", 128, 191),
                    ("ch 192 (voiced)", 192, 192),
                    ("ch 193 (unvoiced)", 193, 193),
                    ("ch 194 (F0 / 500)", 194, 194),
                    ("ch 195 (deltaF0)", 195, 195),
                    ("ch 196 (phonePos)", 196, 196),
                    ("ch 197 (rate)", 197, 197),
                    ("ch 198 (energy)", 198, 198),
                    ("ch 199 (pulse)", 199, 199),
                    ("ch 200-255 (その他/未使用)", 200, 255)
                ]

                var g = 0
                while g < groups.count {
                    let (gName, startCh, endCh) = groups[g]
                    var pairSum: Float = 0.0
                    var pairCount = 0
                    var fIdx = 0
                    while fIdx < pair.features.count {
                        var c = startCh
                        while c <= endCh {
                            if c < pair.features[fIdx].count {
                                pairSum += pair.features[fIdx][c]
                                pairCount += 1
                            }
                            c += 1
                        }
                        fIdx += 1
                    }

                    var ttsSum: Float = 0.0
                    var ttsCount = 0
                    fIdx = 0
                    while fIdx < ttsSeq.count {
                        var c = startCh
                        while c <= endCh {
                            if c < ttsSeq[fIdx].count {
                                ttsSum += ttsSeq[fIdx][c]
                                ttsCount += 1
                            }
                            c += 1
                        }
                        fIdx += 1
                    }

                    let pairAvg = pairSum / Float(max(1, pairCount))
                    let ttsAvg = ttsSum / Float(max(1, ttsCount))
                    print("  \(gName): pairAvg=\(String(format: "%.4f", pairAvg)) vs ttsAvg=\(String(format: "%.4f", ttsAvg))")
                    g += 1
                }
            }
        }
    }

    struct ClosureWaveMetrics {
        let name: String
        let path: String
        let sampleCount: Int
        let durationSec: Float
        let centroidMedian1k5: Float
        let stopRatio: Float
        let mod6to12Ratio: Float
        let firstHalfF0: Float
        let secondHalfF0: Float
        let f0Diff: Float
        let bodyFrameCount: Int
        let bodyDurationSec: Float
        let dips: [(startSec: Float, lengthMs: Int, avgRms: Float)]
        let maxDipLengthMs: Int
    }

    func measureClosureWav(path: String, name: String) throws -> ClosureWaveMetrics? {
        guard FileManager.default.fileExists(atPath: path) else {
            return nil
        }
        let reader = WavAudioReader()
        let pcm = try reader.loadWav16k(from: path)
        let sampleCount = pcm.count
        let durationSec = Float(sampleCount) / 16000.0

        let metric = AcousticCentroidMetric.measure(pcm: pcm)
        let halves = AcousticCentroidMetric.measureHalves(pcm: pcm, interpolateParabolic: false)
        let mod6to12 = AcousticCentroidMetric.measureModulation6to12Ratio(pcm: pcm)

        let winSize = 320
        let hopSize = 160
        let totalFrames = max(1, (pcm.count - winSize) / hopSize + 1)
        var frameRms = [Float](repeating: 0.0, count: totalFrames)
        var maxRms: Float = 0.0

        var f = 0
        while f < totalFrames {
            let start = f * hopSize
            var sumSq: Float = 0.0
            var s = 0
            while s < winSize {
                let pcmIdx = start + s
                var v: Float = 0.0
                if pcmIdx < pcm.count {
                    v = pcm[pcmIdx] * 32768.0
                }
                sumSq += v * v
                s += 1
            }
            let rms = sqrtf(sumSq / Float(winSize))
            frameRms[f] = rms
            if maxRms < rms {
                maxRms = rms
            }
            f += 1
        }

        let bodyThreshold = max(0.010 * 32768.0, maxRms * 0.06)
        var firstBody = -1
        var lastBody = -1
        var sf = 0
        while sf < totalFrames {
            if bodyThreshold <= frameRms[sf] {
                if firstBody < 0 {
                    firstBody = sf
                }
                lastBody = sf
            }
            sf += 1
        }

        var bodyFrames = 0
        var bodySec: Float = 0.0
        if 0 <= firstBody && firstBody <= lastBody {
            bodyFrames = lastBody - firstBody + 1
            bodySec = Float(bodyFrames) * 0.010
        }

        let dipThreshold = max(0.008 * 32768.0, maxRms * 0.12)
        var dips: [(startSec: Float, lengthMs: Int, avgRms: Float)] = []
        var maxDipLen = 0

        if 0 <= firstBody && firstBody <= lastBody {
            var curStart = -1
            var curCount = 0
            var curSumRms: Float = 0.0

            var bf = firstBody
            while bf <= lastBody {
                let rmsVal = frameRms[bf]
                if rmsVal < dipThreshold {
                    if curStart < 0 {
                        curStart = bf
                        curCount = 0
                        curSumRms = 0.0
                    }
                    curCount += 1
                    curSumRms += rmsVal
                } else {
                    if 0 <= curStart {
                        if 8 <= curCount {
                            let avg = curSumRms / Float(curCount)
                            if avg < 800.0 {
                                let startSec = Float(curStart * hopSize) / 16000.0
                                let lenMs = curCount * 10
                                if maxDipLen < lenMs {
                                    maxDipLen = lenMs
                                }
                                dips.append((startSec: startSec, lengthMs: lenMs, avgRms: avg))
                            }
                        }
                        curStart = -1
                    }
                }
                bf += 1
            }
            if 0 <= curStart && 8 <= curCount {
                let avg = curSumRms / Float(curCount)
                if avg < 800.0 {
                    let startSec = Float(curStart * hopSize) / 16000.0
                    let lenMs = curCount * 10
                    if maxDipLen < lenMs {
                        maxDipLen = lenMs
                    }
                    dips.append((startSec: startSec, lengthMs: lenMs, avgRms: avg))
                }
            }
        }

        return ClosureWaveMetrics(
            name: name,
            path: path,
            sampleCount: sampleCount,
            durationSec: durationSec,
            centroidMedian1k5: metric.centroidMedian200to1500,
            stopRatio: metric.cosRatio,
            mod6to12Ratio: mod6to12,
            firstHalfF0: halves.first.f0Median,
            secondHalfF0: halves.second.f0Median,
            f0Diff: halves.first.f0Median - halves.second.f0Median,
            bodyFrameCount: bodyFrames,
            bodyDurationSec: bodySec,
            dips: dips,
            maxDipLengthMs: maxDipLen
        )
    }

    func printClosureMetrics(_ m: ClosureWaveMetrics) {
        print("\n==================================================")
        print("波形分析対象: \(m.name) (\(m.path))")
        print("  ファイル長: \(m.sampleCount) サンプル (\(String(format: "%.3f", m.durationSec)) 秒)")
        print("  本体長: \(m.bodyFrameCount) フレーム (\(String(format: "%.3f", m.bodyDurationSec)) 秒)")
        print("  五つのゲート客観指標:")
        print("    1. 200–1500 Hz 重心変化中央値: \(String(format: "%.1f", m.centroidMedian1k5)) Hz")
        print("    2. 停止割合 (cos > 0.99):       \(String(format: "%.3f", m.stopRatio))")
        print("    3. 6–12 Hz 変調割合:           \(String(format: "%.4f", m.mod6to12Ratio))")
        print("    4. 前半 F0 中央値:             \(String(format: "%.1f", m.firstHalfF0)) Hz")
        print("    5. 後半 F0 中央値:             \(String(format: "%.1f", m.secondHalfF0)) Hz (差: \(String(format: "%+.1f", m.f0Diff)) Hz)")
        print("  深い落ち込み分析 (RMS < max(800, dipThresh), >= 80ms):")
        print("    本数: \(m.dips.count) 本, 最大長: \(m.maxDipLengthMs) ms")
        var dIdx = 0
        while dIdx < m.dips.count {
            let d = m.dips[dIdx]
            print("      [\(dIdx + 1)] 開始: \(String(format: "%.3f", d.startSec))s, 長さ: \(d.lengthMs)ms, 平均RMS: \(String(format: "%.1f", d.avgRms))")
            dIdx += 1
        }
        print("==================================================\n")
    }

    /// 有声フレームの隣接スペクトル余弦 > 0.99 割合およびスペクトル重心変化中央値を計測する
    func testSpectralCosineAndCentroid() throws {
        let files = [
            ("tts_tenki.wav", ".tmp/wave15/tts_tenki.wav"),
            ("tts_mizuwomare.wav", ".tmp/wave15/tts_mizuwomare.wav"),
            ("copy_BASIC5000_0001.wav", ".tmp/wave15/copy_BASIC5000_0001.wav")
        ]
        let reader = WavAudioReader()
        let extractor = MelSpectrogramExtractor(
            sampleRate: Float(AudioConfig.sampleRate),
            melChannels: AudioConfig.melChannels
        )
        let tracker = PitchTracker()

        for (name, path) in files {
            guard FileManager.default.fileExists(atPath: path) else {
                print("[\(name)] ファイルが存在しません: \(path)")
                continue
            }
            let pcm = try reader.loadWav16k(from: path)
            let pitch = tracker.track(pcm: pcm)
            let logMel = extractor.extractLogMel(pcm: pcm)

            var linMel = logMel
            var f = 0
            while f < linMel.count {
                var c = 0
                while c < AudioConfig.melChannels {
                    linMel[f][c] = expf(logMel[f][c])
                    c += 1
                }
                f += 1
            }

            let stftMag = extractor.extractLinearMagnitudeSpectrogram(pcm: pcm)
            let stftCentroids = extractor.computeSpectralCentroids(pcm: pcm)


            func runEval(spec: [[Float]], customCentroids: [Float]?, label: String) {
                var totalVoiced = 0
                var cosOver99 = 0
                var centroidDiffs: [Float] = []

                var t = 1
                let limit = min(spec.count, pitch.frameCount)
                while t < limit {
                    let isVoiced = (0.5 <= pitch.voiced[t] && 0.5 <= pitch.voiced[t - 1])
                    if isVoiced {
                        totalVoiced += 1
                        var dot: Float = 0.0
                        var normA: Float = 0.0
                        var normB: Float = 0.0
                        var c = 0
                        while c < spec[t].count {
                            let a = spec[t][c]
                            let b = spec[t - 1][c]
                            dot += a * b
                            normA += a * a
                            normB += b * b
                            c += 1
                        }
                        let denom = sqrtf(normA) * sqrtf(normB)
                        var cosSim: Float = 0.0
                        if 1e-6 < denom {
                            cosSim = dot / denom
                        }
                        if 0.99 < cosSim {
                            cosOver99 += 1
                        }

                        if let cents = customCentroids {
                            if t < cents.count {
                                centroidDiffs.append(abs(cents[t] - cents[t - 1]))
                            }
                        }
                    }
                    t += 1
                }

                centroidDiffs.sort()
                var medDiff: Float = 0.0
                if centroidDiffs.isEmpty != true {
                    medDiff = centroidDiffs[centroidDiffs.count / 2]
                }
                var ratio: Float = 0.0
                if 0 < totalVoiced {
                    ratio = Float(cosOver99) / Float(totalVoiced)
                }
                print("[\(name)] \(label): 有声数=\(totalVoiced), 余弦>0.99割合=\(String(format: "%.3f", ratio)), 重心変化中央値=\(String(format: "%.1f", medDiff)) Hz")
            }

            print("==================================================")
            print("計測対象: \(name) (\(pcm.count) サンプル, \(Float(pcm.count) / 16000.0) 秒)")
            runEval(spec: logMel, customCentroids: nil, label: "LogMel (64ch)")
            runEval(spec: linMel, customCentroids: nil, label: "LinMel (64ch)")
            runEval(spec: stftMag, customCentroids: stftCentroids, label: "STFT Mag (257 bins)")

            let obj = Self.measureObjectiveAcousticMetrics(pcm: pcm)
            print("[\(name)] 設計仕様基準 (窓512, Hann, 55%tile有声, 200-4000Hz重心): 有声対=\(obj.voicedCount), 余弦>0.99割合=\(String(format: "%.3f", obj.cosRatio)), 重心変化中央値=\(String(format: "%.1f", obj.centroidMedian)) Hz")
            let objDetail = Self.measureObjectiveAcousticMetricsDetail(pcm: pcm)
            print("[\(name)] 詳細基準: 有声対=\(objDetail.voicedCount), 余弦>0.99割合=\(String(format: "%.3f", objDetail.cosRatio)), 200-4000Hz重心変化=\(String(format: "%.1f", objDetail.centroidMedian200to4000)) Hz, 200-1500Hz重心変化=\(String(format: "%.1f", objDetail.centroidMedian200to1500)) Hz")

            let halves = AcousticCentroidMetric.measureHalves(pcm: pcm)
            let f0Full = AcousticCentroidMetric.measureF0Median(pcm: pcm)
            let mod6to12 = AcousticCentroidMetric.measureModulation6to12Ratio(pcm: pcm)
            print("[\(name)] 全体 F0 中央値=\(String(format: "%.1f", f0Full)) Hz")
            print("[\(name)] 前半: 停止割合=\(String(format: "%.3f", halves.first.cosRatio)), 重心=\(String(format: "%.1f", halves.first.centroidMedian200to1500)) Hz, F0=\(String(format: "%.1f", halves.first.f0Median)) Hz")
            print("[\(name)] 後半: 停止割合=\(String(format: "%.3f", halves.second.cosRatio)), 重心=\(String(format: "%.1f", halves.second.centroidMedian200to1500)) Hz, F0=\(String(format: "%.1f", halves.second.f0Median)) Hz")
            print("[\(name)] 6–12 Hz 割合: \(String(format: "%.4f", mod6to12))")
        }
    }

    /// design_cfc_closure.md に基づく閉鎖・パルス修正の客観指標および深い落ち込みの計測評価
    func testClosureDesignEvaluation() throws {
        let targets = [
            ("新しい水", "tts_mizuwomare.wav"),
            ("新しい天気", "tts_tenki.wav"),
            ("既存の copy", ".tmp/wave15/copy_BASIC5000_0001.wav")
        ]
        var tIdx = 0
        while tIdx < targets.count {
            let (label, path) = targets[tIdx]
            if let m = try measureClosureWav(path: path, name: label) {
                printClosureMetrics(m)
                switch label {
                case "新しい水":
                    break
                case "新しい天気":
                    XCTAssertEqual(m.dips.count, 2, "新しい天気の深い落ち込み本数は 2 本であること")
                    XCTAssertTrue(m.maxDipLengthMs <= 100, "新しい天気の深い落ち込み最大長は 100ms 以下であること: \(m.maxDipLengthMs)ms")
                case "既存の copy":
                    XCTAssertEqual(m.dips.count, 2, "既存 copy の深い落ち込み本数は 2 本であること")
                    XCTAssertTrue(m.maxDipLengthMs <= 130, "既存 copy の深い落ち込み最大長は 130ms 以下であること: \(m.maxDipLengthMs)ms")
                default:
                    break
                }
            } else {
                XCTFail("[\(label)] ファイルが存在しません: \(path)")
            }
            tIdx += 1
        }
    }

    /// design_cfc_closure.md 受入基準検証: ep05 継続時間、meanFramesPerMora=16、ch 199 ゼロ化の永続化確認
    func testApplyClosureUpdatesAndSynthesize() throws {
        let ep05URL = URL(fileURLWithPath: "Models/weights.ep05.json")
        let weightsURL = URL(fileURLWithPath: "Models/weights.json")
        guard FileManager.default.fileExists(atPath: ep05URL.path),
              FileManager.default.fileExists(atPath: weightsURL.path) else {
            XCTFail("重みファイルが見つかりません")
            return
        }

        let ep05Data = try Data(contentsOf: ep05URL)
        let ep05Weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: ep05Data)

        let curData = try Data(contentsOf: weightsURL)
        let curWeights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: curData)

        // 1. phonemeAverageDurations が ep05 と完全一致（キー集合および値）
        guard let ep05Durs = ep05Weights.phonemeAverageDurations,
              let curDurs = curWeights.phonemeAverageDurations else {
            XCTFail("phonemeAverageDurations が存在しません")
            return
        }
        XCTAssertEqual(curDurs.count, ep05Durs.count, "継続時間のキー数が ep05 と一致しません")
        for (k, v) in ep05Durs {
            guard let curVal = curDurs[k] else {
                XCTFail("キー \(k) が Models/weights.json に存在しません")
                return
            }
            XCTAssertEqual(curVal, v, accuracy: 1e-5, "キー \(k) の継続時間が ep05 と一致しません: \(curVal) vs \(v)")
        }

        // 2. meanFramesPerMora が 16.0
        XCTAssertEqual(curWeights.meanFramesPerMora, 16.0, "meanFramesPerMora が 16.0 ではありません")

        // 3. wIn は変えない
        XCTAssertEqual(curWeights.wIn, ep05Weights.wIn, "wIn は変えてはなりません")

        // 4. cfcWf[0] と cfcWg[0] の入力 5 および入力 199 L2 ノルムの確認、層 1 以降の一致確認
        guard let wf = curWeights.cfcWf, let wg = curWeights.cfcWg,
              0 < wf.count, 0 < wg.count else {
            XCTFail("cfcWf または cfcWg がありません")
            return
        }

        func l2ForInput(_ arr: [Float], _ inp: Int) -> Float {
            var sumSq: Float = 0.0
            var h = 0
            while h < 256 {
                let v = arr[(inp * 256) + h]
                sumSq += v * v
                h += 1
            }
            return sqrtf(sumSq)
        }

        let l2WfInp5 = l2ForInput(wf[0], 5)
        let l2WfInp199 = l2ForInput(wf[0], 199)
        let l2WgInp5 = l2ForInput(wg[0], 5)
        let l2WgInp199 = l2ForInput(wg[0], 199)

        XCTAssertTrue(abs(l2WfInp5 - 3.7) < 0.2, "cfcWf[0] input 5 L2 が約 3.7 ではありません: \(l2WfInp5)")
        XCTAssertTrue(abs(l2WgInp5 - 3.7) < 0.2, "cfcWg[0] input 5 L2 が約 3.7 ではありません: \(l2WgInp5)")
        XCTAssertEqual(l2WfInp199, 0.0, accuracy: 1e-6, "cfcWf[0] input 199 L2 が 0 ではありません: \(l2WfInp199)")
        XCTAssertEqual(l2WgInp199, 0.0, accuracy: 1e-6, "cfcWg[0] input 199 L2 が 0 ではありません: \(l2WgInp199)")

        if let epWf = ep05Weights.cfcWf, let epWg = ep05Weights.cfcWg {
            var l = 1
            while l < wf.count && l < epWf.count {
                XCTAssertEqual(wf[l], epWf[l], "層 \(l) の cfcWf は ep05 と一致すること")
                XCTAssertEqual(wg[l], epWg[l], "層 \(l) の cfcWg は ep05 と一致すること")
                l += 1
            }
        }

        // 5. 正規波形ファイルの存在および直下と .tmp/wave15/ の同一 PCM 検証
        let dataMizuRoot = try Data(contentsOf: URL(fileURLWithPath: "tts_mizuwomare.wav"))
        let dataMizuTmp = try Data(contentsOf: URL(fileURLWithPath: ".tmp/wave15/tts_mizuwomare.wav"))
        XCTAssertEqual(dataMizuRoot, dataMizuTmp, "水の直下と .tmp/wave15/ の波形 PCM データが完全一致すること")

        let dataTenkiRoot = try Data(contentsOf: URL(fileURLWithPath: "tts_tenki.wav"))
        let dataTenkiTmp = try Data(contentsOf: URL(fileURLWithPath: ".tmp/wave15/tts_tenki.wav"))
        XCTAssertEqual(dataTenkiRoot, dataTenkiTmp, "天気の直下と .tmp/wave15/ の波形 PCM データが完全一致すること")
    }

    /// 設計仕様書に基づく音響客観指標計測:
    /// （窓 512、ホップ 160、Hann、平均振幅が全体の 55 パーセンタイルを超えるフレームを有声、200–4000 Hz の重心）
    public static func measureObjectiveAcousticMetrics(pcm: [Float]) -> (cosRatio: Float, centroidMedian: Float, voicedCount: Int) {
        if pcm.isEmpty {
            return (0.0, 0.0, 0)
        }
        let winSize = 512
        let hopSize = 160
        let frameCount = max(1, (pcm.count - winSize) / hopSize)
        let sampleRate: Float = 16000.0

        var hann = [Float](repeating: 0.0, count: winSize)
        var n = 0
        while n < winSize {
            hann[n] = 0.5 * (1.0 - cosf((2.0 * Float.pi * Float(n)) / Float(winSize)))
            n += 1
        }

        let extractor = MelSpectrogramExtractor(
            sampleRate: sampleRate,
            melChannels: AudioConfig.melChannels,
            hopSize: hopSize,
            frameSize: winSize,
            fftSize: winSize
        )

        var frameAmplitudes = [Float](repeating: 0.0, count: frameCount)
        var frameMags = [[Float]](repeating: [Float](repeating: 0.0, count: 257), count: frameCount)
        var frameCentroids = [Float](repeating: 0.0, count: frameCount)

        let binHz = sampleRate / Float(winSize)
        let kMin = 7
        let kMax = 128

        var f = 0
        while f < frameCount {
            let sampleStart = f * hopSize
            var realBuf = [Float](repeating: 0.0, count: winSize)
            var imagBuf = [Float](repeating: 0.0, count: winSize)

            var ampSum: Float = 0.0
            var s = 0
            while s < winSize {
                let pcmIdx = sampleStart + s
                var sampleVal: Float = 0.0
                if pcmIdx < pcm.count {
                    sampleVal = pcm[pcmIdx]
                }
                ampSum += abs(sampleVal)
                realBuf[s] = sampleVal * hann[s]
                imagBuf[s] = 0.0
                s += 1
            }
            frameAmplitudes[f] = ampSum / Float(winSize)

            extractor.computeFFT(real: &realBuf, imag: &imagBuf)

            var num: Float = 0.0
            var den: Float = 0.0
            var b = 0
            while b <= 256 {
                let r = realBuf[b]
                let im = imagBuf[b]
                let mag = sqrtf(r * r + im * im)
                frameMags[f][b] = mag
                if kMin <= b && b <= kMax {
                    let freq = Float(b) * binHz
                    num += freq * mag
                    den += mag
                }
                b += 1
            }

            if 1e-6 < den {
                frameCentroids[f] = num / den
            }
            f += 1
        }

        let sortedAmps = frameAmplitudes.sorted()
        var p55Idx = Int(Float(frameCount) * 0.55)
        if frameCount <= p55Idx {
            p55Idx = frameCount - 1
        }
        let threshold55 = sortedAmps[p55Idx]

        var voicedPairs = 0
        var cosOver99Count = 0
        var cosAllOver99Count = 0
        var centroidDiffs: [Float] = []

        var t = 1
        while t < frameCount {
            let isVoicedCurr = (threshold55 < frameAmplitudes[t])
            let isVoicedPrev = (threshold55 < frameAmplitudes[t - 1])
            if isVoicedCurr && isVoicedPrev {
                voicedPairs += 1

                var dot: Float = 0.0
                var normA: Float = 0.0
                var normB: Float = 0.0
                var k = kMin
                while k <= kMax {
                    let a = frameMags[t][k]
                    let b = frameMags[t - 1][k]
                    dot += a * b
                    normA += a * a
                    normB += b * b
                    k += 1
                }
                let denom = sqrtf(normA) * sqrtf(normB)
                var cosSim: Float = 0.0
                if 1e-6 < denom {
                    cosSim = dot / denom
                }
                if 0.99 < cosSim {
                    cosOver99Count += 1
                }

                var dotAll: Float = 0.0
                var normAllA: Float = 0.0
                var normAllB: Float = 0.0
                var b = 0
                while b <= 256 {
                    let a = frameMags[t][b]
                    let bVal = frameMags[t - 1][b]
                    dotAll += a * bVal
                    normAllA += a * a
                    normAllB += bVal * bVal
                    b += 1
                }
                let denomAll = sqrtf(normAllA) * sqrtf(normAllB)
                var cosSimAll: Float = 0.0
                if 1e-6 < denomAll {
                    cosSimAll = dotAll / denomAll
                }
                if 0.99 < cosSimAll {
                    cosAllOver99Count += 1
                }

                centroidDiffs.append(abs(frameCentroids[t] - frameCentroids[t - 1]))
            }
            t += 1
        }

        centroidDiffs.sort()
        var medDiff: Float = 0.0
        if centroidDiffs.isEmpty != true {
            medDiff = centroidDiffs[centroidDiffs.count / 2]
        }
        var cosRatio: Float = 0.0
        if 0 < voicedPairs {
            cosRatio = Float(cosOver99Count) / Float(voicedPairs)
        }
        var cosAllRatio: Float = 0.0
        if 0 < voicedPairs {
            cosAllRatio = Float(cosAllOver99Count) / Float(voicedPairs)
        }
        print("  [詳細内訳] 有声対=\(voicedPairs), 200-4000Hz余弦>0.99=\(String(format: "%.3f", cosRatio)) (\(cosOver99Count)/\(voicedPairs)), 全帯域余弦>0.99=\(String(format: "%.3f", cosAllRatio)) (\(cosAllOver99Count)/\(voicedPairs)), 重心変化中央値=\(String(format: "%.1f", medDiff)) Hz")
        return (cosRatio: cosRatio, centroidMedian: medDiff, voicedCount: voicedPairs)
    }

    /// 設計仕様書（design_waveform_grad.md）に基づく詳細音響客観指標計測:
    /// （窓 512、ホップ 160、Hann、平均振幅が全体の 55 パーセンタイルを超えるフレームを有声）
    /// - 有声隣接余弦 0.99 超過割合
    /// - 200–4000 Hz 重心変化中央値
    /// - 200–1500 Hz フォルマント帯重心変化中央値
    public static func measureObjectiveAcousticMetricsDetail(pcm: [Float]) -> (
        cosRatio: Float,
        centroidMedian200to4000: Float,
        centroidMedian200to1500: Float,
        voicedCount: Int
    ) {
        if pcm.isEmpty {
            return (0.0, 0.0, 0.0, 0)
        }
        let winSize = 512
        let hopSize = 160
        let frameCount = max(1, (pcm.count - winSize) / hopSize)
        let sampleRate: Float = 16000.0

        var hann = [Float](repeating: 0.0, count: winSize)
        var n = 0
        while n < winSize {
            hann[n] = 0.5 * (1.0 - cosf((2.0 * Float.pi * Float(n)) / Float(winSize)))
            n += 1
        }

        let extractor = MelSpectrogramExtractor(
            sampleRate: sampleRate,
            melChannels: AudioConfig.melChannels,
            hopSize: hopSize,
            frameSize: winSize,
            fftSize: winSize
        )

        var frameAmplitudes = [Float](repeating: 0.0, count: frameCount)
        var frameMags = [[Float]](repeating: [Float](repeating: 0.0, count: 257), count: frameCount)
        var frameCentroids4k = [Float](repeating: 0.0, count: frameCount)
        var frameCentroids1k5 = [Float](repeating: 0.0, count: frameCount)

        let binHz = sampleRate / Float(winSize)
        let kMin = 7     // ~218.75 Hz
        let kMax4k = 128 // ~4000 Hz
        let kMax1k5 = 48 // ~1500 Hz

        var f = 0
        while f < frameCount {
            let sampleStart = f * hopSize
            var realBuf = [Float](repeating: 0.0, count: winSize)
            var imagBuf = [Float](repeating: 0.0, count: winSize)

            var ampSum: Float = 0.0
            var s = 0
            while s < winSize {
                let pcmIdx = sampleStart + s
                var sampleVal: Float = 0.0
                if pcmIdx < pcm.count {
                    sampleVal = pcm[pcmIdx]
                }
                ampSum += abs(sampleVal)
                realBuf[s] = sampleVal * hann[s]
                imagBuf[s] = 0.0
                s += 1
            }
            frameAmplitudes[f] = ampSum / Float(winSize)

            extractor.computeFFT(real: &realBuf, imag: &imagBuf)

            var num4k: Float = 0.0
            var den4k: Float = 0.0
            var num1k5: Float = 0.0
            var den1k5: Float = 0.0

            var b = 0
            while b <= 256 {
                let r = realBuf[b]
                let im = imagBuf[b]
                let mag = sqrtf(r * r + im * im)
                frameMags[f][b] = mag
                let freq = Float(b) * binHz
                if kMin <= b && b <= kMax4k {
                    num4k += freq * mag
                    den4k += mag
                }
                if kMin <= b && b <= kMax1k5 {
                    num1k5 += freq * mag
                    den1k5 += mag
                }
                b += 1
            }

            if 1e-6 < den4k {
                frameCentroids4k[f] = num4k / den4k
            }
            if 1e-6 < den1k5 {
                frameCentroids1k5[f] = num1k5 / den1k5
            }
            f += 1
        }

        let sortedAmps = frameAmplitudes.sorted()
        var p55Idx = Int(Float(frameCount) * 0.55)
        if frameCount <= p55Idx {
            p55Idx = frameCount - 1
        }
        let threshold55 = sortedAmps[p55Idx]

        var voicedPairs = 0
        var cosOver99Count = 0
        var centroidDiffs4k: [Float] = []
        var centroidDiffs1k5: [Float] = []

        var t = 1
        while t < frameCount {
            let isVoicedCurr = (threshold55 < frameAmplitudes[t])
            let isVoicedPrev = (threshold55 < frameAmplitudes[t - 1])
            if isVoicedCurr && isVoicedPrev {
                voicedPairs += 1

                var dot: Float = 0.0
                var normA: Float = 0.0
                var normB: Float = 0.0
                var k = kMin
                while k <= kMax4k {
                    let a = frameMags[t][k]
                    let b = frameMags[t - 1][k]
                    dot += a * b
                    normA += a * a
                    normB += b * b
                    k += 1
                }
                let denom = sqrtf(normA) * sqrtf(normB)
                var cosSim: Float = 0.0
                if 1e-6 < denom {
                    cosSim = dot / denom
                }
                if 0.99 < cosSim {
                    cosOver99Count += 1
                }

                centroidDiffs4k.append(abs(frameCentroids4k[t] - frameCentroids4k[t - 1]))
                centroidDiffs1k5.append(abs(frameCentroids1k5[t] - frameCentroids1k5[t - 1]))
            }
            t += 1
        }

        centroidDiffs4k.sort()
        var medDiff4k: Float = 0.0
        if centroidDiffs4k.isEmpty != true {
            medDiff4k = centroidDiffs4k[centroidDiffs4k.count / 2]
        }

        centroidDiffs1k5.sort()
        var medDiff1k5: Float = 0.0
        if centroidDiffs1k5.isEmpty != true {
            medDiff1k5 = centroidDiffs1k5[centroidDiffs1k5.count / 2]
        }

        var cosRatio: Float = 0.0
        if 0 < voicedPairs {
            cosRatio = Float(cosOver99Count) / Float(voicedPairs)
        }

        print("  [詳細内訳] 有声対=\(voicedPairs), 余弦>0.99=\(String(format: "%.3f", cosRatio)), 200-4000Hz重心変化中央値=\(String(format: "%.1f", medDiff4k)) Hz, 200-1500Hz重心変化中央値=\(String(format: "%.1f", medDiff1k5)) Hz")
        return (
            cosRatio: cosRatio,
            centroidMedian200to4000: medDiff4k,
            centroidMedian200to1500: medDiff1k5,
            voicedCount: voicedPairs
        )
    }

#if canImport(MLX)
    /// 波形損失勾配受入ゲート（ゲート 1〜4）の自動評価テスト
    func testWaveformGradGates() throws {
        let weightsPath = "Models/weights.json"
        let vocoderWeightsPath = "Models/vocoder_weights.json"
        guard FileManager.default.fileExists(atPath: weightsPath),
              FileManager.default.fileExists(atPath: vocoderWeightsPath) else {
            print("重みファイルが存在しません")
            return
        }

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)

        let vocData = try Data(contentsOf: URL(fileURLWithPath: vocoderWeightsPath))
        let vocWeights = try JSONDecoder().decode(NeuralVocoderWeights.self, from: vocData)

        let network = MLXSpikingAcousticNetwork(weights: weights)
        let vocoder = MLXNeuralVocoder(weights: vocWeights)
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

        let text = "水をマレーシアから買わなくてはならないのです。"
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        guard FileManager.default.fileExists(atPath: wavPath) else {
            print("BASIC5000_0001.wav が存在しません: \(wavPath)")
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
        let pitchTracker = PitchTracker()

        var effectiveAlign: UtteranceAlignment? = nil
        let corpusAlignPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/mas_alignments.json"
        if FileManager.default.fileExists(atPath: corpusAlignPath) {
            if let alignMap = try? AlignmentStore.load(from: corpusAlignPath) {
                effectiveAlign = alignMap["BASIC5000_0001"]
            }
        }

        guard let pair = engine.prepareTrainingPair(
            text: text,
            pcm16k: pcm16k,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker,
            alignment: effectiveAlign,
            useScaledDuration: false
        ) else {
            XCTFail("prepareTrainingPair 失敗")
            return
        }

        let res = WaveformGradGateEvaluator.evaluate(
            network: network,
            vocoder: vocoder,
            features: pair.features,
            targets: pair.targets,
            targetAudio: pair.targetAudio,
            waveformLossWeight: 0.15,
            bpttWindow: 1
        )

        let strPass1: String
        switch res.gate1Passed {
        case true: strPass1 = "PASS"
        case false: strPass1 = "FAIL"
        }
        let strPass2: String
        switch res.gate2Passed {
        case true: strPass2 = "PASS"
        case false: strPass2 = "FAIL"
        }
        let strPass3: String
        switch res.gate3Passed {
        case true: strPass3 = "PASS"
        case false: strPass3 = "FAIL"
        }
        let strPass4: String
        switch res.gate4Passed {
        case true: strPass4 = "PASS"
        case false: strPass4 = "FAIL"
        }
        let strAllPassed: String
        switch res.allPassed {
        case true: strAllPassed = "ALL PASSED"
        case false: strAllPassed = "FAILED"
        }

        print("==================================================")
        print("波形損失勾配受入ゲート（ゲート 1〜4）検証結果:")
        print("  区間開始フレーム: \(res.segStart)")
        print("  ゲート 1 (スライス最大絶対差): \(res.maxSliceDiff) -> \(strPass1)")
        print("  ゲート 2 (STFT損失): 教師=\(res.teacherSTFTLoss) vs 予測=\(res.predSTFTLoss) -> \(strPass2)")
        print("  ゲート 3 (Mel勾配平均絶対値): tanh=\(res.melGradMeanAbsTanh), preTanh=\(res.melGradMeanAbsPreTanh), 飽和率=\(res.outputSaturationRatio), usePreTanh=\(res.usePreTanh) -> \(strPass3)")
        print("  ゲート 4 (wOut 勾配ノルム比率): Mel=\(res.wOutMelGradNorm), Wave=\(res.wOutWaveGradNorm), 比率=\(res.gradNormRatio), 推奨係数=\(res.recommendedWeight) -> \(strPass4)")
        print("  総合判定: \(strAllPassed)")
        print("==================================================")

        XCTAssertTrue(res.gate1Passed, "ゲート 1 (スライス整合性) に失敗")
        XCTAssertTrue(res.gate2Passed, "ゲート 2 (ボコーダ教師STFT損失 < 予測STFT損失) に失敗")
        XCTAssertTrue(res.gate3Passed, "ゲート 3 (Mel勾配平均絶対値 > 1e-4) に失敗")
        XCTAssertTrue(res.gate4Passed, "ゲート 4 (wOut 勾配ノルム比率 0.2〜5.0) に失敗")
        XCTAssertTrue(res.allPassed, "受入ゲート総合判定に失敗")
    }
#endif

    func testProbeProsodyVsTrackerF0() throws {
        let outputDir = ".tmp/wave15"
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "Models/weights.json"))
        let weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: weightsData)
        var vocWeights: NeuralVocoderWeights? = nil
        if let vData = try? Data(contentsOf: URL(fileURLWithPath: "Models/vocoder_weights.json")) {
            vocWeights = try? JSONDecoder().decode(NeuralVocoderWeights.self, from: vData)
        }
        let engine = SpikeSpeechEngine(weights: weights, vocoderWeights: vocWeights)

        let probeText = "水をマレーシアから買わなくてはならないのです。"
        let wavPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
        if FileManager.default.fileExists(atPath: wavPath) != true {
            throw XCTSkip("BASIC5000_0001.wav が存在しないためスキップします: \(wavPath)")
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
            useScaledDuration: false
        ) else {
            XCTFail("prepareTrainingPair に失敗しました")
            return
        }

        // プローブ A: synthesize(text:) と同一パイプライン
        let rawSamplesA = engine.synthesize(text: probeText)
        let pathA = ".tmp/wave15/probe_prosody_f0.wav"
        let dataA = WavEncoder.encode(samples: rawSamplesA, sampleRate: Int(engine.sampleRate))
        try? dataA.write(to: URL(fileURLWithPath: pathA))
        let metricA = AcousticCentroidMetric.measure(pcm: rawSamplesA)
        let centroid1k5A = metricA.centroidMedian200to1500

        // プローブ B: A と同じ特徴行列の ch 194/195 のみを教師 PitchTracker F0 に置換、ボコーダへは A の prosody を渡す
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
        try? dataB.write(to: URL(fileURLWithPath: pathB))
        let metricB = AcousticCentroidMetric.measure(pcm: rawSamplesB)
        let centroid1k5B = metricB.centroidMedian200to1500

        print("プローブ A 200–1500 Hz: \(centroid1k5A) Hz")
        print("プローブ B 200–1500 Hz: \(centroid1k5B) Hz")

        // 仕様書判定: B < 22Hz のため波形分岐が確定
        XCTAssertTrue(centroid1k5B < 22.0, "プローブ B の 200–1500 Hz 重心は 22 Hz 未満であること（実測: \(centroid1k5B) Hz）")
        XCTAssertTrue(0 < rawSamplesA.count, "プローブ A サンプルが空でないこと")
        XCTAssertTrue(0 < rawSamplesB.count, "プローブ B サンプルが空でないこと")
    }

    /// design_envelope_keep19.md の「0. 作業前」受入条件 4 の事前検証テスト
    /// 水の包絡変化が 40–46 Hz、水の一致割合が 0.48–0.54、copy の包絡変化が 70–78 Hz、copy の一致割合が 0.24–0.30
    func testEnvelopeDeltaPreflightValidation() throws {
        let ep19URL = URL(fileURLWithPath: "Models/weights.frame_mel_ep19.json")
        let ep19Weights = try SpikingNetworkWeights.load(from: ep19URL)
        let ep19Engine = SpikeSpeechEngine(weights: ep19Weights)
        let mizuPCM = ep19Engine.synthesize(text: "水をマレーシアから買わなくてはならないのです。")
        let tenkiPCM = ep19Engine.synthesize(text: "今日はいい天気です")

        let reader = WavAudioReader()
        let copyPCM = try reader.loadWav16k(from: ".tmp/wave15/copy_BASIC5000_0001.wav")

        let mizuEnv = AcousticCentroidMetric.measureEnvelopeDelta(pcm: mizuPCM)
        let tenkiEnv = AcousticCentroidMetric.measureEnvelopeDelta(pcm: tenkiPCM)
        let copyEnv = AcousticCentroidMetric.measureEnvelopeDelta(pcm: copyPCM)

        print("[Envelope Delta Preflight Check]")
        print("  水 (ep19):   包絡変化 = \(String(format: "%.2f", mizuEnv.centroidMedian)) Hz, 一致割合 = \(String(format: "%.4f", mizuEnv.matchRatio)), 有声対 = \(mizuEnv.voicedPairs)")
        print("  天気 (ep19): 包絡変化 = \(String(format: "%.2f", tenkiEnv.centroidMedian)) Hz, 一致割合 = \(String(format: "%.4f", tenkiEnv.matchRatio)), 有声対 = \(tenkiEnv.voicedPairs)")
        print("  copy:        包絡変化 = \(String(format: "%.2f", copyEnv.centroidMedian)) Hz, 一致割合 = \(String(format: "%.4f", copyEnv.matchRatio)), 有声対 = \(copyEnv.voicedPairs)")

        // 水の包絡変化が 40–46 Hz、水の一致割合が 0.48–0.54
        XCTAssertTrue(40.0 <= mizuEnv.centroidMedian && mizuEnv.centroidMedian <= 46.0, "作業前水の包絡変化は 40–46 Hz であること（実測: \(mizuEnv.centroidMedian) Hz）")
        XCTAssertTrue(0.48 <= mizuEnv.matchRatio && mizuEnv.matchRatio <= 0.54, "作業前水の一致割合は 0.48–0.54 であること（実測: \(mizuEnv.matchRatio)）")

        // copy の包絡変化が 70–78 Hz、copy の一致割合が 0.24–0.30
        XCTAssertTrue(70.0 <= copyEnv.centroidMedian && copyEnv.centroidMedian <= 78.0, "copy の包絡変化は 70–78 Hz であること（実測: \(copyEnv.centroidMedian) Hz）")
        XCTAssertTrue(0.24 <= copyEnv.matchRatio && copyEnv.matchRatio <= 0.30, "copy の一致割合は 0.24–0.30 であること（実測: \(copyEnv.matchRatio)）")
    }
}



