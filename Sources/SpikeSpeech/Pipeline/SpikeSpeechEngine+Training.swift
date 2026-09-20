import Foundation

extension SpikeSpeechEngine {

    /// 音声波形の短時間 RMS に基づく発話有音区間および前後無音フレーム数の検出
    ///
    /// なぜ固定フレーム切り出しではなく VAD 境界検出を行うか:
    /// 音声録音には発話前後に任意長の無音（環境ノイズ）が存在し、無音部を一律に言語特徴量へ割り当てると
    /// 先頭音素や末尾音素に無音区間のスペクトルが強制アライメントされ、音響モデルが重度のスペクトル歪みを学習してしまうため。
    public func detectSpeechBoundaries(
        pcm: [Float],
        hopSize: Int = AudioConfig.hopSize,
        totalFrames: Int
    ) -> (leadSilence: Int, speechFrames: Int, trailSilence: Int) {
        if totalFrames <= 0 {
            return (leadSilence: 0, speechFrames: 0, trailSilence: 0)
        }

        var frameRms = [Float](repeating: 0.0, count: totalFrames)
        var maxRms: Float = 0.0
        var f = 0
        while f < totalFrames {
            let start = f * hopSize
            var sumSq: Float = 0.0
            var count = 0
            var s = 0
            while s < hopSize {
                let pcmIdx = start + s
                if pcmIdx < pcm.count {
                    let v = pcm[pcmIdx]
                    sumSq += v * v
                    count += 1
                }
                s += 1
            }
            let rms: Float
            if 0 < count {
                rms = sqrtf(sumSq / Float(count))
            } else {
                rms = 0.0
            }
            frameRms[f] = rms
            if maxRms < rms {
                maxRms = rms
            }
            f += 1
        }

        // 発話内ピーク RMS の 8% を無音／有音の物理的境界閾値とする
        let threshold = max(0.015, maxRms * 0.08)

        var firstSpeech = -1
        var lastSpeech = -1
        var sf = 0
        while sf < totalFrames {
            if threshold <= frameRms[sf] {
                if firstSpeech < 0 {
                    firstSpeech = sf
                }
                lastSpeech = sf
            }
            sf += 1
        }

        if firstSpeech < 0 {
            return (leadSilence: 0, speechFrames: totalFrames, trailSilence: 0)
        }

        let margin = 3 // 語頭・語尾の微弱子音（破裂音・摩擦音）を保護する 30ms マージン
        let leadSilence = max(0, firstSpeech - margin)
        let endFrame = min(totalFrames, lastSpeech + 1 + margin)
        let speechFrames = max(1, endFrame - leadSilence)
        let trailSilence = max(0, totalFrames - (leadSilence + speechFrames))

        return (leadSilence: leadSilence, speechFrames: speechFrames, trailSilence: trailSilence)
    }

    /// テキストと音声波形から VAD アライメント・目標 Mel 系列ペアを生成する唯一の正本メソッド
    ///
    /// なぜ本メソッドを唯一の正本として一元化するか:
    /// 学習スクリプト（train CLI、dataset CLI、単体テスト）ごとに目標特徴量やアライメントの計算が分散すると、
    /// 重みスライスやスケール、エネルギー分布の不一致が再発するため、
    /// 推論エンジンと同一の数理基盤（Length Regulator、Vocab）から 1 箇所で生成する。
    public func prepareTrainingPair(
        text: String,
        pcm16k: [Float],
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

        let boundaries = detectSpeechBoundaries(
            pcm: pcm16k,
            hopSize: AudioConfig.hopSize,
            totalFrames: targetFrames
        )
        let leadSilence = boundaries.leadSilence
        let speechFrames = boundaries.speechFrames
        let trailSilence = boundaries.trailSilence

        let morphemes = normalizer.normalize(text: text)
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
        if phrases.isEmpty {
            return nil
        }

        var phoneIds: [Int32] = []
        var pIdx = 0
        while pIdx < phrases.count {
            var mIdx = 0
            while mIdx < phrases[pIdx].moras.count {
                var phIdx = 0
                while phIdx < phrases[pIdx].moras[mIdx].phonemes.count {
                    phoneIds.append(Int32(phrases[pIdx].moras[mIdx].phonemes[phIdx].id))
                    phIdx += 1
                }
                mIdx += 1
            }
            pIdx += 1
        }

        let phoneCount = phoneIds.count
        if speechFrames <= 0 || speechFrames < phoneCount {
            return nil
        }

        // なぜ音響エネルギー同期アライメント器（AcousticEnergyAligner）を唯一の正本とするか:
        // 発話区間をモーラ数で等時間割り（speechFrames / M）すると、実音声波形で子音・母音が鳴る時刻と
        // 音素ラベルの時刻が物理的に乖離し、SNN が誤ったスペクトルを学習して声質だけ残り言葉が消滅するため。
        // 短時間平滑化エネルギーの累積等分および局所エネルギー谷スナップにより、WAV の実音声エネルギー境界で
        // 音素フレーム数を物理的に決定し、教師 Mel と厳密に完全同期させる。
        let energyAligner = AcousticEnergyAligner()
        guard let speechDurations = energyAligner.alignPhonemes(
            pcm: pcm16k,
            hopSize: AudioConfig.hopSize,
            leadSilence: leadSilence,
            speechFrames: speechFrames,
            phrases: phrases,
            lengthRegulator: lengthRegulator
        ) else {
            return nil
        }

        var alignedFeatures: [[Float]]
        var fullPhoneIds: [Int32] = []
        var fullDurations: [Int] = []

        if 0 < leadSilence {
            fullPhoneIds.append(Int32(PhonemeVocabulary.silId))
            fullDurations.append(leadSilence)
        }

        var bIdx = 0
        while bIdx < phoneCount {
            fullPhoneIds.append(phoneIds[bIdx])
            fullDurations.append(speechDurations[bIdx])
            bIdx += 1
        }

        if 0 < trailSilence {
            fullPhoneIds.append(Int32(PhonemeVocabulary.silId))
            fullDurations.append(trailSilence)
        }

        // Pure Swift PitchTracker による実音声からの実測 F0、有声度、および実測短時間 RMS 抽出
        let pitchResult = pitchTracker.track(pcm: pcm16k)

        // なぜ学習データ構築時に発話ピーク正規化を行うか:
        // 実録音のゲインばらつきを吸収し、発話内ピーク（有声母音）を正確に 0.80（推論側の母音エネルギー 0.80）
        // にスケーリングすることで、推論時のエネルギー条件付け特徴量（ch70）の確率分布と 1 対 1 で整合させるため。
        var maxEnergy: Float = 0.0
        var ef = 0
        while ef < pitchResult.frameCount {
            if maxEnergy < pitchResult.energy[ef] {
                maxEnergy = pitchResult.energy[ef]
            }
            ef += 1
        }
        var normScale: Float = 1.0
        if 0.01 < maxEnergy {
            normScale = 0.80 / maxEnergy
        }

        var alignedF0 = [Float](repeating: 0.0, count: targetFrames)
        var alignedVoiced = [Float](repeating: 0.0, count: targetFrames)
        var alignedEnergy = [Float](repeating: 0.0, count: targetFrames)
        var f = 0
        while f < targetFrames {
            if f < pitchResult.frameCount {
                alignedF0[f] = pitchResult.f0[f]
                alignedVoiced[f] = pitchResult.voiced[f]
                let scaledVal = pitchResult.energy[f] * normScale
                if 1.0 < scaledVal {
                    alignedEnergy[f] = 1.0
                } else {
                    alignedEnergy[f] = scaledVal
                }
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

        alignedFeatures = encodeLinguisticFeatures(
            features: alignedLinguistic
        )

        let finalCount = min(alignedFeatures.count, targetMel.count)
        var safeFeatures = alignedFeatures
        if finalCount < safeFeatures.count {
            safeFeatures.removeSubrange(finalCount..<safeFeatures.count)
        }

        var safeTargets = [[Float]](repeating: [Float](repeating: 0.0, count: AudioConfig.melChannels), count: finalCount)

        // 目標 Mel 系列を実音声の絶対対数 Mel スペクトル（targetMel）として直接設定
        // なぜ事前知識の残差学習を完全撤廃するか:
        // SNN 音響モデルが実音声データの絶対対数 Mel スペクトルを直接予測するように学習することで、
        // 不自然なロボット感・機械的歪みを排し、
        // ニューラルボコーダーの学習 Mel 分布と推論 Mel 分布を完全に一致させるため。
        let melCh = AudioConfig.melChannels
        var t = 0
        while t < finalCount {
            safeTargets[t].withUnsafeMutableBufferPointer { pDst in
                targetMel[t].withUnsafeBufferPointer { pSrc in
                    pDst.baseAddress!.update(from: pSrc.baseAddress!, count: melCh)
                }
            }
            t += 1
        }

        return (features: safeFeatures, targets: safeTargets)
    }

    /// テキストと実音声波形から韻律予測器（Duration & F0）学習用サンプルを抽出する
    public func prepareProsodyTrainingSample(
        text: String,
        pcm16k: [Float],
        pitchTracker: PitchTracker
    ) -> ProsodyTrainingSample? {
        if pcm16k.isEmpty {
            return nil
        }
        let targetFrames = pcm16k.count / AudioConfig.hopSize
        if targetFrames <= 0 {
            return nil
        }

        let morphemes = normalizer.normalize(text: text)
        let phrases = prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
        if phrases.isEmpty {
            return nil
        }

        let boundaries = detectSpeechBoundaries(
            pcm: pcm16k,
            hopSize: AudioConfig.hopSize,
            totalFrames: targetFrames
        )
        let speechFrames = boundaries.speechFrames

        let baseLinguistic = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            applyFluctuation: false,
            addBoundarySilence: false
        )
        let origTotalFrames = baseLinguistic.totalFrames
        let phoneCount = baseLinguistic.phoneIds.count

        if speechFrames <= 0 || speechFrames < phoneCount || origTotalFrames <= 0 {
            return nil
        }

        let stretchRatio = Float(speechFrames) / Float(max(1, origTotalFrames))
        var scaledDurations = [Float](repeating: 0.0, count: phoneCount)
        var p = 0
        while p < phoneCount {
            let origD = Float(baseLinguistic.durations[p])
            scaledDurations[p] = max(1.0, origD * stretchRatio)
            p += 1
        }
        let quantizedDurations = lengthRegulator.quantizeDurations(durations: scaledDurations)
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
            var roundIdx = speechDurations.count - 1
            var consecutiveFailures = 0
            while 0 < remaining && 0 <= roundIdx {
                let dIdx = roundIdx % speechDurations.count
                if 1 < speechDurations[dIdx] {
                    speechDurations[dIdx] -= 1
                    remaining -= 1
                    consecutiveFailures = 0
                } else {
                    consecutiveFailures += 1
                    if speechDurations.count <= consecutiveFailures {
                        break
                    }
                }
                roundIdx -= 1
                if roundIdx < 0 && 0 < remaining {
                    roundIdx = speechDurations.count - 1
                }
            }
        }
        if 0 < diff && 0 < speechDurations.count {
            var remainingDiff = diff
            var roundIdx = 0
            while 0 < remainingDiff {
                let dIdx = roundIdx % speechDurations.count
                speechDurations[dIdx] += 1
                remainingDiff -= 1
                roundIdx += 1
            }
        }

        // 量子化された音素長をアクセント句・モーラ・音素構造に反映する
        // なぜ durationFrames を反映するか:
        // buildAccentPhrases 直後の音素は durationFrames=0 で初期化されており、
        // これを更新しないと generateF0Contour が空の F0 輪郭を返してしまい、
        // 有声フレームの有声 F0 MAE 学習が完全にゼロマスクされて学習不能となるため。
        var updatedPhrases = phrases
        var phCount = 0
        var upIdx = 0
        while upIdx < updatedPhrases.count {
            var umIdx = 0
            while umIdx < updatedPhrases[upIdx].moras.count {
                var uphIdx = 0
                while uphIdx < updatedPhrases[upIdx].moras[umIdx].phonemes.count {
                    if phCount < speechDurations.count {
                        updatedPhrases[upIdx].moras[umIdx].phonemes[uphIdx].durationFrames = speechDurations[phCount]
                        phCount += 1
                    }
                    uphIdx += 1
                }
                umIdx += 1
            }
            upIdx += 1
        }

        // 1. 各音素の Duration 特徴量と教師 Duration
        var durFeats = [[Float]]()
        var ruleDurs = [Float]()
        var targetDurs = [Float]()

        var pIdx = 0
        var speechPhonemeIdx = 0
        while pIdx < updatedPhrases.count {
            let phrase = updatedPhrases[pIdx]
            var mIdx = 0
            while mIdx < phrase.moras.count {
                let mora = phrase.moras[mIdx]
                let isLastMora = (mIdx + 1) == phrase.moras.count
                var phIdx = 0
                while phIdx < mora.phonemes.count {
                    let token = mora.phonemes[phIdx]
                    let isLastPhoneme = (phIdx + 1) == mora.phonemes.count
                    let ruleDur = lengthRegulator.floatDurationFrames(category: token.category, symbol: token.symbol, speed: 1.0)

                    let feat = ProsodyPredictor.extractDurationFeatures(
                        token: token,
                        mora: mora,
                        phrase: phrase,
                        isLastMoraInPhrase: isLastMora,
                        isLastPhonemeInMora: isLastPhoneme,
                        ruleDuration: ruleDur,
                        inputDim: 72
                    )

                    var targDur = ruleDur
                    if speechPhonemeIdx < speechDurations.count {
                        targDur = Float(speechDurations[speechPhonemeIdx])
                    }

                    durFeats.append(feat)
                    ruleDurs.append(ruleDur)
                    targetDurs.append(targDur)

                    speechPhonemeIdx += 1
                    phIdx += 1
                }
                mIdx += 1
            }
            pIdx += 1
        }

        // 2. 実測 PitchTracker F0 と藤崎規則 F0
        let pitchResult = pitchTracker.track(pcm: pcm16k)
        var bio = BiologicalFluctuation(seed: 2026)
        let (fujisakiF0, _, totalF) = prosodyModel.generateF0Contour(
            phrases: updatedPhrases,
            vocabulary: vocabulary,
            baseF0: VoiceProfile.female.baseF0,
            fluctuation: &bio,
            applyFluctuation: false
        )

        var f0Feats = [[Float]]()
        var fujiF0s = [Float]()
        var targF0s = [Float]()
        var vMasks = [Float]()

        let leadSilence = boundaries.leadSilence
        var curF = 0
        pIdx = 0
        while pIdx < updatedPhrases.count {
            let phrase = updatedPhrases[pIdx]
            let phraseTotalFrames = phrase.moras.reduce(0) { total, mora in
                total + mora.phonemes.reduce(0) { $0 + $1.durationFrames }
            }
            let phraseStartF = curF

            var mIdx = 0
            while mIdx < phrase.moras.count {
                let mora = phrase.moras[mIdx]
                let isHigh = (mora.tone == .high)

                var phIdx = 0
                while phIdx < mora.phonemes.count {
                    let token = mora.phonemes[phIdx]
                    let dur = max(1, token.durationFrames)
                    let isVoiced = vocabulary.isVoiced(symbol: token.symbol)

                    var f = 0
                    while f < dur {
                        let fujiIdx = curF + f
                        let audioFrameIdx = leadSilence + curF + f

                        var baseF: Float = 0.0
                        if fujiIdx < fujisakiF0.count {
                            baseF = fujisakiF0[fujiIdx]
                        }

                        var actualF0: Float = 0.0
                        var actualVoiced: Float = 0.0
                        if audioFrameIdx < pitchResult.frameCount {
                            actualF0 = pitchResult.f0[audioFrameIdx]
                            actualVoiced = pitchResult.voiced[audioFrameIdx]
                        }

                        // 境界付近（±2フレーム）の微小アライメントズレ耐性
                        // なぜ近傍探索を行うか:
                        // 規則長からの線形伸縮と実音声の発音タイミングの間には 10〜20ms（1〜2フレーム）の微小な物理的ズレがあり、
                        // 1 点サンプリングでは母音の立ち上がりで無声フレーム（0 Hz）を誤って拾ったり mask=0 となって
                        // 学習サンプルが脱落・汚染されるのを防ぎ、真の有声 F0 目標値を安定して捕捉するため。
                        if isVoiced && (actualVoiced < 0.5 || actualF0 < 120.0 || 420.0 < actualF0) {
                            var offset = -2
                            while offset <= 2 {
                                let candIdx = audioFrameIdx + offset
                                if 0 <= candIdx && candIdx < pitchResult.frameCount {
                                    let candVoiced = pitchResult.voiced[candIdx]
                                    let candF0 = pitchResult.f0[candIdx]
                                    if 0.5 <= candVoiced && 120.0 <= candF0 && candF0 <= 420.0 {
                                        actualF0 = candF0
                                        actualVoiced = candVoiced
                                        break
                                    }
                                }
                                offset += 1
                            }
                        }

                        let pProg = Float(f) / Float(dur)
                        let phraseFrame = (curF - phraseStartF) + f
                        let phProg = Float(phraseFrame) / Float(max(1, phraseTotalFrames))
                        let sProg = Float(fujiIdx) / Float(max(1, totalF))
                        let isAccentNucleus = mora.isAccentKernel
                        let isFlatAccent = phrase.moras.allSatisfy { $0.isAccentKernel != true }
                        let isVowel = (token.category == .vowel)

                        let feat = ProsodyPredictor.extractF0Features(
                            phoneId: token.id,
                            phonemeProgress: pProg,
                            duration: dur,
                            fujisakiF0: baseF,
                            isHighTone: isHigh,
                            phraseProgress: phProg,
                            isVoiced: isVoiced,
                            sentenceProgress: sProg,
                            isQuestion: phrase.isQuestion,
                            isAccentNucleus: isAccentNucleus,
                            isFlatAccent: isFlatAccent,
                            isVowel: isVowel,
                            inputDim: 76
                        )

                        var mask: Float = 0.0
                        if isVoiced && 0.5 <= actualVoiced && 120.0 <= actualF0 && actualF0 <= 420.0 {
                            mask = 1.0
                        }

                        f0Feats.append(feat)
                        fujiF0s.append(baseF)
                        targF0s.append(actualF0)
                        vMasks.append(mask)

                        f += 1
                    }
                    curF += dur
                    phIdx += 1
                }
                mIdx += 1
            }
            if phrase.pauseAfter && 0 < phrase.pauseDurationFrames {
                curF += phrase.pauseDurationFrames
            }
            pIdx += 1
        }

        return ProsodyTrainingSample(
            durationFeatures: durFeats,
            ruleDurations: ruleDurs,
            targetDurations: targetDurs,
            f0Features: f0Feats,
            fujisakiF0: fujiF0s,
            targetF0: targF0s,
            voicedMask: vMasks
        )
    }
}
