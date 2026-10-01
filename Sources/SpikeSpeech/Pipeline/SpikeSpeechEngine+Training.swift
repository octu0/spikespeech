import Foundation

extension SpikeSpeechEngine {

    /// 音声波形の短時間 RMS に基づく発話有音区間および前後無音フレーム数の検出
    ///
    /// なぜ固定フレーム切り出しではなく VAD 境界検出を行うか:
    /// 音声録音には発話前後に任意長の無音（環境ノイズ）が存在し、無音部を一律に言語特徴量へ割り当てると
    /// 先頭音素や末尾音素に無音区間のスペクトルが強制アライメントされ、音響モデルが重度のスペクトル歪みを学習してしまうため。
    public static func detectSpeechBoundaries(
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

        // 発話内ピーク RMS の 6% を無音／有音の物理的境界閾値とする
        let threshold = max(0.010, maxRms * 0.06)

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

        let leadMargin = 5 // 語頭の微弱子音（破裂音等）を保護する 50ms マージン
        let trailMargin = 1 // 語尾の摩擦音を保護する 10ms マージン
        let leadSilence = max(0, firstSpeech - leadMargin)
        let endFrame = min(totalFrames, lastSpeech + 1 + trailMargin)
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
        pitchTracker: PitchTracker,
        alignment: UtteranceAlignment? = nil,
        useScaledDuration: Bool = true
    ) -> (features: [[Float]], targets: [[Float]])? {
        if pcm16k.isEmpty {
            return nil
        }
        let targetMel = melExtractor.extractLogMel(pcm: pcm16k)
        let origFrames = targetMel.count
        if origFrames <= 0 {
            return nil
        }

        let boundaries = Self.detectSpeechBoundaries(
            pcm: pcm16k,
            hopSize: AudioConfig.hopSize,
            totalFrames: origFrames
        )
        let leadSilence = boundaries.leadSilence
        let actualSpeechFrames = boundaries.speechFrames
        let trailSilence = boundaries.trailSilence
        if actualSpeechFrames <= 0 {
            return nil
        }

        // 1. 学習入力は合成（synthesize）と全く同一の過程で生成する（設計書 1 項）
        let linguisticFeatures = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            prosodyPredictor: prosodyPredictor,
            speedFactor: 1.0,
            baseF0: VoiceProfile.default.baseF0,
            addBoundarySilence: true,
            meanFramesPerMora: VoiceProfile.default.meanFramesPerMora
        )

        let totalFrames = linguisticFeatures.totalFrames
        if totalFrames <= 0 {
            return nil
        }

        // 実測 F0・有声度・エネルギーを注入せず、processText の出力のみから特徴量を符号化
        let features = encodeLinguisticFeatures(
            features: linguisticFeatures
        )
        if features.count != totalFrames {
            return nil
        }

        // 2. 教師の対数 Mel は processText が切った音素区間の上に載せる（設計書 2 項）
        let phoneCount = linguisticFeatures.phoneIds.count
        if phoneCount < 3 {
            return nil
        }

        let leadSil = Int(linguisticFeatures.durations[0])
        let trailSil = Int(linguisticFeatures.durations[phoneCount - 1])
        let bodyPhoneCount = phoneCount - 2
        let targetSpeechFrames = totalFrames - leadSil - trailSil
        if targetSpeechFrames <= 0 {
            return nil
        }

        // 各音素のソース実測フレーム数を取得（MAS アライメントまたはデータ駆動音素平均比率）
        let effectiveLeadSilence: Int
        let effectiveSpeechFrames: Int
        let effectiveTrailSilence: Int
        var srcDurs: [Int] = []

        switch alignment {
        case .some(let uttAlign) where uttAlign.phonemes.count == bodyPhoneCount && AlignmentStore.isUtteranceAlignmentValid(uttAlign):
            effectiveLeadSilence = uttAlign.leadSilenceFrames
            effectiveSpeechFrames = uttAlign.totalSpeechFrames
            effectiveTrailSilence = uttAlign.trailSilenceFrames
            var p = 0
            while p < uttAlign.phonemes.count {
                srcDurs.append(max(1, uttAlign.phonemes[p].durationFrames))
                p += 1
            }
        case _:
            effectiveLeadSilence = leadSilence
            effectiveSpeechFrames = actualSpeechFrames
            effectiveTrailSilence = trailSilence
            var rawFloatDurs: [Float] = []
            var p = 0
            while p < bodyPhoneCount {
                let pid = linguisticFeatures.phoneIds[1 + p]
                let avgDur = lengthRegulator.phonemeDuration(phoneId: pid, speedFactor: 1.0)
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
            srcDurs = lengthRegulator.quantizeDurations(durations: scaledFloatDurs)
        }

        // 実音声の発話本体区間（speechMel）を抽出
        let speechEnd = min(origFrames, effectiveLeadSilence + effectiveSpeechFrames)
        var speechMel: [[Float]] = []
        var sf = effectiveLeadSilence
        while sf < speechEnd {
            speechMel.append(targetMel[sf])
            sf += 1
        }
        if speechMel.isEmpty {
            return nil
        }
        let srcLen = speechMel.count

        // srcDurs の総和を厳密に srcLen と一致させる
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

        // 音素境界単位で教師 Mel を目標フレーム数へ線形リサンプリング
        let melCh = AudioConfig.melChannels
        var resampledMel = [[Float]](repeating: [Float](repeating: 0.0, count: melCh), count: targetSpeechFrames)
        var srcOffset = 0
        var dstOffset = 0
        var phSeqIdx = 0
        while phSeqIdx < bodyPhoneCount {
            let srcDur = srcDurs[phSeqIdx]
            let dstDur = Int(linguisticFeatures.durations[1 + phSeqIdx])

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
                resampledMel[dstIdx].withUnsafeMutableBufferPointer { pDst in
                    speechMel[safeSrcIdx].withUnsafeBufferPointer { pSrc in
                        pDst.baseAddress!.update(from: pSrc.baseAddress!, count: melCh)
                    }
                }

            case (false, true):
                let safeSrcIdx = min(srcLen - 1, max(0, srcOffset))
                var f = 0
                while f < dstDur {
                    let dstIdx = min(targetSpeechFrames - 1, dstOffset + f)
                    resampledMel[dstIdx].withUnsafeMutableBufferPointer { pDst in
                        speechMel[safeSrcIdx].withUnsafeBufferPointer { pSrc in
                            pDst.baseAddress!.update(from: pSrc.baseAddress!, count: melCh)
                        }
                    }
                    f += 1
                }

            case (false, false):
                let maxDstP = Float(dstDur - 1)
                let maxSrcP = Float(srcDur - 1)
                var f = 0
                while f < dstDur {
                    let dstIdx = min(targetSpeechFrames - 1, dstOffset + f)
                    let posWithinPh = (Float(f) / maxDstP) * maxSrcP
                    var s0 = Int(posWithinPh)
                    if srcDur <= s0 {
                        s0 = srcDur - 1
                    }
                    if s0 < 0 {
                        s0 = 0
                    }
                    var s1 = s0 + 1
                    if srcDur <= s1 {
                        s1 = srcDur - 1
                    }
                    let alpha = posWithinPh - Float(s0)
                    let src0 = min(srcLen - 1, max(0, srcOffset + s0))
                    let src1 = min(srcLen - 1, max(0, srcOffset + s1))

                    var c = 0
                    while c < melCh {
                        resampledMel[dstIdx][c] = (1.0 - alpha) * speechMel[src0][c] + alpha * speechMel[src1][c]
                        c += 1
                    }
                    f += 1
                }
            }

            srcOffset += srcDur
            dstOffset += dstDur
            phSeqIdx += 1
        }

        // 先頭無音・発話本体・末尾無音の結合
        var alignedMel = [[Float]](repeating: [Float](repeating: 0.0, count: melCh), count: totalFrames)

        // 先頭無音区間: WAV の実測無音フレームの Mel を目標とする
        var lf = 0
        while lf < leadSil {
            let srcF: Int
            switch 0 < effectiveLeadSilence {
            case true:
                srcF = min(lf, effectiveLeadSilence - 1)
            case false:
                srcF = 0
            }
            let safeSrcF = min(targetMel.count - 1, max(0, srcF))
            alignedMel[lf].withUnsafeMutableBufferPointer { pDst in
                targetMel[safeSrcF].withUnsafeBufferPointer { pSrc in
                    pDst.baseAddress!.update(from: pSrc.baseAddress!, count: melCh)
                }
            }
            lf += 1
        }

        // 発話本体区間: 音素境界単位でリサンプリングされた Mel を配置
        var bf = 0
        while bf < targetSpeechFrames {
            let dstF = leadSil + bf
            alignedMel[dstF].withUnsafeMutableBufferPointer { pDst in
                resampledMel[bf].withUnsafeBufferPointer { pSrc in
                    pDst.baseAddress!.update(from: pSrc.baseAddress!, count: melCh)
                }
            }
            bf += 1
        }

        // 末尾無音区間: WAV の実測無音フレームの Mel を目標とする
        var trf = 0
        while trf < trailSil {
            let dstF = leadSil + targetSpeechFrames + trf
            let srcF: Int
            switch 0 < effectiveTrailSilence {
            case true:
                srcF = min(origFrames - 1, effectiveLeadSilence + effectiveSpeechFrames + trf)
            case false:
                srcF = origFrames - 1
            }
            let safeSrcF = min(targetMel.count - 1, max(0, srcF))
            alignedMel[dstF].withUnsafeMutableBufferPointer { pDst in
                targetMel[safeSrcF].withUnsafeBufferPointer { pSrc in
                    pDst.baseAddress!.update(from: pSrc.baseAddress!, count: melCh)
                }
            }
            trf += 1
        }

        return (features: features, targets: alignedMel)
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

        let boundaries = Self.detectSpeechBoundaries(
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
                    let ruleDur = lengthRegulator.phonemeDuration(phoneId: Int32(token.id), speedFactor: 1.0)

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
