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

        // なぜ女性話者の baseF0 でアライメントし applyFluctuation: false にするか:
        // 教師データは成人女性単一話者の実録音音声であり、実音声のピッチ帯域（~220Hz）と
        // 一致させる必要がある。また学習時は実音声波形との決定論的な時間軸対応関係を確立する必要があり、
        // 1/f ゆらぎを混入させるとアライメントが汚染されて音素境界が不正確になるため。
        let baseLinguistic = lengthRegulator.processText(
            text: text,
            normalizer: normalizer,
            prosodyModel: prosodyModel,
            vocabulary: vocabulary,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0,
            applyFluctuation: false
        )

        let origTotalFrames = baseLinguistic.totalFrames
        let phoneCount = baseLinguistic.phoneIds.count

        let boundaries = detectSpeechBoundaries(
            pcm: pcm16k,
            hopSize: AudioConfig.hopSize,
            totalFrames: targetFrames
        )
        let leadSilence = boundaries.leadSilence
        let speechFrames = boundaries.speechFrames
        let trailSilence = boundaries.trailSilence

        // なぜ音素数と発話フレーム数の境界検査を行うか:
        // 発話区間フレーム数がゼロまたは音素数未満の場合、各音素に最低 1 フレームを割り当てることが物理的に不可能となり
        // 時間軸アライメントが破綻するため、安全に nil を返してデータセットの品質を保護する。
        if speechFrames <= 0 || speechFrames < phoneCount {
            return nil
        }

        var alignedFeatures: [[Float]]
        var fullPhoneIds: [Int32] = []
        var fullDurations: [Int] = []

        if origTotalFrames <= 0 || phoneCount <= 0 {
            alignedFeatures = [[Float]](repeating: [Float](repeating: 0.0, count: weights.inputDim), count: targetFrames)
            fullPhoneIds = [Int32(PhonemeVocabulary.silId)]
            fullDurations = [targetFrames]
        } else {
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
                // 負の差分: 1 フレームを超えて短縮可能な音素に対して均等に巡回削減
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
                            // なぜ全音素が 1 フレームに達した場合に break するか:
                            // 全音素が下限値 1 フレームとなった場合は物理的にそれ以上削減不能であり、
                            // ガード条件の有無にかかわらず無限ループを未然に確実に防止するため。
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
                // なぜ打ち切りなしの巡回配分を行うか:
                // 差分フレームが音素数を超える場合であっても、余りフレームを一切捨てずに全音素へ均等に 1 フレームずつ
                // 何周でも巡回加算することで、合計フレーム数を speechFrames と厳密に完全一致させるため。
                var remainingDiff = diff
                var roundIdx = 0
                while 0 < remainingDiff {
                    let dIdx = roundIdx % speechDurations.count
                    speechDurations[dIdx] += 1
                    remainingDiff -= 1
                    roundIdx += 1
                }
            }

            if 0 < leadSilence {
                fullPhoneIds.append(Int32(PhonemeVocabulary.silId))
                fullDurations.append(leadSilence)
            }

            var bIdx = 0
            while bIdx < phoneCount {
                fullPhoneIds.append(baseLinguistic.phoneIds[bIdx])
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
        }

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
}
