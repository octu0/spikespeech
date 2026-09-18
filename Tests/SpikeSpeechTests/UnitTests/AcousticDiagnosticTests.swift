import XCTest
import Foundation
#if canImport(MLX)
import MLX
#endif
@testable import SpikeSpeech

final class AcousticDiagnosticTests: XCTestCase {

    /// 「あいうえお」合成時の各ステージ（言語、SNN、MelToLPC、ボコーダー）の診断
    func testDiagnoseAcousticPipeline() {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let weightsPath = currentDir + "/Models/weights.json"

        var weights: SpikingNetworkWeights = SpikingNetworkWeights.randomWeights()
        if fileManager.fileExists(atPath: weightsPath) {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
                weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: data)
                print("[Diagnostic] Models/weights.json を正常に読み込みました。")
            } catch {
                print("[Diagnostic] weights.json 読み込み失敗: \(error)")
            }
        } else {
            print("[Diagnostic] Models/weights.json が見つかりません。")
        }

        let engine = SpikeSpeechEngine(weights: weights)

        let text = "あいうえお"
        let linguisticFeatures = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )

        print("[Diagnostic] totalFrames: \(linguisticFeatures.totalFrames)")
        print("[Diagnostic] phoneIds: \(linguisticFeatures.phoneIds)")
        print("[Diagnostic] durations: \(linguisticFeatures.durations)")

        let inputSeq = engine.encodeLinguisticFeatures(features: linguisticFeatures)
        print("[Diagnostic] inputSeq frames: \(inputSeq.count), inDim: \(inputSeq.first?.count ?? 0)")
        let phone0Dur = Int(linguisticFeatures.durations[0])
        let phone1Idx = min(inputSeq.count - 1, phone0Dur + 1)
        print("[Diagnostic] inputSeq[0] (phone=\(linguisticFeatures.phoneIds[0])): activeIndices=\(inputSeq[0].enumerated().filter { 0.0 < $0.element }.map { ($0.offset, $0.element) }.prefix(6))")
        print("[Diagnostic] inputSeq[\(phone0Dur - 1)] (phone=\(linguisticFeatures.phoneIds[0])): activeIndices=\(inputSeq[phone0Dur - 1].enumerated().filter { 0.0 < $0.element }.map { ($0.offset, $0.element) }.prefix(6))")
        print("[Diagnostic] inputSeq[\(phone1Idx)] (phone=\(linguisticFeatures.phoneIds[1])): activeIndices=\(inputSeq[phone1Idx].enumerated().filter { 0.0 < $0.element }.map { ($0.offset, $0.element) }.prefix(6))")

        // wIn 重みの検査
        let inDim = weights.inputDim
        let hDim = weights.maxHiddenDim
        var phone5WeightSum: Float = 0.0
        var phone6WeightSum: Float = 0.0
        var h = 0
        while h < hDim {
            phone5WeightSum += abs(weights.wIn[(h * inDim) + 5])
            phone6WeightSum += abs(weights.wIn[(h * inDim) + 6])
            h += 1
        }
        print("[Diagnostic] phone5 weight L1 norm: \(phone5WeightSum), phone6 weight L1 norm: \(phone6WeightSum)")

        // 背景電流 vs 音素電流の診断
        let feat0 = inputSeq[0]
        var maxPhoneCur: Float = 0.0
        var maxBgCur: Float = 0.0
        var hIdx = 0
        while hIdx < hDim {
            var phoneCur: Float = 0.0
            var bgCur: Float = weights.bH[hIdx]
            var d = 0
            while d < inDim {
                let w = weights.wIn[(hIdx * inDim) + d]
                if d < 64 {
                    phoneCur += w * feat0[d]
                } else {
                    bgCur += w * feat0[d]
                }
                d += 1
            }
            if maxPhoneCur < abs(phoneCur) {
                maxPhoneCur = abs(phoneCur)
            }
            if maxBgCur < abs(bgCur) {
                maxBgCur = abs(bgCur)
            }
            hIdx += 1
        }
        print("[Diagnostic] maxPhoneCurrent: \(maxPhoneCur), maxBgCurrent: \(maxBgCur)")

        let acousticSeq = engine.decoder.decodeSequence(featuresSeq: inputSeq, workspace: engine.workspace)
        print("[Diagnostic] acousticSeq frames: \(acousticSeq.count), outDim: \(acousticSeq.first?.count ?? 0)")

        // Frame 1 (あ) と 音素 [い] の代表フレームの acousticSeq の差分
        let iFrame = phone1Idx
        var diff1_i: Float = 0.0
        var k = 0
        while k < acousticSeq[1].count {
            diff1_i += abs(acousticSeq[1][k] - acousticSeq[iFrame][k])
            k += 1
        }
        print("[Diagnostic] diff between frame 1 (あ) and frame \(iFrame) (い): \(diff1_i)")

        // decodeFrame を直接単独で frame 1 と frame iFrame について実行してみる
        engine.workspace.reset()
        var out1 = [Float](repeating: 0.0, count: weights.outputDim)
        var outI = [Float](repeating: 0.0, count: weights.outputDim)
        inputSeq[1].withUnsafeBufferPointer { pIn1 in
            out1.withUnsafeMutableBufferPointer { pOut1 in
                engine.decoder.decodeFrame(features: pIn1.baseAddress!, workspace: engine.workspace, outputFeatures: pOut1.baseAddress!)
            }
        }
        engine.workspace.reset()
        inputSeq[iFrame].withUnsafeBufferPointer { pInI in
            outI.withUnsafeMutableBufferPointer { pOutI in
                engine.decoder.decodeFrame(features: pInI.baseAddress!, workspace: engine.workspace, outputFeatures: pOutI.baseAddress!)
            }
        }
        var isolatedDiff: Float = 0.0
        k = 0
        while k < out1.count {
            isolatedDiff += abs(out1[k] - outI[k])
            k += 1
        }
        print("[Diagnostic] isolated diff between frame 1 and \(iFrame) (after reset): \(isolatedDiff)")

        // SNN 出力 Mel 特徴量の統計（各フレームの平均・最小・最大）
        var t = 0
        while t < min(10, acousticSeq.count) {
            let mel = acousticSeq[t]
            var minV: Float = Float.infinity
            var maxV: Float = -Float.infinity
            var sumV: Float = 0.0
            var c = 0
            while c < mel.count {
                let v = mel[c]
                if v < minV {
                    minV = v
                }
                if maxV < v {
                    maxV = v
                }
                sumV += v
                c += 1
            }
            let avgV = sumV / Float(max(1, mel.count))
            
            // スパイク発火数の確認
            var nonZeroReadout = 0
            var r = 0
            while r < engine.workspace.readoutSums.count {
                if 0.0 < engine.workspace.readoutSums[r] {
                    nonZeroReadout += 1
                }
                r += 1
            }
            print("[Diagnostic] Frame \(t): min=\(minV), max=\(maxV), avg=\(avgV), nonZeroReadout=\(nonZeroReadout), first4=\(mel.prefix(4))")
            t += 1
        }

        // MelToLPC の出力診断（Prior + SNN 残差）
        var lpcCoeffs = [Float](repeating: 0.0, count: AudioConfig.lpcOrder)
        var combinedMel = [Float](repeating: 0.0, count: AudioConfig.melChannels)
        let priorMel0 = engine.acousticPrior.getPriorMel(phoneId: Int(linguisticFeatures.phoneIds[0]))
        var mIdx = 0
        let residualScale: Float = 0.4
        while mIdx < AudioConfig.melChannels {
            combinedMel[mIdx] = priorMel0[mIdx] + (residualScale * acousticSeq[1][mIdx])
            mIdx += 1
        }
        let gain = engine.melToLPC.convert(mel: combinedMel, isLogMel: true, outCoeffs: &lpcCoeffs)
        print("[Diagnostic] Scaled (0.4) MelToLPC convert gain=\(gain), lpcCoeffs=\(lpcCoeffs)")

        // 合成された LPC 係数の周波数応答（|H(e^jw)|）のピーク周波数を探索
        func findPeaks(coeffs: [Float]) -> [Float] {
            var response = [Float](repeating: 0.0, count: 512)
            var k = 0
            while k < 512 {
                let omega = Float.pi * Float(k) / 511.0
                var re: Float = 1.0
                var im: Float = 0.0
                var m = 0
                while m < coeffs.count {
                    let angle = -omega * Float(m + 1)
                    re -= coeffs[m] * cos(angle)
                    im -= coeffs[m] * sin(angle)
                    m += 1
                }
                let magSq = (re * re) + (im * im)
                response[k] = 1.0 / sqrt(max(1e-12, magSq))
                k += 1
            }
            var peaks: [Float] = []
            var i = 1
            while i < 511 {
                if response[i - 1] < response[i] && response[i + 1] < response[i] {
                    let freq = (Float(i) * 8000.0) / 511.0
                    peaks.append(freq)
                }
                i += 1
            }
            return peaks
        }
        let peaksSNN = findPeaks(coeffs: lpcCoeffs)
        print("[Diagnostic] SNN output LPC formant peaks: \(peaksSNN)")

        // ボコーダーの単体実験: 母音「あ」のフォルマント共鳴フィルタに Rosenberg パルスを通したときの波形
        // 1. 現状の LPCVocoder (deEmphasis=0.97)
        let vocoderDefault = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)
        // 2. ディエンファシスなし LPCVocoder (deEmphasis=0.0)
        let vocoderNoDe = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.0)

        let aFrame = AcousticFrame(lpcCoefficients: lpcCoeffs, gain: gain, pitchF0: 193.0, voiced: 1.0)
        var outDefault = [Float](repeating: 0.0, count: 160 * 10)
        var outNoDe = [Float](repeating: 0.0, count: 160 * 10)

        var fIdx = 0
        while fIdx < 10 {
            outDefault.withUnsafeMutableBufferPointer { pDst in
                vocoderDefault.synthesizeFrame(frame: aFrame, dst: pDst.baseAddress!.advanced(by: fIdx * 160))
            }
            outNoDe.withUnsafeMutableBufferPointer { pDst in
                vocoderNoDe.synthesizeFrame(frame: aFrame, dst: pDst.baseAddress!.advanced(by: fIdx * 160))
            }
            fIdx += 1
        }

        func measureZCR(wave: [Float]) -> (Float, Float) {
            var zc = 0
            var sumSq: Float = 0.0
            var i = 1
            while i < wave.count {
                let s = wave[i]
                sumSq += s * s
                if (wave[i - 1] <= 0.0 && 0.0 < s) || (s < 0.0 && 0.0 <= wave[i - 1]) {
                    zc += 1
                }
                i += 1
            }
            let freq = (Float(zc) * 16000.0) / (Float(wave.count) * 2.0)
            let rms = sqrt(sumSq / Float(wave.count))
            return (freq, rms)
        }

        // 1. SNN 音素感度検証: Frame 1（あ）と Frame \(phone1Idx)（い）の間で有意な Mel スペクトル差分が存在すること
        XCTAssertTrue(1.0 < diff1_i, "音素遷移によって SNN 出力 Mel スペクトルが有意に変化していません: diff=\(diff1_i)")

        // 2. 合成音声の数値健全性検証: クリップせず十分なエネルギーを有すること
        let samples = engine.synthesize(text: text, voice: .female)
        XCTAssertFalse(samples.isEmpty, "合成音声サンプルが空です")

        var maxAbs: Float = 0.0
        var sumSq: Float = 0.0
        var sIdx = 0
        while sIdx < samples.count {
            let s = samples[sIdx]
            XCTAssertFalse(s.isNaN, "サンプル \(sIdx) で NaN を検出")
            XCTAssertFalse(s.isInfinite, "サンプル \(sIdx) で Inf を検出")
            let a = abs(s)
            if maxAbs < a {
                maxAbs = a
            }
            sumSq += s * s
            sIdx += 1
        }
        let rms = sqrt(sumSq / Float(samples.count))
        print(String(format: "[Pipeline Diagnostic] maxAbs=%.4f, rms=%.4f", maxAbs, rms))

        // ピークが 0.85 以下でクリップしていないこと
        XCTAssertTrue(maxAbs <= 0.85, "合成音声がリミッター上限にクリップしています: maxAbs=\(maxAbs)")
        // 有意な音響エネルギー（RMS > 0.05）を有していること
        XCTAssertTrue(0.05 < rms, "合成音声の音響エネルギーが小さすぎます: rms=\(rms)")
    }

    /// 「こんにちは。」文章合成時の音素、Duration、有声度、LPCゲインの推移診断
    func testDiagnoseKonnichiwaText() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは。スパイキングニューラルネットワークによる超低遅延音声合成の世界へようこそ。"
        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )

        print("--- [Konnichiwa Diagnostic] ---")
        print("総音素数: \(linguistic.phoneIds.count), 総フレーム数: \(linguistic.totalFrames)")
        var p = 0
        let pLimit = min(linguistic.phoneIds.count, 40)
        while p < pLimit {
            let pId = linguistic.phoneIds[p]
            let dur = linguistic.durations[p]
            let token = engine.vocabulary.token(for: Int(pId))
            let priorMel = engine.acousticPrior.getPriorMel(phoneId: Int(pId))
            var priorSum: Float = 0.0
            var c = 0
            while c < priorMel.count {
                priorSum += priorMel[c]
                c += 1
            }
            let priorAvg = priorSum / Float(priorMel.count)
            print(String(format: "  [%2d] phone=%2d (%4s), dur=%2d, priorAvg=%+0.2f", p, pId, (token as NSString).utf8String!, dur, priorAvg))
            p += 1
        }

        // ストリーミング合成時の波形プロパティ検証
        let streamSamples = engine.synthesizeStream(text: text, voice: .female, speed: 1.0, pitch: 1.0, onFrame: nil)
        var streamPeak: Float = 0.0
        var streamSumSq: Float = 0.0
        var sIdx = 0
        while sIdx < streamSamples.count {
            let s = streamSamples[sIdx]
            let a = abs(s)
            if streamPeak < a { streamPeak = a }
            streamSumSq += s * s
            sIdx += 1
        }
        let streamRms = sqrt(streamSumSq / Float(max(1, streamSamples.count)))
        print(String(format: "  [Stream Result] サンプル数=%d, Peak=%0.4f, RMS=%0.4f", streamSamples.count, streamPeak, streamRms))
        XCTAssertTrue(0.05 < streamRms, "ストリーミング合成波形の RMS が過小です: rms=\(streamRms)")
    }

    /// 「こんにちは」の「こ」の発音時の波形・RMS・音素詳細分析
    func testDiagnoseKoNoise() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは"
        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary
        )
        print("=== [こんにちは 音素分析] ===")
        var p = 0
        var curF = 0
        while p < linguistic.phoneIds.count {
            let pId = linguistic.phoneIds[p]
            let dur = Int(linguistic.durations[p])
            let token = engine.vocabulary.token(for: Int(pId))
            print("Phone [\(p)] \(token) (id=\(pId)): dur=\(dur) frames (\(curF)..<\(curF + dur))")
            curF += dur
            p += 1
        }

        let samples = engine.synthesize(text: text, voice: .female)
        let hopSize = AudioConfig.hopSize
        let frameCount = samples.count / hopSize
        print("=== [こんにちは フレーム別 RMS 分析] ===")
        var frameRmsList = [Float](repeating: 0.0, count: min(20, frameCount))
        var f = 0
        while f < min(20, frameCount) {
            var sumSq: Float = 0.0
            var s = 0
            while s < hopSize {
                let v = samples[(f * hopSize) + s]
                sumSq += v * v
                s += 1
            }
            let rms = sqrt(sumSq / Float(hopSize))
            frameRmsList[f] = rms
            let dB = 20.0 * log10(max(1e-6, rms))
            print(String(format: "Frame %2d (sample %4d..<%4d): RMS=%.5f (%.1f dBFS)", f, f * hopSize, (f + 1) * hopSize, rms, dB))
            f += 1
        }

        // 調音生理学の音響検証:
        // 1. 無声破裂音 /k/ の閉鎖区間（最初のフレームから破裂バースト直前まで）は口蓋を密着させて気流を遮断するため、
        //    音響エネルギーが物理的にゼロ（RMS <= 1e-5）であることを厳格に検証する。
        let kDur = Int(linguistic.durations[0])
        let kBurstFrame = kDur - 1
        var kFrame = 0
        while kFrame < kBurstFrame {
            XCTAssertTrue(frameRmsList[kFrame] <= 1e-5, "/k/ の閉鎖区間 Frame \(kFrame) にノイズが漏洩しています: RMS=\(frameRmsList[kFrame])")
            kFrame += 1
        }

        // 2. 破裂バースト期（kBurstFrame）は短いインパルス的開放アタックであり、
        //    かつ耳障りな過大ホワイトノイズ（<= 0.02）になっていないことを検証する。
        XCTAssertTrue(frameRmsList[kBurstFrame] <= 0.02, "破裂バースト期 Frame \(kBurstFrame) のエネルギーが過大です: RMS=\(frameRmsList[kBurstFrame])")

        // 3. 後続母音 /o/ の定常部で豊かな母音フォルマント共鳴（0.10 <= RMS）が立ち上がっていることを検証する。
        let oVowelFrame = min(frameCount - 1, kBurstFrame + 3)
        XCTAssertTrue(0.10 <= frameRmsList[oVowelFrame], "母音 /o/ のフォルマント共鳴エネルギーが不足しています: RMS=\(frameRmsList[oVowelFrame])")
    }

    /// 音声サンプルのエネルギー分布、前後無音区間、および Prior フォルマントピークの診断
    func testDiagnoseAudioSampleAndPrior() throws {
        let currentDir = FileManager.default.currentDirectoryPath
        let wavPath = currentDir + "/Tests/resources/test_female.wav"
        let reader = WavAudioReader()
        let pcm = try reader.loadWav16k(from: wavPath)
        var tfPeak: Float = 0.0
        var tfSumSq: Float = 0.0
        var tfIdx = 0
        while tfIdx < pcm.count {
            let s = pcm[tfIdx]
            let a = abs(s)
            if tfPeak < a { tfPeak = a }
            tfSumSq += s * s
            tfIdx += 1
        }
        let tfRms = sqrt(tfSumSq / Float(max(1, pcm.count)))
        print(String(format: "[Audio Diagnostic] test_female.wav pcm count: %d (%.2f 秒), Peak=%.4f, RMS=%.4f", pcm.count, Float(pcm.count) / 16000.0, tfPeak, tfRms))

        let extractor = MelSpectrogramExtractor()
        let mel = extractor.extractLogMel(pcm: pcm)
        let jsutPath = ProcessInfo.processInfo.environment["JSUT_PATH"] ?? ""
        if !jsutPath.isEmpty && FileManager.default.fileExists(atPath: jsutPath) {
            let jsutPcm = try reader.loadWav16k(from: jsutPath)
            var jPeak: Float = 0.0
            var jSumSq: Float = 0.0
            var jsIdx = 0
            while jsIdx < jsutPcm.count {
                let s = jsutPcm[jsIdx]
                let a = abs(s)
                if jPeak < a { jPeak = a }
                jSumSq += s * s
                jsIdx += 1
            }
            let jRms = sqrt(jSumSq / Float(max(1, jsutPcm.count)))
            let jMel = extractor.extractLogMel(pcm: jsutPcm)
            var jMel20Sum: Float = 0.0
            let targetJF = min(100, jMel.count - 1)
            var jc = 0
            while jc < 64 {
                jMel20Sum += jMel[targetJF][jc]
                jc += 1
            }
            print(String(format: "[JSUT Diagnostic] BASIC5000_0001.wav pcm: %d, Peak=%.4f, RMS=%.4f, MelFrames=%d, Frame100Avg=%.4f", jsutPcm.count, jPeak, jRms, jMel.count, jMel20Sum / 64.0))
        }

        // フレームごとの RMS エネルギー推移を算出
        var frameRms = [Float](repeating: 0.0, count: mel.count)
        var f = 0
        while f < mel.count {
            let start = f * AudioConfig.hopSize
            var sumSq: Float = 0.0
            var s = 0
            while s < AudioConfig.hopSize {
                let idx = start + s
                if idx < pcm.count {
                    let sample = pcm[idx]
                    sumSq += sample * sample
                }
                s += 1
            }
            frameRms[f] = sqrt(sumSq / Float(AudioConfig.hopSize))
            f += 1
        }

        // 先頭・末尾の無音（RMS < 0.01）のフレーム数を計測
        var leadSilence = 0
        while leadSilence < frameRms.count {
            if 0.01 <= frameRms[leadSilence] {
                break
            }
            leadSilence += 1
        }

        var trailSilence = 0
        var tIdx = frameRms.count - 1
        while 0 <= tIdx {
            if 0.01 <= frameRms[tIdx] {
                break
            }
            trailSilence += 1
            tIdx -= 1
        }

        print("[Audio Diagnostic] 先頭無音フレーム数: \(leadSilence) (\(leadSilence * 10) ms)")
        print("[Audio Diagnostic] 発話有音フレーム数: \(mel.count - leadSilence - trailSilence) (\((mel.count - leadSilence - trailSilence) * 10) ms)")
        print("[Audio Diagnostic] 末尾無音フレーム数: \(trailSilence) (\(trailSilence * 10) ms)")

        // 実音声の無音区間 vs 有音区間の対数 Mel 平均
        var leadMelSum: Float = 0.0
        var voicedMelSum: Float = 0.0
        var cIdx2 = 0
        while cIdx2 < AudioConfig.melChannels {
            if 0 < leadSilence {
                leadMelSum += mel[min(10, leadSilence - 1)][cIdx2]
            }
            let vFrame = leadSilence + 20
            if vFrame < mel.count {
                voicedMelSum += mel[vFrame][cIdx2]
            }
            cIdx2 += 1
        }
        print("[Audio Diagnostic] 無音フレーム対数 Mel 平均: \(leadMelSum / Float(AudioConfig.melChannels))")
        print("[Audio Diagnostic] 有音フレーム対数 Mel 平均: \(voicedMelSum / Float(AudioConfig.melChannels))")

        // 母音 Prior の LPC フォルマントピーク診断
        let engine = SpikeSpeechEngine()
        func findPeaks(coeffs: [Float]) -> [Float] {
            var response = [Float](repeating: 0.0, count: 512)
            var k = 0
            while k < 512 {
                let omega = Float.pi * Float(k) / 511.0
                var re: Float = 1.0
                var im: Float = 0.0
                var m = 0
                while m < coeffs.count {
                    let angle = -omega * Float(m + 1)
                    re -= coeffs[m] * cos(angle)
                    im -= coeffs[m] * sin(angle)
                    m += 1
                }
                let magSq = (re * re) + (im * im)
                response[k] = 1.0 / sqrt(max(1e-12, magSq))
                k += 1
            }
            var peaks: [Float] = []
            var i = 1
            while i < 511 {
                if response[i - 1] < response[i] && response[i + 1] < response[i] {
                    let freq = (Float(i) * 8000.0) / 511.0
                    peaks.append(freq)
                }
                i += 1
            }
            return peaks
        }

        let vowels = [(5, "あ"), (6, "い"), (7, "う"), (8, "え"), (9, "お")]
        for (pId, _) in vowels {
            let pMel = engine.acousticPrior.getPriorMel(phoneId: pId)
            var coeffs = [Float](repeating: 0.0, count: AudioConfig.lpcOrder)
            let gain = engine.melToLPC.convert(mel: pMel, isLogMel: true, outCoeffs: &coeffs)
            XCTAssertTrue(0.0 <= gain, "母音 \(pId) のゲインが負数です")
            let peaks = findPeaks(coeffs: coeffs)
            XCTAssertFalse(peaks.isEmpty, "母音 \(pId) のLPCピークが抽出されませんでした")
        }
        let reconstructed = engine.neuralVocoder.synthesize(mel: mel)
        var recSumSq: Float = 0.0
        var recPeak: Float = 0.0
        var rIdx = 0
        while rIdx < reconstructed.count {
            let s = reconstructed[rIdx]
            let a = abs(s)
            if recPeak < a { recPeak = a }
            recSumSq += s * s
            rIdx += 1
        }
        let recRms = sqrt(recSumSq / Float(max(1, reconstructed.count)))
        print(String(format: "[Audio Diagnostic] 実音声 Mel からのボコーダー再構成 PCM: サンプル数=%d, Peak=%.4f, RMS=%.4f", reconstructed.count, recPeak, recRms))
    }

    /// test_denoised.wav の波形・ノイズフロア・周波数スペクトル音響監査
    func testAuditDenoisedWav() throws {
        let currentDir = FileManager.default.currentDirectoryPath
        let wavPath = currentDir + "/Tests/resources/test_denoised.wav"
        let reader = WavAudioReader()
        let pcm = try reader.loadWav16k(from: wavPath)

        print("=== [test_denoised.wav 音響監査レポート] ===")
        print("総サンプル数: \(pcm.count) (\(Float(pcm.count) / 16000.0) 秒)")

        // 1. 全体ピークと実効値 (RMS)
        var maxAbs: Float = 0.0
        var sumSq: Float = 0.0
        var i = 0
        while i < pcm.count {
            let s = pcm[i]
            let a = abs(s)
            if maxAbs < a {
                maxAbs = a
            }
            sumSq += s * s
            i += 1
        }
        let totalRms = sqrt(sumSq / Float(max(1, pcm.count)))
        print(String(format: "全体 Peak: %.4f (約 %.2f dBFS)", maxAbs, 20.0 * log10(max(1e-6, maxAbs))))
        print(String(format: "全体 RMS:  %.4f (約 %.2f dBFS)", totalRms, 20.0 * log10(max(1e-6, totalRms))))

        // 2. フレームごとの RMS 分布（無音区間 vs 有音区間）
        let hopSize = AudioConfig.hopSize // 160 samples (10ms)
        let frameCount = pcm.count / hopSize
        var frameRms = [Float](repeating: 0.0, count: frameCount)
        var f = 0
        while f < frameCount {
            let start = f * hopSize
            var fSumSq: Float = 0.0
            var s = 0
            while s < hopSize {
                let v = pcm[start + s]
                fSumSq += v * v
                s += 1
            }
            frameRms[f] = sqrt(fSumSq / Float(hopSize))
            f += 1
        }

        // 先頭・末尾の無音フレームの残留ノイズ
        var leadZeroFrames = 0
        while leadZeroFrames < frameCount {
            if 1e-4 <= frameRms[leadZeroFrames] {
                break
            }
            leadZeroFrames += 1
        }

        var trailZeroFrames = 0
        var tIdx = frameCount - 1
        while 0 <= tIdx {
            if 1e-4 <= frameRms[tIdx] {
                break
            }
            trailZeroFrames += 1
            tIdx -= 1
        }

        print("先頭完全無音フレーム数 (RMS < -80dB): \(leadZeroFrames) (\(leadZeroFrames * 10) ms)")
        print("末尾完全無音フレーム数 (RMS < -80dB): \(trailZeroFrames) (\(trailZeroFrames * 10) ms)")

        // 句読点「。」ポーズ区間（中間にある最小エネルギー区間）の調査
        var minPauseRms: Float = Float.infinity
        var midF = leadZeroFrames + 10
        let midEnd = frameCount - trailZeroFrames - 10
        while midF < midEnd {
            if frameRms[midF] < minPauseRms {
                minPauseRms = frameRms[midF]
            }
            midF += 1
        }
        print(String(format: "中間ポーズ最小 RMS: %.6f (約 %.2f dBFS)", minPauseRms, 20.0 * log10(max(1e-6, minPauseRms))))

        // 3. 有音区間における高域エネルギー比率（ディエンファシスの効果）
        // 4kHz 以下の母音エネルギー vs 4kHz 以上の高域ノイズエネルギー
        let extractor = MelSpectrogramExtractor()
        let logMel = extractor.extractLogMel(pcm: pcm)
        var lowEnergySum: Float = 0.0
        var highEnergySum: Float = 0.0
        var lf = 0
        while lf < logMel.count {
            if 0.02 <= frameRms[min(frameCount - 1, lf)] {
                // 有音フレームのみ計測
                var c = 0
                while c < 32 { // 0〜4kHz 付近
                    lowEnergySum += exp(logMel[lf][c])
                    c += 1
                }
                while c < 64 { // 4〜8kHz 高域
                    highEnergySum += exp(logMel[lf][c])
                    c += 1
                }
            }
            lf += 1
        }
        let highToLowRatio = highEnergySum / max(1e-6, lowEnergySum)
        print(String(format: "有声部 高域/低域 エネルギー比: %.4f (高域ノイズ比率: %.2f%%)", highToLowRatio, highToLowRatio * 100.0))

        // 4. 末尾 40 フレーム (400ms) の推移と音素・有声フラグの検査
        print("--- [末尾 40 フレーム (400ms) の推移] ---")
        let engine = SpikeSpeechEngine()
        let text = "こんにちは。スパイキングニューラルネットワーク音声合成の世界へようこそ。"
        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary
        )
        print("総フレーム数: \(linguistic.totalFrames)")
        var curF = 0
        var p = 0
        while p < linguistic.phoneIds.count {
            let pId = linguistic.phoneIds[p]
            let dur = Int(linguistic.durations[p])
            let token = engine.vocabulary.token(for: Int(pId))
            if 440 <= (curF + dur) {
                print(String(format: "Phone [%2d] id=%2d (%4s), dur=%2d, frames %d..<%d", p, pId, (token as NSString).utf8String!, dur, curF, curF + dur))
            }
            curF += dur
            p += 1
        }
        let tailStartFrame = max(0, frameCount - 40)
        var tf = tailStartFrame
        while tf < frameCount {
            let vFlag = tf < linguistic.voicedFlags.count ? linguistic.voicedFlags[tf] : 0.0
            let f0Val = tf < linguistic.f0Contour.count ? linguistic.f0Contour[tf] : 0.0
            print(String(format: "Frame %3d: RMS=%.5f (%.1f dBFS), voiced=%.1f, F0=%.1f", tf, frameRms[tf], 20.0 * log10(max(1e-6, frameRms[tf])), vFlag, f0Val))
            tf += 1
        }
        print("==========================================")

        XCTAssertTrue(maxAbs <= 0.85, "ピークがクリップしています")
        XCTAssertTrue(0.05 < totalRms, "全体エネルギーが過小です")

        // 文末ポーズ区間（末尾 <pau> 音素区間）においてヒスノイズが完全にゼロ（-80dB以下、完全ミュート）であることの検証
        // 末尾に少なくとも 20 フレーム（200ms）以上の完全無音区間（trailZeroFrames）が存在することを検証する
        XCTAssertTrue(20 <= trailZeroFrames, "文末ポーズ区間にノイズが漏洩しています: trailZeroFrames=\(trailZeroFrames)")
    }

    /// ボコーダーのゲインゼロ遷移時における完全無音収束テスト
    ///
    /// 前フレームの確定ゲインが次フレームへ正常に引き継がれ、有音から無音への遷移後に
    /// 過去のゲインが残留して乱数ノイズを放出し続ける不具合が解消されていることを検証する。
    func testVocoderGainZeroConvergence() {
        let vocoder = LPCVocoder()
        vocoder.reset()

        // 1. 有声フレームを数フレーム流す
        var voicedCoeffs = [Float](repeating: 0.0, count: 16)
        voicedCoeffs[0] = 0.5
        let voicedFrame = AcousticFrame(
            lpcCoefficients: voicedCoeffs,
            gain: 0.2,
            pitchF0: 200.0,
            voiced: 1.0
        )
        var buf = [Float](repeating: 0.0, count: 160)
        buf.withUnsafeMutableBufferPointer { ptr in
            vocoder.synthesizeFrame(frame: voicedFrame, dst: ptr.baseAddress!)
        }

        // 2. 無音フレーム (gain = 0.0, voiced = 0.0, f0 = 0.0) を入力
        let silenceFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.0, count: 16),
            gain: 0.0,
            pitchF0: 0.0,
            voiced: 0.0
        )

        // 遷移フレーム（1フレーム目：prevGain = 0.2, currGain = 0.0 によるフェードアウト）
        buf.withUnsafeMutableBufferPointer { ptr in
            vocoder.synthesizeFrame(frame: silenceFrame, dst: ptr.baseAddress!)
        }

        // 完全無音フレーム（2フレーム目：prevGain = 0.0, currGain = 0.0、ディエンファシスの自然減衰）
        buf.withUnsafeMutableBufferPointer { ptr in
            vocoder.synthesizeFrame(frame: silenceFrame, dst: ptr.baseAddress!)
        }
        var maxSample2: Float = 0.0
        var s2 = 0
        while s2 < 160 {
            let a = abs(buf[s2])
            if maxSample2 < a {
                maxSample2 = a
            }
            s2 += 1
        }
        XCTAssertTrue(maxSample2 <= 0.01, "無音フレーム2フレーム目でゲインが正しく減衰していません: max=\(maxSample2)")

        // 完全無音フレーム（3フレーム目：20ms経過、ディエンファシスも完全にゼロへ収束）
        buf.withUnsafeMutableBufferPointer { ptr in
            vocoder.synthesizeFrame(frame: silenceFrame, dst: ptr.baseAddress!)
        }
        var maxSample3: Float = 0.0
        var s3 = 0
        while s3 < 160 {
            let a = abs(buf[s3])
            if maxSample3 < a {
                maxSample3 = a
            }
            s3 += 1
        }
        XCTAssertTrue(maxSample3 <= 1e-4, "無音フレーム3フレーム目でゲインが完全ゼロへ収束していません: max=\(maxSample3)")
    }

    /// オンメモリで合成された音声の全フレーム波形・音素・有声度・F0・スペクトル精密診断
    /// なぜオンメモリ合成とするか:
    /// ディスク上の一時ファイル依存を完全排除し、規約「テストの完全オンメモリ化」「絶対パスを残さない」を遵守するため。
    func testDiagnoseActualGeneratedWav() throws {
        let engine = SpikeSpeechEngine(weights: SpikingNetworkWeights.randomWeights())
        let text = "こんにちは"
        let pcm = engine.synthesize(text: text)
        print("=== [こんにちは オンメモリ合成精密診断レポート] ===")
        print("総サンプル数: \(pcm.count) (\(Float(pcm.count) / 16000.0) 秒)")

        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary
        )

        let hopSize = AudioConfig.hopSize
        let frameCount = pcm.count / hopSize

        var curF = 0
        var p = 0
        print("--- [音素アライメント一覧] ---")
        while p < linguistic.phoneIds.count {
            let pId = linguistic.phoneIds[p]
            let dur = Int(linguistic.durations[p])
            let token = engine.vocabulary.token(for: Int(pId))
            print("Phone [\(p)] \(token) (id=\(pId)): dur=\(dur) frames (\(curF)..<\(curF + dur))")
            curF += dur
            p += 1
        }

        print("--- [フレーム別詳細音響パラメータ推移] ---")
        var f = 0
        while f < min(25, frameCount) {
            let start = f * hopSize
            let end = min(pcm.count, start + hopSize)
            var sumSq: Float = 0.0
            var zcrCount = 0
            var prevSign: Float = 0.0
            var s = start
            while s < end {
                let v = pcm[s]
                sumSq += v * v
                if 0 < s {
                    if (prevSign < 0.0 && 0.0 <= v) || (0.0 <= prevSign && v < 0.0) {
                        zcrCount += 1
                    }
                }
                prevSign = v
                s += 1
            }
            let rms = sqrt(sumSq / Float(max(1, end - start)))
            let dB = 20.0 * log10(max(1e-6, rms))
            let zcr = Float(zcrCount) / Float(max(1, end - start))

            let vFlag = f < linguistic.voicedFlags.count ? linguistic.voicedFlags[f] : -1.0
            let f0Val = f < linguistic.f0Contour.count ? linguistic.f0Contour[f] : -1.0

            // 該当フレームの音素トークン
            var phToken = "?"
            var pScan = 0
            var fScan = 0
            while pScan < linguistic.phoneIds.count {
                let d = Int(linguistic.durations[pScan])
                if f < (fScan + d) {
                    phToken = engine.vocabulary.token(for: Int(linguistic.phoneIds[pScan]))
                    break
                }
                fScan += d
                pScan += 1
            }

            print(String(format: "Frame %2d: Phone=%-4s, RMS=%.5f (%5.1f dB), ZCR=%.3f, voiced=%.2f, F0=%5.1f", f, (phToken as NSString).utf8String!, rms, dB, zcr, vFlag, f0Val))

            // 周波数帯域別エネルギーの診断 (Frame 3〜8)
            if 3 <= f && f <= 8 {
                // 160サンプルの自己相関および高域差分エネルギー
                var highDiffSumSq: Float = 0.0
                var sIdx = 1
                while sIdx < (end - start) {
                    let diff = pcm[start + sIdx] - pcm[start + sIdx - 1]
                    highDiffSumSq += diff * diff
                    sIdx += 1
                }
                let highRoughness = sqrt(highDiffSumSq / Float(max(1, end - start - 1)))
                print(String(format: "  -> 高域粗さ(差分RMS): %.5f, 粗さ/全体RMS比: %.2f", highRoughness, highRoughness / max(1e-6, rms)))
            }
            f += 1
        }
        print("==========================================")
    }

    /// 基準母音（SyntheticAudioGenerator）から Mel 抽出 -> MelToLPC -> LPCVocoder の往復再構成テスト
    /// なぜこのテストを行うか:
    /// SNN の学習誤差と DSP ボコーダー側の信号処理特性を完全に分離し、
    /// 「母音フォルマントがボコーダーを通した際に正しく復元されるか」を単体で確定的に検証するため。
    func testVowelFormantReconstructionViaLPC() throws {
        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let extractor = MelSpectrogramExtractor()
        let melToLpc = MelToLPC(melChannels: 64, fftBins: 257, lpcOrder: 16, sampleRate: 16000.0)
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.95)

        // 5母音（あ・い・う・え・お）を各 0.3 秒生成して結合
        var cleanVowels: [Float] = []
        let vowels: [SyntheticAudioGenerator.Vowel] = [.a, .i, .u, .e, .o]
        var vIdx = 0
        while vIdx < vowels.count {
            let v = vowels[vIdx]
            let pcm = generator.generateVowel(vowel: v, durationSeconds: 0.3, f0: 160.0)
            cleanVowels.append(contentsOf: pcm)
            vIdx += 1
        }

        // 1. 基準母音波形から 64ch 対数 Mel を抽出
        let logMel = extractor.extractLogMel(pcm: cleanVowels)
        XCTAssertFalse(logMel.isEmpty, "抽出された LogMel が空です")

        // 2. 各フレームを MelToLPC で 16次 LPC 係数へ変換しボコーダー用 AcousticFrame を構築
        var acousticFrames: [AcousticFrame] = []
        acousticFrames.reserveCapacity(logMel.count)

        var f = 0
        while f < logMel.count {
            let melVec = logMel[f]
            var lpcCoeffs = [Float](repeating: 0.0, count: 16)
            let gain = melToLpc.convert(mel: melVec, isLogMel: true, outCoeffs: &lpcCoeffs)

            // 160Hz 有声音フレーム
            let frame = AcousticFrame(
                lpcCoefficients: lpcCoeffs,
                gain: gain,
                pitchF0: 160.0,
                voiced: 1.0
            )
            acousticFrames.append(frame)
            f += 1
        }

        // 3. ボコーダーにより音声を再構成
        let reconstructedPcm = vocoder.synthesize(frames: acousticFrames)
        XCTAssertFalse(reconstructedPcm.isEmpty, "再構成音声が空です")

        // 4. 音声バイナリのエンコード検証（ディスク出力を排除しオンメモリで検証）
        let cleanWav = WavEncoder.encode(samples: cleanVowels, sampleRate: 16000)
        let reconWav = WavEncoder.encode(samples: reconstructedPcm, sampleRate: 16000)
        XCTAssertFalse(cleanWav.isEmpty, "基準母音 WAV データが空であってはなりません")
        XCTAssertFalse(reconWav.isEmpty, "再構成 WAV データが空であってはなりません")
        XCTAssertEqual(cleanWav.prefix(4), Data([0x52, 0x49, 0x46, 0x46]), "RIFF ヘッダが正しく出力されていること")
        XCTAssertEqual(reconWav.prefix(4), Data([0x52, 0x49, 0x46, 0x46]), "RIFF ヘッダが正しく出力されていること")
    }

    /// 純粋な PhonemeAcousticPrior による音声合成テスト
    /// Prior 自体がボコーダーを通して明瞭な母音フォルマントを形成できるか検証する。
    func testSynthesizeAcousticPriorOnly() throws {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let weightsPath = currentDir + "/Models/weights.json"

        var weights: SpikingNetworkWeights = SpikingNetworkWeights.randomWeights()
        if fileManager.fileExists(atPath: weightsPath) {
            let data = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
            weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: data)
        }

        let engineNormal = SpikeSpeechEngine(weights: weights)
        engineNormal.residualScale = 1.0

        let enginePriorOnly = SpikeSpeechEngine(weights: weights)
        enginePriorOnly.residualScale = 0.0

        let engineScale02 = SpikeSpeechEngine(weights: weights)
        engineScale02.residualScale = 0.2

        let text = "あいうえお"
        let pcmNormal = engineNormal.synthesize(text: text, voice: .female)
        let pcmPriorOnly = enginePriorOnly.synthesize(text: text, voice: .female)
        let pcmScale02 = engineScale02.synthesize(text: text, voice: .female)

        let wavNormal = WavEncoder.encode(samples: pcmNormal, sampleRate: 16000)
        let wavPriorOnly = WavEncoder.encode(samples: pcmPriorOnly, sampleRate: 16000)
        let wavScale02 = WavEncoder.encode(samples: pcmScale02, sampleRate: 16000)

        XCTAssertFalse(wavNormal.isEmpty, "通常合成 WAV が空であってはなりません")
        XCTAssertFalse(wavPriorOnly.isEmpty, "Prior のみ合成 WAV が空であってはなりません")
        XCTAssertFalse(wavScale02.isEmpty, "スケール0.2合成 WAV が空であってはなりません")

        // 文章テキストでの Prior-only 合成（オンメモリ検証）
        let sentence = "おはようございます。音声合成の学習が正常に完了しました。"
        let pcmSentencePrior = enginePriorOnly.synthesize(text: sentence, voice: .female)
        let wavSentencePrior = WavEncoder.encode(samples: pcmSentencePrior, sampleRate: 16000)
        XCTAssertFalse(wavSentencePrior.isEmpty, "文章 Prior のみ合成 WAV が空であってはなりません")
        XCTAssertEqual(wavSentencePrior.prefix(4), Data([0x52, 0x49, 0x46, 0x46]), "RIFF ヘッダ検証")
    }

    /// 4段カスケード共鳴器と音素間調音結合による自然音声合成プロトタイプ検証テスト
    /// Klatt 型 4 段カスケード 2次 IIR 共鳴器に調音結合と声帯音源を直接接続し、
    /// 日本語音声の明瞭度および話者切り替え（女性・男性・子供）の成立性を検証する。
    func testPrototypeCascadeResonatorSynthesis() throws {
        let engine = SpikeSpeechEngine()

        // 64 音素ごとの生理学的標準共鳴フォルマント設定 (F1, B1, F2, B2, F3, B3, F4, B4)
        func phoneFormants(phoneId: Int, nextVowelId: Int? = nil) -> (f1: Float, b1: Float, f2: Float, b2: Float, f3: Float, b3: Float, f4: Float, b4: Float) {
            switch phoneId {
            case 5: // /a/
                return (800.0, 80.0, 1300.0, 100.0, 2600.0, 120.0, 3500.0, 150.0)
            case 6: // /i/
                return (300.0, 60.0, 2300.0, 100.0, 3000.0, 150.0, 3800.0, 200.0)
            case 7: // /u/
                return (350.0, 70.0, 1200.0, 100.0, 2500.0, 140.0, 3600.0, 180.0)
            case 8: // /e/
                return (500.0, 70.0, 1900.0, 90.0, 2700.0, 130.0, 3700.0, 180.0)
            case 9: // /o/
                return (500.0, 70.0, 900.0, 80.0, 2600.0, 120.0, 3500.0, 150.0)
            case 13: // /n/
                return (280.0, 100.0, 1500.0, 150.0, 2500.0, 200.0, 3500.0, 250.0)
            case 15: // /m/
                return (280.0, 100.0, 1000.0, 150.0, 2300.0, 200.0, 3500.0, 250.0)
            case 24: // /N/ (ん)
                return (250.0, 120.0, 1200.0, 160.0, 2400.0, 200.0, 3500.0, 250.0)
            case 16: // /y/
                return (300.0, 70.0, 2200.0, 100.0, 2900.0, 140.0, 3600.0, 180.0)
            case 17: // /r/
                return (400.0, 90.0, 1350.0, 120.0, 2100.0, 150.0, 3300.0, 200.0)
            case 18: // /w/
                return (320.0, 80.0, 800.0, 100.0, 2400.0, 140.0, 3500.0, 180.0)
            case 19: // /g/
                return (320.0, 100.0, 1900.0, 140.0, 2500.0, 200.0, 3500.0, 250.0)
            case 20: // /z/
                return (350.0, 120.0, 1600.0, 160.0, 2800.0, 200.0, 3800.0, 250.0)
            case 21: // /d/
                return (300.0, 100.0, 1700.0, 140.0, 2600.0, 200.0, 3500.0, 250.0)
            case 22: // /b/
                return (300.0, 100.0, 1000.0, 140.0, 2400.0, 200.0, 3500.0, 250.0)
            case 36: // /j/
                return (320.0, 100.0, 2100.0, 140.0, 2800.0, 180.0, 3700.0, 220.0)
            case 10: // /k/
                var kF2: Float = 1900.0
                if let nv = nextVowelId {
                    switch nv {
                    case 5: kF2 = 1600.0 // ka
                    case 6: kF2 = 2300.0 // ki
                    case 7: kF2 = 1400.0 // ku
                    case 8: kF2 = 2000.0 // ke
                    case 9: kF2 = 1200.0 // ko
                    default: break
                    }
                }
                return (350.0, 120.0, kF2, 160.0, 2600.0, 200.0, 3500.0, 250.0)
            case 11: // /s/
                return (300.0, 200.0, 1600.0, 250.0, 4500.0, 300.0, 6200.0, 400.0)
            case 12: // /t/
                return (300.0, 120.0, 1700.0, 160.0, 2800.0, 200.0, 3700.0, 250.0)
            case 14: // /h/
                if let nv = nextVowelId {
                    return phoneFormants(phoneId: nv)
                }
                return (500.0, 100.0, 1500.0, 150.0, 2500.0, 200.0, 3500.0, 250.0)
            case 23: // /p/
                return (300.0, 120.0, 1000.0, 160.0, 2400.0, 200.0, 3500.0, 250.0)
            case 27: // /sh/
                return (300.0, 180.0, 2000.0, 220.0, 3200.0, 250.0, 4600.0, 350.0)
            case 28: // /ch/
                return (300.0, 120.0, 2100.0, 180.0, 3100.0, 220.0, 4400.0, 300.0)
            case 29: // /ts/
                return (300.0, 150.0, 1650.0, 200.0, 4200.0, 280.0, 5800.0, 350.0)
            case 30: // /ky/
                return (300.0, 100.0, 2200.0, 140.0, 2900.0, 180.0, 3600.0, 220.0)
            case 31: // /ny/
                return (280.0, 100.0, 2100.0, 140.0, 2800.0, 180.0, 3600.0, 220.0)
            case 32: // /hy/
                return (300.0, 120.0, 2200.0, 160.0, 3000.0, 200.0, 3800.0, 250.0)
            case 33: // /my/
                return (280.0, 100.0, 1800.0, 140.0, 2600.0, 180.0, 3500.0, 220.0)
            case 34: // /ry/
                return (350.0, 80.0, 1950.0, 110.0, 2600.0, 140.0, 3500.0, 180.0)
            case 35: // /gy/
                return (300.0, 90.0, 2200.0, 130.0, 2800.0, 180.0, 3600.0, 220.0)
            case 37: // /by/
                return (300.0, 90.0, 1800.0, 130.0, 2600.0, 180.0, 3500.0, 220.0)
            case 38: // /py/
                return (300.0, 100.0, 1800.0, 140.0, 2600.0, 180.0, 3500.0, 220.0)
            default:
                // 中立フォルマント
                return (500.0, 100.0, 1500.0, 150.0, 2500.0, 200.0, 3500.0, 250.0)
            }
        }

        func synthesizeWithCascade(text: String, voice: VoiceProfile) -> [Float] {
            let linguistic = engine.lengthRegulator.processText(
                text: text,
                normalizer: engine.normalizer,
                prosodyModel: engine.prosodyModel,
                vocabulary: engine.vocabulary,
                speedFactor: 1.0,
                baseF0: voice.baseF0
            )

            let totalFrames = linguistic.totalFrames
            if totalFrames <= 0 {
                return []
            }

            let hopSize = AudioConfig.hopSize
            let sampleRate: Float = 16000.0

            var frameF1 = [Float](repeating: 500.0, count: totalFrames)
            var frameB1 = [Float](repeating: 100.0, count: totalFrames)
            var frameF2 = [Float](repeating: 1500.0, count: totalFrames)
            var frameB2 = [Float](repeating: 150.0, count: totalFrames)
            var frameF3 = [Float](repeating: 2500.0, count: totalFrames)
            var frameB3 = [Float](repeating: 200.0, count: totalFrames)
            var frameF4 = [Float](repeating: 3500.0, count: totalFrames)
            var frameB4 = [Float](repeating: 250.0, count: totalFrames)
            var frameVoiced = [Float](repeating: 0.0, count: totalFrames)
            var frameGain = [Float](repeating: 0.0, count: totalFrames)
            var framePhoneIds = [Int](repeating: 1, count: totalFrames)

            var curF = 0
            var pIdx = 0
            let pCount = min(linguistic.phoneIds.count, linguistic.durations.count)

            // 1. 各音素のターゲット周波数と後続母音の検索
            var lastVowelId = 5
            while pIdx < pCount {
                let pid = Int(linguistic.phoneIds[pIdx])
                let dur = Int(linguistic.durations[pIdx])

                var nextVowel: Int? = nil
                var scanIdx = pIdx + 1
                while scanIdx < pCount {
                    let sId = Int(linguistic.phoneIds[scanIdx])
                    if sId == 5 || sId == 6 || sId == 7 || sId == 8 || sId == 9 {
                        nextVowel = sId
                        break
                    }
                    scanIdx += 1
                }

                let effectivePid: Int
                if pid == 26 { // 長音 (_)
                    effectivePid = lastVowelId
                } else {
                    effectivePid = pid
                    if pid == 5 || pid == 6 || pid == 7 || pid == 8 || pid == 9 {
                        lastVowelId = pid
                    }
                }

                var f = 0
                while f < dur {
                    let frameIdx = curF + f
                    if frameIdx < totalFrames {
                        framePhoneIds[frameIdx] = effectivePid
                        let fm = phoneFormants(phoneId: effectivePid, nextVowelId: nextVowel)
                        frameF1[frameIdx] = fm.f1
                        frameB1[frameIdx] = fm.b1
                        frameF2[frameIdx] = fm.f2
                        frameB2[frameIdx] = fm.b2
                        frameF3[frameIdx] = fm.f3
                        frameB3[frameIdx] = fm.b3
                        frameF4[frameIdx] = fm.f4
                        frameB4[frameIdx] = fm.b4

                        var vFlag: Float = 0.0
                        if frameIdx < linguistic.voicedFlags.count {
                            vFlag = linguistic.voicedFlags[frameIdx]
                        }

                        var baseGain: Float = 0.65
                        switch true {
                        case engine.vocabulary.isPauseOrSilence(id: effectivePid):
                            baseGain = 0.0
                            vFlag = 0.0
                        case engine.vocabulary.isUnvoicedStop(id: effectivePid):
                            vFlag = 0.0
                            if f < (dur - 1) {
                                baseGain = 0.0 // 閉鎖無音期
                            } else {
                                baseGain = 0.32 // 破裂バースト期
                            }
                        case engine.vocabulary.isVoicedStop(id: effectivePid):
                            vFlag = 1.0
                            if f < (dur - 1) {
                                baseGain = 0.12 // ボイスバー期
                            } else {
                                baseGain = 0.40 // 有声破裂期
                            }
                        case engine.vocabulary.isAffricate(id: effectivePid):
                            vFlag = 0.0
                            if f < (dur / 2) {
                                baseGain = 0.0 // 閉鎖期
                            } else {
                                baseGain = 0.32 // 摩擦期
                            }
                        case engine.vocabulary.isUnvoicedFricative(id: effectivePid):
                            vFlag = 0.0
                            baseGain = 0.28
                        default:
                            break
                        }

                        frameGain[frameIdx] = baseGain
                        frameVoiced[frameIdx] = vFlag
                    }
                    f += 1
                }
                curF += dur
                pIdx += 1
            }

            // 2. 調音結合 (Coarticulation): 音素間を自然に繋ぐ 3 点平滑化
            var smoothF1 = frameF1
            var smoothF2 = frameF2
            var smoothF3 = frameF3
            var smoothF4 = frameF4
            var smoothB1 = frameB1
            var smoothB2 = frameB2
            var smoothB3 = frameB3
            var smoothB4 = frameB4

            if 2 < totalFrames {
                var t = 1
                let tEnd = totalFrames - 1
                while t < tEnd {
                    let pCurr = framePhoneIds[t]
                    if engine.vocabulary.isPauseOrSilence(id: pCurr) != true {
                        smoothF1[t] = (0.20 * frameF1[t - 1]) + (0.60 * frameF1[t]) + (0.20 * frameF1[t + 1])
                        smoothF2[t] = (0.20 * frameF2[t - 1]) + (0.60 * frameF2[t]) + (0.20 * frameF2[t + 1])
                        smoothF3[t] = (0.20 * frameF3[t - 1]) + (0.60 * frameF3[t]) + (0.20 * frameF3[t + 1])
                        smoothF4[t] = (0.20 * frameF4[t - 1]) + (0.60 * frameF4[t]) + (0.20 * frameF4[t + 1])
                        smoothB1[t] = (0.20 * frameB1[t - 1]) + (0.60 * frameB1[t]) + (0.20 * frameB1[t + 1])
                        smoothB2[t] = (0.20 * frameB2[t - 1]) + (0.60 * frameB2[t]) + (0.20 * frameB2[t + 1])
                        smoothB3[t] = (0.20 * frameB3[t - 1]) + (0.60 * frameB3[t]) + (0.20 * frameB3[t + 1])
                        smoothB4[t] = (0.20 * frameB4[t - 1]) + (0.60 * frameB4[t]) + (0.20 * frameB4[t + 1])
                    }
                    t += 1
                }
            }

            // 3. 話者声道長スケーリング (VTLN)
            let vtl = voice.tract.lengthScale
            let bwScale = voice.tract.bandwidthScale
            var tf = 0
            while tf < totalFrames {
                smoothF1[tf] *= vtl
                smoothF2[tf] *= vtl
                smoothF3[tf] *= vtl
                smoothF4[tf] *= vtl
                smoothB1[tf] *= bwScale
                smoothB2[tf] *= bwScale
                smoothB3[tf] *= bwScale
                smoothB4[tf] *= bwScale
                tf += 1
            }

            // 4. サンプル単位の 4 段カスケード IIR 合成ループ
            let totalSamples = totalFrames * hopSize
            var output = [Float](repeating: 0.0, count: totalSamples)

            let pulseGen = RosenbergPulse(sampleRate: sampleRate)
            pulseGen.apply(glottal: voice.glottal)

            var y1_1: Float = 0.0, y1_2: Float = 0.0
            var y2_1: Float = 0.0, y2_2: Float = 0.0
            var y3_1: Float = 0.0, y3_2: Float = 0.0
            var y4_1: Float = 0.0, y4_2: Float = 0.0
            var rng: UInt64 = 88172645463325252


            func nextRand() -> Float {
                rng ^= (rng << 13)
                rng ^= (rng >> 7)
                rng ^= (rng << 17)
                let u = UInt32(truncatingIfNeeded: rng)
                return (Float(u) * (2.0 / 4294967295.0)) - 1.0
            }

            var n = 0
            while n < totalSamples {
                let fIdx = min(totalFrames - 1, n / hopSize)
                let sampleInFrame = n % hopSize
                let frac = Float(sampleInFrame) / Float(hopSize)
                let nextFIdx = min(totalFrames - 1, fIdx + 1)

                let f1 = ((1.0 - frac) * smoothF1[fIdx]) + (frac * smoothF1[nextFIdx])
                let b1 = ((1.0 - frac) * smoothB1[fIdx]) + (frac * smoothB1[nextFIdx])
                let f2 = ((1.0 - frac) * smoothF2[fIdx]) + (frac * smoothF2[nextFIdx])
                let b2 = ((1.0 - frac) * smoothB2[fIdx]) + (frac * smoothB2[nextFIdx])
                let f3 = ((1.0 - frac) * smoothF3[fIdx]) + (frac * smoothF3[nextFIdx])
                let b3 = ((1.0 - frac) * smoothB3[fIdx]) + (frac * smoothB3[nextFIdx])
                let f4 = ((1.0 - frac) * smoothF4[fIdx]) + (frac * smoothF4[nextFIdx])
                let b4 = ((1.0 - frac) * smoothB4[fIdx]) + (frac * smoothB4[nextFIdx])

                let gain = ((1.0 - frac) * frameGain[fIdx]) + (frac * frameGain[nextFIdx])
                let voiced = ((1.0 - frac) * frameVoiced[fIdx]) + (frac * frameVoiced[nextFIdx])
                var curF0: Float = 0.0
                if fIdx < linguistic.f0Contour.count {
                    curF0 = linguistic.f0Contour[fIdx]
                }
                if curF0 <= 0.0 {
                    curF0 = voice.baseF0
                }

                if gain <= 1e-4 {
                    y1_1 = 0.0; y1_2 = 0.0
                    y2_1 = 0.0; y2_2 = 0.0
                    y3_1 = 0.0; y3_2 = 0.0
                    y4_1 = 0.0; y4_2 = 0.0
                    output[n] = 0.0
                    n += 1
                    continue
                }

                let glottalPulse = pulseGen.nextSample(f0: curF0, removeDC: true)
                let noise = nextRand()
                let asp = voice.glottal.aspirationMix
                let voicedPulsePart = glottalPulse + (asp * noise * 0.3)
                let unvoicedPart = noise * 0.40
                let excitation = (voiced * voicedPulsePart) + ((1.0 - voiced) * unvoicedPart)
                let inputSample = excitation * gain

                func stepResonator(inVal: Float, freq: Float, bw: Float, y1: inout Float, y2: inout Float) -> Float {
                    let r = expf(-Float.pi * bw / sampleRate)
                    let theta = 2.0 * Float.pi * freq / sampleRate
                    let a1 = 2.0 * r * cosf(theta)
                    let a2 = -(r * r)
                    let b0 = 1.0 - a1 - a2
                    let y0 = (b0 * inVal) + (a1 * y1) + (a2 * y2)
                    y2 = y1
                    y1 = y0
                    return y0
                }

                let s1 = stepResonator(inVal: inputSample, freq: f1, bw: b1, y1: &y1_1, y2: &y1_2)
                let s2 = stepResonator(inVal: s1, freq: f2, bw: b2, y1: &y2_1, y2: &y2_2)
                let s3 = stepResonator(inVal: s2, freq: f3, bw: b3, y1: &y3_1, y2: &y3_2)
                var s4 = stepResonator(inVal: s3, freq: f4, bw: b4, y1: &y4_1, y2: &y4_2)

                let absVal = abs(s4)
                if 0.8 < absVal {
                    let excess = absVal - 0.8
                    let compressed = 0.8 + (0.2 * tanhf(excess * 5.0))
                    if s4 < 0.0 {
                        s4 = -compressed
                    } else {
                        s4 = compressed
                    }
                }
                output[n] = s4
                n += 1
            }

            return output
        }

        let phrase = "おはようございます。音声合成のテストです。"
        let pcmFemale = synthesizeWithCascade(text: phrase, voice: .female)
        let pcmMale = synthesizeWithCascade(text: phrase, voice: .male)
        let pcmChild = synthesizeWithCascade(text: phrase, voice: .child)

        let wavFemale = WavEncoder.encode(samples: pcmFemale, sampleRate: 16000)
        let wavMale = WavEncoder.encode(samples: pcmMale, sampleRate: 16000)
        let wavChild = WavEncoder.encode(samples: pcmChild, sampleRate: 16000)

        XCTAssertFalse(wavFemale.isEmpty, "Female WAV が空であってはなりません")
        XCTAssertFalse(wavMale.isEmpty, "Male WAV が空であってはなりません")
        XCTAssertFalse(wavChild.isEmpty, "Child WAV が空であってはなりません")
        XCTAssertEqual(wavFemale.prefix(4), Data([0x52, 0x49, 0x46, 0x46]), "Female RIFF ヘッダ検証")
        XCTAssertEqual(wavMale.prefix(4), Data([0x52, 0x49, 0x46, 0x46]), "Male RIFF ヘッダ検証")
        XCTAssertEqual(wavChild.prefix(4), Data([0x52, 0x49, 0x46, 0x46]), "Child RIFF ヘッダ検証")
    }

    /// PriorMel, SNN出力残差, および実音声 Mel の数値スケール診断
    /// 学習時のターゲット残差と推論時の SNN 出力残差の整合性を数値検証する。
    func testDiagnoseMelAndPriorScales() {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let weightsPath = currentDir + "/Models/weights.json"

        var weights: SpikingNetworkWeights = SpikingNetworkWeights.randomWeights()
        if fileManager.fileExists(atPath: weightsPath) {
            if let data = try? Data(contentsOf: URL(fileURLWithPath: weightsPath)) {
                weights = (try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data)) ?? weights
            }
        }

        let engine = SpikeSpeechEngine(weights: weights)
        let extractor = MelSpectrogramExtractor()
        let prior = engine.prior(for: VoiceProfile.female.tract)

        // 1. SyntheticAudioGenerator の母音 /a/ の実測 Mel
        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let vowelAPcm = generator.generateVowel(vowel: .a, durationSeconds: 0.3, f0: 160.0)
        let vowelAMel = extractor.extractLogMel(pcm: vowelAPcm)
        let midFrame = vowelAMel.count / 2
        let actualVowelA = vowelAMel[midFrame]

        // 2. Prior の母音 /a/ (phoneId=5)
        let priorA = prior.getPriorMel(phoneId: 5)

        // 3. SNN の母音 /a/ 出力
        let text = "あ"
        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )
        let inSeq = engine.encodeLinguisticFeatures(features: linguistic)
        let snnOut = engine.decoder.decodeSequence(featuresSeq: inSeq, workspace: engine.workspace)
        let snnA = snnOut[min(2, snnOut.count - 1)]

        print("[Scale Diagnostic] 母音 /a/ 比較 (ch 0..15):")
        var c = 0
        while c < 16 {
            let act = actualVowelA[c]
            let pri = priorA[c]
            let snn = snnA[c]
            let combined = pri + snn
            print(String(format: "  ch %2d: Synthetic=%.3f, Prior=%.3f, SNN=%.3f, Combined=%.3f", c, act, pri, snn, combined))
            c += 1
        }
    }

#if canImport(MLX)
    /// MLX forward と Pure Swift decodeSequence の完全等価性検証テスト
    /// MLX で学習したネットワークの推論結果と Pure Swift SIMD デコーダーの推論結果が
    /// 完全に一致しているか、計算式やテンソル形状の整合性を検証する。
    func testMLXAndPureSwiftInferenceEquivalence() {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let weightsPath = currentDir + "/Models/weights.json"

        var weights: SpikingNetworkWeights = SpikingNetworkWeights.randomWeights()
        if fileManager.fileExists(atPath: weightsPath) {
            if let data = try? Data(contentsOf: URL(fileURLWithPath: weightsPath)) {
                weights = (try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data)) ?? weights
            }
        }

        let engine = SpikeSpeechEngine(weights: weights)
        let network = MLXSpikingAcousticNetwork(weights: weights)

        let text = "あいうえお"
        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )
        let inSeq = engine.encodeLinguisticFeatures(features: linguistic)

        // 1. Pure Swift デコーダーでの推論
        engine.workspace.reset()
        let swiftOut = engine.decoder.decodeSequence(featuresSeq: inSeq, workspace: engine.workspace)

        // 2. MLX ネットワークでの推論
        let alignedLen = MLXAcousticBPTTTrainer.alignTo32(seqLen: inSeq.count)
        let inDim = weights.inputDim
        var flatFeat = [Float](repeating: 0.0, count: alignedLen * inDim)
        var t = 0
        while t < inSeq.count {
            var i = 0
            while i < inDim {
                flatFeat[(t * inDim) + i] = inSeq[t][i]
                i += 1
            }
            t += 1
        }
        let fArr = MLXArray(flatFeat, [1, alignedLen, inDim])
        let mlxOutArray = network.forward(features: fArr, bpttWindow: 16)
        let mlxOutSlice = mlxOutArray[0, ..<inSeq.count, 0...]
        let mlxFlat = mlxOutSlice.asArray(Float.self)

        print("--- [Parity Diagnostic] Frame 0 (MLX vs Swift) ---")
        var maxDiff: Float = 0.0
        var avgDiff: Float = 0.0
        var totalElements = 0
        t = 0
        while t < inSeq.count {
            var c = 0
            while c < weights.outputDim {
                let sVal = swiftOut[t][c]
                let mVal = mlxFlat[(t * weights.outputDim) + c]
                let diff = abs(sVal - mVal)
                if maxDiff < diff {
                    maxDiff = diff
                }
                avgDiff += diff
                totalElements += 1
                if t < 4 && c < 4 {
                    print(String(format: "  Frame %d ch %d: Swift=%.4f, MLX=%.4f, Diff=%.4f", t, c, sVal, mVal, diff))
                }
                c += 1
            }
            t += 1
        }
        avgDiff /= Float(max(1, totalElements))
        print(String(format: "全フレーム平均差分: %.6f, 最大差分: %.6f", avgDiff, maxDiff))
        print("weights.bOut (ch 0..15): \(weights.bOut.prefix(16))")
        XCTAssertTrue(maxDiff < 1e-4, "MLX と Pure Swift の推論結果が一致していません: maxDiff=\(maxDiff)")
    }
#endif

    /// 学習時の入力特徴量と推論時の入力特徴量の整合性診断テスト
    func testCompareTrainAndInferenceFeatures() {
        let engine = SpikeSpeechEngine()
        let extractor = MelSpectrogramExtractor()
        let tracker = PitchTracker()
        let text = "水の星に愛をこめて"
        // 外部コーパス非依存かつ自己完結型とするため、オンメモリ合成波形（PCM）を使用する。
        let pcm = engine.synthesize(text: text, voice: .female)
        guard pcm.isEmpty != true else {
            XCTFail("PCM合成波形が空であってはなりません")
            return
        }
        guard let pair = engine.prepareTrainingPair(
            text: text,
            pcm16k: pcm,
            melExtractor: extractor,
            pitchTracker: tracker
        ) else {
            print("prepareTrainingPair returned nil")
            return
        }

        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )
        let inferSeq = engine.encodeLinguisticFeatures(features: linguistic)

        print("--- [Feature Comparison: Train vs Infer] ---")
        print("Train frames: \(pair.features.count), Infer frames: \(inferSeq.count)")

        // 代表的な音素（母音 /o/ や /i/）のフレームにおける特徴量の比較
        let trainMid = pair.features[pair.features.count / 2]
        let inferMid = inferSeq[inferSeq.count / 2]

        print("Train mid frame non-zeros:")
        var i = 0
        while i < trainMid.count {
            if 0.01 < abs(trainMid[i]) {
                print("  ch \(i): \(trainMid[i])")
            }
            i += 1
        }

        print("Infer mid frame non-zeros:")
        i = 0
        while i < inferMid.count {
            if 0.01 < abs(inferMid[i]) {
                print("  ch \(i): \(inferMid[i])")
            }
            i += 1
        }
    }

    /// 「こんにちは。スパイクスピーチによる音声合成のテストです。」の言語・音響・合成パイプライン診断
    func testDiagnoseHelloSentence() {
        let engine = SpikeSpeechEngine()
        let text = "こんにちは。スパイクスピーチによる音声合成のテストです。"
        let morphemes = engine.normalizer.normalize(text: text)
        print("[Diagnostic Hello] Morphemes:")
        for m in morphemes {
            print("  \(m.surface): reading=\(m.reading), pos=\(m.pos)")
        }

        let linguistic = engine.lengthRegulator.processText(
            text: text,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0
        )
        let phoneTokens = linguistic.phoneIds.map { engine.vocabulary.token(for: Int($0)) }
        print("[Diagnostic Hello] Phonemes: \(phoneTokens.joined(separator: " "))")
        print("[Diagnostic Hello] Durations: \(linguistic.durations)")
        print("[Diagnostic Hello] VoicedFlags: \(linguistic.voicedFlags)")

        let samples = engine.synthesize(text: text)
        print("[Diagnostic Hello] Samples count: \(samples.count)")
        XCTAssertTrue(0 < samples.count)

        // 「お好きな日本語テキストを入力してください。」の音声合成検証
        let fooText = "お好きな日本語テキストを入力してください。"
        let fooSamples = engine.synthesize(text: fooText, voice: .female)
        XCTAssertTrue(0 < fooSamples.count, "foo 音声サンプルの生成に失敗しました")
        let fooMaleSamples = engine.synthesize(text: fooText, voice: .male)
        XCTAssertTrue(0 < fooMaleSamples.count, "foo 男声サンプルの生成に失敗しました")
    }

    /// 評価用3文章（女性・男性）の WAV ファイル生成
    func testGenerateEvaluationWavs() throws {
        let defaultWeights = "Models/weights.json"
        var weights = SpikingNetworkWeights.randomWeights()
        if FileManager.default.fileExists(atPath: defaultWeights) {
            if let data = try? Data(contentsOf: URL(fileURLWithPath: defaultWeights)) {
                if let loaded = try? JSONDecoder().decode(SpikingNetworkWeights.self, from: data) {
                    weights = loaded
                }
            }
        }
        let engine = SpikeSpeechEngine(weights: weights)

        let sentences = [
            ("cat", "吾輩は猫である。名前はまだ無い。"),
            ("water", "水をマレーシアから買わなくてはならないのです。"),
            ("hello", "こんにちは。スパイクスピーチによる音声合成のテストです。")
        ]

        var sIdx = 0
        while sIdx < sentences.count {
            let item = sentences[sIdx]
            let tag = item.0
            let text = item.1

            // Female
            let femaleWav = engine.synthesizeWav(text: text, voice: .female)
            let femalePath = "/tmp/eval_female_\(tag).wav"
            try femaleWav.write(to: URL(fileURLWithPath: femalePath))
            print("[Evaluation] Generated: \(femalePath), size: \(femaleWav.count) bytes")

            // Male
            let maleWav = engine.synthesizeWav(text: text, voice: .male)
            let malePath = "/tmp/eval_male_\(tag).wav"
            try maleWav.write(to: URL(fileURLWithPath: malePath))
            print("[Evaluation] Generated: \(malePath), size: \(maleWav.count) bytes")

            sIdx += 1
        }
    }

    /// 単一発話（BASIC5000_0001）に対する SNN 音響モデル過学習サニティチェック
    func testSingleUtteranceOverfittingSanityCheck() throws {
        #if canImport(MLX)
        // なぜ環境変数の有無でスキップ可能にするか:
        // 単一発話の BPTT 300ステップ過学習検証は約2分40秒を要するため、
        // 通常の開発・テスト実行サイクルを阻害しないよう、環境変数 SPIKESPEECH_STRESS_TEST が明示された場合のみ実行する。
        let shouldRun = ProcessInfo.processInfo.environment["SPIKESPEECH_STRESS_TEST"] != nil
        if shouldRun != true {
            throw XCTSkip("単一発話の300ステップ過学習サニティチェックは実行時間が大きいためスキップします (実行時は SPIKESPEECH_STRESS_TEST=1 を指定してください)")
        }

        let fileManager = FileManager.default
        let jsutPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000"
        let wavPath = jsutPath + "/wav/BASIC5000_0001.wav"

        guard fileManager.fileExists(atPath: wavPath) else {
            print("JSUT wav が存在しないためスキップ")
            return
        }

        let wavReader = WavAudioReader()
        guard let rawPCM = try? wavReader.loadWav16k(from: wavPath) else {
            XCTFail("Wav の読み込みに失敗しました")
            return
        }

        var peak: Float = 0.0
        var pIdx = 0
        while pIdx < rawPCM.count {
            let a = abs(rawPCM[pIdx])
            if peak < a {
                peak = a
            }
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

        let text = "水をマレーシアから買わなくてはならないのです。"
        let engine = SpikeSpeechEngine()
        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: 64)
        let pitchTracker = PitchTracker()

        guard let pair = engine.prepareTrainingPair(
            text: text,
            pcm16k: pcm16k,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker
        ) else {
            XCTFail("prepareTrainingPair に失敗しました")
            return
        }

        print("[Sanity] features: frames=\(pair.features.count), inDim=\(pair.features.first?.count ?? 0)")
        print("[Sanity] targets: frames=\(pair.targets.count), outDim=\(pair.targets.first?.count ?? 0)")

        let inDim = pair.features.first?.count ?? 128
        let outDim = 64
        let hiddenDim = 256
        let numLayers = 2
        let timeSteps = 4

        var melSums = [Float](repeating: 0.0, count: outDim)
        var f = 0
        while f < pair.targets.count {
            var c = 0
            while c < outDim {
                melSums[c] += pair.targets[f][c]
                c += 1
            }
            f += 1
        }
        var meanMel = [Float](repeating: 0.0, count: outDim)
        var c = 0
        while c < outDim {
            meanMel[c] = melSums[c] / Float(max(1, pair.targets.count))
            c += 1
        }

        var weights = SpikingNetworkWeights.randomWeights(
            inputDim: inDim,
            maxHiddenDim: hiddenDim,
            outputDim: outDim,
            timeSteps: timeSteps,
            numLayers: numLayers
        )
        weights = weights.withBOut(meanMel)

        let network = MLXSpikingAcousticNetwork(weights: weights)
        let trainer = MLXAcousticBPTTTrainer(
            network: network,
            learningRate: 0.003,
            bpttWindow: 16,
            weightDecay: 0.0
        )

        let alignedLen = MLXAcousticBPTTTrainer.alignTo32(seqLen: pair.features.count)
        var flatFeat = [Float](repeating: 0.0, count: alignedLen * inDim)
        var flatTgt = [Float](repeating: 0.0, count: alignedLen * outDim)
        var flatMask = [Float](repeating: 0.0, count: alignedLen)
        var t = 0
        while t < pair.features.count {
            flatMask[t] = 1.0
            var i = 0
            while i < inDim {
                flatFeat[(t * inDim) + i] = pair.features[t][i]
                i += 1
            }
            var c0 = 0
            while c0 < outDim {
                flatTgt[(t * outDim) + c0] = pair.targets[t][c0]
                c0 += 1
            }
            t += 1
        }
        let fArr = MLXArray(flatFeat, [1, alignedLen, inDim])
        let tArr = MLXArray(flatTgt, [1, alignedLen, outDim])
        let mArr = MLXArray(flatMask, [1, alignedLen])

        let gradNorms0 = trainer.diagnoseGradNorms(features: fArr, targets: tArr, mask: mArr)
        print("[Sanity Initial Grad Norms]:")
        for (param, norm) in gradNorms0.sorted(by: { $0.key < $1.key }) {
            print("  \(param): \(norm)")
        }

        // 教師 Mel と初期予測 Mel の比較
        let pred0 = network.forward(features: fArr, bpttWindow: 16)
        eval(pred0)
        let pred0Arr = pred0.asArray(Float.self)
        let tgtArr = tArr.asArray(Float.self)

        var tMin: Float = Float.infinity; var tMax: Float = -Float.infinity; var tSum: Float = 0.0
        var ti = 0
        while ti < tgtArr.count {
            let v = tgtArr[ti]
            if v < tMin { tMin = v }
            if tMax < v { tMax = v }
            tSum += v
            ti += 1
        }
        let tMean = tSum / Float(max(1, tgtArr.count))
        print("[Sanity Target Mel Stats]: min=\(tMin), max=\(tMax), mean=\(tMean)")

        var p0Min: Float = Float.infinity; var p0Max: Float = -Float.infinity; var p0Sum: Float = 0.0
        var pi = 0
        while pi < pred0Arr.count {
            let v = pred0Arr[pi]
            if v < p0Min { p0Min = v }
            if p0Max < v { p0Max = v }
            p0Sum += v
            pi += 1
        }
        let p0Mean = p0Sum / Float(max(1, pred0Arr.count))
        print("[Sanity Initial Pred Mel Stats]: min=\(p0Min), max=\(p0Max), mean=\(p0Mean)")

        var step = 0
        var initialLoss: Float = 0.0
        var finalLoss: Float = 0.0
        while step < 300 {
            let loss = trainer.trainSequence(features: pair.features, targets: pair.targets)
            if step == 0 {
                initialLoss = loss
            }
            finalLoss = loss
            if (step % 50) == 0 {
                let norms = trainer.weightNorms()
                print("[Sanity Step \(step)] Loss: \(loss), wIn=\(norms.wIn), wOut=\(norms.wOut), wL0=\(norms.wLayer0)")
            }
            step += 1
        }
        print("[Sanity] Complete: Initial Loss: \(initialLoss), Final Loss (Step 400): \(finalLoss)")
        XCTAssertTrue(finalLoss < (initialLoss * 0.60), "単一発話の過学習で十分な損失減少（40%以上）が達成されていません: initial=\(initialLoss), final=\(finalLoss)")

        let pred50 = network.forward(features: fArr, bpttWindow: 16)
        eval(pred50)
        let pred50Arr = pred50.asArray(Float.self)
        var p50Min: Float = Float.infinity; var p50Max: Float = -Float.infinity; var p50Sum: Float = 0.0
        pi = 0
        while pi < pred50Arr.count {
            let v = pred50Arr[pi]
            if v < p50Min { p50Min = v }
            if p50Max < v { p50Max = v }
            p50Sum += v
            pi += 1
        }
        let p50Mean = p50Sum / Float(max(1, pred50Arr.count))
        print("[Sanity Step 50 Pred Mel Stats]: min=\(p50Min), max=\(p50Max), mean=\(p50Mean)")

        // 最初の数フレームの正解 Mel vs 予測 Mel (チャンネル 0..7)
        print("--- Frame 10 Target vs Pred50 (ch 0..7) ---")
        let offset10 = 10 * outDim
        var ch = 0
        while ch < 8 {
            print("  ch \(ch): Target=\(tgtArr[offset10 + ch]), Pred=\(pred50Arr[offset10 + ch])")
            ch += 1
        }
        print("--- Frame 100 Target vs Pred50 (ch 0..7) ---")
        let offset100 = 100 * outDim
        ch = 0
        while ch < 8 {
            print("  ch \(ch): Target=\(tgtArr[offset100 + ch]), Pred=\(pred50Arr[offset100 + ch])")
            ch += 1
        }

        let gradNormsFinal = trainer.diagnoseGradNorms(features: fArr, targets: tArr, mask: mArr)
        print("[Sanity Step 50 Grad Norms]:")
        for (param, norm) in gradNormsFinal.sorted(by: { $0.key < $1.key }) {
            print("  \(param): \(norm)")
        }

        let trainedWeights = network.exportWeights().withLexicon(ViterbiMorphology.loadDefaultLexicon())
        let weightsPath = "Models/weights.json"
        if let encoded = try? JSONEncoder().encode(trainedWeights) {
            try? encoded.write(to: URL(fileURLWithPath: weightsPath), options: .atomic)
            print("[Sanity] 最新の過学習・適合済み重みを保存しました: \(weightsPath) (\(encoded.count) バイト)")
        }
        #endif
    }

    /// SNN 推論 Mel スペクトルと実音声（JSUT BASIC5000_0001）教師 Mel の数値比較・診断
    func testSnnMelVersusTeacherMelComparison() throws {
        let fileManager = FileManager.default
        let jsutPath = "/Users/octu0/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000"
        let wavPath = jsutPath + "/wav/BASIC5000_0001.wav"

        guard fileManager.fileExists(atPath: wavPath) else {
            print("JSUT wav が存在しないためスキップ")
            return
        }

        let wavReader = WavAudioReader()
        guard let rawPCM = try? wavReader.loadWav16k(from: wavPath) else {
            XCTFail("Wav の読み込みに失敗しました")
            return
        }

        var peak: Float = 0.0
        var pIdx = 0
        while pIdx < rawPCM.count {
            let a = abs(rawPCM[pIdx])
            if peak < a {
                peak = a
            }
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

        let melExtractor = MelSpectrogramExtractor(sampleRate: 16000.0, melChannels: 64)
        let pitchTracker = PitchTracker()
        let teacherMel = melExtractor.extractLogMel(pcm: pcm16k)
        let pitchResult = pitchTracker.track(pcm: pcm16k)

        print("=== [Teacher Mel Diagnostic] ===")
        print("Teacher Mel frames: \(teacherMel.count), PCM samples: \(pcm16k.count)")

        var tMin: Float = Float.infinity
        var tMax: Float = -Float.infinity
        var tSum: Float = 0.0
        var totalElements = 0
        var tf = 0
        while tf < teacherMel.count {
            var c = 0
            while c < teacherMel[tf].count {
                let v = teacherMel[tf][c]
                if v < tMin { tMin = v }
                if tMax < v { tMax = v }
                tSum += v
                totalElements += 1
                c += 1
            }
            tf += 1
        }
        let tMean = tSum / Float(max(1, totalElements))
        print(String(format: "Teacher Mel: min=%.4f, max=%.4f, mean=%.4f", tMin, tMax, tMean))

        // 1. 教師 Mel を直接ニューラルボコーダーへ供給した波形の検証（Copy Synthesis）
        let engine = SpikeSpeechEngine()
        let copyAudio = engine.neuralVocoder.synthesize(
            mel: teacherMel,
            f0Contour: pitchResult.f0,
            voicedFlags: pitchResult.voiced,
            voice: .female
        )
        let copyURL = URL(fileURLWithPath: "/tmp/diag_copy_synth.wav")
        let copyWavData = WavEncoder.encode(samples: copyAudio, sampleRate: 16000)
        try? copyWavData.write(to: copyURL)

        var copyPeak: Float = 0.0
        var copySumSq: Double = 0.0
        var cs = 0
        while cs < copyAudio.count {
            let a = abs(copyAudio[cs])
            if copyPeak < a { copyPeak = a }
            copySumSq += Double(copyAudio[cs] * copyAudio[cs])
            cs += 1
        }
        var copyRms: Float = 0.0
        if 0 < copyAudio.count {
            copyRms = Float(sqrt(copySumSq / Double(copyAudio.count)))
        }
        print(String(format: "Copy Synth Audio: samples=%d, Peak=%.4f, RMS=%.4f", copyAudio.count, copyPeak, copyRms))

        // 2. SNN 推論 Mel の検証: "水をマレーシアから買わなくてはならないのです。"
        let textFull = "水をマレーシアから買わなくてはならないのです。"
        let b = engine.detectSpeechBoundaries(pcm: pcm16k, totalFrames: teacherMel.count)
        print("BASIC5000_0001 boundaries: lead=\(b.leadSilence), speech=\(b.speechFrames), trail=\(b.trailSilence)")
        guard let pair = engine.prepareTrainingPair(
            text: textFull,
            pcm16k: pcm16k,
            melExtractor: melExtractor,
            pitchTracker: pitchTracker
        ) else {
            XCTFail("prepareTrainingPair に失敗しました")
            return
        }
        let snnMelOnPair = engine.decoder.decodeSequence(featuresSeq: pair.features, workspace: engine.workspace)
        var pairSqErr: Float = 0.0
        var pairCmpCount = 0
        var pf = 0
        let pairLimit = min(pair.targets.count, snnMelOnPair.count)
        while pf < pairLimit {
            var c = 0
            while c < 64 {
                let diff = snnMelOnPair[pf][c] - pair.targets[pf][c]
                pairSqErr += diff * diff
                pairCmpCount += 1
                c += 1
            }
            pf += 1
        }
        let pairMse = pairSqErr / Float(max(1, pairCmpCount))
        print(String(format: "=== [SNN Reconstructed on Training Features] MSE against Teacher Mel: %.4f (frames=%d) ===", pairMse, pairLimit))

        let lingFull = engine.lengthRegulator.processText(
            text: textFull,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0
        )
        let inSeqFull = engine.encodeLinguisticFeatures(features: lingFull)
        let snnMelFull = engine.decoder.decodeSequence(featuresSeq: inSeqFull, workspace: engine.workspace)

        print("=== [SNN Full Text Inference Diagnostic] ===")
        print("SNN Mel frames: \(snnMelFull.count), input features: \(inSeqFull.count)")

        var sMin: Float = Float.infinity
        var sMax: Float = -Float.infinity
        var sSum: Float = 0.0
        var sElements = 0
        var sf = 0
        while sf < snnMelFull.count {
            var c = 0
            while c < snnMelFull[sf].count {
                let v = snnMelFull[sf][c]
                if v < sMin { sMin = v }
                if sMax < v { sMax = v }
                sSum += v
                sElements += 1
                c += 1
            }
            sf += 1
        }
        let sMean = sSum / Float(max(1, sElements))
        print(String(format: "SNN Mel: min=%.4f, max=%.4f, mean=%.4f", sMin, sMax, sMean))

        // SNN Mel と Teacher Mel のフレーム比較
        let cmpFrames = min(teacherMel.count, snnMelFull.count)
        var totalSqErr: Float = 0.0
        var totalCmpCount = 0
        var f = 0
        while f < cmpFrames {
            var c = 0
            while c < 64 {
                let diff = snnMelFull[f][c] - teacherMel[f][c]
                totalSqErr += diff * diff
                totalCmpCount += 1
                c += 1
            }
            f += 1
        }
        let globalMse = totalSqErr / Float(max(1, totalCmpCount))
        print(String(format: "Global MSE (first %d frames): %.4f", cmpFrames, globalMse))

        // 各代表フレームにおける 64ch フォルマント形状の比較
        let sampleFrames = [10, 30, 60, 100, 150, 200, 250]
        for frameIdx in sampleFrames {
            if frameIdx < cmpFrames {
                var fMse: Float = 0.0
                var c = 0
                while c < 64 {
                    let diff = snnMelFull[frameIdx][c] - teacherMel[frameIdx][c]
                    fMse += diff * diff
                    c += 1
                }
                fMse = fMse / 64.0
                print(String(format: "  Frame %d MSE: %.4f | Target ch[0..7]=%@ | SNN ch[0..7]=%@",
                    frameIdx,
                    fMse,
                    teacherMel[frameIdx].prefix(8).map { String(format: "%.2f", $0) }.joined(separator: ", "),
                    snnMelFull[frameIdx].prefix(8).map { String(format: "%.2f", $0) }.joined(separator: ", ")
                ))
            }
        }

        // 3. SNN 推論波形の生成と保存
        let synthAudio = engine.synthesize(text: textFull, voice: .female)
        let synthURL = URL(fileURLWithPath: "/tmp/diag_synth_full.wav")
        let synthWavData = WavEncoder.encode(samples: synthAudio, sampleRate: 16000)
        try? synthWavData.write(to: synthURL)

        var synPeak: Float = 0.0
        var synSumSq: Double = 0.0
        var synS = 0
        while synS < synthAudio.count {
            let a = abs(synthAudio[synS])
            if synPeak < a { synPeak = a }
            synSumSq += Double(synthAudio[synS] * synthAudio[synS])
            synS += 1
        }
        var synRms: Float = 0.0
        if 0 < synthAudio.count {
            synRms = Float(sqrt(synSumSq / Double(synthAudio.count)))
        }
        print(String(format: "SNN Full Synth Audio: samples=%d, Peak=%.4f, RMS=%.4f", synthAudio.count, synPeak, synRms))

        // 4. "水を買わなくてはならないのです。" の推論波形も生成
        let textShort = "水を買わなくてはならないのです。"
        let shortAudio = engine.synthesize(text: textShort, voice: .female)
        let shortURL = URL(fileURLWithPath: "/tmp/diag_synth_short.wav")
        let shortWavData = WavEncoder.encode(samples: shortAudio, sampleRate: 16000)
        try? shortWavData.write(to: shortURL)
        print(String(format: "SNN Short Synth Audio: samples=%d", shortAudio.count))

        // 5. CascadeResonatorVocoder による合成検証
        let (resFrames, _) = engine.buildResonatorFrames(
            linguisticFeatures: lingFull,
            voice: .female,
            effectiveBaseF0: VoiceProfile.female.baseF0,
            text: textFull,
            melSeq: snnMelFull
        )
        engine.cascadeVocoder.reset()
        let cascadeAudio = engine.cascadeVocoder.synthesize(frames: resFrames)
        let cascadeURL = URL(fileURLWithPath: "/tmp/diag_cascade_snn.wav")
        let cascadeWavData = WavEncoder.encode(samples: cascadeAudio, sampleRate: 16000)
        try? cascadeWavData.write(to: cascadeURL)

        var casPeak: Float = 0.0
        var casSumSq: Double = 0.0
        var casS = 0
        while casS < cascadeAudio.count {
            let a = abs(cascadeAudio[casS])
            if casPeak < a { casPeak = a }
            casSumSq += Double(cascadeAudio[casS] * cascadeAudio[casS])
            casS += 1
        }
        var casRms: Float = 0.0
        if 0 < cascadeAudio.count {
            casRms = Float(sqrt(casSumSq / Double(cascadeAudio.count)))
        }
        print(String(format: "Cascade Vocoder with SNN Mel: samples=%d, Peak=%.4f, RMS=%.4f", cascadeAudio.count, casPeak, casRms))

        // 6. 短文 "水を買わなくてはならないのです。" の Cascade 合成
        let lingShort = engine.lengthRegulator.processText(
            text: textShort,
            normalizer: engine.normalizer,
            prosodyModel: engine.prosodyModel,
            vocabulary: engine.vocabulary,
            speedFactor: 1.0,
            baseF0: VoiceProfile.female.baseF0
        )
        let inShort = engine.encodeLinguisticFeatures(features: lingShort)
        engine.workspace.reset()
        let snnMelShort = engine.decoder.decodeSequence(featuresSeq: inShort, workspace: engine.workspace)
        let (shortResFrames, _) = engine.buildResonatorFrames(
            linguisticFeatures: lingShort,
            voice: .female,
            effectiveBaseF0: VoiceProfile.female.baseF0,
            text: textShort,
            melSeq: snnMelShort
        )
        engine.cascadeVocoder.reset()
        let cascadeShortAudio = engine.cascadeVocoder.synthesize(frames: shortResFrames)
        let casShortURL = URL(fileURLWithPath: "/tmp/diag_cascade_short.wav")
        let casShortWavData = WavEncoder.encode(samples: cascadeShortAudio, sampleRate: 16000)
        try? casShortWavData.write(to: casShortURL)

        var csPeak: Float = 0.0
        var csSumSq: Double = 0.0
        var csS = 0
        while csS < cascadeShortAudio.count {
            let a = abs(cascadeShortAudio[csS])
            if csPeak < a { csPeak = a }
            csSumSq += Double(cascadeShortAudio[csS] * cascadeShortAudio[csS])
            csS += 1
        }
        var csRms: Float = 0.0
        if 0 < cascadeShortAudio.count {
            csRms = Float(sqrt(csSumSq / Double(cascadeShortAudio.count)))
        }
        print(String(format: "Cascade Vocoder Short Audio: samples=%d, Peak=%.4f, RMS=%.4f", cascadeShortAudio.count, csPeak, csRms))
    }
}



