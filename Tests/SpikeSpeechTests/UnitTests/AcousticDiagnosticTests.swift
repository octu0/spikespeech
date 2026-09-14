import XCTest
import Foundation
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

        let inputSeq = engine.encodeLinguisticFeatures(features: linguisticFeatures, voice: .female)
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

        // ピークが 0.8 以下でクリップしていないこと
        XCTAssertTrue(maxAbs <= 0.85, "合成音声がリミッター上限にクリップしています: maxAbs=\(maxAbs)")
        // 有意な音響エネルギー（RMS > 0.1）を有していること
        XCTAssertTrue(0.1 < rms, "合成音声の音響エネルギーが小さすぎます: rms=\(rms)")
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
        //    かつ耳障りな過大ホワイトノイズ（<= 0.05）になっていないことを検証する。
        XCTAssertTrue(frameRmsList[kBurstFrame] <= 0.05, "破裂バースト期 Frame \(kBurstFrame) のエネルギーが過大です: RMS=\(frameRmsList[kBurstFrame])")

        // 3. 後続母音 /o/ の定常部で豊かな母音フォルマント共鳴（0.10 <= RMS）が立ち上がっていることを検証する。
        let oVowelFrame = min(frameCount - 1, kBurstFrame + 3)
        XCTAssertTrue(0.10 <= frameRmsList[oVowelFrame], "母音 /o/ のフォルマント共鳴エネルギーが不足しています: RMS=\(frameRmsList[oVowelFrame])")
    }

    /// JSUT 実音声のエネルギー分布、前後無音区間、Prior フォルマントピークの診断
    func testDiagnoseJSUTSampleAndPrior() throws {
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let candidates = [
            currentDir + "/../spiketrans/.tmp/jsut_ver1.1/basic5000",
            fileManager.homeDirectoryForCurrentUser.path + "/workspace/spiketrans/.tmp/jsut_ver1.1/basic5000"
        ]
        var corpusDir: String? = nil
        var cIdx = 0
        while cIdx < candidates.count {
            let cand = candidates[cIdx]
            if fileManager.fileExists(atPath: cand + "/transcript_utf8.txt") {
                corpusDir = cand
                break
            }
            cIdx += 1
        }

        guard let cDir = corpusDir else {
            print("[JSUT Diagnostic] コーパスディレクトリが見つかりません。")
            return
        }

        let wavPath = cDir + "/wav/BASIC5000_0001.wav"
        let reader = WavAudioReader()
        let pcm = try reader.loadWav16k(from: wavPath)
        print("[JSUT Diagnostic] BASIC5000_0001 pcm count: \(pcm.count) (\(Float(pcm.count) / 16000.0) 秒)")

        let extractor = MelSpectrogramExtractor()
        let mel = extractor.extractLogMel(pcm: pcm)
        print("[JSUT Diagnostic] 実音声 Mel フレーム数: \(mel.count)")

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

        print("[JSUT Diagnostic] 先頭無音フレーム数: \(leadSilence) (\(leadSilence * 10) ms)")
        print("[JSUT Diagnostic] 発話有音フレーム数: \(mel.count - leadSilence - trailSilence) (\((mel.count - leadSilence - trailSilence) * 10) ms)")
        print("[JSUT Diagnostic] 末尾無音フレーム数: \(trailSilence) (\(trailSilence * 10) ms)")

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
        print("[JSUT Diagnostic] 無音フレーム対数 Mel 平均: \(leadMelSum / Float(AudioConfig.melChannels))")
        print("[JSUT Diagnostic] 有音フレーム対数 Mel 平均: \(voicedMelSum / Float(AudioConfig.melChannels))")

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
        for (pId, vName) in vowels {
            let pMel = engine.acousticPrior.getPriorMel(phoneId: pId)
            var coeffs = [Float](repeating: 0.0, count: AudioConfig.lpcOrder)
            let g = engine.melToLPC.convert(mel: pMel, isLogMel: true, outCoeffs: &coeffs)
            let peaks = findPeaks(coeffs: coeffs)
            print("[JSUT Diagnostic] Prior [\(vName)] (phone=\(pId)): gain=\(g), formant peaks=\(peaks)")
        }
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

    /// 実際に生成された /tmp/konnichiwa_generated.wav の全フレーム波形・音素・有声度・F0・スペクトル精密診断
    func testDiagnoseActualGeneratedWav() throws {
        let wavPath = "/tmp/konnichiwa_generated.wav"
        let reader = WavAudioReader()
        let pcm = try reader.loadWav16k(from: wavPath)
        print("=== [/tmp/konnichiwa_generated.wav 精密診断レポート] ===")
        print("総サンプル数: \(pcm.count) (\(Float(pcm.count) / 16000.0) 秒)")

        let engine = SpikeSpeechEngine(weights: SpikingNetworkWeights.randomWeights())
        let text = "こんにちは"
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
}


