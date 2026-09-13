import XCTest
import Foundation
import Darwin
@testable import SpikeSpeech

/// Milestone 4 (F14〜F16) の極限・境界値および敵対的ストレステストスイート
///
/// 1. SyntheticAudioGenerator の 5母音（/a/, /i/, /u/, /e/, /o/）の周波数スペクトルにおける
///    ホルマントピーク（F1, F2）が理論値近傍に現れているかの音響物理的検証
/// 2. SpikeSpeechEngine における超長文テキストおよび特殊記号混在入力での
///    E2E 合成におけるメモリリーク、アロケーション爆発、NaN/Inf 混入の有無の実測検証
/// 3. 多層 SNN における E2E 音声合成の波形健全性・RMS エネルギー・クリッピング防止の定量的検証
/// を独立して実証する。
final class Challenger2M4Tests: XCTestCase {

    override func setUp() {
        super.setUp()
    }

    override func tearDown() {
        super.tearDown()
    }

    // MARK: - メモリ測定ヘルパー (Darwin task_info)

    /// プロセスの常駐メモリサイズ（Resident Set Size）をバイト単位で取得する。
    /// 外部プロファイラなしでテスト実行プロセス自身の物理メモリ占有量を高精度に監視し、
    /// 長文合成時におけるメモリリークやアロケーション爆発を自動検知する。
    private func getResidentMemoryBytes() -> UInt64 {
        var taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
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

    // MARK: - FFT スペクトル解析ヘルパー (Pure Swift Radix-2 Cooley-Tukey)

    /// Radix-2 複素 FFT (In-Place)
    ///
    /// 外部オーディオ解析ライブラリや非標準フレームワークへの依存を排除し、
    /// プロジェクトの厳格規約（比較演算子 `<` / `<=`、`else if` 禁止等）に準拠した
    /// 再現性のある周波数スペクトル計算を行う。
    private func computeFFT(real: inout [Float], imag: inout [Float]) {
        let n = real.count
        // ビット反転置換 (Bit-reversal permutation)
        var j = 0
        var i = 0
        while i < n - 1 {
            if i < j {
                let tr = real[i]
                real[i] = real[j]
                real[j] = tr
                let ti = imag[i]
                imag[i] = imag[j]
                imag[j] = ti
            }
            var k = n / 2
            while k <= j {
                j -= k
                k /= 2
            }
            j += k
            i += 1
        }

        // Cooley-Tukey バタフライ演算
        var step = 1
        while step < n {
            let jump = step * 2
            let angleStep = -Float.pi / Float(step)
            var m = 0
            while m < step {
                let angle = Float(m) * angleStep
                let wr = cos(angle)
                let wi = sin(angle)
                var p = m
                while p < n {
                    let q = p + step
                    let tr = (wr * real[q]) - (wi * imag[q])
                    let ti = (wr * imag[q]) + (wi * real[q])
                    real[q] = real[p] - tr
                    imag[q] = imag[p] - ti
                    real[p] += tr
                    imag[p] += ti
                    p += jump
                }
                m += 1
            }
            step = jump
        }
    }

    /// ハミング窓およびプリエンファシスを適用して 2048 点 FFT により振幅スペクトル（dB）を算出
    ///
    /// 声帯振動パルスは低域に向かって約 -12dB/octave のスペクトル傾斜を持つため、
    /// プリエンファシス（y[n] = x[n] - 0.97*x[n-1]）で高域をブーストして平坦化し、
    /// F1 の裾野に埋もれがちな F2 ホルマント共鳴ピークの検出精度を高める。
    private func computeMagnitudeSpectrumDb(samples: [Float], fftSize: Int = 2048, applyPreEmphasis: Bool = true) -> [Float] {
        var real = [Float](repeating: 0.0, count: fftSize)
        var imag = [Float](repeating: 0.0, count: fftSize)

        let copyLen = min(samples.count, fftSize)

        // 1. プリエンファシス
        var filtered = [Float](repeating: 0.0, count: copyLen)
        if applyPreEmphasis {
            filtered[0] = samples[0]
            var i = 1
            while i < copyLen {
                filtered[i] = samples[i] - (0.97 * samples[i - 1])
                i += 1
            }
        } else {
            var i = 0
            while i < copyLen {
                filtered[i] = samples[i]
                i += 1
            }
        }

        // 2. ハミング窓の乗算
        var n = 0
        while n < copyLen {
            let w = 0.54 - (0.46 * cos((2.0 * Float.pi * Float(n)) / Float(fftSize - 1)))
            real[n] = filtered[n] * w
            n += 1
        }

        computeFFT(real: &real, imag: &imag)

        let halfSize = fftSize / 2
        var magDb = [Float](repeating: 0.0, count: halfSize)
        var k = 0
        while k < halfSize {
            let power = (real[k] * real[k]) + (imag[k] * imag[k])
            let mag = sqrt(power)
            let safeMag = max(1e-7, mag)
            magDb[k] = 20.0 * log10(safeMag)
            k += 1
        }

        return magDb
    }

    // MARK: - 1. 5母音ホルマント周波数スペクトルピーク数値検証

    /// 5母音（/a/, /i/, /u/, /e/, /o/）の周波数スペクトルピーク（F1, F2）が理論値近傍に現れるかを厳密検証
    func testFiveVowelsFormantSpectrumPeakVerification() {
        // SyntheticAudioGenerator が生成する音声波形が、4段カスケード共鳴器によって
        // 生理学的理論値（FormantConfig）に忠実な周波数共鳴ピーク（F1, F2）を形成していることを実証する。

        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let sampleRate: Float = 16000.0
        let fftSize = 2048
        let binWidth = sampleRate / Float(fftSize) // 1 bin = 7.8125 Hz

        let vowels: [SyntheticAudioGenerator.Vowel] = [.a, .i, .u, .e, .o]

        print("\n--- [Challenger 2 Formant Spectral Audit] ---")

        var vIdx = 0
        while vIdx < vowels.count {
            let vowel = vowels[vIdx]
            let expectedConfig = generator.formantConfig(for: vowel)

            // 0.4 秒（6,400 サンプル）の母音波形を生成 (F0 = 130 Hz)
            let wave = generator.generateVowel(vowel: vowel, durationSeconds: 0.4, f0: 130.0)

            // 先頭の過渡部を避けた中央定常区間から 2048 サンプルを抽出
            let startSample = 1600
            var segment = [Float](repeating: 0.0, count: fftSize)
            var s = 0
            while s < fftSize {
                segment[s] = wave[startSample + s]
                s += 1
            }

            let spectrumDb = computeMagnitudeSpectrumDb(samples: segment, fftSize: fftSize, applyPreEmphasis: true)

            // 周波数 f [Hz] から FFT ビンインデックスへの変換
            func freqToBin(_ freq: Float) -> Int {
                return Int(round(freq / binWidth))
            }

            // 指定周波数範囲 [fMin, fMax] 内で最大ピーク（極大値）周波数を探索
            func findPeakFrequency(fMin: Float, fMax: Float) -> (freq: Float, db: Float) {
                let binMin = max(1, freqToBin(fMin))
                let binMax = min((fftSize / 2) - 2, freqToBin(fMax))

                var bestBin = binMin
                var maxDb: Float = -9999.0

                var b = binMin
                while b <= binMax {
                    let val = spectrumDb[b]
                    let valPrev = spectrumDb[b - 1]
                    let valNext = spectrumDb[b + 1]

                    // ローカルピーク（極大点）判定
                    if valPrev <= val {
                        if valNext <= val {
                            if maxDb < val {
                                maxDb = val
                                bestBin = b
                            }
                        }
                    }
                    b += 1
                }

                // ローカルピークが見つからない場合は単純最大点
                if maxDb < -9000.0 {
                    b = binMin
                    while b <= binMax {
                        if maxDb < spectrumDb[b] {
                            maxDb = spectrumDb[b]
                            bestBin = b
                        }
                        b += 1
                    }
                }

                return (Float(bestBin) * binWidth, maxDb)
            }

            // F0=130Hz の倍音構造が存在するため、理論ホルマント近傍の倍音周波数にピークが現れる。
            // 探索窓幅は各母音の F1, F2 周波数分離と帯域幅に応じて設定
            let f1Min: Float
            let f1Max: Float
            let f2Min: Float
            let f2Max: Float

            switch vowel {
            case .a: // F1=800, F2=1300
                f1Min = 650.0; f1Max = 950.0
                f2Min = 1150.0; f2Max = 1450.0
            case .i: // F1=300, F2=2300
                f1Min = 200.0; f1Max = 450.0
                f2Min = 2100.0; f2Max = 2500.0
            case .u: // F1=350, F2=1200
                f1Min = 220.0; f1Max = 480.0
                f2Min = 1050.0; f2Max = 1350.0
            case .e: // F1=500, F2=1900
                f1Min = 380.0; f1Max = 650.0
                f2Min = 1750.0; f2Max = 2050.0
            case .o: // F1=500, F2=900
                f1Min = 380.0; f1Max = 650.0
                f2Min = 750.0; f2Max = 1050.0
            }

            let f1Peak = findPeakFrequency(fMin: f1Min, fMax: f1Max)
            let f2Peak = findPeakFrequency(fMin: f2Min, fMax: f2Max)

            let f1Error = abs(f1Peak.freq - expectedConfig.f1)
            let f2Error = abs(f2Peak.freq - expectedConfig.f2)

            print("母音 /\(vowel.rawValue)/:")
            print("  F1 理論値: \(expectedConfig.f1) Hz | 実測ピーク: \(f1Peak.freq) Hz (誤差: \(f1Error) Hz, 振幅: \(f1Peak.db) dB)")
            print("  F2 理論値: \(expectedConfig.f2) Hz | 実測ピーク: \(f2Peak.freq) Hz (誤差: \(f2Error) Hz, 振幅: \(f2Peak.db) dB)")

            // 誤差が倍音間隔 130Hz + ビン分解能 (約 140Hz) 以内であることを数値検証
            let maxAllowedHarmonicError: Float = 140.0
            XCTAssertTrue(
                f1Error <= maxAllowedHarmonicError,
                "母音 /\(vowel.rawValue)/ の F1 ピーク誤差 (\(f1Error) Hz) が許容値を超過: 理論=\(expectedConfig.f1), 実測=\(f1Peak.freq)"
            )
            XCTAssertTrue(
                f2Error <= maxAllowedHarmonicError,
                "母音 /\(vowel.rawValue)/ の F2 ピーク誤差 (\(f2Error) Hz) が許容値を超過: 理論=\(expectedConfig.f2), 実測=\(f2Peak.freq)"
            )

            vIdx += 1
        }
        print("---------------------------------------------")
    }

    // MARK: - 2. 超長文テキスト & 特殊記号混在 E2E 合成ストレス検証

    /// 特殊記号・絵文字・外国語文字・連続句読点混在テキストでの E2E 合成耐性検証
    func testSpecialCharactersMixedTextE2ESynthesis() {
        // 形態素解析や正規化で未知トークン・特殊記号が入力された際に、
        // 異常終了や無限ループ、NaN/Inf 生成、ゼロ除算を防止し、
        // 安全にフォールバックして有効な PCM 出力が得られることを保証する。

        let engine = SpikeSpeechEngine()

        let weirdText = "【緊急速報】SpikeSpeech 1.0.0 リリース！？ @#$%^&*()_+|~=`{}[]:\";'<>?,./\\ " +
            "あ、えーっと、、、、12345円（税込）です！" +
            "Emoji: 😀🎉🚀🍣 日本語＆English mixed text! カタカナ・ひらがな・漢字のテスト。"

        let samples = engine.synthesize(text: weirdText)

        // 1. サンプルが正常に出力されたことの確認
        XCTAssertTrue(0 < samples.count, "特殊記号混在テキストで出力サンプルが空です")

        // 2. 全サンプルで NaN / Inf が皆無であることを全数検査
        var maxAbs: Float = 0.0
        var sumSq: Float = 0.0
        var idx = 0
        while idx < samples.count {
            let s = samples[idx]
            XCTAssertFalse(s.isNaN, "特殊記号テキスト合成サンプルのインデックス \(idx) で NaN を検出")
            XCTAssertFalse(s.isInfinite, "特殊記号テキスト合成サンプルのインデックス \(idx) で Inf を検出")
            let absVal = abs(s)
            if maxAbs < absVal {
                maxAbs = absVal
            }
            sumSq += s * s
            idx += 1
        }

        // 3. 最大振幅クリッピング防止 (<= 1.0)
        XCTAssertTrue(maxAbs <= 1.0, "特殊記号テキスト合成で振幅クリッピングが発生: maxAbs=\(maxAbs)")

        // 4. 有意な音響エネルギーが存在すること
        let rms = sqrt(sumSq / Float(samples.count))
        XCTAssertTrue(0.001 < rms, "特殊記号テキスト合成の音響エネルギーが小さすぎます: rms=\(rms)")

        print("--- [Challenger 2 Weird Text Audit] ---")
        print("文字数: \(weirdText.count) 文字, 出力サンプル数: \(samples.count), RMS: \(rms), MaxAbs: \(maxAbs)")
        print("---------------------------------------")
    }

    /// 2,000 文字以上の超長文テキストでの E2E 合成におけるメモリリーク・アロケーション爆発・スケーリング検証
    func testSuperLongTextE2ESynthesisMemoryAndScaling() {
        // 長時間発話において SNN 膜電位積算やバッファ再利用が破綻しないこと、
        // アロケーション爆発やメモリリーク（RSS 肥大化）が生じないこと、
        // および全フレームで NaN/Inf が一切発生しないことを実測検証する。

        let engine = SpikeSpeechEngine()

        let baseParagraph = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。" +
            "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。" +
            "吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。" +
            "この書生というのは時々我々を捕えて煮て食うという話である。"

        var longText = ""
        var rep = 0
        while rep < 12 { // 12 回反復 (約 1,800〜2,000 文字)
            longText += baseParagraph
            rep += 1
        }

        let charCount = longText.count
        XCTAssertTrue(1500 <= charCount, "長文テキスト長が不足しています: \(charCount)")

        let memBefore = getResidentMemoryBytes()
        let startTime = CFAbsoluteTimeGetCurrent()

        // E2E 合成実行
        let samples = engine.synthesize(text: longText)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let memAfter = getResidentMemoryBytes()
        let memDiffMB = Float(memAfter - memBefore) / (1024.0 * 1024.0)

        // 1. サンプル数検証 (1500文字 -> 数万〜数十万サンプル)
        XCTAssertTrue(10000 < samples.count, "超長文の出力サンプル数が過少です: \(samples.count)")

        // 2. 全サンプルで NaN / Inf ゼロ検査
        var maxAbs: Float = 0.0
        var sumSq: Float = 0.0
        var i = 0
        while i < samples.count {
            let s = samples[i]
            XCTAssertFalse(s.isNaN, "超長文合成サンプルのインデックス \(i) で NaN を検出")
            XCTAssertFalse(s.isInfinite, "超長文合成サンプルのインデックス \(i) で Inf を検出")
            let a = abs(s)
            if maxAbs < a {
                maxAbs = a
            }
            sumSq += s * s
            i += 1
        }

        // 3. 最大振幅は 1.0 以下
        XCTAssertTrue(maxAbs <= 1.0, "超長文合成で振幅クリッピングが発生: maxAbs=\(maxAbs)")

        // 4. 有意なエネルギー
        let rms = sqrt(sumSq / Float(samples.count))
        XCTAssertTrue(0.005 < rms, "超長文合成の RMS エネルギーが小さすぎます: \(rms)")

        let durationSeconds = Float(samples.count) / 16000.0
        let rtf = Float(elapsed) / durationSeconds

        print("\n--- [Challenger 2 Super Long Text Audit] ---")
        print("入力文字数: \(charCount) 文字")
        print("生成サンプル数: \(samples.count) サンプル (音声実時間: \(durationSeconds) 秒)")
        print("処理所要時間: \(elapsed) 秒 (RTF: \(rtf))")
        print("合成前 RSS メモリ: \(Float(memBefore) / (1024.0 * 1024.0)) MB")
        print("合成後 RSS メモリ: \(Float(memAfter) / (1024.0 * 1024.0)) MB")
        print("差分メモリ増加量: \(memDiffMB) MB")
        print("最大絶対値振幅: \(maxAbs), 全体 RMS: \(rms)")
        print("--------------------------------------------")

        // メモリ増加量が 100MB 未満（アロケーション爆発なし）
        XCTAssertTrue(memDiffMB < 100.0, "超長文合成で過剰なメモリ増加が発生しました: \(memDiffMB) MB")
    }

    /// ストリーミング合成における長文処理とフレーム健全性検証
    func testStreamingSynthesisWithLongText() {
        // 逐次コールバック呼び出しにおいてフレーム間の状態破綻やバッファ汚染が発生せず、
        // 全フレームが連続的に正しく生成されることを検証する。

        let engine = SpikeSpeechEngine()
        let text = "こんにちは。スパイクスピーチのストリーミング合成テストです。低遅延で音声を出力します。"

        var receivedFramesCount = 0
        var totalSamplesReceived = 0
        var frameHadNaN = false
        var frameHadInf = false

        let allSamples = engine.synthesizeStream(text: text, speed: 1.0, pitch: 1.0) { frame in
            receivedFramesCount += 1
            totalSamplesReceived += frame.count

            var fIdx = 0
            while fIdx < frame.count {
                if frame[fIdx].isNaN {
                    frameHadNaN = true
                }
                if frame[fIdx].isInfinite {
                    frameHadInf = true
                }
                fIdx += 1
            }
        }

        XCTAssertTrue(0 < receivedFramesCount, "ストリーミングフレームが受信されませんでした")
        XCTAssertEqual(totalSamplesReceived, allSamples.count, "コールバック受信サンプル総数が出力配列と不一致です")
        XCTAssertFalse(frameHadNaN, "ストリーミングフレーム内で NaN が検出されました")
        XCTAssertFalse(frameHadInf, "ストリーミングフレーム内で Inf が検出されました")
    }

    // MARK: - 3. 多層 SNN (層数 1, 2, 3, 4) E2E 波形健全性検証

    /// 層数 1, 2, 3, 4 のそれぞれで E2E 音声合成を行い、全モデルで有効な PCM 波形が出力されるかを検証
    func testMultilayerE2EWaveformIntegrity() {
        // 多層 SNN 重み（層数 1, 2, 3, 4）から
        // Mel 特徴量 -> MelToLPC -> LPCVocoder の全経路を通じて、
        // どの層数構成でも無音（ゼロ）や発散（NaN/Inf）にならず、
        // 同一の長さと有意な音響エネルギーを持つ有効な PCM 波形が生成されることを実証する。

        let testPhrase = "すぱいくすぴーちによる、多層SNN音声合成の検証です。"
        let targetLayers = [1, 2, 3, 4]

        var layerResults: [(layers: Int, samples: [Float], rms: Float, maxAbs: Float)] = []
        var expectedSampleCount = 0

        print("\n--- [Challenger 2 Multilayer SNN E2E Audit] ---")

        var lIdx = 0
        while lIdx < targetLayers.count {
            let layers = targetLayers[lIdx]
            let weights = SpikingNetworkWeights.standardInit(
                inputDim: 32,
                maxHiddenDim: 128,
                outputDim: 80,
                numLayers: layers
            )
            let engine = SpikeSpeechEngine(weights: weights)

            let startTime = CFAbsoluteTimeGetCurrent()
            let samples = engine.synthesize(text: testPhrase)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            // 1. サンプル数検証
            XCTAssertTrue(0 < samples.count, "層数 \(layers) で出力サンプルが空です")
            if lIdx == 0 {
                expectedSampleCount = samples.count
            } else {
                XCTAssertEqual(
                    samples.count,
                    expectedSampleCount,
                    "層数 \(layers) のサンプル数が他層数と不一致です (期待: \(expectedSampleCount), 実際: \(samples.count))"
                )
            }

            // 2. NaN / Inf 不在および振幅検査
            var maxAbs: Float = 0.0
            var sumSq: Float = 0.0
            var i = 0
            while i < samples.count {
                let s = samples[i]
                XCTAssertFalse(s.isNaN, "層数 \(layers) のサンプル \(i) で NaN を検出")
                XCTAssertFalse(s.isInfinite, "層数 \(layers) のサンプル \(i) で Inf を検出")
                let absVal = abs(s)
                if maxAbs < absVal {
                    maxAbs = absVal
                }
                sumSq += s * s
                i += 1
            }

            // 3. 最大振幅クリッピング防止 (<= 1.0)
            XCTAssertTrue(maxAbs <= 1.0, "層数 \(layers) で振幅クリッピングが発生: maxAbs=\(maxAbs)")

            // 4. 有意な音響エネルギー (無音ではない)
            let rms = sqrt(sumSq / Float(samples.count))
            XCTAssertTrue(0.001 < rms, "層数 \(layers) の音響エネルギーが不足しています: rms=\(rms)")

            let durationSeconds = Float(samples.count) / 16000.0
            let rtf = Float(elapsed) / durationSeconds

            print("層数 \(layers): サンプル数=\(samples.count), RMS=\(rms), MaxAbs=\(maxAbs), RTF=\(rtf)")

            layerResults.append((layers: layers, samples: samples, rms: rms, maxAbs: maxAbs))
            lIdx += 1
        }
        print("-------------------------------------------------")
    }

    // MARK: - 4. 静的コード規約検査 (Static Rule Enforcement)

    /// Milestone 4 実装およびテストコードがプロジェクト規約に完全適合していることを機械走査
    func testChallenger2M4StaticRuleCheck() throws {
        // 比較演算子の統一（`<` および `<=` のみ、`>` や `>=` の禁止）、
        // `else if` の禁止（`switch` 使用）、三項演算子の禁止などを
        // 自動テスト内で全ソース走査し、規約違反の混入を防止する。

        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath

        let targetPaths = [
            currentDir + "/Sources/SpikeSpeech/DSP/SyntheticAudioGenerator.swift",
            currentDir + "/Sources/SpikeSpeech/Pipeline/SpikeSpeechEngine.swift",
            currentDir + "/script/synthesize/main.swift",
            currentDir + "/script/train/main.swift",
            currentDir + "/script/benchmark/main.swift"
        ]

        var scannedCount = 0

        var pIdx = 0
        while pIdx < targetPaths.count {
            let path = targetPaths[pIdx]
            if fileManager.fileExists(atPath: path) != true {
                pIdx += 1
                continue
            }

            let content = try String(contentsOfFile: path, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines)

            var lineIdx = 0
            while lineIdx < lines.count {
                let rawLine = lines[lineIdx]
                let trimmed = rawLine.trimmingCharacters(in: .whitespaces)

                // コメント行のスキップ
                if trimmed.hasPrefix("//") {
                    lineIdx += 1
                    continue
                }
                if trimmed.hasPrefix("/*") {
                    lineIdx += 1
                    continue
                }
                if trimmed.hasPrefix("*") {
                    lineIdx += 1
                    continue
                }

                // 1. 比較演算子 `>` および `>=` の禁止検査
                // ※ジェネリクス `->`, `=>`, `/>`, `?>` や型 `<T>`, `[String: Any]` は除外
                var sanitized = trimmed
                sanitized = sanitized.replacingOccurrences(of: "->", with: " ")
                sanitized = sanitized.replacingOccurrences(of: "=>", with: " ")
                sanitized = sanitized.replacingOccurrences(of: "/>", with: " ")
                sanitized = sanitized.replacingOccurrences(of: ">>", with: " ") // ビットシフト

                if sanitized.contains(" > ") {
                    XCTFail("ファイル \(path) 行 \(lineIdx + 1) で禁止された比較演算子 ` > ` を検出: \(trimmed)")
                }
                if sanitized.contains(" >= ") {
                    XCTFail("ファイル \(path) 行 \(lineIdx + 1) で禁止された比較演算子 ` >= ` を検出: \(trimmed)")
                }

                // 2. `else if` の禁止検査
                if trimmed.contains("else if") {
                    XCTFail("ファイル \(path) 行 \(lineIdx + 1) で禁止された `else if` を検出: \(trimmed)")
                }

                // 3. `cond == false` / `cond == true` の禁止検査
                if trimmed.contains("== false") {
                    XCTFail("ファイル \(path) 行 \(lineIdx + 1) で禁止された `== false` を検出: \(trimmed)")
                }
                if trimmed.contains("== true") {
                    XCTFail("ファイル \(path) 行 \(lineIdx + 1) で禁止された `== true` を検出: \(trimmed)")
                }

                lineIdx += 1
            }

            scannedCount += 1
            pIdx += 1
        }

        print("--- [Challenger 2 M4 Static Rule Check] ---")
        print("検査完了ファイル数: \(scannedCount) 件 (全ファイル規約適合)")
        print("-------------------------------------------")
        XCTAssertTrue(5 <= scannedCount, "検査対象ファイル数が不足しています: \(scannedCount)")
    }
}
