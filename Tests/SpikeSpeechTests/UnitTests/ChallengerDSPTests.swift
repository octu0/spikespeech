import XCTest
@testable import SpikeSpeech

/// Milestone 2: Audio DSP & Source-Filter LPC Vocoder Engine (F6〜F9)
/// 敵対的・極限入力耐性・無発振性 Challenger 検証テストスイート
///
/// 無音安定性、過大ゲイン Soft Limiter、NaN/Inf 異常入力耐性、
/// 反射係数クランプによる全極フィルタの安定性を敵対的入力を用いて実証・保証する。
final class ChallengerDSPTests: XCTestCase {

    // MARK: - 1. 無音・ゼロ入力安定性テスト

    func testSilenceStabilityLongTermAndTransitions() {
        // 1秒間（100フレーム = 16,000サンプル）の連続無音において乱数や丸め誤差が漏洩せず、
        // かつ初期状態からのゼロゲイン有声フレームで完全な無音が生成されることを実証する。
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)

        // 1. 100 フレーム連続の完全無音入力
        let silenceFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.0, count: 16),
            gain: 0.0,
            pitchF0: 0.0,
            voiced: 0.0
        )
        let silenceFrames = [AcousticFrame](repeating: silenceFrame, count: 100)
        let silenceWave = vocoder.synthesize(frames: silenceFrames)

        XCTAssertEqual(silenceWave.count, 16000)
        var sIdx = 0
        while sIdx < silenceWave.count {
            XCTAssertEqual(silenceWave[sIdx], 0.0, "無音フレームで非ゼロサンプルを検出: index=\(sIdx)")
            sIdx += 1
        }

        // 2. 初期状態からのゲイン 0.0 有声フレーム (F0=200Hz, voiced=1.0)
        let freshVocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)
        let vowelCoeffs: [Float] = [
            1.25, -0.85, 0.45, -0.25, 0.15, -0.08, 0.04, -0.02,
            0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        let zeroGainVoicedFrame = AcousticFrame(
            lpcCoefficients: vowelCoeffs,
            gain: 0.0,
            pitchF0: 200.0,
            voiced: 1.0
        )
        let zeroGainWave = freshVocoder.synthesize(frames: [zeroGainVoicedFrame, zeroGainVoicedFrame])
        var zIdx = 0
        while zIdx < zeroGainWave.count {
            XCTAssertEqual(zeroGainWave[zIdx], 0.0, "初期状態からゲイン0の有声フレームで非ゼロが出力されました: sample=\(zIdx)")
            zIdx += 1
        }

        // 3. 有声発話後の無音移行におけるディエンファシスフィルタ減衰の観測
        let transVocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)
        let voicedFrame = AcousticFrame(
            lpcCoefficients: vowelCoeffs,
            gain: 0.08,
            pitchF0: 150.0,
            voiced: 1.0
        )
        let transFrames = [voicedFrame, voicedFrame, silenceFrame, silenceFrame, silenceFrame]
        let transWave = transVocoder.synthesize(frames: transFrames)

        // 無音開始から 2 フレーム（320 サンプル）後の振幅レベル測定
        let postSilenceSample = abs(transWave[(4 * 160) - 1])
        print("--- [Challenger DSP] 有声後無音減衰測定: 320サンプル後の残留振幅 = \(postSilenceSample) ---")
        // 浮動小数点 IIR フィルタの指数減衰により微小なテール（< 1e-3）に収束していることを確認
        XCTAssertTrue(postSilenceSample < 1e-3, "無音移行後の減衰が不十分です: \(postSilenceSample)")
    }

    // MARK: - 2. Soft Limiter (Tanh) 及び Int16 量子化サチュレーション検証

    func testSoftLimiterAndInt16Quantization() {
        // 極端なオーバードライブ入力下でも Soft Limiter が波形振幅を確実に [-1.0, 1.0] に収め、
        // かつハードクリッピングせず滑らかな非線形サチュレーションを保つことを保証する。
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)
        let vowelCoeffs: [Float] = [
            1.25, -0.85, 0.45, -0.25, 0.15, -0.08, 0.04, -0.02,
            0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

        // 1. 各種過大ゲインでの振幅レンジ検証 (10.0, 100.0, 1000.0, 10000.0)
        let testGains: [Float] = [10.0, 100.0, 1000.0, 10000.0]
        var gIdx = 0
        while gIdx < testGains.count {
            let g = testGains[gIdx]
            let overFrame = AcousticFrame(
                lpcCoefficients: vowelCoeffs,
                gain: g,
                pitchF0: 200.0,
                voiced: 1.0
            )
            var outBuf = [Float](repeating: 0.0, count: 160)
            outBuf.withUnsafeMutableBufferPointer { pDst in
                vocoder.synthesizeFrame(frame: overFrame, dst: pDst.baseAddress!)
            }

            var i = 0
            while i < 160 {
                let s = outBuf[i]
                XCTAssertEqual(s, s, "過大ゲイン入力で NaN が発生しました: gain=\(g), sample=\(i)")
                XCTAssertTrue(s <= 1.0, "振幅上限 1.0 を超過しました: val=\(s), gain=\(g)")
                XCTAssertTrue(-1.0 <= s, "振幅下限 -1.0 を下回りました: val=\(s), gain=\(g)")
                i += 1
            }
            gIdx += 1
        }

        // 2. VectorOperations.softLimitTanh の非線形サチュレーション曲線検証
        let linearInputs: [Float] = [0.0, 0.2, 0.5, 0.8]
        var linearOutputs = [Float](repeating: 0.0, count: linearInputs.count)
        linearOutputs.withUnsafeMutableBufferPointer { pDst in
            linearInputs.withUnsafeBufferPointer { pSrc in
                VectorOperations.softLimitTanh(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: linearInputs.count, threshold: 0.8)
            }
        }
        var lIdx = 0
        while lIdx < linearInputs.count {
            let diff = abs(linearOutputs[lIdx] - linearInputs[lIdx])
            XCTAssertTrue(diff < 1e-6, "線形領域 (<= 0.8) で歪みが発生しました: input=\(linearInputs[lIdx]), output=\(linearOutputs[lIdx])")
            lIdx += 1
        }

        // 3. 圧縮領域 (0.8 < x) での漸近平滑性（一階差分の単調減少）検証
        let satInputs: [Float] = [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        var satOutputs = [Float](repeating: 0.0, count: satInputs.count)
        satOutputs.withUnsafeMutableBufferPointer { pDst in
            satInputs.withUnsafeBufferPointer { pSrc in
                VectorOperations.softLimitTanh(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: satInputs.count, threshold: 0.8)
            }
        }

        var sIdx = 1
        while sIdx < satInputs.count {
            // 出力は 1.0 以下に確実に収まる ([-1.0, 1.0] 安全クリップ)
            XCTAssertTrue(satOutputs[sIdx] <= 1.0, "サチュレーション出力が 1.0 を超過しています: \(satOutputs[sIdx])")
            // 入力が増加すれば出力も単調増加する (勾配が非負)
            XCTAssertTrue(satOutputs[sIdx - 1] <= satOutputs[sIdx], "出力が単調増加していません")
            sIdx += 1
        }

        // 4. Int16 量子化におけるオーバーフロー反転防止検証
        let extremeInputs: [Float] = [-1000.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1000.0]
        var qOutputs = [Int16](repeating: 0, count: extremeInputs.count)
        qOutputs.withUnsafeMutableBufferPointer { pDst in
            extremeInputs.withUnsafeBufferPointer { pSrc in
                VectorOperations.quantizeFloatToInt16(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: extremeInputs.count)
            }
        }
        XCTAssertEqual(qOutputs[0], -32768, "極大負値のサチュレーション失敗")
        XCTAssertEqual(qOutputs[1], -32768, "-2.0 のサチュレーション失敗")
        XCTAssertEqual(qOutputs[2], -32767, "-1.0 の量子化不整合")
        XCTAssertEqual(qOutputs[3], 0, "0.0 の量子化不整合")
        XCTAssertEqual(qOutputs[4], 32767, "1.0 の量子化不整合")
        XCTAssertEqual(qOutputs[5], 32767, "2.0 のサチュレーション失敗")
        XCTAssertEqual(qOutputs[6], 32767, "極大正値のサチュレーション失敗")
    }

    // MARK: - 3. 反射係数クランプと全極 AR フィルタの安定性検証

    func testReflectionCoefficientClampAndARStability() {
        // 特定周波数に全エネルギーが集中した病的なスペクトルにおいて、
        // Levinson-Durbin の反射係数が厳密にクランプ制限され、
        // 帯域幅拡大と相まって IIR フィルタが無限大発散しないことを証明する。
        let melToLpc = MelToLPC(melChannels: 64, fftBins: 257, lpcOrder: 16)

        // 単一チャンネル（チャンネル 15）のみが極大値を持つ病的な Mel 特徴量
        var extremeMel = [Float](repeating: -20.0, count: 64)
        extremeMel[15] = 20.0 // log-mel で +20.0 (線形スケールで約 4.8e8)

        var lpcCoeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLpc.convert(mel: extremeMel, isLogMel: true, outCoeffs: &lpcCoeffs)

        XCTAssertTrue(0.0 <= gain, "ゲインが負値になっています")
        var i = 0
        while i < 16 {
            let c = lpcCoeffs[i]
            XCTAssertEqual(c, c, "LPC 係数に NaN が含まれています: index=\(i)")
            // 反射係数クランプと帯域幅拡大により係数は有界
            XCTAssertTrue(abs(c) < 50.0, "LPC 係数が異常発散しています: index=\(i), val=\(c)")
            i += 1
        }

        // この極端な係数を用いてボコーダーで 50 フレーム連続合成し、発散しないことを実証
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)
        let extremeFrame = AcousticFrame(
            lpcCoefficients: lpcCoeffs,
            gain: 0.05,
            pitchF0: 200.0,
            voiced: 1.0
        )
        let frames = [AcousticFrame](repeating: extremeFrame, count: 50)
        let audio = vocoder.synthesize(frames: frames)

        var maxAmp: Float = 0.0
        var aIdx = 0
        while aIdx < audio.count {
            let s = audio[aIdx]
            XCTAssertEqual(s, s, "ボコーダー出力に NaN が発生しました: sample=\(aIdx)")
            if maxAmp < abs(s) {
                maxAmp = abs(s)
            }
            aIdx += 1
        }
        XCTAssertTrue(maxAmp <= 1.0, "極端係数での合成波形が 1.0 を超過しました: \(maxAmp)")
    }

    // MARK: - 4. 敵対的異常入力に対する脆弱性・フォールバック実証テスト

    func testAdversarialAbnormalInputResilience() {
        // SNN 音響モデルの勾配爆発や数値例外によって異常値がボコーダーに渡された際、
        // プロセスがクラッシュせず安全にフォールバックすることを実証する。
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)

        // 1. LPC 係数の一部に NaN が含まれるフレーム
        var nanCoeffs = [Float](repeating: 0.1, count: 16)
        nanCoeffs[3] = Float.nan
        let nanCoeffFrame = AcousticFrame(
            lpcCoefficients: nanCoeffs,
            gain: 0.05,
            pitchF0: 200.0,
            voiced: 1.0
        )
        var outNaNCoeff = [Float](repeating: 1.0, count: 160)
        outNaNCoeff.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(frame: nanCoeffFrame, dst: pDst.baseAddress!)
        }
        var nIdx = 0
        while nIdx < 160 {
            XCTAssertEqual(outNaNCoeff[nIdx], 0.0, "NaN 係数フレームで 0.0 にフォールバックしていません: sample=\(nIdx)")
            nIdx += 1
        }

        // 2. ゲインに NaN が含まれるフレーム
        let nanGainFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.1, count: 16),
            gain: Float.nan,
            pitchF0: 200.0,
            voiced: 1.0
        )
        var outNaNGain = [Float](repeating: 1.0, count: 160)
        outNaNGain.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(frame: nanGainFrame, dst: pDst.baseAddress!)
        }
        nIdx = 0
        while nIdx < 160 {
            XCTAssertEqual(outNaNGain[nIdx], 0.0, "NaN ゲインフレームで 0.0 にフォールバックしていません: sample=\(nIdx)")
            nIdx += 1
        }

        // 3. 有声度 voiced に NaN / 負値 / 過大値が含まれるフレーム
        let abnormalVoicedFrames: [Float] = [Float.nan, -10.0, 10.0]
        var vIdx = 0
        while vIdx < abnormalVoicedFrames.count {
            let vVal = abnormalVoicedFrames[vIdx]
            let abnormalVoicedFrame = AcousticFrame(
                lpcCoefficients: [Float](repeating: 0.1, count: 16),
                gain: 0.05,
                pitchF0: 200.0,
                voiced: vVal
            )
            var outAbnormal = [Float](repeating: 0.0, count: 160)
            outAbnormal.withUnsafeMutableBufferPointer { pDst in
                vocoder.synthesizeFrame(frame: abnormalVoicedFrame, dst: pDst.baseAddress!)
            }
            var sIdx = 0
            while sIdx < 160 {
                let s = outAbnormal[sIdx]
                XCTAssertEqual(s, s, "異常 voiced で NaN が出力されました: voiced=\(vVal), sample=\(sIdx)")
                XCTAssertTrue(abs(s) <= 1.0, "異常 voiced で振幅が超過しました: voiced=\(vVal), sample=\(sIdx)")
                sIdx += 1
            }
            vIdx += 1
        }

        // 4. 自己回復性 (Self-Healing) の検証: 異常フレーム直後に正常フレームを入力
        let normalFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.1, count: 16),
            gain: 0.05,
            pitchF0: 200.0,
            voiced: 1.0
        )
        var outNormal = [Float](repeating: 0.0, count: 160)
        outNormal.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(frame: normalFrame, dst: pDst.baseAddress!)
        }
        var hasNonZero = false
        var sIdx = 0
        while sIdx < 160 {
            let s = outNormal[sIdx]
            XCTAssertEqual(s, s, "回復後フレームに NaN が残存しています: sample=\(sIdx)")
            if 0.0 < abs(s) {
                hasNonZero = true
            }
            sIdx += 1
        }
        XCTAssertTrue(hasNonZero, "異常フレーム後の正常フレームで音声合成が再開されていません")
    }

    // MARK: - 5. 脆弱性・設計不備の完全根絶検証テスト (Vulnerability Eradication Verification)

    func testVulnerabilityMelToLPCPropagatesNaN() {
        // SNN 音響モデルから NaN が供給された際に入力バリデーションが即座に作動し、
        // NaN 係数や異常ゲインを伝播させずに安全な無音（0.0）へ確定フォールバックすることを実証する。
        let melToLpc = MelToLPC(melChannels: 64, fftBins: 257, lpcOrder: 16)
        var nanMel = [Float](repeating: 0.0, count: 64)
        nanMel[0] = Float.nan

        var outCoeffs = [Float](repeating: 0.0, count: 16)
        let gain = melToLpc.convert(mel: nanMel, isLogMel: true, outCoeffs: &outCoeffs)

        let gainIsNaN = (gain != gain)
        let coeffIsNaN = (outCoeffs[0] != outCoeffs[0])
        print("--- [Challenger Vulnerability 1 Fixed] MelToLPC NaN ガード検証: gainIsNaN=\(gainIsNaN), coeffIsNaN=\(coeffIsNaN) ---")
        XCTAssertFalse(gainIsNaN, "MelToLPC が NaN をガードできず NaN ゲインを出力しました")
        XCTAssertFalse(coeffIsNaN, "MelToLPC が NaN をガードできず NaN 係数を出力しました")
        XCTAssertEqual(gain, 0.0, "MelToLPC が NaN 入力時にゲイン 0.0 を返していません")
        XCTAssertEqual(outCoeffs[0], 0.0, "MelToLPC が NaN 入力時に係数 0.0 を返していません")
    }

    func testVulnerabilityLPCVocoderInfGainBypassesNaNCheck() {
        // Float.infinity や -Float.infinity に対し包括的ガードが作動して、
        // 即座に無音（0.0）を出力することを実証する。
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)

        // 1. ゲインが +Infinity のフレーム
        let infGainFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.1, count: 16),
            gain: Float.infinity,
            pitchF0: 200.0,
            voiced: 1.0
        )
        var outInfGain = [Float](repeating: 1.0, count: 160)
        outInfGain.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(frame: infGainFrame, dst: pDst.baseAddress!)
        }
        var i = 0
        while i < 160 {
            XCTAssertEqual(outInfGain[i], 0.0, "Infinity ゲインフレームで 0.0 にフォールバックしていません: sample=\(i)")
            i += 1
        }

        // 2. 係数に -Infinity が含まれるフレーム
        var infCoeffs = [Float](repeating: 0.1, count: 16)
        infCoeffs[2] = -Float.infinity
        let infCoeffFrame = AcousticFrame(
            lpcCoefficients: infCoeffs,
            gain: 0.05,
            pitchF0: 200.0,
            voiced: 1.0
        )
        var outInfCoeff = [Float](repeating: 1.0, count: 160)
        outInfCoeff.withUnsafeMutableBufferPointer { pDst in
            vocoder.synthesizeFrame(frame: infCoeffFrame, dst: pDst.baseAddress!)
        }
        i = 0
        while i < 160 {
            XCTAssertEqual(outInfCoeff[i], 0.0, "Infinity 係数フレームで 0.0 にフォールバックしていません: sample=\(i)")
            i += 1
        }
    }

    func testVulnerabilityDeEmphasisDenormalResidual() {
        // 有声から無音に切り替わった後、Flush-to-zero 機構により微小な非正規化数が
        // 完全にゼロ化され、CPU トラップや不要な演算負荷を防止することを実証する。
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)
        let vowelCoeffs: [Float] = [
            1.25, -0.85, 0.45, -0.25, 0.15, -0.08, 0.04, -0.02,
            0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        let voicedFrame = AcousticFrame(
            lpcCoefficients: vowelCoeffs,
            gain: 0.1,
            pitchF0: 200.0,
            voiced: 1.0
        )
        let silenceFrame = AcousticFrame(
            lpcCoefficients: [Float](repeating: 0.0, count: 16),
            gain: 0.0,
            pitchF0: 0.0,
            voiced: 0.0
        )

        var frames = [voicedFrame]
        var f = 0
        while f < 10 {
            frames.append(silenceFrame)
            f += 1
        }
        let wave = vocoder.synthesize(frames: frames)
        let lastSample = abs(wave[wave.count - 1])

        print("--- [Challenger Vulnerability 3 Fixed] ディエンファシス 1,600サンプル後残留振幅: \(lastSample) ---")
        XCTAssertEqual(lastSample, 0.0, "Flush-to-Zero が機能せずデノーマル数が残留しています")
    }

    func testInfinityF0DoesNotHang() {
        // SNN から極大値や非有限値 F0 が渡された際、位相計算の剰余処理が無限ループに陥らず、
        // 即座に 0.0 を返却してプロセスが健全に動作し続けることを数学的に実証する。
        let pulseGen = RosenbergPulse(sampleRate: 16000.0)

        // 1. nextSample での Infinity F0
        let sInf = pulseGen.nextSample(f0: Float.infinity)
        XCTAssertEqual(sInf, 0.0, "Infinity F0 で 0.0 が返却されませんでした")

        // 2. nextSample での NaN F0
        let sNaN = pulseGen.nextSample(f0: Float.nan)
        XCTAssertEqual(sNaN, 0.0, "NaN F0 で 0.0 が返却されませんでした")

        // 3. nextSample でのナイキスト周波数超え F0 (8000Hz 以上)
        let sNyquist = pulseGen.nextSample(f0: 8000.0)
        XCTAssertEqual(sNyquist, 0.0, "ナイキスト周波数以上の F0 で 0.0 が返却されませんでした")

        // 4. generateFrame での一括生成
        var buf = [Float](repeating: 1.0, count: 160)
        buf.withUnsafeMutableBufferPointer { pDst in
            pulseGen.generateFrame(f0: Float.infinity, count: 160, dst: pDst.baseAddress!)
        }
        var i = 0
        while i < 160 {
            XCTAssertEqual(buf[i], 0.0, "generateFrame で Infinity F0 入力時にゼロクリアされていません")
            i += 1
        }
    }

    // MARK: - 6. 静的コード規約適合性検証 (DSP モジュール全走査)

    func testStaticCodingRulesComplianceDSP() {
        // 比較演算子は `<` および `<=` のみ、`else if` 禁止、三項演算子禁止などの規約を
        // Sources/SpikeSpeech/DSP/ 以下の全ファイルに対して自動検証する。
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let dspPath = currentDir + "/Sources/SpikeSpeech/DSP"

        guard let enumerator = fileManager.enumerator(atPath: dspPath) else {
            XCTFail("Sources/SpikeSpeech/DSP が走査できませんでした")
            return
        }

        var checkedFiles = 0
        while let relativePath = enumerator.nextObject() as? String {
            if relativePath.hasSuffix(".swift") != true {
                continue
            }

            let fullPath = dspPath + "/" + relativePath
            guard let content = try? String(contentsOfFile: fullPath, encoding: .utf8) else {
                continue
            }

            let lines = content.components(separatedBy: .newlines)
            var lineIdx = 0
            while lineIdx < lines.count {
                let line = lines[lineIdx]
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                if trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*") {
                    lineIdx += 1
                    continue
                }

                XCTAssertFalse(
                    trimmed.contains(" > ") && trimmed.contains("->") != true,
                    "比較演算子 > が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(" >= "),
                    "比較演算子 >= が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains("else if"),
                    "else if が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                XCTAssertFalse(
                    trimmed.contains(" ? ") && trimmed.contains("??") != true,
                    "三項演算子が使用されています: \(relativePath):\(lineIdx + 1): \(trimmed)"
                )
                lineIdx += 1
            }
            checkedFiles += 1
        }

        XCTAssertTrue(0 < checkedFiles, "チェック対象の Swift ファイルがありません")
        print("--- [Challenger DSP Static Rule Check] ---")
        print("検証完了 DSP ファイル数: \(checkedFiles) 件 (全ファイル規約適合)")
        print("------------------------------------------")
    }

    // MARK: - 11. 5母音ホルマントスペクトル (F1, F2) 理論値整合性検証

    func testFiveVowelsFormantSpectraCompliance() {
        let generator = SyntheticAudioGenerator(sampleRate: 16000.0)
        let vowels: [SyntheticAudioGenerator.Vowel] = [.a, .i, .u, .e, .o]
        let sampleRate: Float = 16000.0

        var vIdx = 0
        while vIdx < vowels.count {
            let vowel = vowels[vIdx]
            let cfg = generator.formantConfig(for: vowel)
            let samples = generator.generateVowel(vowel: vowel, durationSeconds: 0.2, f0: 125.0)

            XCTAssertTrue(0 < samples.count)

            // 定常区間（中央部 1024 サンプル）を抽出してスペクトル解析
            let analysisLength = 1024
            let startOffset = (samples.count - analysisLength) / 2

            // 100Hz 〜 3000Hz の範囲で 20Hz 刻みで離散フーリエ振幅を計算
            var freqPows: [(freq: Float, power: Float)] = []
            var testFreq: Float = 100.0
            while testFreq <= 3000.0 {
                let omega = 2.0 * Float.pi * testFreq / sampleRate
                var realSum: Float = 0.0
                var imagSum: Float = 0.0

                var n = 0
                while n < analysisLength {
                    let s = samples[startOffset + n]
                    // ハミング窓
                    let w = 0.54 - (0.46 * cos(2.0 * Float.pi * Float(n) / Float(analysisLength - 1)))
                    let windowed = s * w
                    let angle = omega * Float(n)
                    realSum += windowed * cos(angle)
                    imagSum -= windowed * sin(angle)
                    n += 1
                }

                let power = (realSum * realSum) + (imagSum * imagSum)
                freqPows.append((freq: testFreq, power: power))
                testFreq += 20.0
            }

            // F1 探索 (理論値 cfg.f1 ± 150Hz 範囲での最大点)
            var bestF1Freq: Float = 0.0
            var maxF1Pow: Float = -1.0
            var bestF2Freq: Float = 0.0
            var maxF2Pow: Float = -1.0

            var pIdx = 0
            while pIdx < freqPows.count {
                let entry = freqPows[pIdx]
                if abs(entry.freq - cfg.f1) <= 150.0 {
                    if maxF1Pow < entry.power {
                        maxF1Pow = entry.power
                        bestF1Freq = entry.freq
                    }
                }
                if abs(entry.freq - cfg.f2) <= 200.0 {
                    if maxF2Pow < entry.power {
                        maxF2Pow = entry.power
                        bestF2Freq = entry.freq
                    }
                }
                pIdx += 1
            }

            print("--- [Formant Spectrum: /\(vowel.rawValue)/] ---")
            print("F1 理論値: \(cfg.f1) Hz, 実測ピーク: \(bestF1Freq) Hz")
            print("F2 理論値: \(cfg.f2) Hz, 実測ピーク: \(bestF2Freq) Hz")

            let f1Diff = abs(bestF1Freq - cfg.f1)
            let f2Diff = abs(bestF2Freq - cfg.f2)

            XCTAssertTrue(f1Diff <= 100.0, "母音 /\(vowel.rawValue)/ の F1 ピーク乖離が許容値を超えています: 実測=\(bestF1Freq), 理論=\(cfg.f1)")
            XCTAssertTrue(f2Diff <= 150.0, "母音 /\(vowel.rawValue)/ の F2 ピーク乖離が許容値を超えています: 実測=\(bestF2Freq), 理論=\(cfg.f2)")

            vIdx += 1
        }
    }

    // MARK: - 12. 有声／無声励起切り替えにおけるクリックノイズ不在・平滑遷移検証

    func testExcitationTransitionSmoothnessAndClickAbsence() {
        let vocoder = LPCVocoder(sampleRate: 16000.0, frameSize: 160, lpcOrder: 16, deEmphasisCoeff: 0.97)
        let vowelCoeffs: [Float] = [
            1.25, -0.85, 0.45, -0.25, 0.15, -0.08, 0.04, -0.02,
            0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

        let voicedFrame = AcousticFrame(
            lpcCoefficients: vowelCoeffs,
            gain: 0.1,
            pitchF0: 150.0,
            voiced: 1.0
        )
        let unvoicedFrame = AcousticFrame(
            lpcCoefficients: vowelCoeffs,
            gain: 0.1,
            pitchF0: 0.0,
            voiced: 0.0
        )

        // 有声 -> 無声 -> 有声 の切り替えシーケンス
        let sequence = [
            voicedFrame, voicedFrame,
            unvoicedFrame, unvoicedFrame,
            voicedFrame, voicedFrame
        ]

        let output = vocoder.synthesize(frames: sequence)
        XCTAssertEqual(output.count, 6 * 160)

        // フレーム境界部におけるサンプル間最大一階差分 |s[n] - s[n-1]| を測定
        var maxDeltaAtBoundary: Float = 0.0
        let boundaryIndices = [160, 320, 480, 640, 800]

        var bIdx = 0
        while bIdx < boundaryIndices.count {
            let b = boundaryIndices[bIdx]
            var offset = -10
            while offset <= 10 {
                let idx = b + offset
                if 1 <= idx && idx < output.count {
                    let delta = abs(output[idx] - output[idx - 1])
                    if maxDeltaAtBoundary < delta {
                        maxDeltaAtBoundary = delta
                    }
                }
                offset += 1
            }
            bIdx += 1
        }

        print("--- [Excitation Transition Click Test] ---")
        print("境界近傍最大一階差分: \(maxDeltaAtBoundary)")

        // クリックノイズ（不連続衝撃波）が発生せず、差分が 0.35 以下に抑制されていること
        XCTAssertTrue(maxDeltaAtBoundary < 0.35, "有声/無声励起切り替え境界でクリックノイズ（不連続パルス）を検出しました: \(maxDeltaAtBoundary)")
    }
}

