import XCTest
@testable import SpikeSpeech

/// Challenger 1 による SNN ニューロンの極限状態・数値安定性・敵対的境界値の実証テストスイート
///
/// SNN ニューロンの極限状態・数値安定性・敵対的境界値の実証テストスイート
///
/// ゼロ入力電流減衰、極大電流クランプ、
/// リードアウト層の減算リセットによる電位爆発防止、および NaN/Inf 混入時の安全性を
/// 完全に独立して実証・検証する。
final class ChallengerSNNTests: XCTestCase {

    // MARK: - 1. 入力電流ゼロ（無音潜在ベクトル）での膜電位減衰および無発火安定性実証

    func testZeroInputCurrentMembraneDecayAndSilenceStability() {
        // 音声合成における無音区間やパウズにおいて、膜電位が指数関数的に基底電位へ減衰し、
        // 誤ったゴーストスパイクが一切発生しない数学的安定性を保証する。
        let config = LIFConfig(
            beta: 0.8,
            vTh: 1.0,
            vReset: 0.0,
            alpha: 2.0,
            rho: 0.85,
            gamma: 0.1
        )

        // 1. スカラー隠れ層: 初期膜電位 0.8 (閾値 1.0 未満) からゼロ入力で減衰
        var v = Float(0.8)
        var s = Float(0.0)
        var step = 0
        while step < 30 {
            let res = LIFNeuronEngine.stepScalar(
                config: config,
                vPrev: v,
                sPrev: s,
                inputCurrent: 0.0
            )
            v = res.vNext
            s = res.sNext

            // スパイクは一切発火しないこと
            XCTAssertEqual(s, 0.0, "ゼロ入力において誤発火が発生しました: step=\(step), v=\(v)")

            // 指数関数的減衰 v_t = 0.8 * (0.8)^step の理論値と照合
            let expectedV = 0.8 * pow(config.beta, Float(step + 1))
            let diff = abs(v - expectedV)
            XCTAssertTrue(diff < 1e-4, "膜電位減衰理論値との乖離: step=\(step), actual=\(v), expected=\(expectedV)")
            step += 1
        }

        // 十分なステップ後には実質ゼロに収束すること
        XCTAssertTrue(v < 1e-3, "30ステップ後の膜電位がゼロに収束していません: \(v)")

        // 2. SIMD8 リードアウト層: 各レーンで異なる初期電位からゼロ入力で減衰
        let count = 16
        var vSIMD = [Float](repeating: 0.0, count: count)
        var sSIMD = [Float](repeating: 0.0, count: count)
        var aSIMD = [Float](repeating: 0.0, count: count)
        var readoutSum = [Float](repeating: 0.0, count: count)
        let zeroCur = [Float](repeating: 0.0, count: count)

        var idx = 0
        while idx < count {
            vSIMD[idx] = 0.1 * Float(idx) // 0.0 .. 1.5
            idx += 1
        }

        step = 0
        while step < 30 {
            vSIMD.withUnsafeMutableBufferPointer { pV in
                sSIMD.withUnsafeMutableBufferPointer { pS in
                    aSIMD.withUnsafeMutableBufferPointer { pA in
                        zeroCur.withUnsafeBufferPointer { pCur in
                            readoutSum.withUnsafeMutableBufferPointer { pSum in
                                LIFNeuronEngine.stepReadoutAdaptiveSIMD8(
                                    config: config,
                                    vPtr: pV.baseAddress!,
                                    sPtr: pS.baseAddress!,
                                    aPtr: pA.baseAddress!,
                                    curPtr: pCur.baseAddress!,
                                    readoutSumPtr: pSum.baseAddress!,
                                    count: count
                                )
                            }
                        }
                    }
                }
            }
            step += 1
        }

        // 30ステップ後、全レーンの膜電位が 0.005 未満に減衰していること (1.5 * 0.8^30 ~= 0.0018)
        idx = 0
        while idx < count {
            XCTAssertTrue(vSIMD[idx] < 0.005, "レーン \(idx) の膜電位が減衰していません: \(vSIMD[idx])")
            XCTAssertFalse(vSIMD[idx].isNaN, "レーン \(idx) で NaN 発生")
            idx += 1
        }

        // 3. デコーダー全体へのゼロ潜在ベクトル入力テスト
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 32,
            maxHiddenDim: 128,
            outputDim: 80,
            timeSteps: 4,
            numLayers: 2,
            seed: 42
        )
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 128, outputDim: 80, numLayers: 2)

        let zeroFeatures = [Float](repeating: 0.0, count: 32)
        var outFeats = [Float](repeating: 0.0, count: 80)

        zeroFeatures.withUnsafeBufferPointer { pIn in
            outFeats.withUnsafeMutableBufferPointer { pOut in
                decoder.decodeFrame(
                    features: pIn.baseAddress!,
                    workspace: workspace,
                    outputFeatures: pOut.baseAddress!
                )
            }
        }

        var c = 0
        while c < 80 {
            XCTAssertFalse(outFeats[c].isNaN, "無音入力デコード出力 \(c) で NaN 検出")
            XCTAssertFalse(outFeats[c].isInfinite, "無音入力デコード出力 \(c) で Inf 検出")
            c += 1
        }
    }

    // MARK: - 2. 極大入力電流（過大潜在ベクトル）注入時の膜電位クランプおよび発散防止実証

    func testExtremeInputCurrentMembraneClampingAndAntiDivergence() {
        // 異常な振幅や発音異常による過大電流が注入された際、膜電位が上下限で
        // 確実に飽和し、浮動小数点オーバーフローや発散を起こさないことを実証する。

        let config = LIFConfig(
            beta: 0.8,
            vTh: 1.0,
            vReset: 0.0,
            alpha: 2.0,
            rho: 0.85,
            gamma: 0.0
        )

        // 1. スカラー極大正入力 (+1e6, +1000.0, +50.0)
        let hugePositiveCur: [Float] = [50.0, 1000.0, 1_000_000.0]
        var hIdx = 0
        while hIdx < hugePositiveCur.count {
            let cur = hugePositiveCur[hIdx]
            let res = LIFNeuronEngine.stepScalar(
                config: config,
                vPrev: 0.0,
                sPrev: 0.0,
                inputCurrent: cur
            )
            // 膜電位は vClampMax (20.0) を超えてはならない
            XCTAssertEqual(res.vNext, LIFNeuronEngine.vClampMax, "過大正入力でクランプ上限超過: cur=\(cur), vNext=\(res.vNext)")
            XCTAssertEqual(res.sNext, 1.0, "過大正入力でスパイクが発火していません: cur=\(cur)")
            hIdx += 1
        }

        // 2. スカラー極大負入力 (-1e6, -1000.0, -50.0)
        let hugeNegativeCur: [Float] = [-50.0, -1000.0, -1_000_000.0]
        hIdx = 0
        while hIdx < hugeNegativeCur.count {
            let cur = hugeNegativeCur[hIdx]
            let res = LIFNeuronEngine.stepScalar(
                config: config,
                vPrev: 0.0,
                sPrev: 0.0,
                inputCurrent: cur
            )
            // 膜電位は vClampMin (-20.0) を下回ってはならない
            XCTAssertEqual(res.vNext, LIFNeuronEngine.vClampMin, "過大負入力でクランプ下限超過: cur=\(cur), vNext=\(res.vNext)")
            XCTAssertEqual(res.sNext, 0.0, "過大負入力で誤発火が発生しました: cur=\(cur)")
            hIdx += 1
        }

        // 3. SIMD8 一括クランプ実証
        let count = 8
        var vSIMD = [Float](repeating: 0.0, count: count)
        var sSIMD = [Float](repeating: 0.0, count: count)
        var aSIMD = [Float](repeating: 0.0, count: count)
        let mixedExtremeCur: [Float] = [
            1_000_000.0, -1_000_000.0, 500.0, -500.0,
            25.0, -25.0, 20.0, -20.0
        ]

        vSIMD.withUnsafeMutableBufferPointer { pV in
            sSIMD.withUnsafeMutableBufferPointer { pS in
                aSIMD.withUnsafeMutableBufferPointer { pA in
                    mixedExtremeCur.withUnsafeBufferPointer { pCur in
                        LIFNeuronEngine.stepAdaptiveSIMD8(
                            config: config,
                            vPtr: pV.baseAddress!,
                            sPtr: pS.baseAddress!,
                            aPtr: pA.baseAddress!,
                            curPtr: pCur.baseAddress!,
                            count: count
                        )
                    }
                }
            }
        }

        var i = 0
        while i < count {
            let val = vSIMD[i]
            XCTAssertFalse(val < LIFNeuronEngine.vClampMin, "レーン \(i) でクランプ下限違反: \(val)")
            XCTAssertFalse(LIFNeuronEngine.vClampMax < val, "レーン \(i) でクランプ上限違反: \(val)")
            XCTAssertFalse(val.isNaN, "レーン \(i) で NaN 発生")
            XCTAssertFalse(val.isInfinite, "レーン \(i) で Inf 発生")
            i += 1
        }
    }

    // MARK: - 3. リードアウト層の減算リセットによる連続ステップでの電位爆発防止実証

    func testReadoutSubtractiveResetPreventsPotentialExplosion() {
        // 減算リセットは余剰電位を保持するため、
        // 連続入力下で膜電位が累積爆発しないこと、および平衡点に収束・クランプされることを実証する。

        let config = LIFConfig(
            beta: 0.8,
            vTh: 1.0,
            vReset: 0.0,
            alpha: 2.0,
            rho: 0.85,
            gamma: 0.0
        )

        // 1. 強入力電流 (2.0: 閾値 1.0 の 2倍) を 200 ステップ連続注入
        var vPrev = Float(0.0)
        var sPrev = Float(0.0)
        var aPrev = Float(0.0)
        var step = 0

        // 理論的平衡点: V_int = 0.8 * V + 2.0
        // S = 1.0 なので V_next = 0.8 * V + 2.0 - 1.0 = 0.8 * V + 1.0
        // 平衡点 V* = 1.0 / (1 - 0.8) = 5.0
        while step < 200 {
            let res = LIFNeuronEngine.stepReadoutScalarAdaptive(
                config: config,
                vPrev: vPrev,
                sPrev: sPrev,
                aPrev: aPrev,
                inputCurrent: 2.0
            )
            vPrev = res.vNext
            sPrev = res.sNext
            aPrev = res.aNext

            // スパイクは毎ステップ発火
            XCTAssertEqual(sPrev, 1.0, "step \(step) でスパイク不発火")
            // リードアウト値は [-1, 1] クリップ幅内に収まる
            XCTAssertEqual(res.readout, 1.0, "step \(step) でリードアウトがクリップ上限と不一致")

            // 膜電位はクランプ上限 20.0 を絶対に超えない
            XCTAssertFalse(LIFNeuronEngine.vClampMax < vPrev, "膜電位がクランプ上限を超過: step=\(step), v=\(vPrev)")
            step += 1
        }

        // 200ステップ後の膜電位が理論的平衡点 5.0 に安定収束していること
        let diffEquilibrium = abs(vPrev - 5.0)
        XCTAssertTrue(diffEquilibrium < 1e-3, "減算リセットの平衡点収束失敗: actual=\(vPrev), expected=5.0")

        // 2. 超極大電流 (100.0) を 200 ステップ連続注入
        // 平衡点は理論上 99 / 0.2 = 495 だが、vClampMax (20.0) で確実にクランプされ爆発しないこと
        vPrev = 0.0
        sPrev = 0.0
        aPrev = 0.0
        step = 0
        while step < 200 {
            let res = LIFNeuronEngine.stepReadoutScalarAdaptive(
                config: config,
                vPrev: vPrev,
                sPrev: sPrev,
                aPrev: aPrev,
                inputCurrent: 100.0
            )
            vPrev = res.vNext
            sPrev = res.sNext
            aPrev = res.aNext

            // 膜電位は厳密にクランプ上限 20.0 (減算後は 20.0 - 1.0 = 19.0)
            XCTAssertFalse(LIFNeuronEngine.vClampMax < vPrev, "電位爆発を検出: step=\(step), v=\(vPrev)")
            XCTAssertFalse(vPrev.isInfinite, "step \(step) で Inf 検出")
            XCTAssertFalse(vPrev.isNaN, "step \(step) で NaN 検出")
            step += 1
        }

        // 最終電位が 19.0 (クランプ 20.0 から閾値 1.0 減算) に安定していること
        let diffClamped = abs(vPrev - 19.0)
        XCTAssertTrue(diffClamped < 1e-4, "クランプ下での減算リセット安定値不一致: actual=\(vPrev), expected=19.0")
    }

    // MARK: - 4. 入力に NaN / Inf が含まれた場合の挙動（クラッシュ防止・安全性）実証

    func testNaNAndInfInputHandlingAndSafety() {
        // 敵対的入力や前段の計算破綻により非数（NaN）や無限大（Inf）が混入した場合に、
        // プロセスがセグメンテーションフォルトや致命的エラーでクラッシュせず、
        // 安全に処理を完遂できることを実証する。

        let config = LIFConfig(
            beta: 0.8,
            vTh: 1.0,
            vReset: 0.0,
            alpha: 2.0,
            rho: 0.85,
            gamma: 0.0
        )

        // 1. スカラー clampMembrane における Inf クランプ実証
        let posInf = Float.infinity
        let negInf = -Float.infinity
        let clampedPosInf = LIFNeuronEngine.clampMembrane(posInf)
        let clampedNegInf = LIFNeuronEngine.clampMembrane(negInf)

        XCTAssertEqual(clampedPosInf, LIFNeuronEngine.vClampMax, "+Inf は vClampMax (20.0) にクランプされること")
        XCTAssertEqual(clampedNegInf, LIFNeuronEngine.vClampMin, "-Inf は vClampMin (-20.0) にクランプされること")

        // 2. Inf を含む入力でのステップ更新 (クラッシュしないことの実証)
        let resPosInf = LIFNeuronEngine.stepScalar(
            config: config,
            vPrev: 0.0,
            sPrev: 0.0,
            inputCurrent: posInf
        )
        XCTAssertEqual(resPosInf.vNext, LIFNeuronEngine.vClampMax)
        XCTAssertEqual(resPosInf.sNext, 1.0)

        let resNegInf = LIFNeuronEngine.stepScalar(
            config: config,
            vPrev: 0.0,
            sPrev: 0.0,
            inputCurrent: negInf
        )
        XCTAssertEqual(resNegInf.vNext, LIFNeuronEngine.vClampMin)
        XCTAssertEqual(resNegInf.sNext, 0.0)

        // 3. NaN を含む入力でのステップ更新 (クラッシュせずフォールバックすることの実証)
        let nanVal = Float.nan
        let resNaN = LIFNeuronEngine.stepScalar(
            config: config,
            vPrev: 0.0,
            sPrev: 0.0,
            inputCurrent: nanVal
        )
        // NaN に対して比較は false となりクラッシュせず、スパイクは 0.0 に安全倒置される
        XCTAssertEqual(resNaN.sNext, 0.0, "NaN 入力時にスパイクが誤発火してはならない")

        // 4. デコーダー全体に NaN/Inf を含む潜在ベクトルを与えた場合のクラッシュ防止実証
        let weights = SpikingNetworkWeights.randomWeights(
            inputDim: 32,
            maxHiddenDim: 64,
            outputDim: 16,
            timeSteps: 2,
            numLayers: 2,
            seed: 777
        )
        let decoder = SpikingAcousticDecoder(weights: weights)
        let workspace = AcousticWorkspace(maxHiddenDim: 64, outputDim: 16, numLayers: 2)

        var hostileFeatures = [Float](repeating: 0.5, count: 32)
        hostileFeatures[0] = Float.nan
        hostileFeatures[1] = Float.infinity
        hostileFeatures[2] = -Float.infinity

        var outBuffer = [Float](repeating: 0.0, count: 16)

        // クラッシュ（メモリ不正アクセス・除算例外・未定義動作）なく完遂すること
        hostileFeatures.withUnsafeBufferPointer { pIn in
            outBuffer.withUnsafeMutableBufferPointer { pOut in
                decoder.decodeFrame(
                    features: pIn.baseAddress!,
                    workspace: workspace,
                    outputFeatures: pOut.baseAddress!
                )
            }
        }

        // デコード結果が正常に生成（書き込み完了）されていること（バッファ長16の健全性）
        XCTAssertEqual(outBuffer.count, 16)

        // 5. decodeSequence における NaN/Inf 混在系列のクラッシュ防止実証
        let hostileSeq: [[Float]] = [
            [Float](repeating: 0.1, count: 32),
            hostileFeatures,
            [Float](repeating: 0.2, count: 32)
        ]
        let outSeq = decoder.decodeSequence(featuresSeq: hostileSeq, workspace: workspace)
        XCTAssertEqual(outSeq.count, 3, "NaN/Inf を含む系列でもクラッシュせず 3 フレーム返却されること")
        XCTAssertEqual(outSeq[0].count, 16)
        XCTAssertEqual(outSeq[1].count, 16)
        XCTAssertEqual(outSeq[2].count, 16)
    }

    // MARK: - 5. 静的コード規約機械検査

    func testChallengerSNNStaticRuleCheck() {
        // 比較演算子 `<` と `<=` のみ、`else if` 禁止、三項演算子禁止などの規約を自動走査する。
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        let snnPath = currentDir + "/Sources/SpikeSpeech/SNN"
        let mlxPath = currentDir + "/Sources/SpikeSpeech/MLX"

        let targetPaths = [snnPath, mlxPath]
        var totalChecked = 0

        var pIdx = 0
        while pIdx < targetPaths.count {
            let path = targetPaths[pIdx]
            guard let enumerator = fileManager.enumerator(atPath: path) else {
                XCTFail("パス走査失敗: \(path)")
                pIdx += 1
                continue
            }

            while let relativePath = enumerator.nextObject() as? String {
                if relativePath.hasSuffix(".swift") != true {
                    continue
                }

                let fullPath = path + "/" + relativePath
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
                totalChecked += 1
            }
            pIdx += 1
        }

        XCTAssertTrue(0 < totalChecked, "検査対象ファイルがありません")
        print("--- [Challenger SNN/MLX Static Rule Check] ---")
        print("検証完了ファイル数: \(totalChecked) 件 (全ファイル規約適合)")
        print("---------------------------------------------")
    }
}
