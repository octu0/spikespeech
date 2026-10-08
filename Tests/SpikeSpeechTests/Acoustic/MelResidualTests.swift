import Foundation
import XCTest
@testable import SpikeSpeech
#if canImport(MLX)
import MLX
#endif

final class MelResidualTests: XCTestCase {
    /// エポック 19 の既存 JSON との下位互換性および残差重みの Codable ラウンドトリップ検証
    func testMelResidualWeightsCodableCompatibility() throws {
        let ep19URL = URL(fileURLWithPath: "Models/weights.frame_mel_ep19.json")
        let ep19Weights = try SpikingNetworkWeights.load(from: ep19URL)
        let fmw = try XCTUnwrap(ep19Weights.frameMelWeights, "エポック 19 に FrameMelWeights が存在すること")

        // 1. エポック 19 の JSON には残差キーが無いため nil として読み込まれること
        XCTAssertNil(fmw.resW1, "エポック 19 の resW1 は nil であること")
        XCTAssertNil(fmw.resB1, "エポック 19 の resB1 は nil であること")
        XCTAssertNil(fmw.resW2, "エポック 19 の resW2 は nil であること")
        XCTAssertNil(fmw.resB2, "エポック 19 の resB2 は nil であること")

        // 2. 残差重み生成と付加
        let initialRes = FrameMelWeights.makeInitialResidualWeights(seed: 2026)
        XCTAssertEqual(initialRes.resW1.count, 256 * 3 * 260)
        XCTAssertEqual(initialRes.resB1.count, 256)
        XCTAssertEqual(initialRes.resW2.count, 64 * 1 * 256)
        XCTAssertEqual(initialRes.resB2.count, 64)

        // 2層目の重みとバイアスがすべて 0 で初期化されていること
        var i = 0
        while i < initialRes.resW2.count {
            XCTAssertEqual(initialRes.resW2[i], 0.0, "resW2 は初期状態ですべて 0 であること")
            i += 1
        }
        i = 0
        while i < initialRes.resB2.count {
            XCTAssertEqual(initialRes.resB2[i], 0.0, "resB2 は初期状態ですべて 0 であること")
            i += 1
        }

        let fmwWithRes = fmw.withMelResidual(
            resW1: initialRes.resW1,
            resB1: initialRes.resB1,
            resW2: initialRes.resW2,
            resB2: initialRes.resB2
        )
        let fullWithRes = ep19Weights.withFrameMelWeights(fmwWithRes)

        // 3. エンコード & デコードのラウンドトリップ
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let encodedData = try encoder.encode(fullWithRes)

        let decoder = JSONDecoder()
        let decodedFull = try decoder.decode(SpikingNetworkWeights.self, from: encodedData)
        let decodedFMW = try XCTUnwrap(decodedFull.frameMelWeights)

        XCTAssertEqual(decodedFMW.resW1, initialRes.resW1)
        XCTAssertEqual(decodedFMW.resB1, initialRes.resB1)
        XCTAssertEqual(decodedFMW.resW2, initialRes.resW2)
        XCTAssertEqual(decodedFMW.resB2, initialRes.resB2)

        // 4. 残差重みの破棄 (withoutMelResidual)
        let fmwWithout = decodedFMW.withoutMelResidual()
        XCTAssertNil(fmwWithout.resW1)
        XCTAssertNil(fmwWithout.resB1)
        XCTAssertNil(fmwWithout.resW2)
        XCTAssertNil(fmwWithout.resB2)

        let encodedWithout = try encoder.encode(ep19Weights.withFrameMelWeights(fmwWithout))
        let decodedWithout = try decoder.decode(SpikingNetworkWeights.self, from: encodedWithout)
        XCTAssertNil(decodedWithout.frameMelWeights?.resW1)
    }

    /// 初期化直後の残差は完全に出力 0 であり、PostNet 後メルと同一であること
    func testMelResidualInitialStateIsZero() throws {
        let ep19URL = URL(fileURLWithPath: "Models/weights.frame_mel_ep19.json")
        let ep19Weights = try SpikingNetworkWeights.load(from: ep19URL)
        let fmw = try XCTUnwrap(ep19Weights.frameMelWeights)

        let initialRes = FrameMelWeights.makeInitialResidualWeights(seed: 2026)
        let fmwWithRes = fmw.withMelResidual(
            resW1: initialRes.resW1,
            resB1: initialRes.resB1,
            resW2: initialRes.resW2,
            resB2: initialRes.resB2
        )

        let modelWithRes = FrameMelModel(weights: fmwWithRes)

        let totalFrames = 10
        var dummyCondition = [Float](repeating: 0.0, count: totalFrames * 260)
        var i = 0
        while i < dummyCondition.count {
            dummyCondition[i] = Float(i % 17) * 0.05
            i += 1
        }

        let resOutput = modelWithRes.computeMelResidual(condition: dummyCondition, totalFrames: totalFrames)
        let unwrapRes = try XCTUnwrap(resOutput)

        var maxAbsRes: Float = 0.0
        i = 0
        while i < unwrapRes.count {
            let absV = abs(unwrapRes[i])
            if maxAbsRes < absV {
                maxAbsRes = absV
            }
            i += 1
        }
        XCTAssertEqual(maxAbsRes, 0.0, "初期状態の残差出力は厳密に 0 であること")
    }

    #if canImport(MLX)
    /// 短い文での Pure Swift と MLX の残差および合成メルの最大絶対差が 1e-3 未満であることの検証
    func testMelResidualPureSwiftAndMLXNumericalConsistency() throws {
        let ep19URL = URL(fileURLWithPath: "Models/weights.frame_mel_ep19.json")
        let ep19Weights = try SpikingNetworkWeights.load(from: ep19URL)
        let fmw = try XCTUnwrap(ep19Weights.frameMelWeights)

        // ランダムな重み（非ゼロ）で残差を設定
        let initialRes = FrameMelWeights.makeInitialResidualWeights(seed: 2026)
        // 2層目をランダム値に変更して非ゼロ残差をテスト
        var nonZeroW2 = initialRes.resW2
        var nonZeroB2 = initialRes.resB2
        var i = 0
        while i < nonZeroW2.count {
            nonZeroW2[i] = sinf(Float(i) * 0.1) * 0.01
            i += 1
        }
        i = 0
        while i < nonZeroB2.count {
            nonZeroB2[i] = cosf(Float(i) * 0.2) * 0.01
            i += 1
        }

        let fmwWithRes = fmw.withMelResidual(
            resW1: initialRes.resW1,
            resB1: initialRes.resB1,
            resW2: nonZeroW2,
            resB2: nonZeroB2
        )

        let swiftModel = FrameMelModel(weights: fmwWithRes)
        let mlxResidual = MLXMelResidual(
            w1: initialRes.resW1,
            b1: initialRes.resB1,
            w2: nonZeroW2,
            b2: nonZeroB2
        )

        let totalFrames = 8
        var dummyCondition = [Float](repeating: 0.0, count: totalFrames * 260)
        i = 0
        while i < dummyCondition.count {
            dummyCondition[i] = sinf(Float(i) * 0.05) * 0.5
            i += 1
        }

        // Pure Swift 計算
        let swiftRes = try XCTUnwrap(swiftModel.computeMelResidual(condition: dummyCondition, totalFrames: totalFrames))

        // MLX 計算
        let mlxCondition = MLXArray(dummyCondition, [1, totalFrames, 260])
        let mlxResArray = mlxResidual.forward(condition: mlxCondition)
        eval(mlxResArray)
        let mlxRes = mlxResArray.asArray(Float.self)

        XCTAssertEqual(swiftRes.count, mlxRes.count)

        var maxDiff: Float = 0.0
        i = 0
        while i < swiftRes.count {
            let diff = abs(swiftRes[i] - mlxRes[i])
            if maxDiff < diff {
                maxDiff = diff
            }
            i += 1
        }

        print("[MelResidual Numerical Consistency] Swift vs MLX MaxDiff: \(maxDiff)")
        XCTAssertTrue(maxDiff < 1e-3, "Pure Swift と MLX の残差最大絶対差は 1e-3 未満であること（実測: \(maxDiff)）")
    }
    #endif
}
