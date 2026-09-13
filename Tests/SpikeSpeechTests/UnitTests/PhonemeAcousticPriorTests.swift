import XCTest
import Foundation
@testable import SpikeSpeech

final class PhonemeAcousticPriorTests: XCTestCase {

    /// 5母音および無音トークンの事前 Mel スペクトルの健全性と分離度を検証
    func testPhonemeAcousticPriorProperties() {
        let prior = PhonemeAcousticPrior()
        let melChannels = prior.melChannels

        XCTAssertEqual(melChannels, AudioConfig.melChannels)

        let aMel = prior.getPriorMel(phoneId: 5) // a
        let iMel = prior.getPriorMel(phoneId: 6) // i
        let uMel = prior.getPriorMel(phoneId: 7) // u
        let eMel = prior.getPriorMel(phoneId: 8) // e
        let oMel = prior.getPriorMel(phoneId: 9) // o
        let silMel = prior.getPriorMel(phoneId: 1) // sil

        XCTAssertEqual(aMel.count, melChannels)
        XCTAssertEqual(iMel.count, melChannels)
        XCTAssertEqual(uMel.count, melChannels)
        XCTAssertEqual(eMel.count, melChannels)
        XCTAssertEqual(oMel.count, melChannels)
        XCTAssertEqual(silMel.count, melChannels)

        // 1. 各チャンネルの有限性検証
        var c = 0
        while c < melChannels {
            XCTAssertTrue(aMel[c].isFinite, "aMel[\(c)] が非有限値です")
            XCTAssertTrue(iMel[c].isFinite, "iMel[\(c)] が非有限値です")
            XCTAssertTrue(silMel[c].isFinite, "silMel[\(c)] が非有限値です")
            c += 1
        }

        // 2. 無音スペクトルが母音より有意に低エネルギー（対数値が小さい）であること
        var aSum: Float = 0.0
        var silSum: Float = 0.0
        c = 0
        while c < melChannels {
            aSum += aMel[c]
            silSum += silMel[c]
            c += 1
        }
        let aAvg = aSum / Float(melChannels)
        let silAvg = silSum / Float(melChannels)
        XCTAssertTrue(silAvg < aAvg, "無音スペクトルの平均が母音 /a/ より大きいです: sil=\(silAvg), a=\(aAvg)")

        // 3. /a/ と /i/ の間の明確なスペクトル分離（L1 差分）
        var diffAI: Float = 0.0
        c = 0
        while c < melChannels {
            diffAI += abs(aMel[c] - iMel[c])
            c += 1
        }
        XCTAssertTrue(10.0 < diffAI, "母音 /a/ と /i/ のスペクトル分離度が不十分です: diff=\(diffAI)")
    }

    /// 事前 Mel スペクトルから LPC 係数を算出した際の共鳴特性と安定性を検証
    func testPhonemeAcousticPriorMelToLPCConversion() {
        let prior = PhonemeAcousticPrior()
        let melToLPC = MelToLPC()

        let vowels = [5, 6, 7, 8, 9] // a, i, u, e, o
        var vIdx = 0
        while vIdx < vowels.count {
            let pId = vowels[vIdx]
            let mel = prior.getPriorMel(phoneId: pId)
            var lpcCoeffs = [Float](repeating: 0.0, count: AudioConfig.lpcOrder)
            let gain = melToLPC.convert(mel: mel, isLogMel: true, outCoeffs: &lpcCoeffs)

            XCTAssertTrue(gain.isFinite, "音素 \(pId) の LPC ゲインが非有限値です")
            XCTAssertTrue(0.0001 < gain, "音素 \(pId) の LPC ゲインが極小すぎます: gain=\(gain)")

            var k = 0
            while k < lpcCoeffs.count {
                XCTAssertTrue(lpcCoeffs[k].isFinite, "音素 \(pId) の LPC 係数 \(k) が非有限値です")
                k += 1
            }
            vIdx += 1
        }
    }
}
