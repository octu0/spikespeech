import XCTest
@testable import SpikeSpeech
@testable import SpikeSpeechWeb

/// WebSocket サーバー・クライアント結合テストスイート
final class WebSocketTests: XCTestCase {

    /// プロトコル JSON メッセージの相互エンコード・デコード検証
    func testProtocolJSONSerialization() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // 1. SynthesizeRequest
        let req = SynthesizeRequest(type: "synthesize", text: "テスト発話", speed: 1.2, pitch: 0.9, mode: "stream")
        let reqData = try encoder.encode(req)
        let decodedReq = try decoder.decode(SynthesizeRequest.self, from: reqData)
        XCTAssertEqual(decodedReq.text, "テスト発話")
        XCTAssertEqual(decodedReq.speed, 1.2)
        XCTAssertEqual(decodedReq.pitch, 0.9)
        XCTAssertEqual(decodedReq.mode, "stream")

        // 2. ServerStartEvent
        let startEvt = ServerStartEvent(type: "start", sampleRate: 16000, channels: 1, format: "pcm_f32")
        let startData = try encoder.encode(startEvt)
        let decodedStart = try decoder.decode(ServerStartEvent.self, from: startData)
        XCTAssertEqual(decodedStart.sampleRate, 16000)
        XCTAssertEqual(decodedStart.format, "pcm_f32")

        // 3. ServerDoneEvent
        let doneEvt = ServerDoneEvent(type: "done", duration: 1.5, samples: 24000, rtf: 0.015)
        let doneData = try encoder.encode(doneEvt)
        let decodedDone = try decoder.decode(ServerDoneEvent.self, from: doneData)
        XCTAssertEqual(decodedDone.samples, 24000)
        XCTAssertEqual(decodedDone.rtf, 0.015)

        // 4. ServerErrorEvent
        let errEvt = ServerErrorEvent(type: "error", message: "エラーメッセージ")
        let errData = try encoder.encode(errEvt)
        let decodedErr = try decoder.decode(ServerErrorEvent.self, from: errData)
        XCTAssertEqual(decodedErr.message, "エラーメッセージ")

        // 5. CancelRequest
        let cancelReq = CancelRequest(type: "cancel")
        let cancelData = try encoder.encode(cancelReq)
        let decodedCancel = try decoder.decode(CancelRequest.self, from: cancelData)
        XCTAssertEqual(decodedCancel.type, "cancel")
    }

    /// 内蔵 Web UI の HTML/CSS/JS 構文整合性検証
    func testWebClientHTMLIntegrity() {
        let html = WebClientHTML.content
        XCTAssertTrue(0 < html.count, "WebClientHTML が空であってはならない")
        XCTAssertTrue(html.contains("<!DOCTYPE html>"), "DOCTYPE宣言を含むこと")
        XCTAssertTrue(html.contains("SpikeSpeech Web"), "タイトル表記を含むこと")
        XCTAssertTrue(html.contains("AudioContext"), "Web Audio API 初期化コードを含むこと")
        XCTAssertTrue(html.contains("new WebSocket"), "WebSocket 接続コードを含むこと")
        XCTAssertTrue(html.contains("waveform-canvas"), "波形可視化 Canvas を含むこと")
    }

    /// HTTP 静的ページ配信（GET /）の結合検証
    func testHTTPGetWebUI() throws {
        let testPort: UInt16 = 18081
        let server = SpikeSpeechWebServer(port: testPort, host: "127.0.0.1")
        try server.start()

        let expectation = self.expectation(description: "HTTP GET /")
        let url = URL(string: "http://127.0.0.1:\(testPort)/")!

        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            XCTAssertNil(error, "HTTP リクエストでエラーが発生してはならない")
            guard let httpResponse = response as? HTTPURLResponse else {
                XCTFail("レスポンスが HTTPURLResponse ではない")
                expectation.fulfill()
                return
            }

            XCTAssertEqual(httpResponse.statusCode, 200, "ステータスコードは 200 OK であること")
            let contentType = httpResponse.value(forHTTPHeaderField: "Content-Type")
            XCTAssertTrue(contentType?.contains("text/html") ?? false, "Content-Type に text/html を含むこと")

            guard let data = data, let bodyString = String(data: data, encoding: .utf8) else {
                XCTFail("レスポンスボディが空または不正")
                expectation.fulfill()
                return
            }
            XCTAssertTrue(bodyString.contains("SpikeSpeech Web"), "HTML に SpikeSpeech Web が含まれること")
            expectation.fulfill()
        }
        task.resume()

        wait(for: [expectation], timeout: 5.0)
        server.stop()
    }

    /// WebSocket E2E 音声合成検証 (WAV 一括モード)
    func testWebSocketSynthesizeWavMode() throws {
        let testPort: UInt16 = 18082
        let server = SpikeSpeechWebServer(port: testPort, host: "127.0.0.1")
        try server.start()

        let client = SpikeSpeechWebClient(host: "127.0.0.1", port: testPort)

        let expStart = self.expectation(description: "WebSocket Start Event")
        let expAudio = self.expectation(description: "WebSocket Audio WAV Data")
        let expDone = self.expectation(description: "WebSocket Done Event")

        var receivedStart: ServerStartEvent?
        var receivedWavData = Data()
        var receivedDone: ServerDoneEvent?

        client.onStart = { start in
            receivedStart = start
            expStart.fulfill()
        }

        client.onAudioChunk = { data in
            receivedWavData.append(data)
            expAudio.fulfill()
        }

        client.onDone = { done in
            receivedDone = done
            expDone.fulfill()
        }

        client.onError = { msg in
            XCTFail("WebSocket エラー発生: \(msg)")
        }

        client.connect()

        // 接続確立を待機
        Thread.sleep(forTimeInterval: 0.2)

        try client.synthesize(text: "こんにちは", speed: 1.0, pitch: 1.0, mode: "wav")

        wait(for: [expStart, expAudio, expDone], timeout: 10.0)

        // 受信データの詳細検証
        XCTAssertEqual(receivedStart?.format, "wav")
        XCTAssertEqual(receivedStart?.sampleRate, 16000)

        XCTAssertTrue(44 < receivedWavData.count, "WAVデータはヘッダ (44バイト) を超えるサイズであること")

        // WAV ヘッダの規格検証
        let riffHeader = String(data: receivedWavData.subdata(in: 0..<4), encoding: .ascii)
        let waveHeader = String(data: receivedWavData.subdata(in: 8..<12), encoding: .ascii)
        XCTAssertEqual(riffHeader, "RIFF")
        XCTAssertEqual(waveHeader, "WAVE")

        guard let done = receivedDone else {
            XCTFail("Done イベントが未受信")
            client.disconnect()
            server.stop()
            return
        }
        XCTAssertTrue(0 < done.samples, "サンプル数が 0 より大きいこと")
        XCTAssertTrue(0.0 < done.duration, "音声長が 0 より大きいこと")
        XCTAssertTrue(done.rtf < 0.1, "RTF は 0.1 未満であること")

        client.disconnect()
        server.stop()
    }

    /// WebSocket E2E 音声合成検証 (ストリーミング PCM モード)
    func testWebSocketSynthesizeStreamingMode() throws {
        let testPort: UInt16 = 18083
        let server = SpikeSpeechWebServer(port: testPort, host: "127.0.0.1")
        try server.start()

        let client = SpikeSpeechWebClient(host: "127.0.0.1", port: testPort)

        let expStart = self.expectation(description: "WebSocket Streaming Start")
        let expDone = self.expectation(description: "WebSocket Streaming Done")

        var receivedStart: ServerStartEvent?
        var receivedChunks: [Data] = []
        var receivedDone: ServerDoneEvent?

        client.onStart = { start in
            receivedStart = start
            expStart.fulfill()
        }

        client.onAudioChunk = { data in
            receivedChunks.append(data)
        }

        client.onDone = { done in
            receivedDone = done
            expDone.fulfill()
        }

        client.onError = { msg in
            XCTFail("WebSocket エラー発生: \(msg)")
        }

        client.connect()
        Thread.sleep(forTimeInterval: 0.2)

        try client.synthesize(text: "リアルタイム音声合成のテストです。", speed: 1.0, pitch: 1.0, mode: "stream")

        wait(for: [expStart, expDone], timeout: 10.0)

        XCTAssertEqual(receivedStart?.format, "pcm_f32")
        XCTAssertTrue(0 < receivedChunks.count, "ストリーミングチャンクを 1 個以上受信すること")

        // 全チャンクの合計 Float サンプル数を算出
        var totalFloats = 0
        var nanCount = 0
        var infCount = 0
        for chunk in receivedChunks {
            let floatCount = chunk.count / MemoryLayout<Float>.size
            totalFloats += floatCount
            chunk.withUnsafeBytes { rawPtr in
                let floatPtr = rawPtr.bindMemory(to: Float.self)
                var fIdx = 0
                while fIdx < floatCount {
                    let s = floatPtr[fIdx]
                    if s.isNaN {
                        nanCount += 1
                    }
                    if s.isInfinite {
                        infCount += 1
                    }
                    fIdx += 1
                }
            }
        }

        XCTAssertEqual(nanCount, 0, "ストリーミングサンプルに NaN が含まれてはならない")
        XCTAssertEqual(infCount, 0, "ストリーミングサンプルに Inf が含まれてはならない")

        guard let done = receivedDone else {
            XCTFail("Done イベントが未受信")
            client.disconnect()
            server.stop()
            return
        }
        XCTAssertEqual(totalFloats, done.samples, "受信サンプル総数が Done イベントのサンプル数と完全一致すること")
        XCTAssertTrue(done.rtf < 0.1, "ストリーミング合成 RTF は 0.1 未満であること")

        client.disconnect()
        server.stop()
    }

    /// 空文字送信時のエラー通知ハンドリング検証
    func testWebSocketEmptyTextErrorHandling() throws {
        let testPort: UInt16 = 18084
        let server = SpikeSpeechWebServer(port: testPort, host: "127.0.0.1")
        try server.start()

        let client = SpikeSpeechWebClient(host: "127.0.0.1", port: testPort)
        let expError = self.expectation(description: "WebSocket Empty Text Error")

        var receivedErrorMessage: String?
        client.onError = { msg in
            receivedErrorMessage = msg
            expError.fulfill()
        }

        client.connect()
        Thread.sleep(forTimeInterval: 0.2)

        try client.synthesize(text: "   ", speed: 1.0, pitch: 1.0, mode: "stream")

        wait(for: [expError], timeout: 5.0)

        XCTAssertNotNil(receivedErrorMessage)
        XCTAssertTrue(receivedErrorMessage?.contains("空") ?? false, "エラーメッセージに『空』が含まれること")

        client.disconnect()
        server.stop()
    }

    /// ストリーミング音声合成の即時キャンセル機能の検証
    func testWebSocketCancellation() throws {
        let testPort: UInt16 = 18085
        let server = SpikeSpeechWebServer(port: testPort, host: "127.0.0.1")
        try server.start()

        let client = SpikeSpeechWebClient(host: "127.0.0.1", port: testPort)
        let expStart = self.expectation(description: "WebSocket Start for Cancellation")
        let expFirstChunk = self.expectation(description: "First Chunk Received")

        var receivedChunkCount = 0
        var doneReceived = false
        var cancelSent = false

        client.onStart = { _ in
            expStart.fulfill()
        }

        client.onAudioChunk = { _ in
            receivedChunkCount += 1
            if cancelSent != true {
                cancelSent = true
                expFirstChunk.fulfill()
                // 最初のチャンクを受信した直後にキャンセル要求を送信し、後続の生成・配信が停止されるかを検証する
                do {
                    try client.cancel()
                } catch {
                    XCTFail("キャンセル送信失敗: \(error)")
                }
            }
        }

        client.onDone = { _ in
            doneReceived = true
        }

        client.onError = { msg in
            XCTFail("予期せぬエラー: \(msg)")
        }

        client.connect()
        Thread.sleep(forTimeInterval: 0.2)

        // 中断を確実に検知できるよう、十分なフレーム数を要する複数文テキストを指定する
        var longText = ""
        var rep = 0
        while rep < 6 {
            longText += "ストリーミング音声合成のキャンセルテストです。音声が即座に中断されるか検証します。"
            rep += 1
        }
        try client.synthesize(text: longText, speed: 1.0, pitch: 1.0, mode: "stream")

        // 配信開始と最初のチャンク到着を確実に待機
        wait(for: [expStart, expFirstChunk], timeout: 5.0)

        // キャンセル信号の到達とバックグラウンド処理の停止を待機
        Thread.sleep(forTimeInterval: 0.5)

        // 通常完了であれば 300 チャンク程度届くところ、キャンセルによって大幅に少ないフレーム数で打ち切られていることを確認する
        XCTAssertTrue(0 < receivedChunkCount, "キャンセル前に少なくとも 1 個以上のチャンクを受信していること")
        XCTAssertTrue(receivedChunkCount < 200, "キャンセルによりチャンク送信が早期打ち切りされていること (受信数: \(receivedChunkCount))")
        XCTAssertTrue(doneReceived != true, "キャンセルされた場合は Done イベントが送信されないこと")

        client.disconnect()
        server.stop()
    }
}
