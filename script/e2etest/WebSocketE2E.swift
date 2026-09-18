import Foundation
import SpikeSpeech
import SpikeSpeechWeb

/// WebSocket / WebServer E2E 結合テスト
func runWebSocketE2ETests() {
    print("--- Running WebSocket / WebServer E2E Tests ---")

    let testPort: UInt16 = 18095
    let server = SpikeSpeechWebServer(port: testPort, host: "127.0.0.1")

    do {
        try server.start()
        print("  [Server] Started test server on port \(testPort)")

        let url = URL(string: "http://127.0.0.1:\(testPort)/")!
        let semaphore = DispatchSemaphore(value: 0)

        final class ResponseHolder: @unchecked Sendable {
            var statusCode = 0
            var hasContent = false
        }
        let holder = ResponseHolder()

        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let httpResponse = response as? HTTPURLResponse {
                holder.statusCode = httpResponse.statusCode
            }
            if let data = data, let bodyString = String(data: data, encoding: .utf8) {
                if bodyString.contains("SpikeSpeech Web") {
                    holder.hasContent = true
                }
            }
            semaphore.signal()
        }
        task.resume()

        let timeout = DispatchTime.now() + .seconds(5)
        let waitResult = semaphore.wait(timeout: timeout)
        if waitResult == .timedOut {
            E2ETestContext.shared.recordFail("WebSocketE2E: HTTP GET / timed out")
        } else {
            e2eAssertEqual(holder.statusCode, 200, "WebSocketE2E: status code is 200 OK")
            e2eAssertTrue(holder.hasContent, "WebSocketE2E: body contains SpikeSpeech Web")
        }

        server.stop()
        print("  [Server] Stopped test server")
    } catch {
        E2ETestContext.shared.recordFail("WebSocketE2E: Server failed to start: \(error)")
    }
}
