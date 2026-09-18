import Foundation

/// E2E テスト実行コンテキストおよびアサーションユーティリティ
final class E2ETestContext {
    nonisolated(unsafe) static let shared = E2ETestContext()

    var passedCount = 0
    var failedCount = 0
    var skippedCount = 0
    var corpusDirectory: String? = nil

    func recordPass() {
        passedCount += 1
    }

    func recordFail(_ message: String, file: String = #file, line: Int = #line) {
        failedCount += 1
        let fileName = URL(fileURLWithPath: file).lastPathComponent
        print("  [FAIL] \(fileName):\(line) - \(message)")
    }

    func recordSkip(_ reason: String) {
        skippedCount += 1
        print("  [SKIP] \(reason)")
    }
}

func e2eAssert(_ condition: Bool, _ message: String = "", file: String = #file, line: Int = #line) {
    if condition {
        E2ETestContext.shared.recordPass()
    } else {
        var msg = "Assertion failed"
        if message.isEmpty != true {
            msg = message
        }
        E2ETestContext.shared.recordFail(msg, file: file, line: line)
    }
}

func e2eAssertEqual<T: Equatable>(_ a: T, _ b: T, _ message: String = "", file: String = #file, line: Int = #line) {
    if a == b {
        E2ETestContext.shared.recordPass()
    } else {
        var msg = "\(a) != \(b)"
        if message.isEmpty != true {
            msg = "\(message) (\(a) != \(b))"
        }
        E2ETestContext.shared.recordFail(msg, file: file, line: line)
    }
}

func e2eAssertNotEqual<T: Equatable>(_ a: T, _ b: T, _ message: String = "", file: String = #file, line: Int = #line) {
    if a != b {
        E2ETestContext.shared.recordPass()
    } else {
        var msg = "\(a) == \(b)"
        if message.isEmpty != true {
            msg = "\(message) (\(a) == \(b))"
        }
        E2ETestContext.shared.recordFail(msg, file: file, line: line)
    }
}

func e2eAssertTrue(_ condition: Bool, _ message: String = "", file: String = #file, line: Int = #line) {
    e2eAssert(condition, message, file: file, line: line)
}

func e2eAssertFalse(_ condition: Bool, _ message: String = "", file: String = #file, line: Int = #line) {
    e2eAssert(condition != true, message, file: file, line: line)
}
