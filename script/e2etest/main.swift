import Foundation

/// SpikeSpeech E2E テストスイート CLI エントリポイント
func main() {
    let args = CommandLine.arguments
    var targetTier = "all"
    var corpusDir: String? = nil

    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "-d", "--corpus-dir":
            let nextIdx = i + 1
            if nextIdx < args.count {
                corpusDir = args[nextIdx]
                i += 1
            }
        case "-t", "--tier":
            let nextIdx = i + 1
            if nextIdx < args.count {
                targetTier = args[nextIdx].lowercased()
                i += 1
            }
        case "-h", "--help":
            print("Usage: e2etest [-d <corpus-dir>] [-t all|1|2|3|4|5|phasec|websocket]")
            exit(0)
        default:
            break
        }
        i += 1
    }

    E2ETestContext.shared.corpusDirectory = corpusDir

    print("==========================================================")
    print("SpikeSpeech E2E テストスイート")
    print("==========================================================")
    print("ターゲット Tier: \(targetTier)")
    if let dir = corpusDir {
        print("コーパスディレクトリ: \(dir)")
    } else {
        print("コーパスディレクトリ: (未指定: コーパス依存テストはスキップ)")
    }
    print("----------------------------------------------------------")

    switch targetTier {
    case "1":
        runTier1Tests()
    case "2":
        runTier2Tests()
    case "3":
        runTier3Tests()
    case "4":
        runTier4Tests()
    case "5":
        runTier5Tests()
    case "phasec":
        runPhaseCTests()
    case "websocket":
        runWebSocketE2ETests()
    default:
        runTier1Tests()
        runTier2Tests()
        runTier3Tests()
        runTier4Tests()
        runTier5Tests()
        runPhaseCTests()
        runWebSocketE2ETests()
    }

    print("==========================================================")
    let passed = E2ETestContext.shared.passedCount
    let failed = E2ETestContext.shared.failedCount
    let skipped = E2ETestContext.shared.skippedCount
    print("E2E 結果: 成功: \(passed), 失敗: \(failed), スキップ: \(skipped)")
    print("==========================================================")

    if 0 < failed {
        exit(1)
    } else {
        exit(0)
    }
}

main()
