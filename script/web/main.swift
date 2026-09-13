import Foundation
import SpikeSpeech
import SpikeSpeechWeb

/// spikespeech-web CLI
///
/// SpikeSpeech WebSocket & HTTP 音声合成サーバーを起動するコマンドラインツール。

func printUsage() {
    print("""
    Usage: spikespeech-web -w <weights.json> [-p <port>] [-h <host>]

    Options:
      -w, --weights <path>        SNN 重み JSON ファイルパス (必須)
      -p, --port <port>           バインドポート (デフォルト: 8080)
      -h, --host <host>           バインドホスト (デフォルト: 0.0.0.0)
      --help                      このヘルプを表示
    """)
}

var port: UInt16 = 8080
var host: String = "0.0.0.0"
var weightsPath: String?

let args = CommandLine.arguments
var i = 1
while i < args.count {
    let arg = args[i]
    switch arg {
    case "-p", "--port":
        if i + 1 < args.count {
            i += 1
            if let p = UInt16(args[i]) {
                port = p
            }
        }
    case "-h", "--host":
        if i + 1 < args.count {
            i += 1
            host = args[i]
        }
    case "-w", "--weights":
        if i + 1 < args.count {
            i += 1
            weightsPath = args[i]
        }
    case "--help":
        printUsage()
        exit(0)
    default:
        break
    }
    i += 1
}

// 疑似値やランダム重みへのフォールバックを排除し、学習済み重みファイルの指定を必須とする
guard let path = weightsPath else {
    print("エラー: SNN 音響モデルの重みファイル (-w / --weights <path>) の指定は必須です。")
    print("例: spikespeech-web -w Models/weights.json --port 8080")
    print("ヘルプ表示: spikespeech-web --help")
    exit(1)
}

let url = URL(fileURLWithPath: path)
let weights: SpikingNetworkWeights
do {
    let data = try Data(contentsOf: url)
    weights = try JSONDecoder().decode(SpikingNetworkWeights.self, from: data)
    print("[SpikeSpeechWeb] 学習済み重みファイルを読み込みました: \(path)")
} catch {
    print("エラー: 重みファイル '\(path)' の読み込みまたは JSON 解析に失敗しました: \(error)")
    exit(1)
}

let engine = SpikeSpeechEngine(weights: weights)
let server = SpikeSpeechWebServer(port: port, host: host, engine: engine)

do {
    try server.start()
    print("==================================================")
    print("SpikeSpeech Web サーバーが起動しました")
    print("==================================================")
    var displayHost = host
    if host == "0.0.0.0" {
        displayHost = "localhost"
    }
    print("URL:           http://\(displayHost):\(port)/")
    print("WebSocket:     ws://\(displayHost):\(port)/ws")
    print("サンプリング:  16,000 Hz (16-bit Mono)")
    print("SNN 層数:      \(weights.numLayers) 層 (隠れ層: \(weights.maxHiddenDim))")
    print("--------------------------------------------------")
    print("ブラウザで上記 URL にアクセスすると、GUI から音声を試聴できます。")
    print("Ctrl+C で停止します。")
    print("--------------------------------------------------")

    // シグナル待機でイベントループを維持
    signal(SIGINT) { _ in
        print("\n[SpikeSpeechWeb] サーバーを終了します...")
        exit(0)
    }
    signal(SIGTERM) { _ in
        print("\n[SpikeSpeechWeb] サーバーを終了します...")
        exit(0)
    }

    dispatchMain()
} catch {
    print("[SpikeSpeechWeb] サーバーの起動に失敗しました: \(error)")
    exit(1)
}
