// swift-tools-version: 6.0
import Foundation
import PackageDescription

var products: [Product] = [
    .library(
        name: "SpikeSpeech",
        targets: ["SpikeSpeech"]
    ),
    .executable(
        name: "synthesize",
        targets: ["synthesize"]
    ),
    .executable(
        name: "train",
        targets: ["train"]
    ),
    .executable(
        name: "benchmark",
        targets: ["benchmark"]
    ),
    .library(
        name: "SpikeSpeechWeb",
        targets: ["SpikeSpeechWeb"]
    ),
    .executable(
        name: "spikespeech-web",
        targets: ["spikespeech-web"]
    ),
]

var targets: [Target] = [
    .target(
        name: "SpikeSpeech",
        dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXOptimizers", package: "mlx-swift")
        ],
        path: "Sources/SpikeSpeech"
    ),
    .target(
        name: "SpikeSpeechWeb",
        dependencies: [
            "SpikeSpeech"
        ],
        path: "Sources/SpikeSpeechWeb"
    ),
    .executableTarget(
        name: "synthesize",
        dependencies: [
            "SpikeSpeech"
        ],
        path: "script/synthesize"
    ),
    .executableTarget(
        name: "train",
        dependencies: [
            "SpikeSpeech",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXOptimizers", package: "mlx-swift")
        ],
        path: "script/train"
    ),
    .executableTarget(
        name: "benchmark",
        dependencies: [
            "SpikeSpeech"
        ],
        path: "script/benchmark"
    ),
    .executableTarget(
        name: "spikespeech-web",
        dependencies: [
            "SpikeSpeech",
            "SpikeSpeechWeb"
        ],
        path: "script/web"
    ),
    .testTarget(
        name: "SpikeSpeechTests",
        dependencies: [
            "SpikeSpeech",
            "SpikeSpeechWeb",
            .product(name: "MLX", package: "mlx-swift")
        ],
        path: "Tests/SpikeSpeechTests"
    ),
]

// script/dataset/jsut.swift が実在する場合のみ dataset 実行可能ターゲットを動的追加
// これにより .gitignore でスクリプトが除外されている初期環境でもビルドを破損させず、実在時は SPM で安全に実行可能にする
let datasetScriptPath = "script/dataset/jsut.swift"
if FileManager.default.fileExists(atPath: datasetScriptPath) {
    products.append(.executable(name: "dataset", targets: ["dataset"]))
    targets.append(
        .executableTarget(
            name: "dataset",
            dependencies: [
                "SpikeSpeech",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
            ],
            path: "script/dataset",
            exclude: ["run_dataset.sh", ".gitkeep"],
            sources: ["jsut.swift"]
        )
    )
}

let package = Package(
    name: "SpikeSpeech",
    platforms: [
        .macOS(.v14)
    ],
    products: products,
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0")
    ],
    targets: targets
)
