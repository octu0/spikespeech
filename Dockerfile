# ==============================================================================
# Stage 1: Build stage (Pure Swift 推論バイナリのビルド)
# ==============================================================================
FROM swift:6.0-noble AS builder

WORKDIR /workspace

# ソースコード一式をコピー
COPY Package.swift ./
COPY Sources ./Sources
COPY script ./script

# spikespeech-web 実行可能バイナリを Release モードでビルド
# ※ Linux 上では Package.swift の #if os(Linux) により MLX や C++/Fortran 等の外部依存ゼロで Pure Swift ビルドされる
RUN swift build -c release --product spikespeech-web

# ==============================================================================
# Stage 2: Runtime stage (最小ランタイムコンテナ)
# ==============================================================================
FROM swift:6.0-noble-slim AS runner

# Cloud Run セキュリティベストプラクティス: 非 root ユーザーの作成
RUN useradd -u 1000 -m -s /bin/bash appuser

WORKDIR /app

# ビルド成果物および SNN 重みファイルを配置
COPY --from=builder /workspace/.build/release/spikespeech-web /app/spikespeech-web
COPY Models /app/Models

RUN chown -R appuser:appuser /app

USER appuser

# Cloud Run 標準のポート環境変数
ENV PORT=8080
ENV WEIGHTS_PATH=/app/Models/weights.json

EXPOSE 8080

ENTRYPOINT ["/app/spikespeech-web"]
