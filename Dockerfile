# ==============================================================================
FROM swift:6.2-noble AS builder

WORKDIR /workspace

COPY Package.swift ./
COPY Sources ./Sources
COPY script ./script

RUN swift build -c release --product spikespeech-web

# ==============================================================================
FROM swift:6.2-noble-slim AS runner

WORKDIR /app

COPY --from=builder /workspace/.build/release/spikespeech-web /app/spikespeech-web
COPY Models /app/Models

ENV PORT=8080
ENV WEIGHTS_PATH=/app/Models/weights.json

EXPOSE 8080

ENTRYPOINT ["/app/spikespeech-web"]
