# ---- Stage 1: build the React (Vite) frontend ----
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json ./
RUN npm install
COPY web/ ./
RUN npm run build          # -> /web/dist

# ---- Stage 2: the Python media worker ----
# Runs on Linux because Agora's server SDK ships Linux binaries.
# Python 3.12 for the widest wheel availability (Agora SDK, torch, onnx).
FROM python:3.12-slim

# libgomp1: needed by onnxruntime / torch. ca-certificates: TLS to the APIs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch FIRST so silero-vad's torch dependency is satisfied
# without pulling the multi-GB CUDA build.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY media-worker/requirements.txt ./media-worker/requirements.txt
RUN pip install --no-cache-dir -r media-worker/requirements.txt

# App code
COPY shared ./shared
COPY media-worker ./media-worker
COPY models ./models
# Built frontend from stage 1 (the worker serves this at "/")
COPY --from=web /web/dist ./web/dist

# So `import shared...` works; local modules resolve from the media-worker cwd.
ENV PYTHONPATH=/app
ENV WEB_DIR=/app/web/dist
WORKDIR /app/media-worker

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
