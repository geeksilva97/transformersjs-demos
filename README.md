# Transformers.js Demos

Demo code for the talk **"Forget OpenAI - Run Hugging Face Inside Your Node.js Process"** at Node Congress 2026.

## Prerequisites

- Node.js 18+
- Internet connection on first run (models are downloaded and cached locally)

```bash
npm install
```

## Demos

### Demo 1 - Sentiment Analysis

```bash
npm run demo1
```

Classifies sentiment on a set of sentences using DistilBERT. No API, no network call at inference time. Kills the myth that AI features require an LLM or an external service.

### Demo 2 - Semantic Search

```bash
npm run demo2
```

Embeds a small corpus and runs semantic search using cosine similarity. Queries match documents with no shared keywords - the model understands meaning, not just words. No vector database, no API.

### Demo 3 - Text Generation with Streaming

```bash
npm run demo3
```

Runs a quantized Llama 3.2 1B model entirely inside the Node.js process with token streaming. Falls back to `HuggingFaceTB/SmolLM2-360M-Instruct` if load time is a concern at demo time.

### Demo 4 - WebGPU Acceleration

```bash
npm run demo4
```

Same embedding model from demo2, now running on the GPU via WebGPU. Works on any WebGPU-capable device (Nvidia, AMD, Apple Silicon). On macOS, WebGPU routes to Metal automatically.

> **Note:** This demo requires `@huggingface/transformers` v4, which is currently in pre-release (`4.0.0-next.5`). The API is the same as v3 - only the device support changed. Demos 1-3 also run fine on v4.

## Model caching

On first run, each demo downloads its model from Hugging Face Hub and caches it locally. Subsequent runs are offline. Cache location: `~/.cache/huggingface/hub/`.
