// Demo 3 - Text Generation with Streaming
// The self-aware concession: we spent 15 minutes arguing against LLMs,
// and now we're running one. The point isn't that LLMs are wrong - it's
// that they shouldn't be your *default*. But when you genuinely need
// generation, you still don't need an external API.

import { pipeline, TextStreamer } from "@huggingface/transformers";

// Pass --gpu to run on WebGPU (routes to Metal on macOS). Default: CPU.
const useGPU = process.argv.includes("--gpu");
const device = useGPU ? "webgpu" : "cpu";

// Quantized Llama 3.2 1B - runs entirely inside this Node.js process.
// Fallback: "HuggingFaceTB/SmolLM2-360M-Instruct" if load time is too slow at demo time.
console.log(`Loading Llama 3.2 1B Instruct (quantized) on ${device}...`);
const generator = await pipeline(
  "text-generation",
  "onnx-community/Llama-3.2-1B-Instruct-ONNX",
  { dtype: "q4", device }
);

const streamer = new TextStreamer(generator.tokenizer, {
  skip_prompt: true,
  callback_function(token) {
    process.stdout.write(token);
  },
});

const messages = [
  {
    role: "system",
    content:
      "You are a grumpy senior engineer who has seen every JS framework come and go. " +
      "Answer concisely with sacasm.",
  },
  {
    role: "user",
    content: "Guess what? A new JavaScript framework was just released!",
  },
];

console.log("Generating response from a local LLM (no API, no network)...\n");

await generator(messages, {
  max_new_tokens: 256,
  do_sample: true,
  temperature: 0.7,
  streamer,
});

// Newline after the streamed output
console.log();
