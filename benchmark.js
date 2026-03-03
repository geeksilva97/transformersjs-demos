/**
 * benchmark.js - CPU vs WebGPU performance comparison
 *
 * Usage:
 *   node benchmark.js           # all demos, cpu + webgpu
 *   node benchmark.js --cpu     # cpu only
 *   node benchmark.js --gpu     # webgpu only
 *   node benchmark.js --demo 1  # single demo
 */

import { pipeline, TextStreamer } from '@huggingface/transformers';

const args    = process.argv.slice(2);
const cpuOnly = args.includes('--cpu');
const gpuOnly = args.includes('--gpu');
const demoFilter = args.includes('--demo') ? Number(args[args.indexOf('--demo') + 1]) : null;
const devices = gpuOnly ? ['webgpu'] : cpuOnly ? ['cpu'] : ['cpu', 'webgpu'];

function col(str, w) { return String(str ?? '').padEnd(w); }
function fmt(ms)     { return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`; }
function sep(c = '─', n = 70) { return c.repeat(n); }
function log(msg = '') { process.stdout.write(msg + '\n'); }
function t() { return performance.now(); }

// ── Demo 1: Sentiment Analysis ──

async function runDemo1(device) {
  const sentences = [
    "This is the best developer experience I have ever had. Absolutely love it.",
    "The documentation is completely useless and the examples do not even run.",
    "Perfect deployment, zero errors, zero downtime. Could not be happier.",
    "Tried to cancel my subscription three times and I am still being charged.",
    "The onboarding flow is so smooth I had my whole team set up in minutes.",
    "I love NodeCongress",
  ];

  const t0 = t();
  const classifier = await pipeline('text-classification', undefined, { device });
  const loadTime = t() - t0;

  const t1 = t();
  for (const s of sentences) await classifier(s);

  return { loadTime, inferTime: t() - t1, extra: `${sentences.length} sentences` };
}

// ── Demo 2: Semantic Search ──

async function runDemo2(device) {
  const corpus = [
    "Redis stores data structures in memory for sub-millisecond access times",
    "The garbage collector pauses your program to reclaim unused heap objects",
    "WebSockets maintain a persistent bidirectional channel between client and server",
    "Database indexes trade extra disk space and slower writes for faster lookups",
    "Docker containers share the host kernel but isolate processes with namespaces",
    "Node.js uses an event loop to handle many connections on a single thread without blocking",
    "JWT tokens encode claims as base64 JSON signed with a secret or key pair",
    "Load balancers distribute incoming traffic across multiple server instances",
    "Git stores snapshots of your project as a directed acyclic graph of commits",
    "Rate limiting protects services by capping how many requests a client can make",
  ];
  const queries = [
    "How does Node.js handle concurrency with a single thread?",
    "Why does my application sometimes freeze for a few milliseconds?",
    "How do developers track and manage version history of source code?",
  ];

  const t0 = t();
  const extractor = await pipeline('feature-extraction', undefined, { device });
  const loadTime = t() - t0;

  const t1 = t();
  for (const doc of [...corpus, ...queries]) {
    await extractor(doc, { pooling: 'mean', normalize: true });
  }

  return { loadTime, inferTime: t() - t1, extra: `${corpus.length} docs + ${queries.length} queries` };
}

// ── Demo 3: Translation ──

async function runDemo3(device) {
  const texts = [
    'The event loop is what allows Node.js to perform non-blocking I/O operations.',
    'The event loop is what allows Node.js to perform non-blocking I/O operations.',
    'The event loop is what allows Node.js to perform non-blocking I/O operations.',
    'The event loop is what allows Node.js to perform non-blocking I/O operations.',
  ];
  const pairs = [
    { src_lang: 'eng_Latn', tgt_lang: 'por_Latn' },
    { src_lang: 'eng_Latn', tgt_lang: 'spa_Latn' },
    { src_lang: 'eng_Latn', tgt_lang: 'fra_Latn' },
    { src_lang: 'eng_Latn', tgt_lang: 'deu_Latn' },
  ];

  const t0 = t();
  const translator = await pipeline('translation', 'Xenova/nllb-200-distilled-600M', { device });
  const loadTime = t() - t0;

  const t1 = t();
  for (let i = 0; i < texts.length; i++) {
    await translator(texts[i], { ...pairs[i], max_new_tokens: 256, num_beams: 4 });
  }

  return { loadTime, inferTime: t() - t1, extra: `${texts.length} translations (EN→PT/ES/FR/DE)` };
}

// ── Demo 4: Text Generation ──

async function runDemo4(device) {
  const t0 = t();
  const generator = await pipeline(
    'text-generation',
    'onnx-community/Llama-3.2-1B-Instruct-ONNX',
    { dtype: 'q4', device },
  );
  const loadTime = t() - t0;

  let tokenCount = 0;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    callback_function() { tokenCount++; },
  });

  const t1 = t();
  await generator(
    [
      { role: 'system', content: 'You are a grumpy senior engineer. Answer concisely with sarcasm.' },
      { role: 'user',   content: 'Guess what? A new JavaScript framework was just released!' },
    ],
    { max_new_tokens: 128, do_sample: false, streamer },
  );
  const inferTime = t() - t1;
  const tokPerSec = (tokenCount / (inferTime / 1000)).toFixed(1);

  return { loadTime, inferTime, extra: `${tokenCount} tokens @ ${tokPerSec} tok/s` };
}

// ── Runner ──

const DEMOS = [
  { id: 1, name: 'Sentiment Analysis', fn: runDemo1 },
  { id: 2, name: 'Semantic Search',    fn: runDemo2 },
  { id: 3, name: 'Translation',        fn: runDemo3 },
  { id: 4, name: 'Text Generation',    fn: runDemo4 },
].filter(d => demoFilter === null || d.id === demoFilter);

const results = [];

for (const demo of DEMOS) {
  log(`\n${sep('═')}`);
  log(`Demo ${demo.id}: ${demo.name}`);
  log(sep('═'));

  for (const device of devices) {
    log(`\n  Running on ${device.toUpperCase()}...`);
    let r;
    try {
      r = await demo.fn(device);
      log(`  Load time : ${fmt(r.loadTime)}`);
      log(`  Infer time: ${fmt(r.inferTime)}`);
      log(`  Total     : ${fmt(r.loadTime + r.inferTime)}`);
      if (r.extra) log(`  Info      : ${r.extra}`);
    } catch (err) {
      r = { error: err.message };
      log(`  ERROR: ${err.message}`);
    }
    results.push({ demo: demo.id, demoName: demo.name, device, ...r });
  }
}

// ── Summary Table ──

log(`\n\n${sep('═')}`);
log('BENCHMARK SUMMARY');
log(sep('═'));
log(`  ${col('Demo', 24)} ${col('Device', 8)} ${col('Load', 10)} ${col('Infer', 10)} ${col('Total', 10)}  Info`);
log(`  ${sep('─', 66)}`);

for (const r of results) {
  const label = `Demo ${r.demo} (${r.demoName})`;
  if (r.error) {
    log(`  ${col(label, 24)} ${col(r.device, 8)} ERROR: ${r.error}`);
    continue;
  }
  const total = r.loadTime + r.inferTime;
  log(`  ${col(label, 24)} ${col(r.device, 8)} ${col(fmt(r.loadTime), 10)} ${col(fmt(r.inferTime), 10)} ${col(fmt(total), 10)}  ${r.extra ?? ''}`);
}

// ── Speedup ──

if (devices.length > 1) {
  log(`\n${sep('─', 50)}`);
  log('SPEEDUP  (CPU infer / GPU infer)');
  log(sep('─', 50));

  for (const demo of DEMOS) {
    const cpu = results.find(r => r.demo === demo.id && r.device === 'cpu');
    const gpu = results.find(r => r.demo === demo.id && r.device === 'webgpu');
    if (!cpu || !gpu || cpu.error || gpu.error) continue;
    const speedup = (cpu.inferTime / gpu.inferTime).toFixed(2);
    const winner  = cpu.inferTime > gpu.inferTime ? 'GPU faster' : 'CPU faster';
    log(`  Demo ${demo.id} (${demo.name}): ${speedup}x  [${winner}]`);
  }
}

log('');
