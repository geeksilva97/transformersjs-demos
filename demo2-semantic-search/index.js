/**
 * Demo 2 - Semantic Search with Embeddings
 *
 * No vector database. No API. Just a model that
 * understands language, running inside Node.js.
 */

import { pipeline } from '@huggingface/transformers';

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

// ── Cosine Similarity ──

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ── Search ──

function rankDocuments(queryEmbedding, corpusEmbeddings) {
  return corpusEmbeddings
    .map((embedding, index) => ({
      text: corpus[index],
      score: cosineSimilarity(queryEmbedding, embedding),
    }))
    .sort((a, b) => b.score - a.score);
}

function printResults(query, results, topK = 5) {
  console.log(`\nQuery: "${query}"\n`);
  console.log('   Rank  Score   Document');
  console.log('   ────  ─────   ────────');

  for (let i = 0; i < topK; i++) {
    const { text, score } = results[i];
    const rank = String(i + 1).padStart(4);
    console.log(`   ${rank}  ${score.toFixed(3)}   ${text}`);
  }
}

// ── Main ──

async function main() {
  console.log('Loading embedding model...\n');

  const extractor = await pipeline(
    'feature-extraction',
    // 'Xenova/all-MiniLM-L6-v2',
  );

  // Embed the entire corpus once
  console.log(`Embedding ${corpus.length} documents...`);

  const corpusEmbeddings = [];
  for (const doc of corpus) {
    const output = await extractor(doc, {
      pooling: 'mean',
      normalize: true,
    });
    corpusEmbeddings.push(Array.from(output.data));
  }

  console.log('Done. Ready to search.\n');
  console.log('═'.repeat(60));

  // ── Queries that prove semantic understanding ──
  //
  // None of these share keywords with their best match.
  // The model has to UNDERSTAND meaning, not match words.

  const queries = [
    // Matches "event loop" doc - no shared keywords
    "How does Node.js handle concurrency with a single thread?",

    // Matches "garbage collector" doc - no shared keywords
    "Why does my application sometimes freeze for a few milliseconds?",

    // Matches "Git" doc - no shared keywords
    "How do developers track and manage version history of source code?",
  ];

  for (const query of queries) {
    const output = await extractor(query, {
      pooling: 'mean',
      normalize: true,
    });
    const queryEmbedding = Array.from(output.data);
    const results = rankDocuments(queryEmbedding, corpusEmbeddings);
    printResults(query, results);
  }

  console.log('\n' + '═'.repeat(60));
  console.log('\nNo network calls. No API keys. Just npm install.');
}

main();
