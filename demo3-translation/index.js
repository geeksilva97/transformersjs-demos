// Demo 3 - Translation
//
// opus-mt-en-es is a lightweight English→Spanish model.
// Unlike multilingual models (NLLB-200), no src_lang/tgt_lang needed.

import { pipeline } from '@huggingface/transformers';

const translator = await pipeline(
  'translation',
  'Xenova/opus-mt-en-es',
);

const texts = [
  'Hello, how are you doing?',
  'The event loop is what allows Node.js to perform non-blocking I/O operations.',
];

for (const text of texts) {
  const [result] = await translator(text);
  console.log(`EN: "${text}"`);
  console.log(`ES: "${result.translation_text}"\n`);
}
