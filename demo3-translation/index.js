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
  'The weather today is sunny with a slight chance of rain in the afternoon.',
  'She decided to take a walk in the park after finishing her work.',
];

for (const text of texts) {
  const [result] = await translator(text);
  console.log(`EN: "${text}"`);
  console.log(`ES: "${result.translation_text}"\n`);
}
