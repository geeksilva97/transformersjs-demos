// Demo 4 - Multilingual Translation
//
// NLLB-200 (No Language Left Behind) supports 200 languages.
// Unlike demo1-3, the translation pipeline takes explicit parameters:
// src_lang, tgt_lang, max_new_tokens, num_beams — no magic defaults.

import { pipeline } from '@huggingface/transformers';

const translator = await pipeline(
  'translation',
  'Xenova/nllb-200-distilled-600M',
);

const translations = [
  {
    text: 'The event loop is what allows Node.js to perform non-blocking I/O operations.',
    src_lang: 'eng_Latn',
    tgt_lang: 'por_Latn',
    label: 'EN → PT',
  },
  {
    text: 'The event loop is what allows Node.js to perform non-blocking I/O operations.',
    src_lang: 'eng_Latn',
    tgt_lang: 'spa_Latn',
    label: 'EN → ES',
  },
  {
    text: 'The event loop is what allows Node.js to perform non-blocking I/O operations.',
    src_lang: 'eng_Latn',
    tgt_lang: 'fra_Latn',
    label: 'EN → FR',
  },
  {
    text: 'The event loop is what allows Node.js to perform non-blocking I/O operations.',
    src_lang: 'eng_Latn',
    tgt_lang: 'deu_Latn',
    label: 'EN → DE',
  },
];

console.log('Source:');
console.log(`  "${translations[0].text}"\n`);

for (const { text, src_lang, tgt_lang, label } of translations) {
  const [result] = await translator(text, {
    src_lang,
    tgt_lang,
    max_new_tokens: 256,
    num_beams: 4,
    forced_bos_token_id: null,
  });

  console.log(`${label}:`);
  console.log(`  "${result.translation_text}"\n`);
}
