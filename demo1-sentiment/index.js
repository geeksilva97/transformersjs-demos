import { pipeline } from "@huggingface/transformers";

const classifier = await pipeline(
  "text-classification",
  // "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
);

const sentences = [
  "This is the best developer experience I have ever had. Absolutely love it.",
  "The documentation is completely useless and the examples do not even run.",
  "Perfect deployment, zero errors, zero downtime. Could not be happier.",
  "Tried to cancel my subscription three times and I am still being charged.",
  "The onboarding flow is so smooth I had my whole team set up in minutes.",
  "I love NodeCongress"
];

console.log("Sentiment Analysis Results\n");

for (const text of sentences) {
  const [result] = await classifier(text);
  console.log(result);
  const score = (result.score * 100).toFixed(1);

  console.log(`${result.label} (${score}%) - "${text}"`);
}
