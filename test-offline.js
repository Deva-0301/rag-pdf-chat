// test-offline.js — checks the parts of the pipeline that need no API key.
// Run with:  npm run test:offline

const assert = require("assert");
const { chunkText, cosineSimilarity, topK, buildPrompt } = require("./rag");

// --- chunkText ---
const sample =
  "A deadlock occurs when processes wait for each other forever. " +
  "It needs four conditions to hold at the same time. ".repeat(40);
const chunks = chunkText(sample, 300, 60);
assert(chunks.length > 1, "long text should produce several chunks");
assert(chunks.every((c) => c.length <= 300), "no chunk should exceed the size limit");
assert.deepStrictEqual(chunkText("   "), [], "blank text should give no chunks");
assert.throws(() => chunkText("abc", 100, 100), "overlap >= size should throw");
console.log(`chunkText OK (${chunks.length} chunks, longest ${Math.max(...chunks.map((c) => c.length))} chars)`);

// --- cosineSimilarity ---
assert.strictEqual(cosineSimilarity([1, 0], [1, 0]), 1);
assert.strictEqual(cosineSimilarity([1, 0], [0, 1]), 0);
assert.strictEqual(cosineSimilarity([0, 0], [1, 1]), 0);
assert(Math.abs(cosineSimilarity([1, 1], [2, 2]) - 1) < 1e-9, "same direction => 1");
console.log("cosineSimilarity OK");

// --- topK ---
const fake = [
  { text: "about cats", page: 1, embedding: [1, 0, 0] },
  { text: "about dogs", page: 2, embedding: [0, 1, 0] },
  { text: "about cats and dogs", page: 3, embedding: [0.7, 0.7, 0] },
];
const best = topK([1, 0, 0], fake, 2);
assert.strictEqual(best[0].text, "about cats");
assert.strictEqual(best[1].text, "about cats and dogs");
assert.strictEqual(best.length, 2);
console.log("topK OK");

// --- buildPrompt ---
const prompt = buildPrompt("What is a deadlock?", best);
assert(prompt.includes("ONLY the context"));
assert(prompt.includes("What is a deadlock?"));
assert(prompt.includes("page 1"));
console.log("buildPrompt OK");

console.log("\nAll offline tests passed.");
