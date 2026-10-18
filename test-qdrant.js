// test-qdrant.js — checks qdrant.js against your real Qdrant cluster using FAKE vectors.
// Run with:  npm run test:qdrant
//
// It uses a separate collection called "pdf_chunks_test" and deletes it at the end,
// so your real documents are never touched. No Gemini key or PDF is needed.

// Must be set BEFORE requiring qdrant.js so the test collection name is used.
process.env.QDRANT_COLLECTION = "pdf_chunks_test";

const assert = require("assert");
const q = require("./qdrant");

// A fake "embedding": all zeros except one 1. Different positions = different directions,
// so vec(0) is identical to vec(0) (score 1) and unrelated to vec(1) (score 0).
function vec(position) {
  const v = new Array(3072).fill(0);
  v[position] = 1;
  return v;
}

async function main() {
  console.log("Connecting to", process.env.QDRANT_URL ? "your cluster..." : "(QDRANT_URL missing!)");

  // 1. Creating the collection twice must be safe.
  await q.ensureCollection();
  await q.ensureCollection();
  console.log("ok  ensureCollection (twice)");

  // 2. Save two fake documents. Chunk "alpha" in doc A and chunk "delta" in doc B
  //    have the SAME vector on purpose, to prove the docId filter really separates them.
  await q.saveChunks(
    "doc-A",
    "a.pdf",
    [
      { text: "alpha about cats", page: 1, embedding: vec(0) },
      { text: "beta about dogs", page: 2, embedding: vec(1) },
      { text: "gamma about birds", page: 3, embedding: vec(2) },
    ],
    "hash-A" // the PDF's fingerprint
  );
  await q.saveChunks(
    "doc-B",
    "b.pdf",
    [
      { text: "delta from another document", page: 1, embedding: vec(0) },
      { text: "epsilon from another document", page: 2, embedding: vec(3) },
    ],
    "hash-B"
  );
  console.log("ok  saveChunks");

  // Duplicate detection: a known fingerprint finds its document, an unknown one finds nothing.
  assert.deepStrictEqual(await q.findDocumentByHash("hash-A"), {
    docId: "doc-A",
    name: "a.pdf",
    chunks: 3,
  });
  assert.strictEqual((await q.findDocumentByHash("hash-B")).docId, "doc-B");
  assert.strictEqual(await q.findDocumentByHash("hash-unknown"), null);
  console.log("ok  findDocumentByHash (known and unknown fingerprints)");

  // 3. Search doc A with the "cats" vector.
  const hits = await q.searchChunks("doc-A", vec(0), 4);
  assert.strictEqual(hits.length, 3, "doc A has exactly 3 chunks");
  assert.strictEqual(hits[0].text, "alpha about cats", "closest chunk should come first");
  assert(hits[0].score > 0.99, "identical vectors should score about 1");
  assert(hits.every((h) => !h.text.includes("another document")), "doc B must not leak in");
  assert.strictEqual(hits[0].page, 1);
  console.log("ok  searchChunks (closest first, filtered to one document)");

  // 4. Limit k works.
  const top1 = await q.searchChunks("doc-A", vec(1), 1);
  assert.strictEqual(top1.length, 1);
  assert.strictEqual(top1[0].text, "beta about dogs");
  console.log("ok  searchChunks limit");

  // 5. Listing and existence.
  const docs = await q.listDocuments();
  assert.strictEqual(docs.length, 2);
  assert.deepStrictEqual(docs.map((d) => [d.docId, d.name, d.chunks]), [
    ["doc-A", "a.pdf", 3],
    ["doc-B", "b.pdf", 2],
  ]);
  assert.strictEqual(await q.documentExists("doc-A"), true);
  assert.strictEqual(await q.documentExists("nope"), false);
  console.log("ok  listDocuments, documentExists");

  // 6. Deleting one document leaves the other alone.
  await q.deleteDocument("doc-A");
  assert.strictEqual(await q.documentExists("doc-A"), false);
  assert.strictEqual(await q.documentExists("doc-B"), true);
  assert.strictEqual((await q.listDocuments()).length, 1);
  console.log("ok  deleteDocument");

  // After deleting, the fingerprint is gone too, so the same PDF can be uploaded again.
  assert.strictEqual(await q.findDocumentByHash("hash-A"), null);
  console.log("ok  deleted document can be uploaded again");

  console.log("\nAll Qdrant tests passed.");
}

main()
  .catch((err) => {
    console.error("\nTEST FAILED:", err.message);
    if (err.data) console.error("details:", JSON.stringify(err.data));
    process.exitCode = 1;
  })
  .finally(async () => {
    // Remove the test collection, but only if it really is the test one.
    try {
      if (process.env.QDRANT_COLLECTION.endsWith("_test")) {
        await q.client().deleteCollection(process.env.QDRANT_COLLECTION);
        console.log("(test collection removed)");
      }
    } catch {
      /* nothing to clean up, or no connection */
    }
  });
