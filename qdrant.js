// qdrant.js — everything that talks to the Qdrant vector database lives in this file.
//
// How Qdrant stores our data:
//   collection = one big table (we use a single one, called "pdf_chunks")
//   point      = one row = one chunk of a PDF. It has three parts:
//                  id      a unique id
//                  vector  the chunk's embedding (the list of 3072 numbers from Gemini)
//                  payload extra info we attach: { docId, docName, fileHash, page, text, uploadedAt }
//
// To search inside ONE pdf we add a filter on docId, so chunks from other PDFs are ignored.
// fileHash is a fingerprint of the PDF's contents. If the same PDF is uploaded again,
// its fingerprint is already stored, so we can skip the upload.

require("dotenv").config();
const crypto = require("crypto");
const { QdrantClient } = require("@qdrant/js-client-rest");

const COLLECTION = process.env.QDRANT_COLLECTION || "pdf_chunks";
const VECTOR_SIZE = 3072; // the size of Gemini's default embedding (gemini-embedding-001)

// Created the first time it's needed, so the offline tests work without Qdrant keys.
let _client = null;
function client() {
  if (!_client) {
    if (!process.env.QDRANT_URL) {
      throw new Error("QDRANT_URL is missing. Add it to your .env file.");
    }
    _client = new QdrantClient({
      url: process.env.QDRANT_URL,
      apiKey: process.env.QDRANT_API_KEY || undefined, // a local Qdrant needs no key
      checkCompatibility: false, // skip the client/server version check
    });
  }
  return _client;
}

// A filter that means "only points whose docId equals this one".
function docFilter(docId) {
  return { must: [{ key: "docId", match: { value: docId } }] };
}

// A filter that means "only points whose fileHash equals this one".
function hashFilter(fileHash) {
  return { must: [{ key: "fileHash", match: { value: fileHash } }] };
}

/* ------------------------------------------------------------------ */
/* Create the collection (first run) and make sure it has its indexes  */
/* ------------------------------------------------------------------ */
async function ensureCollection() {
  const { exists } = await client().collectionExists(COLLECTION);
  if (!exists) {
    // size must match the embedding model's output. "Cosine" = compare by meaning-direction,
    // the same measure our hand-written cosineSimilarity() uses.
    await client().createCollection(COLLECTION, {
      vectors: { size: VECTOR_SIZE, distance: "Cosine" },
    });
  }

  // An index on a field makes filtering by it fast. We filter by docId (one PDF)
  // and by fileHash (duplicate check). This also adds a missing index to a collection
  // that was created before fileHash existed.
  const info = await client().getCollection(COLLECTION);
  const alreadyIndexed = info.payload_schema || {};
  for (const field of ["docId", "fileHash"]) {
    if (!alreadyIndexed[field]) {
      await client().createPayloadIndex(COLLECTION, {
        field_name: field,
        field_schema: "keyword",
        wait: true,
      });
    }
  }
}

/* ------------------------------------------------------------------ */
/* Save all chunks of one PDF                                          */
/* chunks = [{ text, page, embedding }]                                */
/* fileHash = fingerprint of the PDF (used to spot duplicate uploads)  */
/* ------------------------------------------------------------------ */
async function saveChunks(docId, docName, chunks, fileHash) {
  const uploadedAt = Date.now();
  const points = chunks.map((c) => ({
    id: crypto.randomUUID(), // Qdrant ids must be numbers or UUIDs
    vector: c.embedding,
    payload: { docId, docName, fileHash, page: c.page, text: c.text, uploadedAt },
  }));

  // Each vector is big (3072 numbers), so send the points in small groups,
  // the same idea as the batch of 50 in embedTexts().
  const BATCH = 25;
  for (let i = 0; i < points.length; i += BATCH) {
    await client().upsert(COLLECTION, {
      wait: true, // wait until Qdrant has stored them
      points: points.slice(i, i + BATCH),
    });
  }
}

/* ------------------------------------------------------------------ */
/* Find the k chunks of ONE pdf closest in meaning to the question     */
/* This replaces topK() from rag.js. Returns [{ text, page, score }]   */
/* ------------------------------------------------------------------ */
async function searchChunks(docId, questionVector, k = 4) {
  const res = await client().query(COLLECTION, {
    query: questionVector,
    filter: docFilter(docId),
    limit: k,
    with_payload: true,
  });
  return res.points.map((p) => ({
    text: p.payload.text,
    page: p.payload.page,
    score: p.score, // higher = closer in meaning (same idea as our cosine score)
  }));
}

/* ------------------------------------------------------------------ */
/* Has a PDF with this fingerprint already been uploaded?              */
/* Returns { docId, name, chunks } for the existing copy, or null.     */
/* ------------------------------------------------------------------ */
async function findDocumentByHash(fileHash) {
  const res = await client().scroll(COLLECTION, {
    filter: hashFilter(fileHash),
    limit: 1, // one matching chunk is enough to tell us which document it belongs to
    with_payload: ["docId", "docName"],
    with_vector: false,
  });
  if (res.points.length === 0) return null;

  const { docId, docName } = res.points[0].payload;
  const { count } = await client().count(COLLECTION, { filter: docFilter(docId), exact: true });
  return { docId, name: docName, chunks: count };
}

/* ------------------------------------------------------------------ */
/* Does this document exist?                                           */
/* ------------------------------------------------------------------ */
async function documentExists(docId) {
  const res = await client().count(COLLECTION, { filter: docFilter(docId), exact: true });
  return res.count > 0;
}

/* ------------------------------------------------------------------ */
/* List documents: read every chunk's small payload and group by docId */
/* (simple and fine for a few documents; a bigger app would keep a     */
/* separate list of documents)                                         */
/* ------------------------------------------------------------------ */
async function listDocuments() {
  const docs = new Map();
  let offset; // "scroll" returns the points page by page; offset says where to continue
  do {
    const res = await client().scroll(COLLECTION, {
      limit: 256,
      offset,
      with_payload: ["docId", "docName", "uploadedAt"], // skip the big text field
      with_vector: false, // skip the vectors, we don't need them here
    });
    for (const p of res.points) {
      const { docId, docName, uploadedAt } = p.payload;
      if (!docs.has(docId)) docs.set(docId, { docId, name: docName, chunks: 0, uploadedAt });
      docs.get(docId).chunks += 1;
    }
    offset = res.next_page_offset ?? undefined; // null when there are no more pages
  } while (offset !== undefined);

  // Oldest first, so the newest upload is last in the dropdown.
  return [...docs.values()]
    .sort((a, b) => a.uploadedAt - b.uploadedAt)
    .map(({ docId, name, chunks }) => ({ docId, name, chunks }));
}

/* ------------------------------------------------------------------ */
/* Delete one document = delete every point with that docId            */
/* ------------------------------------------------------------------ */
async function deleteDocument(docId) {
  await client().delete(COLLECTION, { wait: true, filter: docFilter(docId) });
}

module.exports = {
  client, // exported so the test script can clean up its own test collection
  ensureCollection,
  saveChunks,
  searchChunks,
  findDocumentByHash,
  documentExists,
  listDocuments,
  deleteDocument,
};
