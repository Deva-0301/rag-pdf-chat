// rag.js — the whole RAG pipeline, written by hand so every step is visible.
//
//   PDF  ->  pages of text  ->  chunks  ->  embeddings  ->  stored in Qdrant
//   question -> embedding -> Qdrant finds the closest chunks -> prompt -> Gemini -> answer + sources

require("dotenv").config();
const { GoogleGenAI } = require("@google/genai");
const { PDFParse } = require("pdf-parse");
const qdrant = require("./qdrant"); // the vector database (see qdrant.js)

const EMBED_MODEL = process.env.GEMINI_EMBED_MODEL || "gemini-embedding-001";
const CHAT_MODEL = process.env.GEMINI_CHAT_MODEL || "gemini-3.6-flash";

// The exact sentence the model must use when the document has no answer.
const NOT_FOUND = "I couldn't find that in the document.";

// Created lazily so the offline tests work without an API key.
let _ai = null;
function ai() {
  if (!_ai) {
    if (!process.env.GEMINI_API_KEY) {
      throw new Error("GEMINI_API_KEY is missing. Add it to your .env file.");
    }
    _ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  }
  return _ai;
}

/* ------------------------------------------------------------------ */
/* STEP 1: extract text from the PDF, one entry per page               */
/* ------------------------------------------------------------------ */
async function extractPages(pdfBuffer) {
  const parser = new PDFParse({ data: pdfBuffer });
  try {
    const result = await parser.getText();
    // result.pages = [{ num: 1, text: "..." }, ...]
    // Keeping the page number lets us tell the user "this answer came from page 14".
    return result.pages.map((p) => ({ page: p.num, text: p.text }));
  } finally {
    await parser.destroy(); // free memory used by the PDF engine
  }
}

/* ------------------------------------------------------------------ */
/* STEP 2: chunking — split text into overlapping pieces               */
/* ------------------------------------------------------------------ */
// size    = max characters per chunk
// overlap = how many characters of the previous chunk are repeated at the start
//           of the next one, so a sentence cut at the boundary is not lost.
function chunkText(text, size = 800, overlap = 150) {
  if (overlap >= size) throw new Error("overlap must be smaller than size");
  const clean = text.replace(/\s+/g, " ").trim();
  const chunks = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(start + size, clean.length);

    // Try to end on a sentence boundary so chunks read naturally.
    if (end < clean.length) {
      const lastStop = Math.max(
        clean.lastIndexOf(". ", end),
        clean.lastIndexOf("? ", end),
        clean.lastIndexOf("! ", end)
      );
      if (lastStop > start + size * 0.5) end = lastStop + 1;
    }

    const piece = clean.slice(start, end).trim();
    if (piece.length > 0) chunks.push(piece);

    if (end >= clean.length) break;
    start = end - overlap; // step back by `overlap` characters
  }
  return chunks;
}

// Chunk page by page so every chunk knows which page it came from.
function chunkPages(pages, size = 800, overlap = 150) {
  const out = [];
  for (const { page, text } of pages) {
    for (const piece of chunkText(text, size, overlap)) {
      if (piece.length < 10) continue; // skip page numbers and other noise
      out.push({ text: piece, page });
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* STEP 3: embeddings — turn text into a list of numbers               */
/* ------------------------------------------------------------------ */
async function embedTexts(texts, taskType) {
  const vectors = [];
  const BATCH = 50; // send a limited number of texts per API call
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH);
    const res = await ai().models.embedContent({
      model: EMBED_MODEL,
      contents: batch,
      config: { taskType },
    });
    for (const e of res.embeddings) vectors.push(e.values);
  }
  return vectors;
}

// Embed document chunks and attach the vector to each chunk.
async function embedChunks(chunks) {
  const vectors = await embedTexts(
    chunks.map((c) => c.text),
    "RETRIEVAL_DOCUMENT"
  );
  return chunks.map((c, i) => ({ ...c, embedding: vectors[i] }));
}

async function embedQuestion(question) {
  const [v] = await embedTexts([question], "RETRIEVAL_QUERY");
  return v;
}

/* ------------------------------------------------------------------ */
/* STEP 4: cosine similarity + retrieval                               */
/* ------------------------------------------------------------------ */
// NOTE: the app now lets Qdrant do the searching (see answerQuestion below).
// cosineSimilarity, topK and retrieve are the hand-written version I built first
// to understand how retrieval works. They are kept here as the reference
// ("this is what Qdrant does for us"), and they are still unit-tested.
//
// Cosine similarity = how closely two vectors point in the same direction.
// 1 = same meaning, around 0 = unrelated.
function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Score every chunk against the question vector and keep the best k.
function topK(questionVector, chunks, k = 4) {
  return chunks
    .map((c) => ({ ...c, score: cosineSimilarity(questionVector, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

async function retrieve(question, chunks, k = 4) {
  const qVec = await embedQuestion(question);
  return topK(qVec, chunks, k);
}

/* ------------------------------------------------------------------ */
/* STEP 5: build the prompt and ask Gemini                             */
/* ------------------------------------------------------------------ */
function buildPrompt(question, sources) {
  const context = sources
    .map((s, i) => `[Source ${i + 1}, page ${s.page}]\n${s.text}`)
    .join("\n\n");

  return `You are a helpful assistant that answers questions about a document.

Rules:
- Answer using ONLY the context below.
- If the answer is not in the context, reply exactly: "${NOT_FOUND}"
- Do not use outside knowledge. Do not guess.
- Mention which source numbers you used, like (Source 2).

Context:
${context}

Question: ${question}`;
}

// docId says WHICH pdf to search. Qdrant returns that pdf's k closest chunks.
async function answerQuestion(question, docId, k = 4) {
  
  const questionVector = await embedQuestion(question);
  const sources = await qdrant.searchChunks(docId, questionVector, k);
  const prompt = buildPrompt(question, sources);

  const res = await ai().models.generateContent({
    model: CHAT_MODEL,
    contents: prompt,
  });

  // The model sometimes writes a curly apostrophe; make it a plain one before comparing.
  const answer = (res.text || "").trim().replace(/’/g, "'");
  const found = !answer.startsWith(NOT_FOUND);

  return {
    answer,
    found,
    // Only show sources when the document actually contained the answer.
    sources: found
      ? sources.map((s) => ({
          page: s.page,
          score: Number(s.score.toFixed(3)),
          text: s.text,
        }))
      : [],
  };
}

// True when Gemini says "too many requests" or the free quota is used up (HTTP 429).
// We check the status code first, then the message text, because error shapes can vary.
function isRateLimitError(err) {
  const text = String((err && err.message) || "");
  return (err && err.status === 429) || /429|RESOURCE_EXHAUSTED|quota/i.test(text);
}

// True when Gemini says it is temporarily overloaded (HTTP 503 / UNAVAILABLE).
function isOverloadedError(err) {
  const text = String((err && err.message) || "");
  return (err && err.status === 503) || /503|UNAVAILABLE|high demand/i.test(text);
}

module.exports = {
  extractPages,
  chunkText,
  chunkPages,
  embedTexts,
  embedChunks,
  embedQuestion,
  cosineSimilarity,
  topK,
  retrieve,
  buildPrompt,
  answerQuestion,
  isRateLimitError,
  isOverloadedError,
};
