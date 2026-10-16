
// server.js — Express API with two endpoints:
//   POST /upload  (multipart, field "pdf")   -> ingests a PDF
//   POST /chat    (JSON {docId, question})   -> answers from that PDF

require("dotenv").config();
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { rateLimit } = require("express-rate-limit");
const rag = require("./rag");

const app = express();
const PORT = process.env.PORT || 3000;

// Behind Render's proxy, trust the first proxy so rate limits see each visitor's real IP.
app.set("trust proxy", 1);

// Protect your Gemini quota: each IP gets a limited number of uploads and questions.
const uploadLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  limit: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many uploads. Please try again in an hour." },
});
const chatLimiter = rateLimit({
  windowMs: 10 * 60 * 1000, // 10 minutes
  limit: 40,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many questions. Please wait a few minutes and try again." },
});

// Keep memory and disk bounded: when this many documents are stored, the oldest is dropped.
const MAX_DOCS = 30;

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public"))); // frontend goes here on Day 2

// Keep uploads in memory (we only need the text, not the file itself) and cap the size.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
});

/* ------------------------------------------------------------------ */
/* The "database": an in-memory Map, saved to a JSON file so it        */
/* survives restarts. Each document = { name, chunks: [{text,page,embedding}] } */
/* ------------------------------------------------------------------ */
const DATA_DIR = path.join(__dirname, "data");
const STORE_FILE = path.join(DATA_DIR, "store.json");
const documents = new Map();

function loadStore() {
  try {
    const raw = JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));
    for (const [id, doc] of Object.entries(raw)) documents.set(id, doc);
    console.log(`Loaded ${documents.size} document(s) from disk`);
  } catch {
    // no saved store yet — that's fine
  }
}

function saveStore() {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.writeFileSync(STORE_FILE, JSON.stringify(Object.fromEntries(documents)));
}

loadStore();

/* ------------------------------------------------------------------ */
/* POST /upload                                                        */
/* ------------------------------------------------------------------ */
app.post("/upload", uploadLimiter, upload.single("pdf"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Attach a PDF in the 'pdf' field." });
    if (req.file.mimetype !== "application/pdf") {
      return res.status(400).json({ error: "Only PDF files are supported." });
    }

    const pages = await rag.extractPages(req.file.buffer);
    const totalChars = pages.reduce((n, p) => n + p.text.trim().length, 0);
    if (totalChars < 50) {
      return res.status(400).json({
        error: "No readable text found. This may be a scanned PDF (images only).",
      });
    }

    const chunks = rag.chunkPages(pages);
    const embedded = await rag.embedChunks(chunks);

    const docId = crypto.randomUUID();
    documents.set(docId, { name: req.file.originalname, chunks: embedded });
    while (documents.size > MAX_DOCS) {
      documents.delete(documents.keys().next().value); // Map keeps insertion order: first = oldest
    }
    saveStore();

    res.json({
      docId,
      name: req.file.originalname,
      pages: pages.length,
      chunks: embedded.length,
    });
  } catch (err) {
    console.error("Upload failed:", err);
    res.status(500).json({ error: "Could not process the PDF.", detail: err.message });
  }
});

/* ------------------------------------------------------------------ */
/* POST /chat                                                          */
/* ------------------------------------------------------------------ */
app.post("/chat", chatLimiter, async (req, res) => {
  try {
    const { docId, question } = req.body || {};
    if (!docId || typeof question !== "string" || !question.trim()) {
      return res.status(400).json({ error: "Send { docId, question }." });
    }
    // Resume mode sends a longer prompt (instructions + pasted job description).
    if (question.length > 6000) {
      return res.status(400).json({ error: "Question is too long (max 6000 characters)." });
    }

    const doc = documents.get(docId);
    if (!doc) return res.status(404).json({ error: "Document not found. Upload a PDF first." });

    const result = await rag.answerQuestion(question.trim(), doc.chunks, 4);
    res.json(result);
  } catch (err) {
    console.error("Chat failed:", err);
    res.status(500).json({ error: "Could not answer the question.", detail: err.message });
  }
});

// Look up documents by id. A browser only asks for the ids it uploaded itself, so
// visitors never see each other's files. Without ?ids= this returns nothing.
app.get("/documents", (req, res) => {
  const ids = String(req.query.ids || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 50);
  res.json(
    ids
      .filter((id) => documents.has(id))
      .map((id) => ({
        docId: id,
        name: documents.get(id).name,
        chunks: documents.get(id).chunks.length,
      }))
  );
});

app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));