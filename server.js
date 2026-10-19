// server.js — Express API:
//   POST   /upload          (multipart, field "pdf")  -> ingests a PDF into Qdrant
//   POST   /chat            (JSON {docId, question})  -> answers from that PDF
//   GET    /documents                                 -> lists uploaded PDFs
//   DELETE /documents/:docId                          -> deletes a PDF

require("dotenv").config();
const express = require("express");
const multer = require("multer");
const path = require("path");
const crypto = require("crypto");
const rag = require("./rag");
const qdrant = require("./qdrant");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

// Simple gate: API calls must send the right access code in a header.
// The code lives in an environment variable, never in the repo.
function requireAccessCode(req, res, next) {
  const expected = process.env.ACCESS_CODE;
  if (!expected) return next(); // no code set (local development): allow everything
  if (req.get("x-access-code") === expected) return next();
  res.status(401).json({ error: "Wrong or missing access code." });
}
app.use(["/upload", "/chat", "/documents"], requireAccessCode);

// Keep uploads in memory (we only need the text, not the file itself) and cap the size.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
});

/* ------------------------------------------------------------------ */
/* POST /upload                                                        */
/* ------------------------------------------------------------------ */
app.post("/upload", upload.single("pdf"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Attach a PDF in the 'pdf' field." });
    if (req.file.mimetype !== "application/pdf") {
      return res.status(400).json({ error: "Only PDF files are supported." });
    }

    // Fingerprint of the file's contents. The same PDF always gives the same fingerprint,
    // even if it is renamed, so we can recognise a repeat upload and skip the slow work.
    const fileHash = crypto.createHash("sha256").update(req.file.buffer).digest("hex");
    const existing = await qdrant.findDocumentByHash(fileHash);
    if (existing) {
      return res.json({ ...existing, duplicate: true });
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

    // Save the chunks and their vectors in Qdrant under a new document id.
    const docId = crypto.randomUUID();
    try {
      await qdrant.saveChunks(docId, req.file.originalname, embedded, fileHash);
    } catch (err) {
      // If saving stopped halfway, remove the partly saved document so it isn't left behind.
      await qdrant.deleteDocument(docId).catch(() => {});
      throw err;
    }

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
app.post("/chat", async (req, res) => {
  try {
    const { docId, question } = req.body || {};
    if (typeof docId !== "string" || !docId || typeof question !== "string" || !question.trim()) {
      return res.status(400).json({ error: "Send { docId, question }." });
    }
    if (question.length > 6000) {
      return res.status(400).json({ error: "Question is too long (max 6000 characters)." });
    }

    if (!(await qdrant.documentExists(docId))) {
      return res.status(404).json({ error: "Document not found. Upload a PDF first." });
    }

    const result = await rag.answerQuestion(question.trim(), docId, 4);
    res.json(result);
  } catch (err) {
    console.error("Chat failed:", err);
    res.status(500).json({ error: "Could not answer the question.", detail: err.message });
  }
});

/* ------------------------------------------------------------------ */
/* GET /documents and DELETE /documents/:docId                         */
/* ------------------------------------------------------------------ */
app.get("/documents", async (req, res) => {
  try {
    res.json(await qdrant.listDocuments());
  } catch (err) {
    console.error("Listing documents failed:", err);
    res.status(500).json({ error: "Could not list documents.", detail: err.message });
  }
});

app.delete("/documents/:docId", async (req, res) => {
  try {
    const { docId } = req.params;
    if (!(await qdrant.documentExists(docId))) {
      return res.status(404).json({ error: "Document not found." });
    }
    await qdrant.deleteDocument(docId);
    res.json({ ok: true });
  } catch (err) {
    console.error("Delete failed:", err);
    res.status(500).json({ error: "Could not delete the document.", detail: err.message });
  }
});

/* ------------------------------------------------------------------ */
/* Start: make sure the Qdrant collection exists, then listen          */
/* ------------------------------------------------------------------ */
qdrant
  .ensureCollection()
  .then(() => {
    app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
  })
  .catch((err) => {
    console.error("Could not connect to Qdrant:", err.message);
    console.error("Check QDRANT_URL and QDRANT_API_KEY in your .env file.");
    process.exit(1);
  });
