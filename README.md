# RAG PDF Chat

Upload a PDF and ask questions about it. Answers come only from the document, with the source passages and page numbers shown.

Built from scratch with Node.js, Express and the Gemini API. The RAG pipeline (chunking, embeddings, cosine-similarity retrieval) is written by hand, with no vector database or LangChain.

## How it works

1. **Extract** text from each PDF page (`pdf-parse`)
2. **Chunk** the text into ~800-character pieces with 150 characters of overlap
3. **Embed** every chunk with a Gemini embedding model and store the vectors
4. On a question: **embed the question**, rank all chunks by **cosine similarity**, keep the top 4
5. **Prompt** Gemini to answer using only those chunks, or say it can't find the answer
6. Return the answer plus the source chunks and page numbers

## Run locally

```bash
npm install
cp .env.example .env        # then put your Gemini API key in .env
npm start
```

Test with curl:

```bash
# 1. upload a PDF -> returns a docId
curl -F "pdf=@notes.pdf;type=application/pdf" localhost:3000/upload

# 2. ask a question
curl -X POST localhost:3000/chat \
  -H "content-type: application/json" \
  -d '{"docId":"PASTE_DOC_ID_HERE","question":"What is a deadlock?"}'
```

Offline tests (no API key needed): `npm run test:offline`

## API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /upload` | multipart, field `pdf` (max 10 MB) | `{ docId, name, pages, chunks }` |
| `POST /chat` | `{ docId, question }` | `{ answer, sources: [{ page, score, text }] }` |
| `GET /documents` | none | list of uploaded documents |

## Known limitations

- Vectors are kept in memory and saved to `data/store.json`, which is fine for a handful of documents but not for scale. Next step: MongoDB Atlas Vector Search or Pinecone.
- No user accounts yet, so all documents are visible to everyone using the server.
- Scanned (image-only) PDFs are not supported because there is no OCR.
