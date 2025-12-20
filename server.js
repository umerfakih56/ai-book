const express = require('express');
const path = require('path');
const { GoogleGenerativeAI } = require("@google/generative-ai");
require('dotenv').config({ path: '.env.local' });
const fs = require('fs');
const fetch = globalThis.fetch || require('node-fetch');

// Simple in-memory TF-IDF vector store for RAG
const CHUNK_SIZE = 800; // characters per chunk

function normalizeText(t) {
  return t.replace(/\r?\n/g, ' ').replace(/[^\w\s']/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
}

function tokenize(text) {
  return normalizeText(text).split(/\s+/).filter(Boolean);
}

const stopwords = new Set(['the','and','is','in','to','of','a','for','on','it','with','as','that','this','are','be','or','an','by','from','at','which']);

function buildVectorStore(docsPath) {
  const files = [];
  function walk(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (/\.mdx?$/.test(e.name)) files.push(p);
    }
  }

  if (!fs.existsSync(docsPath)) return { chunks: [], vocab: {}, idf: {} };
  walk(docsPath);

  const chunks = [];
  for (const f of files) {
    try {
      const raw = fs.readFileSync(f, 'utf8');
      let idx = 0;
      while (idx < raw.length) {
        const slice = raw.substr(idx, CHUNK_SIZE);
        chunks.push({ file: path.relative(process.cwd(), f), text: slice });
        idx += CHUNK_SIZE;
      }
    } catch (e) {
      console.error('Error reading file', f, e.message);
    }
  }

  // Build vocabulary and tf vectors
  const vocab = {};
  const docTermFreqs = [];
  for (const c of chunks) {
    const tokens = tokenize(c.text).filter(t => !stopwords.has(t));
    const freqs = {};
    for (const t of tokens) {
      freqs[t] = (freqs[t] || 0) + 1;
      vocab[t] = true;
    }
    docTermFreqs.push(freqs);
  }

  const vocabList = Object.keys(vocab);
  const df = {};
  for (const t of vocabList) df[t] = 0;
  for (const freqs of docTermFreqs) {
    for (const t in freqs) df[t]++;
  }

  const idf = {};
  const N = docTermFreqs.length || 1;
  for (const t of vocabList) idf[t] = Math.log((N + 1) / (df[t] + 1)) + 1;

  const tfidfVectors = docTermFreqs.map(freqs => {
    const vec = {};
    for (const t in freqs) vec[t] = freqs[t] * idf[t];
    // normalize
    const norm = Math.sqrt(Object.values(vec).reduce((s, v) => s + v * v, 0)) || 1;
    for (const k in vec) vec[k] = vec[k] / norm;
    return vec;
  });

  // attach vectors to chunks
  for (let i = 0; i < chunks.length; i++) chunks[i].vec = tfidfVectors[i];

  return { chunks, vocabList, idf };
}

function queryTopK(store, query, k = 3) {
  const qTokens = tokenize(query).filter(t => !stopwords.has(t));
  const qFreq = {};
  for (const t of qTokens) qFreq[t] = (qFreq[t] || 0) + 1;
  const qVec = {};
  for (const t in qFreq) {
    if (store.idf[t]) qVec[t] = qFreq[t] * store.idf[t];
  }
  // normalize
  const norm = Math.sqrt(Object.values(qVec).reduce((s, v) => s + v * v, 0)) || 1;
  for (const k2 in qVec) qVec[k2] = qVec[k2] / norm;

  function cosine(a, b) {
    let s = 0;
    for (const k in a) if (b[k]) s += a[k] * b[k];
    return s;
  }

  const scored = store.chunks.map(c => ({ score: cosine(qVec, c.vec || {}), chunk: c }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).filter(s => s.score > 0.01).map(s => s.chunk);
}

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static('build'));

// Build the vector store at startup from the `docs/` directory
const STORE = buildVectorStore(path.join(__dirname, 'docs'));
console.log('RAG store initialized with', STORE.chunks ? STORE.chunks.length : 0, 'chunks');

// RAG chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) return res.status(400).json({ error: 'question is required' });

    // Find top relevant chunks
    const top = queryTopK(STORE, question, 4);
    const contextText = top.map(t => `Source: ${t.file}\n\n${t.text}`).join('\n\n---\n\n');

    const prompt = `You are an assistant answering user questions using the provided context. Use only the information in the context; if the answer is not present, say you don't know.\n\nContext:\n${contextText}\n\nUser question: ${question}\n\nAnswer concisely and, at the end, list the source file names you used.`;

    // Choose model provider: 'qwen' if MODEL_PROVIDER=qwen, otherwise Gemini
    async function generateWithModel(p) {
      if (process.env.MODEL_PROVIDER && process.env.MODEL_PROVIDER.toLowerCase() === 'qwen') {
        // Qwen requires QWEN_API_URL and QWEN_API_KEY in env
        const url = process.env.QWEN_API_URL;
        const key = process.env.QWEN_API_KEY;
        if (!url || !key) throw new Error('Qwen provider selected but QWEN_API_URL or QWEN_API_KEY not set');
        // Send both `prompt` and `input` to cover different Qwen endpoints
        const body = { prompt: p, input: p };
        const resp = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${key}`,
          },
          body: JSON.stringify(body),
        });
        const text = await resp.text();
        let j;
        try { j = text ? JSON.parse(text) : null; } catch(e) { j = null; }
        if (!resp.ok) {
          throw new Error(`Qwen API error: ${resp.status} ${text}`);
        }
        // Try to extract text from common response shapes
        const candidate = (j && (j.translation || j.answer || j.output || j.result?.output_text || j.choices?.[0]?.text || j.choices?.[0]?.content || j.data?.[0]?.text || j.data?.[0]?.content)) || text;
        if (typeof candidate === 'object') return JSON.stringify(candidate);
        return candidate || '';
      } else {
        const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
        const result = await model.generateContent(p);
        const response = await result.response;
        return response.text();
      }
    }

    const answer = await generateWithModel(prompt);

    const sources = top.map(t => t.file);
    res.status(200).json({ answer, sources });
  } catch (err) {
    console.error('Chat error', err);
    res.status(500).json({ error: 'Chat failed' });
  }
});

// API Routes
app.post('/api/translate', async (req, res) => {
  console.log('Translation request received');
  
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    console.log('Translating text:', text.substring(0, 100) + '...');

    const prompt = `Translate to Urdu: ${text}`;

    // Reuse provider selection logic for translation too
    async function generateText(p) {
      if (process.env.MODEL_PROVIDER && process.env.MODEL_PROVIDER.toLowerCase() === 'qwen') {
        const url = process.env.QWEN_API_URL;
        const key = process.env.QWEN_API_KEY;
        if (!url || !key) throw new Error('Qwen provider selected but QWEN_API_URL or QWEN_API_KEY not set');
        const body = { prompt: p, input: p };
        const resp = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
          body: JSON.stringify(body),
        });
        const text = await resp.text();
        let j;
        try { j = text ? JSON.parse(text) : null; } catch(e) { j = null; }
        if (!resp.ok) throw new Error(`Qwen API error: ${resp.status} ${text}`);
        const candidate = (j && (j.translation || j.answer || j.output || j.result?.output_text || j.choices?.[0]?.text || j.choices?.[0]?.content || j.data?.[0]?.text || j.data?.[0]?.content)) || text;
        if (typeof candidate === 'object') return JSON.stringify(candidate);
        return candidate || '';
      } else {
        const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        const result = await model.generateContent(p);
        const response = await result.response;
        return response.text();
      }
    }

    const translation = await generateText(prompt);

    console.log('Translation completed');
    res.status(200).json({ translation });
  } catch (error) {
    console.error("Translation error:", error);
    res.status(500).json({ error: "Translation failed" });
  }
});

// Serve Docusaurus static files
app.use(express.static(path.join(__dirname, 'build')));

// Handle client-side routing (fallback for SPA)
app.use((req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
