const fetch = globalThis.fetch || require('node-fetch');

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: 'Text is required' });

    const prompt = `Translate to Urdu: ${text}`;

    if (process.env.MODEL_PROVIDER && process.env.MODEL_PROVIDER.toLowerCase() === 'qwen') {
      const url = process.env.QWEN_API_URL;
      const key = process.env.QWEN_API_KEY;
      if (!url || !key) throw new Error('Qwen provider selected but QWEN_API_URL or QWEN_API_KEY not set');
      const body = { prompt, input: prompt };
      const resp = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` }, body: JSON.stringify(body) });
      const txt = await resp.text();
      let j;
      try { j = txt ? JSON.parse(txt) : null; } catch(e) { j = null; }
      if (!resp.ok) return res.status(500).json({ error: `Qwen API error: ${resp.status} ${txt}` });
      const candidate = (j && (j.translation || j.answer || j.output || j.result?.output_text || j.choices?.[0]?.text || j.data?.[0]?.text)) || txt;
      return res.status(200).json({ translation: typeof candidate === 'object' ? JSON.stringify(candidate) : candidate });
    }

    const { GoogleGenerativeAI } = require('@google/generative-ai');
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const translation = response.text();
    return res.status(200).json({ translation });
  } catch (error) {
    console.error('Translation error:', error);
    return res.status(500).json({ error: 'Translation failed' });
  }
}
