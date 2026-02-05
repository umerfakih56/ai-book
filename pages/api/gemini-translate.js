const { GoogleGenerativeAI } = require("@google/generative-ai");

// Simple in-memory cache for translations
const translationCache = new Map();

// Generate cache key from text
function getCacheKey(text) {
  // Create a simple hash from the text
  return text.toLowerCase().replace(/\s+/g, ' ').trim();
}

// Get cached translation
function getCachedTranslation(text) {
  const key = getCacheKey(text);
  return translationCache.get(key);
}

// Cache translation
function cacheTranslation(text, urduText) {
  const key = getCacheKey(text);
  translationCache.set(key, urduText);
  
  // Limit cache size to prevent memory issues
  if (translationCache.size > 1000) {
    const firstKey = translationCache.keys().next().value;
    translationCache.delete(firstKey);
  }
}

export default async function handler(req, res) {
  // Enable CORS for all requests
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { text } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    if (typeof text !== 'string') {
      return res.status(400).json({ error: "Text must be a string" });
    }

    // Check cache first
    const cachedTranslation = getCachedTranslation(text);
    if (cachedTranslation) {
      console.log('Returning cached translation');
      return res.status(200).json({ 
        translation: cachedTranslation,
        cached: true 
      });
    }

    // Check API key
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: "Gemini API key not configured" });
    }

    console.log('Translating text:', text.substring(0, 100) + '...');

    // Initialize Gemini API
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    
    // Translate with specific prompt
    const prompt = `Translate this to Urdu, provide only the Urdu text: ${text}`;
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const translation = response.text().trim();

    if (!translation) {
      return res.status(500).json({ error: "Translation failed - no response from API" });
    }

    // Cache the translation
    cacheTranslation(text, translation);

    console.log('Translation completed successfully');
    res.status(200).json({ 
      translation,
      cached: false 
    });

  } catch (error) {
    console.error("Translation error:", error);
    
    // Handle specific Gemini API errors
    if (error.message.includes('API_KEY')) {
      return res.status(500).json({ error: "Invalid API key" });
    }
    
    if (error.message.includes('quota')) {
      return res.status(429).json({ error: "API quota exceeded" });
    }

    res.status(500).json({ 
      error: "Translation failed",
      details: error.message 
    });
  }
}
