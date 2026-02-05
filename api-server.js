const express = require('express');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config({ path: '.env.local' });

const app = express();
const PORT = 8000;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage for chat history (in production, use a database)
const chatHistory = {};

// Initialize Gemini AI
console.log('Checking API key:', process.env.GEMINI_API_KEY ? 'API key found' : 'API key not found');
console.log('API key length:', process.env.GEMINI_API_KEY?.length || 0);

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Try different model names with safety settings (prefer lighter model first to avoid rate limits)
const modelNames = ['gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-3-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-1.0-pro'];
let model;

for (const modelName of modelNames) {
  try {
    model = genAI.getGenerativeModel({ 
      model: modelName,
      safetySettings: [
        {
          category: "HARM_CATEGORY_HARASSMENT",
          threshold: "BLOCK_NONE"
        },
        {
          category: "HARM_CATEGORY_HATE_SPEECH", 
          threshold: "BLOCK_NONE"
        },
        {
          category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          threshold: "BLOCK_NONE"
        },
        {
          category: "HARM_CATEGORY_DANGEROUS_CONTENT",
          threshold: "BLOCK_NONE"
        }
      ]
    });
    console.log(`Successfully loaded model: ${modelName}`);
    break;
  } catch (error) {
    console.log(`Failed to load model ${modelName}:`, error.message);
  }
}

if (!model) {
  console.error('Could not load any Gemini model');
  process.exit(1);
}

// Helper: sleep
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Helper: call Gemini with retry/backoff to handle rate limits
async function callGeminiWithRetry(promptText, maxAttempts = 3) {
  let attempt = 0;
  let lastError;
  while (attempt < maxAttempts) {
    try {
      const result = await model.generateContent(promptText);
      const response = await result.response;
      return response.text();
    } catch (error) {
      lastError = error;
      const status = error?.status;
      const isRateLimited = status === 429 || status === 403; // some quotas surface as 403
      const backoffMs = Math.min(1000 * Math.pow(2, attempt), 8000);
      console.warn(`Gemini call failed (status: ${status}). Attempt ${attempt + 1}/${maxAttempts}. ${isRateLimited ? 'Backing off ' + backoffMs + 'ms' : 'No backoff'}`);
      if (!isRateLimited) break;
      await sleep(backoffMs);
    }
    attempt += 1;
  }
  throw lastError || new Error('Unknown error calling Gemini');
}

// Helper function to read all markdown files
async function getAllMarkdownFiles(dir) {
  const files = [];
  
  async function traverse(currentDir) {
    const items = await fs.readdir(currentDir, { withFileTypes: true });
    
    for (const item of items) {
      const fullPath = path.join(currentDir, item.name);
      
      if (item.isDirectory()) {
        await traverse(fullPath);
      } else if (item.name.endsWith('.md')) {
        try {
          const content = await fs.readFile(fullPath, 'utf-8');
          const relativePath = path.relative(process.cwd(), fullPath);
          
          // Extract title from front matter or filename
          let title = item.name.replace('.md', '');
          const frontMatterMatch = content.match(/^---\s*\ntitle:\s*["']?([^"'\n]+)["']?\n/);
          if (frontMatterMatch) {
            title = frontMatterMatch[1];
          }
          
          files.push({
            path: relativePath,
            title: title,
            content: content
          });
        } catch (error) {
          console.error(`Error reading file ${fullPath}:`, error);
        }
      }
    }
  }
  
  await traverse(dir);
  return files;
}

// Generate response using Gemini AI with context from documentation
async function generateResponse(question, docs, selectedText = null) {
  console.log('=== generateResponse called ===');
  console.log('Question:', question);
  console.log('Docs count:', docs?.length || 0);
  console.log('Selected text:', selectedText);
  
  // Try multiple approaches in order
  const approaches = [
    {
      name: 'Gemini API',
      fn: async () => {
        console.log('--- Trying Gemini API ---');
        const text = await callGeminiWithRetry(prompt);
        console.log('Gemini response received (truncated):', (text || '').slice(0, 120));
        return text;
      }
    },
    {
      name: 'OpenAI (if configured)',
      fn: async () => {
        console.log('--- Trying OpenAI ---');
        // Placeholder for OpenAI integration
        return null;
      }
    },
    {
      name: 'Local LLM (if configured)',
      fn: async () => {
        console.log('--- Trying Local LLM ---');
        // Placeholder for local LLM integration
        return null;
      }
    },
    {
      name: 'Simple keyword matching',
      fn: async () => {
        console.log('--- Trying keyword matching ---');
        const lowerQuestion = question.toLowerCase();
        
        if (lowerQuestion.includes('ros') || lowerQuestion.includes('robot operating system')) {
          return {
            answer: "ROS (Robot Operating System) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms. ROS 2 is the latest version with improved security, real-time capabilities, and support for multiple operating systems.",
            sources: docs.filter(doc => 
              doc.title.toLowerCase().includes('ros') || 
              doc.content.toLowerCase().includes('robot operating system')
            ).map(doc => ({ title: doc.title, file: doc.path }))
          };
        }
        
        if (lowerQuestion.includes('gazebo') || lowerQuestion.includes('simulation')) {
          return {
            answer: "Gazebo is a 3D simulation environment for robotics. It provides realistic physics simulation, sensor modeling, and rendering capabilities. Gazebo is commonly used with ROS for testing robot algorithms in simulation before deploying them on real hardware, saving time and reducing risks.",
            sources: docs.filter(doc => 
              doc.title.toLowerCase().includes('gazebo') || 
              doc.content.toLowerCase().includes('simulation')
            ).map(doc => ({ title: doc.title, file: doc.path }))
          };
        }
        
        if (lowerQuestion.includes('nvidia') || lowerQuestion.includes('isaac')) {
          return {
            answer: "NVIDIA Isaac Sim is a robotics simulation platform built on Omniverse. It provides photorealistic rendering, physics simulation, and AI integration capabilities. Isaac Sim is particularly useful for developing and testing AI-powered robots, including humanoid robots with advanced perception and manipulation capabilities.",
            sources: docs.filter(doc => 
              doc.title.toLowerCase().includes('isaac') || 
              doc.content.toLowerCase().includes('nvidia')
            ).map(doc => ({ title: doc.title, file: doc.path }))
          };
        }
        
        if (lowerQuestion.includes('humanoid') || lowerQuestion.includes('kinematics')) {
          return {
            answer: "Humanoid robotics involves creating robots with human-like body structures. Key challenges include bipedal locomotion, balance control, human-robot interaction, and complex kinematics. Modern humanoid robots use advanced sensors, AI algorithms, and sophisticated control systems to navigate and interact with human environments.",
            sources: docs.filter(doc => 
              doc.title.toLowerCase().includes('humanoid') || 
              doc.content.toLowerCase().includes('kinematics')
            ).map(doc => ({ title: doc.title, file: doc.path }))
          };
        }
        
        if (lowerQuestion.includes('vla') || lowerQuestion.includes('vision language action')) {
          return {
            answer: "Vision-Language-Action (VLA) models are AI systems that can process visual information, understand natural language, and generate appropriate actions. These models enable robots to understand complex commands, perceive their environment, and execute tasks based on multimodal inputs. VLA represents a cutting-edge approach to creating more intelligent and capable robots.",
            sources: docs.filter(doc => 
              doc.title.toLowerCase().includes('vla') || 
              doc.content.toLowerCase().includes('vision language action')
            ).map(doc => ({ title: doc.title, file: doc.path }))
          };
        }
        
        // Default response
        return {
          answer: "I'm here to help you learn about Physical AI and Humanoid Robotics! You can ask me about:\n\n• ROS 2 (Robot Operating System)\n• Gazebo simulation\n• NVIDIA Isaac Sim\n• Humanoid robot kinematics\n• Vision-Language-Action (VLA) models\n• Robot perception and navigation\n• AI integration in robotics\n\nWhat specific topic would you like to explore?",
          sources: []
        };
      }
    }
  ];
  
  // Try each approach until one succeeds
  for (const approach of approaches) {
    try {
      console.log(`Trying approach: ${approach.name}`);
      
      // Prepare context from documentation (trimmed for TPM)
      let context = "";
      if (docs && docs.length > 0) {
        const relevantDocs = docs.slice(0, 3);
        context = "\n\nRelevant Documentation:\n" + 
          relevantDocs.map(doc => `--- ${doc.title} ---\n${doc.content.substring(0, 1200)}...`).join('\n\n');
      }

      // Compose prompt
      const prompt = `You are an expert AI tutor specializing in Physical AI and Humanoid Robotics. Answer clearly and concisely.\n\nContext from course materials:${context}\n\nUser Question: ${question}\n${selectedText ? `\nSelected Text: "${selectedText}"\nFocus your answer on the selected text if relevant.` : ''}`;
      
      console.log('Prompt prepared, calling approach function...');
      const result = await approach.fn();
      console.log('Approach result:', result ? (typeof result === 'string' ? 'string length ' + result.length : 'object') : 'null/undefined');
      
      if (result && typeof result === 'string' && result.length > 10) {
        console.log(`✅ ${approach.name} succeeded!`);
        return {
          answer: result,
          sources: docs.slice(0, 3).map(doc => ({ title: doc.title, file: doc.path })),
          mode: selectedText ? 'selected_text' : 'general'
        };
      }
    } catch (error) {
      console.warn(`❌ ${approach.name} failed:`, error.message);
      console.warn('Error stack:', error.stack);
      continue; // Try next approach
    }
  }
  
  // If all approaches fail, return final fallback
  console.log('=== All approaches failed, returning fallback ===');
  return {
    answer: "I'm having trouble connecting to my AI brain right now. Please try again in a moment. In the meantime, you can explore the course materials or ask me about:\n\n• ROS 2 fundamentals\n• Gazebo simulation\n• NVIDIA Isaac Sim\n• Humanoid robot kinematics\n• Vision-Language-Action models",
    sources: [],
    mode: selectedText ? 'selected_text' : 'general'
  };
}

// Routes
app.post('/api/chat', async (req, res) => {
  try {
    const { question, session_id } = req.body;
    
    if (!question || !session_id) {
      return res.status(400).json({ error: 'Question and session_id are required' });
    }
    
    // Get all documentation files
    const docsDir = path.join(process.cwd(), 'docs');
    const docs = await getAllMarkdownFiles(docsDir);
    
    // Generate response
    const response = await generateResponse(question, docs);
    
    // Store in chat history
    if (!chatHistory[session_id]) {
      chatHistory[session_id] = [];
    }
    
    chatHistory[session_id].push({
      question,
      answer: response.answer,
      sources: response.sources,
      created_at: new Date().toISOString()
    });
    
    res.json(response);
  } catch (error) {
    console.error('Chat API error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/chat/selected', async (req, res) => {
  try {
    const { question, selected_text, session_id } = req.body;
    
    if (!question || !selected_text || !session_id) {
      return res.status(400).json({ error: 'Question, selected_text, and session_id are required' });
    }
    
    // Get all documentation files
    const docsDir = path.join(process.cwd(), 'docs');
    const docs = await getAllMarkdownFiles(docsDir);
    
    // Generate response based on selected text
    const response = await generateResponse(question, docs, selected_text);
    
    // Store in chat history
    if (!chatHistory[session_id]) {
      chatHistory[session_id] = [];
    }
    
    chatHistory[session_id].push({
      question,
      answer: response.answer,
      sources: response.sources,
      created_at: new Date().toISOString(),
      selected_text
    });
    
    res.json(response);
  } catch (error) {
    console.error('Chat API error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/history/:session_id', (req, res) => {
  try {
    const { session_id } = req.params;
    const history = chatHistory[session_id] || [];
    res.json(history);
  } catch (error) {
    console.error('History API error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.delete('/api/history/:session_id', (req, res) => {
  try {
    const { session_id } = req.params;
    delete chatHistory[session_id];
    res.json({ success: true });
  } catch (error) {
    console.error('Delete history API error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Chat API server running on http://localhost:${PORT}`);
  console.log(`📚 Documentation will be loaded from: ${path.join(process.cwd(), 'docs')}`);
});
