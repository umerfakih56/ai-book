const express = require('express');
const cors = require('cors');
require('dotenv').config({ path: '.env.local' });

const app = express();
const PORT = 8000;

app.use(cors());
app.use(express.json());

// Simple test endpoint
app.post('/api/test', async (req, res) => {
  try {
    const { question } = req.body;
    console.log('=== TEST ENDPOINT CALLED ===');
    console.log('Question:', question);
    
    // Simple keyword response
    let answer = "I don't understand that question.";
    if (question.toLowerCase().includes('ros')) {
      answer = "ROS is Robot Operating System - a framework for robot software development.";
    }
    
    console.log('Returning answer:', answer);
    res.json({ answer });
  } catch (error) {
    console.error('Test error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Test API server running on http://localhost:${PORT}`);
});
