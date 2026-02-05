from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import google.generativeai as genai
import uuid
import json

# Load environment variables
load_dotenv()

# Check environment variables
print("Environment variables loaded:")
print(f"GEMINI_API_KEY: {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
print(f"QDRANT_URL: {'SET' if os.getenv('QDRANT_URL') else 'NOT SET'}")
print(f"DATABASE_URL: {'SET' if os.getenv('DATABASE_URL') else 'NOT SET'}")

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini initialized successfully")
except Exception as e:
    print(f"Error initializing Gemini: {e}")
    gemini_model = None

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

# Initialize FastAPI
app = FastAPI(title="Physical AI ChatBot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://textbook-three.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Simple chat endpoint without RAG or database"""
    try:
        print(f"Received chat request: {request.question}")
        
        if not gemini_model:
            raise HTTPException(status_code=500, detail="Gemini model not initialized")
        
        # Build prompt without RAG context
        prompt = f"""You are an AI tutor specializing in Physical AI, Humanoid Robotics, ROS 2, and NVIDIA Isaac. 
        Answer the following question clearly and concisely:

        Question: {request.question}

        Provide a helpful, educational response. If the question is outside your domain, politely explain your expertise areas."""
        
        print("Generating response...")
        # Generate response
        response = gemini_model.generate_content(prompt)
        answer = response.text
        print(f"Generated response: {answer[:100]}...")
        
        return ChatResponse(answer=answer, sources=[])
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/selected", response_model=ChatResponse)
async def chat_selected(request: ChatRequest):
    """Chat endpoint for selected text"""
    try:
        print(f"Received selected chat request: {request.question}")
        
        if not gemini_model:
            raise HTTPException(status_code=500, detail="Gemini model not initialized")
        
        # Build prompt for selected text
        prompt = f"""You are an AI tutor specializing in Physical AI and Humanoid Robotics.
        Answer the following question based on the context of Physical AI and robotics.

        Question: {request.question}

        Provide a focused answer based on your knowledge of Physical AI and robotics."""
        
        # Generate response
        response = gemini_model.generate_content(prompt)
        answer = response.text
        
        return ChatResponse(answer=answer, sources=[], mode="selected_text")
        
    except Exception as e:
        print(f"Error in selected chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session (mock implementation)"""
    try:
        print(f"Getting history for session: {session_id}")
        # Return empty history for now
        return {"history": []}
    except Exception as e:
        print(f"Error in history endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
