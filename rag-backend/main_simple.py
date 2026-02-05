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
from database import save_chat, get_history, ChatHistory as DBChatHistory

# Load environment variables
load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

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
    """Simple chat endpoint without RAG (for testing)"""
    try:
        # Build prompt without RAG context
        prompt = f"""You are an AI tutor specializing in Physical AI, Humanoid Robotics, ROS 2, and NVIDIA Isaac. 
        Answer the following question clearly and concisely:

        Question: {request.question}

        Provide a helpful, educational response. If the question is outside your domain, politely explain your expertise areas."""
        
        # Generate response
        response = gemini_model.generate_content(prompt)
        answer = response.text
        
        # Save to database
        try:
            save_chat(
                session_id=request.session_id,
                question=request.question,
                answer=answer,
                sources=[]
            )
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Continue even if database fails
        
        return ChatResponse(answer=answer, sources=[])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/selected", response_model=ChatResponse)
async def chat_selected(request: ChatRequest):
    """Chat endpoint for selected text"""
    try:
        # Build prompt for selected text
        prompt = f"""You are an AI tutor specializing in Physical AI and Humanoid Robotics.
        Answer the following question based ONLY on the context provided. If the context doesn't contain enough information, say so politely.

        Question: {request.question}

        Provide a focused answer based on the selected text context."""
        
        # Generate response
        response = gemini_model.generate_content(prompt)
        answer = response.text
        
        # Save to database
        try:
            save_chat(
                session_id=request.session_id,
                question=request.question,
                answer=answer,
                sources=[],
                mode="selected_text"
            )
        except Exception as db_error:
            print(f"Database error: {db_error}")
        
        return ChatResponse(answer=answer, sources=[], mode="selected_text")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = get_history(session_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
