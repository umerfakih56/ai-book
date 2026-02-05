from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import google.generativeai as genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid
import json
from database import save_chat, get_history, ChatHistory as DBChatHistory

# Load environment variables
load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Qdrant
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'humanoid_robotics_book')
if qdrant_url and qdrant_api_key:
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
else:
    qdrant_client = QdrantClient(host='localhost', port=6333)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    session_id: str

class SelectedTextRequest(BaseModel):
    question: str
    selected_text: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = None
    mode: str = None

class ChatHistory(BaseModel):
    id: int
    question: str
    answer: str
    session_id: str
    created_at: datetime
    sources: Optional[List[dict]] = None

# Create FastAPI app
app = FastAPI(title="RAG Backend", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://textbook-three.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG Backend API is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG retrieval"""
    try:
        # Generate embedding for the question
        question_embedding = embedding_model.encode(request.question)
        
        # Search Qdrant for relevant chunks
        search_result = qdrant_client.search(
            collection_name=qdrant_collection_name,
            query_vector=question_embedding.tolist(),
            limit=4,
            with_payload=True
        )
        
        # Extract chunks and build context
        chunks = []
        sources = []
        for hit in search_result:
            payload = hit.payload
            chunks.append(payload['chunk_text'])
            sources.append({
                "title": payload['title'],
                "file": payload['file_path']
            })
        
        # Build prompt
        context = "\n\n".join(chunks)
        prompt = f"""You are an AI tutor for Physical AI & Humanoid Robotics. Use this context to answer the question accurately and helpfully.

Context:
{context}

Question: {request.question}

Answer:"""
        
        # Generate response with Gemini
        response = gemini_model.generate_content(prompt)
        answer = response.text
        
        # Save to database using SQLAlchemy
        save_chat(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            sources=sources
        )
        
        return ChatResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/selected", response_model=ChatResponse)
async def chat_selected(request: SelectedTextRequest):
    """Chat endpoint using only selected text"""
    try:
        # Build prompt with only selected text
        prompt = f"""Based ONLY on the provided text, answer the question accurately. Do not use any external knowledge.

Text:
{request.selected_text}

Question: {request.question}

Answer:"""
        
        # Generate response with Gemini
        response = gemini_model.generate_content(prompt)
        answer = response.text
        
        # Save to database using SQLAlchemy
        save_chat(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            sources=None
        )
        
        return ChatResponse(answer=answer, mode="selected_text")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{session_id}", response_model=List[ChatHistory])
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = get_history(session_id=session_id, limit=10)
        
        # Convert DB models to Pydantic models
        return [
            ChatHistory(
                id=item.id,
                question=item.question,
                answer=item.answer,
                session_id=item.session_id,
                created_at=item.created_at,
                sources=item.sources
            )
            for item in history
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
