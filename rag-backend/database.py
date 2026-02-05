from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# SQLAlchemy setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ChatHistory model
class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sources = Column(JSON, nullable=True)

# Dependency to get database session
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to save chat
def save_chat(session_id: str, question: str, answer: str, sources: Optional[List[Dict[str, Any]]] = None) -> ChatHistory:
    """Save a chat exchange to the database"""
    db = SessionLocal()
    try:
        chat_history = ChatHistory(
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources
        )
        db.add(chat_history)
        db.commit()
        db.refresh(chat_history)
        return chat_history
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

# Function to get chat history
def get_history(session_id: str, limit: int = 10) -> List[ChatHistory]:
    """Get chat history for a session"""
    db = SessionLocal()
    try:
        history = db.query(ChatHistory)\
                   .filter(ChatHistory.session_id == session_id)\
                   .order_by(ChatHistory.created_at.desc())\
                   .limit(limit)\
                   .all()
        return history
    finally:
        db.close()

# Auto-create tables on startup
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

# Initialize database on import
create_tables()
