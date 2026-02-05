import os
import re
from pathlib import Path
from typing import List, Dict, Any
import yaml
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

class DocumentIngestor:
    def __init__(self):
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Qdrant client
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if qdrant_url and qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Fallback to local instance
            self.client = QdrantClient(host='localhost', port=6333)
        
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'humanoid_robotics_book')
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections
            )
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection '{self.collection_name}'")
            else:
                print(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract frontmatter from markdown content"""
        frontmatter = {}
        content_without_frontmatter = content
        
        # Check if content starts with frontmatter
        if content.startswith('---'):
            try:
                # Find the end of frontmatter
                end_index = content.find('---', 3)
                if end_index != -1:
                    frontmatter_text = content[3:end_index].strip()
                    frontmatter = yaml.safe_load(frontmatter_text) or {}
                    content_without_frontmatter = content[end_index + 3:].strip()
            except yaml.YAMLError:
                # If YAML parsing fails, treat as regular content
                pass
        
        return frontmatter, content_without_frontmatter
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into chunks with specified size and overlap"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is the last chunk, take whatever remains
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at a sentence boundary (., !, ?)
            chunk = text[start:end]
            
            # Look for sentence boundaries in the last 100 characters
            sentence_breaks = [
                chunk.rfind('.'),
                chunk.rfind('!'),
                chunk.rfind('?')
            ]
            
            # Find the best break point
            best_break = max(sentence_breaks)
            
            if best_break > start + chunk_size // 2:  # Don't go too far back
                end = start + best_break + 1
                chunk = text[start:end]
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - overlap
            
            # Ensure we make progress
            if start <= 0:
                start = end
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def find_markdown_files(self, docs_path: str) -> List[str]:
        """Recursively find all .md and .mdx files"""
        markdown_files = []
        docs_dir = Path(docs_path)
        
        if not docs_dir.exists():
            print(f"Docs directory not found: {docs_path}")
            return markdown_files
        
        for file_path in docs_dir.rglob('*'):
            if file_path.suffix.lower() in ['.md', '.mdx']:
                markdown_files.append(str(file_path))
        
        return sorted(markdown_files)
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single markdown file and return chunks with metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        
        # Extract frontmatter and content
        frontmatter, content = self.extract_frontmatter(content)
        
        # Get metadata
        title = frontmatter.get('title', Path(file_path).stem)
        description = frontmatter.get('description', '')
        
        # Chunk the content
        chunks = self.chunk_text(content)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_obj = {
                'id': str(uuid.uuid4()),
                'file_path': str(Path(file_path).relative_to('../docs')),
                'title': title,
                'description': description,
                'chunk_text': chunk_text,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def ingest_documents(self, docs_path: str = '../docs'):
        """Main method to ingest all documents"""
        print("Starting document ingestion...")
        
        # Find all markdown files
        markdown_files = self.find_markdown_files(docs_path)
        print(f"Found {len(markdown_files)} markdown files")
        
        total_chunks = 0
        
        for file_path in markdown_files:
            print(f"Processing {Path(file_path).name}...", end=' ')
            
            # Process the file
            chunks = self.process_file(file_path)
            
            if chunks:
                # Create embeddings for all chunks in this file
                chunk_texts = [chunk['chunk_text'] for chunk in chunks]
                embeddings = self.model.encode(chunk_texts)
                
                # Create points for Qdrant
                points = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    point = PointStruct(
                        id=chunk['id'],
                        vector=embedding.tolist(),
                        payload={
                            'file_path': chunk['file_path'],
                            'title': chunk['title'],
                            'description': chunk['description'],
                            'chunk_text': chunk['chunk_text'],
                            'chunk_index': chunk['chunk_index'],
                            'total_chunks': chunk['total_chunks']
                        }
                    )
                    points.append(point)
                
                # Upload to Qdrant
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"{len(chunks)} chunks created")
                    total_chunks += len(chunks)
                except Exception as e:
                    print(f"Error uploading to Qdrant: {e}")
            else:
                print("No chunks created")
        
        print(f"\nIngestion complete! Total chunks processed: {total_chunks}")

if __name__ == "__main__":
    ingestor = DocumentIngestor()
    ingestor.ingest_documents()
