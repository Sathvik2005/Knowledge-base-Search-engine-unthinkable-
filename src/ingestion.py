"""
Data ingestion module for Answer IQ.
Handles document loading, text extraction, and preprocessing.
"""

import re
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib

from pypdf import PdfReader


@dataclass
class Document:
    """Represents a processed document."""
    id: str
    filename: str
    content: str
    page_count: int
    word_count: int


@dataclass
class TextChunk:
    """Represents a chunk of text from a document."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict


class DocumentLoader:
    """Handles loading and text extraction from various file formats."""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.txt': self._load_txt,
        }
    
    def load(self, file_path: Path) -> Optional[Document]:
        """
        Load a document from the given file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object or None if loading fails
        """
        suffix = file_path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        loader_func = self.supported_formats[suffix]
        return loader_func(file_path)
    
    def load_from_upload(self, uploaded_file) -> Optional[Document]:
        """
        Load a document from a Streamlit uploaded file object.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Document object or None if loading fails
        """
        suffix = Path(uploaded_file.name).suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)
        
        try:
            doc = self.supported_formats[suffix](tmp_path)
            if doc:
                doc.filename = uploaded_file.name
            return doc
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def _load_pdf(self, file_path: Path) -> Optional[Document]:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(str(file_path))
            pages = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            
            content = "\n\n".join(pages)
            content = self._clean_text(content)
            
            doc_id = self._generate_id(file_path.name, content)
            
            return Document(
                id=doc_id,
                filename=file_path.name,
                content=content,
                page_count=len(reader.pages),
                word_count=len(content.split())
            )
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return None
    
    def _load_txt(self, file_path: Path) -> Optional[Document]:
        """Load text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            content = self._clean_text(content)
            doc_id = self._generate_id(file_path.name, content)
            
            return Document(
                id=doc_id,
                filename=file_path.name,
                content=content,
                page_count=1,
                word_count=len(content.split())
            )
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()\-]', '', text)
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _generate_id(self, filename: str, content: str) -> str:
        """Generate a unique ID for the document."""
        hash_input = f"{filename}:{content[:1000]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]


class TextSplitter:
    """Splits documents into semantic chunks with configurable overlap."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, document: Document) -> List[TextChunk]:
        """
        Split a document into overlapping chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of TextChunk objects
        """
        text = document.content
        chunks = []
        
        # Split by sentences first for better semantic boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_length = 0
        start_char = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{document.id}_{chunk_index}"
                
                chunks.append(TextChunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        'filename': document.filename,
                        'chunk_index': chunk_index,
                        'total_chunks': -1  # Will be updated later
                    }
                ))
                
                chunk_index += 1
                
                # Keep overlap
                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else chunk_text
                start_char = start_char + len(chunk_text) - len(overlap_text)
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Handle remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{document.id}_{chunk_index}"
            
            chunks.append(TextChunk(
                id=chunk_id,
                document_id=document.id,
                content=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata={
                    'filename': document.filename,
                    'chunk_index': chunk_index,
                    'total_chunks': -1
                }
            ))
        
        # Update total chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]


class IngestionPipeline:
    """Complete document ingestion pipeline."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.processed_documents: Dict[str, Document] = {}
        self.all_chunks: List[TextChunk] = []
    
    def process_files(self, uploaded_files: List) -> Tuple[int, int, List[str]]:
        """
        Process multiple uploaded files.
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            Tuple of (documents_count, chunks_count, error_messages)
        """
        errors = []
        new_chunks = []
        
        for uploaded_file in uploaded_files:
            try:
                # Load document
                doc = self.loader.load_from_upload(uploaded_file)
                
                if doc is None:
                    errors.append(f"Failed to load: {uploaded_file.name}")
                    continue
                
                # Skip if already processed
                if doc.id in self.processed_documents:
                    continue
                
                # Split into chunks
                chunks = self.splitter.split(doc)
                
                # Store
                self.processed_documents[doc.id] = doc
                self.all_chunks.extend(chunks)
                new_chunks.extend(chunks)
                
            except Exception as e:
                errors.append(f"Error processing {uploaded_file.name}: {str(e)}")
        
        return len(self.processed_documents), len(self.all_chunks), errors
    
    def get_statistics(self) -> Dict:
        """Get current pipeline statistics."""
        return {
            'document_count': len(self.processed_documents),
            'chunk_count': len(self.all_chunks),
            'total_words': sum(doc.word_count for doc in self.processed_documents.values()),
            'documents': [
                {
                    'filename': doc.filename,
                    'pages': doc.page_count,
                    'words': doc.word_count
                }
                for doc in self.processed_documents.values()
            ]
        }
    
    def clear(self):
        """Clear all processed documents and chunks."""
        self.processed_documents.clear()
        self.all_chunks.clear()
