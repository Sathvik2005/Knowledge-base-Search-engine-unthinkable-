"""
Generation module for Answer IQ.
Handles context-aware answer generation using local transformer models.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from .config import (
    GENERATION_MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DO_SAMPLE,
    DEVICE
)
from .retrieval import SearchResult


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with metadata."""
    answer: str
    query: str
    context_used: str
    sources: List[str]
    confidence: float
    is_grounded: bool


class AnswerGenerator:
    """
    Context-aware answer generation using local transformer models.
    Designed to minimize hallucinations and stay grounded in retrieved context.
    """
    
    def __init__(self, model_name: str = GENERATION_MODEL_NAME):
        """
        Initialize the answer generator.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self._loaded = False
    
    def load(self) -> None:
        """Load the generation model into memory."""
        if self._loaded:
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Move to appropriate device
        device_id = 0 if DEVICE == "cuda" and torch.cuda.is_available() else -1
        
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE
        )
        
        self._loaded = True
    
    def generate(self, query: str, context: str, 
                 search_results: List[SearchResult],
                 concise: bool = True) -> GeneratedAnswer:
        """
        Generate an answer based on the query and retrieved context.
        
        Args:
            query: User's question
            context: Retrieved context string
            search_results: List of search results used
            concise: Whether to generate a concise answer
            
        Returns:
            GeneratedAnswer object
        """
        if not self._loaded:
            self.load()
        
        # Handle case when no context is available
        if not context or not context.strip():
            return GeneratedAnswer(
                answer="I could not find relevant information in the knowledge base to answer this question. Please ensure relevant documents have been uploaded and indexed.",
                query=query,
                context_used="",
                sources=[],
                confidence=0.0,
                is_grounded=False
            )
        
        # Build the prompt
        prompt = self._build_prompt(query, context, concise)
        
        # Generate answer
        try:
            output = self.pipe(prompt)[0]['generated_text']
            answer = self._postprocess_answer(output)
        except Exception as e:
            return GeneratedAnswer(
                answer=f"An error occurred during answer generation: {str(e)}",
                query=query,
                context_used=context,
                sources=[],
                confidence=0.0,
                is_grounded=False
            )
        
        # Extract sources
        sources = list(set(
            result.chunk.metadata.get('filename', 'Unknown')
            for result in search_results
        ))
        
        # Calculate confidence based on search scores
        avg_score = sum(r.score for r in search_results) / len(search_results) if search_results else 0
        
        return GeneratedAnswer(
            answer=answer,
            query=query,
            context_used=context,
            sources=sources,
            confidence=avg_score,
            is_grounded=True
        )
    
    def _build_prompt(self, query: str, context: str, concise: bool) -> str:
        """Build the generation prompt."""
        style_instruction = "Provide a brief, direct answer." if concise else "Provide a comprehensive, detailed answer."
        
        prompt = f"""Answer the question based only on the provided context. If the context does not contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Instructions: {style_instruction} Do not make up information that is not in the context.

Answer:"""
        
        return prompt
    
    def _postprocess_answer(self, answer: str) -> str:
        """Clean up the generated answer."""
        answer = answer.strip()
        
        # Remove any prompt leakage
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Ensure proper capitalization
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Ensure proper ending
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        return answer


class QueryProcessor:
    """
    Processes user queries and coordinates retrieval and generation.
    """
    
    def __init__(self, retriever, generator: AnswerGenerator):
        """
        Initialize the query processor.
        
        Args:
            retriever: Retriever instance for document search
            generator: AnswerGenerator instance
        """
        self.retriever = retriever
        self.generator = generator
    
    def process(self, query: str, top_k: int = 5, 
                concise: bool = True) -> GeneratedAnswer:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            concise: Whether to generate concise answer
            
        Returns:
            GeneratedAnswer object
        """
        # Validate query
        query = query.strip()
        if not query:
            return GeneratedAnswer(
                answer="Please provide a valid question.",
                query="",
                context_used="",
                sources=[],
                confidence=0.0,
                is_grounded=False
            )
        
        # Retrieve relevant context
        context, search_results = self.retriever.get_context(query, top_k)
        
        # Generate answer
        answer = self.generator.generate(query, context, search_results, concise)
        
        return answer
