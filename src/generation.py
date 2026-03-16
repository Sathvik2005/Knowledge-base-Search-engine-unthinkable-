"""
Generation module for Answer IQ.
Handles context-aware answer generation via OpenAI and Groq APIs.
"""

from typing import List
from dataclasses import dataclass
from openai import OpenAI

from .config import (
    GENERATION_API_MODEL,
    GROQ_API_BASE,
    MAX_NEW_TOKENS,
    TEMPERATURE
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
    Context-aware answer generation using API-based LLMs.
    Designed to minimize hallucinations and stay grounded in retrieved context.
    """
    
    def __init__(self, api_key: str, model_name: str = GENERATION_API_MODEL,
                 provider: str = "groq"):
        """
        Initialize the answer generator.

        Args:
            api_key: API key for the chosen provider
            model_name: Model identifier (provider-specific)
            provider: "openai" or "groq"
        """
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider.lower()
        self.client = None
        self._loaded = False
    
    def load(self) -> None:
        """Load the generation model into memory."""
        if self._loaded:
            return

        if not self.api_key or not self.api_key.strip():
            raise ValueError("API key is required for answer generation.")

        client_kwargs = {"api_key": self.api_key.strip()}
        if self.provider == "groq":
            client_kwargs["base_url"] = GROQ_API_BASE

        self.client = OpenAI(**client_kwargs)
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
        system_prompt, user_prompt = self._build_prompt(query, context, concise)
        
        # Generate answer
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.2 if concise else TEMPERATURE
            )

            output = completion.choices[0].message.content or ""
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
    
    def _build_prompt(self, query: str, context: str, concise: bool):
        """Build the generation prompt."""
        style_instruction = "Provide a brief, direct answer." if concise else "Provide a comprehensive, detailed answer."

        system_prompt = (
            "You are a retrieval-grounded assistant. "
            "Answer only using the provided context. "
            "If the context is insufficient, say that clearly. "
            "Do not hallucinate or invent facts."
        )

        user_prompt = f"""Context:
{context}

Question: {query}

Instructions: {style_instruction}"""

        return system_prompt, user_prompt
    
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
