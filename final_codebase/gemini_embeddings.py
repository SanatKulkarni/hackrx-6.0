"""
Gemini Embeddings Engine
Uses Gemini's native embedding model for semantic search
"""

import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from .config import Config

class GeminiEmbeddingsEngine:
    """Gemini-native embeddings for optimal semantic search"""
    
    def __init__(self):
        self.embedding_model = "text-embedding-004"  # Latest Gemini embedding model
        self.setup_client()
    
    def setup_client(self):
        """Setup Gemini embeddings client"""
        try:
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found")
            
            genai.configure(api_key=Config.GEMINI_API_KEY)
            print("✅ Gemini Embeddings client initialized")
            
        except Exception as e:
            print(f"❌ Error setting up Gemini Embeddings: {e}")
            raise
    
    def embed_documents(self, texts: List[str], show_progress: bool = True) -> Optional[np.ndarray]:
        """
        Generate embeddings for document chunks
        
        Args:
            texts: List of document texts
            show_progress: Whether to show progress
            
        Returns:
            numpy array of embeddings
        """
        try:
            print(f"🔮 Generating Gemini embeddings for {len(texts)} documents...")
            
            embeddings = []
            for i, text in enumerate(texts):
                if show_progress and i % 50 == 0:
                    print(f"   Processing document {i+1}/{len(texts)}")
                
                # Use RETRIEVAL_DOCUMENT task type for documents
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                embeddings.append(response['embedding'])
            
            embeddings_array = np.array(embeddings)
            print(f"✅ Generated {len(embeddings_array)} Gemini embeddings")
            return embeddings_array
            
        except Exception as e:
            print(f"❌ Error generating document embeddings: {e}")
            return None
    
    def embed_queries(self, queries: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for search queries
        
        Args:
            queries: List of query texts
            
        Returns:
            numpy array of query embeddings
        """
        try:
            print(f"🔍 Generating Gemini query embeddings for {len(queries)} queries...")
            
            embeddings = []
            for query in queries:
                # Use RETRIEVAL_QUERY task type for queries
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=query,
                    task_type="RETRIEVAL_QUERY"
                )
                
                embeddings.append(response['embedding'])
            
            embeddings_array = np.array(embeddings)
            print(f"✅ Generated {len(embeddings_array)} query embeddings")
            return embeddings_array
            
        except Exception as e:
            print(f"❌ Error generating query embeddings: {e}")
            return None
    
    def compute_similarity(self, query_embeddings: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute dot product similarities between queries and documents
        
        Args:
            query_embeddings: Query embedding vectors
            document_embeddings: Document embedding vectors
            
        Returns:
            Similarity matrix
        """
        try:
            # Compute dot product (Gemini embeddings are already normalized)
            similarities = np.dot(query_embeddings, document_embeddings.T)
            return similarities
            
        except Exception as e:
            print(f"❌ Error computing similarities: {e}")
            return np.array([])
    
    def semantic_search(self, questions: List[str], document_chunks: List[dict], 
                       chunk_embeddings: Optional[np.ndarray] = None) -> str:
        """
        Perform semantic search using Gemini embeddings
        
        Args:
            questions: List of questions
            document_chunks: List of document chunks
            chunk_embeddings: Pre-computed document embeddings (optional)
            
        Returns:
            Concatenated relevant context
        """
        try:
            print("🧠 Performing Gemini-native semantic search...")
            
            # Generate enhanced queries
            from .query_generator import QueryGenerator
            query_gen = QueryGenerator()
            
            search_queries = questions.copy()
            topic_queries = query_gen.generate_enhanced_topic_queries(questions)
            search_queries.extend(topic_queries[:Config.MAX_SEARCH_QUERIES])
            
            print(f"🔍 Searching with {len(search_queries)} Gemini-optimized queries...")
            
            # Generate query embeddings
            query_embeddings = self.embed_queries(search_queries)
            if query_embeddings is None:
                return ""
            
            # Use provided embeddings or generate new ones
            if chunk_embeddings is None:
                chunk_texts = [chunk['text'] for chunk in document_chunks]
                chunk_embeddings = self.embed_documents(chunk_texts, show_progress=False)
            
            if chunk_embeddings is None:
                return ""
            
            # Compute similarities using dot product
            similarities = self.compute_similarity(query_embeddings, chunk_embeddings)
            
            # Score and select chunks
            selected_chunks = self._score_and_select_chunks(
                similarities, document_chunks, questions
            )
            
            if not selected_chunks:
                print("⚠️ No semantically relevant context found")
                return ""
            
            context_texts = [chunk['text'] for chunk in selected_chunks]
            print(f"✅ Selected {len(context_texts)} diverse Gemini-optimized chunks")
            
            return "\n\n--- DOCUMENT SECTION ---\n\n".join(context_texts)
            
        except Exception as e:
            print(f"❌ Error in Gemini semantic search: {e}")
            return ""
    
    def _score_and_select_chunks(self, similarities, document_chunks, questions):
        """Score chunks based on Gemini semantic similarity and select diverse ones"""
        chunk_scores = []
        
        for i, chunk in enumerate(document_chunks):
            # Get maximum and average similarity across all queries
            max_similarity = np.max(similarities[:, i])
            avg_similarity = np.mean(similarities[:, i])
            final_score = (max_similarity * 0.7) + (avg_similarity * 0.3)
            
            # Apply keyword boosting
            text_lower = chunk['text'].lower()
            keyword_boost = self._calculate_keyword_boost(text_lower, questions)
            final_score += min(keyword_boost, 0.5)
            
            if final_score > 0.2:  # Higher threshold for Gemini embeddings
                chunk_scores.append({
                    'index': i,
                    'score': final_score,
                    'text': chunk['text'],
                    'semantic_score': max_similarity,
                    'avg_score': avg_similarity
                })
        
        # Sort by score
        chunk_scores.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📊 Found {len(chunk_scores)} semantically relevant chunks")
        if chunk_scores:
            print(f"🎯 Gemini score range: {chunk_scores[0]['score']:.3f} - {chunk_scores[-1]['score']:.3f}")
        
        # Select diverse chunks
        return self._select_diverse_chunks(chunk_scores)
    
    def _calculate_keyword_boost(self, text_lower, questions):
        """Calculate keyword boost for domain-specific terms"""
        keyword_boost = 0
        
        # Boost for domain-specific terms
        domain_terms = ['waiting period', 'grace period', 'cataract', 'maternity', 'donor', 
                       'pre-existing', 'coverage', 'excluded', 'benefits', 'premium']
        for term in domain_terms:
            if term in text_lower:
                keyword_boost += 0.1
        
        # Boost for question-specific terms
        for question in questions:
            question_words = question.lower().split()
            for word in question_words:
                if len(word) > 4 and word in text_lower:
                    keyword_boost += 0.05
        
        return keyword_boost
    
    def _select_diverse_chunks(self, chunk_scores, max_chunks=35):
        """Select diverse chunks to avoid redundancy"""
        selected_chunks = []
        seen_content = set()
        
        for chunk in chunk_scores[:100]:  # Consider top 100 for diversity
            # Content diversity check
            chunk_words = frozenset(chunk['text'].lower().split()[:15])
            
            is_similar = False
            for seen_words in seen_content:
                if len(chunk_words.intersection(seen_words)) > 10:
                    is_similar = True
                    break
            
            if not is_similar:
                selected_chunks.append(chunk)
                seen_content.add(chunk_words)
                
                if len(selected_chunks) >= max_chunks:
                    break
        
        if selected_chunks:
            print(f"📈 Gemini semantic score range: {selected_chunks[0]['semantic_score']:.3f} - {selected_chunks[-1]['semantic_score']:.3f}")
        
        return selected_chunks
