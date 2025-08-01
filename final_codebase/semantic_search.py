"""
Semantic Search Engine
Handles semantic similarity search using sentence transformers
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .config import Config

class SemanticSearchEngine:
    """Semantic search engine using sentence transformers"""
    
    def __init__(self):
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """Setup Sentence Transformer model for semantic similarity"""
        try:
            print("🤖 Loading semantic similarity model...")
            self.model = SentenceTransformer(Config.SEMANTIC_MODEL)
            print("✅ Semantic model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error setting up semantic model: {e}")
            print("⚠️  Falling back to keyword-based search")
            self.model = None
    
    def encode_texts(self, texts, show_progress=True):
        """
        Encode texts using the semantic model
        
        Args:
            texts (List[str]): List of texts to encode
            show_progress (bool): Whether to show progress bar
            
        Returns:
            numpy.ndarray: Encoded embeddings
        """
        if not self.model:
            return None
            
        try:
            return self.model.encode(
                texts, 
                convert_to_tensor=False,
                show_progress_bar=show_progress
            )
        except Exception as e:
            print(f"❌ Error encoding texts: {e}")
            return None
    
    def compute_similarity(self, query_embeddings, chunk_embeddings):
        """
        Compute cosine similarity between query and chunk embeddings
        
        Args:
            query_embeddings: Query embeddings
            chunk_embeddings: Chunk embeddings
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        try:
            return cosine_similarity(query_embeddings, chunk_embeddings)
        except Exception as e:
            print(f"❌ Error computing similarity: {e}")
            return None
    
    def semantic_search(self, questions, document_chunks, chunk_embeddings):
        """
        Perform semantic search to find relevant chunks
        
        Args:
            questions (List[str]): Questions to search for
            document_chunks (List[dict]): Document chunks
            chunk_embeddings: Precomputed chunk embeddings
            
        Returns:
            str: Concatenated relevant context
        """
        try:
            if not self.model or chunk_embeddings is None:
                print("⚠️  Semantic model not available")
                return ""
            
            print("🧠 Performing advanced semantic context retrieval...")
            
            # Combine questions with enhanced topic queries
            from .query_generator import QueryGenerator
            query_gen = QueryGenerator()
            
            search_queries = questions.copy()
            topic_queries = query_gen.generate_enhanced_topic_queries(questions)
            search_queries.extend(topic_queries[:Config.MAX_SEARCH_QUERIES])
            
            print(f"🔍 Performing semantic search with {len(search_queries)} queries...")
            
            # Encode search queries
            query_embeddings = self.encode_texts(search_queries, show_progress=False)
            if query_embeddings is None:
                return ""
            
            # Compute similarities
            similarities = self.compute_similarity(query_embeddings, chunk_embeddings)
            if similarities is None:
                return ""
            
            # Score and select chunks
            selected_chunks = self._score_and_select_chunks(
                similarities, document_chunks, questions
            )
            
            if not selected_chunks:
                print("⚠️ No semantically relevant context found")
                return ""
            
            context_texts = [chunk['text'] for chunk in selected_chunks]
            print(f"✅ Selected {len(context_texts)} diverse semantic chunks")
            
            return "\n\n--- DOCUMENT SECTION ---\n\n".join(context_texts)
            
        except Exception as e:
            print(f"❌ Error in semantic search: {e}")
            return ""
    
    def _score_and_select_chunks(self, similarities, document_chunks, questions):
        """Score chunks based on semantic similarity and select diverse ones"""
        chunk_scores = []
        
        for i, chunk in enumerate(document_chunks):
            # Get maximum and average similarity across all queries
            max_similarity = np.max(similarities[:, i])
            avg_similarity = np.mean(similarities[:, i])
            final_score = (max_similarity * 0.7) + (avg_similarity * 0.3)
            
            # Apply keyword boosting
            text_lower = chunk['text_lower']
            keyword_boost = self._calculate_keyword_boost(text_lower, questions)
            final_score += min(keyword_boost, 0.5)
            
            if final_score > 0.1:  # Only include reasonably similar chunks
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
            print(f"🎯 Top score: {chunk_scores[0]['score']:.3f}, Bottom score: {chunk_scores[-1]['score']:.3f}")
        
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
            print(f"📈 Semantic score range: {selected_chunks[0]['semantic_score']:.3f} - {selected_chunks[-1]['semantic_score']:.3f}")
        
        return selected_chunks
