"""
Context Retriever
Handles context retrieval using both semantic and keyword-based search
"""

from typing import List
from .semantic_search import SemanticSearchEngine
from .query_generator import QueryGenerator

class ContextRetriever:
    """Handles context retrieval for Q&A system"""
    
    def __init__(self, chunks: List[str] = None, pinecone_index=None):
        self.chunks = chunks or []
        self.semantic_search = SemanticSearchEngine()
        self.query_generator = QueryGenerator()
    
    def get_relevant_context(self, questions: List[str], document_chunks, chunk_embeddings=None) -> str:
        """
        Get relevant context using semantic search with keyword fallback
        
        Args:
            questions (List[str]): Questions to find context for
            document_chunks: List of document chunks
            chunk_embeddings: Precomputed chunk embeddings
            
        Returns:
            str: Relevant context text
        """
        # Try semantic search first
        if self.semantic_search.model and chunk_embeddings is not None:
            context = self.semantic_search.semantic_search(questions, document_chunks, chunk_embeddings)
            if context:
                return context
            
            print("🔄 Semantic search yielded no results, falling back to keyword search")
        
        # Fallback to keyword search
        return self.fast_context_retrieval(questions, document_chunks)
    
    def fast_context_retrieval(self, questions: List[str], document_chunks) -> str:
        """
        Ultra-fast context retrieval using advanced in-memory search with semantic understanding
        Uses fuzzy matching and enhanced scoring for better results
        """
        try:
            print("⚡ Performing ultra-fast semantic in-memory context retrieval...")
            
            if not document_chunks:
                print("⚠️ No document chunks available")
                return ""
            
            # Generate enhanced topic queries for comprehensive search
            topic_queries = self.query_generator.generate_enhanced_topic_queries(questions)
            
            # Extract key terms from questions and topic queries for searching
            search_terms = set()
            
            # Add insurance company specific terms
            company_terms = [
                "national", "insurance", "company", "star", "health", "icici", "lombard",
                "hdfc", "ergo", "bajaj", "allianz", "max", "bupa", "new", "india", 
                "assurance", "oriental", "united", "mediclaim", "policy", "issuer", "provider"
            ]
            search_terms.update(company_terms)
            
            # Add terms from original questions
            for question in questions:
                words = question.lower().replace('?', '').split()
                filtered_words = [w for w in words if len(w) > 3 and w not in 
                                ['what', 'does', 'this', 'policy', 'under', 'with', 'from', 'they', 'have', 'been', 'will', 'are']]
                search_terms.update(filtered_words)
            
            # Add terms from enhanced topic queries
            for query in topic_queries:
                words = query.lower().split()
                filtered_words = [w for w in words if len(w) > 2]
                search_terms.update(filtered_words)
            
            print(f"🔍 Searching with {len(search_terms)} enhanced terms including synonyms...")
            
            # Score chunks using advanced matching techniques
            chunk_scores = self._score_chunks(document_chunks, search_terms, topic_queries, questions)
            
            # Select top relevant chunks with diversity
            selected_chunks = self._select_diverse_chunks(chunk_scores, questions, document_chunks)
            
            context_texts = [chunk['text'] for chunk in selected_chunks]
            
            print(f"✅ Found {len(context_texts)} diverse relevant chunks with semantic scoring")
            if selected_chunks:
                print(f"📊 Score range: {selected_chunks[0]['score']:.1f} - {selected_chunks[-1]['score']:.1f}")
                print(f"🎯 Average matches per chunk: {sum(c['unique_matches'] for c in selected_chunks)/len(selected_chunks):.1f}")
            
            if not context_texts:
                print("⚠️ No relevant context found despite enhanced semantic search")
                return ""
            
            # Combine chunks with clear separators and metadata
            return "\n\n--- DOCUMENT SECTION ---\n\n".join(context_texts)
            
        except Exception as e:
            print(f"❌ Error in enhanced semantic context retrieval: {e}")
            return ""
    
    def _score_chunks(self, document_chunks, search_terms, topic_queries, questions):
        """Score chunks using advanced matching techniques"""
        chunk_scores = []
        
        for i, chunk in enumerate(document_chunks):
            score = 0
            text_lower = chunk['text_lower']
            
            # 1. Exact term matching with enhanced scoring
            for term in search_terms:
                if term in text_lower:
                    # Base score for exact matches
                    term_count = text_lower.count(term)
                    score += term_count * 12
                    
                    # Bonus for terms at start of chunk (likely headers/important)
                    if text_lower.startswith(term) or f' {term}' in text_lower[:100]:
                        score += 20
                    
                    # Extra bonus for longer, more specific terms
                    if len(term) > 6:
                        score += 15
                    
                    # Bonus for terms that appear in multiple search contexts
                    term_frequency = sum(1 for t in search_terms if term in t or t in term)
                    if term_frequency > 1:
                        score += term_frequency * 8
            
            # 2. Fuzzy/partial matching for better recall
            for term in search_terms:
                if len(term) > 4:  # Only for longer terms to avoid noise
                    # Check for partial matches (substring matching)
                    partial_matches = [word for word in text_lower.split() if term in word or word in term]
                    if partial_matches:
                        score += len(partial_matches) * 8
            
            # 3. Phrase matching bonus for topic queries
            for query in topic_queries[:20]:  # Check more topic queries for exact phrases
                if len(query) > 8 and query.lower() in text_lower:
                    score += 35  # High bonus for exact phrase matches
            
            # 4. Semantic proximity bonus - chunks with multiple related terms
            unique_matches = sum(1 for term in search_terms if term in text_lower)
            if unique_matches > 1:
                score += unique_matches * 12  # Higher multiplier for multiple matches
            
            # 5. Context-aware scoring - boost chunks with domain-specific patterns
            domain_indicators = ['waiting period', 'grace period', 'coverage', 'excluded', 'included', 
                               'benefits', 'premium', 'policy', 'treatment', 'medical', 'surgery',
                               'procedure', 'expenses', 'costs', 'donor', 'transplant', 'cataract',
                               'maternity', 'pregnancy', 'childbirth', 'delivery', 'ophthalmic',
                               'eye condition', 'lens replacement', 'intraocular', 'exclusion',
                               'not covered', 'covered under', 'eligible', 'payable', 'reimbursement']
            domain_matches = sum(1 for indicator in domain_indicators if indicator in text_lower)
            if domain_matches > 0:
                score += domain_matches * 10
            
            # 6. Special boost for sections that commonly contain specific medical procedures
            section_indicators = ['exclusions', 'covered services', 'benefits', 'schedule', 'appendix', 
                                'list of', 'table', 'waiting periods', 'specific conditions']
            section_matches = sum(1 for indicator in section_indicators if indicator in text_lower)
            if section_matches > 0:
                score += section_matches * 15  # Higher boost for sections likely to contain details
            
            # 7. Length penalty for very short chunks (likely incomplete)
            if len(chunk['text']) < 100:
                score *= 0.8  # Reduce score for very short chunks
            
            if score > 0:
                chunk_scores.append({
                    'index': i,
                    'score': score,
                    'text': chunk['text'],
                    'unique_matches': unique_matches
                })
        
        # Sort by score (highest first)
        chunk_scores.sort(key=lambda x: x['score'], reverse=True)
        return chunk_scores
    
    def _select_diverse_chunks(self, chunk_scores, questions, document_chunks):
        """Select diverse chunks to avoid redundancy"""
        selected_chunks = []
        seen_content = set()
        
        for chunk in chunk_scores[:50]:  # Consider top 50 for diversity selection
            # Simple content similarity check to avoid near-duplicates
            chunk_words = frozenset(chunk['text'].lower().split()[:20])  # Use frozenset for hashing
            
            is_similar = False
            for seen_words in seen_content:
                if len(chunk_words.intersection(seen_words)) > 15:  # High overlap
                    is_similar = True
                    break
            
            if not is_similar:
                selected_chunks.append(chunk)
                seen_content.add(chunk_words)
                
                if len(selected_chunks) >= 30:  # Increased for better coverage
                    break
        
        # Fallback search if initial results are insufficient
        if len(selected_chunks) < 15:
            selected_chunks = self._fallback_search(selected_chunks, seen_content, questions, document_chunks)
        
        return selected_chunks
    
    def _fallback_search(self, selected_chunks, seen_content, questions, document_chunks):
        """Perform fallback search with broader criteria"""
        print("🔄 Initial search yielded few results, applying enhanced fallback search...")
        
        # Broader search with relaxed criteria AND medical-specific terms
        broad_search_terms = set()
        medical_terms = set()
        
        for question in questions:
            # Extract more terms including shorter ones
            words = question.lower().replace('?', '').replace(',', '').split()
            broad_terms = [w for w in words if len(w) > 2 and w not in 
                         ['what', 'does', 'this', 'the', 'and', 'for', 'are', 'can', 'how']]
            broad_search_terms.update(broad_terms)
            
            # Add medical-specific search terms based on question content
            if 'cataract' in question.lower():
                medical_terms.update(['eye', 'lens', 'ophthalmic', 'vision', 'intraocular', 'IOL'])
            if 'maternity' in question.lower() or 'pregnancy' in question.lower():
                medical_terms.update(['pregnancy', 'childbirth', 'delivery', 'newborn', 'infant', 'confinement'])
            if 'donor' in question.lower():
                medical_terms.update(['transplant', 'donation', 'transplantation', 'organ'])
        
        # Combine all search terms
        all_fallback_terms = broad_search_terms.union(medical_terms)
        
        # Re-score with broader criteria
        fallback_scores = []
        for i, chunk in enumerate(document_chunks):
            if i in [c['index'] for c in selected_chunks]:
                continue  # Skip already selected
            
            score = 0
            text_lower = chunk['text_lower']
            
            # Standard term matching
            for term in all_fallback_terms:
                if term in text_lower:
                    score += text_lower.count(term) * 6
            
            # Bonus for medical procedure keywords that might be in lists/tables
            procedure_keywords = ['surgery', 'treatment', 'procedure', 'condition', 'disease', 'illness']
            for keyword in procedure_keywords:
                if keyword in text_lower:
                    score += text_lower.count(keyword) * 4
            
            # Extra bonus for exclusion/inclusion sections
            if any(section in text_lower for section in ['exclusion', 'inclusion', 'covered', 'not covered', 'eligible']):
                score += 20
            
            if score > 0:
                fallback_scores.append({
                    'index': i,
                    'score': score,
                    'text': chunk['text'],
                    'unique_matches': sum(1 for term in all_fallback_terms if term in text_lower)
                })
        
        # Sort and add best fallback results
        fallback_scores.sort(key=lambda x: x['score'], reverse=True)
        
        for chunk in fallback_scores[:15]:  # Add up to 15 more chunks
            chunk_words = set(chunk['text'].lower().split()[:20])
            is_similar = any(len(chunk_words.intersection(seen_words)) > 10 
                           for seen_words in seen_content)
            
            if not is_similar:
                selected_chunks.append(chunk)
                seen_content.add(frozenset(chunk_words))
                
                if len(selected_chunks) >= 35:  # Increased total limit
                    break
        
        print(f"🔄 Enhanced fallback search added {len(selected_chunks) - 30} additional chunks with medical focus")
        return selected_chunks
