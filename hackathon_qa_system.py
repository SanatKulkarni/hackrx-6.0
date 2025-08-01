"""
Hackathon Q&A System - Complete RAG Pipeline
Processes documents from URLs and answers multiple questions
Format: {"documents": "URL", "questions": [...]} -> {"answers": [...]}
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import asyncio
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Add paths for imports
current_dir = os.path.dirname(__file__)
loader_dir = os.path.join(current_dir, 'loader_&_extractor')
chunking_dir = os.path.join(current_dir, 'chunking')
embeddings_dir = os.path.join(current_dir, 'embeddings')

sys.path.extend([loader_dir, chunking_dir, embeddings_dir, current_dir])

# Import our modules
def import_modules():
    """Import required modules with error handling"""
    try:
        # Change to loader directory for import
        original_dir = os.getcwd()
        os.chdir(loader_dir)
        sys.path.insert(0, loader_dir)
        from url_document_loader import process_document
        
        # Change to chunking directory
        os.chdir(chunking_dir) 
        sys.path.insert(0, chunking_dir)
        from main_chunking import chunk_text
        
        # Change back to root for embeddings
        os.chdir(original_dir)
        sys.path.insert(0, current_dir)
        from embeddings import embed_and_store_chunks, query_embeddings
        
        # Import Gemini for answer generation
        from google import genai
        
        return process_document, chunk_text, embed_and_store_chunks, query_embeddings, genai
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_dir)

# Import all modules
process_document, chunk_text, embed_and_store_chunks, query_embeddings, genai = import_modules()

class HackathonQASystem:
    """Complete Q&A system for hackathon requirements"""
    
    def __init__(self, index_name="hackathon-policies", namespace="documents"):
        """
        Initialize the Q&A system
        
        Args:
            index_name (str): Pinecone index name
            namespace (str): Pinecone namespace
        """
        self.index_name = index_name
        self.namespace = namespace
        self.gemini_client = None
        self.document_chunks = []  # Cache for faster retrieval
        self.chunks_loaded = False
        self.semantic_model = None  # Sentence transformer model
        self.chunk_embeddings = None  # Precomputed embeddings
        self.setup_gemini()
        self.setup_semantic_model()
    
    def setup_semantic_model(self):
        """Setup Sentence Transformer model for semantic similarity"""
        try:
            print("🤖 Loading semantic similarity model...")
            # Use a lightweight, fast model optimized for semantic search
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error setting up semantic model: {e}")
            print("⚠️  Falling back to keyword-based search")
            self.semantic_model = None
    
    def setup_gemini(self):
        """Setup Gemini API client"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            self.gemini_client = genai.Client(api_key=api_key)
            print("✅ Gemini API client initialized")
            
        except Exception as e:
            print(f"❌ Error setting up Gemini: {e}")
            raise
    
    def check_document_exists(self, document_url: str) -> bool:
        """
        Check if document already exists in the vector database
        
        Args:
            document_url (str): URL of the document to check
            
        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            # Create a unique identifier for the document based on URL
            import hashlib
            doc_id = hashlib.md5(document_url.encode()).hexdigest()
            
            # Fast existence check: Use minimal query with very low n_results  
            test_results = query_embeddings(
                query_text="policy",  # Simple, short query
                index_name=self.index_name,
                namespace=self.namespace,
                n_results=1,  # Just need one result to confirm existence
                filter={"document_id": doc_id}  # Filter by document ID
            )
            
            # If we got any matches with this document_id, the document exists
            exists = test_results and 'matches' in test_results and len(test_results['matches']) > 0
            
            if exists:
                print(f"📋 Document ID {doc_id[:8]}... found in vector database")
            else:
                print(f"🆕 Document ID {doc_id[:8]}... not found - new document")
                
            return exists
            
        except Exception as e:
            print(f"⚠️  Warning: Could not check document existence: {e}")
            # If we can't check, assume it doesn't exist to be safe
            return False
    
    def process_and_store_document(self, document_url: str) -> bool:
        """
        Process document from URL and store in Pinecone (only if not already processed)
        
        Args:
            document_url (str): URL of the document to process
            
        Returns:
            bool: Success status
        """
        try:
            print(f"📄 Processing document from URL...")
            
            # Check if document already exists
            if self.check_document_exists(document_url):
                print(f"✅ Document already exists in vector database - skipping processing")
                print(f"🚀 Using cached document embeddings for faster response")
                return True
            
            print(f"📥 New document detected - processing and storing...")
            
            # Step 1: Process document
            doc_result = process_document(document_url, 'url')
            if not doc_result['success']:
                print(f"❌ Document processing failed: {doc_result.get('error')}")
                return False
            
            text = doc_result['text']
            print(f"✅ Document processed: {len(text)} characters")
            
            # Step 2: Chunk text
            print(f"🔪 Chunking text...")
            chunks = chunk_text(text, chunk_size=1500, chunk_overlap=300)
            if not chunks:
                print("❌ Text chunking failed")
                return False
            
            print(f"✅ Text chunked: {len(chunks)} chunks")
            
            # Step 3: Add document metadata to chunks
            import hashlib
            doc_id = hashlib.md5(document_url.encode()).hexdigest()
            
            # Add metadata to each chunk
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                enriched_chunk = {
                    'text': chunk,
                    'document_id': doc_id,
                    'document_url': document_url,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                enriched_chunks.append(enriched_chunk)
            
            # Step 4: Generate embeddings and store with metadata
            print(f"🔮 Generating embeddings and storing...")
            index = embed_and_store_chunks(enriched_chunks, self.index_name, self.namespace)
            if not index:
                print("❌ Embedding and storage failed")
                return False
            
            print(f"✅ Document processed and stored successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            return False

    def load_document_chunks(self) -> bool:
        """
        Load all document chunks into memory for fast retrieval
        This replaces slow Pinecone queries with in-memory search
        """
        try:
            if self.chunks_loaded:
                return True
                
            print("📚 Loading document chunks for fast retrieval...")
            
            # Get all chunks from Pinecone once
            results = query_embeddings(
                query_text="policy insurance coverage",  # Generic query to get all chunks
                index_name=self.index_name,
                namespace=self.namespace,
                n_results=1000  # Get all available chunks
            )
            
            if not results or 'matches' not in results:
                print("⚠️  No document chunks found in database")
                return False
            
            # Store chunks with metadata for fast search
            self.document_chunks = []
            chunk_texts = []
            
            for match in results['matches']:
                if 'metadata' in match and 'text' in match['metadata']:
                    chunk_text = match['metadata']['text']
                    chunk_data = {
                        'text': chunk_text,
                        'text_lower': chunk_text.lower(),  # For fast searching
                        'score': match.get('score', 0.0)
                    }
                    self.document_chunks.append(chunk_data)
                    chunk_texts.append(chunk_text)
            
            # Precompute semantic embeddings if model is available
            if self.semantic_model and chunk_texts:
                print("🔮 Computing semantic embeddings for chunks...")
                self.chunk_embeddings = self.semantic_model.encode(
                    chunk_texts, 
                    convert_to_tensor=False,
                    show_progress_bar=True
                )
                print(f"✅ Computed embeddings for {len(chunk_texts)} chunks")
            
            self.chunks_loaded = True
            print(f"✅ Loaded {len(self.document_chunks)} document chunks into memory")
            return True
            
        except Exception as e:
            print(f"❌ Error loading document chunks: {e}")
            return False

    def generate_enhanced_topic_queries(self, questions: List[str]) -> List[str]:
        """
        Generate comprehensive topic queries for better context retrieval.
        Uses advanced NLP techniques for better semantic understanding.
        """
        topic_queries = []
        
        # Add the original questions as primary queries
        topic_queries.extend(questions)
        
        # Extract enhanced key terms from each question dynamically
        for question in questions:
            question_lower = question.lower()
            
            # Extract key nouns, verbs, and domain-specific terms
            key_terms = []
            
            # Split question into words and extract meaningful terms
            words = question_lower.replace('?', '').replace(',', '').split()
            
            # Extract important terms (skip common question words)
            skip_words = {'what', 'is', 'the', 'are', 'does', 'do', 'can', 'how', 'when', 'where', 'why', 'which', 'who', 'under', 'this', 'that', 'these', 'those', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'by', 'from', 'to', 'in', 'on', 'at', 'of'}
            meaningful_words = [word for word in words if len(word) > 2 and word not in skip_words]
            
            # Add synonyms and related terms using domain knowledge
            enhanced_terms = self._generate_synonyms_and_variants(meaningful_words)
            key_terms.extend(enhanced_terms)
            
            # Create variations and combinations of key terms
            for word in meaningful_words:
                # Add the word itself
                key_terms.append(word)
                
                # Add common variations for document search
                if len(word) > 4:
                    key_terms.append(f"{word}s")  # plural
                    if word.endswith('y'):
                        key_terms.append(f"{word[:-1]}ies")  # e.g., policy -> policies
                    if word.endswith('e'):
                        key_terms.append(f"{word}s")  # e.g., service -> services
                
                # Add contextual phrases for the word
                key_terms.append(f"{word} details")
                key_terms.append(f"{word} information")
                key_terms.append(f"{word} conditions")
                key_terms.append(f"{word} terms")
                key_terms.append(f"{word} benefits")
                key_terms.append(f"{word} coverage")
            
            # Create 2-word and 3-word combinations from meaningful words
            for i, word1 in enumerate(meaningful_words):
                # 2-word combinations
                for j, word2 in enumerate(meaningful_words[i+1:], i+1):
                    key_terms.append(f"{word1} {word2}")
                    key_terms.append(f"{word2} {word1}")  # reverse order too
                
                # 3-word combinations for very specific searches
                for j, word2 in enumerate(meaningful_words[i+1:], i+1):
                    for k, word3 in enumerate(meaningful_words[j+1:], j+1):
                        if len(f"{word1} {word2} {word3}") < 50:  # reasonable length
                            key_terms.append(f"{word1} {word2} {word3}")
            
            # Add domain-specific expansions based on detected context
            context_expansions = self._generate_context_expansions(question_lower, meaningful_words)
            key_terms.extend(context_expansions)
            
            # Add all key terms to topic queries
            topic_queries.extend(key_terms)
        
        # Remove duplicates while preserving order and filter out very short terms
        seen = set()
        unique_queries = []
        for query in topic_queries:
            clean_query = query.strip()
            if clean_query not in seen and len(clean_query) > 2:
                seen.add(clean_query)
                unique_queries.append(clean_query)
        
        # Limit to reasonable number for performance
        unique_queries = unique_queries[:150]  # Increased for better coverage
        
        print(f"🎯 Generated {len(unique_queries)} enhanced topic queries with synonyms and variants")
        return unique_queries

    def _generate_synonyms_and_variants(self, meaningful_words: List[str]) -> List[str]:
        """
        Generate synonyms and variants for better semantic matching.
        This is a simplified version - in production, use libraries like NLTK, spaCy, or word embeddings.
        """
        synonyms = []
        
        # Domain-specific synonym dictionary
        synonym_dict = {
            # Medical/Insurance terms
            'surgery': ['operation', 'procedure', 'treatment', 'medical procedure', 'surgical treatment'],
            'cataract': ['eye surgery', 'lens surgery', 'eye procedure', 'lens replacement', 'eye treatment', 'ophthalmic', 'ophthalmology', 'eye condition', 'lens condition'],
            'donor': ['organ donor', 'transplant donor', 'donor expenses', 'donor treatment', 'donation', 'transplantation'],
            'organ': ['transplant', 'organ transplant', 'organ donation', 'transplantation', 'donor organ'],
            'expenses': ['costs', 'charges', 'fees', 'payments', 'bills', 'expenditure', 'amount'],
            'coverage': ['benefits', 'protection', 'insurance', 'covered', 'benefit', 'cover'],
            'waiting': ['wait', 'delay', 'period before', 'moratorium'],
            'period': ['time', 'duration', 'term', 'months', 'years', 'days'],
            'grace': ['grace period', 'additional time', 'extension', 'extra time'],
            'premium': ['payment', 'installment', 'fee', 'contribution', 'amount due'],
            'maternity': ['pregnancy', 'childbirth', 'delivery', 'maternal', 'obstetric', 'confinement', 'newborn', 'baby', 'infant'],
            'pre-existing': ['existing', 'prior condition', 'previous disease', 'pre existing'],
            'ped': ['pre-existing disease', 'existing condition', 'prior disease'],
            'policy': ['insurance', 'plan', 'coverage', 'scheme'],
            'medical': ['health', 'healthcare', 'treatment', 'clinical'],
            'treatment': ['therapy', 'care', 'medical care', 'cure'],
            'covered': ['included', 'eligible', 'benefits', 'payable'],
            'exclude': ['not covered', 'excluded', 'not included', 'exclusion'],
            'hospitalization': ['hospital stay', 'admission', 'inpatient', 'indoor treatment'],
            'claim': ['reimbursement', 'payment', 'settlement', 'benefit payment'],
            # Additional specific terms for better matching
            'eye': ['ophthalmic', 'ophthalmology', 'ocular', 'vision'],
            'lens': ['intraocular', 'IOL', 'artificial lens'],
            'birth': ['delivery', 'confinement', 'labor', 'childbirth'],
            'newborn': ['infant', 'baby', 'child', 'neonate']
        }
        
        for word in meaningful_words:
            if word in synonym_dict:
                synonyms.extend(synonym_dict[word])
            
            # Add partial matches for compound terms
            for key, values in synonym_dict.items():
                if word in key or key in word:
                    synonyms.extend(values)
        
        return synonyms

    def _generate_context_expansions(self, question_lower: str, meaningful_words: List[str]) -> List[str]:
        """
        Generate context-specific expansions based on detected domain and question type.
        """
        expansions = []
        
        # Detect question types and add relevant expansions
        
        # Time-related questions (periods, dates, duration)
        if any(term in question_lower for term in ['period', 'time', 'duration', 'when', 'how long', 'days', 'months', 'years']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} period",
                    f"{word} duration",
                    f"{word} time",
                    f"{word} timeline",
                    f"time for {word}",
                    f"duration of {word}"
                ])
        
        # Coverage/inclusion questions
        if any(term in question_lower for term in ['cover', 'include', 'benefit', 'eligible', 'qualify']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} coverage",
                    f"{word} benefits",
                    f"{word} included",
                    f"covered {word}",
                    f"{word} eligible",
                    f"{word} qualification"
                ])
        
        # Exclusion/limitation questions  
        if any(term in question_lower for term in ['exclude', 'not cover', 'limitation', 'restrict']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} exclusion",
                    f"{word} limitation",
                    f"{word} restriction",
                    f"excluded {word}",
                    f"{word} not covered",
                    f"{word} restrictions"
                ])
        
        # Cost/amount questions
        if any(term in question_lower for term in ['cost', 'price', 'amount', 'fee', 'charge', 'pay']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} cost",
                    f"{word} price",
                    f"{word} amount",
                    f"{word} fee",
                    f"cost of {word}",
                    f"amount for {word}"
                ])
        
        # Process/procedure questions
        if any(term in question_lower for term in ['how', 'process', 'procedure', 'step', 'method']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} process",
                    f"{word} procedure",
                    f"{word} method",
                    f"how to {word}",
                    f"{word} steps",
                    f"process for {word}"
                ])
        
        # Requirements/conditions questions
        if any(term in question_lower for term in ['require', 'condition', 'criteria', 'need', 'must']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} requirements",
                    f"{word} conditions",
                    f"{word} criteria",
                    f"required {word}",
                    f"{word} needed",
                    f"conditions for {word}"
                ])
        
        return expansions

    def semantic_context_retrieval(self, questions: List[str]) -> str:
        """
        Advanced semantic context retrieval using sentence transformers
        Uses cosine similarity for semantic matching
        """
        try:
            print("🧠 Performing advanced semantic context retrieval...")
            
            # Load chunks if not already loaded
            if not self.load_document_chunks():
                return ""
            
            # If semantic model not available, fall back to keyword search
            if not self.semantic_model or self.chunk_embeddings is None:
                print("⚠️  Semantic model not available, using keyword search")
                return self.fast_context_retrieval(questions)
            
            # Combine all questions into search queries
            search_queries = questions.copy()
            
            # Add enhanced topic queries for better coverage
            topic_queries = self.generate_enhanced_topic_queries(questions)
            search_queries.extend(topic_queries[:50])  # Add top 50 topic queries
            
            print(f"🔍 Performing semantic search with {len(search_queries)} queries...")
            
            # Encode search queries
            query_embeddings = self.semantic_model.encode(
                search_queries, 
                convert_to_tensor=False
            )
            
            # Calculate semantic similarity scores
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute similarities between queries and all chunks
            similarities = cosine_similarity(query_embeddings, self.chunk_embeddings)
            
            # Aggregate similarity scores for each chunk
            chunk_scores = []
            for i, chunk in enumerate(self.document_chunks):
                # Get maximum similarity across all queries for this chunk
                max_similarity = np.max(similarities[:, i])
                # Get average similarity for this chunk
                avg_similarity = np.mean(similarities[:, i])
                # Combine max and average for final score
                final_score = (max_similarity * 0.7) + (avg_similarity * 0.3)
                
                # Apply keyword boosting for important terms
                text_lower = chunk['text_lower']
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
                
                final_score += min(keyword_boost, 0.5)  # Cap keyword boost
                
                if final_score > 0.1:  # Only include reasonably similar chunks
                    chunk_scores.append({
                        'index': i,
                        'score': final_score,
                        'text': chunk['text'],
                        'semantic_score': max_similarity,
                        'avg_score': avg_similarity
                    })
            
            # Sort by semantic similarity score
            chunk_scores.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"📊 Found {len(chunk_scores)} semantically relevant chunks")
            if chunk_scores:
                print(f"🎯 Top score: {chunk_scores[0]['score']:.3f}, Bottom score: {chunk_scores[-1]['score']:.3f}")
            
            # Select diverse top chunks
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
                    
                    if len(selected_chunks) >= 35:  # Get more chunks for better coverage
                        break
            
            # If we don't have enough high-quality semantic matches, supplement with keyword search
            if len(selected_chunks) < 15:
                print("🔄 Supplementing semantic search with keyword search...")
                keyword_context = self.fast_context_retrieval(questions)
                if keyword_context:
                    # Mix semantic and keyword results
                    semantic_context = "\n\n--- DOCUMENT SECTION ---\n\n".join([chunk['text'] for chunk in selected_chunks])
                    return f"{semantic_context}\n\n--- DOCUMENT SECTION ---\n\n{keyword_context}"
            
            context_texts = [chunk['text'] for chunk in selected_chunks]
            
            print(f"✅ Selected {len(context_texts)} diverse semantic chunks")
            if selected_chunks:
                print(f"📈 Semantic score range: {selected_chunks[0]['semantic_score']:.3f} - {selected_chunks[-1]['semantic_score']:.3f}")
            
            if not context_texts:
                print("⚠️ No semantically relevant context found")
                return ""
            
            return "\n\n--- DOCUMENT SECTION ---\n\n".join(context_texts)
            
        except Exception as e:
            print(f"❌ Error in semantic context retrieval: {e}")
            print("🔄 Falling back to keyword search")
            return self.fast_context_retrieval(questions)

    def fast_context_retrieval(self, questions: List[str]) -> str:
        """
        Ultra-fast context retrieval using advanced in-memory search with semantic understanding
        Uses fuzzy matching and enhanced scoring for better results
        """
        try:
            print("⚡ Performing ultra-fast semantic in-memory context retrieval...")
            
            # Load chunks if not already loaded
            if not self.load_document_chunks():
                return ""
            
            # Generate enhanced topic queries for comprehensive search
            topic_queries = self.generate_enhanced_topic_queries(questions)
            
            # Extract key terms from questions and topic queries for searching
            search_terms = set()
            
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
            chunk_scores = []
            for i, chunk in enumerate(self.document_chunks):
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
            
            # Select top relevant chunks with diversity
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
            if len(selected_chunks) < 15:  # Increased threshold for more comprehensive search
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
                for i, chunk in enumerate(self.document_chunks):
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

    def get_relevant_context(self, questions: List[str]) -> str:
        """
        Use advanced semantic context retrieval with sentence transformers fallback to keyword search
        """
        return self.semantic_context_retrieval(questions)

    def answer_questions_batch(self, questions: List[str]) -> List[str]:
        """
        Answer multiple questions using efficient semantic search + Gemini answering.
        Replaces the two-stage LLM approach with direct vector retrieval.
        """
        try:
            print(f"📝 Processing {len(questions)} questions with ultra-fast retrieval + LLM answering...")
            
            # Step 1: Get relevant context using ultra-fast in-memory retrieval
            relevant_context = self.get_relevant_context(questions)
            
            if not relevant_context:
                print("⚠️  No relevant context available")
                return ["No relevant information found in the document to answer this question."] * len(questions)
            
            # Step 2: Create questions list
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            print(f"🤖 Processing with Gemini using ultra-fast retrieved context...")
            
            # Step 3: Single-stage LLM processing with high-quality context
            qa_prompt = f"""You are an expert insurance policy analyst. Answer each question based strictly on the relevant policy context provided below.

CONTEXT FROM POLICY DOCUMENTS:
{relevant_context}

QUESTIONS TO ANSWER:
{questions_text}

IMPORTANT INSTRUCTIONS:
- Answer each question in exactly ONE paragraph only
- Use strictly plain text format with NO formatting, NO markdown, NO bullet points
- Keep answers concise but complete
- If information is not in the context, say "Information not available in the provided policy documents"
- Format as: "1. [Single paragraph answer]", "2. [Single paragraph answer]", etc.

Provide concise, numbered single-paragraph answers in plain text only:"""

            print("🎯 Generating comprehensive answers with Gemini...")
            
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=qa_prompt
            )
            
            print("✅ Gemini response received, parsing answers...")
            
            batch_response = response.text.strip()
            
            # Parse the response into individual answers
            import re
            
            # Strategy 1: Split by numbered answers
            answers = []
            parts = re.split(r'\n(?=\d+\.)', batch_response)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                # Remove the number prefix (1., 2., etc.)
                clean_answer = re.sub(r'^\d+\.\s*', '', part).strip()
                
                # Handle multi-line answers by preserving structure but cleaning whitespace
                clean_answer = ' '.join(clean_answer.split())
                
                if clean_answer:
                    answers.append(clean_answer)
            
            # Fallback parsing if numbered format fails
            if len(answers) != len(questions):
                print(f"⚠️  Primary parsing yielded {len(answers)} answers, expected {len(questions)}. Trying fallback...")
                
                # Alternative parsing approach
                answers = []
                lines = batch_response.split('\n')
                current_answer = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if re.match(r'^\d+\.', line):
                        if current_answer:
                            clean_answer = current_answer.strip()
                            clean_answer = re.sub(r'^\d+\.\s*', '', clean_answer).strip()
                            clean_answer = ' '.join(clean_answer.split())
                            answers.append(clean_answer)
                        current_answer = line
                    else:
                        current_answer += " " + line
                
                # Add the last answer
                if current_answer:
                    clean_answer = current_answer.strip()
                    clean_answer = re.sub(r'^\d+\.\s*', '', clean_answer).strip()
                    clean_answer = ' '.join(clean_answer.split())
                    answers.append(clean_answer)
            
            # Ensure we have the right number of answers
            while len(answers) < len(questions):
                answers.append("Unable to find specific information for this question in the provided policy context.")
            
            # Trim if we have too many answers
            answers = answers[:len(questions)]
            
            print(f"✅ All {len(questions)} questions processed with ultra-fast retrieval + LLM approach")
            return answers
            
        except Exception as e:
            print(f"❌ Error in ultra-fast processing: {e}")
            return ["An error occurred while processing this question."] * len(questions)
    
    def process_hackathon_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete hackathon request format
        
        Args:
            request_data (dict): {"documents": "URL", "questions": [...]}
            
        Returns:
            dict: {"answers": [...]}
        """
        try:
            start_time = time.time()
            
            print("🚀 Processing Hackathon Request with Ultra-Fast System")
            print("="*50)
            
            # Validate input format
            if 'documents' not in request_data or 'questions' not in request_data:
                return {
                    "error": "Invalid request format. Required: {'documents': 'URL', 'questions': [...]}"
                }
            
            document_url = request_data['documents']
            questions = request_data['questions']
            
            print(f"📄 Document URL: {document_url}")
            print(f"❓ Questions: {len(questions)}")
            
            # Step 1: Process and store document (with timing)
            doc_start_time = time.time()
            success = self.process_and_store_document(document_url)
            if not success:
                return {
                    "error": "Failed to process document from URL"
                }
            doc_processing_time = time.time() - doc_start_time
            
            # Step 2: Answer all questions (with timing)
            qa_start_time = time.time()
            print(f"\n🎯 Answering {len(questions)} questions with ultra-fast system...")
            answers = self.answer_questions_batch(questions)
            qa_processing_time = time.time() - qa_start_time
            
            # Step 3: Prepare response
            processing_time = time.time() - start_time
            
            response = {
                "answers": answers
            }
            
            print(f"\n🎉 REQUEST COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"⏱️  TOTAL PROCESSING TIME: {processing_time:.2f} seconds")
            print(f"📄 Document processing: {doc_processing_time:.2f}s ({doc_processing_time/processing_time*100:.1f}%)")
            print(f"🤖 Q&A processing: {qa_processing_time:.2f}s ({qa_processing_time/processing_time*100:.1f}%)")
            if answers:
                print(f"📊 Questions processed: {len(answers)}")
                print(f"⚡ Average time per question: {qa_processing_time/len(questions):.2f}s")
                print(f"🚀 Processing speed: {len(questions)/processing_time:.1f} questions/second")
                if doc_processing_time < 5:  # If using cache
                    print(f"🎯 Q&A-only speed: {len(questions)/qa_processing_time:.1f} questions/second")
            print("="*60)
            
            return response
            
        except Exception as e:
            print(f"❌ Error processing hackathon request: {e}")
            return {
                "error": f"Processing failed: {str(e)}"
            }

def test_hackathon_qa_system():
    """Test the Q&A system with hackathon sample data"""
    
    test_start_time = time.time()
    
    print("🧪 TESTING ULTRA-FAST HACKATHON Q&A SYSTEM")
    print("="*60)
    
    # Initialize Q&A system
    qa_system = HackathonQASystem(
        index_name="hackathon-qa-test",
        namespace="test-docs"
    )
    
    # Sample hackathon request
    hackathon_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    # Process request
    result = qa_system.process_hackathon_request(hackathon_request)
    
    # Calculate total test time
    total_test_time = time.time() - test_start_time
    
    # Display results
    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
    else:
        print("\n🎉 TEST SUCCESSFUL!")
        print("="*60)
        print("📋 ANSWERS:")
        for i, answer in enumerate(result['answers'], 1):
            print(f"\n{i}. Q: {hackathon_request['questions'][i-1]}")
            print(f"   A: {answer}")
        
        # Validate JSON format
        try:
            json_output = json.dumps(result, indent=2)
            print(f"\n✅ Valid JSON output generated")
            print(f"📊 Response size: {len(json_output)} characters")
        except Exception as e:
            print(f"❌ JSON serialization error: {e}")
    
    # Display total test execution time
    print(f"\n🕒 TOTAL TEST EXECUTION TIME: {total_test_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    test_hackathon_qa_system()
