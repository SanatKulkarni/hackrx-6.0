"""
Query Generator
Generates enhanced topic queries for better context retrieval
"""

from typing import List
from .config import Config

class QueryGenerator:
    """Generate enhanced topic queries for semantic search"""
    
    def __init__(self):
        self.synonym_dict = self._build_synonym_dictionary()
    
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
            
            # Extract key terms
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
            key_terms.extend(self._create_term_variations(meaningful_words))
            
            # Create word combinations
            key_terms.extend(self._create_word_combinations(meaningful_words))
            
            # Add domain-specific expansions based on detected context
            context_expansions = self._generate_context_expansions(question_lower, meaningful_words)
            key_terms.extend(context_expansions)
            
            # Add all key terms to topic queries
            topic_queries.extend(key_terms)
        
        # Remove duplicates while preserving order and filter out very short terms
        unique_queries = self._deduplicate_and_filter(topic_queries)
        
        print(f"🎯 Generated {len(unique_queries)} enhanced topic queries with synonyms and variants")
        return unique_queries
    
    def _build_synonym_dictionary(self):
        """Build domain-specific synonym dictionary"""
        return {
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
    
    def _generate_synonyms_and_variants(self, meaningful_words: List[str]) -> List[str]:
        """Generate synonyms and variants for better semantic matching"""
        synonyms = []
        
        for word in meaningful_words:
            if word in self.synonym_dict:
                synonyms.extend(self.synonym_dict[word])
            
            # Add partial matches for compound terms
            for key, values in self.synonym_dict.items():
                if word in key or key in word:
                    synonyms.extend(values)
        
        return synonyms
    
    def _create_term_variations(self, meaningful_words: List[str]) -> List[str]:
        """Create variations of terms for document search"""
        variations = []
        
        for word in meaningful_words:
            # Add the word itself
            variations.append(word)
            
            # Add common variations for document search
            if len(word) > 4:
                variations.append(f"{word}s")  # plural
                if word.endswith('y'):
                    variations.append(f"{word[:-1]}ies")  # e.g., policy -> policies
                if word.endswith('e'):
                    variations.append(f"{word}s")  # e.g., service -> services
            
            # Add contextual phrases for the word
            variations.extend([
                f"{word} details",
                f"{word} information",
                f"{word} conditions",
                f"{word} terms",
                f"{word} benefits",
                f"{word} coverage"
            ])
        
        return variations
    
    def _create_word_combinations(self, meaningful_words: List[str]) -> List[str]:
        """Create 2-word and 3-word combinations from meaningful words"""
        combinations = []
        
        for i, word1 in enumerate(meaningful_words):
            # 2-word combinations
            for j, word2 in enumerate(meaningful_words[i+1:], i+1):
                combinations.append(f"{word1} {word2}")
                combinations.append(f"{word2} {word1}")  # reverse order too
            
            # 3-word combinations for very specific searches
            for j, word2 in enumerate(meaningful_words[i+1:], i+1):
                for k, word3 in enumerate(meaningful_words[j+1:], j+1):
                    if len(f"{word1} {word2} {word3}") < 50:  # reasonable length
                        combinations.append(f"{word1} {word2} {word3}")
        
        return combinations
    
    def _generate_context_expansions(self, question_lower: str, meaningful_words: List[str]) -> List[str]:
        """Generate context-specific expansions based on detected domain and question type"""
        expansions = []
        
        # Time-related questions
        if any(term in question_lower for term in ['period', 'time', 'duration', 'when', 'how long', 'days', 'months', 'years']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} period", f"{word} duration", f"{word} time",
                    f"{word} timeline", f"time for {word}", f"duration of {word}"
                ])
        
        # Coverage/inclusion questions
        if any(term in question_lower for term in ['cover', 'include', 'benefit', 'eligible', 'qualify']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} coverage", f"{word} benefits", f"{word} included",
                    f"covered {word}", f"{word} eligible", f"{word} qualification"
                ])
        
        # Exclusion/limitation questions
        if any(term in question_lower for term in ['exclude', 'not cover', 'limitation', 'restrict']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} exclusion", f"{word} limitation", f"{word} restriction",
                    f"excluded {word}", f"{word} not covered", f"{word} restrictions"
                ])
        
        # Cost/amount questions
        if any(term in question_lower for term in ['cost', 'price', 'amount', 'fee', 'charge', 'pay']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} cost", f"{word} price", f"{word} amount",
                    f"{word} fee", f"cost of {word}", f"amount for {word}"
                ])
        
        # Process/procedure questions
        if any(term in question_lower for term in ['how', 'process', 'procedure', 'step', 'method']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} process", f"{word} procedure", f"{word} method",
                    f"how to {word}", f"{word} steps", f"process for {word}"
                ])
        
        # Requirements/conditions questions
        if any(term in question_lower for term in ['require', 'condition', 'criteria', 'need', 'must']):
            for word in meaningful_words:
                expansions.extend([
                    f"{word} requirements", f"{word} conditions", f"{word} criteria",
                    f"required {word}", f"{word} needed", f"conditions for {word}"
                ])
        
        return expansions
    
    def _deduplicate_and_filter(self, topic_queries: List[str]) -> List[str]:
        """Remove duplicates and filter out very short terms"""
        seen = set()
        unique_queries = []
        
        for query in topic_queries:
            clean_query = query.strip()
            if clean_query not in seen and len(clean_query) > 2:
                seen.add(clean_query)
                unique_queries.append(clean_query)
        
        # Limit to reasonable number for performance
        return unique_queries[:Config.MAX_TOPIC_QUERIES]
