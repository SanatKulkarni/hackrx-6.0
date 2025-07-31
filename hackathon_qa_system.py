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
        self.setup_gemini()
    
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
    
    def answer_question(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate answer for a single question using Gemini
        
        Args:
            question (str): The question to answer
            context_chunks (List[str]): Relevant document chunks
            
        Returns:
            str: Generated answer
        """
        try:
            # Create context from chunks
            context = "\n\n".join(context_chunks)
            
            # Create prompt for detailed answers matching API expectations
            prompt = f"""You are an expert insurance policy analyst specializing in the National Parivar Mediclaim Plus Policy. Answer the question based ONLY on the provided context from the insurance policy document.

Context from Insurance Policy:
{context}

Question: {question}

Instructions:
1. Provide a detailed and comprehensive answer based strictly on the policy document context
2. Include specific numbers, percentages, time periods, and conditions exactly as mentioned
3. For waiting periods, grace periods, and coverage limits, provide exact values with units
4. When mentioning benefits or coverage, include any relevant conditions, exclusions, or sub-limits
5. If the context doesn't contain sufficient information, state "Information not available in the provided document"
6. Use professional insurance terminology as found in the source document

Answer:"""

            # Generate answer using Gemini
            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            answer = response.text.strip()
            return answer
            
        except Exception as e:
            print(f"❌ Error generating answer for question: {e}")
            return "An error occurred while generating the answer."
    
    def answer_questions_batch(self, questions: List[str]) -> List[str]:
        """
        Answer multiple questions using batch processing for efficiency
        
        Args:
            questions (List[str]): List of questions to answer
            
        Returns:
            List[str]: List of answers corresponding to questions
        """
        try:
            print(f"📝 Processing {len(questions)} questions in batch...")
            
            # Improve retrieval: Get more context chunks for better coverage
            print("🔍 Performing enhanced context retrieval...")
            
            # Strategy 1: Combined query for overall context
            combined_query = " ".join(questions)
            main_results = query_embeddings(
                query_text=combined_query,
                index_name=self.index_name,
                namespace=self.namespace,
                n_results=15  # Increased for better coverage
            )
            
            # Strategy 2: Individual queries for specific topics that might be missed
            individual_results = []
            for question in questions:
                try:
                    individual_result = query_embeddings(
                        query_text=question,
                        index_name=self.index_name,
                        namespace=self.namespace,
                        n_results=5  # Fewer per question but more targeted
                    )
                    if individual_result and 'matches' in individual_result:
                        individual_results.extend(individual_result['matches'])
                except Exception as e:
                    print(f"⚠️  Individual query failed for: {question[:50]}...")
                    continue
            
            # Combine all results
            all_matches = []
            if main_results and 'matches' in main_results:
                all_matches.extend(main_results['matches'])
            all_matches.extend(individual_results)
            
            
            # Extract unique context chunks from all results
            all_context_chunks = []
            seen_chunks = set()  # Track duplicates more efficiently
            
            for match in all_matches:
                if 'metadata' in match and 'text' in match['metadata']:
                    chunk_text = match['metadata']['text']
                    # Use hash for efficient duplicate detection
                    chunk_hash = hash(chunk_text)
                    if chunk_hash not in seen_chunks:
                        seen_chunks.add(chunk_hash)
                        all_context_chunks.append(chunk_text)
            
            if not all_context_chunks:
                print("⚠️  No relevant context found for any questions")
                return ["No relevant information found in the document to answer this question."] * len(questions)
            
            print(f"✅ Retrieved {len(all_context_chunks)} unique context chunks")
            
            # Create comprehensive context - increased for better coverage
            context = "\n\n".join(all_context_chunks[:20])  # Increased from 12 for better coverage
            
            # Create batch prompt with all questions
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            prompt = f"""You are an expert insurance policy analyst. Your task is to answer ALL questions based STRICTLY on the provided policy context.

POLICY CONTEXT:
{context}

QUESTIONS TO ANSWER:
{questions_text}

CRITICAL INSTRUCTIONS:
1. Read the entire context carefully before answering any question
2. For each question, search through ALL the context for relevant information
3. Provide specific, detailed answers with exact numbers, time periods, and conditions
4. Quote relevant policy terms and conditions when applicable
5. If the exact information is not explicitly stated in the context, try to infer from related information
6. ONLY use "Information not available in the provided document" as a last resort when absolutely no related information exists
7. Format your response as a numbered list (1., 2., 3., etc.) matching the question numbers
8. Each answer should be complete and self-contained

ANSWER ALL {len(questions)} QUESTIONS:"""

            print("🤖 Generating answers with Gemini...")
            
            # Generate all answers in one API call
            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            print("✅ Gemini response received, parsing answers...")
            
            # Improved answer parsing with better number detection
            batch_response = response.text.strip()
            
            # Split response into individual answers with enhanced parsing
            answers = []
            
            # Try to split by numbered answers (1., 2., 3., etc.)
            import re
            answer_pattern = r'^\d+\.\s*'
            
            # Split the response by numbered answers
            parts = re.split(r'\n(?=\d+\.)', batch_response)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                # Remove the number prefix (1., 2., etc.)
                clean_answer = re.sub(answer_pattern, '', part, count=1).strip()
                
                # Handle multi-line answers
                clean_answer = ' '.join(clean_answer.split())
                
                if clean_answer:
                    answers.append(clean_answer)
            
            # Fallback parsing if numbered format fails
            if len(answers) != len(questions):
                print(f"⚠️  Numbered parsing yielded {len(answers)} answers, expected {len(questions)}. Trying fallback...")
                
                # Try alternative parsing
                answers = []
                lines = batch_response.split('\n')
                current_answer = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if line starts with a number (new answer)
                    if re.match(r'^\d+\.', line):
                        if current_answer:
                            # Clean up the previous answer
                            clean_answer = current_answer.strip()
                            # Remove number prefix if present
                            clean_answer = re.sub(r'^\d+\.\s*', '', clean_answer).strip()
                            answers.append(clean_answer)
                        current_answer = line
                    else:
                        current_answer += " " + line
                
                # Add the last answer
                if current_answer:
                    clean_answer = current_answer.strip()
                    clean_answer = re.sub(r'^\d+\.\s*', '', clean_answer).strip()
                    answers.append(clean_answer)
            
            # Quality check: Re-process questions that got generic "not available" answers
            final_answers = []
            questions_needing_retry = []
            retry_indices = []
            
            for i, answer in enumerate(answers):
                if ("Information not available" in answer or 
                    "Unable to generate answer" in answer or 
                    len(answer.strip()) < 10):  # Very short answers might be incomplete
                    questions_needing_retry.append(questions[i])
                    retry_indices.append(i)
                    final_answers.append(answer)  # Keep original for now
                else:
                    final_answers.append(answer)
            
            # Retry individual questions that failed
            if questions_needing_retry:
                print(f"🔄 Retrying {len(questions_needing_retry)} questions individually for better results...")
                
                for idx, (retry_idx, retry_question) in enumerate(zip(retry_indices, questions_needing_retry)):
                    try:
                        # More targeted search for individual question
                        individual_result = query_embeddings(
                            query_text=retry_question,
                            index_name=self.index_name,
                            namespace=self.namespace,
                            n_results=8  # Focused search
                        )
                        
                        if individual_result and 'matches' in individual_result:
                            # Build context specific to this question
                            question_context = []
                            for match in individual_result['matches']:
                                if 'metadata' in match and 'text' in match['metadata']:
                                    question_context.append(match['metadata']['text'])
                            
                            if question_context:
                                context_text = "\n\n".join(question_context[:8])
                                
                                individual_prompt = f"""Based on the following insurance policy context, answer this specific question with detailed information:

CONTEXT:
{context_text}

QUESTION: {retry_question}

Provide a detailed answer with specific policy terms, numbers, and conditions. If the information is truly not available, explain what related information is available instead.

ANSWER:"""
                                
                                individual_response = self.gemini_client.models.generate_content(
                                    model='gemini-2.0-flash-exp',
                                    contents=individual_prompt
                                )
                                
                                improved_answer = individual_response.text.strip()
                                if improved_answer and len(improved_answer) > 20:  # Valid improved answer
                                    final_answers[retry_idx] = improved_answer
                                    print(f"✅ Improved answer for question {retry_idx + 1}")
                    
                    except Exception as e:
                        print(f"⚠️  Individual retry failed for question {retry_idx + 1}: {e}")
                        continue
            
            # Ensure we have the right number of answers
            while len(final_answers) < len(questions):
                final_answers.append("Unable to generate answer for this question.")
            
            # Trim if we have too many answers
            final_answers = final_answers[:len(questions)]
            
            print(f"✅ All {len(questions)} questions processed with quality check")
            return final_answers
            
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            # Fallback to individual processing if batch fails
            print("🔄 Falling back to individual question processing...")
            return self.answer_questions_individual(questions)
    
    def answer_questions_individual(self, questions: List[str]) -> List[str]:
        """
        Fallback method: Answer questions individually
        """
        answers = []
        
        for i, question in enumerate(questions, 1):
            print(f"📝 Processing question {i}/{len(questions)}: {question[:60]}...")
            
            try:
                # Retrieve relevant chunks for this question
                retrieval_results = query_embeddings(
                    query_text=question,
                    index_name=self.index_name,
                    namespace=self.namespace,
                    n_results=5
                )
                
                # Extract text from retrieval results
                context_chunks = []
                if retrieval_results and 'matches' in retrieval_results:
                    for match in retrieval_results['matches']:
                        if 'metadata' in match and 'text' in match['metadata']:
                            context_chunks.append(match['metadata']['text'])
                
                if not context_chunks:
                    print(f"⚠️  No relevant context found for question {i}")
                    answers.append("No relevant information found in the document to answer this question.")
                    continue
                
                # Generate answer
                answer = self.answer_question(question, context_chunks)
                answers.append(answer)
                
                print(f"✅ Question {i} answered")
                
            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                answers.append("An error occurred while processing this question.")
        
        return answers
    
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
            
            print("🚀 Processing Hackathon Request")
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
            print(f"\n🎯 Answering {len(questions)} questions...")
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
    
    print("🧪 TESTING HACKATHON Q&A SYSTEM")
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
