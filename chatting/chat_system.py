import os
import getpass
import json
from dotenv import load_dotenv
from google import genai
from langchain_nomic import NomicEmbeddings
import chromadb
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain_community.vectorstores import Chroma
from typing import List

# Load environment variables from .env file
load_dotenv()

# Define prompt template for insurance adjudicator
prompt_template = """
You are an experienced insurance adjudicator with extensive knowledge of insurance policies and claim assessment. Your role is to evaluate insurance claims based on policy terms and claimant circumstances.

## Policy Clauses (Context):
{context}

## Claimant Facts (Question):
{question}

## Instructions:
Follow these steps to assess the claim:

1. **Policy Analysis**: Carefully review the policy clauses provided in the context section above.

2. **Claim Evaluation**: Analyze the claimant facts against the policy terms to determine coverage eligibility.

3. **Coverage Determination**: Assess whether the claim falls within the scope of coverage, considering:
   - Policy limits and deductibles
   - Exclusions and limitations
   - Pre-existing conditions
   - Waiting periods
   - Geographic coverage areas

4. **Amount Calculation**: If coverage applies, calculate the payable amount based on:
   - Policy benefits and limits
   - Applicable deductibles
   - Co-payment requirements
   - Maximum coverage amounts

5. **Decision Justification**: Provide clear reasoning for your decision, referencing specific policy clauses.

## Critical Output Requirements:
Do not output any text, markdown, or explanations before or after the JSON block. Your entire response must be the JSON object itself.

You must format your response as a single, valid JSON object with exactly this structure:

{{
  "decision": "APPROVED" | "DENIED" | "PARTIALLY_APPROVED",
  "amount_payable": <numeric_value>,
  "justification": [
    {{
      "step": "Policy Analysis",
      "finding": "description of policy review findings",
      "impact": "how this affects the claim decision"
    }},
    {{
      "step": "Coverage Assessment",
      "finding": "description of coverage determination",
      "impact": "how this affects the claim decision"
    }},
    {{
      "step": "Amount Calculation",
      "finding": "description of amount calculation process",
      "impact": "how this affects the final payout"
    }}
  ]
}}

Remember: Your entire response must be only the JSON object above. No additional text, explanations, or markdown formatting.
"""

# Configure Google API key from environment
def configure_google_api():
    """Configure Google Generative AI API key"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment variables.")
        api_key = getpass.getpass("Enter your Google Generative AI API key: ")
        os.environ["GEMINI_API_KEY"] = api_key
    
    # Initialize Google Generative AI client
    client = genai.Client(api_key=api_key)
    return client

# Initialize Nomic embeddings (same as in embeddings/__init__.py)
def initialize_nomic_embeddings():
    """Initialize Nomic embeddings with same configuration as embeddings folder"""
    # Check for Nomic API key
    if not os.getenv("NOMIC_API_KEY"):
        print("Nomic API key not found. Please get one from https://atlas.nomic.ai/")
        os.environ["NOMIC_API_KEY"] = getpass.getpass("Enter your Nomic API key: ")
    
    # Initialize Nomic embeddings with same parameters as embeddings/__init__.py
    embeddings = NomicEmbeddings(
        model="nomic-embed-text-v1.5",
        dimensionality=768,  
        inference_mode="remote",  
    )
    
    return embeddings

# Load existing Chroma vector database
def load_chroma_db(embeddings, db_path="../database/"):
    """
    Load existing Chroma vector database from persistent directory
    
    Args:
        embeddings: Initialized embeddings model
        db_path (str): Path to the persistent vector database directory
    
    Returns:
        chromadb.Collection: Chroma collection object
    """
    try:
        # Create the database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize Chroma client with persistent storage
        chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Try to get existing collection, create if it doesn't exist
        try:
            collection = chroma_client.get_collection(name="policy_documents")
            print(f"Loaded existing collection 'policy_documents' from {db_path}")
        except:
            # If collection doesn't exist, create it
            collection = chroma_client.create_collection(name="policy_documents")
            print(f"Created new collection 'policy_documents' in {db_path}")
        
        return collection
        
    except Exception as e:
        print(f"Error loading Chroma database: {e}")
        return None

def create_retriever(collection, embeddings, k=5):
    """
    Create a retriever from the loaded Chroma vector store using LangChain's built-in Chroma
    
    Args:
        collection: Chroma collection object
        embeddings: Initialized embeddings model
        k (int): Number of top relevant documents to retrieve (default: 5)
    
    Returns:
        retriever: Configured retriever object
    """
    try:
        # Check if collection has any documents
        collection_count = collection.count()
        print(f"Collection contains {collection_count} documents")
        
        if collection_count == 0:
            print("⚠️  Warning: Collection is empty. Please add documents to the database first.")
            print("You can add documents using the embedding and loading scripts in your project.")
        
        # Use LangChain's Chroma vectorstore
        vectorstore = Chroma(
            collection_name="policy_documents",
            embedding_function=embeddings,
            persist_directory="../database/"
        )
        
        # Create retriever from vectorstore
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        print(f"✅ Retriever created successfully using LangChain Chroma (k={k})")
        return retriever
        
    except Exception as e:
        print(f"Error creating retriever: {e}")
        print("Attempting fallback method...")
        
        # Fallback: Create a simple custom retriever
        try:
            class SimpleChromaRetriever(BaseRetriever):
                def __init__(self, collection, embeddings, k=5):
                    super().__init__()
                    self.collection = collection
                    self.embeddings = embeddings
                    self.k = k
                
                def _get_relevant_documents(self, query: str) -> List[Document]:
                    """Retrieve relevant documents for a given query"""
                    try:
                        # Generate embedding for the query
                        query_embedding = self.embeddings.embed_query(query)
                        
                        # Search the collection
                        results = self.collection.query(
                            query_embeddings=[query_embedding],
                            n_results=self.k
                        )
                        
                        # Convert results to Document objects
                        documents = []
                        if results.get('documents') and results['documents'][0]:
                            for i, doc in enumerate(results['documents'][0]):
                                metadata = {
                                    'id': results['ids'][0][i] if results.get('ids') else f"doc_{i}",
                                    'distance': results['distances'][0][i] if results.get('distances') else None
                                }
                                documents.append(Document(page_content=doc, metadata=metadata))
                        
                        return documents
                        
                    except Exception as e:
                        print(f"Error in retrieval: {e}")
                        return []
            
            # Create fallback retriever
            fallback_retriever = SimpleChromaRetriever(collection, embeddings, k)
            print(f"✅ Fallback retriever created successfully (k={k})")
            return fallback_retriever
            
        except Exception as fallback_error:
            print(f"Fallback retriever also failed: {fallback_error}")
            return None

def setup_retrieval_qa_chain(retriever, api_key):
    """
    Set up a RetrievalQA chain using LangChain with Gemini LLM
    
    Args:
        retriever: Configured retriever object
        api_key (str): Google API key for Gemini
    
    Returns:
        RetrievalQA: Configured RetrievalQA chain
    """
    try:
        # Initialize ChatGoogleGenerativeAI model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
            google_api_key=api_key
        )
        print("✅ ChatGoogleGenerativeAI model initialized")
        
        # Create PromptTemplate object from the prompt_template string
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        print("✅ PromptTemplate created from prompt_template")
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        print("✅ RetrievalQA chain created successfully")
        
        return qa_chain
        
    except Exception as e:
        print(f"Error setting up RetrievalQA chain: {e}")
        return None

def query_insurance_claim(qa_chain, question):
    """
    Query the insurance claim using the RetrievalQA chain
    
    Args:
        qa_chain: Configured RetrievalQA chain
        question (str): User question about insurance claim
    
    Returns:
        dict: Response containing result and source documents
    """
    try:
        # Query the chain
        response = qa_chain.invoke({"query": question})
        
        # Extract result and source documents
        result = {
            "answer": response.get("result", ""),
            "source_documents": response.get("source_documents", [])
        }
        
        return result
        
    except Exception as e:
        print(f"Error querying insurance claim: {e}")
        return {"answer": "", "source_documents": []}

def main():
    """Main function to initialize all components"""
    print("=== Chat System Initialization ===")
    
    # Step 1: Configure Google Generative AI
    print("1. Configuring Google Generative AI...")
    genai_client = configure_google_api()
    print("✅ Google Generative AI configured successfully")
    
    # Step 2: Initialize Nomic embeddings
    print("2. Initializing Nomic embeddings...")
    embeddings = initialize_nomic_embeddings()
    print("✅ Nomic embeddings initialized successfully")
    
    # Step 3: Load Chroma vector database
    print("3. Loading Chroma vector database...")
    collection = load_chroma_db(embeddings)
    if collection:
        print("✅ Chroma vector database loaded successfully")
    else:
        print("❌ Failed to load Chroma vector database")
        return
    
    # Step 4: Create retriever from vector store
    print("4. Creating retriever from vector store...")
    retriever = create_retriever(collection, embeddings, k=5)
    if retriever:
        print("✅ Retriever created successfully")
        # Test the retriever
        try:
            test_docs = retriever.get_relevant_documents("test query")
            print(f"✅ Retriever test successful - found {len(test_docs)} documents")
        except Exception as e:
            print(f"⚠️  Retriever test failed: {e}")
    else:
        print("❌ Failed to create retriever")
        return
    
    # Step 5: Set up RetrievalQA chain
    print("5. Setting up RetrievalQA chain...")
    api_key = os.getenv("GEMINI_API_KEY")
    qa_chain = setup_retrieval_qa_chain(retriever, api_key)
    if qa_chain:
        print("✅ RetrievalQA chain set up successfully")
    else:
        print("❌ Failed to set up RetrievalQA chain")
        return
    
    # Step 6: Placeholder for user query and demonstration
    print("\n=== Ready for Chat ===")
    user_query = "What are the coverage details for knee surgery in Pune?"  # Placeholder query
    
    print(f"Placeholder user query: {user_query}")
    print("\n--- System Components Ready ---")
    print("✅ Google Generative AI Client:", type(genai_client).__name__)
    print("✅ Nomic Embeddings Model:", type(embeddings).__name__)
    print("✅ Chroma Collection:", type(collection).__name__)
    print("✅ Retriever:", type(retriever).__name__)
    print("✅ RetrievalQA Chain:", type(qa_chain).__name__)
    
    # Demonstrate retriever functionality
    print("\n=== Testing Retriever ===")
    try:
        relevant_docs = retriever.get_relevant_documents(user_query)
        print(f"Retrieved {len(relevant_docs)} relevant documents:")
        
        for i, doc in enumerate(relevant_docs[:3]):  # Show first 3 documents
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"Error testing retriever: {e}")
        print("Note: This is expected if the database is empty. Add some documents first.")
    
    # Demonstrate RetrievalQA chain functionality
    print("\n=== Testing RetrievalQA Chain ===")
    try:
        # Test the complete RAG pipeline
        claim_question = "46-year-old male needs knee surgery in Pune with 3-month-old insurance policy"
        print(f"Testing claim question: {claim_question}")
        
        response = query_insurance_claim(qa_chain, claim_question)
        
        print(f"\nRetrievalQA Response:")
        print(f"Answer: {response['answer'][:500]}...")
        print(f"Source Documents Used: {len(response['source_documents'])}")
        
        # Show source documents
        for i, doc in enumerate(response['source_documents'][:2]):
            print(f"\nSource Document {i+1}:")
            print(f"Content: {doc.page_content[:150]}...")
            
    except Exception as e:
        print(f"Error testing RetrievalQA chain: {e}")
        print("Note: This is expected if the database is empty or API keys are not configured.")
    
    # TODO: Add actual chat functionality here
    print("\n🚀 Complete RAG Chat system is ready!")
    print("You can now:")
    print("- Process insurance claims with semantic search")
    print("- Generate structured JSON responses using Gemini")
    print("- Access source documents for transparency")

if __name__ == "__main__":
    main()
    
    # Final demonstration: Process a specific insurance claim
    print("\n" + "="*60)
    print("🏥 INSURANCE CLAIM PROCESSING DEMONSTRATION")
    print("="*60)
    
    try:
        # Step 1: Initialize all components
        print("\n1. Initializing system components...")
        
        # Configure API keys
        genai_client = configure_google_api()
        embeddings = initialize_nomic_embeddings()
        collection = load_chroma_db(embeddings)
        retriever = create_retriever(collection, embeddings, k=5)
        api_key = os.getenv("GEMINI_API_KEY")
        qa_chain = setup_retrieval_qa_chain(retriever, api_key)
        
        if not qa_chain:
            print("❌ Failed to initialize RetrievalQA chain. Exiting.")
            exit(1)
        
        print("✅ All components initialized successfully")
        
        # Step 2: Define user query
        query = "46M, knee surgery, Pune, 3-month policy"
        print(f"\n2. Processing insurance claim query: '{query}'")
        
        # Step 3: Invoke RetrievalQA chain
        print("\n3. Invoking RetrievalQA chain...")
        result_dict = qa_chain.invoke({"query": query})
        
        # Step 4: Extract result from dictionary
        print("\n4. Extracting result from chain response...")
        if "result" not in result_dict:
            print("❌ No 'result' key found in chain response")
            print("Available keys:", list(result_dict.keys()))
            exit(1)
        
        gemini_response = result_dict["result"]
        print(f"✅ Raw Gemini response extracted (length: {len(gemini_response)} characters)")
        
        # Step 5: Parse JSON response
        print("\n5. Parsing JSON response...")
        try:
            # Clean the response (remove any potential markdown formatting)
            cleaned_response = gemini_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "").strip()
            
            # Parse JSON
            parsed_result = json.loads(cleaned_response)
            
            # Success! Pretty-print the result
            print("✅ JSON parsing successful!")
            print("\n" + "="*50)
            print("📋 INSURANCE CLAIM DECISION")
            print("="*50)
            print(json.dumps(parsed_result, indent=2, ensure_ascii=False))
            print("="*50)
            
            # Display source documents if available
            if "source_documents" in result_dict and result_dict["source_documents"]:
                print(f"\n📚 Source Documents Used: {len(result_dict['source_documents'])}")
                for i, doc in enumerate(result_dict["source_documents"][:3]):
                    print(f"\nSource {i+1}:")
                    print(f"  Content: {doc.page_content[:200]}...")
                    print(f"  Metadata: {doc.metadata}")
            
        except json.JSONDecodeError as e:
            print("❌ JSON parsing failed!")
            print(f"Error: {e}")
            print("\n🔍 Raw Gemini output for debugging:")
            print("-" * 60)
            print(gemini_response)
            print("-" * 60)
            print("\nPossible issues:")
            print("- Gemini may have included extra text before/after JSON")
            print("- JSON structure might be malformed")
            print("- Check if the prompt template is working correctly")
            
        except Exception as e:
            print(f"❌ Unexpected error during JSON parsing: {e}")
            print("\n🔍 Raw Gemini output for debugging:")
            print("-" * 60)
            print(gemini_response)
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error during claim processing: {e}")
        print("Please check your API keys and database configuration.")
        
    print("\n🎉 Insurance claim processing demonstration complete!")