"""
HackRX 6.0 API - Document Q&A System
FastAPI endpoint for processing documents and answering questions
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our Q&A system
from hackathon_qa_system import HackathonQASystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRX 6.0 Document Q&A API",
    description="Process documents from URLs and answer questions using RAG pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# API Key validation - Using the exact token from api_details.txt
API_KEY = "f640dfb77d63b79eb5904987e526bc51ae6391eb3c86f8345b1c1341c0b32b08"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Request/Response models
class HackRXRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]  # List of questions to answer

class HackRXResponse(BaseModel):
    answers: List[str]  # List of answers corresponding to questions

class ErrorResponse(BaseModel):
    error: str
    detail: str = None

# Initialize Q&A system with same configuration as testing
qa_system = HackathonQASystem(
    index_name="hackathon-qa-test",
    namespace="test-docs"
)

@app.post("/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint for HackRX 6.0 competition
    
    Process documents from URL and answer questions using RAG pipeline.
    """
    try:
        start_time = time.time()
        
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Validate request
        if not request.documents:
            raise HTTPException(
                status_code=400,
                detail="Document URL is required"
            )
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one question is required"
            )
        
        # Process the request using our Q&A system
        request_data = {
            "documents": request.documents,
            "questions": request.questions
        }
        
        result = qa_system.process_hackathon_request(request_data)
        
        # Check for errors in processing
        if "error" in result:
            logger.error(f"Processing error: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result['error']}"
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Request completed successfully in {processing_time:.2f} seconds")
        
        return HackRXResponse(answers=result["answers"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "HackRX 6.0 Document Q&A API"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HackRX 6.0 Document Q&A API",
        "version": "1.0.0",
        "endpoint": "/hackrx/run",
        "method": "POST",
        "authentication": "Bearer token required",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
