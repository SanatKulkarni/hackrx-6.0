"""
Gemini AI Client
Handles Gemini AI model integration for answer generation
"""

import os
import google.generativeai as genai
from .config import Config

class GeminiAIClient:
    """Gemini AI client for answer generation"""
    
    def __init__(self):
        self.model = None
        self.model_name = Config.GEMINI_MODEL
        self.setup_client()
    
    def setup_client(self):
        """Setup Gemini AI client"""
        try:
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            # Configure the API key
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            # Initialize the model
            self.model = genai.GenerativeModel(self.model_name)
            
            print("✅ Gemini AI client initialized")
            
        except Exception as e:
            print(f"❌ Error setting up Gemini AI: {e}")
            raise
    
    def generate_answers(self, questions, context):
        """
        Generate answers for multiple questions using Gemini AI
        
        Args:
            questions (List[str]): List of questions to answer
            context (str): Relevant context from documents
            
        Returns:
            List[str]: List of answers
        """
        try:
            # Create questions text
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            # Single-stage LLM processing with high-quality context
            qa_prompt = f"""You are an expert insurance policy analyst. Answer each question based strictly on the relevant policy context provided below.

CONTEXT FROM POLICY DOCUMENTS:
{context}

QUESTIONS TO ANSWER:
{questions_text}

CRITICAL INSTRUCTIONS:
- Answer each question in exactly ONE paragraph only
- Use strictly plain text format with NO formatting, NO markdown, NO bullet points
- ALWAYS include both written numbers AND numerical values (e.g., "thirty-six (36) months", "twenty-four (24) months", "two (2) years")
- ALWAYS convert percentages to both written and numerical form (e.g., "twenty percent (20%)")
- Include ALL specific amounts, percentages, time periods, and exact references from the policy
- Search thoroughly through the entire context - information may be scattered across different sections
- If partial information is found, provide what is available and specify what is missing
- Only say "Information not available" if absolutely no relevant information exists in the context
- Format as: "1. [Single paragraph answer]", "2. [Single paragraph answer]", etc.
- Be comprehensive - include ALL relevant details, conditions, exceptions, and sub-clauses
- Always prioritize numerical accuracy and completeness

Provide numbered single-paragraph answers with ALL numerical values in both written and numeric format:"""

            print("🎯 Generating comprehensive answers with Gemini AI...")
            
            response = self.model.generate_content(qa_prompt)
            
            print("✅ Gemini AI response received, parsing answers...")
            
            return self._parse_answers(response.text.strip(), len(questions))
            
        except Exception as e:
            print(f"❌ Error in Gemini AI processing: {e}")
            return ["An error occurred while processing this question."] * len(questions)
    
    def _parse_answers(self, batch_response, expected_count):
        """Parse the batch response into individual answers"""
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
        if len(answers) != expected_count:
            print(f"⚠️  Primary parsing yielded {len(answers)} answers, expected {expected_count}. Trying fallback...")
            
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
        while len(answers) < expected_count:
            answers.append("Unable to find specific information for this question in the provided policy context.")
        
        # Trim if we have too many answers
        answers = answers[:expected_count]
        
        return answers
