"""
Gemini AI Client
Handles Gemini AI model integration for answer generation
"""

import os
from google import genai
from .config import Config

class GeminiAIClient:
    """Gemini AI client for answer generation"""
    
    def __init__(self):
        self.client = None
        self.model_name = Config.GEMINI_MODEL
        self.setup_client()
    
    def setup_client(self):
        """Setup Gemini AI client"""
        try:
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
            
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

IMPORTANT INSTRUCTIONS:
- Answer each question in exactly ONE paragraph only
- Use strictly plain text format with NO formatting, NO markdown, NO bullet points
- Include specific numbers, percentages, time periods, and exact references from the policy
- Keep answers precise but complete with all relevant details
- If information is not in the context, say "Information not available in the provided policy documents"
- Format as: "1. [Single paragraph answer]", "2. [Single paragraph answer]", etc.
- Always include specific amounts, percentages, waiting periods, and conditions exactly as mentioned in the policy

Provide concise, numbered single-paragraph answers in plain text only with specific numbers and references:"""

            print("🎯 Generating comprehensive answers with Gemini AI...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=qa_prompt
            )
            
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
