"""
Gemini AI Client
Handles Gemini AI model integration with Google Search grounding for answer generation
"""

import os
import google.generativeai as genai
from google import genai as genai_client
from google.genai import types
from .config import Config

class GeminiAIClient:
    """Gemini AI client for answer generation with Google Search grounding"""
    
    def __init__(self):
        self.model = None
        self.client = None
        self.model_name = Config.GEMINI_MODEL
        self.setup_client()
    
    def setup_client(self):
        """Setup Gemini AI client with Google Search grounding"""
        try:
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            # Configure the API key for both clients
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            # Initialize the standard model for document-based answers
            self.model = genai.GenerativeModel(self.model_name)
            
            # Initialize the client for grounded search
            self.client = genai_client.Client()
            
            print("✅ Gemini AI client with Google Search grounding initialized")
            
        except Exception as e:
            print(f"❌ Error setting up Gemini AI: {e}")
            raise
    
    def generate_answers(self, questions, context):
        """
        Generate answers for multiple questions using Gemini AI with Google Search fallback
        
        Args:
            questions (List[str]): List of questions to answer
            context (str): Relevant context from documents
            
        Returns:
            List[str]: List of answers
        """
        try:
            # First, try to answer using the document context
            document_answers = self._generate_document_based_answers(questions, context)
            
            # Check which answers need grounded search
            final_answers = []
            for i, answer in enumerate(document_answers):
                if "Unable to find specific information" in answer or len(answer.strip()) < 20:
                    print(f"🔍 Question {i+1} needs grounded search - searching online...")
                    grounded_answer = self._generate_grounded_answer(questions[i], context)
                    final_answers.append(grounded_answer)
                else:
                    final_answers.append(answer)
            
            return final_answers
            
        except Exception as e:
            print(f"❌ Error in Gemini AI processing: {e}")
            return ["An error occurred while processing this question."] * len(questions)
    
    def _generate_document_based_answers(self, questions, context):
        """Generate answers using only the document context"""
        try:
            # Create questions text
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            # Document-based prompt
            qa_prompt = f"""You are an expert insurance policy analyst. Answer each question based strictly on the relevant policy context provided below.

CONTEXT FROM POLICY DOCUMENTS:
{context}

QUESTIONS TO ANSWER:
{questions_text}

CRITICAL INSTRUCTIONS:
- Answer each question in 1-3 sentences with key details and conditions
- Use strictly plain text format with NO formatting, NO markdown, NO bullet points
- For key numerical values, use the format that appears in the context (may be numeric only or written+numeric)
- Include essential details like eligibility criteria, limits, conditions, and specific requirements
- For yes/no questions, start with "Yes" or "No" then provide the supporting details
- Include relevant sub-limits, time periods, and important conditions from the policy
- Be specific about amounts, percentages, waiting periods, and coverage limits
- Format as: "1. [Detailed answer with key conditions]", "2. [Detailed answer with key conditions]", etc.
- Keep answers focused but comprehensive - include the main answer plus important conditions/limits
- Match the style of clear, professional insurance policy explanations
- If information is not clearly available in the context, respond with "Unable to find specific information for this question in the provided policy context."

Provide numbered detailed answers with key conditions and numerical values:"""

            print("🎯 Generating document-based answers with Gemini AI...")
            
            response = self.model.generate_content(qa_prompt)
            
            print("✅ Document-based answers received, parsing...")
            
            return self._parse_answers(response.text.strip(), len(questions))
            
        except Exception as e:
            print(f"❌ Error in document-based processing: {e}")
            return ["Unable to find specific information for this question."] * len(questions)
    
    def _generate_grounded_answer(self, question, context):
        """Generate answer using Google Search grounding for missing information"""
        try:
            # Extract insurance company/policy info from context for better search
            context_snippet = context[:500] if context else ""
            
            # Extract insurance company name from context
            insurance_company = self._extract_insurance_company(context)
            
            # Define the grounding tool
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            # Configure generation settings
            config = types.GenerateContentConfig(
                tools=[grounding_tool]
            )
            
            # Enhanced prompt for grounded search with company context
            search_context = f"Insurance Company: {insurance_company}\n" if insurance_company else ""
            
            grounded_prompt = f"""You are an expert insurance policy analyst. Answer the following question about insurance policies using online search to provide definitive, accurate information.

{search_context}QUESTION: {question}

INSTRUCTIONS:
- Use Google Search to find current, accurate information about insurance policies and practices
- {f"Focus specifically on {insurance_company} policies and practices when searching" if insurance_company else "Focus on general insurance industry standards and practices"}
- Search for current insurance regulations, industry standards, and typical policy terms
- Provide a clear, definitive answer in 1-3 sentences
- Include specific numbers, percentages, or time periods when available
- If this is a yes/no question, start with "Yes" or "No"
- Be authoritative and confident - avoid phrases like "typically", "usually", "may vary", "policy context does not specify"
- State facts directly based on your search results
- Maintain professional insurance policy explanation style

Answer the question with authority and specificity:"""
            
            print(f"🌐 Performing Google Search for: {question[:50]}...")
            if insurance_company:
                print(f"🏢 Targeting {insurance_company} specific information...")
            
            # Make the grounded request
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=grounded_prompt,
                config=config,
            )
            
            answer = response.text.strip()
            print(f"✅ Grounded answer received: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            print(f"❌ Error in grounded search: {e}")
            return f"Based on general insurance practices, this information may vary by policy. Please refer to your specific policy document or contact your insurance provider for accurate details."
    
    def _extract_insurance_company(self, context):
        """Extract insurance company name from context for better search targeting"""
        if not context:
            return None
            
        import re
        
        # Common insurance company patterns
        company_patterns = [
            r'([A-Z][a-z]+ (?:Insurance|Life|General|Health|Assurance|Mediclaim))',
            r'((?:National|Star|ICICI|HDFC|Bajaj|Reliance|Max|Care|Apollo) [A-Z][a-z]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+ (?:Insurance|Life|General|Health|Assurance))',
            r'(New India Assurance)',
            r'(Life Insurance Corporation)',
            r'(LIC)',
            r'(Star Health)',
            r'(Max Bupa)',
            r'(Care Health)',
            r'(Apollo Munich)',
            r'(Religare Health)',
            r'(Oriental Insurance)',
            r'(United India Insurance)',
            r'(National Insurance)',
        ]
        
        # Search for company names in context
        context_lower = context.lower()
        context_snippet = context[:1000]  # Check first 1000 chars for company info
        
        for pattern in company_patterns:
            matches = re.findall(pattern, context_snippet, re.IGNORECASE)
            if matches:
                company_name = matches[0].strip()
                print(f"🏢 Detected insurance company: {company_name}")
                return company_name
        
        # Look for specific keywords that might indicate company
        if "national parivar mediclaim" in context_lower:
            return "National Insurance Company"
        elif "star health" in context_lower:
            return "Star Health Insurance"
        elif "max bupa" in context_lower:
            return "Max Bupa Health Insurance"
        elif "icici lombard" in context_lower:
            return "ICICI Lombard"
        elif "hdfc ergo" in context_lower:
            return "HDFC ERGO"
        elif "bajaj allianz" in context_lower:
            return "Bajaj Allianz"
        elif "new india" in context_lower:
            return "New India Assurance"
        elif "oriental insurance" in context_lower:
            return "Oriental Insurance"
        elif "united india" in context_lower:
            return "United India Insurance"
        
        return None
    
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
