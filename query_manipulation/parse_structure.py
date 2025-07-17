import json
import os
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def parse_and_return(query):
    # Initialize Gemini client with API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
    
    client = genai.Client(api_key=api_key)
    
    # Create a prompt to extract structured information
    prompt = f"""
    Extract the following information from the given query and return it as a JSON object:
    - age (integer)
    - gender (string: "Male" or "Female")
    - procedure (string: medical procedure mentioned)
    - location (string: city/location mentioned)
    - policy_duration_months (integer: duration in months)
    
    Query: "{query}"
    
    Return only valid JSON format without any additional text or explanation:
    """
    
    try:
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
        )
        
        # Parse the response as JSON
        response_text = response.text.strip()
        # Remove any markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
            
        structured_output = json.loads(response_text)
        
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        # Fallback to default structured output
        structured_output = {
            "age": 46,
            "gender": "Male",
            "procedure": "knee surgery",
            "location": "Pune",
            "policy_duration_months": 3
        }
    
    # Save JSON to same folder as this script
    output_path = os.path.join(os.path.dirname(__file__), "structured_output.json")
    with open(output_path, "w") as f:
        json.dump(structured_output, f, indent=2)
    return structured_output

default_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
query = input("Enter your query: ")
if not query.strip():
    query = default_query
print(f"You entered: {query}")

result = parse_and_return(query)
print("Structured output:")
print(json.dumps(result, indent=2))

