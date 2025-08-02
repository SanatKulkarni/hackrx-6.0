#!/usr/bin/env python3
"""
Test script for HackRX 6.0 Q&A System
Run this from the root directory to test the system
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from final_codebase import HackathonQASystem

def test_system():
    """Test the Q&A system with a simple example"""
    try:
        print("🚀 Initializing HackRX 6.0 Q&A System...")
        
        # Initialize the system
        qa_system = HackathonQASystem(
            index_name="hackathon-qa-test",
            namespace="test-docs"
        )
        
        print("✅ System initialized successfully!")
        
        # Test with the same hackathon request as in the main system
        test_request = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                "What is the waiting period for pre-existing diseases (PED) to be covered?",
                "Does this policy cover maternity expenses, and what are the conditions?",
                "What is the waiting period for cataract surgery?",
                "Are the medical expenses for an organ donor covered under this policy?"
            ]
        }
        
        print("🧪 Testing with hackathon request...")
        print(f"Document: {test_request['documents']}")
        print(f"Questions ({len(test_request['questions'])}):")
        for i, q in enumerate(test_request['questions'], 1):
            print(f"  {i}. {q}")
        
        # Actually process the request
        print("\n🚀 Processing hackathon request...")
        result = qa_system.process_hackathon_request(test_request)
        
        if "error" in result:
            print(f"❌ Processing failed: {result['error']}")
            return False
        
        print(f"\n✅ Successfully processed {len(result['answers'])} answers!")
        for i, answer in enumerate(result['answers'], 1):
            print(f"\n{i}. {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_system()
    if success:
        print("🎉 System test completed!")
    else:
        print("💥 System test failed!")
