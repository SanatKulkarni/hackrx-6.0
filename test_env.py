#!/usr/bin/env python3
"""
Quick Environment Variable Test
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Environment Variables Test")
print("="*40)

# Check API keys
gemini_key = os.getenv("GEMINI_API_KEY")
nomic_key = os.getenv("NOMIC_API_KEY") 
pinecone_key = os.getenv("PINECONE_API_KEY")

print(f"GEMINI_API_KEY: {'✅ Found' if gemini_key else '❌ Missing'}")
print(f"NOMIC_API_KEY: {'✅ Found' if nomic_key else '❌ Missing'}")
print(f"PINECONE_API_KEY: {'✅ Found' if pinecone_key else '❌ Missing'}")

if pinecone_key:
    print(f"Pinecone key preview: {pinecone_key[:15]}...")

print("="*40)

if all([gemini_key, nomic_key, pinecone_key]):
    print("✅ All API keys found!")
else:
    print("❌ Some API keys missing!")
