#!/usr/bin/env python3
"""Debug script to identify startup issues"""

import sys
import os

print("=" * 60)
print("NL2SQL STARTUP DEBUG")
print("=" * 60)

# Step 1: Check Python version
print(f"\n✓ Python Version: {sys.version}")
print(f"✓ Python Executable: {sys.executable}")

# Step 2: Check .env file
print(f"\n✓ Current Directory: {os.getcwd()}")
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    print(f"✓ .env file exists: {env_path}")
    with open(env_path, 'r') as f:
        env_content = f.read()
    if 'OPENAI_API_KEY' in env_content:
        print("✓ OPENAI_API_KEY found in .env")
else:
    print("✗ .env file NOT found")

# Step 3: Check imports
print("\n--- Checking Imports ---")
try:
    from dotenv import load_dotenv
    print("✓ dotenv imported")
except Exception as e:
    print(f"✗ dotenv import failed: {e}")
    sys.exit(1)

try:
    load_dotenv()
    print("✓ load_dotenv() executed")
except Exception as e:
    print(f"✗ load_dotenv() failed: {e}")
    sys.exit(1)

# Step 4: Check API key
print("\n--- Checking API Keys ---")
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✓ OPENAI_API_KEY loaded: {openai_key[:15]}...")
else:
    print("✗ OPENAI_API_KEY not loaded")

# Step 5: Check LangChain imports
print("\n--- Checking LangChain Imports ---")
try:
    from langchain_openai import ChatOpenAI
    print("✓ ChatOpenAI imported")
except Exception as e:
    print(f"✗ ChatOpenAI import failed: {e}")
    sys.exit(1)

try:
    from langchain_community.utilities.sql_database import SQLDatabase
    print("✓ SQLDatabase imported")
except Exception as e:
    print(f"✗ SQLDatabase import failed: {e}")
    sys.exit(1)

try:
    from langchain_community.vectorstores import Chroma
    print("✓ Chroma imported")
except Exception as e:
    print(f"✗ Chroma import failed: {e}")

try:
    import gradio as gr
    print("✓ Gradio imported")
except Exception as e:
    print(f"✗ Gradio import failed: {e}")
    sys.exit(1)

# Step 6: Check Spider dataset
print("\n--- Checking Spider Dataset ---")
spider_path = './spider'
if os.path.exists(spider_path):
    print(f"✓ Spider path exists: {spider_path}")
    db_folder = os.path.join(spider_path, 'database')
    if os.path.exists(db_folder):
        db_count = len([d for d in os.listdir(db_folder) if os.path.isdir(os.path.join(db_folder, d))])
        print(f"✓ Database folder exists with ~{db_count} databases")
    else:
        print("✗ Database folder not found")
else:
    print("✗ Spider dataset not found")

# Step 7: Try to initialize LLM
print("\n--- Initializing OpenAI LLM ---")
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    print("✓ ChatOpenAI initialized successfully")
except Exception as e:
    print(f"✗ ChatOpenAI initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
