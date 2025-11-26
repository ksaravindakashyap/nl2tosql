import os
import json
import sqlite3
import time
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import gradio as gr
import re

# Required: pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()  # Automatically reads .env file

print("\n" + "="*70)
print("NL2SQL SPIDER PROJECT — HEALTH CHECK STARTING...")
print("="*70 + "\n")

# LangChain / Google Gemini imports
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_core.prompts import PromptTemplate

# SQLAlchemy imports for robust DB loading
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import StaticPool

# Pydantic for output parsing
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

# ==================== MODEL SWITCH (CHANGE ONLY HERE) ====================
USE_OPENAI = True                  # ← Set to False when you have Gemini credits
OPENAI_MODEL = "gpt-4o-mini"       # Options: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
GEMINI_MODEL = "gemini-1.5-flash"  # or gemini-1.5-pro
TEMPERATURE = 0.0
# =========================================================================

# ======== FORCE API KEY LOADING BASED ON MODEL CHOICE ========
if USE_OPENAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise ValueError(
            "\nOPENAI_API_KEY not found or invalid!\n"
            "Create a file named '.env' in the project folder with:\n"
            "OPENAI_API_KEY=sk-your-real-key-here\n"
        )
    os.environ["OPENAI_API_KEY"] = api_key
    print("[OK] OpenAI API key loaded successfully")
else:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "\nGOOGLE_API_KEY not found!\n"
            "Add it to .env file when you want to use Gemini.\n"
        )
    os.environ["GOOGLE_API_KEY"] = api_key
    print("[OK] Gemini API key loaded successfully")
# ============================================================

# ======== HEALTH CHECK SYSTEM ========
def run_health_check():
    """Run comprehensive health check and report status"""
    print("\n1. PROJECT FOLDER STRUCTURE...")
    required = ["spider/train_spider.json", "spider/tables.json", "spider/database/"]
    all_ok = True
    for path in required:
        if Path(path).exists():
            print(f"   [OK] {path}")
        else:
            print(f"   [FAIL] MISSING: {path}")
            print(f"   → Download Spider dataset from https://yale-lily.github.io/spider")
            all_ok = False
    
    print("\n2. API KEY CHECK...")
    if USE_OPENAI:
        key = os.getenv("OPENAI_API_KEY")
        if key and key.startswith("sk-"):
            print(f"   [OK] OPENAI_API_KEY found ({key[:15]}...)")
        else:
            print(f"   [FAIL] OPENAI_API_KEY missing or invalid")
            print(f"   → Add to .env file: OPENAI_API_KEY=sk-...")
            all_ok = False
    else:
        key = os.getenv("GOOGLE_API_KEY")
        if key:
            print(f"   [OK] GOOGLE_API_KEY found")
        else:
            print(f"   [FAIL] GOOGLE_API_KEY missing")
            all_ok = False
    
    print("\n3. REQUIRED PACKAGES...")
    packages_ok = True
    try:
        import langchain
        print(f"   [OK] langchain")
    except Exception as e:
        print(f"   [FAIL] langchain: {e}")
        packages_ok = False
    
    try:
        import gradio
        print(f"   [OK] gradio")
    except Exception as e:
        print(f"   [FAIL] gradio: {e}")
        packages_ok = False
    
    try:
        import sqlalchemy
        print(f"   [OK] sqlalchemy")
    except Exception as e:
        print(f"   [FAIL] sqlalchemy: {e}")
        packages_ok = False
    
    try:
        import chromadb
        print(f"   [OK] chromadb")
    except Exception as e:
        print(f"   [FAIL] chromadb: {e}")
        packages_ok = False
    
    try:
        import openai
        print(f"   [OK] openai")
    except Exception as e:
        print(f"   [FAIL] openai: {e}")
        packages_ok = False
    
    if not packages_ok:
        print(f"   → Run: pip install -r requirements.txt")
        all_ok = False
    
    print("\n" + "="*70)
    if all_ok:
        print("HEALTH CHECK: [OK] ALL GREEN -- APP IS READY TO START!")
    else:
        print("HEALTH CHECK: [FAIL] FAILED -- Fix the items above and rerun")
        print("="*70 + "\n")
        sys.exit(1)
    print("="*70 + "\n")

# Run health check immediately
try:
    run_health_check()
except SystemExit:
    raise
except Exception as e:
    print(f"\n✗ Health check error: {e}")
    sys.exit(1)
# ============================================================

# Constants
SPIDER_PATH = './spider'
TRAIN_FILE = os.path.join(SPIDER_PATH, 'train_spider.json')
DB_FOLDER = os.path.join(SPIDER_PATH, 'database')
PERSIST_DIR = './chroma_db'
LOG_FILE = 'interaction_log.jsonl'

# Hard-coded DB descriptions for guaranteed router accuracy
DB_DESCRIPTIONS = {
    "concert_singer": "Concerts, singers, stadiums, and performances",
    "pets_1": "Pet owners, pets (dogs, cats, rabbits), breeds and treatments",
    "department_management": "Company departments, employees, salaries, management hierarchy",
    "scholar": "Scholars, papers, citations, and academic publications",
    "car_1": "Cars, owners, repairs, and maintenance records",
    "flight_1": "Airports, airlines, flights, and passengers",
    "employee_hire_evaluation": "Employees, hires, evaluations, and job applications",
    "museum_visit": "Museums, visitors, tickets, and exhibits",
    "wta_1": "Tennis players, matches, rankings, and tournaments",
    "battle_death": "Battles, deaths, locations, and military conflicts",
    "student_assessment": "Students, assessments, courses, and grades",
    "tvshow": "TV shows, episodes, actors, and ratings",
    "poker_player": "Poker players, games, winnings, and tournaments",
    "voter_1": "Voters, elections, candidates, and votes",
    "world_1": "Countries, cities, populations, and geography",
    "orchestra": "Orchestras, musicians, concerts, and performances",
    "network_1": "Networks, devices, connections, and protocols",
    "restaurant_1": "Restaurants, customers, orders, and menus",
    "hospital_1": "Hospitals, patients, doctors, and treatments",
    "insurance_fnol": "Insurance claims, policies, accidents, and settlements"
}


def get_llm():
    if USE_OPENAI:
        from langchain_openai import ChatOpenAI
        print(f"Using OpenAI ({OPENAI_MODEL})")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=TEMPERATURE)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print(f"Using Google Gemini ({GEMINI_MODEL})")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=TEMPERATURE)


def get_embeddings():
    if USE_OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def load_spider_dataset(train_file: str = TRAIN_FILE):
    """Load the Spider train JSON file and prepare SQLite DBs in memory robustly.

    Returns:
        spider_data: list of examples from Spider train file
        databases: dict mapping db_id -> SQLDatabase (LangChain wrapper)
    """
    with open(train_file, 'r', encoding='utf-8') as f:
        spider_data = json.load(f)

    # Load databases including cre_* databases for cross-schema testing
    all_db_ids = sorted({d['db_id'] for d in spider_data})
    
    # Priority: Load cre_* databases first for cross-schema testing, then others
    cre_dbs = [db for db in all_db_ids if db.startswith('cre_')]
    other_dbs = [db for db in all_db_ids if not db.startswith('cre_')]
    
    # Must-have databases for testing
    priority_dbs = ['concert_singer', 'singer', 'pets_1', 'car_1', 'world_1', 'student_transcripts_tracking', 
                    'flight_2', 'museum_visit', 'employee_hire_evaluation', 'dog_kennels', 'tvshow']
    
    # Load cre databases + priority databases + remaining up to 30 total
    priority_set = set(priority_dbs)
    remaining_dbs = [db for db in other_dbs if db not in priority_set]
    
    db_ids = cre_dbs[:5] + priority_dbs + remaining_dbs[:14]  # 5 + 11 + 14 = 30 databases
    
    databases = {}
    for db_id in db_ids:
        db_dir = os.path.join(DB_FOLDER, db_id)
        if not os.path.exists(db_dir):
            continue
        
        try:
            # Use in-memory database with check_same_thread=False to allow reuse
            engine = create_engine("sqlite:///:memory:", 
                                 connect_args={"check_same_thread": False},
                                 poolclass=StaticPool)
            sql_files = sorted([f for f in os.listdir(db_dir) if f.endswith('.sql')])
            
            for sql_file in sql_files:
                sql_path = os.path.join(db_dir, sql_file)
                with open(sql_path, 'r', encoding='utf-8') as f:
                    sql_script = f.read()
                
                # Split and execute statements
                statements = [s.strip() for s in sql_script.split(';') if s.strip()]
                with engine.begin() as conn:  # Use begin() for auto-commit
                    for stmt in statements[:100]:  # Limit statements per file to avoid hangs
                        if not stmt or len(stmt) < 5:
                            continue
                        try:
                            conn.execute(text(stmt))
                        except Exception:
                            pass  # skip broken statements
            
            # Enable foreign keys
            with engine.begin() as conn:
                conn.execute(text("PRAGMA foreign_keys = ON;"))
            
            # Create SQLDatabase
            databases[db_id] = SQLDatabase(engine=engine)
            table_names = inspect(engine).get_table_names()
            print(f"Loaded {db_id}: {len(table_names)} tables")
        except Exception as e:
            print(f"Skipped {db_id}: {e}")

    return spider_data, databases


def get_complete_schema_with_foreign_keys(db: SQLDatabase) -> str:
    """Generate complete schema with PKs, FKs, and nullable constraints.
    
    All table and column names are normalized to lowercase for consistency
    and to avoid case-sensitivity issues with the LLM.
    """
    inspector = inspect(db._engine)
    lines = ["-- FULL DATABASE SCHEMA (all names in lowercase for consistency):"]
    schemas_seen = set()
    relationships = []  # Track all foreign key relationships
    
    # Debug: check what tables the inspector sees
    all_tables = inspector.get_table_names()
    print(f"[DEBUG get_complete_schema] Inspector found {len(all_tables)} tables: {all_tables}")
    
    for schema in inspector.get_schema_names():
        if schema in (None, "information_schema", "pg_catalog", "sqlite_master", "sys", "main", "temp"):
            continue
        for table in inspector.get_table_names(schema=schema):
            qualified = f"{schema}.{table}".lower() if schema else table.lower()
            lines.append(f"-- Table: {qualified}")
            schemas_seen.add(schema or "default")
            
            # Columns with nullable info (normalized to lowercase)
            for col in inspector.get_columns(table, schema=schema):
                nullable = " NOT NULL" if not col["nullable"] else ""
                col_name = col['name'].lower()
                lines.append(f"--   {col_name} {col['type']}{nullable}")
            
            # Primary Key
            pk = inspector.get_pk_constraint(table, schema=schema)
            if pk.get("constrained_columns"):
                pk_cols = ', '.join([c.lower() for c in pk['constrained_columns']])
                lines.append(f"--   PRIMARY KEY ({pk_cols})")
            
            # Foreign Keys with full qualification
            for fk in inspector.get_foreign_keys(table, schema=schema):
                local = ", ".join([c.lower() for c in fk["constrained_columns"]])
                ref_table = fk["referred_table"].lower()
                ref_schema = fk.get("referred_schema") or schema
                remote = ", ".join([c.lower() for c in fk["referred_columns"]])
                ref_qualified = f"{ref_schema}.{ref_table}".lower() if ref_schema else ref_table
                lines.append(f"--   FOREIGN KEY ({local}) → {ref_qualified} ({remote})")
                # Track relationship for summary
                table_qualified = f"{schema}.{table}".lower() if schema else table.lower()
                relationships.append(f"{table_qualified}.{local} → {ref_qualified}.{remote}")
    
    # Fallback for SQLite (no explicit schemas)
    if not schemas_seen:
        for table in inspector.get_table_names():
            table_lower = table.lower()
            lines.append(f"-- Table: {table_lower}")
            # Don't add 'default' - SQLite doesn't use schema prefixes
            
            for col in inspector.get_columns(table):
                nullable = " NOT NULL" if not col["nullable"] else ""
                col_name = col['name'].lower()
                lines.append(f"--   {col_name} {col['type']}{nullable}")
            
            pk = inspector.get_pk_constraint(table)
            if pk.get("constrained_columns"):
                pk_cols = ', '.join([c.lower() for c in pk['constrained_columns']])
                lines.append(f"--   PRIMARY KEY ({pk_cols})")
            
            for fk in inspector.get_foreign_keys(table):
                local = ", ".join([c.lower() for c in fk["constrained_columns"]])
                ref_table = fk["referred_table"].lower()
                remote = ", ".join([c.lower() for c in fk["referred_columns"]])
                lines.append(f"--   FOREIGN KEY ({local}) → {ref_table} ({remote})")
                # Track relationship for summary
                relationships.append(f"{table_lower}.{local} → {ref_table}.{remote}")
    
    # Only show schema detection if there are actual schemas (not SQLite)
    if schemas_seen:
        lines.insert(1, f"-- Detected schemas: {', '.join(sorted(schemas_seen))}")
    else:
        lines.insert(1, f"-- SQLite database (case-insensitive, use lowercase)")
    
    # Add relationship summary at the top for easy reference
    if relationships:
        lines.insert(2, "")
        lines.insert(3, "-- TABLE RELATIONSHIPS (use these for JOINs):")
        for rel in relationships:
            lines.insert(4, f"--   {rel}")
        lines.insert(4 + len(relationships), "")
    
    return "\n".join(lines)


def select_relevant_tables(question: str, schema_text: str, llm) -> list:
    """Pre-select top 7 most relevant tables including junction tables - Phase 3 accuracy boost."""
    # Extract actual table names from schema
    import re
    actual_tables = []
    for line in schema_text.split('\n'):
        if line.startswith('-- Table: '):
            table_name = line.replace('-- Table: ', '').strip()
            actual_tables.append(table_name)
    
    print(f"[DEBUG select_relevant_tables] Actual tables from schema: {actual_tables}")
    
    if not actual_tables:
        return []
    
    # Extract table info including FOREIGN KEYS (critical for finding junction tables)
    table_hints = []
    current_table = None
    for line in schema_text.split('\n'):
        if line.startswith('-- Table: '):
            if current_table:
                table_hints.append(current_table)
            current_table = line + '\n'
        elif current_table and (line.startswith('--   ') or line.strip() == ''):
            # Include first few columns AND all foreign keys
            if 'FOREIGN KEY' in line or 'PRIMARY KEY' in line or current_table.count('\n') < 5:
                current_table += line + '\n'
    if current_table:
        table_hints.append(current_table)
    
    # Extract relationship summary from schema
    relationships = []
    for line in schema_text.split('\n'):
        if line.strip().startswith('--   ') and '→' in line:
            relationships.append(line.strip())
    
    rel_summary = '\n'.join(relationships[:15]) if relationships else ''  # Show first 15 relationships
    
    prompt = f"""Given the question and database schema below, return the top 7 most relevant table names as a JSON array.

⚠️ CRITICAL: Include junction/linking tables that connect entities (tables with multiple foreign keys).

Question: {question}

Available tables: {', '.join(actual_tables)}

TABLE RELATIONSHIPS (look for junction tables):
{rel_summary}

Table structures:
{''.join(table_hints)}  

SELECTION STRATEGY:
1. Include main entity tables mentioned in the question (e.g., students, courses, departments)
2. **MUST INCLUDE junction/linking tables** that connect the main entities (e.g., student_enrolment, student_enrolment_courses)
3. Include tables that filter/qualify the query (e.g., departments if filtering by department)

Return format: ["table1", "table2", "table3", "table4", "table5", "table6", "table7"]
Return ONLY the JSON array with table names from the available tables list above."""
    
    try:
        response = llm.invoke(prompt)
        # Extract JSON from response
        text = str(response.content if hasattr(response, 'content') else response)
        print(f"[DEBUG select_relevant_tables] LLM response: {text[:200]}")
        # Find JSON array
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            tables = json.loads(match.group())
            print(f"[DEBUG select_relevant_tables] Parsed tables: {tables}")
            # Strip any schema prefixes (like 'main.') and validate
            cleaned_tables = []
            for t in tables:
                # Remove schema prefix if present
                if '.' in t:
                    t = t.split('.')[-1]
                # Check if it matches any actual table (case-insensitive for SQLite)
                for actual_table in actual_tables:
                    if t.lower() == actual_table.lower():
                        cleaned_tables.append(actual_table)
                        print(f"[DEBUG select_relevant_tables] Matched '{t}' to '{actual_table}'")
                        break
            print(f"[DEBUG select_relevant_tables] Final cleaned tables: {cleaned_tables}")
            return cleaned_tables[:5]
        else:
            print(f"[DEBUG select_relevant_tables] No JSON array found in response")
    except Exception as e:
        print(f"[DEBUG select_relevant_tables] Table selection failed: {e}")
    
    # Fallback: return empty list (use all tables)
    print(f"[DEBUG select_relevant_tables] Returning empty list (fallback)")
    return []


def get_db_router_chain(llm, db_descriptions):
    """Create a LangChain chain for predicting the correct db_id from a question."""
    class DBPrediction(BaseModel):
        db_id: str

    parser = PydanticOutputParser(pydantic_object=DBPrediction)
    
    db_list_text = "\n".join(f"- {db_id}: {desc}" for db_id, desc in db_descriptions.items())

    prompt = PromptTemplate.from_template(
        """Given the user question, predict the most relevant database from the list below.

Question: {question}

Databases:
{db_list}

{format_instructions}

Output only the JSON with the predicted db_id."""
    )

    chain = prompt | llm | parser

    return chain, db_list_text


def predict_db_id(router_chain, db_list, question):
    """Predict the db_id for a given question using LLM with keyword fallback."""
    # First try LLM prediction
    try:
        class DBPrediction(BaseModel):
            db_id: str
        
        parser = PydanticOutputParser(pydantic_object=DBPrediction)
        
        result = router_chain.invoke({
            "question": question,
            "db_list": db_list,
            "format_instructions": parser.get_format_instructions()
        })
        print(f"[DEBUG] LLM predicted database: {result.db_id}")
        
        # Verify the predicted database exists
        if result.db_id in databases:
            return result.db_id
        else:
            print(f"[DEBUG] LLM predicted '{result.db_id}' but it doesn't exist in loaded databases")
    except Exception as e:
        print(f"[DEBUG] LLM prediction failed: {e}")
    
    # Fallback to keyword matching if LLM fails
    question_lower = question.lower()
    keyword_matches = {
        'apartment_rentals': ['apartment', 'rental', 'guest', 'booking', 'building', 'view_unit'],
        'concert_singer': ['concert', 'singer', 'stadium', 'performance'],
        'car_1': ['car', 'vehicle', 'model', 'maker', 'continents'],
        'museum_visit': ['museum', 'visitor', 'visit'],
        'flight_2': ['flight', 'airline', 'airport'],
        'pets_1': ['pet', 'pettype', 'has_pet'],
        'singer': ['singer', 'song'],
        'employee_hire_evaluation': ['employee', 'hire', 'shop', 'evaluation'],
        'world_1': ['country', 'city', 'language', 'population'],
        'orchestra': ['orchestra', 'conductor', 'performance'],
        'network_1': ['highschooler', 'friend', 'likes'],
        'dog_kennels': ['dog', 'kennel', 'owner', 'treatment'],
        'student_transcripts_tracking': ['student', 'transcript', 'course', 'degree', 'enrollment', 'department'],
        'cre_Doc_Template_Mgt': ['document', 'template', 'paragraph'],
        'battle_death': ['battle', 'death', 'ship'],
        'tvshow': ['tv', 'show', 'cartoon', 'channel'],
        'poker_player': ['poker', 'player'],
        'voter_1': ['vote', 'contestant', 'area'],
        'real_estate_properties': ['property', 'estate', 'feature'],
        'course_teach': ['course', 'teacher'],
    }
    
    # Check for keyword matches as fallback
    for db_id, keywords in keyword_matches.items():
        if any(keyword in question_lower for keyword in keywords):
            if db_id in databases:
                print(f"[DEBUG] Keyword fallback match: '{db_id}' (matched: {[k for k in keywords if k in question_lower]})")
                return db_id
    
    print(f"[DEBUG] No database match found")
    return None


def init_embeddings_and_vectorstore(spider_data, persist_dir: str = PERSIST_DIR):
    """Initialize embeddings and Chroma vectorstore (with on-disk cache).

    Returns:
        embeddings, vectorstore
    """
    embeddings = get_embeddings()

    # Prepare Documents for Chroma
    docs = [
        Document(page_content=d.get('question', ''), metadata={'sql': d.get('query', ''), 'db_id': d.get('db_id', ''), 'id': f"spider_{i}"})
        for i, d in enumerate(spider_data)
    ]

    if os.path.exists(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    return embeddings, vectorstore


def extract_llm_text(response) -> str:
    """Extract text content from various LLM response shapes."""
    if response is None:
        return ''
    if hasattr(response, 'content'):
        return getattr(response, 'content')
    if isinstance(response, dict):
        return response.get('content', '')
    return str(response)


def generate_sql(llm, vectorstore, databases, db_descriptions, router_chain, db_list, question: str, db_id=None, k: int = 5) -> str:
    """Generate a SQL query given a natural-language question using LangChain SQL chain with cross-database support.

    - Predicts db_id if not provided using semantic router.
    - Retrieves few-shot examples from the vectorstore filtered by db_id.
    - Uses create_sql_query_chain with custom prompt injecting schema and examples.
    - Returns the raw SQL string.
    """
    # Predict db_id if not provided
    if not db_id:
        db_id = predict_db_id(router_chain, db_list, question)
    if not db_id or db_id not in databases:
        return "-- Error: Could not determine or find the database. Please specify the database in the question, e.g., 'In database concert_singer, ...'"

    db = databases[db_id]

    # Get complete schema with foreign keys - Phase 3 upgrade
    try:
        schema = get_complete_schema_with_foreign_keys(db)
    except Exception as e:
        # Fallback to basic schema
        try:
            schema = db.get_table_info(sample_rows_in_table_info=0)
        except:
            schema = str(db.get_usable_table_names())

    # Pre-select relevant tables for better accuracy - Phase 3
    selected_tables = select_relevant_tables(question, schema, llm)
    print(f"[DEBUG] Selected tables: {selected_tables}")
    
    if selected_tables:
        # Filter schema to show ONLY selected tables with ALL their columns
        # Create lowercase set for case-insensitive matching
        selected_lower = set(t.lower() for t in selected_tables)
        filtered_schema_lines = ["-- RELEVANT TABLES (with complete column information):"]
        current_table = None
        include_current = False
        
        for line in schema.split('\n'):
            if line.startswith('-- Table: '):
                table_name = line.replace('-- Table: ', '').strip()
                # Case-insensitive check
                include_current = table_name.lower() in selected_lower
                print(f"[DEBUG] Checking table '{table_name}' - include: {include_current}")
                if include_current:
                    filtered_schema_lines.append(line)
            elif include_current:
                filtered_schema_lines.append(line)
        
        # Use filtered schema if we found the tables, otherwise use full schema
        if len(filtered_schema_lines) > 1:
            schema = '\n'.join(filtered_schema_lines)
        # If no tables matched, keep original schema

    # Retrieve few-shot examples (defensive API call)
    try:
        docs = vectorstore.similarity_search(question, k=k, filter={'db_id': db_id})
    except TypeError:
        try:
            docs = vectorstore.similarity_search(question, k=k, where={'db_id': db_id})
        except Exception:
            docs = []
    except Exception:
        docs = []

    # Add examples to schema for context
    if docs:
        examples = "\n\n".join([f"-- Example: {doc.page_content}\n-- SQL: {doc.metadata.get('sql','')}" for doc in docs])
        schema = f"{schema}\n\n/* Few-shot examples:\n{examples}\n*/"

    # Detect database type and schema requirements
    is_sqlite = "SQLite database" in schema
    multi_schema = "Detected schemas:" in schema and len(schema.split("Detected schemas:")[1].split(",")) > 1
    
    # Build enhanced prompt with qualified name enforcement - Phase 3
    if is_sqlite:
        qualified_rules = """
CRITICAL SQLite RULES:
- Do NOT use schema prefixes (like 'default.' or 'main.')
- Use table names EXACTLY as shown in the schema (case-sensitive)
- Use column names EXACTLY as shown in the schema
- Example: SELECT * FROM Guests NOT FROM default.guests
"""
    elif multi_schema:
        qualified_rules = """
CRITICAL MULTI-SCHEMA RULES:
- ALWAYS use fully qualified names: schema.table.column
- NEVER use bare table names if more than one schema is present
- Example: cre_Drama_Workshop_Groups.member NOT just member
"""
    else:
        qualified_rules = ""

    # Generate SQL directly with LLM using our enhanced schema and rules
    if is_sqlite:
        # Extract actual table and column names from schema for explicit listing
        schema_info = {}
        current_table = None
        
        # Debug: print first 500 chars of schema to see format
        print(f"\n[DEBUG] Schema first 500 chars:\n{schema[:500]}")
        
        for line in schema.split('\n'):
            if line.startswith('-- Table: '):
                current_table = line.replace('-- Table: ', '').strip()
                schema_info[current_table] = []
                print(f"[DEBUG] Found table: {current_table}")
            elif current_table and line.startswith('--   ') and not line.strip().startswith('-- PRIMARY') and not line.strip().startswith('-- FOREIGN'):
                col_name = line.replace('--   ', '').strip().split()[0]
                if col_name and not col_name.startswith('PRIMARY') and not col_name.startswith('FOREIGN'):
                    schema_info[current_table].append(col_name)
        
        # Build explicit column listing - SHOW ALL COLUMNS (no truncation)
        explicit_columns = []
        for table, columns in schema_info.items():
            if columns:
                explicit_columns.append(f"Table '{table}': {', '.join(columns)}")  # Show ALL columns
        
        table_count = len(schema_info)
        
        # Debug: print what we extracted
        print(f"\n[DEBUG] Extracted schema info for {table_count} tables:")
        for ec in explicit_columns:
            print(f"  {ec}")
        
        system_message = f"""You are a SQL expert. You MUST use ONLY the table and column names from the schema below.

🚨 CRITICAL: The schema below is COMPLETE and AUTHORITATIVE.

🚫 ABSOLUTELY FORBIDDEN JOIN PATTERNS (will cause errors):
❌ JOIN table_name ON column IN (SELECT ...) 
❌ JOIN table_name ON unrelated_column_1 = unrelated_column_2
❌ Joining tables without using their FOREIGN KEY relationships

✅ ONLY ALLOWED JOIN PATTERN:
✅ JOIN table_name ON parent_table.foreign_key = child_table.primary_key
✅ Use the FOREIGN KEY relationships shown in the schema below

AVAILABLE TABLES AND THEIR COLUMNS:
{chr(10).join(explicit_columns)}

ABSOLUTE RULES:
1. Use ONLY table names and column names listed above
2. DO NOT invent, guess, or assume any column names
3. If a column name sounds logical but isn't in the list above, IT DOESN'T EXIST
4. All names in lowercase
5. When aggregating or listing entities (people, places, things), prefer human-readable fields (name, title, description) over IDs
6. **CRITICAL**: Study FOREIGN KEY relationships - they show how tables connect
7. **JOINING TABLES**: Use ONLY foreign key columns for JOIN conditions (never join on unrelated columns)
8. **MANY-TO-MANY**: Look for junction/linking tables (e.g., Student_Enrolment_Courses links students to courses)

FOREIGN KEY & JOIN RULES:
🔑 ONLY join tables using their FOREIGN KEY relationships
🔑 If table A has a foreign key to table B, join using: `A JOIN B ON A.foreign_key_id = B.primary_key_id`
🔑 For many-to-many relationships, you MUST use the junction/linking table
🔑 NEVER create JOIN conditions like `ON table1.id IN (SELECT ...)` - use proper equality joins
🔑 NEVER join unrelated columns (e.g., student_id = section_id is WRONG)

COMMON PATTERNS:
✅ Students → Student_Enrolment → Student_Enrolment_Courses → Courses
✅ Concerts → Singer_In_Concert → Singers (through junction table)
❌ Students → Sections directly (NO direct relationship - need junction table)

USER EXPERIENCE GUIDELINES:
✅ PREFER: SELECT name, title, description (human-readable)
❌ AVOID: SELECT id (unless specifically asked for IDs)
✅ PREFER: GROUP_CONCAT(singer.name) to show "John, Mary, Bob"
❌ AVOID: GROUP_CONCAT(singer_id) to show "1, 2, 3"
✅ When listing entities, JOIN to the main table to get descriptive fields

EXAMPLES OF WHAT **NOT** TO DO:
❌ "bookings" table - WRONG! The actual table is "apartment_bookings"
❌ "booking_id" column - WRONG! The actual column is "apt_booking_id"  
❌ "name" column in guests - WRONG! Actual columns are "guest_first_name" and "guest_last_name"
❌ "apt_type" column - WRONG! Actual column is "apt_type_code"
❌ GROUP_CONCAT(singer_id) - WRONG! Use GROUP_CONCAT(singer.name) for readability
❌ JOIN sections ON student_id = section_id - WRONG! These are unrelated columns
❌ JOIN sections ON section_id IN (...) - WRONG! Use proper equality joins with foreign keys

IF YOU USE A NAME NOT IN THE LIST ABOVE, THE QUERY WILL FAIL."""
    else:
        system_message = "You are a SQL expert. Generate valid SQL queries using the exact schema provided."
    
    prompt = f"""{system_message}

═══════════════════════════════════════════════════════════════
COMPLETE DATABASE SCHEMA WITH ALL COLUMNS:
═══════════════════════════════════════════════════════════════
{schema}
═══════════════════════════════════════════════════════════════

{qualified_rules}

QUESTION: {question}

🔍 CRITICAL: Study the FOREIGN KEY relationships in the schema above BEFORE writing SQL!

For student-course queries in THIS database, the relationship path is:
  students (has student_id)
    ↓ JOIN ON students.student_id = student_enrolment.student_id
  student_enrolment (links students to degree programs)
    ↓ JOIN ON student_enrolment.student_enrolment_id = student_enrolment_courses.student_enrolment_id  
  student_enrolment_courses (links enrollments to courses)
    ↓ JOIN ON student_enrolment_courses.course_id = courses.course_id
  courses (has course details)

⚠️ CRITICAL LIMITATION: There is NO direct link between courses and departments in this schema.
⚠️ The ONLY way to associate courses with departments is through student enrollments:
   courses → student_enrolment_courses → student_enrolment → degree_programs → departments

🔴 FORBIDDEN: Do NOT join courses.course_id = degree_programs.degree_program_id (different entities!)
🔴 FORBIDDEN: Do NOT create non-existent relationships between tables

✅ CORRECT interpretation for "courses offered by a department":
   = "courses taken by students enrolled in degree programs of that department"
   This requires going through the student_enrolment chain.

STEP-BY-STEP APPROACH:
1. **Identify required tables**: Which tables contain the data needed?
2. **Find the relationship path**: How are these tables connected via FOREIGN KEYS?
3. **Check for junction tables**: Do I need a linking table for many-to-many relationships?
4. **Build proper JOINs**: Use foreign key columns for equality joins (A.fk_id = B.pk_id)
5. **Choose readable columns**: Use names/descriptions, not IDs (unless specifically requested)

BEFORE YOU WRITE SQL:
✓ Look at the FOREIGN KEY relationships in the schema
✓ Trace the path between tables using foreign keys
✓ Use junction/linking tables for many-to-many relationships
✓ NEVER join on unrelated columns
✓ Use ONLY exact table/column names from the list above
✓ Prefer human-readable fields (names) over IDs

COMMON PATTERN - Aggregating related entities:
When you need to show "list of singers", "list of students", "list of products", etc.:
❌ WRONG: GROUP_CONCAT(junction_table.entity_id)  
✅ CORRECT: JOIN entity_table, then GROUP_CONCAT(entity_table.name)

Example: For "list of singers who performed at each concert":
❌ BAD:  GROUP_CONCAT(singer_in_concert.singer_id)  → Shows "1,2,3"
✅ GOOD: JOIN singer, GROUP_CONCAT(singer.name)    → Shows "John,Mary,Bob"

Example: For "students enrolled in courses":
❌ BAD:  students JOIN sections (no direct relationship)
✅ GOOD: students → student_enrolment → student_enrolment_courses → courses (follow foreign keys)

CRITICAL PATTERN - "enrolled in EVERY course" queries:
When the question asks for entities that have "all", "every", "each" of something:

❌ WRONG HAVING clause:
  HAVING COUNT(DISTINCT item_id) = (SELECT COUNT(*) FROM items)
  ↑ This counts ALL items in the table, not filtered items

✅ CORRECT HAVING clause:
  HAVING COUNT(DISTINCT item_id) = (
      SELECT COUNT(DISTINCT item_id) 
      FROM [same_tables_and_joins_as_main_query]
      WHERE [same_filter_conditions]
  )
  ↑ This counts only the filtered subset

Example: "Students enrolled in every course offered by computer science department"
❌ WRONG:
  HAVING COUNT(DISTINCT c.course_id) = (SELECT COUNT(*) FROM courses)
  ↑ Counts ALL courses (all departments)

✅ CORRECT:
  HAVING COUNT(DISTINCT c.course_id) = (
      SELECT COUNT(DISTINCT c2.course_id)
      FROM courses c2
      JOIN student_enrolment_courses sec2 ON c2.course_id = sec2.course_id
      JOIN student_enrolment se2 ON sec2.student_enrolment_id = se2.student_enrolment_id
      JOIN degree_programs dp2 ON se2.degree_program_id = dp2.degree_program_id
      JOIN departments d2 ON dp2.department_id = d2.department_id
      WHERE d2.department_name = 'computer science'
  )
  ↑ Counts only CS courses (through the student_enrolment chain - the ONLY valid path)

Generate the SQL query:

Return ONLY the SQL query without any explanation, markdown formatting, or additional text.

SQL QUERY:"""
    
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        # Fallback to basic chain if LLM invoke fails
        try:
            chain = create_sql_query_chain(llm, db)
            response = chain.invoke({"question": question})
        except:
            return f"-- Error generating SQL: {e}"

    sql = extract_llm_text(response).strip()
    
    # Remove any explanatory text before or after the SQL
    # Look for SELECT statement and extract only that
    import re
    select_match = re.search(r'(SELECT\s+.*?;)', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        sql = select_match.group(1)
    
    # Also check for other SQL statements (INSERT, UPDATE, DELETE, CREATE, etc.)
    if not select_match:
        other_match = re.search(r'((?:INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*?;)', sql, re.IGNORECASE | re.DOTALL)
        if other_match:
            sql = other_match.group(1)
    # Strip any markdown code blocks if present
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.endswith("```"):
        sql = sql[:-3]
    sql = sql.strip()
    
    # Validate SQL for common forbidden patterns
    validation_errors = []
    
    # Check for forbidden JOIN ON ... IN (...) pattern
    if re.search(r'JOIN\s+\w+\s+(?:ON|on)\s+[^=]+\s+IN\s*\(', sql, re.IGNORECASE):
        validation_errors.append("⚠️ INVALID: Using 'JOIN ... ON column IN (...)' - Use equality JOIN with foreign keys")
    
    # Check for comparing unrelated ID columns (e.g., student_id = section_id)
    invalid_joins = re.findall(r'ON\s+(\w+_id)\s*=\s*(\w+_id)', sql, re.IGNORECASE)
    for col1, col2 in invalid_joins:
        if col1 != col2 and not any(x in col1 for x in ['student', 'course', 'section']) or \
           not any(x in col2 for x in ['student', 'course', 'section']):
            # Only warn if they're clearly unrelated (different entity types)
            pass  # Too aggressive, skip for now
    
    if validation_errors:
        error_msg = "\n".join(validation_errors)
        print(f"[VALIDATION ERROR]\n{error_msg}\n{sql}")
        # Return error comment + the bad SQL so user can see what went wrong
        return f"-- VALIDATION FAILED:\n-- {error_msg}\n\n-- Generated (but invalid) SQL:\n{sql}"
    
    # For SQLite: Remove schema prefixes like 'main.' or 'default.'
    if is_sqlite:
        import re
        # Remove schema prefixes from table names (main.table_name -> table_name)
        sql = re.sub(r'\b(main|default)\s*\.\s*', '', sql, flags=re.IGNORECASE)
    
    # Validate SQL against schema (basic check for common errors)
    if is_sqlite:
        # Check for common hallucinated column names (case-insensitive check)
        sql_lower = sql.lower()
        hallucinated_patterns = [
            ('select name from', 'guest_first_name, guest_last_name'),
            (' from bookings', 'apartment_bookings'),
            ('from singers', 'singer'),
            ('from concerts', 'concert'),
            ('apartment_id', 'apt_id'),
        ]
        
        for wrong_pattern, correction_hint in hallucinated_patterns:
            if wrong_pattern in sql_lower:
                sql = f"-- Warning: Possible error detected. Check schema.\n-- Hint: Use '{correction_hint}' instead\n{sql}"
                break
    
    return sql


def execute_sql_and_summarize(llm, sql: str, db_id: str, databases):
    """Execute SQL on the selected Spider DB and return (summary_text, dataframe).

    The function returns a short natural-language summary produced by the LLM
    and the full Pandas DataFrame for display.
    """
    try:
        db = databases[db_id]
        with db.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
    except Exception as e:
        return (f"Error executing SQL: {str(e)}", pd.DataFrame())

    # Prepare a short preview for the LLM to summarize
    preview = df.head(10).to_csv(index=False)
    cols = df.columns.tolist()
    prompt = f"""Summarize the following SQL query results in concise natural language.

Columns: {cols}

Preview (first rows as CSV):
{preview}

Provide a short summary (2-4 sentences) describing the result, number of rows, and any notable values.
"""
    try:
        response = llm.invoke(prompt)
        summary = extract_llm_text(response).strip()
    except Exception:
        summary = f"Returned {len(df)} rows. Columns: {cols}."

    return summary, df


def log_interaction(log_file: str, entry: dict):
    """Append a JSONL log entry to `log_file` with the exact structure.
    """
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    except Exception:
        # Never crash the main flow for logging errors
        pass


def build_gradio_app(llm, vectorstore, databases, db_descriptions, router_chain, db_list):
    """Construct and return the Gradio Blocks app for interactive NL2SQL with multi-turn chat and cross-database support.

    Features:
    - Stateful chatbot with conversation history.
    - Automatic DB switching with semantic router.
    - Cross-schema support with foreign keys.
    - Table pre-selection for accuracy boost.
    - Save corrections to improve few-shot learning.
    """
    model_name = f"OpenAI {OPENAI_MODEL}" if USE_OPENAI else f"Gemini {GEMINI_MODEL}"
    title = f"NL2SQL Spider Copilot - Intelligent SQL Generation"
    
    # Stats tracking
    corrections_file = "saved_corrections.jsonl"
    
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(f"**Model:** {model_name} │ **Databases Loaded:** {len(databases)} │ **Auto-detection & Cross-schema Support**")
        
        gr.Markdown("💡 **Tips:** Type your question to auto-detect database | Use `/schema <database>` to view schema | Say 'Use database <db_id>' to force switch")
        
        # Main Chat Interface
        chatbot = gr.Chatbot(label="Conversation History", height=450)
        msg = gr.Textbox(label="Your Question", placeholder="Ask a question in natural language...", lines=2)
        sql_box = gr.Textbox(label="Generated SQL (Edit if needed)", lines=8, interactive=True, visible=False)
        
        with gr.Row():
            regenerate_btn = gr.Button("🔄 Regenerate SQL", visible=False)
            run_query_btn = gr.Button("▶️ Run Query", visible=False)
            save_correction_btn = gr.Button("💾 Save Correction", visible=False)
        
        correction_status = gr.Textbox(label="Status", visible=False, interactive=False)
        
        # State for history, current db_id, etc.
        history = gr.State([])
        current_db_id = gr.State("")
        current_question = gr.State("")
        current_generated_sql = gr.State("")
        current_edited_sql = gr.State("")
        few_shot_ids = gr.State([])
        predicted_db_id = gr.State("")
        manual_override = gr.State(False)


        def user_message(message, history, current_db_id):
            """Handle user input: /schema, force DB switch, or generate SQL with auto DB detection."""
            message = message.strip()
            if not message:
                return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_db_id, "", "", [], "", False
            
            if message.startswith("/schema"):
                # Handle /schema or /schema <database>
                parts = message.split()
                if len(parts) > 1:
                    db_id = parts[1].strip()
                    if db_id in databases:
                        db = databases[db_id]
                        schema = get_complete_schema_with_foreign_keys(db)
                        response = f"**Schema for {db_id}:**\n\n```\n{schema}\n```"
                    else:
                        response = f"Database '{db_id}' not found. Available databases: {', '.join(sorted(databases.keys())[:15])}..."
                else:
                    if not current_db_id:
                        response = "No database selected. Use `/schema <database>` or ask a question first."
                    else:
                        db = databases[current_db_id]
                        schema = get_complete_schema_with_foreign_keys(db)
                        response = f"**Schema for {current_db_id}:**\n\n```\n{schema}\n```"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_db_id, "", "", [], "", False
            
            elif message.lower().startswith("use database"):
                parts = message.lower().split("use database")
                new_db_id = parts[1].strip().replace(",", "").strip()
                if new_db_id in databases:
                    current_db_id = new_db_id
                    response = f"Switched to database: {new_db_id}"
                else:
                    response = f"Database '{new_db_id}' not found. Available: {', '.join(sorted(databases.keys())[:10])}"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_db_id, "", "", [], "", False
            
            else:
                # Auto-detect or predict database
                pred_db_id = predict_db_id(router_chain, db_list, message)
                print(f"[DEBUG] Database prediction for '{message[:50]}...': {pred_db_id}")
                manual_override = False
                
                # If router failed, use current_db_id
                if not pred_db_id or pred_db_id not in databases:
                    if current_db_id and current_db_id in databases:
                        pred_db_id = current_db_id
                    else:
                        # Try to use first available database
                        pred_db_id = sorted(databases.keys())[0] if databases else None
                
                print(f"[DEBUG] Final selected database: {pred_db_id}")
                
                if not pred_db_id:
                    response = "Error: No databases available or could not determine database."
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response})
                    return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_db_id, message, "", [], "", False
                
                # Generate SQL
                sql = generate_sql(llm, vectorstore, databases, DB_DESCRIPTIONS, router_chain, db_list, message, pred_db_id)
                
                # Update current_db_id if different
                if pred_db_id != current_db_id:
                    current_db_id = pred_db_id
                
                # Retrieve few_shot examples
                try:
                    docs = vectorstore.similarity_search(message, k=5, filter={'db_id': pred_db_id})
                    fs_ids = [doc.metadata.get('id', '') for doc in docs]
                except Exception:
                    fs_ids = []
                
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"**Database:** {pred_db_id}\n\n**Generated SQL:**\n```sql\n{sql}\n```"})
                
                return "", history, gr.update(value=sql, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), current_db_id, message, sql, fs_ids, pred_db_id, manual_override

        def save_correction(question, sql, db_id):
            """Save user-corrected SQL to few-shot store."""
            try:
                correction = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "question": question,
                    "sql": sql,
                    "db_id": db_id
                }
                
                # Save to corrections file
                with open(corrections_file, 'a', encoding='utf-8') as f:
                    json.dump(correction, f, ensure_ascii=False)
                    f.write('\n')
                
                # Also add to vectorstore for immediate use
                doc = Document(
                    page_content=question,
                    metadata={"sql": sql, "db_id": db_id, "source": "user_correction"}
                )
                vectorstore.add_documents([doc])
                
                # Count corrections
                try:
                    with open(corrections_file, 'r', encoding='utf-8') as f:
                        count = sum(1 for line in f)
                except:
                    count = 1
                
                return gr.update(value=f"✅ Correction saved! Total corrections: {count}", visible=True)
            except Exception as e:
                return gr.update(value=f"❌ Error saving correction: {e}", visible=True)

        def regenerate_sql(history, db_id, question):
            """Regenerate SQL for the current question."""
            if not question:
                return history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            sql = generate_sql(llm, vectorstore, databases, DB_DESCRIPTIONS, router_chain, db_list, question, db_id)
            # Update history
            history[-1] = {"role": "assistant", "content": f"**Regenerated SQL:**\n```sql\n{sql}\n```"}
            return history, gr.update(value=sql, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), sql

        def run_query(sql, db_id, question, generated_sql, few_shot_ids, predicted_db_id, manual_override, history):
            """Execute the (possibly edited) SQL and append to history."""
            if not db_id or db_id not in databases:
                response = "Error: Database not found."
                error = response
                result = ""
            else:
                summary, df = execute_sql_and_summarize(llm, sql, db_id, databases)
                response = f"Summary: {summary}\n\nTable:\n{df.to_string() if not df.empty else 'No results'}"
                error = None
                result = df.to_string() if not df.empty else ""

            history.append({"role": "user", "content": "Executed SQL"})
            history.append({"role": "assistant", "content": response})

            # Detect schemas
            db = databases[db_id]
            inspector = inspect(db._engine)
            detected_schemas = [s for s in inspector.get_schema_names() if s not in ['information_schema', 'pg_catalog', 'sqlite_master']]

            # Log interaction
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "db_id": db_id,
                "predicted_db_id": predicted_db_id,
                "router_confidence": None,
                "manual_override": manual_override,
                "detected_schemas": detected_schemas,
                "question": question,
                "generated_sql": generated_sql,
                "edited_sql": sql if sql != generated_sql else generated_sql,
                "final_sql_used": sql,
                "result": result,
                "error": error,
                "few_shot_ids": few_shot_ids
            }
            log_interaction(LOG_FILE, log_entry)

            return history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        msg.submit(user_message, inputs=[msg, history, current_db_id], outputs=[msg, chatbot, sql_box, regenerate_btn, run_query_btn, save_correction_btn, correction_status, current_db_id, current_question, current_generated_sql, few_shot_ids, predicted_db_id, manual_override])
        regenerate_btn.click(regenerate_sql, inputs=[history, current_db_id, current_question], outputs=[chatbot, sql_box, regenerate_btn, run_query_btn, save_correction_btn, current_generated_sql])
        run_query_btn.click(run_query, inputs=[sql_box, current_db_id, current_question, current_generated_sql, few_shot_ids, predicted_db_id, manual_override, history], outputs=[chatbot, sql_box, regenerate_btn, run_query_btn, save_correction_btn, correction_status])
        save_correction_btn.click(save_correction, inputs=[current_question, sql_box, current_db_id], outputs=[correction_status])

    return demo


if __name__ == '__main__':
    print('Preparing NL2SQL Phase 2 application...')
    spider_data, databases = load_spider_dataset()
    embeddings, vectorstore = init_embeddings_and_vectorstore(spider_data)
    llm = get_llm()
    router_chain, db_list = get_db_router_chain(llm, DB_DESCRIPTIONS)
    demo = build_gradio_app(llm, vectorstore, databases, DB_DESCRIPTIONS, router_chain, db_list)

    try:
        demo.launch(server_name='127.0.0.1', server_port=7861, share=False)
    except TypeError:
        demo.launch(server_name='127.0.0.1', server_port=7861, share=False)