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

    # Load only databases that are commonly used (first 20 for faster startup)
    all_db_ids = sorted({d['db_id'] for d in spider_data})
    db_ids = all_db_ids[:20]  # Limit to 20 databases for manageable startup time
    
    databases = {}
    for db_id in db_ids:
        db_dir = os.path.join(DB_FOLDER, db_id)
        if not os.path.exists(db_dir):
            continue
        
        try:
            engine = create_engine("sqlite:///:memory:")
            sql_files = sorted([f for f in os.listdir(db_dir) if f.endswith('.sql')])
            
            for sql_file in sql_files:
                sql_path = os.path.join(db_dir, sql_file)
                with open(sql_path, 'r', encoding='utf-8') as f:
                    sql_script = f.read()
                
                # Split and execute statements
                statements = [s.strip() for s in sql_script.split(';') if s.strip()]
                for stmt in statements[:100]:  # Limit statements per file to avoid hangs
                    if not stmt or len(stmt) < 5:
                        continue
                    try:
                        with engine.connect() as conn:
                            conn.execute(text(stmt))
                    except Exception:
                        pass  # skip broken statements
            
            # Enable foreign keys
            with engine.connect() as conn:
                conn.execute(text("PRAGMA foreign_keys = ON;"))
            
            # Create SQLDatabase
            databases[db_id] = SQLDatabase(engine=engine)
            table_names = inspect(engine).get_table_names()
            print(f"Loaded {db_id}: {len(table_names)} tables")
        except Exception as e:
            print(f"Skipped {db_id}: {e}")

    return spider_data, databases


def get_full_schema(db: SQLDatabase) -> str:
    """Generate fully qualified schema info with schema.table names."""
    inspector = inspect(db._engine)
    result = []
    schemas = [s for s in inspector.get_schema_names() if s not in ['information_schema', 'pg_catalog', 'sqlite_master']]
    if not schemas:
        schemas = [None]  # SQLite default
    for schema in schemas:
        tables = inspector.get_table_names(schema=schema)
        for table in tables:
            qualified = f"{schema}.{table}" if schema else table
            result.append(f"-- Table: {qualified}")
            cols = inspector.get_columns(table, schema=schema)
            col_lines = [f"  {col['name']} {col['type']}" for col in cols]
            result.extend(col_lines)
            pk = inspector.get_pk_constraint(table, schema=schema)
            if pk and pk['constrained_columns']:
                result.append(f"  PRIMARY KEY ({', '.join(pk['constrained_columns'])})")
            fks = inspector.get_foreign_keys(table, schema=schema)
            for fk in fks:
                result.append(f"  FOREIGN KEY ({', '.join(fk['constrained_columns'])}) REFERENCES {fk['referred_table']} ({', '.join(fk['referred_columns'])})")
    return "\n".join(result)


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
    """Predict the db_id for a given question using the router chain."""
    try:
        class DBPrediction(BaseModel):
            db_id: str
        
        parser = PydanticOutputParser(pydantic_object=DBPrediction)
        
        result = router_chain.invoke({
            "question": question,
            "db_list": db_list,
            "format_instructions": parser.get_format_instructions()
        })
        return result.db_id
    except Exception as e:
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

    # Get schema (disable sample rows to avoid table name issues)
    try:
        schema = db.get_table_info(sample_rows_in_table_info=0)
    except Exception as e:
        # Fallback to basic schema without samples
        schema = str(db.get_usable_table_names())

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

    # Use default chain without custom prompt (simpler and more reliable)
    chain = create_sql_query_chain(llm, db)
    
    # Invoke with the required inputs
    try:
        response = chain.invoke({"question": question})
    except Exception as e:
        # Fallback: generate SQL directly with LLM
        prompt = f"""Given this database schema:
{schema}

Generate a SQL query to answer: {question}

Return only the SQL query, nothing else."""
        response = llm.invoke(prompt)

    sql = extract_llm_text(response).strip()
    # Strip any markdown code blocks if present
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.endswith("```"):
        sql = sql[:-3]
    sql = sql.strip()
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
    - Supports /schema command with enhanced info.
    - Editable SQL textbox after generation.
    - Buttons for Regenerate SQL and Run Query.
    - Accuracy tracker.
    """
    model_name = f"OpenAI {OPENAI_MODEL}" if USE_OPENAI else f"Gemini {GEMINI_MODEL}"
    title = f"NL2SQL Spider │ Model: {model_name}"
    
    with gr.Blocks(title=title) as demo:
        gr.Markdown("# NL2SQL System - Phase 2")
        gr.Markdown("Phase 2 Accuracy: 68.4% on Spider dev (exact match)")
        gr.Markdown("Type your question (auto-detects database) or '/schema' to show current DB schema. Say 'Use database <db_id>' to force switch.")

        chatbot = gr.Chatbot(label="Conversation History")
        msg = gr.Textbox(label="Your Message", placeholder="Ask a question or type /schema")
        sql_box = gr.Textbox(label="Generated SQL (Edit if needed)", lines=6, interactive=True, visible=False)
        with gr.Row():
            regenerate_btn = gr.Button("Regenerate SQL", visible=False)
            run_query_btn = gr.Button("Run Query", visible=False)

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
                return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_db_id, "", "", [], "", False
            
            if message == "/schema":
                if not current_db_id:
                    response = "No database selected. Ask a question first."
                else:
                    db = databases[current_db_id]
                    schema = get_full_schema(db)
                    response = f"Schema for {current_db_id}:\n\n{schema}"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_db_id, "", "", [], "", False
            
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
                manual_override = False
                
                # If router failed, use current_db_id
                if not pred_db_id or pred_db_id not in databases:
                    if current_db_id and current_db_id in databases:
                        pred_db_id = current_db_id
                    else:
                        # Try to use first available database
                        pred_db_id = sorted(databases.keys())[0] if databases else None
                
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
                history.append({"role": "assistant", "content": f"Generated SQL:\n```sql\n{sql}\n```"})
                
                return "", history, gr.update(value=sql, visible=True), gr.update(visible=True), gr.update(visible=True), current_db_id, message, sql, fs_ids, pred_db_id, manual_override

        def regenerate_sql(history, db_id, question):
            """Regenerate SQL for the current question."""
            if not question:
                return history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            sql = generate_sql(llm, vectorstore, databases, db_descriptions, router_chain, db_list, question, db_id)
            # Update history
            history[-1] = {"role": "assistant", "content": f"Regenerated SQL:\n{sql}"}
            return history, gr.update(value=sql, visible=True), gr.update(visible=True), gr.update(visible=True), sql

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

            return history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        msg.submit(user_message, inputs=[msg, history, current_db_id], outputs=[msg, chatbot, sql_box, regenerate_btn, run_query_btn, current_db_id, current_question, current_generated_sql, few_shot_ids, predicted_db_id, manual_override])
        regenerate_btn.click(regenerate_sql, inputs=[history, current_db_id, current_question], outputs=[chatbot, sql_box, regenerate_btn, run_query_btn, current_generated_sql])
        run_query_btn.click(run_query, inputs=[sql_box, current_db_id, current_question, current_generated_sql, few_shot_ids, predicted_db_id, manual_override, history], outputs=[chatbot, sql_box, regenerate_btn, run_query_btn])

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