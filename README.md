# NL2SQL System

Natural Language to SQL query generation system with human-in-the-loop learning.

## Quick Start

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key-here" > .env
python main.py
```

Open http://127.0.0.1:7861

## Features

**SQL Generation**
- Natural language to SQL conversion
- Automatic database detection from 29+ Spider databases
- Foreign key relationship awareness
- Human-in-the-loop query refinement

**User Interface**
- Glassmorphism design
- Real-time query execution
- Interactive SQL editing
- Query regeneration and improvement saving

**Learning System**
- Vector-based few-shot retrieval
- User correction storage
- Continuous accuracy improvement

## Architecture

```
Question → Database Router → Schema Extraction
    ↓
Table Selection → Few-Shot Retrieval
    ↓
SQL Generation → Human Review → Execution
    ↓
Save Improvement → Vectorstore Update
```

## Project Structure

```
main.py                  # Application (2,144 lines)
requirements.txt         # Dependencies
spider/                  # Spider dataset
  database/              # 200+ SQLite databases
  train_spider.json
  dev.json
  tables.json
chroma_db/              # Vectorstore
interaction_log.jsonl   # Query logs
```

## Configuration

Edit main.py:

```python
USE_OPENAI = True
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
```

## Requirements

- Python 3.11+
- OpenAI API key
- 4GB+ RAM
- Internet connection

## Dependencies

```
langchain>=0.2.0
langchain-openai>=0.1.0
chromadb
sqlalchemy
pandas>=2.2.0
gradio
python-dotenv
```

## Usage

Ask natural language questions:

```
"Show all singers"
"How many stadiums are there?"
"List students with GPA above 3.5"
```

Click Run Query to execute. Edit SQL if needed. Save improvements for future queries.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | Change server_port in main.py |
| Import errors | pip install -r requirements.txt |
| API key error | Create .env with OPENAI_API_KEY |
| Database not found | Type: Use database <name> |

## License

MIT License

## Contact

GitHub: @ksaravindakashyap
Repository: github.com/ksaravindakashyap/nl2sql
