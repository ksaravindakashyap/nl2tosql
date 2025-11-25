# NL2SQL Phase 1 — Gemini + LangChain + Spider

This repository contains a Phase 1 implementation of an NL→SQL system using Google Gemini + LangChain and the Spider dataset.

Files
- `main.py` — single-file application (Gradio) to generate, show/edit, and execute SQL against Spider databases.
- `spider/` — local Spider dataset (you've already downloaded this).
- `requirements.txt` — Python dependencies for the project.
- `conversation_log.jsonl` — (created at runtime) stores user questions, generated/edited SQL, and execution results.

Quick setup (recommended: use a virtual environment)

1) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

(If you use bash)

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes on packages
- `langchain` should be the 0.3.x series if possible (the code uses the 0.3 style). If you already have a different version installed, you may need small import adjustments.
- `google-generativeai` and `langchain-google-genai` are required for the Gemini LLM integration. Make sure your environment has network access and credentials.

3) Set your Google API key

PowerShell (temporary for the session):

```powershell
$env:GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
```

PowerShell (persist across sessions):

```powershell
setx GOOGLE_API_KEY "YOUR_GOOGLE_API_KEY"
# then open a new terminal window to use it
```

Bash:

```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

4) Start the app

```powershell
python main.py
```

Open the Gradio link printed in the terminal (usually `http://127.0.0.1:7860`).

Usage
- Ask questions in the format: `In database <db_id>, <natural language question>` (e.g., `In database concert_singer, list the singers with more than 3 performances`).
- Click `Generate SQL` to retrieve a generated SQL snippet (few-shot examples and schema injected).
- Edit the SQL in the "Generated SQL" box if you want to change it.
- Click `Execute SQL` to run it. Results are shown in the `Execution Results` box and logged to `conversation_log.jsonl`.
- Click `Show Schema` to view the currently selected DB schema.

Troubleshooting
- `ModuleNotFoundError: No module named 'pandas'` or similar → activate your virtual environment and run `pip install -r requirements.txt`.
- `GOOGLE_API_KEY` missing → ensure the environment variable is set in the same shell where you run `python main.py`.
- If you see LLM-related errors (authentication, API changes), check the versions of `langchain`, `langchain-google-genai`, and `google-generativeai`.

Kaggle / Spider dataset
- You already downloaded the Spider dataset into `spider/`. `train_spider.json` and the `database/` folder should be present.
- `main.py` will attempt to create per-database SQLite files from `schema.sql` in each database folder (if present).

Security note
- Do not commit your `GOOGLE_API_KEY` to source control. If you accidentally exposed it in a public repo, rotate the key immediately.

Next steps I can do for you
- (A) Run `pip install -r requirements.txt` inside a created virtualenv here (I can run it if you want me to).
- (B) Launch `main.py` and capture any runtime errors.
- (C) Add a small `run.ps1` script that creates the venv and installs dependencies for you.

Tell me which you'd like me to do next.