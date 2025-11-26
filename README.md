# NL2SQL Spider Copilot - Phase 3 Complete 🎉

**Production-ready Natural Language to SQL system with cross-schema support, Spider leaderboard evaluation, and human-in-the-loop learning.**

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (create .env file)
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 3. Run the application
python main.py

# 4. Open browser
# http://127.0.0.1:7861
```

---

## 📚 Documentation

- **[PHASE3_README.md](PHASE3_README.md)** - Complete feature documentation
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Test cases and validation
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference
- **[PHASE3_SUMMARY.md](PHASE3_SUMMARY.md)** - Implementation details

---

## ✨ Phase 3 Features

### 🎯 Core Capabilities

1. **Cross-Schema SQL Generation**
   - Automatic foreign key detection
   - Fully qualified name enforcement
   - Nullable constraint awareness

2. **Spider Leaderboard Evaluation**
   - Official exact-match metrics
   - Execution accuracy validation
   - Configurable sample size (10-1034)

3. **Human-in-the-Loop Learning**
   - Save user corrections
   - Automatic vectorstore updates
   - Continuous accuracy improvement

4. **Professional 3-Tab UI**
   - 💬 Chat: Main NL2SQL interface
   - 🏆 Leaderboard: Spider evaluation
   - 📊 Statistics: System metrics

---

## 📊 System Architecture

```
Question → Database Router → Schema Extraction (FK + Nullable)
   ↓
Table Pre-Selection (Top-5) → Few-Shot Retrieval (Chroma)
   ↓
SQL Generation (LangChain + OpenAI/Gemini)
   ↓
Human Review & Edit → Execution (SQLAlchemy)
   ↓
Save Correction → Vectorstore Update
```

---

## 📁 Project Structure

```
Dbms-project/
├── main.py                    # Main application (849 lines)
├── spider_eval.py             # Evaluation module
├── requirements.txt           # Dependencies
├── .env                       # API keys (create this)
├── .gitignore                # Git exclusions
├── PHASE3_README.md          # Full documentation
├── TESTING_GUIDE.md          # Test instructions
├── QUICK_REFERENCE.md        # Command reference
├── PHASE3_SUMMARY.md         # Implementation details
├── spider/                   # Spider dataset (download)
│   ├── train_spider.json
│   ├── dev.json
│   ├── tables.json
│   └── database/             # 200+ SQLite databases
├── chroma_db/               # Vectorstore (auto-generated)
├── interaction_log.jsonl    # Query logs
└── saved_corrections.jsonl  # User corrections
```

---

## 🎓 Usage Examples

### Basic Query
```
User: "Show all students with GPA above 3.5"
System: 
  Database: college_1
  SQL: SELECT * FROM student WHERE GPA > 3.5
```

### Edit & Save
```
1. Ask question
2. Review generated SQL
3. Edit if needed
4. Click "💾 Save Correction"
5. Future queries improve automatically
```

### Evaluation
```
1. Go to 🏆 Leaderboard tab
2. Set samples: 100
3. Click "Run Evaluation"
4. Get: Exact-match + Execution accuracy
```

---

## ⚙️ Configuration

**Switch Models** (main.py lines 37-40):
```python
USE_OPENAI = True              # True = OpenAI, False = Gemini
OPENAI_MODEL = "gpt-4o-mini"   # Options: gpt-4o-mini, gpt-4o
GEMINI_MODEL = "gemini-1.5-flash"
TEMPERATURE = 0.0
```

**Adjust Databases** (main.py line 215):
```python
for i, (db_id, db_file) in enumerate(sorted(all_dbs.items())[:20]):  # Change 20 to 200
```

---

## 📈 Expected Performance

| Metric | Target |
|--------|--------|
| Exact Match Accuracy | 72-76% (with corrections) |
| Execution Accuracy | 75-80% |
| Databases Supported | 20 (default), 200+ (configurable) |
| Response Time | < 5 seconds per query |

---

## 🛠️ Requirements

- Python 3.11+
- OpenAI API key (or Google Gemini API key)
- Spider dataset (download from [yale-lily.github.io/spider](https://yale-lily.github.io/spider))
- 4GB+ RAM (for 20 databases)
- Internet connection (for API calls)

---

## 📦 Dependencies

```
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.2.0
chromadb
sqlalchemy
pandas>=2.2.0
gradio
python-dotenv
```

---

## 🧪 Testing

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for:
- Sample queries to test
- Expected results
- Evaluation workflow
- Troubleshooting tips

**Quick Test:**
```
1. python main.py
2. Open http://127.0.0.1:7861
3. Ask: "Show all activities"
4. Click "Run Query"
5. ✅ Should return results from activity_1 database
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | Change `server_port=7861` to `7862` |
| Import errors | `pip install -r requirements.txt` |
| API key error | Check `.env` file exists with `OPENAI_API_KEY=sk-...` |
| Slow startup | Reduce database count (line 215) |
| Database not found | Type: `Use database <db_id>` |

---

## 🎯 Phase Progression

| Feature | Phase 1 | Phase 2 | Phase 3 |
|---------|---------|---------|---------|
| Databases | Single | Multiple (auto-route) | 20+ (configurable to 200) |
| Schema | Basic | Table info | FK + Nullable + Qualified |
| UI | Simple | Chat interface | 3-tab professional |
| Learning | None | Few-shot | User corrections |
| Evaluation | Manual | None | Official Spider metrics |

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- **Spider Dataset:** Yale University
- **LangChain:** LangChain AI
- **Gradio:** Hugging Face
- **OpenAI/Google:** API providers

---

## 📧 Contact

- GitHub: [@ksaravindakashyap](https://github.com/ksaravindakashyap)
- Repository: [Dbms-project](https://github.com/ksaravindakashyap/Dbms-project)

---

## 🎓 For Academic Use

This project is suitable for:
- DBMS course projects
- NLP/Database research
- Machine learning demonstrations
- Software engineering portfolios

**Citation:**
```bibtex
@software{nl2sql_spider_phase3,
  title = {NL2SQL Spider Copilot - Phase 3},
  year = {2025},
  url = {https://github.com/ksaravindakashyap/Dbms-project}
}
```

---

**Status:** ✅ Production-Ready │ Leaderboard-Comparable │ Research-Grade

**Built with ❤️ for DBMS Project - Phase 3**