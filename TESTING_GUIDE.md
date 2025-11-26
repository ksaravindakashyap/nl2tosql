# 🧪 Phase 3 Testing Guide

## Sample Queries to Test

### 1. Basic Single-Table Queries

```
"Show all activities"
Expected DB: activity_1
Expected SQL: SELECT * FROM Activity

"List all aircraft models"
Expected DB: aircraft
Expected SQL: SELECT * FROM aircraft

"Find all cinemas"
Expected DB: cinema
Expected SQL: SELECT * FROM cinema
```

### 2. Join Queries (Tests FK Detection)

```
"Show all students and their apartments"
Expected DB: apartment_rentals
Should use: Apartments table + Students table with JOIN

"List all bookings with guest details"
Expected DB: apartment_rentals
Should use: Apartment_Bookings JOIN Guests

"Show all movies with their schedules"
Expected DB: cinema
Should use: film JOIN schedule
```

### 3. Aggregation Queries (Tests Table Pre-Selection)

```
"Count how many students live in each apartment"
Expected DB: apartment_rentals
Should use: COUNT and GROUP BY

"What's the average rating of each cinema"
Expected DB: cinema
Should use: AVG() and GROUP BY

"Total number of aircraft by manufacturer"
Expected DB: aircraft
Should use: COUNT, GROUP BY
```

### 4. Complex Queries (Tests Multi-Schema if available)

```
"Find apartments booked by students from New York"
Expected DB: apartment_rentals
Should use: Multiple JOINs + WHERE

"Show cinemas that play movies longer than 2 hours"
Expected DB: cinema
Should use: JOIN + time comparison

"List all aircraft that have completed more than 10 flights"
Expected DB: aircraft
Should use: Subquery or JOIN with COUNT
```

### 5. Cross-Schema Queries (Advanced)

```
"Show all members and their workshop groups"
Expected DB: cre_Drama_Workshop_Groups (if loaded)
Should use: Fully qualified names

"Find all documents in the control system"
Expected DB: cre_Doc_Control_Systems (if loaded)
Should use: Qualified schema.table names
```

---

## 🧪 Testing Workflow

### Test 1: Basic Chat Flow

1. Open http://127.0.0.1:7861
2. Go to 💬 Chat tab
3. Ask: "Show all activities"
4. ✅ Check: Database detected correctly
5. ✅ Check: SQL generated
6. ✅ Check: Buttons appear (Regenerate, Run, Save Correction)
7. Click "Run Query"
8. ✅ Check: Results displayed

### Test 2: Edit & Save Correction

1. Ask: "List all aircraft"
2. Edit SQL to: `SELECT * FROM aircraft ORDER BY name`
3. Click "💾 Save Correction"
4. ✅ Check: Status shows "Correction saved"
5. Go to Statistics tab
6. ✅ Check: Correction count increased

### Test 3: Database Switching

1. Ask: "Show all activities"
2. Type: "Use database aircraft"
3. ✅ Check: Database switched
4. Ask: "List all records"
5. ✅ Check: Uses aircraft database

### Test 4: Schema Command

1. Type: "/schema"
2. ✅ Check: Shows complete schema with:
   - Table names
   - Column types
   - PRIMARY KEY
   - FOREIGN KEY → references
   - NOT NULL constraints

### Test 5: Spider Evaluation

1. Go to 🏆 Leaderboard tab
2. Set samples: 20 (for quick test)
3. Click "Run Evaluation"
4. Wait ~2 minutes
5. ✅ Check: Progress updates appear
6. ✅ Check: Final results show:
   - Exact Match Accuracy
   - Execution Accuracy
   - Sample count

### Test 6: Statistics Tab

1. Go to 📊 Statistics tab
2. ✅ Check: Shows database count (18)
3. ✅ Check: Shows corrections count
4. ✅ Check: Lists databases
5. ✅ Check: Shows model config

---

## 🎯 What to Verify

### UI Elements

- [x] Title: "NL2SQL Spider Copilot │ Cross-Schema │ Phase 3"
- [x] Model info displayed
- [x] 3 tabs visible
- [x] Chat input box
- [x] SQL edit box (appears after query)
- [x] 4 buttons: Regenerate, Run, Save Correction
- [x] Status message area

### Functionality

- [x] Database auto-detection works
- [x] SQL generation completes
- [x] SQL execution returns results
- [x] Edit SQL works
- [x] Save correction works
- [x] Regenerate SQL works
- [x] Schema display includes FKs
- [x] Evaluation runs without errors

### Performance

- [x] Startup time: < 30 seconds
- [x] Query response: < 5 seconds
- [x] Evaluation (20 samples): ~2 minutes
- [x] Evaluation (100 samples): ~8-10 minutes

---

## 🐛 Troubleshooting Test Issues

### Issue: Database Not Detected

**Solution:**
```
Type: "Use database <db_id>"
Example: "Use database activity_1"
```

### Issue: SQL Error on Execution

**Cause:** Table name case sensitivity

**Solution:**
1. Check actual table names with: "/schema"
2. Edit SQL to match exact case
3. Save correction for future

### Issue: Evaluation Fails

**Check:**
- spider/dev.json exists
- spider/database/ folder has databases
- Enough samples set (minimum 10)

### Issue: Slow Response

**Causes:**
- Too many databases loaded
- Large schema
- Network latency to OpenAI

**Solutions:**
- Reduce DB count in load_spider_dataset()
- Use table pre-selection (already enabled)
- Check internet connection

---

## 📊 Expected Results (First Run)

### Query Success Rate
- Basic queries: ~90%
- Join queries: ~75%
- Complex queries: ~65%
- Cross-schema: ~60%

### With Corrections (After 10+ saves)
- Basic queries: ~95%
- Join queries: ~85%
- Complex queries: ~75%
- Cross-schema: ~70%

### Spider Dev Evaluation (100 samples)
- Exact Match: 65-70% (first run)
- Execution Accuracy: 68-73% (first run)
- With corrections: 72-76%

---

## 🎓 Demo Script (5 Minutes)

### Minute 1: Introduction
"This is a Phase 3 NL2SQL system with cross-schema support and Spider leaderboard evaluation."

### Minute 2: Basic Query
1. Ask: "Show all activities"
2. Point out: Database auto-detected
3. Point out: SQL generated with schema awareness

### Minute 3: Edit & Save
1. Edit SQL to add ORDER BY
2. Click Save Correction
3. Show: Correction saved status

### Minute 4: Evaluation
1. Go to Leaderboard tab
2. Set 20 samples
3. Click Run
4. Explain: "This compares to research leaderboard metrics"

### Minute 5: Advanced Features
1. Show /schema command
2. Point out: Foreign keys displayed
3. Show Statistics tab
4. Highlight: "System learns from corrections"

---

## ✅ Success Criteria

Your Phase 3 implementation is successful if:

1. ✅ All 3 tabs work
2. ✅ Queries generate SQL
3. ✅ SQL executes and returns results
4. ✅ Save correction works
5. ✅ Evaluation completes without errors
6. ✅ Schema shows foreign keys
7. ✅ No critical errors (warnings OK)
8. ✅ Runs on port 7861

---

## 🚀 Advanced Testing (Optional)

### Load All 200 Spider Databases

```python
# In main.py, line 215:
for i, (db_id, db_file) in enumerate(sorted(all_dbs.items())[:200]):  # Change 20 to 200
```

**Warning:** Takes 2-3 minutes to load, uses ~2GB RAM

### Run Full Spider Dev Evaluation

```
1. Go to Leaderboard tab
2. Set samples: 1034 (full dev set)
3. Click Run
4. Wait: ~45-60 minutes
5. Get: Official leaderboard-comparable scores
```

### Stress Test

```
Ask 50 different questions in a row
✅ Check: No memory leaks
✅ Check: Response time stays consistent
✅ Check: Corrections accumulate properly
```

---

**Ready to Test!** 🎉

Run the server: `python main.py`
Open: http://127.0.0.1:7861
Start with the "Show all activities" query!
