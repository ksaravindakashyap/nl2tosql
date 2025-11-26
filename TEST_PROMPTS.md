# 🧪 Test Prompts for Phase 3 NL2SQL System

**Server Running:** http://127.0.0.1:7861

---

## 🎯 Quick Test Prompts (Copy & Paste)

### 1️⃣ Basic Single-Table Queries

```
Show all activities
```
**Expected:** activity_1 database, SELECT * FROM Activity

```
List all aircraft
```
**Expected:** aircraft database, SELECT * FROM aircraft

```
What cinemas are there?
```
**Expected:** cinema database, SELECT * FROM cinema

```
Show me all cities
```
**Expected:** city_record database, SELECT * FROM city

---

### 2️⃣ Simple Filtering

```
Find activities with the name "Canoeing"
```
**Expected:** WHERE activity_name = 'Canoeing'

```
Show aircraft with more than 1000 passengers
```
**Expected:** WHERE passengers > 1000

```
List cinemas in London
```
**Expected:** WHERE location = 'London' (or similar)

---

### 3️⃣ Join Queries (Tests FK Detection)

```
Show all students with their apartment details
```
**Expected:** apartment_rentals, JOIN between Students and Apartments

```
List all movies with their cinema information
```
**Expected:** cinema database, JOIN film and cinema tables

```
Show climbing locations with mountain details
```
**Expected:** climbing database, JOIN location and mountain

---

### 4️⃣ Aggregation Queries

```
Count how many activities exist
```
**Expected:** SELECT COUNT(*) FROM Activity

```
What's the average number of passengers per aircraft?
```
**Expected:** SELECT AVG(passengers) FROM aircraft

```
How many students are in each apartment?
```
**Expected:** COUNT with GROUP BY

```
Count cinemas by location
```
**Expected:** GROUP BY location_town

---

### 5️⃣ Ordering & Limiting

```
Show top 5 largest aircraft
```
**Expected:** ORDER BY passengers DESC LIMIT 5

```
List activities alphabetically
```
**Expected:** ORDER BY activity_name

```
Show the 3 newest buildings in architecture database
```
**Expected:** ORDER BY year DESC LIMIT 3

---

### 6️⃣ Complex Multi-Table Queries

```
Find all apartments rented by students from New York
```
**Expected:** Multiple JOINs with WHERE condition

```
Show all cinemas that show films longer than 120 minutes
```
**Expected:** JOIN cinema and film with WHERE duration > 120

```
List students who live in apartments with rent over 1000
```
**Expected:** JOIN with filtering on price

---

### 7️⃣ Special Commands

```
/schema
```
**Expected:** Shows complete schema with PKs, FKs for current database

```
Use database aircraft
```
**Expected:** Switches to aircraft database

```
Use database cinema
```
**Expected:** Switches to cinema database

---

## 🎨 Advanced Test Scenarios

### Test Database Auto-Detection

1. **Ask:** "Show all activities"
2. **Check:** System detects `activity_1` database
3. **Then ask:** "Now show me aircraft"
4. **Check:** System switches to `aircraft` database

### Test SQL Editing

1. **Ask:** "Show all activities"
2. **Edit SQL** to add: `ORDER BY activity_name`
3. **Click:** Run Query
4. **Check:** Results are ordered

### Test Save Correction

1. **Ask:** "List all aircraft"
2. **Edit SQL** to: `SELECT name, passengers FROM aircraft ORDER BY passengers DESC`
3. **Click:** 💾 Save Correction
4. **Check:** Status shows "Correction saved!"
5. **Go to:** Statistics tab
6. **Verify:** Correction count increased

### Test Regeneration

1. **Ask:** "Show activities"
2. **Click:** 🔄 Regenerate SQL
3. **Check:** New SQL is generated
4. **Compare:** May be slightly different

---

## 📊 Expected Results by Database

### activity_1 (5 tables)
- Activity, Faculty, Faculty_Participates_in, Participates_in, Student
- **Test:** "Show all activities and their participating students"
- **Expected:** JOIN Activity with Participates_in

### aircraft (5 tables)
- aircraft, airport, airport_aircraft, employee, flight
- **Test:** "Which airports have Boeing aircraft?"
- **Expected:** JOIN aircraft, airport_aircraft, airport

### apartment_rentals (6 tables)
- Apartments, Apartment_Bookings, Apartment_Facilities, Guests, View_Unit_Status
- **Test:** "Show all guests and their bookings"
- **Expected:** JOIN Guests with Apartment_Bookings

### cinema (3 tables)
- cinema, film, schedule
- **Test:** "What films are showing at each cinema?"
- **Expected:** JOIN all three tables

### city_record (4 tables)
- city, hosting_city, party, party_host
- **Test:** "Which cities are hosting parties?"
- **Expected:** JOIN city with hosting_city

---

## 🎯 Testing Workflow

### Quick 5-Minute Test

1. **Open:** http://127.0.0.1:7861
2. **Try these in order:**
   - "Show all activities"
   - "List all aircraft"
   - "Count how many cinemas exist"
   - "/schema"
   - "Use database climbing"
   - "Show all mountains"

### Full Feature Test (15 minutes)

**Chat Tab:**
- [x] 3-5 basic queries
- [x] 2-3 join queries
- [x] 1-2 aggregation queries
- [x] Test /schema command
- [x] Test database switching
- [x] Edit SQL and run
- [x] Save a correction

**Leaderboard Tab:**
- [x] Set samples to 10
- [x] Click Run Evaluation
- [x] Wait for results
- [x] Check accuracy scores

**Statistics Tab:**
- [x] View database count
- [x] Check corrections saved
- [x] Refresh stats

---

## 🐛 What to Check

### ✅ Good Signs
- SQL generates in < 5 seconds
- Database auto-detected correctly
- SQL is syntactically correct
- Results display properly
- Buttons appear/work
- No Python errors in terminal

### ⚠️ Warning Signs (OK)
- SQLAlchemy warnings (compatibility, non-critical)
- Chroma deprecation warning (future upgrade needed)
- SAWarning for INTEGER (cosmetic, doesn't affect queries)

### ❌ Error Signs (Need fixing)
- Python exceptions/tracebacks
- "Database not found" errors
- SQL syntax errors consistently
- Server crashes
- No response after 30+ seconds

---

## 💡 Pro Tips

### Get Best Results
1. **Be specific:** "Show all students with GPA above 3.5"
2. **Use database context:** "In cinema database, list all films"
3. **Natural language:** "What's the average aircraft capacity?"

### If SQL is Wrong
1. Edit it in the SQL box
2. Click Run Query
3. Click Save Correction (helps future queries!)

### If Database Detection Fails
1. Type: `Use database <db_id>`
2. Then ask your question
3. Example: `Use database aircraft` then `Show all planes`

### See Available Databases
1. Go to Statistics tab
2. Scroll to "Database List"
3. See all 18 loaded databases

---

## 📝 Sample Test Session

```
You: Show all activities
Bot: [Detects activity_1, generates SQL]
You: [Click Run Query]
Bot: [Shows results]

You: /schema
Bot: [Shows complete schema with FKs]

You: Now show me all aircraft
Bot: [Detects aircraft, generates new SQL]
You: [Edit SQL to add ORDER BY name]
You: [Click Run Query]
You: [Click Save Correction]
Bot: ✅ Correction saved!

You: Count how many cinemas exist
Bot: [Detects cinema, generates COUNT query]
You: [Click Run Query]
Bot: [Shows count]

Go to 🏆 Leaderboard tab:
- Set samples: 20
- Click Run Evaluation
- Wait ~2 minutes
- See accuracy scores!

Go to 📊 Statistics tab:
- See 18 databases loaded
- See 1 correction saved
- See model: gpt-4o-mini
```

---

## 🎉 Success Criteria

Your system is working perfectly if:

- ✅ All queries generate SQL
- ✅ SQL executes without errors
- ✅ Database auto-detection works
- ✅ /schema shows foreign keys
- ✅ Save correction works
- ✅ Evaluation tab runs
- ✅ Statistics tab shows data

---

**Server URL:** http://127.0.0.1:7861

**Start testing now!** Try the first query: `Show all activities`
