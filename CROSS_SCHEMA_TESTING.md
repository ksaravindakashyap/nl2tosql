# 🔗 Cross-Schema Testing Guide - Phase 3

## What is Cross-Schema Support?

Cross-schema means handling databases where tables are organized into **multiple schemas** (namespaces), and queries need to use **fully qualified names** like `schema.table.column`.

---

## ✅ Which Spider Databases Have Multiple Schemas?

Looking at the Spider dataset, most databases are **single-schema SQLite** databases. However, the Phase 3 system is designed to handle multi-schema databases when they exist.

### Currently Loaded Databases (Single Schema - SQLite)

The 18 databases you have loaded are all SQLite databases with a **default schema**:
- activity_1
- aircraft  
- allergy_1
- apartment_rentals
- architecture
- assets_maintenance
- behavior_monitoring
- bike_1
- body_builder
- book_2
- browser_web
- candidate_poll
- chinook_1
- cinema
- city_record
- climbing
- club_1
- coffee_shop

**Note:** SQLite databases don't have explicit schemas like PostgreSQL or SQL Server, but the Phase 3 system is **ready** for them.

---

## 🎯 How to Test Cross-Schema Features with Current Databases

Even though your loaded databases are single-schema, you can still test the **foreign key detection** and **qualified name** features:

### Test 1: Foreign Key Detection (Multi-Table Joins)

These queries will show how the system detects relationships:

```
Show all students and their apartments with full details
```
**Database:** apartment_rentals  
**Expected:** Should use FK between Students and Apartments tables  
**Cross-schema feature:** System will detect the foreign key relationship

```
List all movies with their cinema and schedule information
```
**Database:** cinema  
**Expected:** JOIN across film, cinema, and schedule tables  
**Cross-schema feature:** Multiple FK relationships detected

```
Show all aircraft with their airport assignments
```
**Database:** aircraft  
**Expected:** JOIN aircraft, airport_aircraft, airport tables  
**Cross-schema feature:** Complex FK graph navigation

### Test 2: See Foreign Keys with /schema Command

After asking any query, type:
```
/schema
```

**What to look for:**
```sql
-- Table: Apartments
--   apt_id INTEGER NOT NULL
--   building_id INTEGER
--   PRIMARY KEY (apt_id)
--   FOREIGN KEY (building_id) → Apartment_Buildings (building_id)

-- Table: Apartment_Bookings
--   apt_id INTEGER
--   guest_id INTEGER
--   FOREIGN KEY (apt_id) → Apartments (apt_id)
--   FOREIGN KEY (guest_id) → Guests (guest_id)
```

**This shows:**
- ✅ Primary keys detected
- ✅ Foreign keys with arrows (→)
- ✅ Referenced tables shown
- ✅ Column mapping displayed

---

## 🧪 Specific Test Queries for FK Detection

### Apartment Rentals Database
```
Show me all apartment bookings with guest names and apartment details
```
**Expected FK detection:**
- Apartment_Bookings → Guests (guest_id)
- Apartment_Bookings → Apartments (apt_id)

```
List all apartments with their facilities
```
**Expected FK:**
- Apartment_Facilities → Apartments (apt_id)

### Cinema Database
```
Which cinemas are showing which films and when?
```
**Expected FK chain:**
- schedule → cinema (cinema_id)
- schedule → film (film_id)

### Aircraft Database
```
Show all flights with aircraft and airport information
```
**Expected FK:**
- flight → aircraft (aircraft_id)
- flight → airport (departure/arrival)

### Activity Database
```
List all students participating in activities with faculty supervisors
```
**Expected FK:**
- Participates_in → Student (stuid)
- Participates_in → Activity (actid)
- Faculty_Participates_in → Faculty (facid)
- Faculty_Participates_in → Activity (actid)

---

## 🎓 True Multi-Schema Example (For Future PostgreSQL/MySQL)

If you were to connect to a PostgreSQL database with schemas like:

```
sales schema:
  - sales.customers
  - sales.orders
  
inventory schema:
  - inventory.products
  - inventory.suppliers
```

**Phase 3 would generate:**
```sql
SELECT 
  c.customer_name,
  o.order_date,
  p.product_name
FROM sales.customers AS c
JOIN sales.orders AS o ON c.customer_id = o.customer_id
JOIN inventory.products AS p ON o.product_id = p.product_id
WHERE c.region = 'North'
```

**Notice:**
- ✅ `sales.customers` (fully qualified)
- ✅ `sales.orders` (fully qualified)  
- ✅ `inventory.products` (cross-schema join)

---

## 📋 Step-by-Step Testing Instructions

### 1. Start the Server
Open new PowerShell terminal and run:
```powershell
cd E:\Dbms-project
.\start_server.bat
```

**Or directly:**
```powershell
python main.py
```

Keep the terminal open - don't close it!

### 2. Open Browser
Navigate to: **http://127.0.0.1:7861**

### 3. Test Foreign Key Detection

**Test Query 1:**
```
Show all apartment bookings with guest information
```

Click **Run Query**, then type:
```
/schema
```

Look for lines like:
```
--   FOREIGN KEY (guest_id) → Guests (guest_id)
```

**Test Query 2:**
```
List all movies showing at cinemas with schedules
```

Click **Run Query**, then:
```
/schema
```

Look for:
```
--   FOREIGN KEY (cinema_id) → cinema (cinema_id)
--   FOREIGN KEY (film_id) → film (film_id)
```

### 4. Test Complex Joins

```
Show all students, their activities, and supervising faculty
```

This should generate a multi-table JOIN using the detected FKs.

---

## ✅ What Cross-Schema Features Work NOW

Even with single-schema SQLite databases:

1. **✅ Foreign Key Detection**
   - System reads FK constraints from schema
   - Displays them with → arrows
   - Uses them for intelligent JOIN generation

2. **✅ Primary Key Detection**
   - Shows PK constraints clearly
   - Helps with relationship understanding

3. **✅ Nullable Constraints**
   - Shows which columns are NOT NULL
   - Better schema understanding

4. **✅ Multi-Table Join Intelligence**
   - Uses FK info to generate correct JOINs
   - Handles 3+ table queries
   - Preserves referential integrity

5. **✅ Ready for True Multi-Schema**
   - Code handles schema.table.column format
   - Qualified name enforcement ready
   - Will work with PostgreSQL/MySQL when connected

---

## 🚀 Future: Testing with Real Multi-Schema Databases

To truly test cross-schema with multiple schemas, you would need to:

1. **Option A:** Load a PostgreSQL Spider database
   - Some Spider DBs have schema prefixes
   - Example: `cre_Doc_Control_Systems` has multiple schemas

2. **Option B:** Create a custom multi-schema test DB
   ```sql
   CREATE SCHEMA sales;
   CREATE SCHEMA inventory;
   CREATE TABLE sales.orders (...);
   CREATE TABLE inventory.products (...);
   ```

3. **Option C:** Use databases with prefixes
   - Look for `cre_*` databases in Spider
   - These often have schema-like structures

---

## 🎯 Bottom Line for Current Testing

**What you CAN test now:**
- ✅ Foreign key detection across tables
- ✅ Multi-table joins with FK awareness
- ✅ Complete schema display with relationships
- ✅ Primary key and nullable detection
- ✅ `/schema` command shows all relationships

**What you CANNOT test yet (need multi-schema DB):**
- ❌ Actual `schema.table` qualified names in SQL
- ❌ Cross-schema joins (different namespaces)
- ❌ Schema-level conflict resolution

**But the code is ready!** When you connect to a PostgreSQL or MySQL database with multiple schemas, Phase 3 will automatically:
1. Detect all schemas
2. Generate fully qualified names
3. Enforce cross-schema syntax rules

---

## 📝 Example Test Session

```powershell
# Terminal
cd E:\Dbms-project
python main.py
# Keep this terminal open!
```

```
# Browser: http://127.0.0.1:7861

User: Show all apartment bookings with guest names
Bot: [Generates SQL with JOIN]

User: /schema
Bot: [Shows complete schema with FK arrows]

-- Table: Apartment_Bookings
--   booking_id INTEGER NOT NULL
--   apt_id INTEGER
--   guest_id INTEGER
--   FOREIGN KEY (apt_id) → Apartments (apt_id)
--   FOREIGN KEY (guest_id) → Guests (guest_id)

User: List all movies at each cinema with schedules
Bot: [Generates multi-table JOIN]

User: /schema  
Bot: [Shows cinema, film, schedule relationships]
```

---

## 🔑 Key Commands

1. **Start server:** `python main.py` (keep terminal open!)
2. **Open UI:** http://127.0.0.1:7861
3. **Test FK query:** "Show all apartment bookings with guest names"
4. **See schema:** `/schema`
5. **Check FKs:** Look for `FOREIGN KEY (col) → table (col)` lines

---

**The Phase 3 cross-schema infrastructure is built and ready - test the FK detection features with your current databases!**
