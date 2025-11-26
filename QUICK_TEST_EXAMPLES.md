# 🧪 Quick Test Examples with Expected Answers

## 1. apartment_rentals Database

### Query 1: List all guests
**Prompt:** `List all guests with their names and date of birth`

**Expected SQL:**
```sql
SELECT 
    guest_id,
    guest_first_name,
    guest_last_name,
    date_of_birth
FROM Guests;
```

**Expected Result:** 15 guests (Kip DuBuque, Rebeca Runolfsdottir, etc.)

---

### Query 2: Apartments in specific building
**Prompt:** `Show all apartments in building 225 with their type and room count`

**Expected SQL:**
```sql
SELECT 
    apt_id,
    apt_type_code,
    apt_number,
    room_count,
    bedroom_count,
    bathroom_count
FROM Apartments
WHERE building_id = 225;
```

**Expected Result:** Multiple apartments (apt_id: 3, 4, 10, 14)

---

### Query 3: Confirmed bookings only
**Prompt:** `Show all confirmed bookings with guest names`

**Expected SQL:**
```sql
SELECT 
    ab.apt_booking_id,
    g.guest_first_name || ' ' || g.guest_last_name AS guest_name,
    ab.booking_start_date,
    ab.booking_end_date
FROM Apartment_Bookings ab
JOIN Guests g ON ab.guest_id = g.guest_id
WHERE ab.booking_status_code = 'Confirmed';
```

**Expected Result:** 6 confirmed bookings (booking IDs: 343, 365, 497, 889, 920, 924)

---

### Query 4: Count bookings per guest
**Prompt:** `How many bookings does each guest have?`

**Expected SQL:**
```sql
SELECT 
    g.guest_first_name || ' ' || g.guest_last_name AS guest_name,
    COUNT(ab.apt_booking_id) AS booking_count
FROM Guests g
LEFT JOIN Apartment_Bookings ab ON g.guest_id = ab.guest_id
GROUP BY g.guest_id, g.guest_first_name, g.guest_last_name
ORDER BY booking_count DESC;
```

**Expected Result:** Guest #2 (Rebeca Runolfsdottir) has 4 bookings, Guest #5 (Lou Grady) has 2, etc.

---

## 2. cinema Database

### Query 5: All movies with their titles
**Prompt:** `List all movies with their titles and directors`

**Expected SQL:**
```sql
SELECT 
    movie_id,
    title,
    director
FROM movie;
```

**Expected Result:** List of movies (Harry Potter, The Avengers, etc.)

---

### Query 6: Movie schedules
**Prompt:** `Show all movie schedules with film titles and show times`

**Expected SQL:**
```sql
SELECT 
    s.schedule_id,
    m.title,
    s.date,
    s.time,
    s.price
FROM schedule s
JOIN movie m ON s.movie_id = m.movie_id
ORDER BY s.date, s.time;
```

**Expected Result:** Movie schedules with dates and times

---

### Query 7: Count schedules per movie
**Prompt:** `How many schedules does each movie have?`

**Expected SQL:**
```sql
SELECT 
    m.title,
    COUNT(s.schedule_id) AS schedule_count
FROM movie m
LEFT JOIN schedule s ON m.movie_id = s.movie_id
GROUP BY m.movie_id, m.title
ORDER BY schedule_count DESC;
```

**Expected Result:** Movies with their schedule counts

---

## 3. concert_singer Database

### Query 8: All singers
**Prompt:** `List all singers with their names and countries`

**Expected SQL:**
```sql
SELECT 
    Singer_ID,
    Name,
    Country,
    Age
FROM singer;
```

**Expected Result:** Multiple singers from different countries

---

### Query 9: Concerts with stadium info
**Prompt:** `Show all concerts with stadium names and years`

**Expected SQL:**
```sql
SELECT 
    c.concert_ID,
    c.concert_Name,
    c.Year,
    s.Name AS stadium_name,
    s.Location
FROM concert c
JOIN stadium s ON c.Stadium_ID = s.Stadium_ID
ORDER BY c.Year;
```

**Expected Result:** Concerts with their venues

---

### Query 10: Singers per concert
**Prompt:** `Which singers performed in which concerts?`

**Expected SQL:**
```sql
SELECT 
    c.concert_Name,
    s.Name AS singer_name
FROM concert c
JOIN singer_in_concert sic ON c.concert_ID = sic.concert_ID
JOIN singer s ON sic.Singer_ID = s.Singer_ID
ORDER BY c.concert_Name;
```

**Expected Result:** Concert-singer mappings

---

## 4. Cross-Schema Tests (apartment_rentals)

### Query 11: Buildings with most apartments
**Prompt:** `Which buildings have the most apartments?`

**Expected SQL:**
```sql
SELECT 
    ab.building_full_name,
    ab.building_address,
    COUNT(a.apt_id) AS apartment_count
FROM Apartment_Buildings ab
LEFT JOIN Apartments a ON ab.building_id = a.building_id
GROUP BY ab.building_id, ab.building_full_name, ab.building_address
ORDER BY apartment_count DESC
LIMIT 5;
```

**Expected Result:** Top 5 buildings (building 225 has 4 apartments, etc.)

---

### Query 12: Apartment facilities
**Prompt:** `What facilities does apartment 1 have?`

**Expected SQL:**
```sql
SELECT 
    a.apt_number,
    af.facility_code
FROM Apartments a
LEFT JOIN Apartment_Facilities af ON a.apt_id = af.apt_id
WHERE a.apt_id = 1;
```

**Expected Result:** Apartment Suite 645 has Broadband

---

### Query 13: Guest booking history
**Prompt:** `Show booking history for guest Kip DuBuque`

**Expected SQL:**
```sql
SELECT 
    g.guest_first_name || ' ' || g.guest_last_name AS guest_name,
    ab.apt_booking_id,
    a.apt_number,
    ab.booking_start_date,
    ab.booking_end_date,
    ab.booking_status_code
FROM Guests g
JOIN Apartment_Bookings ab ON g.guest_id = ab.guest_id
JOIN Apartments a ON ab.apt_id = a.apt_id
WHERE g.guest_first_name = 'Kip' AND g.guest_last_name = 'DuBuque';
```

**Expected Result:** No bookings (guest #1 has no bookings in the data)

---

## 5. Aggregate Queries

### Query 14: Average rooms per apartment type
**Prompt:** `What's the average number of rooms for each apartment type?`

**Expected SQL:**
```sql
SELECT 
    apt_type_code,
    AVG(CAST(room_count AS REAL)) AS avg_rooms,
    COUNT(*) AS apartment_count
FROM Apartments
GROUP BY apt_type_code
ORDER BY avg_rooms DESC;
```

**Expected Result:** Duplex ~6.8, Flat ~6.8, Studio ~7.6 rooms on average

---

### Query 15: Longest bookings
**Prompt:** `Show the 5 longest apartment bookings`

**Expected SQL:**
```sql
SELECT 
    ab.apt_booking_id,
    g.guest_first_name || ' ' || g.guest_last_name AS guest_name,
    ab.booking_start_date,
    ab.booking_end_date,
    JULIANDAY(ab.booking_end_date) - JULIANDAY(ab.booking_start_date) AS days
FROM Apartment_Bookings ab
JOIN Guests g ON ab.guest_id = ab.guest_id
ORDER BY days DESC
LIMIT 5;
```

**Expected Result:** Longest bookings (500+ days)

---

## 6. Subquery Tests

### Query 16: Guests who never booked
**Prompt:** `Which guests have never made a booking?`

**Expected SQL:**
```sql
SELECT 
    guest_id,
    guest_first_name,
    guest_last_name
FROM Guests
WHERE guest_id NOT IN (
    SELECT DISTINCT guest_id 
    FROM Apartment_Bookings
);
```

**Expected Result:** Guests 1, 6, 9, 10, 11 (5 guests with no bookings)

---

### Query 17: Most popular apartment
**Prompt:** `Which apartment has the most bookings?`

**Expected SQL:**
```sql
SELECT 
    a.apt_number,
    a.apt_type_code,
    COUNT(ab.apt_booking_id) AS booking_count
FROM Apartments a
LEFT JOIN Apartment_Bookings ab ON a.apt_id = ab.apt_id
GROUP BY a.apt_id, a.apt_number, a.apt_type_code
ORDER BY booking_count DESC
LIMIT 1;
```

**Expected Result:** Apartments 8 and 10 (3 bookings each)

---

## 7. Date Range Queries

### Query 18: Bookings in 2017
**Prompt:** `Show all bookings that start in 2017`

**Expected SQL:**
```sql
SELECT 
    ab.apt_booking_id,
    g.guest_first_name || ' ' || g.guest_last_name AS guest_name,
    ab.booking_start_date,
    ab.booking_status_code
FROM Apartment_Bookings ab
JOIN Guests g ON ab.guest_id = g.guest_id
WHERE ab.booking_start_date >= '2017-01-01' 
  AND ab.booking_start_date < '2018-01-01'
ORDER BY ab.booking_start_date;
```

**Expected Result:** 5 bookings in 2017

---

## 8. Multiple Database Test

### Query 19: Database detection
**Prompt:** `In database cinema, show all movies`

**Expected SQL:**
```sql
SELECT * FROM movie;
```

**Expected Database:** cinema (not apartment_rentals)

---

### Query 20: Automatic detection
**Prompt:** `Which singers are from USA?`

**Expected Database:** concert_singer (auto-detected)

**Expected SQL:**
```sql
SELECT 
    Singer_ID,
    Name,
    Age
FROM singer
WHERE Country = 'USA';
```

---

## 🎯 Testing Checklist

For each query, verify:
- ✅ **No `default.` prefix** in SQL
- ✅ **Correct table names** (case-sensitive: `Guests` not `guests`)
- ✅ **Correct column names** from actual schema
- ✅ **Proper JOINs** using foreign keys
- ✅ **Database auto-detected** correctly
- ✅ **Query executes** without errors
- ✅ **Results make sense** (row count, values)

## 🔍 How to Test

1. **Enter query** in Chat tab
2. **Check SQL** - Should have no `default.` prefix
3. **Run Query** - Click execute button
4. **Verify results** - Check row count and values
5. **Test /schema** - Type `/schema apartment_rentals` to see FK relationships

## 💡 Pro Tips

- Use `/schema <database>` to explore any database
- Try variations: "confirmed bookings", "provisional bookings"
- Test edge cases: "guests with no bookings", "empty apartments"
- Mix databases: Ask about singers, then apartments, then movies
- Test aggregates: COUNT, AVG, SUM, MAX, MIN
- Test sorting: "top 5", "most popular", "longest"

---

**All expected results are based on the actual Spider dataset!** 🚀
