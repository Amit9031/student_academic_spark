"""Catalogue of individually runnable Spark SQL demonstrations.

Views available to every query (registered by spark_sql_analysis.py):
  students      - student-mat.csv plus a generated student_id (1..395)
  performance   - derived table: student_id, category, support_action
                  (only students 1-300, plus two ids that do not exist in students)
  column_names  - one row per column of `students` (so LIKE can be used on names)

Each entry: (id, title, concept, sql)
"""

QUERIES = [
    (1, "Preview records", "SELECT, LIMIT",
     "SELECT student_id, school, sex, age, studytime, G1, G2, G3 FROM students LIMIT 10"),

    (2, "Distinct study-time levels", "DISTINCT",
     "SELECT DISTINCT studytime FROM students ORDER BY studytime"),

    (3, "High achievers with no failures", "WHERE with AND",
     "SELECT student_id, school, G1, G2, G3 FROM students "
     "WHERE G3 >= 15 AND failures = 0 ORDER BY G3 DESC, student_id LIMIT 10"),

    (4, "Young students with low absences", "WHERE with IN and BETWEEN",
     "SELECT student_id, age, absences, G3 FROM students "
     "WHERE age IN (15, 16) AND absences BETWEEN 0 AND 5 LIMIT 10"),

    (5, "LIKE on values: mother's job starts with 't'", "LIKE with % wildcard",
     "SELECT Mjob, COUNT(*) AS students FROM students WHERE Mjob LIKE 't%' GROUP BY Mjob"),

    (6, "LIKE on values: father's job matches '_ther'", "LIKE with _ (one character)",
     "SELECT Fjob, COUNT(*) AS students FROM students WHERE Fjob LIKE '_ther' GROUP BY Fjob"),

    (7, "NOT LIKE: mother's job does not end with 'home'", "NOT LIKE",
     "SELECT Mjob, COUNT(*) AS students FROM students "
     "WHERE Mjob NOT LIKE '%home' GROUP BY Mjob ORDER BY students DESC"),

    (8, "Column names matching 'G%'", "LIKE on column NAMES (via column_names view)",
     "SELECT column_name FROM column_names WHERE column_name LIKE 'G%'"),

    (9, "Column names matching '_edu'", "LIKE with _ on column names",
     "SELECT column_name FROM column_names WHERE column_name LIKE '_edu'"),

    (10, "Column names matching '_P%'", "Pattern that matches nothing (0 rows is a valid answer)",
     "SELECT column_name FROM column_names WHERE column_name LIKE '_P%'"),

    (11, "Inspect all columns with SHOW COLUMNS", "SHOW COLUMNS",
     "SHOW COLUMNS IN students"),

    (12, "Top students ordered by G3 then absences", "ORDER BY multiple keys",
     "SELECT student_id, G3, absences FROM students "
     "ORDER BY G3 DESC, absences ASC, student_id LIMIT 10"),

    (13, "Overall statistics", "COUNT, SUM, AVG, MIN, MAX",
     "SELECT COUNT(*) AS students, SUM(absences) AS total_absences, "
     "ROUND(AVG(G3), 2) AS avg_g3, MIN(G3) AS min_g3, MAX(G3) AS max_g3 FROM students"),

    (14, "Average marks by sex", "GROUP BY",
     "SELECT sex, COUNT(*) AS students, ROUND(AVG(G1),2) AS avg_g1, "
     "ROUND(AVG(G2),2) AS avg_g2, ROUND(AVG(G3),2) AS avg_g3 FROM students GROUP BY sex"),

    (15, "Average G3 by school and sex", "GROUP BY two columns",
     "SELECT school, sex, COUNT(*) AS students, ROUND(AVG(G3),2) AS avg_g3 "
     "FROM students GROUP BY school, sex ORDER BY school, sex"),

    (16, "Study time versus marks", "GROUP BY + ORDER BY",
     "SELECT studytime, COUNT(*) AS students, ROUND(AVG(G3),2) AS avg_g3 "
     "FROM students GROUP BY studytime ORDER BY studytime"),

    (17, "Study-time groups with average G3 above 10", "HAVING (filters groups, not rows)",
     "SELECT studytime, COUNT(*) AS students, ROUND(AVG(G3),2) AS avg_g3 "
     "FROM students GROUP BY studytime HAVING AVG(G3) > 10 ORDER BY studytime"),

    (18, "Failure counts with at least 5 students", "HAVING COUNT",
     "SELECT failures, COUNT(*) AS students, ROUND(AVG(G3),2) AS avg_g3 "
     "FROM students GROUP BY failures HAVING COUNT(*) >= 5 ORDER BY failures"),

    (19, "Students per performance category", "CASE WHEN + GROUP BY",
     "SELECT CASE WHEN G3 >= 15 THEN 'Excellent' WHEN G3 >= 10 THEN 'Pass' "
     "ELSE 'At-Risk' END AS category, COUNT(*) AS students, ROUND(AVG(absences),2) AS avg_absences "
     "FROM students GROUP BY 1 ORDER BY students DESC"),

    (20, "Students above the overall average G3", "Subquery",
     "SELECT student_id, G3 FROM students WHERE G3 > (SELECT AVG(G3) FROM students) "
     "ORDER BY G3 DESC, student_id LIMIT 10"),

    (21, "Improvement from G1 to G3 by study time", "Calculated column + GROUP BY",
     "SELECT studytime, ROUND(AVG(G3 - G1), 2) AS avg_improvement FROM students "
     "GROUP BY studytime ORDER BY studytime"),

    (22, "INNER JOIN students with performance", "INNER JOIN on student_id",
     "SELECT s.student_id, s.G3, p.category, p.support_action "
     "FROM students s INNER JOIN performance p ON s.student_id = p.student_id "
     "ORDER BY s.student_id LIMIT 10"),

    (23, "LEFT JOIN: every student, category if present", "LEFT JOIN (shows NULLs)",
     "SELECT s.student_id, s.G3, p.category FROM students s "
     "LEFT JOIN performance p ON s.student_id = p.student_id "
     "WHERE s.student_id BETWEEN 298 AND 303 ORDER BY s.student_id"),

    (24, "RIGHT JOIN: every performance row", "RIGHT JOIN (ids 9001/9002 have no student)",
     "SELECT p.student_id, s.G3, p.category FROM students s "
     "RIGHT JOIN performance p ON s.student_id = p.student_id "
     "WHERE p.student_id >= 299 ORDER BY p.student_id"),

    (25, "FULL OUTER JOIN: unmatched rows from both sides", "FULL OUTER JOIN",
     "SELECT COALESCE(s.student_id, p.student_id) AS student_id, s.G3, p.category "
     "FROM students s FULL OUTER JOIN performance p ON s.student_id = p.student_id "
     "WHERE s.student_id IS NULL OR p.student_id IS NULL "
     "ORDER BY student_id LIMIT 10"),

    (26, "Join + aggregate: average absences per category", "JOIN then GROUP BY",
     "SELECT p.category, COUNT(*) AS students, ROUND(AVG(s.absences),2) AS avg_absences "
     "FROM students s INNER JOIN performance p ON s.student_id = p.student_id "
     "GROUP BY p.category ORDER BY students DESC"),
]


def get_query(query_id):
    for q in QUERIES:
        if q[0] == query_id:
            return q
    return None
