import sqlite3
import pandas as pd

# merge using sql, because of memory error
db_file = "../output/temp_db.sqlite"
conn = sqlite3.connect(db_file)
#strSQL = 'SELECT * FROM train t INNER JOIN yeah y ON ;'
# MIGHT HAVE TO ADJUST ABOVE FOR CELL AND PATIENT PARAMS IN DEFINED FUNCTION
strSQL = 'SELECT * FROM train t INNER JOIN yeah y  '
merged = pd.read_sql(strSQL, conn)
print(merged.tail())
strSQL = 'SELECT * FROM yeah t ;'
merged = pd.read_sql(strSQL, conn)
print(merged.tail())