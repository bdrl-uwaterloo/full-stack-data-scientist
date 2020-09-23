import mysql.connector

#connect to database server
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="password"
)
print(conn)

#create a database
mycursor = conn.cursor()
mycursor.execute("CREATE DATABASE myfirstdatabase")
mycursor.execute("SHOW DATABASES")
for db in mycursor:
    print(db)

#create a table
mycursor.execute("USE myfirstdatabase")
sql = "CREATE TABLE Persons (id INT AUTO_INCREMENT PRIMARY KEY, LastName VARCHAR(255) NOT NULL, FirstName VARCHAR(255), Age INT)"
mycursor.execute(sql)
mycursor.execute("SHOW TABLES")
for table in mycursor:
    print(table)

#insert a single line
sql = "INSERT INTO Persons (LastName, FirstName, Age) VALUES (%s, %s, %d)"
val = ("Sam", "Witwicky", 21)
mycursor.execute(sql, val)

#insert many lines
sql = "INSERT INTO Persons (LastName, FirstName, Age) VALUES (%s, %s, %d)"
val = [
    ("Peter", "Parker", 20),
    ("Harry", "Potter", 16),
    ("Amy", "Wang", 22),
    ("John", "Taylor", 20)
]
mycursor.executemany(sql, val)
conn.commit()

#update data
sql = "UPDATE Persons SET LastName = 'Henry' WHERE LastName = 'Taylor'"
mycursor.execute(sql)
conn.commit()

#delete data
sql = "DELETE FROM Persons WHERE LastName = 'Wang'"
mycursor.execute(sql)
conn.commit()

#select from a table
sql = "SELECT ORDERNUMBER, STATUS FROM myfirstdatabase.sales_data_sample"
sql = "SELECT * FROM myfirstdatabase.sales_data_sample"
sql = "SELECT ORDERNUMBER, STATUS FROM myfirstdatabase.sales_data_sample ORDER BY STATUS"
sql = "SELECT * FROM myfirstdatabase.sales_data_sample ORDER BY ORDERNUMBER, STATUS DESC"
result = mycursor.fetchall()
for rows in result:
    print(rows)

#filter data
sql = "SELECT * FROM myfirstdatabase.sales_data_sample WHERE COUNTRY = 'USA' OR COUNTRY = 'France'"
sql = "SELECT * FROM myfirstdatabase.sales_data_sample WHERE COUNTRY = 'USA' AND STATE = 'NY'"
sql = "SELECT * FROM myfirstdatabase.sales_data_sample WHERE YEAR_ID BETWEEN 2003 AND 2005"
sql = "SELECT * FROM myfirstdatabase.sales_data_sample WHERE YEAR_ID IN (2003, 2005, 2006)"
sql = "SELECT * FROM myfirstdatabase.sales_data_sample WHERE YEAR_ID < 2008"
sql = "SELECT * FROM myfirstdatabase.sales_data_sample WHERE STATE IS NULL"
sql = "SELECT * FROM myfirstdatabase.sales_data_sample WHERE PRODUCTCODE LIKE 'S10%'"

#group data
sql = "SELECT STATUS, COUNT(*) FROM Sales_Data_Sample GROUP BY STATUS"
sql = "SELECT YEAR, COUNT(*) FROM Sales_Data_Sample GROUP BY YEAR HAVING YEAR > 2004"

#join table
sql = "SELECT sd.ORDERNUMBER, sd.PRODUCTLINE, pl.MAINBUS FROM myfirstdatabase.sales_data_sample AS sd INNER JOIN myfirstdatabase.product_line AS pl ON sd.PRODUCTLINE = pl.PRODUCTLINE"

#subquery
sql = "SELECT sd.ORDERNUMBER, sd.PRODUCTLINE FROM myfirstdatabase.sales_data_sample AS sd WHERE sd.PRODUCTLINE IN (SELECT pl.PRODUCTLINE FROM myfirstdatabase.product_line AS pl WHERE pl.MAINBUS = 1)"