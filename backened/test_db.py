
from database import get_connection

try:
    connection = get_connection()

    if connection.is_connected():
        print("MySQL connected successfully!")

        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE()")
        print("Database:", cursor.fetchone()[0])

        cursor.close()
        connection.close()

except Exception as e:
    print("Connection failed:", e)
