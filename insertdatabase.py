import mysql.connector
import random

from datetime import date


start_dt = date.today().replace(day=1, month=1).toordinal()
end_dt = date.today().toordinal()

conndb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345",
    database="imagesprocessing_db"
)


def insert_detecthelmet_log(timestamp, img_path):
        cursor = conndb.cursor()
        sql = "INSERT INTO detecthelmet (timestamp, image_path) VALUES (%s, %s)"
        val = (timestamp, img_path)
        cursor.execute(sql, val)
        conndb.commit()

        #print(cursor.rowcount, "record inserted.")

def get_logs():
        cursor = conndb.cursor()
        cursor.execute('SELECT * FROM detecthelmet ORDER BY timestamp DESC')
        return cursor.fetchall()