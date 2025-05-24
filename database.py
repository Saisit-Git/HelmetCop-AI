import sqlite3
import os

DB_NAME = 'imagesprocessing_db'

def init_db():
    if not os.path.exists(DB_NAME):
        with sqlite3.connect(DB_NAME) as connectionsql:
            cursor = connectionsql.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detecthelmet (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL
                )
            ''')
            connectionsql.commit()

def insert_violation_log(timestamp, img_path):
    with sqlite3.connect(DB_NAME) as connectionsql:
        cursor = connectionsql.cursor()
        cursor.execute("INSERT INTO detecthelmet (timestamp, image_path) VALUES (%s, %s)", (timestamp, img_path))
        connectionsql.commit()

def get_logs():
    with sqlite3.connect(DB_NAME) as connectionsql:
        cursor = connectionsql.cursor()
        cursor.execute('SELECT * FROM detecthelmet ORDER BY timestamp DESC')
        return cursor.fetchall()