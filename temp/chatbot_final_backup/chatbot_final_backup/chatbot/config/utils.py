import pymysql
import random
import string
import re
import platform
import os
from setuptools import glob


def connect_to_database(host, username, password, database):
    return pymysql.connect(
        user=username,
        passwd=password,
        host=host,
        database=database,
        cursorclass=pymysql.cursors.DictCursor
    )


def execute_sql_query(connection, query, values=None):
    cursor = connection.cursor()

    try:
        if values:
            cursor.execute(query, values)
        else:
            cursor.execute(query)

        results = cursor.fetchall() if cursor.description else None
        connection.commit()
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        results = None
    finally:
        cursor.close()

    return results


if __name__ == "__main__":
    print("To test!")
