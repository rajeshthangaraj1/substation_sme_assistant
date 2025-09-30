from config.constant import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USERNAME,
    MYSQL_PASSWORD,
    MYSQL_DATABASE
)
import pymysql
from model.feedback import feedback, feedback_update
from logger_config import setup_logger
logger = setup_logger()
def connect_to_database(host, username, password, database):
    return pymysql.connect(
        user=username,
        passwd=password,
        host=host,
        database=database,
        cursorclass=pymysql.cursors.DictCursor
    )


class FeedbackService:
    def __init__(self, logging_service: logger):
        self.logging_service = logging_service
        self.connection = connect_to_database(MYSQL_HOST, MYSQL_USERNAME, MYSQL_PASSWORD,
                                              MYSQL_DATABASE)

    def submit(self, request: feedback):
        try:
            with self.connection.cursor() as cursor:
                sql = """ insert into cog_bot_feedback (usr_id,usr_quest , usr_ans , session_id ,is_like, sql_query, collection_name, temp_file_name)
                values (%s, %s, %s,%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    request.usr_id, request.usr_quest, request.usr_ans, request.session_id, request.is_like , request.sql_query , request.collection_name , request.temp_file_name))
                self.connection.commit()

                feedback_id = cursor.lastrowid
                cursor.execute("SELECT * FROM cog_bot_feedback WHERE id = %s", (feedback_id,))
                return cursor.fetchone()

        except Exception as e:
            print(f"Exception submit {str(e)}")
        finally:
            self.connection.close()

    def update(self, request: feedback_update):
        try:
            with self.connection.cursor() as cursor:
                sql = """update cog_bot_feedback set is_like = %s where id = %s """
                cursor.execute(sql, (request.is_like, request.id))
                self.connection.commit()
                return {"message": "Comment updated successfully"}

        except Exception as e:
            print(f"Exception update {str(e)}")
        finally:
            self.connection.close()
