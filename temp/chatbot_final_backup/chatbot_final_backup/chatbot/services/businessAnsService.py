import pymysql
from config.constant import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USERNAME,
    MYSQL_PASSWORD,
    MYSQL_DATABASE
)
import weaviate
from weaviate.classes.query import MetadataQuery
from datetime import datetime
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import ollama
import pymysql
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")
import traceback

from config.constant import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_PATH,
    LONG_TERM_MODEL_PATH,
    SHORT_TERM_MODEL_PATH,
    CREW_MODEL_NAME,
    CREW_OLLAMA_SERVER_URL,
    CREW_TEMPERATURE
)
from src.utils.embedder_utils import (
    SentenceTransformerEmbedder,
    LangchainSentenceTransformer
)
from model.feedback import feedback, feedback_response, feedback_update
from services.feedbackService import FeedbackService

from logger_config import setup_logger
logger = setup_logger()

def get_weaviate_client() -> weaviate.Client:
    """Connect to Weaviate instance running locally."""
    return weaviate.connect_to_custom(
        http_host="10.183.142.105",
        http_port=8081,
        http_secure=False,
        grpc_host="10.183.142.105",
        grpc_port=50051,
        grpc_secure=False
    )

def connect_to_database(host, username, password, database):
    return pymysql.connect(
        user=username,
        passwd=password,
        host=host,
        database=database,
        cursorclass=pymysql.cursors.DictCursor
    )

embedding_model = LangchainSentenceTransformer(EMBEDDING_MODEL_PATH)


def query_articles(collection, query_text: str, limit: int = 1):
    """Search articles using vector similarity."""
    # Generate query embedding
    query_vector = embedding_model.embed_query(query_text)
    # Perform vector search
    result = collection.query.near_vector(
        near_vector=query_vector,
        limit=limit,
        distance=0.6,
        return_metadata=MetadataQuery(distance=True),
    )
    print(f"\nSearch results for '{query_text}':")

    final_result = result.objects
    return final_result

def clean_sql(raw_response: str) -> str:
    """Aggressive SQL cleaning with multiple safeguards"""
    # Remove all markdown code blocks
    cleaned = re.sub(r'```.*?\n', '', raw_response, flags=re.DOTALL)
    cleaned = cleaned.replace('```', '')
    # Extract first complete SQL statement
    sql_match = re.search(
        r'(SELECT|WITH).*?(?:;|$)',
        cleaned,
        re.IGNORECASE | re.DOTALL
    )
    if sql_match:
        result = sql_match.group(0).strip()
        # Ensure proper termination
        if not result.endswith(';'):
            result += ';'
        return result
    return "ERROR: No valid SQL detected"


def sql_generation(weaviate_result, natural_language_query):
    try:
        logger.info(f"Begin sql_generation {weaviate_result}")
        ollama_client = ollama.Client(host="http://10.183.142.105:11434")
        current_date = datetime.now().strftime("%Y-%m-%d")

        for obj in weaviate_result:
            table_name = obj.properties["table_name"]
            kpi_name = obj.properties["kpi_name"]
            schema = obj.properties["schema"]

            kpi_name_column = obj.properties["kpi_name_column"]
            node_name_column = obj.properties["node_name_column"]
            kpi_value_column = obj.properties["kpi_value_column"]
            kpi_time_column = obj.properties["kpi_time_column"]

            # Format the prompt with schema and query
            prompt = f"""
                ### Task:
                You are an expert in writing SQL queries, Your task is to convert a question into a SQL query. Use the below details to generate the query accurately:

                ### Instruction
                Answer this question:{natural_language_query}

                ### Database Schema:
                {schema}

                ### Table Name (Use this table only):
                {table_name}

                ### Key Columns:
                - {kpi_name_column} (filter value: '{kpi_name}')
                - {node_name_column} (filter value)
                - {kpi_value_column} (filter value)
                - {kpi_time_column} (datetime column)

                ### Date Handling:
                1. TODAY'S DATE is {current_date}
                2. For date-only queries: DATE({kpi_time_column}) = 'YYYY-MM-DD'
                3. For exact timestamps: {kpi_time_column} = 'YYYY-MM-DD HH:MM:SS'
                4. For relative dates: DATE({kpi_time_column}) = CURDATE() - INTERVAL X DAY
                5. NEVER mix date functions with timestamp literals

                ### Query Examples:
                1. Example of query format if the user asked for the utilization at 10:00 AM on April 1rd-2024:
                   SELECT {kpi_name_column}, {node_name_column}, {kpi_value_column}, {kpi_time_column}
                   FROM {schema}.{table_name}
                   WHERE {kpi_name_column} = {kpi_name}
                     AND {kpi_time_column} = "2024-04-01 10:00:00";
                2. Example of the query format if user asked for the minimum utilization of Yesterday:
                   SELECT {kpi_name_column}, {node_name_column}, MIN({kpi_value_column}), {kpi_time_column}
                   FROM {schema}.{table_name}
                   WHERE {kpi_name_column} = {kpi_name}
                     AND {node_name_column} = :node_name
                     AND DATE({kpi_time_column}) = CURDATE() - INTERVAL 1 DAY;
                3. Example of the query format if user asked: Which node had was utilized the most for yesterday?:
                   SELECT {node_name_column}, MAX({kpi_value_column}),  DATE({kpi_time_column})
                   FROM {schema}.{table_name}
                   WHERE DATE({kpi_time_column}) = CURDATE() - INTERVAL 1 DAY;

                ### Critical Instructions
                1. Provide EXACTLY One query based on the user question and the instructions.
                2. DON'T use GROUP BY in the queries.
                3. Adhere to this rules: **Deliberately go through the question and database schema word by word** to appropriately answer the question
                4. Be careful when the user question includes specific hour, like 10AM. You need to handle this. The way to handle is by filtering on the specific hour. Example: {kpi_time_column} = "%Y-%m-%d 10:00:00"
                5. For dates only: DATE({kpi_time_column}) = 'YYYY-MM-DD'
                6. For exact timestamps: {kpi_time_column} = 'YYYY-MM-DD HH:MM:SS'
                7. For relative dates: DATE({kpi_time_column}) = CURDATE() - INTERVAL X DAY
                8. USE THIS EXACT KPI NAME: '{kpi_name}' 
                   - This value comes from the database schema
                   - IGNORE any similar names in the question

                ### Strict Formatting Rules:
                - Return ONLY raw SQL code
                - Never use markdown (no ``` or code blocks)
                - No explanations or commentary
                - No line wrapping

                ### SQL Query:
                """

            # Generate SQL
            response = ollama_client.generate(
                model='llama3.2:3b',  # Replace with your Ollama model name
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Reduce randomness
                    'max_tokens': 500,  # Limit response length
                    'stop': ['###', '---']  # Prevent commentary
                }
            )
            logger.info("---------Begin------------")
            logger.info(f"Process clean_response {response['response'].strip()}")
            clean_response = clean_sql(response['response'].strip())

            logger.info(f"Begin clean_response {clean_response}")
            logger.info("---------End------------")

            # Extract SQL from response
            return clean_response, kpi_name
    except Exception as e:
        logger.error("Error :sql_generation is: %s", str(e))


def sql_query_executer(query):
    logger.info(f"Begin sql_query_executer {query}")
    # mydb = pymysql.connect(
    #     host='10.237.90.21',
    #     port=3306,
    #     user='root',
    #     passwd='Test@123!',
    #     db='iplan_system',
    #     charset='utf8',
    # )
    # cursor = mydb1.cursor()

    mydb = pymysql.connect(
        host='10.183.130.99',
        port=3306,
        user='bigdata',
        passwd='Auto@2020!$',
        db='iplan_system',
        charset='utf8',
    )
    cursor = mydb.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    mydb.commit()
    mydb.close()

    return results



def format_results_to_natural_language(original_question: str, sql_results: tuple, kpi_name: str) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    """
    Converts raw SQL results into natural language summaries.
    Args:
        original_question: User's original question (for context)
        sql_results: Raw results from SQL execution
        query_execution_date: Optional timestamp of when query ran
    Returns:
        Human-readable summary of results
    """
    ollama_client = ollama.Client(host="http://10.183.142.105:11434")

    # Build context-aware prompt
    prompt = f"""
        ### Task:
        Transform these SQL results into a natural language answer for the original question.

        ### Date Handling:
        1. TODAY'S DATE is {current_date}

        ### Original Question:
        {original_question}

        ### Query Execution Context:
        - Current date: {current_date}
        - Results generated at: {datetime.now()}

        ### SQL Results:
        {sql_results}

        ### Formatting Rules:
        1. Use COMPLETE SENTENCES with proper punctuation
        2. Include exact values from the data
        3. Format dates as "Month Day, Year" (e.g., "April 02, 2025")
        4. For utilization values, add percentage symbol (e.g., "1.84%")
        5. For multiple results, use bullet points
        6. Never show raw numbers without context
        7. Provide in the output response ONLY the natural humann text, no automated generated text.
        8. Don't add text indicates that the answer is AI generated. Don't add text like: 'here's a natural language answer to the original question'
        9. In case KPI or KPI Name is mentioned in the question, use this {kpi_name}

        ### Example Output:
        1. The maximum utilization of {kpi_name} for yesterday was 4%.
        2. The node with maximum utilization for yesterday was "node_name" with 1.84%.
        3. The Utilization for KPI {kpi_name} and Node :NODE recorded on :DATE is 1.5%.

        ### Critical Instructions:
        1. Provide only a SINGLE sentence output. NO BULLET POINTS. NO PARAGRAPHS.
        2. Use the Example Output just as samples to understand how can the result be formatted. DON'T USE EXACTLY THE EXAMPLE AS OUTPUT RESULT.


        ### Your Response:
        """

    # Generate natural language response
    response = ollama_client.generate(
        model='llama3.2:3b',
        prompt=prompt,
        options={
            'temperature': 0.1,  # Balanced creativity/accuracy
            'num_predict': 300,
        }
    )

    return response['response'].strip()


def sqlagenttool(user_question):
    logger.info(f"sql generate for user question: {user_question}")
    try:
        client = get_weaviate_client()
        collection = client.collections.get("sql_generator")
        result_query = query_articles(collection, user_question)
        # print(f"result_query is: {result_query}")

        result_sql, kpi_name = sql_generation(result_query, user_question)
        logger.info(f"result_sql is: {result_sql}")

        result_data = sql_query_executer(result_sql)
        logger.info(f"result_data is: {result_data}")

        human_text = format_results_to_natural_language(user_question, result_data,kpi_name)
        logger.info(f"human_text is: {human_text}")
        return human_text, result_sql

    except Exception as e:
        error_msg = f"SQLAgentTool ERROR: {str(e)}"
        logger.info(error_msg)
        logger.info(f"SQLAgentTool Exception:\n{traceback.format_exc()}")
        # Also log to the system logger
        # logger.error(f"SQLAgentTool Exception:\n{traceback.format_exc()}")
        # Important: Raise or return a recognizable error string
        return f"ERROR_SQL_TOOL: {error_msg}"





class BusinessAnsService:
    def __init__(self, logging_service: logger):
        self.logging_service = logging_service
        self.connection = connect_to_database(MYSQL_HOST, MYSQL_USERNAME, MYSQL_PASSWORD,
                                              MYSQL_DATABASE)


    def submit(self, user_input: str, user_id: str, session_id: str):
        logger.info(
            f"Begin submit {user_input} user_id {user_id} , session {session_id}")

        human_text, result_sql = sqlagenttool(user_input)
        model = feedback(usr_id=user_id, usr_quest=user_input, usr_ans=human_text, session_id=session_id,
                         sql_query=result_sql, is_like=1)

        logger.info(f"Feedback model {model}")
        feedback_result = self.save_feedback(model)
        logger.info(f"feedback_result model {feedback_result}")

        return human_text, feedback_result

    def save_feedback(self, request: feedback):
        logger.info("Begin save_feedback")
        feedback_service = FeedbackService(logger)
        result = feedback_service.submit(request)
        logger.info("End save_feedback")
        return result

