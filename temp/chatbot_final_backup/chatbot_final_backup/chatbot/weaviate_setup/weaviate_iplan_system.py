import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery
from typing import List
from datetime import datetime, timezone
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import ollama
import warnings
warnings.filterwarnings("ignore")

EMBEDDING_MODEL_PATH = "/app1/python/code/weaviate_setup/embedding_model/instructor_base/"

# ======================================
# 1. Weaviate Client Configuration
# ======================================
def get_weaviate_client() -> weaviate.Client:
    """Connect to Weaviate instance running locally."""
    return weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8081,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False
    )

# ======================================
# 2. Custom Embedding Model (YOUR CODE)
# ======================================
class LangchainSentenceTransformer(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_name_or_path=model_path, use_auth_token=False)  # Load local model

    def embed_documents(self, texts):
        """Embeds a list of texts and returns list of vectors"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """Embeds a single query and returns vector"""
        return self.model.encode(text, convert_to_numpy=True).tolist()


embedding_model = LangchainSentenceTransformer(EMBEDDING_MODEL_PATH)

# ======================================
# 3. Schema Management
# ======================================
def create_schema(client: weaviate.Client) -> None:
    """Create collection with custom vector configuration."""
    client.collections.create(
        name="iplan_system",
        properties=[
            Property(name="table_name", data_type=DataType.TEXT),
            Property(name="kpi_name", data_type=DataType.TEXT),
            Property(name="kpi_name_column", data_type=DataType.TEXT),
            Property(name="node_name_column", data_type=DataType.TEXT),
            Property(name="kpi_value_column", data_type=DataType.TEXT),
            Property(name="kpi_time_column", data_type=DataType.TEXT),
            Property(name="schema", data_type=DataType.TEXT),
            Property(name="sample_queries", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        )
    )
    print("Schema created successfully")

# ======================================
# 4. Data Operations
# ======================================
def insert_articles(collection, client: weaviate.Client, articles: List[dict]) -> None:
    """Insert articles with custom embeddings."""
    # Configure batch settings
    client.batch.batch_size = 100  # Max objects per batch
    client.batch.dynamic = True     # Auto-retry failed objects

    question_objs = list()
    for i, d in enumerate(articles):
        text_to_embed = f"{d['table_name']} {d['kpi_name']} {d['description']}"
        vector = embedding_model.embed_query(text_to_embed)
        question_objs.append(wvc.data.DataObject(
            properties={
                "table_name": d["table_name"],
                "kpi_name": d["kpi_name"],
                "kpi_name_column": d["kpi_name_column"],
                "node_name_column": d["node_name_column"],
                "kpi_value_column": d["kpi_value_column"],
                "kpi_time_column": d["kpi_time_column"],
                "schema": d["schema"],
                "sample_queries": d["sample_queries"],
                "description": d["description"],
            },
            vector=vector
        ))

    collection.data.insert_many(question_objs)
    print("Data Inserted successfully")

def query_articles(collection, query_text: str, limit: int = 1) -> None:
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
    for obj in result.objects:
        print(f"\nTHE FULL PROPERTIES: {obj.properties}")
        print(f"TABLE NAME: {obj.properties['table_name']}")
        print(f"KPI NAME: {obj.properties['kpi_name']}")
        print(f"Distance: {obj.metadata.distance}")


    return result.objects


# ======================================
# 5. TEXT TO SQL
# ======================================
def sql_generation(weaviate_result, user_question):
    ollama_client = ollama.Client(host="http://10.183.142.105:11434")
    print(f"THE QUESTION IS: {user_question}")

    current_date = datetime.now().strftime("%Y-%m-%d")

    for obj in weaviate_result:
        print(f"\nTHE FULL PROPERTIES: {obj}")
        table_name = obj["table_name"]
        kpi_name = obj["kpi_name"]
        schema = obj["schema"]

        kpi_name_column = obj["kpi_name_column"]
        node_name_column = obj["node_name_column"]
        kpi_value_column = obj["kpi_value_column"]
        kpi_time_column = obj["kpi_time_column"]

        # Format the prompt with schema and query
        prompt = f"""
        ### Task:
        You are an expert in writing SQL queries, Generate a single SQL query to answer this question: {user_question}. Use the below details to generate the query accurately

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
        1. TODAY'S DATE is '{current_date}'
        2. Date ranges should use SQL date functions, NOT hardcoded values
        3. To query today's value, filter using: DATE({kpi_time_column}) = CURDATE()
        4. To query yesterday's value, filter using: DATE({kpi_time_column}) = CURDATE() - INTERVAL 1 DAY
        5. To query a specific date values, filter using: DATE({kpi_time_column}) = :specific_date
        6. To query a specific month values, filter using BETWEEN: 
            DATE('{kpi_time_column}') BETWEEN :start_of_the_month AND :end_of_the_month

        ### Critical Instructions:
        1. Example of query format if the user asked for the utilization at 10:00 AM on April 3rd-2024:
           SELECT {kpi_value_column}
           FROM {table_name}
           WHERE {kpi_name_column} = '{kpi_name}'
             AND {node_name_column} = :node_name
             AND {kpi_time_column} = "2024-04-03 10:00:00";
        2. Example of the query format if user asked for the min utilization of Yesterday:
           SELECT MIN({kpi_value_column})
           FROM {table_name}
           WHERE {kpi_name_column} = '{kpi_name}'
             AND {node_name_column} = :node_name
             AND DATE({kpi_time_column}) = CURDATE() - INTERVAL 1 DAY;
        3. Provide EXACTLY One query based on the user question and the instructions

        ### SQL Query:
        """

        # Generate SQL
        response = ollama_client.generate(
            model='llama3.2:3b',  # Replace with your Ollama model name
            prompt=prompt,
            options={
                'temperature': 0.1,  # Reduce randomness
                'max_tokens': 500  # Limit response length
            }
        )

        # Extract SQL from response
        return response['response'].strip()


# ======================================
# 6. Example Usage
# ======================================
client = get_weaviate_client()
try:
    # RFC3339 Date Format
    published_date = datetime.now(timezone.utc).isoformat()

    # Clean existing schema (optional)
    for collection_name in client.collections.list_all():
        client.collections.delete(collection_name)

    # Create new schema
    create_schema(client)
    # Sample data
    articles = [
        # ---------- CSCF KPIs ---------- #
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoWiFi Calls",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": ("""
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL,
                );
            """),
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi Calls'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
                 
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE {kpi_name_column} = 'VoWiFi Calls'
                 AND {node_name_column} = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": (
                "This table has all data related to VoWiFi Calls KPI. It is storing the hourly, daily, weekly, " 
                "monthly, and yearly Capacity Planning Utilization value."
            ),
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoLTE Erlang",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": ("""
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL,
                );
            """),
            "sample_queries": ("""
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi Calls'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
        
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE {kpi_name_column} = 'VoWiFi Calls'
                 AND {node_name_column} = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """),
            "description": (
                "This table has all data related to VoLTE Erlang KPI. It is storing the hourly, daily, weekly, "
                "monthly, and yearly Capacity Planning Utilization value."
            ),
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "CSCF Cx Congestions on HSS Side",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF Cx Congestions on HSS Side'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF Cx Congestions on HSS Side'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to CSCF Cx Congestions on HSS Side KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "CSCF SIP UDP Congestions",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF SIP UDP Congestions'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF SIP UDP Congestions'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to CSCF SIP UDP Congestions KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "CSCF Throttled HSS Diameter Requests",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF Throttled HSS Diameter Requests'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF Throttled HSS Diameter Requests'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to CSCF Throttled HSS Diameter Requests KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "CSCF SIP TCP Congestions",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF SIP TCP Congestions'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF SIP TCP Congestions'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to CSCF SIP TCP Congestions KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "vCSCF Memory Usage",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'vCSCF Memory Usage'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'vCSCF Memory Usage'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to vCSCF Memory Usage KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "vCSCF vCPU Usage",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'vCSCF vCPU Usage'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'vCSCF vCPU Usage'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to vCSCF vCPU Usage KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "CSCF Mean Holding Time",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF Mean Holding Time'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF Mean Holding Time'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to CSCF Mean Holding Time KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoWiFi Erlang",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi Erlang'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi Erlang'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to VoWiFi Erlang KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "CSCF Load",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF Load'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF Load'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to CSCF Load KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoWiFi Users",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi Users'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi Users'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to VoWiFi Users KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoWiFi BHCA",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi BHCA'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi BHCA'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to VoWiFi BHCA KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoWiFi Calls Time",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi Calls Time'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoWiFi Calls Time'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to VoWiFi Calls Time KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoLTE Calls",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoLTE Calls'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoLTE Calls'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to VoLTE Calls KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoLTE Calls Time",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoLTE Calls Time'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoLTE Calls Time'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to VoLTE Calls Time KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "CSCF BHCA",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF BHCA'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'CSCF BHCA'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to CSCF BHCA KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "IMS Total BHCA",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'IMS Total BHCA'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'IMS Total BHCA'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to IMS Total BHCA KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "VoLTE BHCA",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoLTE BHCA'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'VoLTE BHCA'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to VoLTE BHCA KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_cscf_util_planning",
            "kpi_name": "Total IMS Calls",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_cscf_util_planning` (
                  `kpi_name` varchar(255) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'Total IMS Calls'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_cscf_util_planning
               WHERE kpi_name = 'Total IMS Calls'
                 AND node_name = 'ACEIM1_vcscf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to Total IMS Calls KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },

        # ---------- SBG KPIs ---------- #
        {
            "table_name": "cog_vims_sbg_util_planning",
            "kpi_name": "SBG SIP Ingress Overload Rejection",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_sbg_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL,
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_sbg_util_planning
               WHERE kpi_name = 'SBG SIP Ingress Overload Rejection'
                 AND node_name = 'ACEIM1_vsbg03'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_sbg_util_planning
               WHERE kpi_name = 'SBG SIP Ingress Overload Rejection'
                 AND node_name = 'ACEIM1_vsbg03'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": (
                "This table has all data related to SBG SIP Ingress Overload Rejection KPI. It is storing the hourly, "
                "daily, weekly, monthly, and yearly Capacity Planning Utilization value."
            ),
        },
        {
            "table_name": "cog_vims_sbg_util_planning",
            "kpi_name": "SBG SIP Core Overload Rejection",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
            CREATE TABLE `cog_vims_sbg_util_planning` (
              `kpi_name` varchar(45) DEFAULT NULL,
              `node_name` varchar(45) DEFAULT NULL,
              `kpi_value` double(20,2) DEFAULT NULL,
              `kpi_time` datetime DEFAULT NULL,
            );
        """,
            "sample_queries": """
           SELECT kpi_name
           FROM cog_vims_sbg_util_planning
           WHERE kpi_name = 'SBG SIP Core Overload Rejection'
             AND node_name = 'ACEIM1_vsbg03'
             AND kpi_time = "2024-04-03 10:00:00";
           SELECT MIN(kpi_value)
           FROM cog_vims_sbg_util_planning
           WHERE kpi_name = 'SBG SIP Core Overload Rejection'
             AND node_name = 'ACEIM1_vsbg03'
             AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
        """,
            "description": (
                "This table has all data related to SBG SIP Core Overload Rejection KPI. It is storing the hourly, "
                "daily, weekly, monthly, and yearly Capacity Planning Utilization value."
            ),
        },
        {
            "table_name": "cog_vims_sbg_util_planning",
            "kpi_name": "SBG SIP BGF Overload Rejection",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
            CREATE TABLE `cog_vims_sbg_util_planning` (
              `kpi_name` varchar(45) DEFAULT NULL,
              `node_name` varchar(45) DEFAULT NULL,
              `kpi_value` double(20,2) DEFAULT NULL,
              `kpi_time` datetime DEFAULT NULL,
            );
        """,
            "sample_queries": """
           SELECT kpi_name
           FROM cog_vims_sbg_util_planning
           WHERE kpi_name = 'SBG SIP BGF Overload Rejection'
             AND node_name = 'ACEIM1_vsbg03'
             AND kpi_time = "2024-04-03 10:00:00";
           SELECT MIN(kpi_value)
           FROM cog_vims_sbg_util_planning
           WHERE kpi_name = 'SBG SIP BGF Overload Rejection'
             AND node_name = 'ACEIM1_vsbg03'
             AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
        """,
            "description": (
                "This table has all data related to SBG SIP BGF Overload Rejection KPI. It is storing the hourly, "
                "daily, weekly, monthly, and yearly Capacity Planning Utilization value."
            ),
        },

        # ---------- DNS KPIs ---------- #
        {
            "table_name": "cog_vims_dns_util_planning",
            "kpi_name": "DNS Query Failure Rate",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_dns_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL,
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_dns_util_planning
               WHERE kpi_name = 'DNS Query Failure Rate'
                 AND node_name = 'ACEIM1_vdns01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_dns_util_planning
               WHERE kpi_name = 'DNS Query Failure Rate'
                 AND node_name = 'ACEIM1_vdns01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": (
                "This table has all data related to DNS Query Failure Rate KPI. It is storing the hourly, "
                "daily, weekly, monthly, and yearly Capacity Planning Utilization value."
            ),
        },
        {
            "table_name": "cog_vims_dns_util_planning",
            "kpi_name": "DNS Query Failure",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_dns_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL,
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_dns_util_planning
               WHERE kpi_name = 'DNS Query Failure'
                 AND node_name = 'ACEIM1_vdns01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_dns_util_planning
               WHERE kpi_name = 'DNS Query Failure'
                 AND node_name = 'ACEIM1_vdns01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": (
                "This table has all data related to DNS Query Failure KPI. It is storing the hourly, "
                "daily, weekly, monthly, and yearly Capacity Planning Utilization value."
            ),
        },

        # ---------- MTAS KPIs ---------- #
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "vMTAS vCPU Usage",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL,
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'vMTAS vCPU Usage'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'vMTAS vCPU Usage'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": (
                "This table has all data related to vMTAS vCPU Usage KPI. It is storing the hourly, "
                "daily, weekly, monthly, and yearly Capacity Planning Utilization value."
            ),
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "MTAS SIP PSI Overload Rejection",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS SIP PSI Overload Rejection'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS SIP PSI Overload Rejection'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to MTAS SIP PSI Overload Rejection KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "MTAS MRFC Overload Rejection",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS MRFC Overload Rejection'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS MRFC Overload Rejection'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to MTAS MRFC Overload Rejection KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "MTAS Diameter Overload Discard",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS Diameter Overload Discard'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS Diameter Overload Discard'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to MTAS Diameter Overload Discard KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "MTAS SIP Overload Rejection",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS SIP Overload Rejection'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS SIP Overload Rejection'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to MTAS SIP Overload Rejection KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "Active Conference Session",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Active Conference Session'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Active Conference Session'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to Active Conference Session KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "vMTAS Memory Usage",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'vMTAS Memory Usage'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'vMTAS Memory Usage'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to vMTAS Memory Usage KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "Active Call Sessions",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Active Call Sessions'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Active Call Sessions'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to Active Call Sessions KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "MTAS DNS Overload Rejection",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS DNS Overload Rejection'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'MTAS DNS Overload Rejection'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to MTAS DNS Overload Rejection KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "Total IMS Calls Time",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Total IMS Calls Time'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Total IMS Calls Time'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to Total IMS Calls Time KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "Registered Users",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Registered Users'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Registered Users'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to Registered Users KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "Active Users",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Active Users'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'Active Users'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to Active Users KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },
        {
            "table_name": "cog_vims_mtas_util_planning",
            "kpi_name": "vIMS Erlang",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_mtas_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'vIMS Erlang'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_mtas_util_planning
               WHERE kpi_name = 'vIMS Erlang'
                 AND node_name = 'ACEIM1_vmtas01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": "This table contains all data related to vIMS Erlang KPI. It stores the hourly, daily, weekly, monthly, and yearly Capacity Planning Utilization value."
        },

        # ---------- vBGF KPIs ---------- #
        {
            "table_name": "cog_vims_vbgf_util_planning",
            "kpi_name": "Packet Loss Ratio Core",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_vbgf_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL,
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_vbgf_util_planning
               WHERE kpi_name = 'Packet Loss Ratio Core'
                 AND node_name = 'ACEIM1_vbgf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_vbgf_util_planning
               WHERE kpi_name = 'Packet Loss Ratio Core'
                 AND node_name = 'ACEIM1_vbgf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": (
                "This table has all data related to Packet Loss Ratio Core KPI. It is storing the hourly, "
                "daily, weekly, monthly, and yearly Capacity Planning Utilization value."
            ),
        },
        {
            "table_name": "cog_vims_vbgf_util_planning",
            "kpi_name": "Packet Loss Ratio Access",
            "kpi_name_column": "kpi_name",
            "node_name_column": "node_name",
            "kpi_value_column": "kpi_value",
            "kpi_time_column": "kpi_time",
            "schema": """
                CREATE TABLE `cog_vims_vbgf_util_planning` (
                  `kpi_name` varchar(45) DEFAULT NULL,
                  `node_name` varchar(45) DEFAULT NULL,
                  `kpi_value` double(20,2) DEFAULT NULL,
                  `kpi_time` datetime DEFAULT NULL,
                );
            """,
            "sample_queries": """
               SELECT kpi_name
               FROM cog_vims_vbgf_util_planning
               WHERE kpi_name = 'Packet Loss Ratio Access'
                 AND node_name = 'ACEIM1_vbgf01'
                 AND kpi_time = "2024-04-03 10:00:00";
               SELECT MIN(kpi_value)
               FROM cog_vims_vbgf_util_planning
               WHERE kpi_name = 'Packet Loss Ratio Access'
                 AND node_name = 'ACEIM1_vbgf01'
                 AND DATE(kpi_time) = CURDATE() - INTERVAL 1 DAY;
            """,
            "description": (
                "This table has all data related to Packet Loss Ratio Access KPI. It is storing the hourly, "
                "daily, weekly, monthly, and yearly Capacity Planning Utilization value."
            ),
        },
    ]

    collection = client.collections.get("iplan_system")

    # Insert data
    insert_articles(collection, client, articles)
    # Perform search
    # result = query_articles(collection, "Provide me the table name that i can get Mtas DNS Overload Rejection Utilization values from it")
except KeyboardInterrupt:
    print("\nOperation cancelled")
finally:
    client.close()
    print("Client connection closed")
