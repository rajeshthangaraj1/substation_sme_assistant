import sys
import os
import json
import traceback
from config.constant import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_PATH,
    CREW_MODEL_NAME,
    CREW_OLLAMA_SERVER_URL,
    CREW_TEMPERATURE
)
from logger_config import setup_logger
from file_handler import FileHandler
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tools import tool
from langchain_community.vectorstores import FAISS
from llm_summarizer import summarize_document_tools
from src.utils.embedder_utils import (
    LangchainSentenceTransformer
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from weaviate_setup.complete_flow import sqlagenttool
from model.feedback import feedback, feedback_response, feedback_update
from services.feedbackService import FeedbackService
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# Environment variables
os.environ["STREAMLIT_WATCHER_ENHANCED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_LOCAL_MODE"] = "True"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = setup_logger()
# torch class issue
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

custom_llm = LLM(
    model=CREW_MODEL_NAME,
    base_url=CREW_OLLAMA_SERVER_URL,
    streaming=False,
    max_tokens=256,
    handle_parsing_errors=True,
    temperature=CREW_TEMPERATURE,
)

class SummarizeService:
    g_collection = None  # class-level global variable
    g_uploaded_file_path = None
    g_last_response = None
    g_sql_query = None

    def __init__(self, logging_service: logger):
        self.logging_service = logging_service
        # self.file_handler_method = FileHandler(VECTOR_DB_PATH)
        # self.collection: str = ""

    def buildContext(self, user_input: str, user_id: str, session_id: str, session_state):

        logger.info(
            f"Begin buildContext {user_input} session state {session_state} user_id {user_id} , session {session_id}")
        g_collection = session_state.get("collection")
        logger.info(f"g_collection {g_collection}")
        g_uploaded_file_path = session_state.get("uploaded_file_path")
        logger.info(f"g_uploaded_file_path {g_uploaded_file_path}")
        context = {
            "query": user_input.lower(),
            "document_uploaded": "yes" if g_collection else "no",
            "document_status": "Uploaded" if g_collection else "Not uploaded",
            "explicit_summary_request": any(
                keyword in user_input.lower() for keyword in ["summarize", "summary", "overview"])
        }
        logger.info(f"context {context}")

        # Enhanced Manager Agent with Clear Decision Tree
        manager_agent = Agent(
            role="Senior Support Manager",
            goal="Route user queries explicitly and efficiently to exactly ONE specialized agent at a time.",
            backstory="Expert at selecting the right agent without ambiguity. Retries once if first choice fails.",
            verbose=True,
            llm=custom_llm,
            max_iter=2,
            description=f"""
                Explicit Routing Rules:
        
                1. IF summarization keywords detected ({context['explicit_summary_request']}) 
                   AND Document Uploaded? ({context['document_uploaded']}):
                   → Document Summarization Expert
        
                2. ELSE IF Document Uploaded? ({context['document_uploaded']}) 
                   AND user explicitly mentions document ("according to the document"):
                   → Document Analysis Specialist
                   - IF this returns ERROR_NO_RESULTS, immediately fallback to:
                     → General Chat Agent
        
                3. ELSE (ALL other cases):
                   → General Chat Agent
        
                STOP immediately after successful routing or after exactly one fallback attempt.
                """,
        )
        logger.info(f"manager_agent {manager_agent}")

        # Summarization Tool
        @tool("summarize_document")
        def summarize_document() -> str:
            """Summarize the currently uploaded document. NO INPUT NEEDED."""

            try:
                logger.info(f"summarize_document {g_collection}")
                if not g_collection:
                    return "Final Answer: ERROR_NO_DOCUMENT: Please upload a document first"

                document_path = g_uploaded_file_path
                logger.info(f"document_path {document_path}")
                if not document_path or not os.path.exists(document_path):
                    return "Final Answer: ERROR_INVALID_PATH: No valid document found"

                result = summarize_document_tools(document_path)
                final_response = f"Final Answer: {result}"
                logger.info(f"Final Answer from tool:   {final_response}")

                # st.session_state["last_response"] = final_response
                SummarizeService.g_last_response = final_response
                logger.info(f"final_response {final_response}")
                return final_response
            except Exception as e:
                logger.error(f"Summarize Document Error:\n{traceback.format_exc()}")
                return f"Final Answer: ERROR_SUMMARIZE_TOOL: {str(e)}"

        @tool("sql_retriever")
        def sql_retriever(query: dict) -> str:
            """Handles SQL-style capacity planning questions and returns the final output. Returns error codes for agent routing:
            - ERROR_SQL_TOOL: When Sql Tool Error
            """
            try:
                logger.info(f"Rahul sql_retriever {query}")
                if isinstance(query, dict):
                    query = query.get("description", "")
                sqlanswer, sql_query = sqlagenttool(query)
                g_sql_query = sql_query
                logger.info(f"Rahul sqlanswer {sqlanswer}")
                logger.info(f"Rahul sql_query {sql_query}")
                final_response = f"Final Answer: {sqlanswer}"
                logger.info(f"Final Answer from SQL tool: {final_response}")
                return final_response
            except Exception as e:
                logger.error(f"Sql Retriever Error:\n{traceback.format_exc()}")
                return f"ERROR_SQL_TOOL: {str(e)}"

        # Document Retriever Tool
        @tool("document_retriever")
        def document_retriever(query: dict) -> str:
            """Retrieves relevant context from the uploaded document based on a given input. Returns error codes for agent routing:
            - ERROR_NO_DOCUMENT: When no document uploaded
            - ERROR_INVALID_QUERY: When query not about document
            - ERROR_NO_RESULTS: When no matching content"""

            try:
                logger.info(f" document_retriever {query}")
                if isinstance(query, dict):
                    query = query.get("description", "")  # Extract the actual query string

                collection_name = g_collection
                logger.info(f" document_retriever collection_name {collection_name}")
                if not collection_name:
                    return "ERROR_NO_DOCUMENT: Please upload a document first"

                embeddings = LangchainSentenceTransformer(EMBEDDING_MODEL_PATH)
                index_path = os.path.join(VECTOR_DB_PATH, collection_name, "index.faiss")

                if os.path.exists(index_path):
                    vector_store = FAISS.load_local(
                        os.path.join(VECTOR_DB_PATH, collection_name),
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    response_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=5)
                    filtered_responses = [doc.page_content for doc, score in response_with_scores]
                    if filtered_responses:
                        combined_responses = "\n\n".join(filtered_responses)

                        final_response = f"Final Answer: {combined_responses}"
                        SummarizeService.g_last_response = combined_responses
                        return final_response
                    else:
                        return "ERROR_NO_RESULTS: No relevant content found in document"
                else:
                    logger.info("PATH NOT FOUND")
                    return "PATH NOT FOUND"
            except Exception as e:
                logger.error(f"Document Retriever Error:\n{traceback.format_exc()}")
                return f"ERROR_DOCUMENT_TOOL: {str(e)}"

        # ✅ Agent Definitions with Enhanced Descriptions
        summarizer_agent = Agent(
            role="Document Summarization Expert",
            goal="Directly return summarized content from tool output",
            backstory="Specializes in executing summarization tools",
            tools=[summarize_document],
            verbose=True,
            llm=custom_llm,
            max_iter=1,
            description="""
                STRICT USAGE RULES:
                  - Only activate if BOTH:
                    1. A document is uploaded.
                    2. The query contains explicit summary request keywords: 'summarize', 'summary', or 'overview'.
                  - ALWAYS call the summarize_document tool.
                  - IMMEDIATELY RETURN the tool's response as your final answer.
                  - DO NOT modify, reword, expand, or add anything to the tool's output.
                  - DO NOT add any comments or extra information.
                  - STOP after the first successful tool use.
                """

        )

        rag_chat_agent = Agent(
            role="Document Analysis Specialist",
            backstory="Answers document-specific queries via retrieval augmented generation (RAG).",
            goal="Provide accurate answers from uploaded files",
            tools=[document_retriever],
            verbose=True,
            llm=custom_llm,
            max_iter=1,
            description="""
            STRICT RULES:
            1. ONLY activate if document uploaded AND query explicitly references the document.
            2. Handle queries like "According to the document..." or "From the document..."
            3. ALWAYS return response directly from the document_retriever tool without additional interpretation.
            4. IF no relevant content found, return 'ERROR_NO_RESULTS'.
            STOP after one tool use.
            """,
        )

        general_chat_agent = Agent(
            role="General Conversationalist",
            goal="Briefly respond to general or non-document queries in ONE or TWO concise lines MAXIMUM.",
            backstory="Skilled in natural conversations about various topics",
            verbose=True,
            llm=custom_llm,
            max_iter=2,
            description="""
                DEFAULT AGENT FOR ALL NON-DOCUMENT QUERIES.
                STRICT RESPONSE RULES:
                  1. Respond strictly in ONE or TWO short sentences only.
                  2. NEVER provide long explanations or details.
                  3. If unsure, reply briefly indicating uncertainty in one line.
                STOP immediately after one response.
                """,
        )

        # ✅ Corrected Autonomous Task Configuration
        def create_autonomous_task(user_query: str):
            return Task(
                description=f"Route immediately to ONE agent for query: {user_query}",
                expected_output="Immediate final response prefixed with 'Final Answer:'",
                context=[{
                    "query": user_query,
                    "description": user_query,
                    "expected_output": "Relevant response matching query intent"
                }],
                config={
                    "allow_delegation": True,
                    "stop_on_success": True,
                }
            )

        chat_task = create_autonomous_task(user_input)
        crew = Crew(
            agents=[summarizer_agent, rag_chat_agent, general_chat_agent],
            tasks=[chat_task],
            process=Process.hierarchical,
            verbose=True,
            max_iter=2,
            manager_agent=manager_agent,
        )

        try:
            crew_input = {
                "query": user_input,
                "context": json.dumps(context)
            }
            logger.info(f"crew_input {crew_input}")

            response_content = crew.kickoff(inputs=crew_input)
            # print(f"After crew hit the agent")
            logger.info(f"After crew hit the agent {context}")

            #Directly return the tool output if it's already in session state
            if SummarizeService.g_last_response:
                logger.info(f"SummarizeService.g_last_response check {SummarizeService.g_last_response}")
                # response_content = st.session_state["last_response"]
                response_content = SummarizeService.g_last_response
                # st.session_state["last_response"] = None  # Reset state after use
                SummarizeService.g_last_response = None

        except Exception as e:

            traceback_str = traceback.format_exc()
            logger.error(f"Full traceback:\n{traceback_str}")
            logger.error(f"System error: {str(e)}")
            response_content = "Apologies, I encountered an error. Please retry that again..."

        logger.info(f"final output {response_content}")
        # Handle response parsing
        if isinstance(response_content, list):
            final_response = "\n".join(response_content)
        elif hasattr(response_content, 'output'):
            final_response = response_content.output
        else:
            final_response = str(response_content)

        if final_response:
            final_response = final_response.removeprefix("Final Answer:").strip()

        # Fallback to general chat on errors
        if any(err in final_response for err in ["ERROR", "Error"]):
            logger.error(f"final_response came into error, pls trigger general chat")
            final_response = general_chat_agent.execute({"task": user_input})

        model = feedback(usr_id=user_id, usr_quest=user_input, usr_ans=final_response, session_id=session_id,
                         sql_query="" , is_like=1,collection_name=g_collection,temp_file_name=g_uploaded_file_path)
        logger.info(f"Feedback model {model}")
        feedback_id = self.save_feedback(model)
        logger.info(f"feedback_id model {feedback_id}")

        return final_response, feedback_id




    def save_feedback(self, request: feedback):
        logger.info("Begin save_feedback")
        feedback_service = FeedbackService(logger)
        result = feedback_service.submit(request)
        logger.info("End save_feedback")
        return result
