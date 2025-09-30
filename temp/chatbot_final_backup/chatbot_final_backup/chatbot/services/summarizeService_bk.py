import sys
import streamlit as st
import tempfile
import os
import json
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
from logger_config import setup_logger
from file_handler import FileHandler
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tools import tool
from langchain_community.vectorstores import FAISS
from llm_summarizer import summarize_document_tools
# from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
# from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
# from crewai.memory.storage.rag_storage import RAGStorage
from src.utils.embedder_utils import (
    SentenceTransformerEmbedder,
    LangchainSentenceTransformer
)

# Add the parent folder (/app1/python/code) to the Python path
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
    handle_parsing_errors=True,
    max_iter=2,
    temperature=CREW_TEMPERATURE,
)


# Summarization Tool
@tool("summarize_document")
def summarize_document() -> str:
    """Summarize the currently uploaded document. NO INPUT NEEDED."""

    try:
        logger.info(f"summarize_document {SummarizeService.g_collection}")
        if not SummarizeService.g_collection:
            return "Final Answer: ERROR_NO_DOCUMENT: Please upload a document first"

        document_path = SummarizeService.g_uploaded_file_path
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
        SummarizeService.g_sql_query = sql_query
        logger.info(f"Rahul sqlanswer {sqlanswer}")
        logger.info(f"Rahul sql_query {sql_query}")
        final_response = f"Final Answer: {sqlanswer}"
        logger.info(f"Final Answer from SQL tool: {final_response}")
        return final_response
    except Exception as e:
        logger.error(f"Sql Retriever Error:\n{traceback.format_exc()}")
        return f"ERROR_SQL_TOOL: {str(e)}"


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
        g_uploaded_file_path = session_state.get("uploaded_file_path")
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
            goal=f"Accurately route user queries to the appropriate specialist.User Query: {user_input}",
            backstory=f"""Highly experienced in triaging user requests and selecting the most suitable agent.
                            - The goal is to SELECT ONLY ONE agent based on query type.
                            - IF the query mentions summarization keywords, SELECT the Document Summarization Expert.
                            - IF the query references document content, SELECT the Document Analysis Specialist.
                            - IF the query is general or does not mention document-specific content, SELECT the General Chat Agent.
                            - NO FOLLOW-UP questions allowed.
                            - ONCE AN AGENT IS SELECTED, DO NOT RETRY or delegate to another agent.
                            """,
            verbose=True,
            llm=custom_llm,
            max_iter=2,
            description=f"""You are the Senior Support Manager. Your ONLY responsibility is to accurately route user queries 
                        to the correct specialist agent based strictly on the conditions below:
                            User Query: {user_input}
                            Explicit Routing Examples:
                            - Query: "Summarize the uploaded document about renewable energy."
                              - Selected Agent: Document Summarization Expert
                            - Query: "According to the document, what are the benefits of solar power?"
                              - Selected Agent: Document Analysis Specialist
                            - Query: "What's the capital of France?"
                              - Selected Agent: General Chat Agent

                            Routing Conditions:
                            1. FIRST CHECK - Document Uploaded? {context['document_uploaded']}
                               - If NO: 
                                   a. ALWAYS select General Chat Agent.
                               - If YES:
                                   a. Is Summary Requested? {context['explicit_summary_request']} â†’ Route to Document Summarization Expert
                                   b. Else If Question References Document? {"document" in context['query']} â†’ Route to Document Analysis Specialist
                                   c. Else If Route to General Chat Agent
                            2. FINAL DEFAULT: General Chat Agent
                            3. RULES:
                               - SELECT ONLY ONE agent based on above logic.
                               - NO delegation or retry allowed.
                               - STOP once agent is selected.
                            <note>
                                  When delegating tasks, ensure you clearly define these fields:
                                  - task: The task description.
                                  - context: Relevant context.
                                  - coworker: Role/name of the selected agent.

                                  Example:
                                  {{
                                    "task": "Summarize the uploaded document on renewable energy.",
                                    "context": "User explicitly asked for a summary.",
                                    "coworker": "Document Summarization Expert"
                                  }}
                            </note>
                        """,
        )
        logger.info(f"manager_agent {manager_agent}")

    # Summarization Tool
    @tool("summarize_document")
    def summarize_document(self) -> str:
        """Summarize the currently uploaded document. NO INPUT NEEDED."""

        try:
            logger.info(f"summarize_document {SummarizeService.g_collection}")
            if not SummarizeService.g_collection:
                return "Final Answer: ERROR_NO_DOCUMENT: Please upload a document first"

            document_path = SummarizeService.g_uploaded_file_path
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

        # âœ… Corrected Autonomous Task Configuration
        def create_autonomous_task(user_query: str):
            has_document = SummarizeService.g_collection is not None
            # is_sql_query = any(
            #     keyword in user_query.lower()
            #     for keyword in ["utilization", "capacity", "vowifi", "report", "kpi", "node", "network"]
            # )
            is_summary_request = any(
                keyword in user_query.lower()
                for keyword in ["summarize", "summary", "overview"]
            )
            return Task(
                description=f"""Analyze and process the user query:
                            User Query: {user_query}
                            Document Status: {'Uploaded' if has_document else 'Not Uploaded'}
                            Summary Request Detected: {is_summary_request}

                            Available agent specializations:
                            1. Document Summarization Expert - .docx/.pdf summaries
                            2. Document Analysis Specialist - Document content questions
                            3. Conversational Assistant - General chat

                            Routing Rules:

                            1. if summary keywords and document exists â†’ Summarizer
                            2. Else if document exists and query references content â†’ Document Analyst
                            3. All other cases â†’ General Chat

                            """,
                expected_output="Immediate, final response prefixed with 'Final Answer:'",
                context=[{
                    "has_document": has_document,
                    "query": user_query,
                    "description": user_query,
                    "explicit_summary_request": is_summary_request,
                    "expected_output": "Relevant response matching query intent"
                }],
                config={
                    "allow_delegation": True,
                    "stop_on_success": True,
                }
            )

        chat_task = create_autonomous_task(user_input)
        crew = Crew(
            agents=[self.summarizer_agent, self.rag_chat_agent, self.general_chat_agent],
            tasks=[chat_task],
            process=Process.hierarchical,
            verbose=True,
            max_iter=3,
            manager_agent=manager_agent,
        )

        try:
            crew_input = {
                "query": user_input,
                "context": json.dumps(context)
            }
            logger.info(f"crew_input {crew_input}")

            response_content = crew.kickoff(inputs=crew_input)
            print(f"After crew hit the agent")
            logger.info(f"After crew hit the agent {context}")

            # âœ… Directly return the tool output if it's already in session state
            if SummarizeService.g_last_response:
                logger.info(f"SummarizeService.g_last_response check {SummarizeService.g_last_response}")
                # response_content = st.session_state["last_response"]
                response_content = SummarizeService.g_last_response
                # st.session_state["last_response"] = None  # Reset state after use
                SummarizeService.g_last_response = None

        except Exception as e:
            response_content = "Apologies, I encountered an error. Let me try that again..."
            traceback_str = traceback.format_exc()
            logger.error(f"Full traceback:\n{traceback_str}")
            response_content = f"System error: {str(e)}"

        logger.info(f"final output {response_content}")
        # Handle response parsing
        if isinstance(response_content, list):
            final_response = "\n".join(response_content)
        elif hasattr(response_content, 'output'):
            final_response = response_content.output
        else:
            final_response = str(response_content)

        # Fallback to general chat on errors
        if any(err in final_response for err in ["ERROR", "Error"]):
            final_response = self.general_chat_agent.execute({"task": user_input})

        # st.session_state["messages"].append({"role": "assistant", "content": response_content})
        # with st.chat_message("assistant"):
        #     st.markdown(response_content)
        # return "".join({"role": "assistant", "content": response_content})
        model = feedback(usr_id=user_id, usr_quest=user_input, usr_ans=final_response, session_id=session_id,
                         sql_query="", is_like=1)
        logger.info(f"Feedback model {model}")
        feedback_id = self.save_feedback(model)
        logger.info(f"feedback_id model {feedback_id}")

        return final_response, feedback_id

    # âœ… Agent Definitions with Enhanced Descriptions
    sql_agent = Agent(
        role="SQL Data Specialist",
        goal="Answer queries related to capacity planning and utilization using SQL",
        backstory="Expert in generating and analyzing SQL data insights, especially for capacity and network utilization metrics.",
        tools=[sql_retriever],
        verbose=True,
        llm=custom_llm,
        max_iter=2,
        description="""
        ACTIVATE ONLY FOR SQL OR DATA-RELATED QUESTIONS, e.g. capacity/utilization reports.
        Activation keywords: 'utilization', 'capacity', 'VoWiFi', 'report', 'yesterday', 'specific date'
        STRICT RULES:
        1. Always use the sql_retriever tool.
        2. Do NOT generate SQL manually â€“ let the tool handle the logic.
        3. Stop after tool returns Final Answer.
        4. NEVER modify or process tool output
        5. Return tool response directly
        6. Never delegate or reroute to other agents.
        """,
    )

    # âœ… Agent Definitions with Enhanced Descriptions
    summarizer_agent = Agent(
        role="Document Summarization Expert",
        goal="Directly return summarized content from tool output",
        backstory="Specializes in executing summarization tools",
        tools=[summarize_document],
        verbose=True,
        llm=custom_llm,
        max_iter=2,
        description="""STRICTLY FOR DOCUMENT SUMMARIES ONLY WHEN EXPLICITLY REQUESTED.
        SYSTEM INSTRUCTIONS:
        - When you receive a task to summarize a document, use the summarize_document tool.
        - The output from summarize_document is already formatted as "Final Answer: [summary]", so return it verbatim as your final answer.
        - Do not call any other tools or perform any additional processing.
        - STOP after returning the tool's output.
        Requirements:
        1. User MUST use words like 'summarize', 'summary', or 'overview'
        2. Document MUST be uploaded first
        3. Never handle questions or analysis
        STRICT USAGE RULES:
        - ONLY activates when BOTH conditions are met:
        1. Document uploaded
        2. Query contains explicit summary request keywords
        3. ALWAYS use summarize_document tool
        4. NEVER modify or process tool output
        5. Return tool response directly
        6. STOP after first successful tool use
        """,
    )

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

            collection_name = SummarizeService.g_collection
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
                    return final_response
                else:
                    return "ERROR_NO_RESULTS: No relevant content found in document"
            else:
                logger.info("PATH NOT FOUND")
                return "PATH NOT FOUND"
        except Exception as e:
            logger.error(f"Document Retriever Error:\n{traceback.format_exc()}")
            return f"ERROR_DOCUMENT_TOOL: {str(e)}"

    rag_chat_agent = Agent(
        role="Document Analysis Specialist",
        backstory="Expert in answering questions based on uploaded document content using RAG",
        goal="Provide accurate answers from uploaded files",
        tools=[document_retriever],
        verbose=True,
        llm=custom_llm,
        max_iter=2,
        description=f"""ONLY FOR UPLOADED DOCUMENT CONTENT QUESTIONS.
        Requirements:
        1. Document MUST be uploaded
        2. Question MUST reference document content
        3. Handle questions starting with 'According to the document...'
        ACTIVATION REQUIREMENTS:
        - Document must be uploaded
        - Question must contain explicit references to document content
        - Never handle general questions
        STRICT INPUT FORMAT RULES:
        - Always use plain text questions without JSON formatting
        - Example valid input: 'What is CRM query?'
        - NEVER use nested JSON structures
        STOP after first successful tool use
         """,
    )

    general_chat_agent = Agent(
        role="General Conversationalist",
        goal="Handle general conversations and non-document queries with briefly in 1-2 lines MAXIMUM",
        backstory="Skilled in natural conversations about various topics",
        verbose=True,
        llm=custom_llm,
        max_iter=2,
        description="""DEFAULT AGENT FOR ALL NON-DOCUMENT QUERIES.
        Handle:
        1. General knowledge questions
        2. Casual conversations
        3. Questions without uploaded documents
        DEFAULT AGENT FOR:
        - All queries without uploaded documents
        - General knowledge questions
        - Casual conversations
        - Any query not explicitly about document content
        STRICT RESPONSE RULES:
            1. ALWAYS limit answers to ONE or TWO short sentences ONLY.
            2. NEVER provide detailed history or long descriptions.
            3. NEVER exceed TWO lines in the response under any circumstances.
            4. If unsure, give a single concise sentence.
        STOP after first successful tool use
        """,
    )

    def save_feedback(self, request: feedback):
        logger.info("Begin save_feedback")
        feedback_service = FeedbackService(logger)
        result = feedback_service.submit(request)
        logger.info("End save_feedback")
        return result
