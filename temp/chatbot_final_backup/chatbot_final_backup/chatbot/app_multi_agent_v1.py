import sys
import streamlit as st
import tempfile
import os
import json
import traceback
from config.constant import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_PATH ,
    LONG_TERM_MODEL_PATH,
    SHORT_TERM_MODEL_PATH,
    CREW_MODEL_NAME,
    CREW_OLLAMA_SERVER_URL,
    CREW_TEMPERATURE
)
from logger_config import setup_logger
from file_handler import FileHandler
from crewai import Agent, Task, Crew,Process
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

logger = setup_logger()

# Environment variables
os.environ["STREAMLIT_WATCHER_ENHANCED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_LOCAL_MODE"] = "True"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# torch class issue
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

# Initialize file handler
file_handler_method = FileHandler(VECTOR_DB_PATH)


system_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the routing manager responsible for delegating tasks to coworker agents.

⚠️ VERY IMPORTANT:
When you pass task and context to the tool, you MUST follow this format:
{
  "task": "Plain text only",
  "context": "Plain text only",
  "coworker": "Agent name exactly as defined"
}

❌ NEVER do this:
- Do NOT pass objects or dictionaries like {"description": "...", "type": "str"}.
- Do NOT nest the task or context in any way.
- Do NOT add metadata or explanation keys.

✅ ALWAYS:
- Use string for both `task` and `context`
- Coworker must be a valid agent name
- Everything must be clean and minimal

Example:
{
  "task": "Summarize the uploaded document on renewable energy.",
  "context": "User asked for a summary and a document is uploaded.",
  "coworker": "Document Summarization Expert"
}
<|eot_id|>"""


prompt_template = """<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>"""
response_template = """<|start_header_id|>assistant<|end_header_id|>{{ .Response }}<|eot_id|>"""


# Initialize LLM
custom_llm = LLM(
    model=CREW_MODEL_NAME,
    base_url=CREW_OLLAMA_SERVER_URL,
    streaming=False,
    handle_parsing_errors=True,
    max_iter=2,
    temperature=CREW_TEMPERATURE,
)

# SQL Retriever Tool
@tool("sql_retriever")
def sql_retriever(query: dict) -> str:
    """Handles SQL-style capacity planning questions and returns the final output. Returns error codes for agent routing:
    - ERROR_SQL_TOOL: When Sql Tool Error
    """
    try:
        if isinstance(query, dict):
            query = query.get("description", "")
        sqlanswer = sqlagenttool(query)
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
        if isinstance(query, dict):
            query = query.get("description", "")  # Extract the actual query string


        collection_name = st.session_state.get("collection")
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

# Summarization Tool
@tool("summarize_document")
def summarize_document() -> str:
    """Summarize the currently uploaded document. NO INPUT NEEDED."""

    try:
        if not st.session_state.get("collection"):
            return "Final Answer: ERROR_NO_DOCUMENT: Please upload a document first"

        document_path = st.session_state.get("uploaded_file_path")
        if not document_path or not os.path.exists(document_path):
            return "Final Answer: ERROR_INVALID_PATH: No valid document found"

        result = summarize_document_tools(document_path)
        final_response = f"Final Answer: {result}"
        logger.info(f"Final Answer from tool:   {final_response}")

        st.session_state["last_response"] = final_response
        return final_response
    except Exception as e:
        logger.error(f"Summarize Document Error:\n{traceback.format_exc()}")
        return f"Final Answer: ERROR_SUMMARIZE_TOOL: {str(e)}"

# ✅ Agent Definitions with Enhanced Descriptions
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
    2. Do NOT generate SQL manually – let the tool handle the logic.
    3. Stop after tool returns Final Answer.
    4. NEVER modify or process tool output
    5. Return tool response directly
    6. Never delegate or reroute to other agents.
    """,
)


# ✅ Agent Definitions with Enhanced Descriptions
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

# ========================
# Streamlit UI Setup
# ========================
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("📄 Ask Eand")
st.write("Chat, summarize, and interact with your documents and DB.")

# ✅ Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ✅ Display previous chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Add file upload inside the chat box using columns
col1, col2 = st.columns([0.70, 0.30])

with col1:
    user_input = st.chat_input("Type your question...")
with col2:
    uploaded_file = st.file_uploader(" ", type=["docx", "pdf", "txt"], label_visibility='collapsed')

# ✅ Handle file upload
if uploaded_file:
    original_filename = uploaded_file.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    response = file_handler_method.handle_file_upload(temp_file_path, original_filename)
    if response.get("collection"):
        st.session_state["collection"] = response["collection"]
    st.session_state["uploaded_file_path"] = temp_file_path

    st.session_state["messages"].append(
        {"role": "user", "content": f"📎 Uploaded file: **{uploaded_file.name}**"}
    )

# ✅ Handle user input
if user_input:
    print(f"user input: {user_input}")
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    context = {
        "query": user_input.lower(),
        "document_uploaded": "yes" if st.session_state.get("collection") else "no",
        "document_status": "Uploaded" if st.session_state.get("collection") else "Not uploaded",
        "explicit_summary_request": any(keyword in user_input.lower() for keyword in ["summarize", "summary", "overview"]),
        "is_sql_query": any(keyword in user_input.lower() for keyword in ["utilization", "capacity", "vowifi", "report", "kpi", "node", "network"])
    }

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
            - Query: "What is the utilization of VoWiFi Calls at 10:00 AM on April 3rd-2024 ?"
              - Selected Agent: SQL Data Specialist
            - Query: "What's the capital of France?"
              - Selected Agent: General Chat Agent

            Routing Conditions:
            1. FIRST CHECK - Document Uploaded? {context['document_uploaded']}
               - If NO: 
                   a. If SQL keywords → Route to SQL Data Specialist
                   b. Else → General Chat
               - If YES:
                   a. Is Summary Requested? {context['explicit_summary_request']} → Route to Document Summarization Expert
                   b. Else If SQL Query Detected? {context['is_sql_query']} → Route to SQL Data Specialist
                   c. Else If Question References Document? {"document" in context['query']} → Route to Document Analysis Specialist
                   d. Else → Route to General Chat Agent
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
        # system_template=system_template,
        # prompt_template=prompt_template,
        # response_template=response_template
    )

    # ✅ Corrected Autonomous Task Configuration
    def create_autonomous_task(user_query: str):
        has_document = "collection" in st.session_state
        is_sql_query = any(
            keyword in user_query.lower()
            for keyword in ["utilization", "capacity", "vowifi", "report", "kpi", "node", "network"]
        )
        is_summary_request = any(
            keyword in user_query.lower()
            for keyword in ["summarize", "summary", "overview"]
        )
        return Task(
            description=f"""Analyze and process the user query:
            User Query: {user_query}
            Document Status: {'Uploaded' if has_document else 'Not Uploaded'}
            SQL Query Detected: {is_sql_query}
            Summary Request Detected: {is_summary_request}

            Available agent specializations:
            1. Document Summarization Expert - .docx/.pdf summaries
            2. Document Analysis Specialist - Document content questions
            3. SQL Data Specialist - Capacity planning, KPI, VoWiFi or utilization queries
            4. Conversational Assistant - General chat

            Routing Rules:
            1. If SQL keywords found → Route to SQL Data Specialist
            2. Else if summary keywords and document exists → Summarizer
            3. Else if document exists and query references content → Document Analyst
            4. All other cases → General Chat
            
            """,
            expected_output="Immediate, final response prefixed with 'Final Answer:'",
            context=[{
                "has_document": has_document,
                "query": user_query,
                "description": user_query,
                "is_sql_query": is_sql_query,
                "explicit_summary_request": is_summary_request,
                "expected_output": "Relevant response matching query intent"
            }],
            config={
                "allow_delegation": True,
                "stop_on_success": True,
            }
        )

    chat_task = create_autonomous_task(user_input)
    # Pre-process input before sending to CrewAI
    crew = Crew(
        agents=[summarizer_agent, rag_chat_agent, general_chat_agent, sql_agent],
        tasks=[chat_task],
        process=Process.hierarchical,
        verbose=True,
        max_iter=3,
        manager_agent=manager_agent,
        # memory=True,
        # embedder={
        #     "provider": "custom",
        #     "config": {
        #         "embedder": SentenceTransformerEmbedder(EMBEDDING_MODEL_PATH)
        #     }
        # },
        # long_term_memory=LongTermMemory(
        #     storage=LTMSQLiteStorage(db_path=LONG_TERM_MODEL_PATH),
        # ),
        # short_term_memory=ShortTermMemory(
        #     storage=RAGStorage(
        #         embedder_config={
        #             "provider": "custom",
        #             "config": {
        #                 "embedder": SentenceTransformerEmbedder(EMBEDDING_MODEL_PATH)
        #             }
        #         },
        #         type="short_term",
        #         path=SHORT_TERM_MODEL_PATH,
        #     )
        # ),
        # entity_memory=EntityMemory(
        #     storage=RAGStorage(
        #         embedder_config={
        #             "provider": "custom",
        #             "config": {
        #                 "embedder": SentenceTransformerEmbedder(EMBEDDING_MODEL_PATH)
        #             }
        #         },
        #         type="short_term",
        #         path=SHORT_TERM_MODEL_PATH
        #     )
        # ),
        # memory_config=None
    )
    # ✅ Let CrewAI handle task assignment automatically
    try:
        crew_input = {
            "query": user_input,
            "context": json.dumps(context)
        }
        response_content = crew.kickoff(inputs=crew_input)
        print(f"After crew hit the agent")
        # ✅ Directly return the tool output if it's already in session state
        if st.session_state.get("last_response"):
            response_content = st.session_state["last_response"]
            st.session_state["last_response"] = None  # Reset state after use

    except Exception as e:
        response_content = "Apologies, I encountered an error. Let me try that again..."
        traceback_str = traceback.format_exc()
        logger.error(f"Full traceback:\n{traceback_str}")
        response_content = f"System error: {str(e)}"

    print(f"final output {response_content}")
    # Handle response parsing
    if isinstance(response_content, list):
        final_response = "\n".join(response_content)
    elif hasattr(response_content, 'output'):
        final_response = response_content.output
    else:
        final_response = str(response_content)

    # Fallback to general chat on errors
    if any(err in final_response for err in ["ERROR", "Error"]):
        final_response = general_chat_agent.execute({"task": user_input})

    st.session_state["messages"].append({"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)