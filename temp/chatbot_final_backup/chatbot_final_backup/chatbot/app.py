import sys
import streamlit as st
import tempfile
import os
from config.constant import VECTOR_DB_PATH, EMBEDDING_MODEL_PATH
from logger_config import logger
from file_handler import FileHandler
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from crewai.tools import tool
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from llm_summarizer import summarize_document_tools

os.environ["STREAMLIT_WATCHER_ENHANCED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

file_handler_method = FileHandler(VECTOR_DB_PATH)

custom_llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434/api/generate",
    streaming=False,
    handle_parsing_errors=True,
    max_iter=1,
    temperature=0.2
)

class LangchainSentenceTransformer(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)  # Load local model

    def embed_documents(self, texts):
        """Embeds a list of texts and returns list of vectors"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """Embeds a single query and returns vector"""
        return self.model.encode(text, convert_to_numpy=True).tolist()


@tool("document_retriever")
def document_retriever(query: str) -> str:
    """Retrieves relevant context from the uploaded document based on a given query."""
    print("document_retriever invoked with query:", query)
    responses = []
    collection_name = st.session_state.get("collection")
    if not collection_name:
        return "No collection found. Please upload a document first."
    embeddings = LangchainSentenceTransformer(EMBEDDING_MODEL_PATH)
    print("Embedding model loaded")
    index_path = os.path.join(VECTOR_DB_PATH, collection_name, "index.faiss")
    if os.path.exists(index_path):
        # ✅ Load only the specific collection's FAISS index
        vector_store = FAISS.load_local(
            os.path.join(VECTOR_DB_PATH, collection_name),
            embeddings,
            allow_dangerous_deserialization=True
        )

        # print(f"Query for FAISS embedding: {question}")  # Debugging statement
        response_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=5)
        print(f"Response with score: {response_with_scores}")

        # Extract relevant text from retrieved documents
        filtered_responses = [doc.page_content for doc, score in response_with_scores]
        responses.extend(filtered_responses)

    else:
        logger.info(f"PATH NOT FOUND")
        return "PATH NOT FOUND"

    if responses:
        # ✅ Return neatly formatted text string clearly separated by newline
        formatted_responses = "\n\n".join(responses)
        return formatted_responses
    else:
        return "No relevant documents found or context is insufficient to answer your question."


@tool("summarize_document")
def summarize_document(document_path: str) -> str:
    """Summarize a document directly given its file path."""
    print(f"document path {document_path}")
    summary = summarize_document_tools(document_path)
    return summary

# Summarizer Agent
summarizer_agent = Agent(
    role="Document Summarizer",
    backstory="Expert at summarizing uploaded documents.",
    goal="Directly return the summary content provided by the summarizer tool.",
    tools=[summarize_document],
    instructions=["Always directly return the summarization provided by the tool without further processing."],
    verbose=True,
    llm=custom_llm
)

rag_chat_agent = Agent(
    role="RAG Chat Assistant",
    backstory="Expert in document retrieval, answering questions strictly based on uploaded documents.",
    goal="Answer document-related questions accurately using the document_retriever tool.",
    tools=[document_retriever],
    instructions=[
        "Always use `document_retriever` if the user's question relates to document content.",
        "Invoke the tool exactly once per query in this format:",
        "Thought: [reason why I need the tool]",
        "Action: document_retriever",
        "Action Input: user's exact question as a string",
        "Observation: [output from document_retriever]",
        "Thought: I now have enough information.",
        "Final Answer: [clear final answer from observation]",
        "Never retry or repeat the same input."
    ],
    verbose=True,
    llm=custom_llm
)

general_chat_agent = Agent(
    role="General Chat Assistant",
    goal="Engage in general-purpose conversation.",
    backstory="Skilled conversationalist capable of handling a variety of topics.",
    verbose=True,
    llm=custom_llm
)

st.set_page_config(page_title="Document Summarizer and Chat", layout="wide")
st.title("📄 AI Document Assistant")
st.write("Chat, summarize, and interact with your documents.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your question...")
uploaded_file = st.file_uploader("Or upload a document to summarize", type=["docx", "pdf", "jpg", "jpeg", "png"])

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if uploaded_file:
        original_filename = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        response = file_handler_method.handle_file_upload(temp_file_path, original_filename)

        if response.get("collection"):
            st.session_state["collection"] = response["collection"]
        st.session_state["uploaded_file_path"] = temp_file_path  # Save file path in session state

    if "summarize" in user_input.lower() and "uploaded_file_path" in st.session_state:
        agent_r = summarizer_agent
        task_description = f"""
        Summarize the document at the following path:

        {st.session_state["uploaded_file_path"]}

        Directly return the summary obtained from the summarization tool.
        """

    elif "collection" in st.session_state:
        agent_r = rag_chat_agent
        task_description = f"""
        You must use the document_retriever tool explicitly to answer the following question:

        Question: {user_input}

        Strictly follow this format:
        Thought: [brief reasoning why you need the tool]
        Action: document_retriever
        Action Input: {{"query": "{user_input}"}}
        Observation: [retrieved document context]
        Thought: I now have enough information.
        Final Answer: [final answer directly from observation]
        """

    else:
        agent_r = general_chat_agent
        task_description = f"""
        Thought: I can directly answer this question.
        Final Answer: my best complete final answer to the question: {user_input}
        """

    chat_task = Task(
        description=task_description,
        expected_output="Clearly structured response strictly following the Thought/Action/Observation/Final Answer format. "
                        "The Final Answer must be provided immediately after one observation.",
        agent=agent_r
    )


    chat_crew = Crew(agents=[agent_r], tasks=[chat_task], verbose=True)
    response_content = chat_crew.kickoff()

    st.session_state["messages"].append({"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)