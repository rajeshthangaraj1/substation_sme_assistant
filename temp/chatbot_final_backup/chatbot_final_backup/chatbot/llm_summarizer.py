from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config.constant import MODEL_NAME,OLLAMA_SERVER_URL,LLM_CHUNK_SIZE,LLM_CHUNK_OVERLAP,TEMPERATURE,TIMEOUT
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import NLTKTextSplitter
from helper.common import load_document
from logger_config import setup_logger

logger = setup_logger()
# class StatusCallbackHandler(BaseCallbackHandler):
#     def __init__(self, status_container):
#         self.status_container = status_container
#         self.chunk_status = {}  # Dictionary to track all chunks
#
#     def on_chain_start(self, serialized, inputs, **kwargs):
#         """When a new chunk starts processing, update the UI."""
#         chunk_id = len(self.chunk_status) + 1
#         self.chunk_status[chunk_id] = "🟡 Processing Chunk..."
#         self.update_status()
#
#     def on_chain_end(self, outputs, **kwargs):
#         """When a chunk is completed, mark it as done."""
#         chunk_id = len(self.chunk_status)
#         self.chunk_status[chunk_id] = "✅ Completed Chunk"
#         self.update_status()
#
#     def update_status(self):
#         """Update the Streamlit UI to show progress of all chunks."""
#         status_text = "\n".join([f"{self.chunk_status[k]} {k}" for k in sorted(self.chunk_status.keys())])
#         self.status_container.write(status_text)

class StatusCallbackHandler(BaseCallbackHandler):
    def __init__(self, status_container):
        self.status_container = status_container
        self.chunk_status = {}  # Track chunk status using run_id

    def on_chain_start(self, serialized, inputs, run_id, **kwargs):
        """Triggered when a chunk starts processing."""
        self.chunk_status[run_id] = "🟡 Processing Chunk..."
        self.update_status()

    def on_chain_end(self, outputs, run_id, **kwargs):
        """Triggered when a chunk finishes processing."""
        self.chunk_status[run_id] = "✅ Completed Chunk"
        self.update_status()

    def update_status(self):
        """Update Streamlit UI with accurate progress."""
        sorted_status = sorted(self.chunk_status.items(), key=lambda x: x[0])
        status_text = "\n".join(
            [f"{status} ({idx+1})" for idx, (run_id, status) in enumerate(sorted(self.chunk_status.items()))]
        )
        self.status_container.write(status_text)

def summarize_document_tools(document_file_or_text):
    """Summarize a Word document using LangChain"""
    try:
        text = load_document(document_file_or_text)
        if not text:
            logger.error("No text extracted. Exiting...")
            exit()
        # Initialize your callback handler
        logger.info("Starting document summarization")
        print("Starting document summarization")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=LLM_CHUNK_SIZE, chunk_overlap=int(0.5 * LLM_CHUNK_SIZE))
        docs = text_splitter.create_documents([text])
        # text_splitter = NLTKTextSplitter()
        # docs = text_splitter.split_text(text)
        logger.info("Document successfully split into chunks.")
        print("Document successfully split into chunks.")
        llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_SERVER_URL, temperature=TEMPERATURE,timeout=TIMEOUT)
        logger.info("LLM Model Initialized: %s", MODEL_NAME)

        map_prompt = PromptTemplate.from_template(
            "You are a highly skilled **Summarization Assistant** specializing in extracting key insights from technical and business documents. "
            "Your task is to produce a clear and concise summary focusing on actionable insights, key takeaways, and technical details of upcoming projects and features. "
            "Ensure that the summary reflects future requirements and planned developments, avoiding language that implies current or past completion.\n\n"
            "**Instructions:**\n"
            "- Identify the **main topic** and its significance.\n"
            "- Highlight **key points**, processes, or technical details.\n"
            "- Exclude redundant, generic, or unimportant information.\n"
            "- Present the summary in a **single paragraph** without bullet points or headings.\n\n"
            "- Use **future-oriented language** to accurately represent planned developments.\n\n"
            "- **Omit all personal names and identifiers** to ensure privacy.\n\n"
            "**Text:**\n\n{text}\n\n"
            "**Summary:**"
        )

        combine_prompt = PromptTemplate.from_template(
            "You are a professional summarizer tasked with refining and merging multiple partial summaries into a **concise, well-structured** final summary that emphasizes future plans and requirements.\n\n"
            "**Instructions:**\n"
            "- Merge all summaries into a **single, cohesive paragraph**.\n"
            "- Remove redundant or repetitive content.\n"
            "- Ensure the summary remains under **40 lines** while maintaining clarity.\n"
            "- Avoid using headings or bullet points; present the information as a unified narrative.\n"
            "- Use **future-oriented language** to accurately represent planned developments.\n\n"
            "- **Omit all personal names and identifiers** to ensure privacy.\n\n"
            "**Now refine the summaries below into a consolidated final summary:**\n\n"
            "**Summaries:**\n\n{text}\n\n"
            "**Final Summary:**"
        )

        # Create processing chain
        map_chain = map_prompt | llm
        reduce_chain = combine_prompt | llm

        map_reduce_chain = (
            RunnableLambda(lambda x: [{"text": doc.page_content} for doc in x])  # Convert list of docs to dict
            # RunnableLambda( lambda x:[{"text": doc} for doc in x])  # Directly use the chunked text
            | map_chain.map()  # Apply map chain
            | reduce_chain  # Combine results
        )

        logger.info("Running summarization pipeline.")
        print("Running summarization pipeline.")
        result = map_reduce_chain.invoke(docs)
        logger.info("Summary generated successfully.")
        print("Summary generated successfully.")

        return result.content if hasattr(result, 'content') else result

    except Exception as e:
        logger.error("Error in summarization: %s", str(e))
        raise e
