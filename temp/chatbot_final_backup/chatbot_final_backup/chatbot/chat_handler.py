import os
import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from config.constant import EMBEDDING_MODEL_PATH,MODEL_NAME,OLLAMA_SERVER_URL
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from logger_config import setup_logger
from langchain_ollama import OllamaLLM

logger = setup_logger()
class LangchainSentenceTransformer(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)  # Load local model

    def embed_documents(self, texts):
        """Embeds a list of texts and returns list of vectors"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """Embeds a single query and returns vector"""
        return self.model.encode(text, convert_to_numpy=True).tolist()


class ChatHandler:
    def __init__(self, vector_db_path):
        self.vector_db_path = vector_db_path
        local_model_path = EMBEDDING_MODEL_PATH  # Change this to your actual path
        # ✅ Use the correct embeddings wrapper instead of a raw function
        self.embeddings = LangchainSentenceTransformer(local_model_path)
        self.llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_SERVER_URL)  # ✅ Initialize LLaMA model

    def answer_question(self, question):
        responses = []
        # ✅ Get the collection name from session state
        collection_name = st.session_state.get("collection")  # ✅ Retrieve collection name
        logger.info(f"Chat Collection Name Retrieved: {collection_name}")  # Debugging log

        if not collection_name:
            return "No document has been processed yet. Please upload a document first."

        index_path = os.path.join(self.vector_db_path, collection_name, "index.faiss")
        if os.path.exists(index_path):
            # ✅ Load only the specific collection's FAISS index
            vector_store = FAISS.load_local(
                os.path.join(self.vector_db_path, collection_name),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # print(f"Query for FAISS embedding: {question}")  # Debugging statement
            response_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=5)
            print(f"Response with score: {response_with_scores}")

            # Extract relevant text from retrieved documents
            filtered_responses = [doc.page_content for doc, score in response_with_scores]
            responses.extend(filtered_responses)

        else:
            logger.info(f"PATH NOT FOUND")
            return "PATH NOT FOUND"

        if responses:
            # print(f"🔍 FAISS Retrieved Chunks:\n{responses}")  # ✅ Debugging
            logger.info(f"🔍 FAISS Retrieved Chunks:\n{responses}")


            prompt = self._generate_prompt(question, responses)

            # print(f"📝 Generated Prompt Sent to LLaMA:\n{prompt}")  # ✅ Debugging
            logger.info(f"📝 Generated Prompt Sent to LLaMA:\n{prompt}")

            llm_response = self.call_ollama(prompt)
            return llm_response

        return "No relevant documents found or context is insufficient to answer your question."

    def _generate_prompt(self, question, documents):
        """
        Generate a structured prompt to ensure the AI assistant extracts relevant information
        from the document, even if the answer is not explicitly stated.
        """
        context = "\n\n".join([f"{i + 1}. {doc.strip()}" for i, doc in enumerate(documents[:6])])

        prompt = f"""
        You are an AI assistant that extracts precise answers from a given document.
        Your task is to analyze the document context and provide a clear, direct, and relevant answer.

        ### **Document Context:**
        {context}

        ### **User's Question:**
        {question}

        ### **Guidelines:**
        - **Base your answer only on the document.** Do not use external knowledge.
        - **If the answer is explicit in the document, provide the exact text.**
        - **If the document contains related information, summarize key points.**
        - **If the answer is not directly available, infer a reasonable explanation based on the given context.**
        - **Avoid stating "no relevant information" unless absolutely nothing related is found.**
        - **Your response should be structured, with bullet points for clarity.**

        ### **Expected Response Format:**
        - **Direct Answer (if available)**: Extract the relevant passage.
        - **Summary of Key Points**: If the answer is indirect, summarize the key details.
        - **Inference (if applicable)**: Provide a logical interpretation based on the context.

        **Now, analyze the document and answer the question:**
        """
        return prompt

    def call_ollama(self, prompt):
        """
        Calls the Ollama LLaMA model and gets the response.
        """
        try:
            logger.info("Sending request to Ollama LLaMA model...")

            response = self.llm.generate([prompt])  # ✅ Change from .invoke() to .generate()
            # print(f"Ollama content check : {response}")
            # logger.info(f"Ollama content check : {response}")

            answer = response.generations[0][0].text.strip() if hasattr(response, "generations") else response

            # logger.info(f"Ollama Response: {answer}")
            return answer

        except Exception as e:
            logger.error(f"Error in calling LLaMA model: {str(e)}")
            return "Error occurred while generating an answer. Please try again."

