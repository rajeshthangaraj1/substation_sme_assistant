# D:\rajesh\python\substation_SME_assistant\src\chat_handler.py

import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from config.constants import (
    VECTOR_DB_PATH,
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
)
from src.file_handler import GeminiEmbeddings
from src.utils.logger import get_logger

logger = get_logger()

# ✅ Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class ChatHandler:
    def __init__(self, vector_db_path=VECTOR_DB_PATH):
        self.vector_db_path = vector_db_path
        self.embeddings = GeminiEmbeddings()

    def _load_all_vectorstores(self):
        """Load all FAISS collections from vector_db_path"""
        collections = {}
        if not os.path.exists(self.vector_db_path):
            return collections

        for folder in os.listdir(self.vector_db_path):
            folder_path = os.path.join(self.vector_db_path, folder)
            index_file = os.path.join(folder_path, "index.faiss")
            if os.path.exists(index_file):
                try:
                    vs = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)
                    collections[folder] = vs
                except Exception as e:
                    logger.error(f"Error loading FAISS {folder}: {e}")
        return collections

    def search_all_files(self, query, k=5):
        """Search query across ALL FAISS collections"""
        results = []
        collections = self._load_all_vectorstores()

        if not collections:
            return ["No documents found. Please upload files first."]

        for name, vs in collections.items():
            try:
                docs = vs.similarity_search_with_relevance_scores(query, k=k)
                for doc, score in docs:
                    results.append({
                        "collection": name,
                        "content": doc.page_content,
                        "score": score
                    })
            except Exception as e:
                logger.error(f"Error searching in {name}: {e}")

        # sort results by relevance
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:k]

    def build_prompt(self, query, results):
        """Build structured prompt for Gemini"""
        context = "\n\n".join(
            [f"From {r['collection']}:\n{r['content']}" for r in results]
        )

        prompt = f"""
You are an expert assistant helping with multi-document Q&A.

### User Question:
{query}

### Retrieved Document Context:
{context}

### Instructions:
- Answer ONLY using the information in the retrieved context.
- If context is insufficient, say: "The documents do not contain enough information."
- Provide a clear, structured answer.
- Use bullet points if multiple points exist.
"""
        return prompt

    def answer_question(self, query, k=5):
        """Retrieve context → Ask Gemini → Return Answer"""
        results = self.search_all_files(query, k=k)

        if not results or (len(results) == 1 and "No documents found" in results[0]["content"]):
            return "No documents available. Please upload files."

        prompt = self.build_prompt(query, results)

        try:
            response = genai.GenerativeModel(GEMINI_CHAT_MODEL).generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Error occurred while generating answer with Gemini."
