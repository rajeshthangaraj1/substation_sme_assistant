# D:\rajesh\python\substation_SME_assistant\src\chat_handler.py

import os
import json
import numpy as np
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
genai.configure(api_key=GEMINI_API_KEY)


def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


class ChatHandler:
    def __init__(self, vector_db_path=VECTOR_DB_PATH):
        self.vector_db_path = vector_db_path
        self.embeddings = GeminiEmbeddings()

    def _load_all_vectorstores(self):
        collections = {}
        if not os.path.exists(self.vector_db_path):
            return collections

        for folder in os.listdir(self.vector_db_path):
            folder_path = os.path.join(self.vector_db_path, folder)
            index_file = os.path.join(folder_path, "index.faiss")
            if os.path.exists(index_file):
                try:
                    vs = FAISS.load_local(
                        folder_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    collections[folder] = vs
                except Exception as e:
                    logger.error(f"Error loading FAISS {folder}: {e}")
        return collections

    def search_all_files(self, query, k=5):
        results = []
        collections = self._load_all_vectorstores()

        if not collections:
            return [{"content": "No documents found. Please upload files first."}]

        query_vector = self.embeddings.embed_query(query)

        for name, vs in collections.items():
            try:
                docs = vs.similarity_search_with_relevance_scores(query, k=k)
                for doc, score in docs:
                    results.append({
                        "collection": name,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                    })
            except Exception as e:
                logger.error(f"Error searching in {name}: {e}")

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:k]

    def build_prompt(self, query, results):
        context_parts = []
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("type") == "image":
                context_parts.append(
                    f"[Relevant image from {meta.get('filename')} page {meta.get('page','?')}]"
                )
            else:
                context_parts.append(
                    f"From {meta.get('filename')} (page {meta.get('page','?')}):\n{r['content']}"
                )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are an expert assistant helping with multimodal document Q&A.

### User Question:
{query}

### Retrieved Context:
{context}

### Instructions:
- Answer ONLY using the retrieved context.
- If context is insufficient, say: "The documents do not contain enough information."
- Provide a clear, structured answer.
- Use bullet points if multiple points exist.
"""
        return prompt

    def answer_question(self, query, k=5):
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
