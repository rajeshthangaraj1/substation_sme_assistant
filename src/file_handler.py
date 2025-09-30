# D:\rajesh\python\substation_SME_assistant\src\file_handler.py

import os
import io
import json
import hashlib
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import google.generativeai as genai

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredImageLoader,
)

from config.constants import (
    VECTOR_DB_PATH,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_CHUNK_OVERLAP,
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
)
from src.utils.logger import get_logger

logger = get_logger()

# ========== GEMINI SETUP ==========
genai.configure(api_key=GEMINI_API_KEY)


class GeminiEmbeddings(Embeddings):
    """Custom Embedding class using Gemini Embedding API"""

    def __init__(self, model=GEMINI_EMBED_MODEL):
        self.model = model

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            try:
                resp = genai.embed_content(model=self.model, content=text)
                vectors.append(resp["embedding"])
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                vectors.append([0.0] * 768)  # fallback vector
        return vectors

    def embed_query(self, text):
        try:
            resp = genai.embed_content(model=self.model, content=text)
            return resp["embedding"]
        except Exception as e:
            logger.error(f"Embedding query error: {e}")
            return [0.0] * 768


class FileHandler:
    def __init__(self, vector_db_path=VECTOR_DB_PATH):
        self.vector_db_path = vector_db_path
        os.makedirs(self.vector_db_path, exist_ok=True)
        self.embeddings = GeminiEmbeddings()

    def handle_file_upload(self, file_path, original_filename):
        """Process uploaded file -> extract text+images -> chunk -> embed -> save FAISS"""

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            file_hash = hashlib.md5(content).hexdigest()
            normalized_filename = original_filename.lower().replace(" ", "_")
            file_key = f"{normalized_filename}_{file_hash}"
            vector_store_dir = os.path.join(self.vector_db_path, file_key)
            os.makedirs(vector_store_dir, exist_ok=True)

            index_path = os.path.join(vector_store_dir, "index.faiss")
            if os.path.exists(index_path):
                logger.info("File already processed. Skipping reprocessing.")
                return {"message": "File already processed", "collection": file_key}

            # Step 1: Extract text (and OCR if needed)
            texts = self.load_document(file_path)

            if not texts:
                return {"message": "No text extracted", "collection": file_key}

            # Step 2: Chunk text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=EMBEDDING_CHUNK_SIZE,
                chunk_overlap=EMBEDDING_CHUNK_OVERLAP,
            )
            chunks = splitter.split_text(texts)

            # Step 3: Embed & Save FAISS
            vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
            vector_store.save_local(vector_store_dir)

            metadata = {
                "filename": original_filename,
                "size": len(content),
                "chunks": len(chunks),
            }
            with open(os.path.join(vector_store_dir, "metadata.json"), "w") as md:
                json.dump(metadata, md)

            logger.info(f"✅ File processed: {original_filename}")
            return {"message": "File processed successfully", "collection": file_key}

        except Exception as e:
            logger.error(f"Error handling file upload: {str(e)}")
            return {"message": f"Error: {str(e)}"}

    def load_document(self, file_path):
        """Extract text from doc/pdf/images with OCR fallback"""
        ext = file_path.split(".")[-1].lower()
        text = ""

        try:
            if ext == "pdf":
                loader = UnstructuredPDFLoader(file_path)
                docs = loader.load()
                text = " ".join([d.page_content for d in docs])

            elif ext in ["docx", "doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                text = " ".join([d.page_content for d in docs])

            elif ext in ["jpg", "jpeg", "png"]:
                loader = UnstructuredImageLoader(file_path, mode="elements", strategy="ocr_only")
                docs = loader.load()
                text = " ".join([d.page_content for d in docs])

            elif ext == "txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            elif ext in ["csv", "xlsx"]:
                text = self.load_table(file_path)

            else:
                raise ValueError("Unsupported file type")

        except Exception as e:
            logger.error(f"Document load error: {e}")
            text = ""

        return text.strip()

    def load_table(self, file_path):
        """Convert CSV/XLSX into readable text"""
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            df = df.fillna("N/A")
            rows = []
            for _, row in df.iterrows():
                row_text = ", ".join([f"{k}: {v}" for k, v in row.to_dict().items()])
                rows.append(row_text)
            return "\n".join(rows)
        except Exception as e:
            logger.error(f"Table load error: {e}")
            return ""
