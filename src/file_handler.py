# D:\rajesh\python\substation_SME_assistant\src\file_handler.py

import os
import json
import hashlib
import pandas as pd
import fitz  # PyMuPDF for PDF rendering
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from docx import Document  # for extracting DOCX images

from config.constants import (
    VECTOR_DB_PATH,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_CHUNK_OVERLAP,
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
)
from src.utils.logger import get_logger

logger = get_logger()

# ========== CONFIG ==========
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
                vectors.append([0.0] * 768)
        return vectors

    def embed_query(self, text):
        try:
            resp = genai.embed_content(model=self.model, content=text)
            return resp["embedding"]
        except Exception as e:
            logger.error(f"Embedding query error: {e}")
            return [0.0] * 768

    def embed_image_bytes(self, img_bytes):
        try:
            resp = genai.embed_content(
                model=self.model,
                content={"image": {"bytes": img_bytes}},
            )
            return resp["embedding"]
        except Exception as e:
            logger.error(f"Image embedding error: {e}")
            return [0.0] * 768

    def embed_image(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            return self.embed_image_bytes(img_bytes)
        except Exception as e:
            logger.error(f"Image embedding error: {e}")
            return [0.0] * 768


class FileHandler:
    def __init__(self, vector_db_path=VECTOR_DB_PATH):
        self.vector_db_path = vector_db_path
        os.makedirs(self.vector_db_path, exist_ok=True)
        self.embeddings = GeminiEmbeddings()

    def handle_file_upload(self, file_path, original_filename):
        """Process uploaded file → extract text/images → embed → save FAISS"""

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

            # Extract multimodal docs (text + images)
            docs = self.load_document(file_path, original_filename)

            if not docs:
                return {"message": "No content extracted", "collection": file_key}

            texts = []
            metadatas = []
            vectors = []

            for d in docs:
                if d["type"] == "text":
                    texts.append(d["content"])
                    metadatas.append(d["metadata"])
                elif d["type"] == "image":
                    # placeholder text for FAISS docstore
                    texts.append("[Image]")
                    metadatas.append(d["metadata"])
                    vectors.append(d["vector"])

            # Build FAISS with text entries
            vector_store = FAISS.from_texts(
                texts,
                embedding=self.embeddings,
                metadatas=metadatas,
            )

            # Replace FAISS vectors for image entries with real image embeddings
            for i, d in enumerate(docs):
                if d["type"] == "image":
                    vector_store.index.replace(i, d["vector"])

            # Save FAISS + metadata
            vector_store.save_local(vector_store_dir)

            with open(os.path.join(vector_store_dir, "metadata.json"), "w") as md:
                json.dump({"filename": original_filename}, md)

            logger.info(f"✅ File processed: {original_filename}")
            return {"message": "File processed successfully", "collection": file_key}

        except Exception as e:
            logger.error(f"Error handling file upload: {str(e)}")
            return {"message": f"Error: {str(e)}"}

    def load_document(self, file_path, filename):
        """Extract multimodal content: text + images"""
        ext = file_path.split(".")[-1].lower()
        docs = []

        try:
            # -------- PDF --------
            if ext == "pdf":
                loader = PyPDFLoader(file_path)
                all_docs = loader.load()

                pdf_doc = fitz.open(file_path)

                for page_num, page in enumerate(pdf_doc, start=1):
                    # Extract text for this page
                    page_text = " ".join(
                        [d.page_content for d in all_docs if d.metadata.get("page") == page_num]
                    ).strip()
                    if page_text:
                        docs.append({
                            "type": "text",
                            "content": page_text,
                            "metadata": {"filename": filename, "page": page_num, "type": "text"},
                        })

                    # Render page image → embedding
                    pix = page.get_pixmap(dpi=144)
                    img_bytes = pix.tobytes("png")
                    img_vector = self.embeddings.embed_image_bytes(img_bytes)
                    docs.append({
                        "type": "image",
                        "vector": img_vector,
                        "metadata": {"filename": filename, "page": page_num, "type": "image"},
                    })

            # -------- DOCX --------
            elif ext in ["docx", "doc"]:
                loader = Docx2txtLoader(file_path)
                all_docs = loader.load()
                text = " ".join([d.page_content for d in all_docs])
                if text:
                    docs.append({
                        "type": "text",
                        "content": text,
                        "metadata": {"filename": filename, "type": "text"},
                    })

                # Extract embedded images
                try:
                    doc = Document(file_path)
                    for rel in doc.part.rels.values():
                        if "image" in rel.target_ref:
                            img_bytes = rel.target_part.blob
                            img_vector = self.embeddings.embed_image_bytes(img_bytes)
                            docs.append({
                                "type": "image",
                                "vector": img_vector,
                                "metadata": {"filename": filename, "type": "image"},
                            })
                except Exception as e:
                    logger.warning(f"No images extracted from DOCX: {e}")

            # -------- TXT --------
            elif ext == "txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if text:
                    docs.append({
                        "type": "text",
                        "content": text,
                        "metadata": {"filename": filename, "type": "text"},
                    })

            # -------- CSV / Excel --------
            elif ext in ["csv", "xlsx"]:
                text = self.load_table(file_path)
                if text:
                    docs.append({
                        "type": "text",
                        "content": text,
                        "metadata": {"filename": filename, "type": "text"},
                    })

            # -------- Images --------
            elif ext in ["jpg", "jpeg", "png"]:
                img_vector = self.embeddings.embed_image(file_path)
                docs.append({
                    "type": "image",
                    "vector": img_vector,
                    "metadata": {"filename": filename, "type": "image"},
                })

            else:
                raise ValueError("Unsupported file type")

        except Exception as e:
            logger.error(f"Document load error: {e}")

        return docs

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
