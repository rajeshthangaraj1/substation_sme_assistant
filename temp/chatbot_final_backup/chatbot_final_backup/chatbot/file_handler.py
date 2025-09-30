import os
import hashlib
import io
import json
import pandas as pd
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document
from config.constant import EMBEDDING_MODEL_PATH,EMBEDDING_CHUNK_SIZE,EMBEDDING_CHUNK_OVERLAP
from logger_config import setup_logger
from langchain.embeddings.base import Embeddings
from helper.common import load_document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

class FileHandler:
    def __init__(self, vector_db_path):
        self.vector_db_path = vector_db_path
        local_model_path = EMBEDDING_MODEL_PATH  # Change this to your actual path
        self.embeddings = SentenceTransformer(local_model_path)  # Load locally

    def handle_file_upload(self, file_path, original_filename):
        try:
            # content = file_path
            with open(file_path, "rb") as file:
                content = file.read()
            file_hash = hashlib.md5(content).hexdigest()
            # normalized_filename = file.name.lower().replace(' ', '_')
            normalized_filename = original_filename.lower().replace(' ', '_')  # ✅ Fix here
            file_key = f"{normalized_filename}_{file_hash}"
            vector_store_dir = os.path.join(self.vector_db_path, file_key)
            os.makedirs(vector_store_dir, exist_ok=True)
            vector_store_path = os.path.join(vector_store_dir, "index.faiss")
            logger.info(f"File received: {file_path}, Size: {len(content)} bytes")
            logger.info(f"Generated File Key: {file_key}")
            logger.info(f"Vector Store Directory: {vector_store_dir}")
            logger.info(f"File name: {original_filename}")
            if os.path.exists(vector_store_path):
                logger.info("File already processed. Skipping processing.")
                # return {"message": "File already processed."}
                return {"message": "File already processed.", "collection": file_key}  # ✅ Return collection name

            # Process file based on type
            if file_path.endswith(".pdf"):
                texts, metadatas = self.load_and_split_pdf(file_path)
            elif file_path.endswith(".docx"):
                texts, metadatas = self.load_and_split_docx(file_path)
            elif file_path.endswith(".txt"):
                texts, metadatas = self.load_and_split_txt(content)
            elif file_path.endswith(".xlsx"):
                texts, metadatas = self.load_and_split_table(content)
            elif file_path.endswith(".csv"):
                texts, metadatas = self.load_and_split_csv(content)
            else:
                logger.error("Unsupported file format.")
                raise ValueError("Unsupported file format.")

            logger.info(f"before chunking: {texts}")
            if not texts:
                logger.warning("No text extracted from the file. Check the file content.")
                return {"message": "No text extracted from the file. Check the file content."}

            # ✅ **Step 1: Chunk the Text**
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=EMBEDDING_CHUNK_SIZE, chunk_overlap=EMBEDDING_CHUNK_OVERLAP)
            chunks = text_splitter.split_text(texts)  # 🔥 **This creates meaningful chunks**
            # ✅ **Step 2: Embed Each Chunk Separately**
            embeddings = LangchainSentenceTransformer(EMBEDDING_MODEL_PATH)
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)  # 🔥 **Chunk-wise Embeddings**
            # ✅ **Step 3: Save to FAISS**
            vector_store.save_local(vector_store_dir)

            metadata = {
                "filename": original_filename,
                "file_size": len(content),
            }
            metadata_path = os.path.join(vector_store_dir, "metadata.json")
            with open(metadata_path, 'w') as md_file:
                json.dump(metadata, md_file)
            logger.info("File processed successfully.")

            # return {"message": "File processed successfully."}
            return {"message": "File processed successfully.", "collection": file_key}  # ✅ Return collection name
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {"message": f"Error processing file: {str(e)}"}

    def load_and_split_pdf(self, file):
        texts = ""
        metadatas = []
        text = load_document(file)
        if not text:
            logger.error("No text extracted. Exiting...")
            exit()
        # texts.append(text)
        texts = text
        metadatas.append({"page_number": 10})
        return texts, metadatas

    def load_and_split_docx(self, file):
        texts = ""
        metadatas = []

        logger.info(f"Loading document from: {file}")
        text = load_document(file)  # Extract text from DOCX
        logger.info(f"❌ Extracted text is of unexpected type: {type(text)}")
        logger.info(f"extracted content: {text}")
        # Debugging: Print extracted text preview
        if not text:
            logger.error("❌ Error: Extracted text is empty.")
            raise ValueError("No valid text found in the document. Ensure the document is readable.")

        if not isinstance(text, str):
            logger.error(f"❌ Extracted text is of unexpected type: {type(text)}")
            raise TypeError("Extracted text is not a string. Possible issue with document parsing.")

        # texts.append(text)
        texts=text
        # logger.info(f"extracted content 1: {texts}")
        metadatas.append({"page_number": 10})
        return texts, metadatas

    def load_and_split_txt(self, content):
        text = content.decode("utf-8")
        lines = text.split('\n')
        texts = [line for line in lines if line.strip()]
        metadatas = [{}] * len(texts)
        return texts, metadatas

    def load_and_split_table(self, content):
        excel_data = pd.read_excel(io.BytesIO(content), sheet_name=None)
        texts = []
        metadatas = []
        for sheet_name, df in excel_data.items():
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            df = df.fillna('N/A')
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Combine key-value pairs into a string
                row_text = ', '.join([f"{key}: {value}" for key, value in row_dict.items()])
                texts.append(row_text)
                metadatas.append({"sheet_name": sheet_name})
        return texts, metadatas

    def load_and_split_csv(self, content):
        csv_data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        texts = []
        metadatas = []
        csv_data = csv_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
        csv_data = csv_data.fillna('N/A')
        for _, row in csv_data.iterrows():
            row_dict = row.to_dict()
            row_text = ', '.join([f"{key}: {value}" for key, value in row_dict.items()])
            texts.append(row_text)
            metadatas.append({"row_index": _})
        return texts, metadatas