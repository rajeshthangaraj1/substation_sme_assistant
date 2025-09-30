# D:\rajesh\python\substation_SME_assistant\config\constants.py

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ========== GEMINI CONFIG ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CHAT_MODEL = "gemini-2.0-flash"
GEMINI_EMBED_MODEL = "models/embedding-001"

# ========== PATH CONFIG ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectordb")
STATIC_TEMP_DIR = os.path.join(BASE_DIR, "file_temp")
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")

# ========== EMBEDDING CONFIG ==========
EMBEDDING_CHUNK_SIZE = 800
EMBEDDING_CHUNK_OVERLAP = 200
