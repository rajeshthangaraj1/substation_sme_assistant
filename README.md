# 📄 Substation SME Assistant

An AI-powered assistant built with **Streamlit**, **Google Gemini 2.0 Flash**, and **FAISS** that lets you **upload multiple files** (PDF, DOCX, TXT, CSV, XLSX, images) and **chat with your documents**.

---

## 🚀 Features
- Multi-file upload with persistent FAISS vector database.
- Document Q&A powered by **Gemini 2.0 Flash**.
- Support for **PDFs, Word docs, images (OCR), spreadsheets, and text files**.
- Automatic **text + image OCR** using `unstructured`, `pytesseract`, and `poppler`.
- Manager Agent architecture → future ready for multiple specialized agents.
- Streamlit chat interface with history.

---

## 📂 Project Structure
substation_SME_assistant/
│ main.py
│ requirements.txt
│ .env
│ .gitignore
│ README.md
│
├── config/
│ └── constants.py
│
├── src/
│ ├── file_handler.py
│ ├── chat_handler.py
│ ├── manager_agent.py
│ └── utils/
│ └── logger.py
│
├── vectordb/ # Auto-created: FAISS DBs
└── file_temp/ # Auto-created: temp uploads

---

## ⚙️ Installation

1. Clone repo:
   ```bash
   git clone https://github.com/your-username/substation_SME_assistant.git
   cd substation_SME_assistant
   
## Create virtual environment:

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac

Install dependencies:
pip install -r requirements.txt

Add .env with your Gemini API key:

GEMINI_API_KEY=your_google_api_key_here

▶️ Run the App
streamlit run main.py