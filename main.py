# D:\rajesh\python\substation_SME_assistant\main.py
import os
import tempfile
import streamlit as st
from config.constants import STATIC_TEMP_DIR
from src.file_handler import FileHandler
from src.manager_agent import ManagerAgent
from src.utils.logger import get_logger


logger = get_logger()
# ✅ Ensure temp dir exists
os.makedirs(STATIC_TEMP_DIR, exist_ok=True)

# ========================
# Streamlit Page Config
# ========================
st.set_page_config(page_title="📄 Substation SME Assistant", layout="wide")
st.title("📄 Substation SME Assistant")
st.write("Upload multiple files and chat with your documents (powered by Gemini).")

# ========================
# Initialize session state
# ========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "collections" not in st.session_state:
    st.session_state["collections"] = []

file_handler = FileHandler()
manager_agent = ManagerAgent()

# ========================
# Display previous chat
# ========================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========================
# File Upload Section
# ========================
col1, col2 = st.columns([0.7, 0.3])

with col1:
    user_input = st.chat_input("Type your question...")

with col2:
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "csv", "xlsx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

# ✅ Handle multiple file uploads
if uploaded_files:
    for uploaded_file in uploaded_files:
        original_filename = uploaded_file.name
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{uploaded_file.name.split('.')[-1]}",
            dir=STATIC_TEMP_DIR,
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        response = file_handler.handle_file_upload(temp_file_path, original_filename)

        if "collection" in response:
            st.session_state["collections"].append(response["collection"])
            st.session_state["messages"].append(
                {"role": "user", "content": f"📎 Uploaded file: **{original_filename}**"}
            )

# ========================
# Handle User Input
# ========================
if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Route through Manager Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = manager_agent.route(user_input)

        st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
