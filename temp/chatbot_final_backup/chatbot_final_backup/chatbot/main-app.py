import uvicorn as uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from logger_config import setup_logger
from services.summarizeService import SummarizeService
from services.feedbackService import FeedbackService
from services.businessAnsService import BusinessAnsService
from typing import Optional
from file_handler import FileHandler
from fastapi.responses import JSONResponse
from typing import Union
from config.constant import (
    VECTOR_DB_PATH,
    STATIC_TEMP_DIR,
    EMBEDDING_MODEL_PATH,
    LONG_TERM_MODEL_PATH,
    SHORT_TERM_MODEL_PATH,
    CREW_MODEL_NAME,
    CREW_OLLAMA_SERVER_URL,
    CREW_TEMPERATURE
)
from model.feedback import feedback, feedback_response, feedback_update
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = setup_logger()

#
# def get_logging_service():
#     return LoggingService(logger)
#
#
# logging_serv = get_logging_service()

# log_now = datetime.now()

session_state = {
    "collection": None,
    "uploaded_file_path":None,
    "messages":[],
    "last_response":None
}

class ChatResponse(BaseModel):
    response: str


@app.post("/askend/chat_sql")
async def ask_end_sql(message: Optional[str] = Form(None),
                  user_id: Optional[str] = Form(None), session_id: Optional[str] = Form(None)):
    logger.info(message)
    logger.info(user_id)
    response_content = {}

    if message:
        session_state["messages"].append({
            "role":"user",
            "content": message
        })
        businessAnsService = BusinessAnsService(logger)
        context, feedback_result = businessAnsService.submit(message, user_id, session_id)
        # Replace this with actual NLP or summary logic
        response_content["context"] = context
        response_content["reply"] = f"Message received: {message}"
        response_content["feedback_result"] = feedback_result
    return response_content



@app.post("/askend/chat/")
async def ask_end(message: Optional[str] = Form(None), file: Optional[Union[UploadFile, str]] = File(None),
                  user_id: Optional[str] = Form(None), session_id: Optional[str] = Form(None)):
    logger.info(message)
    logger.info(file)
    logger.info(user_id)
    response_content = {}
    if isinstance(file, str):
        file = None

    if file:
        original_filename = file.filename
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False,suffix=f".{file.filename.split('.')[-1]}", dir=STATIC_TEMP_DIR) as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        file_handler_method = FileHandler(VECTOR_DB_PATH)
        response = file_handler_method.handle_file_upload(temp_file_path, original_filename)
        logger.info(f"response {response}")
        session_state["collection"] = response.get("collection")
        session_state["uploaded_file_path"] = temp_file_path
        session_state["messages"].append({
            "role":'user',
            "content":f"Uploaded file: **{file.filename}**"
        })
        response_content["file_upload"] = f"{file.filename} uploaded successfully"

    if message:
        session_state["messages"].append({
            "role":"user",
            "content": message
        })
        summary_service = SummarizeService(logger)
        logger.info(f"session_state {session_state}")
        context, feedback_id = summary_service.buildContext(message, user_id, session_id, session_state)
        # Replace this with actual NLP or summary logic
        response_content["context"] = context
        response_content["reply"] = f"Message received: {message}"
        response_content["feedback_result"] = feedback_id

    if not message and not file:
        return JSONResponse(content={"error": "Please provide a message or a file."}, status_code=400)

    return response_content


@app.post("/askend/feedback/", response_model=feedback_response)
async def feedback(request: feedback):
    feedback_service = FeedbackService(logger)
    result = feedback_service.submit(request)
    return result


@app.put("/askend/feedback_update/")
async def update_feedback(request: feedback_update):
    feedback_service = FeedbackService(logger)
    result = feedback_service.update(request)
    return result


# @fastapi_app.on_event("shutdown")
# async def shutdown_event():
#     shutdown_logging()


#if __name__ == "__main__":
    #uvicorn.run(app, host="127.0.0.1", port=8081)
