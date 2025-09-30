from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class feedback(BaseModel):
    usr_id: Optional[str]
    usr_quest: Optional[str]
    usr_ans: Optional[str]
    session_id: Optional[str]
    is_like: Optional[int]
    sql_query:Optional[str]
    collection_name: Optional[str]
    temp_file_name: Optional[str]


class feedback_response(BaseModel):
    id: int
    # usr_id: str
    # usr_quest: str
    # usr_ans: str
    # session_id: str
    # is_like: int
    inserted_on: datetime


class feedback_update(BaseModel):
    id: int
    is_like: int
