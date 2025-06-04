from typing import Optional
from pydantic import BaseModel

class ResultModel(BaseModel):
    image_path: str = ""
    index: int = 0
    issue_type: str = ""
    issue_location: list[int] = []
    issue_description: str = ""

class Result(BaseModel):
    issue_location: list[int] = []
    issue_description: str = ""
    