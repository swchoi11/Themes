from pydantic import BaseModel
from typing import List

class Issue(BaseModel):
    """검출된 이슈"""
    issue_type: int
    component_id : str
    component_type: str
    severity: float
    bbox: List[float]
    description: str