from dataclasses import dataclass
from typing import Optional

@dataclass
class ResultModel:
    image_path: str = ""
    issue_type: str = ""
    issue_location: list[int] = None
    issue_description: str = ""

    def __post_init__(self):
        if self.issue_location is None:
            self.issue_location = []

    