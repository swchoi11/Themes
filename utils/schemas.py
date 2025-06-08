from typing import List, Dict, Optional
from pydantic import BaseModel


class UIElement(BaseModel):
    id: str
    type: str
    bbox: List[float]
    content: str
    confidence: float
    interactivity: bool
    parent_id: str
    children: List[str]
    layout_role: str
    visual_features: Dict


class LayoutElement(BaseModel):
    skeleton: Dict
    layout_regions: Dict
    parsed_regions: Dict
    forms: List[Dict]
    navigation: Dict
    grid_structure: Dict
    interaction_map: Dict
    accessibility: Dict
    statistics: Dict


class Issue(BaseModel):
    filename: Optional[str] = None
    issue_type: str             # EvalKPI.DESCRIPTION.keys()
    component_id: str
    component_type: str
    ui_component_id: str        # EvalKPI.UI_COMPONENT.keys()
    ui_component_type : str     # EvalKPI.UI_COMPONENT.value()
    severity: str
    location_id: str            # EvalKPI.LOCATION.keys()
    location_type: str          # EvalKPI.LOCATION.value()
    bbox: List[float]
    description_id: str         # EvalKPI.DESCRIPTION.keys()
    description_type: str       # EvalKPI.DESCRIPTION.value()
    ai_description: str         # Gemini.response.text()
