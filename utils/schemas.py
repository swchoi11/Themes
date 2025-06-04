from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class UIElement(BaseModel):
    """UI 요소 정보를 담는 데이터 클래스"""
    id: str
    type: str
    bbox: List[float]
    content: Optional[str]
    confidence: float
    interactivity: bool = False
    parent_id: str
    children: List[str] = Field(default_factory=list)
    layout_role: str
    visual_features: Dict


class Skeleton(BaseModel):
    structure_type: str
    elements: List[UIElement]
    hierarchy: Dict[str, List[str]]


class Region(BaseModel):
    elements: List[UIElement]
    bbox: Optional[List[float]] = None


class LayoutRegions(BaseModel):
    header: Region
    navigation: Region
    sidebar: Region
    content: Region
    footer: Region


class OCR(BaseModel):
    text: List[str]
    bbox: List[List[float]]


class ElementTypes(BaseModel):
    text: int
    button: int
    input: Optional[int] = None


class ParsedElement(BaseModel):
    cropped_ocr: OCR
    elements_count: int
    element_types: ElementTypes


class ParsedRegions(BaseModel):
    header: ParsedElement
    content: ParsedElement
    footer: ParsedElement


class Form(BaseModel):
    inputs: List[UIElement]
    submit_button: UIElement


class Layout(BaseModel):
    """레이아웃 정보를 담는 데이터 클래스"""
    skeleton: Skeleton
    layout_regions: LayoutRegions
    parsed_regions: ParsedRegions
    forms: List[Form]
    navigation: Dict
    grid_structure: Dict
    interaction_map: Dict
    accessibility: Dict
    statistics: Dict


class Issue(BaseModel):
    """검출된 이슈"""
    issue_type: str
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
    suggestion: str
