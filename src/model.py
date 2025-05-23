from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class UIElement:
    """UI 요소 정보를 담는 데이터 클래스"""
    id: str
    type: str
    bbox: List[float]
    content: Optional[str] = None
    confidence: float = 0.0
    interactivity: bool = False
    parent_id: Optional[str] = None
    children: List[str] = None
    layout_role: Optional[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class Skeleton:
    structure_type: str
    elements: List[UIElement]
    hierarchy: Dict[str, List[str]]

@dataclass
class Element:
    elements: List[UIElement]
    bbox: List[float]

@dataclass
class LayoutRegions:
    header: Dict[str, Element]
    navigation: Dict[str, Element]
    sidebar: Dict[str, Element]
    content: Dict[str, Element]
    footer: Dict[str, Element]

@dataclass
class OCR:
    text: List[str]
    bbox: List[List[float]]

@dataclass
class ParsedElement:
    cropped_ocr: OCR
    elements_count: str
    elements_types: Dict[str, int]

@dataclass
class ParsedRegions:
    header: ParsedElement
    content: ParsedElement
    footer: ParsedElement

@dataclass
class Form:
    inputs: List[UIElement]
    submit_button: UIElement

@dataclass
class Layout:
    """레이아웃 정보를 담는 데이터 클래스"""
    skeleton: Skeleton
    layout_regions: LayoutRegions
    parsed_regions: ParsedRegions
    forms: Form
    navigation: Dict
