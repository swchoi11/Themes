import re
import cv2
import itertools
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from xml.etree import ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

from common.eval_kpi import EvalKPI

UI_COMPONENT = EvalKPI.UI_COMPONENT

def get_bounds(bounds_str: str) -> Optional[tuple]:
    """바운딩 박스 문자열을 파싱하여 (x1, y1, x2, y2) 튜플 반환."""
    match = re.search(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
    if not match:
        raise ValueError(f"Invalid bounds format: {bounds_str}")
    return tuple(map(int, match.groups()))

@dataclass
class UIComponent:
    id: str
    type: str
    bbox: List[float]
    content: str
    content_desc: str
    parent_id: Optional[str]
    children: List[str]
    interactivity: bool
    visual_features: Dict
    ocr_matched: bool = False

    def __post_init__(self):
        self.visual_features = self.visual_features or {}


class XMLParser:
    """안드로이드 XML 파일을 파싱하여 UI 컴포넌트를 추출"""

    def __init__(self, image_path: str = None, xml_path: str = None):
        self.image_path = Path(image_path) if image_path else None
        self.xml_path = Path(xml_path) if xml_path else None

        self.config = {
            'text_classes': ['TextView', 'EditText'],
            'button_classes': ['Button', 'ImageButton','RadioButton', 'CheckBox','Switch', 'ToggleButton', 'Spinner'],
            'etc_classes': ['ImageView','SeekBar', 'ProgressBar'],
            'icon_size_range': (20, 100),
            'max_size_ratio': 0.5,
            'iou_threshold': 0.3,
            'similarity_threshold': 0.5
        }

        self.component_counters = {component_type: 0 for component_type in UI_COMPONENT.values()}

        try:
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot().find('node')
            if self.root is None:
                raise ValueError("No root node found in XML")
        except FileNotFoundError:
            print(f"XML file not found: {self.xml_path}")
            raise
        except ET.ParseError:
            print(f"Invalid XML format: {self.xml_path}")
            raise

        try:
            self.image = cv2.imread(str(self.image_path))
            if self.image is None:
                raise ValueError("Failed to load image")
            self.image_height, self.image_width  = self.image.shape[:2]
        except Exception as e:
            print(f"Image loading failed: {self.image_path}, {e}")
            raise

    def check_bounds_validity(self):
        all_nodes = self.tree.getroot().findall('.//node')
        invalid_bounds = []
        for node in all_nodes:
            bounds_str = node.get('bounds', '[0,0][0,0]')
            try:
                bounds = get_bounds(bounds_str)
                if not self._is_valid_bounds(bounds):
                    invalid_bounds.append((bounds_str, node.get('class')))
            except ValueError as e:
                invalid_bounds.append((bounds_str, node.get('class'), str(e)))
        print(f"Invalid bounds found: {len(invalid_bounds)}")
        for bounds_info in invalid_bounds:
            print(f"Invalid: {bounds_info}")
        return invalid_bounds

    def check_xml_structure(self):
        try:
            all_nodes = self.tree.getroot().findall('.//node')
            print(f"Total nodes in XML: {len(all_nodes)}")
            for i, node in enumerate(all_nodes):
                print(f"Node {i}: class={node.get('class')}, bounds={node.get('bounds')}, text={node.get('text')}, clickable={node.get('clickable')}")
            return len(all_nodes)
        except Exception as e:
            print(f"Error checking XML structure: {e}")
            return 0

    def _get_similar_ui_component(self, class_name: str) -> str:
        if not class_name:
            return 'Unknown'
        class_name = class_name.split('.')[-1].lower()
        class_mapping = {
            'textview': 'TextView',
            'edittext': 'EditText',
            'button': 'Button',
            'imagebutton': 'ImageButton',
            'imageview': 'ImageView',
            'radiobutton': 'RadioButton',
            'checkbox': 'CheckBox',
            'switch': 'Switch',
            'togglebutton': 'ToggleButton',
            'spinner': 'Spinner',
            'seekbar': 'SeekBar',
            'progressbar': 'ProgressBar',
            'framelayout': 'FrameLayout',
            'linearlayout': 'LinearLayout',
            'relativelayout': 'RelativeLayout',
            'viewgroup': 'ViewGroup',
            'scrollview': 'ScrollView',
            'viewpager': 'ViewPager',
            'view': 'View'
        }
        component_type = class_mapping.get(class_name, 'Unknown')
        # print(f"Class {class_name} mapped to: {component_type}")
        return component_type

    def _get_next_component_id(self, component_type: str) -> str:
        """컴포넌트 타입에 따른 고유 ID 생성"""
        if component_type not in self.component_counters:
            self.component_counters[component_type] = 0

        self.component_counters[component_type] += 1
        return f'{component_type}_{self.component_counters[component_type]}'

    def _normalize_bbox(self, bounds: Tuple[int, int, int, int]) -> List[float]:
        # print(self.image_width, self.image_height)
        x1, y1, x2, y2 = bounds
        x1 = min(max(x1, 0), self.image_width)
        y1 = min(max(y1, 0), self.image_height)
        x2 = min(max(x2, 0), self.image_width)
        y2 = min(max(y2, 0), self.image_height)
        normalized = [
            x1 / self.image_width,
            y1 / self.image_height,
            x2 / self.image_width,
            y2 / self.image_height
        ]
        return normalized

    def _is_valid_bounds(self, bounds: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bounds
        return x1 < x2 and y1 < y2

    def _extract_content(self, node: ET.Element) -> str:
        """노드에서 콘텐츠 추출 (text/resource-id)"""
        if text := node.get('text', '').strip():
            return text
        if resource_id := node.get('resource-id', ''):
            return resource_id.rsplit('/', 1)[-1]
        return ''

    def _parse_xml_recursive(self, node: ET.Element, parent_id: Optional[str] = None) -> Optional[UIComponent]:
        bounds_str = node.get('bounds', '[0,0][0,0]')
        try:
            bounds = get_bounds(bounds_str)
        except ValueError as e:
            print(f"Invalid bounds format: {bounds_str}, Error: {e}")
            return None
        if not self._is_valid_bounds(bounds):
            print(f"Invalid bounds coordinates: {bounds}")
            return None

        class_name = node.get('class', '').split('.')[-1]
        component_type = self._get_similar_ui_component(class_name)
        # print(f"Node class: {class_name}")
        # print(f"Matched component type: {component_type}")
        component_id = self._get_next_component_id(component_type)

        component = UIComponent(
            id=component_id,
            type=component_type,
            bbox=self._normalize_bbox(bounds),
            content=self._extract_content(node),
            content_desc=node.get('content-desc', ''),
            parent_id=parent_id,
            children=[],
            interactivity=node.get('clickable', 'false').lower() == 'true',
            visual_features={}
        )
        # print(f"Created component: {component.id}, type: {component.type}, bbox: {component.bbox}")

        child_ids = []
        child_nodes = node.findall('node')
        # print(f"Found {len(child_nodes)} child nodes")
        for child_node in child_nodes:
            child_component = self._parse_xml_recursive(child_node, component_id)
            if child_component:
                child_ids.append(child_component.id)
                # print(f"Added child: {child_component.id}")

        component.children = child_ids
        # print(f"Component {component.id} has {len(child_ids)} children")
        return component

    def _parse_xml(self) -> List[UIComponent]:
        components = []

        def collect_components(component: UIComponent):
            if component:
                components.append(component)
                # print( f"Collected component: {component.id}, type: {component.type}, children: {component.children}, bbox: {component.bbox}")
                for child_id in component.children:
                    child_component = next((c for c in components if c.id == child_id), None)
                    if child_component:
                        collect_components(child_component)
                    # else:
                    #     print(f"Child {child_id} not found in components")

        try:
            print("Starting XML parsing...")
            all_nodes = self.tree.getroot().findall('.//node')
            print(f"Total nodes found: {len(all_nodes)}")
            for node in all_nodes:
                component = self._parse_xml_recursive(node, None)
                if component:
                    collect_components(component)
            print(f"Parsed {len(components)} components from XML: {[comp.__dict__ for comp in components]}")
            return components
        except Exception as e:
            print(f"Failed to parse XML: {str(e)}")
            raise ValueError(f"Failed to parse XML: {str(e)}")

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """두 바운딩 박스의 IoU(Intersection over Union) 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _find_best_match(self, xml_component: UIComponent, elements: List[Any],
                         element_type: str = 'dict') -> Optional[Any]:
        """XML 컴포넌트와 가장 잘 매칭되는 요소 찾기"""
        best_match, best_iou = None, 0.0

        for element in elements:
            try:
                bbox = self._extract_bbox(element, element_type)
                if bbox is None:
                    continue

                iou = self._calculate_iou(xml_component.bbox, bbox)
                if iou > self.config['iou_threshold'] and iou > best_iou:
                    best_match, best_iou = element, iou

            except Exception as e:
                print(f"Error in IoU calculation: {e}")
                continue

        return best_match

    def _extract_bbox(self, element: Any, element_type: str) -> Optional[List[float]]:
        """요소에서 바운딩 박스 추출"""
        if element_type == 'UIElement':
            return getattr(element, 'bbox', None)
        elif element_type == 'dict' and isinstance(element, dict):
            return element.get('bbox')
        return None

    def _apply_size_filter(self, components: List[UIComponent]) -> List[UIComponent]:
        """크기 기반 필터링 적용"""
        max_width = self.image_width * self.config['max_size_ratio']
        max_height = self.image_height * self.config['max_size_ratio']

        return [
            comp for comp in components
            if (abs(comp.bbox[2] - comp.bbox[0]) * self.image_width < max_width and
                abs(comp.bbox[3] - comp.bbox[1]) * self.image_height < max_height)
        ]

    def _apply_containment_filter(self, components: List[UIComponent]) -> List[UIComponent]:
        """포함 관계 기반 필터링 적용"""
        return [
            comp for i, comp in enumerate(components)
            if not any(
                self._is_contained(comp.bbox, other_comp.bbox)
                for j, other_comp in enumerate(components)
                if i != j
            )
        ]

    def _apply_content_filter(self, components: List[UIComponent]) -> List[UIComponent]:
        """텍스트 콘텐츠 기반 필터링 적용"""
        return [
            comp for comp in components
            if (comp.type not in self.config['text_classes'] or
                (comp.content and comp.content.strip()))
        ]

    def _is_contained(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        """내부 바운딩 박스가 외부 바운딩 박스에 포함되는지 확인"""
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_bbox
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_bbox

        return (inner_x1 >= outer_x1 and inner_y1 >= outer_y1 and
                inner_x2 <= outer_x2 and inner_y2 <= outer_y2)

    def get_components(self, class_names: Optional[List[str]] = None) -> List[UIComponent]:
        """지정된 클래스 이름에 해당하는 컴포넌트 추출"""
        if class_names is None:
            class_names = (self.config['text_classes'] + self.config['button_classes'] + self.config['etc_classes'])
        components = self._parse_xml()

        filtered_components = [comp for comp in components if comp.type in class_names]
        print(f"Initial components: {len(components)}, Details: {[comp.__dict__ for comp in components]}")
        return filtered_components

    def get_icon_components(self) -> List[UIComponent]:
        """아이콘 크기에 해당하는 컴포넌트 추출"""
        components = self._parse_xml()
        min_size, max_size = self.config['icon_size_range']

        icon_components = [
            comp for comp in components
            if (min_size <= (abs(comp.bbox[2] - comp.bbox[0]) * self.image_width) <= max_size and
                min_size <= (abs(comp.bbox[3] - comp.bbox[1]) * self.image_height) <= max_size)
        ]

        icon_components = self._apply_containment_filter(icon_components)
        return icon_components

    def merge_with_elements(self, xml_components: List[UIComponent], other_elements: List[Any],
                            element_type: str = 'dict') -> List[UIComponent]:
        """XML 컴포넌트와 다른 요소들을 병합"""
        merged_components = xml_components.copy()
        merge_count = 0

        for xml_comp in merged_components:
            best_match = self._find_best_match(xml_comp, other_elements, element_type)
            if not best_match:
                continue

            try:
                # 콘텐츠 및 시각적 특징 업데이트
                self._update_component_from_match(xml_comp, best_match, element_type)
                xml_comp.ocr_matched = True
                merge_count += 1

            except Exception as e:
                print(f"Error merging element {xml_comp.id}: {e}")

        print(f"Successfully merged {merge_count}/{len(merged_components)} components")
        return merged_components

    def _update_component_from_match(self, component: UIComponent, match: Any, element_type: str):
        """매칭된 요소로부터 컴포넌트 정보 업데이트"""
        if element_type == 'UIElement':
            new_content = getattr(match, 'content', '') or component.content
            visual_features = getattr(match, 'visual_features', {})
        else:  # dict 타입
            new_content = match.get('content', '') or component.content
            visual_features = match.get('visual_features', {})

        # 더 나은 콘텐츠로 업데이트
        if (not component.content.strip() or
                (new_content and len(new_content) > len(component.content))):
            component.content = new_content

        # 시각적 특징 업데이트
        if visual_features:
            component.visual_features.update(visual_features)

    def merge_xml_json_data(self, xml_elements: List[UIComponent], json_layout: Dict) -> Dict:
        """XML 요소와 JSON 레이아웃 데이터를 병합"""
        if not isinstance(json_layout, dict):
            raise ValueError("json_layout must be a dictionary")

        skeleton = json_layout.get('skeleton', {})
        json_elements = skeleton.get('elements', [])

        if not skeleton or not json_elements:
            raise ValueError("Invalid json_layout structure: 'skeleton' or 'elements' missing")

        merged_components = self.merge_with_elements(xml_elements, json_elements, 'dict')

        # 병합된 레이아웃 생성
        merged_layout = json_layout.copy()
        merged_layout['skeleton'] = {
            **skeleton,
            'elements': [asdict(comp) for comp in merged_components],
            'metadata': {
                **skeleton.get('metadata', {}),
                'source': 'xml_json_merged',
                'total_elements': len(merged_components),
                'image_size': [self.image_width, self.image_height]
            }
        }

        merged_layout['statistics'] = {
            **merged_layout.get('statistics', {}),
            'merged_elements': len(merged_components),
            'xml_elements': len(xml_elements),
            'json_elements': len(json_elements)
        }

        return merged_layout

    def get_all_classes(self) -> List[str]:
        """XML에서 모든 클래스 이름 추출"""
        try:
            all_nodes = self.tree.getroot().findall('.//node')
            classes = {
                node.get('class', '').split('.')[-1]
                for node in all_nodes
                if node.get('class')
            }
            return sorted(list(classes))

        except Exception as e:
            print(f"Error extracting classes: {e}")
            return []

    def get_all_context(self) -> List[str]:
        """모든 노드의 resource-id에서 컨텍스트 부분 추출"""
        try:
            all_nodes = self.tree.getroot().findall('.//node')
            contexts = set()

            for node in all_nodes:
                resource_id = node.get('resource-id', '')
                if '/' in resource_id:
                    try:
                        context = resource_id.split('/', 1)[1]
                        contexts.add(context)
                    except (ValueError, IndexError):
                        continue

            return sorted(list(contexts))

        except Exception as e:
            print(f"Error extracting contexts: {e}")
            return []