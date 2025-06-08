import re
import cv2
import itertools
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from xml.etree import ElementTree as ET


def get_bounds(bounds_str: str) -> Optional[tuple]:
    """바운딩 박스 문자열을 파싱하여 (x1, y1, x2, y2) 튜플 반환."""
    match = re.search(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
    if not match:
        return None
    return tuple(map(int, match.groups()))

@dataclass
class UIComponent:
    id: str
    type: str
    bbox: List[float]
    content: str
    content_desc: str
    resource_id: str
    parent_id: Optional[str]
    children: List[str]
    clickable: bool
    visual_features: Dict
    ocr_matched: bool = False

    def __post_init__(self):
        self.visual_features = self.visual_features or {}

class XMLParser:
    """안드로이드 XML 파일을 파싱하여 UI 컴포넌트를 추출"""

    def __init__(self, image_path: str = None, xml_path: str = None):
        self.image_path = image_path
        self.xml_path = xml_path
        self.config = {
            'text_classes': ['android.widget.TextView', 'android.widget.EditText'],
            'button_classes': ['android.widget.Button', 'android.widget.ImageButton',
                               'android.widget.RadioButton', 'android.widget.CheckBox'],
            'icon_size_range': (20, 100),
            'max_size_ratio': 0.5,
            'iou_threshold': 0.3,
            'layout_classes': ['FrameLayout', 'LinearLayout', 'RelativeLayout', 'ConstraintLayout']
        }

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
            self.image_width, self.image_height = self.image.shape[:2]
        except Exception as e:
            print(f"Image loading failed: {self.image_path}, {e}")
            raise

    def _parse_xml(self) -> List[UIComponent]:
        """XML 파일을 파싱하여 UIComponent 리스트 반환."""
        components = []
        node_counter = itertools.count()

        def extract_node(node: ET.Element, parent_id: Optional[str] = None) -> Optional[str]:
            idx = next(node_counter)
            bounds = get_bounds(node.get('bounds', '[0,0][0,0]'))
            if not bounds:
                print(f"Invalid bounds for node: {node.get('resource-id', f'node_{idx}')}")
                return None

            x1, y1, x2, y2 = bounds
            if x1 >= x2 or y1 >= y2:
                print(f"Invalid bbox dimensions for node: {node.get('resource-id', f'node_{idx}')}")
                return None

            bbox = [x1 / self.image_width, y1 / self.image_height, x2 / self.image_width, y2 / self.image_height]

            comp = UIComponent(
                id=node.get('resource-id', f'node_{idx}'),
                type=node.get('class', 'Unknown').split('.')[-1],
                bbox=bbox,
                content=node.get('text', ''),
                content_desc=node.get('content-desc', ''),
                resource_id=node.get('resource-id', ''),
                parent_id=parent_id,
                children=[],
                clickable=node.get('clickable', 'false').lower() == 'true',
                visual_features={}
            )

            child_ids = []
            for child in node.findall('node'):
                child_id = extract_node(child, comp.id)
                if child_id:
                    child_ids.append(child_id)

            # 필터링 조건 완화: 유효한 bbox와 type만 있으면 포함
            comp.children = child_ids
            components.append(comp)
            return comp.id

        try:
            extract_node(self.root)
        except Exception as e:
            print(f"Error parsing XML: {e}")
            raise ValueError(f"Failed to parse XML: {str(e)}")

        if not components:
            print("No valid components found in XML")
        else:
            print(f"Parsed {len(components)} components from XML")
        return components

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
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

    def _find_best_ocr_match(self, xml_elem: UIComponent, other_elements: List[Any],
                             element_type: str = 'dict') -> Optional[Any]:
        best_match, best_iou = None, 0.0
        for elem in other_elements:
            try:
                if element_type == 'UIElement':
                    bbox = elem.bbox if hasattr(elem, 'bbox') else None
                else:
                    bbox = elem.get('bbox') if isinstance(elem, dict) else None

                if bbox is None:
                    continue

                iou = self._calculate_iou(xml_elem.bbox, bbox)
                if iou > self.config['iou_threshold'] and iou > best_iou:
                    best_match, best_iou = elem, iou

            except Exception as e:
                print(f"Error in IoU calculation: {e}")
                continue
        return best_match

    def merge_with_elements(self, xml_components: List[UIComponent], other_elements: List[Any],
                            element_type: str = 'dict') -> List[UIComponent]:
        merged_components = xml_components.copy()
        merge_count = 0

        for xml_elem in merged_components:
            best_match = self._find_best_ocr_match(xml_elem, other_elements, element_type)
            if best_match:
                try:
                    if element_type == 'UIElement':
                        new_content = getattr(best_match, 'content', '') or xml_elem.content
                        visual_features = getattr(best_match, 'visual_features', {})
                    else:  # dict 타입
                        new_content = best_match.get('content', '') or xml_elem.content
                        visual_features = best_match.get('visual_features', {})

                    if not xml_elem.content.strip() or (new_content and len(new_content) > len(xml_elem.content)):
                        xml_elem.content = new_content

                    if visual_features:
                        xml_elem.visual_features.update(visual_features)

                    xml_elem.ocr_matched = True
                    merge_count += 1

                except Exception as e:
                    print(f"Error merging element {xml_elem.id}: {e}")
                    continue

        print(f"Successfully merged {merge_count}/{len(merged_components)} components")
        return merged_components

    def merge_xml_json_data(self, xml_elements: List[UIComponent], json_layout: Dict) -> Dict:
        """XML 요소와 JSON 레이아웃 데이터를 병합하여 LayoutElement 반환"""
        if not isinstance(json_layout, dict):
            print("json_layout must be a dictionary")
            raise ValueError("json_layout must be a dictionary")

        json_elements = json_layout.get('skeleton', {}).get('elements', [])
        if not json_layout.get('skeleton') or not json_layout['skeleton'].get('elements'):
            print("Invalid json_layout structure: 'skeleton' or 'elements' missing")
            raise ValueError("Invalid json_layout structure: 'skeleton' or 'elements' missing")

        merged_components = self.merge_with_elements(xml_elements, json_elements, 'dict')

        merged_layout = json_layout.copy()
        merged_layout['skeleton'] = merged_layout.get('skeleton', {})
        merged_layout['skeleton']['elements'] = [asdict(comp) for comp in merged_components]
        merged_layout['skeleton']['metadata'] = merged_layout['skeleton'].get('metadata', {})
        merged_layout['skeleton']['metadata'].update({
            'source': 'xml_json_merged',
            'total_elements': len(merged_components),
            'image_size': [self.image_width, self.image_height]
        })

        merged_layout['statistics'] = merged_layout.get('statistics', {})
        merged_layout['statistics']['merged_elements'] = len(merged_components)
        merged_layout['statistics']['xml_elements'] = len(xml_elements)
        merged_layout['statistics']['json_elements'] = len(json_elements)

        return merged_layout

    def _all_nodes(self, node: ET.Element) -> List[ET.Element]:
        """모든 XML 노드를 재귀적으로 수집."""
        nodes = [node]
        for child in node.findall('node'):
            nodes.extend(self._all_nodes(child))
        return nodes

    def get_components(self, class_names: Optional[List[str]] = None) -> List[UIComponent]:
        """지정된 클래스 이름에 해당하는 컴포넌트 추출."""
        class_names = class_names or (self.config['text_classes'] + self.config['button_classes'] + self.config['layout_classes'])
        components = self._parse_xml()
        return self._apply_filters(components, class_names)

    def _apply_filters(self, components: List[UIComponent], class_names: List[str]) -> List[UIComponent]:
        """크기, 포함 관계, 텍스트 콘텐츠 필터링 적용."""
        max_width = self.image_width * self.config['max_size_ratio']
        max_height = self.image_height * self.config['max_size_ratio']

        valid_components = [
            comp for comp in components
            if comp.type in class_names and
               abs(comp.bbox[2] * self.image_width - comp.bbox[0] * self.image_width) < max_width and
               abs(comp.bbox[3] * self.image_height - comp.bbox[1] * self.image_height) < max_height
        ]

        final_components = [
            comp for i, comp in enumerate(valid_components)
            if not any(self._is_contained(comp.bbox, comp_b.bbox) for j, comp_b in enumerate(valid_components) if i != j)
        ]

        filtered_components = [
            comp for comp in final_components
            if comp.type not in self.config['text_classes'] or (comp.content and comp.content.strip())
        ]

        print(f"Filtered {len(filtered_components)} components after applying filters")
        return filtered_components

    def _is_contained(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        """컴포넌트 간 포함 관계 확인."""
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_bbox
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_bbox
        return (inner_x1 >= outer_x1 and inner_y1 >= outer_y1 and
                inner_x2 <= outer_x2 and inner_y2 <= outer_y2)

    def get_icon_components(self) -> List[UIComponent]:
        """아이콘 컴포넌트 추출."""
        components = self._parse_xml()
        min_size, max_size = self.config['icon_size_range']
        icon_components = []

        for comp in components:
            width = abs(comp.bbox[2] - comp.bbox[0]) * self.image_width
            height = abs(comp.bbox[3] - comp.bbox[1]) * self.image_height
            if min_size <= width <= max_size and min_size <= height <= max_size:
                icon_components.append(comp)

        unique_components = []
        seen_bounds = set()
        for comp in icon_components:
            bbox_tuple = tuple(comp.bbox)
            if bbox_tuple not in seen_bounds:
                unique_components.append(comp)
                seen_bounds.add(bbox_tuple)

        print(f"Extracted {len(unique_components)} icon components")
        return unique_components

    def get_all_classes(self) -> List[str]:
        """모든 노드의 클래스 목록 반환."""
        all_nodes = self._all_nodes(self.root)
        return list(set(node.get('class') for node in all_nodes if node.get('class')))