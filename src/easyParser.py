"""
Extracts Skeleton UI Layout from Samsung Themes using GUIParser
"""

import cv2
import numpy as np
import torch
from PIL import Image

import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict


from utils.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
from utils.box_annotator import BoxAnnotator
import supervision as sv
import torchvision.transforms as T


@dataclass
class UIElement:
    """UI 요소 정보를 담는 데이터 클래스"""
    id: str
    type: str  # 'text', 'button', 'input', 'icon', 'image', 'container'
    bbox: List[float]  # [x1, y1, x2, y2] normalized
    content: Optional[str] = None
    confidence: float = 0.0
    interactivity: bool = False
    parent_id: Optional[str] = None
    children: List[str] = None
    layout_role: Optional[str] = None  # 'header', 'navigation', 'content', 'sidebar', 'footer'

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class LayoutStructure:
    """레이아웃 구조 정보"""
    structure_type: str  # 'linear', 'grid', 'nested', 'tabs'
    elements: List[UIElement]
    hierarchy: Dict[str, List[str]]  # parent_id -> children_ids
    layout_regions: Dict[str, List[UIElement]]  # region_name -> elements


class SkeletonUIExtractor:
    """스켈레톤 UI 구조 추출기"""

    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 모델 초기화
        self.som_model = get_yolo_model(model_path=config.get('som_model_path', 'weights/icon_detect/weights.pt'))
        self.caption_model_processor = get_caption_model_processor(
            model_name=config.get('caption_model_name', 'florence2'),
            model_name_or_path=config.get('caption_model_path', 'weights/icon_caption_florence'),
            device=device
        )

        # 레이아웃 분석 매개변수
        self.layout_detector = LayoutDetector()
        self.hierarchy_builder = HierarchyBuilder()

        print("SkeletonUIExtractor 초기화 완료!")

    def extract_skeleton(self, image: Union[str, Image.Image, np.ndarray]) -> LayoutStructure:
        """스켈레톤 UI 구조 추출"""
        # 1. 이미지 전처리
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        w, h = image.size

        # 2. OCR로 텍스트 요소 추출
        ocr_result, _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'text_threshold': 0.8, 'paragraph': False},
            use_paddleocr=False
        )
        text_list, ocr_bbox = ocr_result

        # 3. YOLO로 아이콘/버튼 검출
        ui_elements = self._detect_ui_elements(image, ocr_bbox, text_list, w, h)

        # 4. 레이아웃 영역 분할
        layout_regions = self.layout_detector.detect_layout_regions(image, ui_elements)

        # 5. 계층 구조 분석
        hierarchy = self.hierarchy_builder.build_hierarchy(ui_elements, layout_regions)

        # 6. 스켈레톤 구조 반환
        return LayoutStructure(
            structure_type=self._determine_structure_type(layout_regions),
            elements=ui_elements,
            hierarchy=hierarchy,
            layout_regions=layout_regions
        )

    def _detect_ui_elements(self, image: Image.Image, ocr_bbox: List, text_list: List, w: int, h: int) -> List[
        UIElement]:
        """UI 요소 검출"""
        elements = []

        # OCR 텍스트 요소 추가
        for i, (bbox, text) in enumerate(zip(ocr_bbox, text_list)):
            x1, y1, x2, y2 = bbox
            elements.append(UIElement(
                id=f"text_{i}",
                type="text",
                bbox=[x1 / w, y1 / h, x2 / w, y2 / h],
                content=text,
                confidence=0.95,
                interactivity=False,
                layout_role="content"
            ))

        # YOLO 검출
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # SOM 라벨링 수행
        _, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            self.som_model,
            BOX_TRESHOLD=self.config.get('BOX_TRESHOLD', 0.05),
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text_list,
            use_local_semantics=True,
            iou_threshold=self.config.get('iou_threshold', 0.7),
            batch_size=128
        )

        # 파싱된 내용을 UIElement로 변환
        for i, parsed_item in enumerate(parsed_content_list):
            bbox = parsed_item['bbox']
            element_type = self._classify_element_type(parsed_item)

            elements.append(UIElement(
                id=f"{element_type}_{i}",
                type=element_type,
                bbox=bbox,
                content=parsed_item.get('content'),
                confidence=0.8,
                interactivity=parsed_item.get('interactivity', False),
                layout_role=self._determine_layout_role(bbox, elements)
            ))

        return elements

    def _classify_element_type(self, parsed_item: Dict) -> str:
        """파싱된 아이템의 타입 분류"""
        if parsed_item.get('type') == 'text':
            return 'text'
        elif parsed_item.get('interactivity'):
            # 크기와 내용으로 버튼/입력 필드 구분
            bbox = parsed_item['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # 가로가 세로보다 훨씬 긴 경우 입력 필드
            if width / height > 4:
                return 'input'
            else:
                return 'button'
        else:
            return 'icon'

    def _determine_layout_role(self, bbox: List[float], existing_elements: List[UIElement]) -> str:
        """요소의 레이아웃 역할 결정"""
        x1, y1, x2, y2 = bbox

        # 화면 상단 (0.1 미만)에 있으면 헤더
        if y1 < 0.1:
            return 'header'
        # 화면 하단 (0.9 초과)에 있으면 푸터
        elif y1 > 0.9:
            return 'footer'
        # 화면 좌측 (0.1 미만)에 있고 세로가 길면 사이드바
        elif x1 < 0.1 and (y2 - y1) > 0.3:
            return 'sidebar'
        # 화면 상단 근처 (0.1~0.2)에 있고 가로가 길면 네비게이션
        elif 0.1 < y1 < 0.2 and (x2 - x1) > 0.5:
            return 'navigation'
        else:
            return 'content'

    def _determine_structure_type(self, layout_regions: Dict) -> str:
        """레이아웃 구조 타입 결정"""
        # 레이아웃 영역 분석으로 구조 타입 결정
        if 'sidebar' in layout_regions and 'content' in layout_regions:
            return 'sidebar_layout'
        elif 'header' in layout_regions and 'footer' in layout_regions:
            return 'header_footer_layout'
        elif len(layout_regions.get('navigation', {}).get('elements', [])) > 0:
            return 'navigation_layout'
        else:
            return 'single_column'


class LayoutDetector:
    """레이아웃 영역 검출기"""

    def detect_layout_regions(self, image: Image.Image, elements: List[UIElement]) -> Dict[str, Dict]:
        """레이아웃 영역 검출"""
        regions = {
            'header': {'elements': [], 'bbox': None},
            'navigation': {'elements': [], 'bbox': None},
            'sidebar': {'elements': [], 'bbox': None},
            'content': {'elements': [], 'bbox': None},
            'footer': {'elements': [], 'bbox': None}
        }

        # 요소들을 레이아웃 역할별로 그룹화
        for element in elements:
            if element.layout_role in regions:
                regions[element.layout_role]['elements'].append(element)

        # 각 영역의 바운딩 박스 계산
        for region_name, region_info in regions.items():
            if region_info['elements']:
                bboxes = [elem.bbox for elem in region_info['elements']]
                min_x = min([bbox[0] for bbox in bboxes])
                min_y = min([bbox[1] for bbox in bboxes])
                max_x = max([bbox[2] for bbox in bboxes])
                max_y = max([bbox[3] for bbox in bboxes])
                region_info['bbox'] = [min_x, min_y, max_x, max_y]

        return regions

    def extract_layout_containers(self, image: Image.Image) -> List[UIElement]:
        """레이아웃 컨테이너 추출"""
        # 이미지 분석으로 컨테이너 검출
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # 경계선 검출
        edges = cv2.Canny(gray, 50, 150)

        # 컨투어 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        containers = []
        w, h = image.size

        for i, contour in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(contour)

            # 크기 필터링 - 너무 작거나 전체 화면인 경우 제외
            if cw * ch > (w * h * 0.01) and cw * ch < (w * h * 0.8):
                containers.append(UIElement(
                    id=f"container_{i}",
                    type="container",
                    bbox=[x / w, y / h, (x + cw) / w, (y + ch) / h],
                    confidence=0.7,
                    interactivity=False,
                    layout_role="container"
                ))

        return containers


class HierarchyBuilder:
    """UI 계층 구조 생성기"""

    def build_hierarchy(self, elements: List[UIElement], layout_regions: Dict) -> Dict[str, List[str]]:
        """UI 요소간 계층 구조 구축"""
        hierarchy = {}

        # 1. 레이아웃 영역별 그룹화
        for region_name, region_info in layout_regions.items():
            if region_info['elements']:
                region_id = f"region_{region_name}"
                hierarchy[region_id] = [elem.id for elem in region_info['elements']]

                # 각 요소의 parent_id 설정
                for elem in region_info['elements']:
                    elem.parent_id = region_id

        # 2. 공간적 포함 관계 분석
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i != j and self._is_inside(elem1.bbox, elem2.bbox):
                    # elem1이 elem2 안에 포함됨
                    if elem2.id not in hierarchy:
                        hierarchy[elem2.id] = []
                    if elem1.id not in hierarchy[elem2.id]:
                        hierarchy[elem2.id].append(elem1.id)
                        elem1.parent_id = elem2.id
                        elem2.children.append(elem1.id)

        return hierarchy

    def _is_inside(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        """한 박스가 다른 박스 안에 포함되는지 확인"""
        return (inner_bbox[0] >= outer_bbox[0] and
                inner_bbox[1] >= outer_bbox[1] and
                inner_bbox[2] <= outer_bbox[2] and
                inner_bbox[3] <= outer_bbox[3])


class LayoutAwareParser:
    """레이아웃 인식 파서"""

    def __init__(self, config: Dict):
        self.config = config
        self.skeleton_extractor = SkeletonUIExtractor(config)

    def parse_by_layout(self, image: Union[str, Image.Image]) -> Dict:
        """레이아웃별로 파싱 수행"""
        # 1. 스켈레톤 구조 추출
        layout_structure = self.skeleton_extractor.extract_skeleton(image)

        # 2. 각 레이아웃 영역별 세부 파싱
        parsed_regions = {}

        for region_name, region_info in layout_structure.layout_regions.items():
            if region_info['elements']:
                parsed_regions[region_name] = self._parse_region(
                    image,
                    region_info,
                    layout_structure.elements
                )

        # 3. 폼 구조 추출
        forms = self._extract_forms(layout_structure.elements)

        # 4. 네비게이션 구조 추출
        navigation = self._extract_navigation(layout_structure.elements)

        return {
            'skeleton': {
                'structure_type': layout_structure.structure_type,
                'elements': [asdict(elem) for elem in layout_structure.elements],
                'hierarchy': layout_structure.hierarchy
            },
            'layout_regions': {name: {
                'elements': [asdict(elem) for elem in info['elements']],
                'bbox': info['bbox']
            } for name, info in layout_structure.layout_regions.items()},
            'parsed_regions': parsed_regions,
            'forms': forms,
            'navigation': navigation
        }

    def _parse_region(self, image: Image.Image, region_info: Dict, all_elements: List[UIElement]) -> Dict:
        """특정 영역 세부 파싱"""
        if not region_info['bbox']:
            return {}

        # 영역 크롭
        bbox = region_info['bbox']
        w, h = image.size

        left = int(bbox[0] * w)
        top = int(bbox[1] * h)
        right = int(bbox[2] * w)
        bottom = int(bbox[3] * h)

        cropped_region = image.crop((left, top, right, bottom))

        # 크롭된 영역에서 추가 분석
        # OCR 재실행
        ocr_result, _ = check_ocr_box(
            cropped_region,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'text_threshold': 0.6}
        )

        # 상대 좌표를 전체 이미지 좌표로 변환
        region_text, region_bbox = ocr_result
        absolute_bbox = []
        for box in region_bbox:
            x1, y1, x2, y2 = box
            abs_x1 = (left + x1) / w
            abs_y1 = (top + y1) / h
            abs_x2 = (left + x2) / w
            abs_y2 = (top + y2) / h
            absolute_bbox.append([abs_x1, abs_y1, abs_x2, abs_y2])

        return {
            'cropped_ocr': {
                'text': region_text,
                'bbox': absolute_bbox
            },
            'elements_count': len(region_info['elements']),
            'element_types': self._count_element_types(region_info['elements'])
        }

    def _count_element_types(self, elements: List[UIElement]) -> Dict[str, int]:
        """요소 타입별 개수 계산"""
        type_counts = {}
        for elem in elements:
            type_counts[elem.type] = type_counts.get(elem.type, 0) + 1
        return type_counts

    def _extract_forms(self, elements: List[UIElement]) -> List[Dict]:
        """폼 구조 추출"""
        forms = []
        current_form = []

        for elem in elements:
            if elem.type in ['input', 'text']:
                current_form.append(elem)
            elif elem.type == 'button' and current_form:
                # 버튼 발견 시 폼 완료
                forms.append({
                    'inputs': [asdict(e) for e in current_form],
                    'submit_button': asdict(elem)
                })
                current_form = []

        return forms

    def _extract_navigation(self, elements: List[UIElement]) -> Dict:
        """네비게이션 구조 추출"""
        nav_elements = [elem for elem in elements if elem.layout_role == 'navigation']

        if not nav_elements:
            return {}

        # 네비게이션 요소 정렬 (위치 기준)
        nav_elements.sort(key=lambda x: (x.bbox[1], x.bbox[0]))

        return {
            'type': 'horizontal' if len(nav_elements) > 1 else 'menu',
            'elements': [asdict(elem) for elem in nav_elements]
        }


# 편의 함수
def extract_ui_skeleton(image_path: str, config: Optional[Dict] = None) -> Dict:
    """UI 스켈레톤 추출 편의 함수"""
    if config is None:
        config = {
            'som_model_path': 'weights/icon_detect/weights.pt',
            'caption_model_name': 'florence2',
            'caption_model_path': 'weights/icon_caption_florence',
            'BOX_TRESHOLD': 0.05,
            'iou_threshold': 0.7
        }

    parser = LayoutAwareParser(config)
    return parser.parse_by_layout(image_path)


# 사용 예제
if __name__ == "__main__":
    # 설정
    config = {
        'som_model_path': 'weights/icon_detect/weights.pt',
        'caption_model_name': 'florence2',
        'caption_model_path': 'weights/icon_caption_florence',
        'BOX_TRESHOLD': 0.05,
        'iou_threshold': 0.7
    }

    # 파서 생성
    parser = LayoutAwareParser(config)

    # 이미지 파싱
    result = parser.parse_by_layout("./resource/sample/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.png")

    # 결과 저장
    with open("ui_skeleton_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 결과 출력
    print("=== UI 스켈레톤 구조 ===")
    print(f"구조 타입: {result['skeleton']['structure_type']}")
    print(f"총 요소 수: {len(result['skeleton']['elements'])}")

    print("\n=== 레이아웃 영역별 정보 ===")
    for region_name, region_info in result['layout_regions'].items():
        if region_info['elements']:
            print(f"{region_name}: {len(region_info['elements'])}개 요소")

    print("\n=== 네비게이션 구조 ===")
    if result['navigation']:
        print(f"타입: {result['navigation']['type']}")
        print(f"요소 수: {len(result['navigation']['elements'])}")