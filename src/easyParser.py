"""
Extracts Skeleton UI Layout from Samsung Themes using GUIParser
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)

import glob
import json

import torch
import cv2

import numpy as np
from PIL import Image, ImageFile
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import math
import pandas as pd
from tqdm import tqdm

from src.xmlParser import XMLParser
from utils.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
from src.visualizer import visualize_ui_skeleton_result

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
    visual_features: Optional[Dict] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.visual_features is None:
            self.visual_features = {}


@dataclass
class LayoutStructure:
    """레이아웃 구조 정보"""
    structure_type: str
    elements: List[UIElement]
    hierarchy: Dict[str, List[str]]
    layout_regions: Dict[str, Dict]
    grid_structure: Optional[Dict] = None


class SkeletonUIExtractor:
    """스켈레톤 UI 구조 추출"""
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("사용 가능한 자원:", device)

        # 모델 초기화
        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(
            model_name=config.get('caption_model_name', 'florence2'),
            model_name_or_path=config.get('caption_model_path'),
            device=device
        )
        # 레이아웃 분석 매개변수(
        self.layout_detector = LayoutDetector()
        self.hierarchy_builder = HierarchyBuilder()
        self.component_detector = ComponentDetector()

        print("SkeletonUIExtractor 초기화 완료!!!!!!!!!!!!!!!!")

    def extract_skeleton(self, image: Union[str, Image.Image, np.ndarray]) -> LayoutStructure:
        """스켈레톤 UI 구조 추출"""
        # 1. 이미지 전처리
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 이미지 리사이즈 (성능 최적화)
        if max(image.size) > 640:
            new_w = 640
            new_h = int(image.height * (640 / image.width))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        image = image.convert("RGB")
        w, h = image.size

        # 1. OCR로 텍스트 요소 추출
        ocr_result, _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'text_threshold': 0.8, 'paragraph': False},
            use_paddleocr=True
        )
        text_list, ocr_bbox = ocr_result

        # 2. YOLO로 아이콘/버튼 검출
        ui_elements = self._detect_ui_elements(image, ocr_bbox, text_list, w, h)

        # 3. 컴포넌트 패턴 검출 (그리드, 리스트 등)
        pattern_elements = self.component_detector.detect_patterns(image, ui_elements)
        ui_elements.extend(pattern_elements)

        # 4. 시각적 특성 분석
        ui_elements = self._analyze_visual_features(image, ui_elements)

        # 5. 레이아웃 영역 분할
        layout_regions = self.layout_detector.detect_layout_regions(image, ui_elements)

        # 6.계층 구조 분석
        hierarchy = self.hierarchy_builder.build_hierarchy(ui_elements, layout_regions)

        # 7. 그리드 구조 검출
        grid_structure = self._detect_grid_structure(ui_elements)

        # 스켈레톤 구조 반환
        return LayoutStructure(
            structure_type=self._determine_structure_type(layout_regions),
            elements=ui_elements,
            hierarchy=hierarchy,
            layout_regions=layout_regions,
            grid_structure=grid_structure
        )

    def _enhanced_ocr_detection(self, image: Image.Image) -> Tuple[List[str], List[List[float]]]:
        all_texts = []
        all_bboxes = []

        ocr_configs = [
            {'text_threshold': 0.8, 'paragraph': False},
            {'text_threshold': 0.6, 'paragraph': True},
            {'text_threshold': 0.4, 'paragraph': False}
        ]

        for config in ocr_configs:
            ocr_result, _ = check_ocr_box(
                image,
                display_img=False,
                output_bb_format='xyxy',
                easyocr_args=config,
                use_paddleocr=True
            )
            texts, bboxes = ocr_result

            for text, bbox in zip(texts, bboxes):
                if not self._is_duplicate_text(bbox, all_bboxes):
                    all_texts.append(text)
                    all_bboxes.append(bbox)

        return all_texts, all_bboxes

    def _detect_ui_elements(self, image: Image.Image, ocr_bbox: List, text_list: List, w: int, h: int) -> List[UIElement]:
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
            BOX_TRESHOLD=self.config.get('BOX_TRESHOLD', 0.02),
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text_list,
            use_local_semantics=True,
            iou_threshold=self.config.get('iou_threshold', 0.05),
            batch_size=16
        )

        if self.config.get('debug'):
            df = pd.DataFrame(parsed_content_list)
            bbox_df = pd.DataFrame(df['bbox'].tolist(), columns=['bbox_0', 'bbox_1', 'bbox_2', 'bbox_3'])
            df = pd.concat([df.drop(columns='bbox'), bbox_df], axis=1)
            out_dir = './output/csv'
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"{self.config.get('filename')}.csv"), index=False, encoding='utf-8-sig')

        # 파싱된 내용을 UIElement로 변환
        for i, parsed_item in enumerate(parsed_content_list):
            bbox = parsed_item['bbox']
            element_type = self._classify_element_type(parsed_item)
            elements.append(UIElement(
                id=f"{element_type}_{i}",
                type=element_type,
                bbox=bbox,
                content=parsed_item.get('content'),
                confidence=parsed_item.get('confidence', 0.8),
                interactivity=parsed_item.get('interactivity', False),
                layout_role=self._determine_layout_role(bbox, elements)
            ))

        return elements

    def _analyze_visual_features(self, image: Image.Image, elements: List[UIElement]) -> List[UIElement]:
        image_np = np.array(image)

        for element in elements:
            bbox = element.bbox
            x1, y1, x2, y2 = bbox

            # 픽셀 좌표로 변환
            h, w = image_np.shape[:2]
            x1_px = int(x1 * w)
            y1_px = int(y1 * h)
            x2_px = int(x2 * w)
            y2_px = int(y2 * h)

            # 요소 영역 추출
            element_region = image_np[y1_px:y2_px, x1_px:x2_px]

            if element_region.size > 0:
                # 평균 색상
                avg_color = np.mean(element_region.reshape(-1, 3), axis=0)

                # 엣지 밀도 (경계선 검출)
                gray = cv2.cvtColor(element_region, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size

                # 시각적 특성 저장
                element.visual_features = {
                    'avg_color': avg_color.tolist(),
                    'edge_density': edge_density,
                    'aspect_ratio': (x2 - x1) / (y2 - y1) if y2 - y1 > 0 else 1
                }

        return elements

    def _non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Non-Maximum Suppression for multi-scale detections"""
        if not detections:
            return []

        # 신뢰도 기준 정렬
        detections = sorted(detections, key=lambda x: x.get('confidence', 0.5), reverse=True)

        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)

            # 중복 제거
            detections = [d for d in detections if not self._is_overlapping(current['bbox'], d['bbox'], iou_threshold)]

        return keep

    def _is_overlapping(self, bbox1: List[float], bbox2: List[float], threshold: float) -> bool:
        """두 박스가 겹치는지 확인"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 교집합 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return False

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou > threshold

    def _is_duplicate_text(self, bbox: List[float], existing_bboxes: List[List[float]], threshold: float = 0.8) -> bool:
        """텍스트 박스 중복 확인"""
        for existing_bbox in existing_bboxes:
            if self._is_overlapping(bbox, existing_bbox, threshold):
                return True
        return False

    def _classify_element_type(self, parsed_item: Dict) -> str:
        """파싱된 아이템의 타입 분류"""
        if parsed_item.get('type') == 'text':
            return 'text'

        bbox = parsed_item['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 1

        # 컨텐츠 기반 키워드 분류
        content = parsed_item.get('content', '').lower()
        keyword_map = {
            'button': ['button', 'submit', 'click', 'tap'],
            'input': ['input', 'text', 'search', 'enter'],
            'image': ['image', 'photo', 'picture', 'icon'],
            'navigation': ['menu', 'navigation', 'tab']
        }
        for label, keywords in keyword_map.items():
            if any(kw in content for kw in keywords):
                return label

        # 크기와 내용으로 버튼/입력 필드 구분
        if parsed_item.get('interactivity'):
            if aspect_ratio > 3:  # 가로로 긴 경우
                return 'input'
            elif aspect_ratio < 0.3:  # 세로로 긴 경우
                return 'scrollbar'
            else:
                return 'button'
        else:
            return 'banner' if aspect_ratio > 2 else 'icon'

    def _determine_layout_role(self, bbox: List[float], existing_elements: List[UIElement]) -> str:
        """요소의 레이아웃 역할 결정"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Set Variable
        TOP_REGION = 0.15
        BOTTOM_REGION = 0.85
        WIDE_WIDTH = 0.8
        CENTER_REGION_X = (0.3, 0.7)
        CENTER_REGION_Y = (0.3, 0.7)
        SIDEBAR_THRESHOLD = 0.3

        # 상단 영역
        if y1 < TOP_REGION:
            if width > WIDE_WIDTH:
                return 'header'
            elif x1 < 0.2:  # 좌측
                return 'logo'
            elif x2 > 0.8:  # 우측
                return 'toolbar'
            else:
                return 'navigation'

        # 하단 영역
        if y1 > BOTTOM_REGION:
            return 'footer' if width > WIDE_WIDTH else 'bottom_navigation'

        # 좌우 사이드바
        if x1 < 0.2 and height > SIDEBAR_THRESHOLD:  # 좌측 사이드바
            return 'sidebar_left'
        if x2 > 0.8 and height > SIDEBAR_THRESHOLD: # 우측 사이드바
            return 'sidebar_right'

        # 상단 인근 네비게이션
        if 0.15 < y1 < 0.25 and width > 0.5:
            return 'navigation'

        # 중앙 콘텐츠
        if CENTER_REGION_X[0] < x1 < CENTER_REGION_X[1] and CENTER_REGION_Y[0] < y1 < CENTER_REGION_Y[1]:
            return 'main_content'

        return 'content'

    def _determine_structure_type(self, layout_regions: Dict) -> str:
        """레이아웃 구조 타입 결정 (향상된 버전)"""
        active_regions = [name for name, info in layout_regions.items() if info['elements']]

        # 복잡한 레이아웃 패턴 검출
        if 'sidebar_left' in active_regions and 'sidebar_right' in active_regions:
            return 'three_column_layout'
        elif 'sidebar_left' in active_regions or 'sidebar_right' in active_regions:
            return 'sidebar_layout'
        elif 'header' in active_regions and 'footer' in active_regions and 'navigation' in active_regions:
            return 'full_layout'
        elif 'header' in active_regions and 'footer' in active_regions:
            return 'header_footer_layout'
        elif 'navigation' in active_regions:
            return 'navigation_layout'
        elif 'main_content' in active_regions:
            return 'centered_layout'
        else:
            return 'single_column'

    def _detect_grid_structure(self, elements: List[UIElement]) -> Optional[Dict]:
        """그리드 구조 검출"""
        # 유사한 크기의 요소들 그룹화
        size_groups = defaultdict(list)

        for element in elements:
            bbox = element.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # 크기를 반올림하여 그룹화
            size_key = (round(width, 2), round(height, 2))
            size_groups[size_key].append(element)

        # 그리드 후보 찾기
        grid_candidates = []
        for size_key, group in size_groups.items():
            if len(group) >= 3:  # 최소 3개 이상의 유사한 요소
                grid_candidates.append(group)

        if not grid_candidates:
            return None

        # 가장 큰 그리드 선택
        largest_grid = max(grid_candidates, key=len)

        # 그리드 구조 분석
        grid_info = self._analyze_grid_structure(largest_grid)

        return grid_info

    def _analyze_grid_structure(self, elements: List[UIElement]) -> Dict:
        """그리드 구조 분석"""
        # X, Y 좌표 수집
        x_coords = sorted(set(elem.bbox[0] for elem in elements))
        y_coords = sorted(set(elem.bbox[1] for elem in elements))

        # 열과 행 감지
        cols = self._detect_grid_lines(x_coords)
        rows = self._detect_grid_lines(y_coords)

        return {
            'type': 'grid',
            'columns': len(cols),
            'rows': len(rows),
            'elements': [elem.id for elem in elements],
            'cell_size': {
                'width': np.mean([elem.bbox[2] - elem.bbox[0] for elem in elements]),
                'height': np.mean([elem.bbox[3] - elem.bbox[1] for elem in elements])
            }
        }

    def _detect_grid_lines(self, coords: List[float], threshold: float = 0.02) -> List[List[float]]:
        """그리드 라인 검출"""
        if not coords:
            return []

        lines = [[coords[0]]]

        for coord in coords[1:]:
            if coord - lines[-1][-1] < threshold:
                lines[-1].append(coord)
            else:
                lines.append([coord])

        return lines


class ComponentDetector:
    """컴포넌트 패턴 검출기"""

    def detect_patterns(self, image: Image.Image, existing_elements: List[UIElement]) -> List[UIElement]:
        """반복 패턴 검출"""
        detected_patterns = []

        # 1. 리스트 아이템 패턴 검출
        list_items = self._detect_list_pattern(existing_elements)
        detected_patterns.extend(list_items)

        # 2. 카드 패턴 검출
        cards = self._detect_card_pattern(image, existing_elements)
        detected_patterns.extend(cards)

        # 3. 테이블 패턴 검출
        tables = self._detect_table_pattern(existing_elements)
        detected_patterns.extend(tables)

        return detected_patterns

    def _detect_list_pattern(self, elements: List[UIElement]) -> List[UIElement]:
        """리스트 패턴 검출"""
        # Y 좌표 기준으로 정렬
        sorted_elements = sorted(elements, key=lambda e: e.bbox[1])

        list_items = []
        current_list = []
        prev_y = None

        for element in sorted_elements:
            y = element.bbox[1]
            height = element.bbox[3] - element.bbox[1]

            if prev_y is not None:
                gap = y - prev_y

                # 일정한 간격의 요소들을 리스트로 그룹화
                if abs(gap - height) < height * 0.3:  # 30% 오차 허용
                    current_list.append(element)
                else:
                    if len(current_list) >= 3:  # 최소 3개 이상
                        list_items.extend(self._create_list_group(current_list))
                    current_list = [element]
            else:
                current_list = [element]

            prev_y = y

        if len(current_list) >= 3:
            list_items.extend(self._create_list_group(current_list))

        return list_items

    def _create_list_group(self, elements: List[UIElement]) -> List[UIElement]:
        """리스트 그룹 생성"""
        group_elements = []

        for i, element in enumerate(elements):
            element.layout_role = 'list_item'
            element.visual_features['list_index'] = i
            group_elements.append(element)

        return group_elements

    def _detect_card_pattern(self, image: Image.Image, elements: List[UIElement]) -> List[UIElement]:
        """카드 패턴 검출"""
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # 경계선 검출로 카드 찾기
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cards = []
        h, w = image.size

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # 카드 크기 필터링
            if cw * ch > (w * h * 0.05) and cw * ch < (w * h * 0.3):
                # 사각형 근사
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 사각형인 경우 카드로 간주
                if len(approx) == 4:
                    card_elem = UIElement(
                        id=f"card_{len(cards)}",
                        type="card",
                        bbox=[x / w, y / h, (x + cw) / w, (y + ch) / h],
                        confidence=0.8,
                        interactivity=False,
                        layout_role="card"
                    )

                    # 카드 내부 요소 찾기
                    inner_elements = self._find_elements_inside(card_elem.bbox, elements)
                    if inner_elements:
                        card_elem.children = [elem.id for elem in inner_elements]
                        cards.append(card_elem)

        return cards

    def _detect_table_pattern(self, elements: List[UIElement]) -> List[UIElement]:
        """테이블 패턴 검출"""
        # 텍스트 요소만 필터링
        text_elements = [e for e in elements if e.type == 'text']

        if len(text_elements) < 4:
            return []

        # 그리드 정렬 확인
        x_positions = sorted(set(e.bbox[0] for e in text_elements))
        y_positions = sorted(set(e.bbox[1] for e in text_elements))

        # 규칙적인 간격 확인
        if len(x_positions) >= 2 and len(y_positions) >= 2:
            x_gaps = [x_positions[i + 1] - x_positions[i] for i in range(len(x_positions) - 1)]
            y_gaps = [y_positions[i + 1] - y_positions[i] for i in range(len(y_positions) - 1)]

            # 간격이 일정한지 확인
            x_std = np.std(x_gaps) if x_gaps else float('inf')
            y_std = np.std(y_gaps) if y_gaps else float('inf')

            if x_std < 0.02 and y_std < 0.02:  # 표준편차가 작으면 테이블
                table_elem = UIElement(
                    id="table_0",
                    type="table",
                    bbox=[
                        min(e.bbox[0] for e in text_elements),
                        min(e.bbox[1] for e in text_elements),
                        max(e.bbox[2] for e in text_elements),
                        max(e.bbox[3] for e in text_elements)
                    ],
                    confidence=0.9,
                    interactivity=False,
                    layout_role="table"
                )
                table_elem.visual_features['rows'] = len(y_positions)
                table_elem.visual_features['columns'] = len(x_positions)
                return [table_elem]

        return []

    def _find_elements_inside(self, container_bbox: List[float], elements: List[UIElement]) -> List[UIElement]:
        """컨테이너 내부 요소 찾기"""
        inside_elements = []

        for element in elements:
            if self._is_inside(element.bbox, container_bbox):
                inside_elements.append(element)

        return inside_elements

    def _is_inside(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        """한 박스가 다른 박스 안에 포함되는지 확인"""
        return (inner_bbox[0] >= outer_bbox[0] and
                inner_bbox[1] >= outer_bbox[1] and
                inner_bbox[2] <= outer_bbox[2] and
                inner_bbox[3] <= outer_bbox[3])


class LayoutDetector:
    """레이아웃 영역 검출기"""

    def detect_layout_regions(self, image: Image.Image, elements: List[UIElement]) -> Dict[str, Dict]:
        """레이아웃 영역 검출"""
        regions = {
            'header': {'elements': [], 'bbox': None},
            'navigation': {'elements': [], 'bbox': None},
            'sidebar_left': {'elements': [], 'bbox': None},
            'sidebar_right': {'elements': [], 'bbox': None},
            'main_content': {'elements': [], 'bbox': None},
            'content': {'elements': [], 'bbox': None},
            'footer': {'elements': [], 'bbox': None},
            'bottom_navigation': {'elements': [], 'bbox': None},
            'toolbar': {'elements': [], 'bbox': None},
            'logo': {'elements': [], 'bbox': None}
        }

        # 1. 요소들을 레이아웃 역할별로 그룹화
        for element in elements:
            if element.layout_role in regions:
                regions[element.layout_role]['elements'].append(element)

        # 2. 각 영역의 바운딩 박스 계산
        for region_name, region_info in regions.items():
            if region_info['elements']:
                bboxes = [elem.bbox for elem in region_info['elements']]
                min_x = min([bbox[0] for bbox in bboxes])
                min_y = min([bbox[1] for bbox in bboxes])
                max_x = max([bbox[2] for bbox in bboxes])
                max_y = max([bbox[3] for bbox in bboxes])
                region_info['bbox'] = [min_x, min_y, max_x, max_y]

        # 3. 컨테이너 기반 영역 검출
        container_regions = self._detect_container_regions(image, elements)
        regions.update(container_regions)

        # 4. 시맨틱 그룹화
        semantic_regions = self._semantic_grouping(elements)
        regions.update(semantic_regions)

        return regions

    def _detect_container_regions(self, image: Image.Image, elements: List[UIElement]) -> Dict[str, Dict]:
        """ 컨테이너 기반 영역 검출"""
        # 이미지 분석으로 컨테이너 검출
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # 검출 영역 분리
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 컨투어 찾기
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        container_regions = {}
        w, h = image.size

        for i, contour in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            # 유의미한 크기의 컨테이너만 선택
            if area > (w * h * 0.02) and area < (w * h * 0.8):
                bbox = [x / w, y / h, (x + cw) / w, (y + ch) / h]

                inner_elements = []
                for elem in elements:
                    if self._is_inside(elem.bbox, bbox):
                        inner_elements.append(elem)

                if len(inner_elements) >= 2:  # 최소 2개 이상의 요소 포함
                    region_name = f"container_{i}"
                    container_regions[region_name] = {
                        'elements': inner_elements,
                        'bbox': bbox,
                        'type': 'container'
                    }

        return container_regions

    def _semantic_grouping(self, elements: List[UIElement]) -> Dict[str, Dict]:
        """시맨틱 그룹화"""
        semantic_groups = defaultdict(list)

        # 유사한 타입의 요소들을 그룹화
        for element in elements:
            if element.type in ['button', 'input']:
                semantic_groups['form_elements'].append(element)
            elif element.type in ['navigation', 'menu']:
                semantic_groups['navigation_elements'].append(element)
            elif element.type == 'image':
                semantic_groups['media_elements'].append(element)

        # 그룹을 영역으로 변환
        semantic_regions = {}
        for group_name, group_elements in semantic_groups.items():
            if len(group_elements) >= 2:
                bboxes = [elem.bbox for elem in group_elements]
                min_x = min([bbox[0] for bbox in bboxes])
                min_y = min([bbox[1] for bbox in bboxes])
                max_x = max([bbox[2] for bbox in bboxes])
                max_y = max([bbox[3] for bbox in bboxes])

                semantic_regions[group_name] = {
                    'elements': group_elements,
                    'bbox': [min_x, min_y, max_x, max_y],
                    'type': 'semantic_group'
                }

        return semantic_regions

    def _is_inside(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        """한 박스가 다른 박스 안에 포함되는지 확인"""
        return (inner_bbox[0] >= outer_bbox[0] and
                inner_bbox[1] >= outer_bbox[1] and
                inner_bbox[2] <= outer_bbox[2] and
                inner_bbox[3] <= outer_bbox[3])

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

    def _analyze_spatial_hierarchy(self, elements: List[UIElement], hierarchy: Dict[str, List[str]]):
        """공간적 계층 관계 분석"""
        # # 요소를 크기 순으로 정렬 (큰 것부터)
        sorted_elements = sorted(elements,
                                 key=lambda e: (e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1]),
                                 reverse=True)

        for i, parent in enumerate(sorted_elements):
            for j, child in enumerate(sorted_elements[i + 1:], i + 1):
                if self._is_inside(child.bbox, parent.bbox):
                    # 직접 포함 관계인지 확인
                    is_direct_child = True

                    # 중간에 다른 컨테이너가 있는지 확인
                    for k in range(i + 1, j):
                        intermediate = sorted_elements[k]
                        if (self._is_inside(child.bbox, intermediate.bbox) and
                                self._is_inside(intermediate.bbox, parent.bbox)):
                            is_direct_child = False
                            break

                    if is_direct_child:
                        if parent.id not in hierarchy:
                            hierarchy[parent.id] = []
                        if child.id not in hierarchy[parent.id]:
                            hierarchy[parent.id].append(child.id)
                            child.parent_id = parent.id
                            parent.children.append(child.id)

    def _analyze_visual_hierarchy(self, elements: List[UIElement], hierarchy: Dict[str, List[str]]):
        """시각적 계층 분석 (크기, 위치, 스타일 기반)"""
        # 헤더 요소들을 크기별로 그룹화
        text_elements = [e for e in elements if e.type == 'text']

        if text_elements:
            # 텍스트 크기 추정 (bbox 높이 기반)
            text_sizes = [(e, e.bbox[3] - e.bbox[1]) for e in text_elements]
            text_sizes.sort(key=lambda x: x[1], reverse=True)

            # 제목 계층 구성
            if len(text_sizes) >= 2:
                # 가장 큰 텍스트를 h1으로
                h1_elements = [e for e, size in text_sizes if size > text_sizes[0][1] * 0.8]

                # 중간 크기를 h2로
                avg_size = np.mean([size for _, size in text_sizes])
                h2_elements = [e for e, size in text_sizes
                               if avg_size * 0.8 < size <= text_sizes[0][1] * 0.8]

                # 계층 구성
                for h1 in h1_elements:
                    h1_group_id = f"heading_group_{h1.id}"
                    hierarchy[h1_group_id] = [h1.id]

                    # 근처의 h2 요소들을 자식으로
                    for h2 in h2_elements:
                        if abs(h2.bbox[1] - h1.bbox[3]) < 0.1:  # 수직으로 가까운 경우
                            hierarchy[h1_group_id].append(h2.id)
                            h2.parent_id = h1_group_id

    def _form_logical_groups(self, elements: List[UIElement], hierarchy: Dict[str, List[str]]):
        """논리적 그룹 형성"""
        # 폼 그룹 검출
        form_groups = self._detect_form_groups(elements)
        for i, group in enumerate(form_groups):
            group_id = f"form_group_{i}"
            hierarchy[group_id] = [elem.id for elem in group]
            for elem in group:
                elem.parent_id = group_id

        # 리스트 그룹 검출
        list_groups = self._detect_list_groups(elements)
        for i, group in enumerate(list_groups):
            group_id = f"list_group_{i}"
            hierarchy[group_id] = [elem.id for elem in group]
            for elem in group:
                elem.parent_id = group_id

        # 네비게이션 그룹 검출
        nav_groups = self._detect_navigation_groups(elements)
        for i, group in enumerate(nav_groups):
            group_id = f"nav_group_{i}"
            hierarchy[group_id] = [elem.id for elem in group]
            for elem in group:
                elem.parent_id = group_id

    def _detect_form_groups(self, elements: List[UIElement]) -> List[List[UIElement]]:
        """폼 그룹 검출"""
        form_groups = []
        current_form = []

        # Y 좌표로 정렬
        sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))

        for element in sorted_elements:
            if element.type in ['input', 'text', 'button']:
                if current_form:
                    # 이전 요소와의 거리 확인
                    prev_elem = current_form[-1]
                    y_distance = element.bbox[1] - prev_elem.bbox[3]

                    if y_distance < 0.05:  # 가까운 경우
                        current_form.append(element)
                    else:
                        if len(current_form) >= 2:
                            form_groups.append(current_form)
                        current_form = [element]
                else:
                    current_form = [element]

        if len(current_form) >= 2:
            form_groups.append(current_form)

        return form_groups

    def _detect_list_groups(self, elements: List[UIElement]) -> List[List[UIElement]]:
        """리스트 그룹 검출"""
        list_groups = []

        # 유사한 요소들을 그룹화
        element_groups = defaultdict(list)
        for elem in elements:
            # 크기와 타입을 키로 사용
            width = elem.bbox[2] - elem.bbox[0]
            height = elem.bbox[3] - elem.bbox[1]
            key = (elem.type, round(width, 2), round(height, 2))
            element_groups[key].append(elem)

        # 수직으로 정렬된 그룹 찾기
        for group in element_groups.values():
            if len(group) >= 3:
                # Y 좌표로 정렬
                sorted_group = sorted(group, key=lambda e: e.bbox[1])

                # 일정한 간격인지 확인
                gaps = []
                for i in range(1, len(sorted_group)):
                    gap = sorted_group[i].bbox[1] - sorted_group[i - 1].bbox[3]
                    gaps.append(gap)

                if gaps:
                    avg_gap = np.mean(gaps)
                    std_gap = np.std(gaps)

                    # 간격이 일정한 경우 리스트로 판단
                    if std_gap < avg_gap * 0.3:  # 30% 이내 편차
                        list_groups.append(sorted_group)

        return list_groups

    def _detect_navigation_groups(self, elements: List[UIElement]) -> List[List[UIElement]]:
        """네비게이션 그룹 검출"""
        nav_groups = []

        # 수평으로 정렬된 인터랙티브 요소들 찾기
        interactive_elements = [e for e in elements if e.interactivity or e.type in ['button', 'navigation']]

        # Y 좌표가 비슷한 요소들 그룹화
        y_groups = defaultdict(list)
        for elem in interactive_elements:
            y_center = (elem.bbox[1] + elem.bbox[3]) / 2
            # Y 좌표를 반올림하여 그룹화
            y_key = round(y_center, 2)
            y_groups[y_key].append(elem)

        # 수평으로 3개 이상 정렬된 그룹 찾기
        for group in y_groups.values():
            if len(group) >= 3:
                # X 좌표로 정렬
                sorted_group = sorted(group, key=lambda e: e.bbox[0])

                # 일정한 간격인지 확인
                gaps = []
                for i in range(1, len(sorted_group)):
                    gap = sorted_group[i].bbox[0] - sorted_group[i - 1].bbox[2]
                    gaps.append(gap)

                if gaps:
                    avg_gap = np.mean(gaps)
                    std_gap = np.std(gaps)

                    # 간격이 일정한 경우 네비게이션으로 판단
                    if std_gap < avg_gap * 0.5:  # 50% 이내 편차
                        nav_groups.append(sorted_group)

        return nav_groups

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
        self.overlap_threshold = 0.5
        self.skeleton_extractor = SkeletonUIExtractor(config)

    def parse_by_layout(self, image: Union[str, Image.Image], xml_path: Optional[str] = None) -> Dict:
        """레이아웃별로 파싱 수행"""

        if isinstance(image, str):
            self.filename = os.path.basename(image).split('.')[0]
            image = Image.open(image)

        # 1. 스켈레톤 구조 추출
        layout_structure = self.skeleton_extractor.extract_skeleton(image)

        if xml_path and os.path.exists(xml_path):
            try:
                # XMLParser 생성
                xml_parser = XMLParser(image_path=image_path, xml_path=xml_path)
                # xml_parser.check_xml_structure()
                # xml_parser.check_bounds_validity()
                xml_components = xml_parser.get_components()
                print(f"XML에서 {len(xml_components)}개 컴포넌트 추출")

                merged_elements = self._merge_skeleton_with_xml(
                    layout_structure.elements,
                    xml_components,
                    xml_parser
                )
                layout_structure.elements = merged_elements
                print(f"병합 후 최종: {len(layout_structure.elements)}개 요소")

            except Exception as e:
                print(f"XML 병합 실패: {e}, 기본 스켈레톤만 사용")

        # 2. 각 레이아웃 영역별 세부 파싱
        parsed_regions = {}
        for region_name, region_info in layout_structure.layout_regions.items():
            if region_info['elements']:
                parsed_regions[region_name] = self._parse_region(
                    image,
                    region_info,
                    layout_structure.elements
                )

        # # 3. 폼 구조 추출
        # forms = self._extract_forms(layout_structure.elements)
        #
        # # 4. 네비게이션 구조 추출
        # navigation = self._extract_navigation(layout_structure.elements)
        #
        # # 5. 인터랙션 맵 생성
        # interaction_map = self._create_interaction_map(layout_structure.elements)
        #
        # # 6. 접근성 정보 추출
        # accessibility_info = self._extract_accessibility_info(layout_structure.elements)

        return {
            'skeleton': {
                'structure_type': layout_structure.structure_type,
                'elements': [asdict(elem) for elem in layout_structure.elements],
                'hierarchy': layout_structure.hierarchy,
                'metadata': {
                    'source': 'xml_ocr_yolo_merged' if xml_path else 'ocr_yolo_only',
                    'total_elements': len(layout_structure.elements),
                    'image_path': image_path,
                    'xml_path': xml_path
                }
            },
            'layout_regions': {name: {
                'elements': [asdict(elem) for elem in info['elements']],
                'bbox': info['bbox'],
                'type': info.get('type', 'layout')
            } for name, info in layout_structure.layout_regions.items()},
            'parsed_regions': parsed_regions,
            # 'forms': forms,
            # 'navigation': navigation,
            # 'grid_structure': layout_structure.grid_structure,
            # 'interaction_map': interaction_map,
            # 'accessibility': accessibility_info,
            # 'statistics': self._calculate_statistics(layout_structure.elements)
        }

    # def _merge_skeleton_with_xml(self, ui_elements: List[UIElement], xml_components: List, xml_parser_instance=None) -> List[UIElement]:
    #
    #     ui_elements_dict = [self._uielement_to_dict(elem) for elem in ui_elements]
    #
    #     # XMLParser의 병합 로직 활용
    #     if xml_components and xml_parser_instance:
    #         try:
    #             # XMLParser 인스턴스의 merge_with_elements 메서드 사용
    #             merged_components = xml_parser_instance.merge_with_elements(
    #                 xml_components, ui_elements_dict, element_type='dict'
    #             )
    #
    #             # 병합된 결과를 다시 UIElement로 변환
    #             merged_ui_elements = []
    #             for comp in merged_components:
    #                 ui_elem = self._dict_to_uielement(asdict(comp))
    #                 merged_ui_elements.append(ui_elem)
    #
    #             # 기존 UIElement 중 병합되지 않은 것들도 추가
    #             existing_ids = {comp.id for comp in merged_components}
    #             for ui_elem in ui_elements:
    #                 if ui_elem.id not in existing_ids:
    #                     merged_ui_elements.append(ui_elem)
    #
    #             return merged_ui_elements
    #         except Exception as e:
    #             print(f"XMLParser 병합 실패: {e}")
    #             return ui_elements
    #
    #     return ui_elements

    def _merge_skeleton_with_xml(self, ui_elements: List[UIElement], xml_components: List, xml_parser_instance=None) -> \
    List[UIElement]:
        # print(f"-------------------- * XML 병합 시작 * --------------------")
        # print(f"UI Elements 개수: {len(ui_elements)}")
        # print(f"XML Components 개수: {len(xml_components) if xml_components else 0}")
        # print(f"XML Parser Instance: {xml_parser_instance is not None}")

        if not xml_components:
            return ui_elements

        # XML 컴포넌트 유효성 검사 및 필터링
        valid_xml_components = []
        for comp in xml_components:
            bbox = getattr(comp, 'bbox', None)
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                # 유효한 좌표 범위와 크기 확인
                if (0 <= x1 < 1 and 0 <= y1 < 1 and
                        x1 < x2 <= 1 and y1 < y2 <= 1 and
                        (x2 - x1) > 0.005 and (y2 - y1) > 0.005):
                    valid_xml_components.append(comp)
                else:
                    print(f" Invalid bbox: {comp.id} - {bbox}")
            else:
                print(f"None bbox : {getattr(comp, 'id', 'unknown')}")
        print(f"Valid XML Components: {len(valid_xml_components)}개")

        if not valid_xml_components:
            return ui_elements

        unique_ui_elements = []
        seen_ids = set()
        for elem in ui_elements:
            if elem.id not in seen_ids:
                unique_ui_elements.append(elem)
                seen_ids.add(elem.id)
            # else:
                # print(f"Removed duplicated UI Element: {elem.id}")
        print(f"중복 제거 후 UI Elements: {len(unique_ui_elements)}개")

        if xml_parser_instance:
            try:
                ui_elements_dict = [self._uielement_to_dict(elem) for elem in unique_ui_elements]

                merged_xml_components = xml_parser_instance.merge_with_elements(
                    valid_xml_components, ui_elements_dict, element_type='dict'
                )

                print(f"XMLParser 병합 완료: {len(merged_xml_components)}개")

                merged_ui_elements = []
                for xml_comp in merged_xml_components:
                    try:
                        ui_element_dict = {
                            'id': xml_comp.id,
                            'type': self._map_xml_type_to_ui_type(xml_comp.type),
                            'bbox': xml_comp.bbox,
                            'content': xml_comp.content or xml_comp.content_desc or '',
                            'confidence': 0.9 if getattr(xml_comp, 'ocr_matched', False) else 0.8,
                            'interactivity': xml_comp.interactivity,
                            'parent_id': xml_comp.parent_id,
                            'children': xml_comp.children,
                            'layout_role': self._infer_layout_role_from_xml_type(xml_comp.type),
                            'visual_features': xml_comp.visual_features or {}
                        }

                        ui_elem = self._dict_to_uielement(ui_element_dict)
                        merged_ui_elements.append(ui_elem)

                    except Exception as e:
                        print(f"XML 컴포넌트 변환 실패 {xml_comp.id}: {e}")

                for ui_elem in unique_ui_elements:
                    should_add = True

                    for xml_comp in merged_xml_components:
                        if self._is_spatially_overlapping(ui_elem.bbox, xml_comp.bbox):
                            should_add = False
                            # overlap_ratio = self._calculate_overlap_ratio(ui_elem.bbox, xml_comp.bbox)
                            # print(f"기존 UI Element 제외 (공간적 겹침): {ui_elem.id} <-> {xml_comp.id}")
                            # print(f"  UI bbox: {ui_elem.bbox}")
                            # print(f"  XML bbox: {xml_comp.bbox}")
                            # print(f"  ROI: {overlap_ratio:.3f}")
                            break

                    if should_add:
                        merged_ui_elements.append(ui_elem)
                        # print(f"기존 UI Element 추가: {ui_elem.id}")

                print(f"최종 병합 결과: {len(merged_ui_elements)}개")
                return merged_ui_elements

            except Exception as e:
                print(f"XMLParser 병합 실패: {e}")
                import traceback
                traceback.print_exc()
                return self._fallback_direct_merge(unique_ui_elements, valid_xml_components)

        return self._fallback_direct_merge(unique_ui_elements, valid_xml_components)

    def _calculate_overlap_ratio(self, bbox1, bbox2):
        """두 bounding box의 겹침 비율 계산"""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

        # 겹치는 영역 계산
        overlap_x1 = max(x1_1, x1_2)
        overlap_y1 = max(y1_1, y1_2)
        overlap_x2 = min(x2_1, x2_2)
        overlap_y2 = min(y2_1, y2_2)

        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            return 0.0

        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        min_area = min(area1, area2)
        return overlap_area / min_area if min_area > 0 else 0.0

    def _map_xml_type_to_ui_type(self, xml_type: str) -> str:
        """XML 컴포넌트 타입을 UI 타입으로 매핑"""
        type_mapping = {
            'TextView': 'text',
            'EditText': 'input',
            'Button': 'button',
            'ImageButton': 'button',
            'ImageView': 'image',
            'RadioButton': 'button',
            'CheckBox': 'button',
            'Switch': 'input',
            'ToggleButton': 'button',
            'Spinner': 'input',
            'SeekBar': 'input',
            'ProgressBar': 'progress'
        }
        return type_mapping.get(xml_type, 'unknown')

    def _infer_layout_role_from_xml_type(self, xml_type: str) -> str:
        """XML 타입에서 layout_role 추론"""
        if xml_type in ['Button', 'ImageButton']:
            return 'toolbar'
        elif xml_type in ['TextView']:
            return 'content'
        elif xml_type in ['EditText', 'Switch', 'RadioButton', 'CheckBox']:
            return 'form_input'
        elif xml_type in ['ImageView']:
            return 'content'
        else:
            return 'content'

    def _fallback_direct_merge(self, ui_elements: List[UIElement], xml_components: List) -> List[UIElement]:
        print("FallBack manual merge mode")

        merged_elements = []

        for xml_comp in xml_components:
            try:
                ui_element_dict = {
                    'id': xml_comp.id,
                    'type': self._map_xml_type_to_ui_type(xml_comp.type),
                    'bbox': xml_comp.bbox,
                    'content': xml_comp.content or xml_comp.content_desc or '',
                    'confidence': 0.8,
                    'interactivity': xml_comp.interactivity,
                    'parent_id': xml_comp.parent_id,
                    'children': xml_comp.children,
                    'layout_role': self._infer_layout_role_from_xml_type(xml_comp.type),
                    'visual_features': xml_comp.visual_features or {}
                }

                ui_elem = self._dict_to_uielement(ui_element_dict)
                merged_elements.append(ui_elem)

            except Exception as e:
                print(f"폴백 변환 실패 {xml_comp.id}: {e}")

        xml_ids = {comp.id for comp in xml_components}

        for ui_elem in ui_elements:
            if ui_elem.id not in xml_ids:
                # 공간적 겹침 확인
                is_overlapping = False
                for xml_comp in xml_components:
                    if self._is_spatially_overlapping(ui_elem.bbox, xml_comp.bbox):
                        is_overlapping = True
                        break

                if not is_overlapping:
                    merged_elements.append(ui_elem)

        print(f"폴백 병합 완료: {len(merged_elements)}개")
        return merged_elements

    def _is_spatially_overlapping(self, bbox1, bbox2):
        """공간적 겹침 확인"""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return False

        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

        overlap_x1 = max(x1_1, x1_2)
        overlap_y1 = max(y1_1, y1_2)
        overlap_x2 = min(x2_1, x2_2)
        overlap_y2 = min(y2_1, y2_2)

        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            return False

        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        min_area = min(area1, area2)
        overlap_ratio = overlap_area / min_area if min_area > 0 else 0

        return overlap_ratio > self.overlap_threshold

    def _uielement_to_dict(self, ui_element: UIElement) -> Dict:
        """UIElement를 dict로 변환"""
        return {
            'id': ui_element.id,
            'type': ui_element.type,
            'bbox': ui_element.bbox,
            'content': ui_element.content,
            'confidence': ui_element.confidence,
            'interactivity': ui_element.interactivity,
            'parent_id': ui_element.parent_id,
            'children': ui_element.children,
            'layout_role': ui_element.layout_role,
            'visual_features': ui_element.visual_features
        }

    def _dict_to_uielement(self, elem_dict: Dict) -> UIElement:
        """dict를 UIElement로 변환"""
        return UIElement(
            id=elem_dict.get('id', ''),
            type=elem_dict.get('type', 'unknown'),
            bbox=elem_dict.get('bbox', [0, 0, 0, 0]),
            content=elem_dict.get('content'),
            confidence=elem_dict.get('confidence', 0.0),
            interactivity=elem_dict.get('interactivity', False),  # clickable 참조 제거
            parent_id=elem_dict.get('parent_id'),
            children=elem_dict.get('children', []),
            layout_role=elem_dict.get('layout_role'),
            visual_features=elem_dict.get('visual_features', {})
        )

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

        # 경계 확인
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        if right <= left or bottom <= top:
            return {}

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

        # 영역 특성 분석
        region_features = self._analyze_region_features(cropped_region, region_info['elements'])

        return {
            'cropped_ocr': {
                'text': region_text,
                'bbox': absolute_bbox
            },
            'elements_count': len(region_info['elements']),
            'element_types': self._count_element_types(region_info['elements']),
            'features': region_features
        }

    def _analyze_region_features(self, region_image: Image.Image, elements: List[UIElement]) -> Dict:
        """영역 특성 분석"""
        region_image = region_image.convert("RGB")
        region_np = np.array(region_image)

        # 색상 분석
        avg_color = np.mean(region_np.reshape(-1, 3), axis=0)

        # 밀도 분석
        element_area = sum((e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1]) for e in elements)
        region_area = region_image.width * region_image.height / (region_np.shape[0] * region_np.shape[1])
        density = element_area / region_area if region_area > 0 else 0

        # 복잡도 분석
        gray = cv2.cvtColor(region_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        return {
            'avg_color': avg_color.tolist(),
            'element_density': density,
            'edge_complexity': edge_density,
            'element_alignment': self._check_alignment(elements)
        }

    def _check_alignment(self, elements: List[UIElement]) -> Dict:
        """요소 정렬 확인"""
        if len(elements) < 2:
            return {'horizontal': False, 'vertical': False}

        # X, Y 좌표 수집
        x_coords = [e.bbox[0] for e in elements]
        y_coords = [e.bbox[1] for e in elements]

        # 정렬 확인
        x_aligned = len(set(round(x, 2) for x in x_coords)) < len(elements) / 2
        y_aligned = len(set(round(y, 2) for y in y_coords)) < len(elements) / 2

        return {
            'horizontal': y_aligned,
            'vertical': x_aligned
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

    def _create_interaction_map(self, elements: List[UIElement]) -> Dict:
        """인터랙션 맵 생성"""
        interaction_map = {
            'clickable_areas': [],
            'input_areas': [],
            'scrollable_areas': []
        }

        for elem in elements:
            if elem.interactivity or elem.type in ['button', 'navigation']:
                interaction_map['clickable_areas'].append({
                    'id': elem.id,
                    'bbox': elem.bbox,
                    'type': elem.type
                })
            elif elem.type == 'input':
                interaction_map['input_areas'].append({
                    'id': elem.id,
                    'bbox': elem.bbox
                })
            elif elem.type == 'scrollbar':
                interaction_map['scrollable_areas'].append({
                    'id': elem.id,
                    'bbox': elem.bbox
                })

        return interaction_map

    def _extract_accessibility_info(self, elements: List[UIElement]) -> Dict:
        """접근성 정보 추출"""
        accessibility = {
            'text_elements': len([e for e in elements if e.type == 'text']),
            'interactive_elements': len([e for e in elements if e.interactivity]),
            'navigation_elements': len([e for e in elements if e.layout_role == 'navigation']),
            'form_elements': len([e for e in elements if e.type in ['input', 'button']]),
            'contrast_issues': [],
            'size_issues': []
        }

        # 크기 문제 확인
        for elem in elements:
            width = elem.bbox[2] - elem.bbox[0]
            height = elem.bbox[3] - elem.bbox[1]

            # 너무 작은 인터랙티브 요소
            if elem.interactivity and (width < 0.05 or height < 0.05):
                accessibility['size_issues'].append({
                    'id': elem.id,
                    'size': [width, height]
                })

        return accessibility

    def _calculate_statistics(self, elements: List[UIElement]) -> Dict:
        """통계 정보 계산"""
        stats = {
            'total_elements': len(elements),
            'elements_by_type': self._count_element_types(elements),
            'elements_by_role': defaultdict(int),
            'average_element_size': 0,
            'coverage_ratio': 0,
            'complexity_score': 0
        }

        # 역할별 집계
        for elem in elements:
            stats['elements_by_role'][elem.layout_role] += 1
        stats['elements_by_role'] = dict(stats['elements_by_role'])

        # 평균 크기 계산
        if elements:
            sizes = [(e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1]) for e in elements]
            stats['average_element_size'] = np.mean(sizes)
            stats['coverage_ratio'] = sum(sizes)

        # 복잡도 점수 계산
        stats['complexity_score'] = self._calculate_complexity_score(elements)

        return stats

    def _calculate_complexity_score(self, elements: List[UIElement]) -> float:
        """UI 복잡도 점수 계산"""
        try:
            # 요소 수 기반 점수
            element_score = min(len(elements) / 50, 1.0)

            # 타입 다양성 점수
            unique_types = len(set(e.type for e in elements))
            type_score = min(unique_types / 10, 1.0)

            # 계층 깊이 점수
            try:
                max_depth = self._calculate_max_depth(elements)
                depth_score = min(max_depth / 5, 1.0)
            except RecursionError:
                print("RecursionError in depth calculation, using default")
                depth_score = 0.5  # 기본값

            # 종합 점수
            complexity = (element_score + type_score + depth_score) / 3
            return round(complexity, 2)

        except Exception as e:
            print(f"Error in complexity calculation: {e}")
            return 0.5

    def _calculate_max_depth(self, elements: List[UIElement]) -> int:
        """최대 계층 깊이 계산"""
        depths = {}

        def get_depth(elem_id: str, visited: Optional[set] = None) -> int:
            if visited is None:
                visited = set()

            # 이미 계산된 깊이가 있으면 반환
            if elem_id in depths:
                return depths[elem_id]

            if elem_id in visited:
                # print(f"순환 참조 감지: {elem_id}")
                depths[elem_id] = 0
                return 0

            visited.add(elem_id)

            # 원소 찾기
            elem = next((e for e in elements if e.id == elem_id), None)
            if not elem or not hasattr(elem, 'parent_id') or elem.parent_id is None:
                depths[elem_id] = 0
                return 0

            # 부모의 깊이 계산
            try:
                parent_depth = get_depth(elem.parent_id, visited.copy())
                depth = parent_depth + 1
            except RecursionError:
                print(f"RecursionError for parent {elem.parent_id}")
                depth = 0

            depths[elem_id] = depth
            return depth

        max_depth = 0
        for elem in elements:
            try:
                depth = get_depth(elem.id)
                max_depth = max(max_depth, depth)
            except Exception as e:
                print(f"Error calculating depth for {elem.id}: {e}")
                continue

        return max_depth

    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """두 박스 간 거리 계산"""
        # 중심점 계산
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

        # 유클리드 거리
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

        return distance

    def _get_group_bbox(self, elements: List[UIElement]) -> List[float]:
        """요소 그룹의 전체 바운딩 박스 계산"""
        if not elements:
            return [0, 0, 0, 0]

        min_x = min(e.bbox[0] for e in elements)
        min_y = min(e.bbox[1] for e in elements)
        max_x = max(e.bbox[2] for e in elements)
        max_y = max(e.bbox[3] for e in elements)

        return [min_x, min_y, max_x, max_y]

def extract_ui_skeleton(image_path: str, xml_path: str, config: Optional[Dict] = None) -> Dict:
    """UI 스켈레톤 추출 편의 함수"""
    if config is None:
        config = {
            'som_model_path': 'weights/icon_detect/model_hf.pt',
            'caption_model_name': 'florence2',
            'caption_model_path': 'weights/icon_caption_florence',
            'BOX_TRESHOLD': 0.02,
            'iou_threshold': 0.02
        }

    parser = LayoutAwareParser(config)
    return parser.parse_by_layout(image_path, xml_path)


class EasyParserRunner:
    def __init__(self, base_dir: str, cluster_base_dir: str, json_output_dir: str,
                 visual_output_dir: str, num_cluster: int):
        self.base_dir = base_dir
        self.cluster_base_dir = cluster_base_dir
        self.json_output_dir = json_output_dir
        self.visual_output_dir = visual_output_dir
        self.num_cluster = num_cluster
        self.config = {
            'som_model_path': os.path.join(base_dir, 'src', 'weights', 'icon_detect', 'model.pt'),
            'caption_model_name': 'florence2',
            'caption_model_path': os.path.join(base_dir, 'src', 'weights', 'icon_caption_florence'),
            'BOX_TRESHOLD': 0.05,
            'iou_threshold': 0.4
        }
        self.parser = LayoutAwareParser(self.config)

    def run(self):

        for cluster_id in range(self.num_cluster):
            cluster_dir = os.path.join(self.cluster_base_dir, f"cluster{cluster_id:02d}")
            print(f"\n[INFO] 클러스터 {cluster_id:02d} 처리 중...")

            cluster_images = sorted(glob.glob(os.path.join(cluster_dir, "*.png")))
            print(f"[INFO] 클러스터 {cluster_id:02d} | 이미지 수: {len(cluster_images)}")

            for image_path in cluster_images:
                filename = os.path.splitext(os.path.basename(image_path))[0]

                cluster_json_dir = os.path.join(self.json_output_dir, f"cluster{cluster_id:02d}")
                os.makedirs(cluster_json_dir, exist_ok=True)
                json_path = os.path.join(cluster_json_dir, f"{filename}.json")

                print(f"[PROCESSING] {image_path}")

                try:
                    result = self.parser.parse_by_layout(image_path=image_path, xml_path=None)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"[INFO] 분석 결과: {json_path}")
                except Exception as e:
                    print(f"[ERROR] 분석 실패: {e}")
                    continue

                try:
                    visual_path = os.path.join(self.visual_output_dir, f"cluster{cluster_id:02d}", f"{filename}.png")
                    os.makedirs(os.path.dirname(visual_path), exist_ok=True)

                    visualize_ui_skeleton_result(
                        image_path=image_path,
                        result_path=json_path,
                        output_dir=self.visual_output_dir,
                        cluster_output_name=visual_path
                    )
                except Exception as e:
                    print(f"[ERROR] 시각화 실패: {e}")
                    continue


ImageFile.LOAD_TRUNCATED_IMAGES = False
def is_image_valid(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except Exception:
        return True

if __name__ == "__main__":

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    OUT_DIR = os.path.join(BASE_DIR, 'output/json')
    os.makedirs(OUT_DIR, exist_ok=True)
    config = {
        'som_model_path': os.path.join(BASE_DIR, 'src/weights/icon_detect/model.pt'),
        'caption_model_name': 'florence2',
        'caption_model_path': os.path.join(BASE_DIR, 'src/weights/icon_caption_florence'),
        'BOX_TRESHOLD': 0.01,
        'iou_threshold': 0.01,
        'debug': True,
    }

    parser = LayoutAwareParser(config)
    rootDir = "./resource"

    img_paths = glob.glob(f"{rootDir}/image/*.png", recursive=True)
    broken_files = []
    not_found_component = []

    for image_path in tqdm(img_paths):
        filename = os.path.splitext(os.path.basename(image_path))[0]
        xml_path=f"{rootDir}/xml/{filename}.xml"
        config['filename'] = filename

        if os.path.isfile(f"{OUT_DIR}/{filename}.json"):
            print(f"해당 파일은 존재 합니다.:{OUT_DIR}/{filename}.json")
            continue

        print(f"image path: {image_path}")
        if is_image_valid(image_path):
            broken_files.append(os.path.basename(image_path))
            print(f"해당 파일에 컴포넌트가 존재하지 않습니다.: {image_path}")
            continue

        if not os.path.isfile(xml_path):
            continue

        try:
            result = parser.parse_by_layout(image=image_path, xml_path=xml_path)
        except Exception as e:
            not_found_component.append(os.path.basename(image_path))
            print(f"해당 파일은 손상 되었습니다.: {image_path}")
            continue

        with open(f"{OUT_DIR}/{filename}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("\n=== UI 스켈레톤 구조 ===")
        print(f"   구조 타입: {result['skeleton']['structure_type']}")
        print(f"   총 요소 수: {len(result['skeleton']['elements'])}")

        print("\n=== 레이아웃 영역별 정보 ===")
        for region_name, region_info in result['layout_regions'].items():
            if region_info['elements']:
                print(f"   {region_name}: {len(region_info['elements'])}개 요소")

        # print("\n=== 네비게이션 구조 ===")
        # if result['navigation']:
        #     print(f"   타입: {result['navigation']['type']}")
        #     print(f"   요소 수: {len(result['navigation']['elements'])}")

    df = pd.DataFrame(broken_files, columns=["filename"])
    df.to_csv("broken_images.csv", index=False, encoding="utf-8")
    print(f"[완료] 손상된 이미지 {len(df)}개")

    df = pd.DataFrame(not_found_component, columns=["filename"])
    df.to_csv("not_found_images.csv", index=False, encoding="utf-8")
    print(f"[완료] 컴포넌트 존재 하지 않음 {len(df)}개")