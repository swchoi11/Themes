# 디폴트 이미지 & 이미지의 json / 테마 이미지 & 이미지의 json 비교
import json
from src.model import Layout, UIElement, Issue
from typing import Dict, List, Optional
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from PIL import Image

def json_parser(json_path: str) -> Layout:
    """JSON 파일을 파싱하여 Layout 객체로 변환"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return Layout.model_validate(data)

# 디폴트 이미지와 비교 분석
class Compare:
    def __init__(self):
        pass
    
    def analyze_layout(self, theme: Layout, default: Layout) -> List[LayoutIssue]:
        """레이아웃 분석 수행"""
        issues = []
        
        # 컴포넌트 매핑
        component_mapping = self._map_components(
            default.elements,
            theme.elements
        )
        
        

        # 각 검사 수행
        issues.extend(self._check_element_sort(theme, default, component_mapping))
        issues.extend(self._check_element_brightness(theme, default, component_mapping))
        issues.extend(self._check_element_interaction(theme, default, component_mapping))
        issues.extend(self._check_element_icon(theme, default, component_mapping))
        issues.extend(self._check_element_text(theme))
        issues.extend(self._check_element_calendar(theme))
        issues.extend(self._check_element_highlight(theme))

        return issues

    def _map_components(self,
                        default_elements,
                        themed_elements):
        """두 이미지 간 동일한 컴포넌트 매핑"""
        mapping = {}

        # 텍스트 기반 매핑 (컨텐츠가 동일하면 같은 컴포넌트로 간주)
        content_map = {}
        for elem in default_elements:
            if elem.content:
                content_key = f"{elem.type}_{elem.content}"
                content_map[content_key] = elem

        # 테마 적용된 요소들과 매핑
        for themed_elem in themed_elements:
            if themed_elem.content:
                content_key = f"{themed_elem.type}_{themed_elem.content}"
                if content_key in content_map:
                    default_elem = content_map[content_key]
                    mapping[themed_elem.id] = {
                        'default': default_elem,
                        'themed': themed_elem
                    }

        # 위치 기반 매핑 (컨텐츠가 없는 아이콘, 버튼 등)
        for themed_elem in themed_elements:
            if themed_elem.id not in mapping:
                best_match = None
                best_iou = 0

                for default_elem in default_elements:
                    if default_elem.type == themed_elem.type:
                        iou = self._calculate_iou(
                            default_elem.bbox, themed_elem.bbox
                        )
                        if iou > 0.5 and iou > best_iou:
                            best_match = default_elem
                            best_iou = iou

                if best_match:
                    mapping[themed_elem.id] = {
                        'default': best_match,
                        'themed': themed_elem,
                        'iou': best_iou
                    }

        return mapping
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """두 바운딩 박스의 IoU(Intersection over Union) 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 교집합 영역 계산
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # 합집합 영역 계산
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area


    def _check_element_sort(self, theme: Layout, default: Layout, component_mapping: Dict) -> List[LayoutIssue]:
        """요소 정렬 기준 일치 여부 확인"""
        issues = []
        
        for component_id, mapping in component_mapping.items():
            default_elem = mapping['default']
            themed_elem = mapping['themed']
            
            # 정렬 기준 비교 (예: 수직/수평 정렬, 간격 등)
            if default_elem.alignment != themed_elem.alignment:
                issues.append(LayoutIssue(
                    issue_type='sort',
                    component_id=component_id,
                    component_type=themed_elem.type,
                    severity=0.7,
                    bbox=themed_elem.bbox,
                    description=f"정렬 기준 불일치 (원본: {default_elem.alignment}, 테마: {themed_elem.alignment})"
                ))
        
        return issues

    # def _check_element_brightness(self, theme: Layout, default: Layout, component_mapping: Dict) -> List[LayoutIssue]:
    #     """텍스트/아이콘과 배경 간 명도 대비 확인"""
    #     issues = []
        
    #     for component_id, mapping in component_mapping.items():
    #         themed_elem = mapping['themed']
            
    #         # 대비 계산
    #         contrast = self._calculate_contrast(themed_elem.image_region)
            
    #         if contrast < self.visibility_threshold:
    #             issues.append(LayoutIssue(
    #                 issue_type='brightness',
    #                 component_id=component_id,
    #                 component_type=themed_elem.type,
    #                 severity=1.0 - (contrast / self.visibility_threshold),
    #                 bbox=themed_elem.bbox,
    #                 description=f"낮은 명도 대비 (대비값: {contrast:.2f})",
    #                 screenshot_region=themed_elem.image_region
    #             ))
        
    #     return issues

    # def _check_element_interaction(self, theme: Layout, default: Layout, component_mapping: Dict) -> List[LayoutIssue]:
    #     """상호작용 요소 구분 명확성 확인"""
    #     issues = []
        
    #     for component_id, mapping in component_mapping.items():
    #         default_elem = mapping['default']
    #         themed_elem = mapping['themed']
            
    #         if default_elem.is_interactive and themed_elem.is_interactive:
    #             # 상호작용 요소의 시각적 구분도 계산
    #             if themed_elem.visual_distinctiveness < 0.5:
    #                 issues.append(LayoutIssue(
    #                     issue_type='interaction',
    #                     component_id=component_id,
    #                     component_type=themed_elem.type,
    #                     severity=0.8,
    #                     bbox=themed_elem.bbox,
    #                     description="상호작용 요소의 시각적 구분이 모호함"
    #                 ))
        
    #     return issues

    # def _check_element_icon(self, theme: Layout, default: Layout, component_mapping: Dict) -> List[LayoutIssue]:
    #     """아이콘 중복 및 잘림 확인"""
    #     issues = []
    #     icon_map = {}
        
    #     for component_id, mapping in component_mapping.items():
    #         themed_elem = mapping['themed']
            
    #         if themed_elem.type == 'icon':
    #             # 아이콘 중복 검사
    #             icon_key = f"{themed_elem.content}_{themed_elem.visual_hash}"
    #             if icon_key in icon_map:
    #                 issues.append(LayoutIssue(
    #                     issue_type='icon_duplicate',
    #                     component_id=component_id,
    #                     component_type='icon',
    #                     severity=0.6,
    #                     bbox=themed_elem.bbox,
    #                     description=f"중복된 아이콘 사용 (ID: {icon_map[icon_key]})"
    #                 ))
    #             icon_map[icon_key] = component_id
                
    #             # 아이콘 잘림 검사
    #             if themed_elem.is_cropped:
    #                 issues.append(LayoutIssue(
    #                     issue_type='icon_crop',
    #                     component_id=component_id,
    #                     component_type='icon',
    #                     severity=0.7,
    #                     bbox=themed_elem.bbox,
    #                     description="아이콘이 잘려서 표시됨"
    #                 ))
        
    #     return issues

    # def _check_element_text(self, theme: Layout) -> List[LayoutIssue]:
    #     """텍스트 잘림 확인"""
    #     issues = []
        
    #     for elem in theme.elements:
    #         if elem.type == 'text' and elem.is_truncated:
    #             issues.append(LayoutIssue(
    #                 issue_type='text_truncate',
    #                 component_id=elem.id,
    #                 component_type='text',
    #                 severity=0.8,
    #                 bbox=elem.bbox,
    #                 description="텍스트가 영역을 초과하여 잘림"
    #             ))
        
    #     return issues

    # def _check_element_calendar(self, theme: Layout) -> List[LayoutIssue]:
    #     """달력 관련 요소 확인"""
    #     issues = []
        
    #     for elem in theme.elements:
    #         if elem.type == 'calendar':
    #             # 달력 요일 표시 확인
    #             if elem.has_overflowing_days:
    #                 issues.append(LayoutIssue(
    #                     issue_type='calendar_overflow',
    #                     component_id=elem.id,
    #                     component_type='calendar',
    #                     severity=0.7,
    #                     bbox=elem.bbox,
    #                     description="달력 요일이 테두리를 벗어남"
    #                 ))
                
    #             # 현재 날짜 표시 확인
    #             if not elem.matches_current_date:
    #                 issues.append(LayoutIssue(
    #                     issue_type='calendar_date',
    #                     component_id=elem.id,
    #                     component_type='calendar',
    #                     severity=0.5,
    #                     bbox=elem.bbox,
    #                     description="현재 날짜와 일치하지 않음"
    #                 ))
        
    #     return issues

    # def _check_element_highlight(self, theme: Layout) -> List[LayoutIssue]:
    #     """하이라이트 요소 확인"""
    #     issues = []
        
    #     for elem in theme.elements:
    #         if elem.is_highlighted:
    #             contrast = self._calculate_contrast(elem.image_region)
    #             if contrast < self.visibility_threshold:
    #                 issues.append(LayoutIssue(
    #                     issue_type='highlight_contrast',
    #                     component_id=elem.id,
    #                     component_type=elem.type,
    #                     severity=1.0 - (contrast / self.visibility_threshold),
    #                     bbox=elem.bbox,
    #                     description="하이라이트된 요소의 대비가 낮음",
    #                     screenshot_region=elem.image_region
    #                 ))
        
    #     return issues


# 이미지 자체의 결함 분석
class Alone(Issue):
    def __init__(self, theme_layout: Layout):
        self.layout = theme_layout
        
    def analyse_issue(self) -> Issue:
        pass
    
    def _calculate_contrast(self, img_region: np.ndarray) -> float:
        """이미지 영역의 대비 계산"""
        if img_region.size == 0:
            return 0.0

        if len(img_region.shape) == 3:
            gray = cv2.cvtColor(img_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_region

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        non_zero_vals = hist_norm[hist_norm > 0]
        entropy = -np.sum(non_zero_vals * np.log2(non_zero_vals))
        contrast = entropy / 8.0

        return contrast


if __name__ == "__main__":
    parser = LayoutAwareParser(config={
        'visibility_threshold': 0.3,
        'cut_off_threshold': 0.2,
        'design_threshold': 0.4
    })
    theme_layout = parser.json_parser('./resource/test.json')
    default_layout = parser.json_parser('./resource/test.json')
    
    issues = parser.analyze_layout(theme_layout.skeleton, default_layout)
    print(issues)
    for issue in issues:
        print(issue.issue_type)
        print(issue.description)

