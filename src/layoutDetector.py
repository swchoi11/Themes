import os
import glob

import json
import cv2

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
from collections import defaultdict

from utils.schemas import Issue
from common.eval_kpi import EvalKPI
from common.prompt import IssuePrompt
from src.gemini import Gemini
from src.xmlParser import XMLParser


class VisibilityDetector:
    """ Visibility 이슈(0, 1, 2) 검출 클래스"""

    def __init__(self, parent: 'LayoutDetector'):
        self.parent = parent
        self.base_visibility_threshold = parent.visibility_threshold
        self.gray_cache = {}

    def _get_grayscale(self, image: np.ndarray, image_path: str) -> np.ndarray:
        if image_path not in self.gray_cache:
            self.gray_cache[image_path] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.gray_cache[image_path]

    def detect_issues(self, elements: List[Dict], image: np.ndarray, ai_responses: Dict) -> List[Issue]:
        issues = []

        # 화면 전체 대비 분석
        adaptive_contrast_threshold = self._calculate_contrast_threshold(image)

        for element in elements:
            elem_type = element.get('type', '').lower()

            # 이슈 0: 텍스트/아이콘 대비 문제
            if elem_type in ['textview', 'imageview', 'text', 'icon']:
                contrast_ratio = self._calculate_wcag_contrast_ratio(element, image)
                required_ratio = adaptive_contrast_threshold if elem_type in ['text', 'textview'] else 3.0
                if contrast_ratio < required_ratio:
                    issue = self.parent._create_issue(element, 0, ai_responses.get('0', {}))
                    issues.append(issue)

            # 이슈 1: 하이라이트 요소 대비 문제
            if self._is_highlighted_element(element):
                highlight_ratio = self._calculate_highlight_contrast_ratio(element, image)
                if highlight_ratio < adaptive_contrast_threshold:
                    issue = self.parent._create_issue(element, 1, ai_responses.get('1', {}))
                    issues.append(issue)

            # 이슈 2: 상호작용 요소 시각적 구분성
            if element.get('clickable') and elem_type in ['button', 'imagebutton']:
                affordance_score = self._calculate_button_affordance(element, image)
                if affordance_score < 0.6:
                    issue = self.parent._create_issue(element, 2, ai_responses.get('2', {}))
                    issues.append(issue)

        return issues

    def _calculate_contrast_threshold(self, image: np.ndarray) -> float:
        """화면 전체의 평균 대비를 기반으로 적응적 임계값 계산"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            overall_contrast = np.std(gray) / 255.0
            # 전체 화면 대비가 낮으면 임계값을 낮춤
            adaptive_threshold = max(3.0, min(7.0, overall_contrast * 10))
            return adaptive_threshold
        except:
            return 4.5

    def _calculate_wcag_contrast_ratio(self, element: Dict, image: np.ndarray) -> float:
        """WCAG 표준에 따른 정확한 대비 비율 계산"""
        try:
            # 요소 영역 추출
            element_luminance = self._get_element_luminance(element, image)
            background_luminance = self._get_background_luminance(element, image)

            if element_luminance is None or background_luminance is None:
                return 1.0

            # WCAG 대비 비율 공식: (L1 + 0.05) / (L2 + 0.05)
            lighter = max(element_luminance, background_luminance)
            darker = min(element_luminance, background_luminance)

            contrast_ratio = (lighter + 0.05) / (darker + 0.05)
            return contrast_ratio

        except Exception:
            return 1.0

    def _get_element_luminance(self, element: Dict, image: np.ndarray) -> Optional[float]:
        """요소의 상대 밝기 계산"""
        try:
            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return None

            element_region = image[y1:y2, x1:x2]
            if element_region.size == 0:
                return None

            # RGB 평균값 계산
            avg_rgb = np.mean(element_region.reshape(-1, 3), axis=0)

            # WCAG 상대 밝기 계산
            return self._calculate_relative_luminance(avg_rgb)

        except (KeyError, TypeError, ValueError, IndexError) as e:
            print(f"Error calculating element luminance: {e}")
            return None

    def _get_background_luminance(self, element: Dict, image: np.ndarray) -> Optional[float]:
        """요소 주변 배경의 상대 밝기 계산"""
        try:
            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            background_margin = 10
            # 요소 주변의 배경 영역 정의
            bg_x1 = max(0, x1 - background_margin)
            bg_y1 = max(0, y1 - background_margin)
            bg_x2 = min(w, x2 + background_margin)
            bg_y2 = min(h, y2 + background_margin)

            # 배경 영역에서 요소 영역 제외
            background_region = image[bg_y1:bg_y2, bg_x1:bg_x2]

            if background_region.size == 0:
                return None

            # 배경 마스크 생성 (요소 영역 제외)
            mask = np.ones((bg_y2 - bg_y1, bg_x2 - bg_x1), dtype=bool)
            elem_start_x = max(0, x1 - bg_x1)
            elem_start_y = max(0, y1 - bg_y1)
            elem_end_x = min(mask.shape[1], x2 - bg_x1)
            elem_end_y = min(mask.shape[0], y2 - bg_y1)

            if (elem_end_x > elem_start_x and elem_end_y > elem_start_y):
                mask[elem_start_y:elem_end_y, elem_start_x:elem_end_x] = False

            # 배경 영역의 평균 RGB
            background_pixels = background_region.reshape(-1, 3)[mask.flatten()]

            if len(background_pixels) == 0:
                return None

            avg_bg_rgb = np.mean(background_pixels, axis=0)

            # WCAG 상대 밝기 계산
            return self._calculate_relative_luminance(avg_bg_rgb)

        except Exception:
            return None

    def _calculate_relative_luminance(self, rgb: np.ndarray) -> float:
        """WCAG 표준 상대 밝기 계산"""
        try:
            # RGB 값을 0-1 범위로 정규화
            rgb_normalized = rgb / 255.0

            # WCAG 공식 적용
            def linearize(component):
                if component <= 0.03928:
                    return component / 12.92
                else:
                    return ((component + 0.055) / 1.055) ** 2.4

            r_linear = linearize(rgb_normalized[0])
            g_linear = linearize(rgb_normalized[1])
            b_linear = linearize(rgb_normalized[2])

            # 가중 합계
            luminance = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

            return luminance

        except Exception:
            return 0.5

    def _calculate_highlight_contrast_ratio(self, element: Dict, image: np.ndarray) -> float:
        """하이라이트 요소의 대비 비율 계산"""
        base_ratio = self._calculate_wcag_contrast_ratio(element, image)

        try:
            # 시각적 특성으로 하이라이트 강도 추가 고려
            visual_features = element.get('visual_features', {})
            avg_color = visual_features.get('avg_color', [128, 128, 128])

            if len(avg_color) >= 3:
                r, g, b = avg_color[:3]
                max_val = max(r, g, b)
                min_val = min(r, g, b)
                saturation = (max_val - min_val) / max_val if max_val > 0 else 0

                # 채도가 높을수록 더 높은 대비 요구
                saturation_penalty = saturation * 0.5
                adjusted_ratio = base_ratio * (1 - saturation_penalty)
                return max(1.0, adjusted_ratio)

        except Exception:
            pass

        return base_ratio

    def _calculate_button_affordance(self, element: Dict, image: np.ndarray) -> float:
        """버튼의 어포던스(시각적 단서) 점수 계산"""
        try:
            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return 0.0

            element_region = image[y1:y2, x1:x2]
            if element_region.size == 0:
                return 0.0

            affordance_score = 0.0

            # 1. 경계선 검출
            border_score = self._detect_border_affordance(element_region)
            affordance_score += border_score * 0.3

            # 2. 그림자/깊이 효과 검출
            shadow_score = self._detect_shadow_effect(element_region)
            affordance_score += shadow_score * 0.3

            # 3. 배경과의 대비
            contrast_ratio = self._calculate_wcag_contrast_ratio(element, image)
            contrast_score = min(1.0, contrast_ratio / 4.5)  # 4.5 이상이면 만점
            affordance_score += contrast_score * 0.4

            return min(1.0, affordance_score)

        except Exception:
            return 0.0

    def _detect_border_affordance(self, element_region: np.ndarray) -> float:
        """테두리 검출"""
        try:
            gray = cv2.cvtColor(element_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            h, w = edges.shape
            if h < 4 or w < 4:
                return 0.0

            # 가장자리 픽셀 확인
            border_pixels = np.concatenate([
                edges[0, :],    # 상단
                edges[-1, :],   # 하단
                edges[:, 0],    # 좌측
                edges[:, -1]    # 우측
            ])

            border_ratio = np.sum(border_pixels > 0) / len(border_pixels)
            return min(1.0, border_ratio * 3)  # 30% 이상이면 만점

        except Exception:
            return 0.0

    def _detect_shadow_effect(self, element_region: np.ndarray) -> float:
        """그림자 효과 검출"""
        try:
            if element_region.shape[0] < 6 or element_region.shape[1] < 6:
                return 0.0

            gray = cv2.cvtColor(element_region, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # 영역을 9개영역 나누어 밝기 분석
            third_h = h // 3
            third_w = w // 3

            regions = {
                'top_left': gray[:third_h, :third_w],
                'top_center': gray[:third_h, third_w:2 * third_w],
                'top_right': gray[:third_h, 2 * third_w:],
                'center_left': gray[third_h:2 * third_h, :third_w],
                'center': gray[third_h:2 * third_h, third_w:2 * third_w],
                'center_right': gray[third_h:2 * third_h, 2 * third_w:],
                'bottom_left': gray[2 * third_h:, :third_w],
                'bottom_center': gray[2 * third_h:, third_w:2 * third_w],
                'bottom_right': gray[2 * third_h:, 2 * third_w:]
            }

            # 각 영역의 평균 밝기
            brightness = {k: np.mean(v) if v.size > 0 else 128
                          for k, v in regions.items()}

            # 그림자 패턴 검출 (좌상단이 밝고 우하단이 어두운 패턴)
            shadow_indicators = [
                brightness['top_left'] - brightness['bottom_right'],
                brightness['top_center'] - brightness['bottom_center'],
                brightness['center_left'] - brightness['center_right'],
                brightness['center'] - brightness['bottom_right']
            ]

            shadow_score = np.mean([max(0, indicator) for indicator in shadow_indicators]) / 255
            return min(1.0, shadow_score * 2)

        except Exception:
            return 0.0

    def _is_highlighted_element(self, element: Dict) -> bool:
        """하이라이트 판별"""
        # 키워드 검사를 먼저 수행 (빠른 종료)
        resource_id = element.get('resource_id', '').lower()
        content = element.get('content', '').lower()
        highlight_keywords = ['selected', 'highlight', 'focus', 'active', 'current']

        if any(keyword in resource_id or keyword in content for keyword in highlight_keywords):
            return True

        visual_features = element.get('visual_features', {})
        avg_color = visual_features.get('avg_color', [128, 128, 128])

        if len(avg_color) >= 3:
            r, g, b = avg_color[:3]
            max_val = max(r, g, b)
            if max_val > 0:
                min_val = min(r, g, b)
                saturation = (max_val - min_val) / max_val
                brightness = max_val / 255.0
                return saturation > 0.6 and brightness > 0.4

        return False


class AlignmentDetector:
    """Alignment 이슈(3, 4, 5) 검출 클래스"""

    def __init__(self, parent: 'LayoutDetector'):
        self.parent = parent
        self.base_alignment_threshold = parent.alignment_threshold

    def detect_issues(self, elements: List[Dict], ai_responses: Dict) -> List[Issue]:
        issues = []
        if len(elements) < 3:  # 최소 3개 이상 필요
            return issues

        layout_patterns = self._detect_layout_patterns(elements)
        non_pattern_elements = self._filter_pattern_elements(elements, layout_patterns)
        grouped_elements = self._group_elements_by_context(non_pattern_elements)

        for group_key, group_elements in grouped_elements.items():
            if len(group_elements) < 3:
                continue

            dynamic_threshold = self._calculate_alignment_threshold(group_elements)

            # 이슈 3: 일관된 정렬 기준 위반
            alignment_issues = self._check_contextual_alignment(group_elements, dynamic_threshold)
            for elem in alignment_issues:
                issue = self.parent._create_issue(elem, 3, ai_responses.get('3', {}))
                issues.append(issue)

            # 이슈 4: 수직/수평 정렬 불일치
            vertical_issues = self._check_vertical_alignment(group_elements, dynamic_threshold)
            for elem in vertical_issues:
                issue = self.parent._create_issue(elem, 4, ai_responses.get('4', {}))
                issues.append(issue)

            # 이슈 5: 동일 계층 요소 정렬 기준 불일치
            reference_issues = self._check_hierarchical_alignment(group_elements, dynamic_threshold)
            for elem in reference_issues:
                issue = self.parent._create_issue(elem, 5, ai_responses.get('5', {}))
                issues.append(issue)

        return issues

    def _detect_layout_patterns(self, elements: List[Dict]) -> Dict[str, List[List[Dict]]]:
        """레이아웃 패턴 감지"""
        patterns = {
            'grid': [],
            'list': [],
            'navigation': [],
            'staggered': []
        }

        # 그리드 패턴 감지
        grid_groups = self._find_grid_patterns(elements)
        patterns['grid'] = grid_groups

        # 리스트 패턴 감지 (수직 정렬)
        list_groups = self._find_list_patterns(elements)
        patterns['list'] = list_groups

        # 네비게이션 패턴 감지 (수평 정렬)
        nav_groups = self._find_navigation_patterns(elements)
        patterns['navigation'] = nav_groups

        return patterns

    def _find_grid_patterns(self, elements: List[Dict]) -> List[List[Dict]]:
        """실제 그리드 패턴 감지"""
        grid_groups = []
        size_tolerance = 0.02

        # 비슷한 크기의 요소들을 그룹화
        size_groups = defaultdict(list)
        for elem in elements:
            w = elem['bbox'][2] - elem['bbox'][0]
            h = elem['bbox'][3] - elem['bbox'][1]
            size_key = (round(w, 2), round(h, 2))
            size_groups[size_key].append(elem)

        for size_key, candidates in size_groups.items():
            if len(candidates) >= 4:  # 최소 2x2 그리드
                if self._verify_grid_alignment(candidates):
                    grid_groups.append(candidates)

        return grid_groups

    def _verify_grid_alignment(self, candidates: List[Dict]) -> bool:
        """실제 그리드 정렬 검증"""
        # X, Y 좌표 수집
        x_coords = sorted(set(round(elem['bbox'][0], 2) for elem in candidates))
        y_coords = sorted(set(round(elem['bbox'][1], 2) for elem in candidates))

        # 최소 2x2 그리드인지 확인
        if len(x_coords) < 2 or len(y_coords) < 2:
            return False

        # 그리드 포인트에 실제 요소가 있는지 확인
        grid_positions = set((x, y) for x in x_coords for y in y_coords)
        element_positions = set((round(elem['bbox'][0], 2), round(elem['bbox'][1], 2))
                                for elem in candidates)

        # 대부분의 그리드 포인트에 요소가 있으면 그리드로 판단
        matches = len(element_positions.intersection(grid_positions))
        return matches >= len(candidates) * 0.7

    def _find_list_patterns(self, elements: List[Dict]) -> List[List[Dict]]:
        """리스트 패턴 감지 (수직으로 정렬된 유사 요소들)"""
        list_groups = []

        # 유사한 타입과 크기의 요소들 그룹화
        type_size_groups = defaultdict(list)
        for elem in elements:
            elem_type = elem.get('type', 'unknown')
            w = round(elem['bbox'][2] - elem['bbox'][0], 2)
            h = round(elem['bbox'][3] - elem['bbox'][1], 2)
            key = (elem_type, w, h)
            type_size_groups[key].append(elem)

        for group in type_size_groups.values():
            if len(group) >= 3:
                # Y 좌표로 정렬
                sorted_group = sorted(group, key=lambda e: e['bbox'][1])

                # 일정한 수직 간격인지 확인
                if self._verify_list_alignment(sorted_group):
                    list_groups.append(sorted_group)

        return list_groups

    def _verify_list_alignment(self, sorted_group: List[Dict]) -> bool:
        """리스트 정렬 검증"""
        if len(sorted_group) < 3:
            return False

        gaps = []
        for i in range(1, len(sorted_group)):
            gap = sorted_group[i]['bbox'][1] - sorted_group[i - 1]['bbox'][3]
            gaps.append(gap)

        if gaps:
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            # 간격 편차가 평균의 30% 이내면 리스트로 판단
            return std_gap < avg_gap * 0.3

        return False

    def _find_navigation_patterns(self, elements: List[Dict]) -> List[List[Dict]]:
        """네비게이션 패턴 감지"""
        nav_groups = []

        # 인터랙티브 요소들만 필터링
        interactive_elements = [e for e in elements
                                if e.get('clickable') or e.get('type', '').lower() in ['button', 'imagebutton']]

        # Y 좌표가 비슷한 요소들 그룹화
        y_groups = defaultdict(list)
        for elem in interactive_elements:
            y_center = (elem['bbox'][1] + elem['bbox'][3]) / 2
            y_key = round(y_center, 2)
            y_groups[y_key].append(elem)

        for group in y_groups.values():
            if len(group) >= 3:
                # X 좌표로 정렬
                sorted_group = sorted(group, key=lambda e: e['bbox'][0])

                # 수평 정렬 검증
                if self._verify_navigation_alignment(sorted_group):
                    nav_groups.append(sorted_group)

        return nav_groups

    def _verify_navigation_alignment(self, sorted_group: List[Dict]) -> bool:
        """네비게이션 정렬 검증"""
        if len(sorted_group) < 3:
            return False

        gaps = []
        for i in range(1, len(sorted_group)):
            gap = sorted_group[i]['bbox'][0] - sorted_group[i - 1]['bbox'][2]
            gaps.append(gap)

        if gaps:
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            # 간격 편차가 평균의 50% 이내면 네비게이션으로 판단
            return std_gap < avg_gap * 0.5

        return False

    def _filter_pattern_elements(self, elements: List[Dict], patterns: Dict) -> List[Dict]:
        """패턴에 속하는 요소들 제외"""
        pattern_element_ids = set()

        for pattern_list in patterns.values():
            for group in pattern_list:
                for elem in group:
                    pattern_element_ids.add(elem.get('id', ''))

        return [elem for elem in elements if elem.get('id', '') not in pattern_element_ids]

    def _group_elements_by_context(self, elements: List[Dict]) -> Dict[str, List[Dict]]:
        """컨텍스트 고려한 그룹화"""
        groups = {}

        for element in elements:
            elem_type = element.get('type', 'unknown')

            # 위치 기반 컨텍스트 추가
            bbox = element['bbox']
            center_y = (bbox[1] + bbox[3]) / 2
            center_x = (bbox[0] + bbox[2]) / 2

            # 화면 영역 분류
            if center_y < 0.15:
                region = 'header'
            elif center_y > 0.85:
                region = 'footer'
            elif center_x < 0.2:
                region = 'sidebar'
            else:
                region = 'content'

            group_key = f"{elem_type}_{region}"

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(element)

        return groups

    def _calculate_alignment_threshold(self, elements: List[Dict]) -> float:
        """요소 크기와 분포를 고려한 임계값 계산"""
        if not elements:
            return self.base_alignment_threshold

        # 요소들의 평균 크기 계산
        avg_width = np.mean([elem['bbox'][2] - elem['bbox'][0] for elem in elements])
        avg_height = np.mean([elem['bbox'][3] - elem['bbox'][1] for elem in elements])

        # 크기에 비례한 임계값 설정(평균 크기의 5%)
        size_based_threshold = min(avg_width, avg_height) * 0.05

        # 요소 개수에 따른 조정
        count_factor = max(0.5, 1.0 - len(elements) * 0.05)

        dynamic_threshold = size_based_threshold * count_factor

        # 최소/최대값 제한
        return max(0.005, min(0.05, dynamic_threshold))

    def _check_contextual_alignment(self, elements: List[Dict], threshold: float) -> List[Dict]:
        """컨텍스트 고려한 정렬 검사"""
        if len(elements) < 3:  # 최소 3개 이상일 때만 검사
            return []

        problematic_elements = []

        # 좌측 정렬 검사
        left_positions = [elem['bbox'][0] for elem in elements]
        left_std = np.std(left_positions)

        if left_std > threshold:
            # 정렬되지 않은 요소들 찾기
            mean_left = np.mean(left_positions)
            for element in elements:
                deviation = abs(element['bbox'][0] - mean_left)
                if deviation > threshold:
                    if not self._is_indentation(element, elements):
                        problematic_elements.append(element)

        return problematic_elements

    def _is_indentation(self, element: Dict, group: List[Dict]) -> bool:
        """들여쓰기 판별"""
        # 계층 구조 확인
        parent_id = element.get('parent_id')
        if parent_id:
            # 부모와 다른 들여쓰기 레벨이면 의도적일 가능성
            siblings = [e for e in group if e.get('parent_id') == parent_id]
            if len(siblings) >= 2:
                sibling_positions = [e['bbox'][0] for e in siblings]
                if np.std(sibling_positions) < 0.01:  # 형제들은 같은 레벨
                    return True

        # 리스트 아이템 스타일 확인
        content = element.get('content', '').strip()
        if content and (content.startswith('•') or content.startswith('-') or
                        content.startswith('*') or content[0].isdigit()):
            return True

        return False

    def _check_vertical_alignment(self, elements: List[Dict], threshold: float) -> List[Dict]:
        """수직 정렬 검사"""
        if len(elements) < 3:
            return []

        problematic_elements = []

        # 수직 중심점들 계산
        vertical_centers = [(elem['bbox'][1] + elem['bbox'][3]) / 2 for elem in elements]

        # 요소들이 한 줄에 배치 되어야 하는지 확인
        y_range = max(vertical_centers) - min(vertical_centers)
        avg_height = np.mean([elem['bbox'][3] - elem['bbox'][1] for elem in elements])

        # Y 범위가 평균 높이보다 작으면 같은 줄로 간주
        if y_range < avg_height * 0.5:
            vertical_std = np.std(vertical_centers)

            if vertical_std > threshold:
                mean_center = np.mean(vertical_centers)
                for i, element in enumerate(elements):
                    deviation = abs(vertical_centers[i] - mean_center)
                    if deviation > threshold:

                        elem_height = element['bbox'][3] - element['bbox'][1]
                        if not self._is_vertical_deviation(element, elements, elem_height):
                            problematic_elements.append(element)

        return problematic_elements

    def _is_vertical_deviation(self, element: Dict, group: List[Dict], elem_height: float) -> bool:
        """수직 편차 확인"""
        other_heights = [e['bbox'][3] - e['bbox'][1] for e in group if e != element]
        if other_heights:
            avg_other_height = np.mean(other_heights)
            if abs(elem_height - avg_other_height) > avg_other_height * 0.3:
                return True

        return False

    def _check_hierarchical_alignment(self, elements: List[Dict], threshold: float) -> List[Dict]:
        """계층 구조를 고려한 정렬 검사"""
        problematic_elements = []

        # 부모별로 그룹화
        parent_groups = defaultdict(list)
        for element in elements:
            parent_id = element.get('parent_id', 'root')
            parent_groups[parent_id].append(element)

        # 각 부모 그룹 내에서 정렬 검사
        for parent_id, group in parent_groups.items():
            if len(group) >= 3:  # 최소 3개 이상
                start_positions = [elem['bbox'][0] for elem in group]
                start_std = np.std(start_positions)

                if start_std > threshold:
                    mean_start = np.mean(start_positions)
                    for element in group:
                        deviation = abs(element['bbox'][0] - mean_start)
                        if deviation > threshold:
                            problematic_elements.append(element)

        return problematic_elements


class CutoffDetector:
    """ Cut Off 이슈(6, 7) 검출 클래스"""

    def __init__(self, parent: 'LayoutDetector'):
        self.parent = parent
        self.crop_threshold = parent.crop_threshold
        self.xml_parser = None
        self.icon_components = None

    def detect_issues(self, elements: List[Dict], image: np.ndarray, ai_responses: Dict) -> List[Issue]:
        # XMLParser 초기화
        if self.xml_parser is None:
            if self.parent.xml_path is None or self.parent.image_path is None:
                print(f"Error: xml_path or image_path is None in CutoffDetector")
                return []
            self.xml_parser = XMLParser(image_path=self.parent.image_path, xml_path=self.parent.xml_path)
            self.icon_components = {comp.id for comp in self.xml_parser.get_icon_components()}

        issues = []
        for element in elements:
            elem_type = element.get('type', '').lower()

            # 이슈 6: 텍스트 잘림 검출
            if elem_type in ['textview', 'text'] and element.get('content'):
                if self._detect_actual_text_truncation(element, image):
                    issue = self.parent._create_issue(element, 6, ai_responses.get('6', {}))
                    issues.append(issue)

            # 이슈 7: 아이콘 잘림 검출
            if elem_type in ['imageview', 'icon'] and element['id'] in self.icon_components:
                if self._is_icon_cropped(element, image):
                    issue = self.parent._create_issue(element, 7, ai_responses.get('7', {}))
                    issues.append(issue)

        return issues

    def _detect_actual_text_truncation(self, element: Dict, image: np.ndarray) -> bool:
        """실제 텍스트 렌더링 통한 잘림 검출"""
        try:
            content = element.get('content', '')
            if not content:
                return False

            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 >= x2 or y1 >= y2:
                return False

            text_region = image[y1:y2, x1:x2]
            if text_region.size == 0:
                return False

            # 방법 1: 텍스트 영역의 가장자리에서 텍스트 픽셀 검출
            text_at_edges = self._check_text_at_edges(text_region)
            if text_at_edges:
                return True

            # 방법 2: 개선된 텍스트 너비 추정
            container_width = bbox[2] - bbox[0]
            container_height = bbox[3] - bbox[1]

            # 컨테이너 높이를 기반으로 폰트 크기 추정
            estimated_font_size = container_height * 0.8  # 컨테이너 높이의 80%
            estimated_char_width = estimated_font_size * 0.6  # 일반적인 폰트 비율
            estimated_text_width = len(content) * estimated_char_width

            # 실제 컨테이너 너비와 비교
            width_ratio = estimated_text_width / (container_width * w)

            # 추정 너비가 실제 너비보다 15% 이상 크면 잘림으로 판단
            return width_ratio > 1.15

        except Exception:
            return False

    def _check_text_at_edges(self, text_region: np.ndarray) -> bool:
        """텍스트 영역의 가장자리에 텍스트 존재 여부 확인"""
        try:
            # 이미지를 그레이스케일로 변환
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

            # 적응적 이진화 (다양한 배경에 대응)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

            h, w = binary.shape
            if h < 8 or w < 8:
                return False

            # 가장자리 영역 정의
            margin = 3

            # 좌우 가장자리
            left_edge = binary[:, :margin]
            right_edge = binary[:, -margin:]

            # 각 가장자리에서 텍스트(어두운) 픽셀 비율 계산
            left_text_ratio = np.sum(left_edge < 128) / left_edge.size
            right_text_ratio = np.sum(right_edge < 128) / right_edge.size

            # 가장자리에 일정 비율 이상의 텍스트 픽셀이 있으면 잘림으로 판단
            threshold = 0.15  # 15% 이상
            return left_text_ratio > threshold or right_text_ratio > threshold

        except Exception:
            return False

    def _is_icon_cropped(self, element: Dict, image: np.ndarray) -> bool:
        """아이콘 잘림 검출"""
        try:
            bbox = element['bbox']

            # 방법 1: 화면 경계와의 거리 확인
            margins = [bbox[0], bbox[1], 1.0 - bbox[2], 1.0 - bbox[3]]
            min_margin = min(margins)

            # 화면 가장자리에 가까우면 잘림으로 판단
            if min_margin < self.crop_threshold:
                return True

            # 방법 2: 아이콘 영역의 완전성 검사
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 >= x2 or y1 >= y2:
                return False

            icon_region = image[y1:y2, x1:x2]
            if icon_region.size == 0:
                return False

            # 아이콘의 완전성 점수 계산
            completeness_score = self._calculate_icon_completeness(icon_region)

            # 완전성 점수가 낮으면 잘림으로 판단
            return completeness_score < 0.7

        except Exception:
            return False

    def _calculate_icon_completeness(self, icon_region: np.ndarray) -> float:
        """아이콘 완전성 점수 계산"""
        try:
            gray = cv2.cvtColor(icon_region, cv2.COLOR_BGR2GRAY)

            # 경계선 검출
            edges = cv2.Canny(gray, 50, 150)

            h, w = edges.shape
            if h < 10 or w < 10:
                return 1.0  # 너무 작은 영역은 완전한 것으로 간주

            # 가장자리에서의 경계선 밀도
            edge_margin = 2

            # 각 변의 경계선 밀도 계산
            top_edge_density = np.mean(edges[:edge_margin, :]) / 255.0
            bottom_edge_density = np.mean(edges[-edge_margin:, :]) / 255.0
            left_edge_density = np.mean(edges[:, :edge_margin]) / 255.0
            right_edge_density = np.mean(edges[:, -edge_margin:]) / 255.0

            # 중앙 영역의 경계선 밀도
            center_edges = edges[edge_margin:-edge_margin, edge_margin:-edge_margin]
            center_edge_density = np.mean(center_edges) / 255.0 if center_edges.size > 0 else 0

            # 가장자리 경계선 밀도가 높으면 잘림 가능성
            edge_densities = [top_edge_density, bottom_edge_density, left_edge_density, right_edge_density]
            max_edge_density = max(edge_densities)
            avg_edge_density = np.mean(edge_densities)

            # 대칭성 검사 (잘리지 않은 아이콘은 대체로 대칭적)
            symmetry_score = self._calculate_symmetry_score(gray)

            # 종합 완전성 점수
            edge_penalty = max_edge_density * 0.4 + avg_edge_density * 0.3
            completeness_score = (1.0 - edge_penalty) * 0.7 + symmetry_score * 0.3

            return max(0.0, min(1.0, completeness_score))

        except Exception:
            return 1.0

    def _calculate_symmetry_score(self, gray_region: np.ndarray) -> float:
        """아이콘의 대칭성 점수 계산"""
        try:
            h, w = gray_region.shape

            # 수직 대칭성 (좌우)
            left_half = gray_region[:, :w // 2]
            right_half = gray_region[:, w // 2:]
            right_half_flipped = np.fliplr(right_half)

            # 크기 맞추기
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]

            if left_half.size > 0 and right_half_flipped.size > 0:
                vertical_symmetry = 1.0 - np.mean(
                    np.abs(left_half.astype(float) - right_half_flipped.astype(float))) / 255.0
            else:
                vertical_symmetry = 0.5

            # 수평 대칭성 (상하)
            top_half = gray_region[:h // 2, :]
            bottom_half = gray_region[h // 2:, :]
            bottom_half_flipped = np.flipud(bottom_half)

            # 크기 맞추기
            min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
            top_half = top_half[:min_height, :]
            bottom_half_flipped = bottom_half_flipped[:min_height, :]

            if top_half.size > 0 and bottom_half_flipped.size > 0:
                horizontal_symmetry = 1.0 - np.mean(
                    np.abs(top_half.astype(float) - bottom_half_flipped.astype(float))) / 255.0
            else:
                horizontal_symmetry = 0.5

            # 평균 대칭성 점수
            symmetry_score = (vertical_symmetry + horizontal_symmetry) / 2
            return max(0.0, min(1.0, symmetry_score))

        except Exception:
            return 0.5


class LayoutDetector(Gemini):
    """ layoutParser.py -> 레이아웃 이슈 검출기"""

    def __init__(self, output_dir):
        super().__init__()

        self.contrast_threshold = 4.5
        self.visibility_threshold = 0.6
        self.alignment_threshold = 0.02
        self.crop_threshold = 0.02
        self.max_issues_per_type = 3
        self.image_path = None
        self.xml_path = None

        self.debug = True
        self.output_dir = output_dir
        if self.debug:
            os.makedirs(self.output_dir, exist_ok=True)

        self.prompts = {k: v for k, v in IssuePrompt().items()
                        if k in ['0', '1', '2', '3', '4', '5', '6', '7']}

        self.issue_descriptions = EvalKPI.DESCRIPTION

        self.visibility_detector = VisibilityDetector(self)
        self.alignment_detector = AlignmentDetector(self)
        self.cutoff_detector = CutoffDetector(self)

    def analyze_layout(self, image_path: str, xml_path: str, json_path: str) -> List[Issue]:
        try:
            if not all([image_path, xml_path, json_path]):
                raise ValueError("image_path, xml_path, and json_path must not be None")

            self.image_path = image_path
            self.xml_path = xml_path
            self._validate_files(image_path, xml_path, json_path)
            json_data = self._load_json(json_path)

            xml_parser = XMLParser(image_path=image_path, xml_path=xml_path)
            xml_components = xml_parser.get_components()

            json_elements = json_data.get('skeleton', {}).get('elements', [])
            if not xml_components and json_elements:
                print("XML 요소가 없어서 JSON 요소를 사용합니다.")
                merged_elements = self._convert_json_to_xml_format(json_elements)
            else:
                merged_components = xml_parser.merge_with_elements(xml_components, json_elements, element_type='dict')
                merged_elements = [asdict(comp) for comp in merged_components]

            if not merged_elements:
                print("분석할 UI 요소가 없습니다.")
                return []

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

            ai_responses = self._get_ai_responses_safe(image_path)

            all_issues = []

            # 화면 분석 정보 수집
            screen_info = self._analyze_screen_characteristics(image, merged_elements)
            print(f"화면 특성 분석 완료: 복잡도={screen_info['complexity']:.2f}")

            # 개선된 검출기들로 이슈 검출
            visibility_issues = self.visibility_detector.detect_issues(merged_elements, image, ai_responses)
            all_issues.extend(visibility_issues)
            print(f"Visibility 이슈 {len(visibility_issues)}개 검출")

            alignment_issues = self.alignment_detector.detect_issues(merged_elements, ai_responses)
            all_issues.extend(alignment_issues)
            print(f"Alignment 이슈 {len(alignment_issues)}개 검출")

            cutoff_issues = self.cutoff_detector.detect_issues(merged_elements, image, ai_responses)
            all_issues.extend(cutoff_issues)
            print(f"Cut Off 이슈 {len(cutoff_issues)}개 검출")

            # 이슈 우선 순위 및 필터링
            final_issues = self._prioritize_and_filter_issues(all_issues, screen_info)
            print(f"최종 {len(final_issues)}개의 이슈 선별")

            if self.debug:
                self.verify_bboxes(final_issues, xml_components, json_elements)
                self.visualize_bboxes(image_path, final_issues, suffix="issues")

            return final_issues

        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _analyze_screen_characteristics(self, image: np.ndarray, elements: List[Dict]) -> Dict:
        """화면 특성 분석"""
        h, w = image.shape[:2]

        # 요소 밀도 계산
        total_element_area = sum((e['bbox'][2] - e['bbox'][0]) * (e['bbox'][3] - e['bbox'][1]) for e in elements)
        density = total_element_area

        # 복잡도 계산
        edge_complexity = self._calculate_edge_complexity(image)
        element_complexity = len(set(e.get('type', 'unknown') for e in elements)) / 10.0

        complexity = min(1.0, (density + edge_complexity + element_complexity) / 3)

        return {
            'density': density,
            'complexity': complexity,
            'element_count': len(elements),
            'screen_size': (w, h)
        }

    def _calculate_edge_complexity(self, image: np.ndarray) -> float:
        """화면의 엣지 복잡도 계산"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return min(1.0, edge_density * 5)  # 정규화
        except:
            return 0.5

    def _prioritize_and_filter_issues(self, issues: List[Issue], screen_info: Dict) -> List[Issue]:
        """화면 특성을 고려한 이슈 우선순위 및 필터링"""

        complexity = screen_info['complexity']
        if complexity > 0.7:
            max_issues_per_type = 2
        elif complexity < 0.3:
            max_issues_per_type = 4
        else:
            max_issues_per_type = self.max_issues_per_type

        # 이슈 타입별 그룹화
        issues_by_type = defaultdict(list)
        for issue in issues:
            issues_by_type[issue.issue_type].append(issue)

        final_issues = []
        severity_order = {'high': 3, 'medium': 2, 'low': 1}

        for issue_type, type_issues in issues_by_type.items():
            # 심각도와 위치를 고려한 정렬
            sorted_issues = sorted(
                type_issues,
                key=lambda x: (
                    severity_order.get(x.severity, 0),
                    -self._calculate_visibility_weight(x.bbox),  # 더 보이는 위치 우선
                    x.bbox[0]  # 좌측 우선
                ),
                reverse=True
            )

            # 중복 제거 및 개수 제한
            filtered_issues = self._remove_duplicate_issues(sorted_issues[:max_issues_per_type])
            final_issues.extend(filtered_issues)

        return final_issues

    def _calculate_visibility_weight(self, bbox: List[float]) -> float:
        """요소 위치의 시각적 중요도 계산"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # 중앙에 가까울수록, 상단에 가까울수록 높은 가중치
        x_weight = 1.0 - abs(center_x - 0.5) * 2
        y_weight = 1.0 - center_y * 0.5  # 상단 선호

        return (x_weight + y_weight) / 2

    def _remove_duplicate_issues(self, issues: List[Issue]) -> List[Issue]:
        """중복 되거나 너무 가까운 이슈들 제거"""
        if len(issues) <= 1:
            return issues

        filtered_issues = [issues[0]]

        for issue in issues[1:]:
            is_duplicate = False
            for existing in filtered_issues:
                if issue.component_id == existing.component_id:
                    is_duplicate = True
                    break
                if self._calculate_bbox_distance(issue.bbox, existing.bbox) < 0.05:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_issues.append(issue)

        return filtered_issues

    def _calculate_bbox_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """두 바운딩 박스 간의 거리 계산"""
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        return distance

    # 기존 메서드들 (변경 없음)
    def verify_bboxes(self, issues: List[Issue], xml_elements: List[Dict], json_elements: List[Dict]):
        """Verify that issue bounding boxes match XML or JSON bounding boxes"""
        if not self.debug:
            return

        epsilon = 0.001
        for issue in issues:
            issue_bbox = issue.bbox
            component_id = issue.component_id
            matched = False

            # Check XML elements
            for xml_elem in xml_elements:
                if xml_elem.id == component_id and xml_elem.bbox:
                    if all(abs(a - b) < epsilon for a, b in zip(issue_bbox, xml_elem.bbox)):
                        print(f"Match found in XML for component {component_id}: {issue_bbox}")
                        matched = True
                        break

            # Check JSON elements if no match in XML
            if not matched:
                for json_elem in json_elements:
                    json_bbox = json_elem.get('bbox')
                    elem_id = json_elem.get('id', f"json_element_{json_elements.index(json_elem)}")
                    if elem_id == component_id and json_bbox:
                        if all(abs(a - b) < epsilon for a, b in zip(issue_bbox, json_bbox)):
                            print(f"Match found in JSON for component {component_id}: {issue_bbox}")
                            matched = True
                            break

            if not matched:
                print(f"No match found for component {component_id}: {issue_bbox}")

    def visualize_bboxes(self, image_path: str, issues: List[Issue], suffix: str = ""):
        """Visualize bounding boxes of detected issues on the image."""
        if not self.debug:
            return

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot load image for visualization: {image_path}")
                return

            h, w = image.shape[:2]
            colors = {
                '0': (255, 0, 0),  # 빨강 - 텍스트 대비
                '1': (255, 165, 0),  # 주황 - 하이라이트 대비
                '2': (255, 255, 0),  # 노랑 - 버튼 시각성
                '3': (0, 255, 0),  # 초록 - 정렬 일관성
                '4': (0, 255, 255),  # 청록 - 수직 정렬
                '5': (0, 0, 255),  # 파랑 - 계층 정렬
                '6': (255, 0, 255),  # 자홍 - 텍스트 잘림
                '7': (128, 0, 128),  # 보라 - 아이콘 잘림
            }

            for issue in issues:
                bbox = issue.bbox
                x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

                # 이슈 타입별 색상으로 사각형 그리기
                color = colors.get(issue.issue_type, (128, 128, 128))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # 라벨 추가
                label = f"Issue {issue.issue_type} ({issue.severity})"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 디버그 이미지 저장
            output_path = os.path.join(self.output_dir, f"debug_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, image)
            print(f"Debug image saved: {output_path}")

        except Exception as e:
            print(f"Error visualizing bounding boxes: {str(e)}")

    def _validate_files(self, image_path: str, xml_path: str, json_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML 파일이 없습니다: {xml_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON 파일이 없습니다: {json_path}")

    def _load_json(self, json_path: str) -> Dict:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파일 형식 오류: {json_path}, {str(e)}")

    def _convert_json_to_xml_format(self, json_elements: List[Dict]) -> List[Dict]:
        elements = []
        for i, json_elem in enumerate(json_elements):
            if 'bbox' not in json_elem:
                continue
            element = {
                'id': f"json_element_{i}",
                'type': json_elem.get('type', 'unknown'),
                'bbox': json_elem['bbox'],
                'content': json_elem.get('content', ''),
                'content_desc': json_elem.get('content_desc', ''),
                'resource_id': json_elem.get('resource_id', ''),
                'clickable': json_elem.get('clickable', False),
                'parent_id': None,
                'children': [],
                'visual_features': json_elem.get('visual_features', {}),
                'source': 'json'
            }
            elements.append(element)
        return elements

    def _get_ai_responses_safe(self, image_path: str) -> Dict:
        responses = {}
        for issue_type in range(8):
            issue_type_str = str(issue_type)
            try:
                if hasattr(self, 'gemini') and issue_type_str in self.prompts:
                    response = dict(self.generate_response(self.prompts[issue_type_str], image_path))
                    responses[issue_type_str] = response
                    print(f"AI 응답 수집 완료: 이슈 {issue_type}")
                else:
                    responses[issue_type_str] = {
                        'severity': 'medium',
                        'text': f'이슈 {issue_type}: {self.issue_descriptions[str(issue_type)]}'
                    }
            except Exception as e:
                print(f"AI 응답 생성 오류 (이슈 {issue_type}): {str(e)}")
                responses[issue_type_str] = {
                    'severity': 'medium',
                    'text': f'기본 분석: {self.issue_descriptions[str(issue_type)]}'
                }
        return responses

    def _create_issue(self, element: Dict, issue_type: int, ai_response: Dict) -> Issue:
        elem_type = element.get('type', 'unknown')
        bbox = element.get('bbox', [0, 0, 0, 0])
        location_id, location_type = self._calc_location(bbox)
        ui_component_id, ui_component_type = self._calc_ui_component(elem_type)
        severity = self._map_severity(ai_response.get('severity', 'medium'))
        ai_description = ai_response.get('text', self.issue_descriptions.get(str(issue_type), '이슈 해결 필요'))
        return Issue(
            issue_type=str(issue_type),
            component_id=element.get('id', 'unknown'),
            component_type=elem_type,
            ui_component_id=ui_component_id,
            ui_component_type=ui_component_type,
            severity=severity,
            location_id=location_id,
            location_type=location_type,
            bbox=bbox,
            description_id=str(issue_type),
            description_type=self.issue_descriptions[str(issue_type)],
            ai_description=ai_description
        )

    def _calc_location(self, bbox: List[float]) -> Tuple[str, str]:
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        if center_y < 0.33:
            if center_x < 0.33:
                location_id = '0'  # TL
            elif center_x < 0.67:
                location_id = '1'  # TC
            else:
                location_id = '2'  # TR
        elif center_y < 0.67:
            if center_x < 0.33:
                location_id = '3'  # ML
            elif center_x < 0.67:
                location_id = '4'  # MC
            else:
                location_id = '5'  # MR
        else:
            if center_x < 0.33:
                location_id = '6'  # BL
            elif center_x < 0.67:
                location_id = '7'  # BC
            else:
                location_id = '8'  # BR
        location_type = EvalKPI.LOCATION.get(location_id, 'unknown')
        return location_id, location_type

    def _calc_ui_component(self, element_type: str) -> Tuple[str, str]:
        ui_component_id = EvalKPI.UI_COMPONENT.get(element_type, '-1')
        ui_component_type = EvalKPI.UI_COMPONENT.get(ui_component_id, 'unknown')
        return ui_component_id, ui_component_type

    def _map_severity(self, severity_input) -> str:
        if isinstance(severity_input, str):
            if severity_input.isdigit():
                severity_input = float(severity_input)
            elif severity_input.lower() in ['high', 'medium', 'low']:
                return severity_input.lower()
            else:
                return 'medium'
        if isinstance(severity_input, (int, float)):
            if severity_input >= 0.8 or severity_input >= 3:
                return 'high'
            elif severity_input >= 0.5 or severity_input >= 2:
                return 'medium'
            else:
                return 'low'
        return 'medium'


def save_results_to_csv(filename, issues: List[Issue], output_path: str):
    try:
        results = []
        for issue in issues:
            if hasattr(issue, '__dict__'):
                issue_dict = issue.__dict__.copy()
            else:
                issue_dict = asdict(issue)

            issue_dict['filename'] = filename
            results.append(issue_dict)

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Results saved to CSV: {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {str(e)}")


def batch_analyze(file_triplets: List[Tuple[str, str, str]]) -> Dict[str, List[Issue]]:
    """배치 분석 함수"""
    results = {}
    for i, (image_path, xml_path, json_path) in enumerate(file_triplets, 1):
        print(f"\n=== 배치 분석 {i}/{len(file_triplets)} ===")
        print(f"파일명: {os.path.basename(image_path)}")
        try:
            detector = LayoutDetector()
            issues = detector.analyze_layout(image_path, xml_path, json_path)
            file_key = os.path.basename(image_path).split('.')[0]
            results[file_key] = issues
            print(f"완료: {len(issues)}개 이슈 검출")
        except Exception as e:
            print(f"오류: {str(e)}")
            results[f"error_{i}"] = []
    return results


if __name__ == "__main__":
    image_paths = glob.glob("D:/hnryu/Themes/resource/image/*.png")

    print(f"총 {len(image_paths)}개 이미지 처리 시작...")
    all_issues = []
    for image_path in image_paths:

        filename = os.path.splitext(os.path.basename(image_path))[0]
        xml_path = f"D:/hnryu/Themes/resource/xml/{filename}.xml"
        json_path = f"D:/hnryu/Themes/output/json/{filename}.json"

        print(f"\n=== 처리 중: {filename} ===")
        print(f"  이미지: {image_path}")
        print(f"  XML: {xml_path}")
        print(f"  JSON: {json_path}")

        output_dir = "D:/hnryu/Themes/output/result/20250608"
        # 이슈 검출 실행
        detector = LayoutDetector(output_dir=output_dir)
        issues = detector.analyze_layout(image_path, xml_path, json_path)

        if issues:
            # CSV 저장
            for issue in issues:
                issue.filename = filename
            all_issues.extend(issues)

    if all_issues:
        output_csv = f"{output_dir}/total_issue_results.csv"
        save_results_to_csv("integrated", all_issues, output_csv)
        print(f"\n통합 결과 저장 완료: {output_csv}")
        print(f"총 {len(all_issues)}개 이슈 발견")
    else:
        print("\n발견된 이슈가 없습니다.")
