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


class VisibilityDetector:
    """ Visibility 이슈(0, 1, 2) 검출 클래스"""

    def __init__(self, parent: 'LayoutDetector'):
        self.parent = parent
        self.base_visibility_threshold = parent.visibility_threshold
        self.gray_cache = {}

    def detect_issues(self, elements: List[Dict], image: np.ndarray) -> List[Issue]:
        issues = []

        adaptive_contrast_threshold = self._calculate_contrast_threshold(image)

        for element in elements:
            elem_type = element.get('type', '').lower()
            overlapping_elements = self._find_overlapping_elements(element, elements)

            # 이슈 0: 텍스트/아이콘 대비 문제
            if elem_type in ['textview', 'imageview', 'text', 'icon']:
                contrast_ratio = self._calculate_wcag_contrast_ratio(element, image)
                required_ratio = adaptive_contrast_threshold if elem_type in ['text', 'textview'] else 1.0
                if contrast_ratio < required_ratio:
                    issue = self.parent._create_issue(element, 0)
                    issues.append(issue)

            # 이슈 1: 하이라이트 요소 대비 문제
            if self._is_highlighted_element(element):
                highlight_ratio = self._calculate_highlight_contrast_ratio(element, image)
                if highlight_ratio < adaptive_contrast_threshold:
                    issue = self.parent._create_issue(element, 1)
                    issues.append(issue)

            # 이슈 2: 상호작용 요소 시각적 구분성 (겹치는 요소 고려)
            if element.get('interactivity') and elem_type in ['button', 'imagebutton']:
                affordance_score = self._calculate_button_affordance(element, image, overlapping_elements)
                if affordance_score < 0.6:
                    issue = self.parent._create_issue(element, 2)
                    issues.append(issue)

        return issues

    def _find_overlapping_elements(self, target_element: Dict, all_elements: List[Dict]) -> List[Dict]:
        overlapping = []
        target_bbox = target_element['bbox']

        for element in all_elements:
            if element['id'] == target_element['id']:
                continue

            overlap_ratio = self._calculate_overlap_ratio(target_bbox, element['bbox'])
            if overlap_ratio >= 0.9:
                overlapping.append(element)

        return overlapping

    def _calculate_overlap_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """겹침 비율 계산"""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

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

            visual_features = element.get('visual_features', {})
            avg_color = visual_features.get('avg_color')

            if avg_color and len(avg_color) >= 3:
                element_luminance = self._calculate_relative_luminance(avg_color)
            else:
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
            if isinstance(rgb, list):
                rgb = np.array(rgb)
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

    def _calculate_button_affordance(self, element: Dict, image: np.ndarray,
                                     overlapping_elements: List[Dict] = None) -> float:
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

            visual_features = element.get('visual_features', {})
            edge_density = visual_features.get('edge_density', 0.5)

            # 1. 경계선 검출
            border_score = self._detect_border_affordance(element_region)
            affordance_score += border_score * 0.3

            # 2. 그림자/깊이 효과 검출
            shadow_score = self._detect_shadow_effect(element_region)
            affordance_score += shadow_score * 0.3

            # 3. 배경과의 대비
            contrast_ratio = self._calculate_wcag_contrast_ratio(element, image)
            contrast_score = min(1.0, contrast_ratio / 4.5)
            affordance_score += contrast_score * 0.4

            # 4. 겹치는 요소들과의 차별성 검사
            if overlapping_elements:
                distinction_penalty = self._check_visual_distinction_penalty(element, overlapping_elements)
                affordance_score *= (1 - distinction_penalty)

            return min(1.0, affordance_score)

        except Exception:
            return 0.0

    def _check_visual_distinction_penalty(self, element: Dict, overlapping: List[Dict]) -> float:
        """겹치는 요소들과의 시각적 구별성 페널티 계산 """
        element_features = element.get('visual_features', {})
        element_color = element_features.get('avg_color', [128, 128, 128])
        element_edge = element_features.get('edge_density', 0.5)

        max_penalty = 0.0

        for overlap_elem in overlapping:
            overlap_features = overlap_elem.get('visual_features', {})
            overlap_color = overlap_features.get('avg_color', [128, 128, 128])
            overlap_edge = overlap_features.get('edge_density', 0.5)

            if len(element_color) >= 3 and len(overlap_color) >= 3:
                color_diff = np.mean([abs(a - b) for a, b in zip(element_color[:3], overlap_color[:3])])
                edge_diff = abs(element_edge - overlap_edge)

                # 차이가 작을수록 높은 페널티
                if color_diff < 30 and edge_diff < 0.2:
                    penalty = 1.0 - (color_diff / 30 + edge_diff / 0.2) / 2
                    max_penalty = max(max_penalty, penalty)

        return min(0.5, max_penalty)  # 최대 50% 페널티

    def _detect_border_affordance(self, element_region: np.ndarray) -> float:
        """테두리 검출"""
        try:
            gray = cv2.cvtColor(element_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            h, w = edges.shape
            if h < 4 or w < 4:
                return 0.0

            border_pixels = np.concatenate([
                edges[0, :],    # 상단
                edges[-1, :],   # 하단
                edges[:, 0],    # 좌측
                edges[:, -1]    # 우측
            ])

            border_ratio = np.sum(border_pixels > 0) / len(border_pixels)
            return min(1.0, border_ratio * 3)

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

            brightness = {k: np.mean(v) if v.size > 0 else 128
                          for k, v in regions.items()}

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

        layout_role = element.get('layout_role', '')
        if layout_role in ['main_content', 'navigation', 'toolbar']:
            return True

        # 기존 키워드 검사
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

    def detect_issues(self, elements: List[Dict], ai_responses: Dict, layout_data: Dict = None) -> List[Issue]:

        issues = []

        if layout_data:
            layout_regions = layout_data.get('layout_regions', {})
            hierarchy = layout_data.get('skeleton', {}).get('hierarchy', {})

            # 영역별 정렬 검사
            for region_name, region_info in layout_regions.items():
                region_elements = region_info.get('elements', [])
                if len(region_elements) < 3:
                    continue

                # 이슈 3, 4 검사를 영역별로 수행
                region_issues = self._check_region_specific_alignment(region_elements, region_name, ai_responses)
                issues.extend(region_issues)

            # 이슈 5: 계층 구조 기반 정렬 검사
            hierarchical_issues = self._check_hierarchical_alignment_from_json(hierarchy, layout_data, ai_responses)
            issues.extend(hierarchical_issues)
        else:
            # 기존 로직 유지
            if len(elements) < 3:
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
                    issue = self.parent._create_issue(elem, 3)
                    issues.append(issue)

                # 이슈 4: 수직/수평 정렬 불일치
                vertical_issues = self._check_vertical_alignment(group_elements, dynamic_threshold)
                for elem in vertical_issues:
                    issue = self.parent._create_issue(elem, 4)
                    issues.append(issue)

                # 이슈 5: 동일 계층 요소 정렬 기준 불일치
                reference_issues = self._check_hierarchical_alignment(group_elements, dynamic_threshold)
                for elem in reference_issues:
                    issue = self.parent._create_issue(elem, 5)
                    issues.append(issue)

        return issues

    def _check_region_specific_alignment(self, elements: List[Dict], region_name: str) -> List[
        Issue]:
        """영역별 특성을 고려한 정렬 검사"""
        issues = []

        if region_name in ['navigation', 'toolbar']:
            # 네비게이션/툴바는 수평 정렬 중시
            problematic = self._check_horizontal_alignment_enhanced(elements)
            for elem in problematic:
                issue = self.parent._create_issue(elem, 4)
                issues.append(issue)

        elif region_name in ['content', 'main_content']:
            # 컨텐츠 영역은 좌측 정렬 중시
            problematic = self._check_left_alignment_enhanced(elements)
            for elem in problematic:
                issue = self.parent._create_issue(elem, 3)
                issues.append(issue)

        elif region_name in ['bottom_navigation']:
            # 하단 네비게이션은 균등 분배 검사
            problematic = self._check_even_distribution(elements)
            for elem in problematic:
                issue = self.parent._create_issue(elem, 5)
                issues.append(issue)

        return issues

    def _check_horizontal_alignment_enhanced(self, elements: List[Dict]) -> List[Dict]:
        """수평 정렬 검사 """
        problematic = []

        y_centers = [(elem['bbox'][1] + elem['bbox'][3]) / 2 for elem in elements]
        y_std = np.std(y_centers)

        # JSON visual_features 활용
        avg_height = np.mean([elem['bbox'][3] - elem['bbox'][1] for elem in elements])
        threshold = avg_height * 0.1

        if y_std > threshold:
            mean_y = np.mean(y_centers)
            for i, element in enumerate(elements):
                if abs(y_centers[i] - mean_y) > threshold:
                    # aspect_ratio로 의도적 편차인지 확인
                    visual_features = element.get('visual_features', {})
                    aspect_ratio = visual_features.get('aspect_ratio', 1.0)

                    # 비정상적인 종횡비가 아닌 경우만 문제로 판단
                    if 0.5 <= aspect_ratio <= 3.0:
                        problematic.append(element)

        return problematic

    def _check_left_alignment_enhanced(self, elements: List[Dict]) -> List[Dict]:
        """좌측 정렬 검사 """
        problematic = []

        left_edges = [elem['bbox'][0] for elem in elements]
        left_std = np.std(left_edges)

        threshold = 0.02

        if left_std > threshold:
            mean_left = np.mean(left_edges)
            for element in elements:
                if abs(element['bbox'][0] - mean_left) > threshold:
                    # parent_id를 활용한 들여쓰기 판별
                    if not self._is_intentional_indentation_enhanced(element, elements):
                        problematic.append(element)

        return problematic

    def _is_intentional_indentation_enhanced(self, element: Dict, group: List[Dict]) -> bool:
        """의도적 들여쓰기 판별"""
        # parent_id 기반 판별 (JSON 활용)
        parent_id = element.get('parent_id')
        if parent_id:
            siblings = [e for e in group if e.get('parent_id') == parent_id]
            if len(siblings) >= 2:
                sibling_positions = [e['bbox'][0] for e in siblings]
                if np.std(sibling_positions) < 0.01:
                    return True

        # 기존 리스트 스타일 확인
        content = element.get('content', '').strip()
        if content and (content.startswith('•') or content.startswith('-') or
                        content.startswith('*') or content[0].isdigit()):
            return True

        # layout_role 기반 판별
        layout_role = element.get('layout_role', '')
        if layout_role in ['list_item']:
            return True

        return False

    def _check_even_distribution(self, elements: List[Dict]) -> List[Dict]:
        """균등 분배 검사 """
        problematic = []

        if len(elements) < 3:
            return problematic

        sorted_elements = sorted(elements, key=lambda e: e['bbox'][0])

        gaps = []
        for i in range(1, len(sorted_elements)):
            gap = sorted_elements[i]['bbox'][0] - sorted_elements[i - 1]['bbox'][2]
            gaps.append(gap)

        if gaps:
            gap_std = np.std(gaps)
            avg_gap = np.mean(gaps)

            if gap_std > avg_gap * 0.3:
                for i, gap in enumerate(gaps):
                    if abs(gap - avg_gap) > gap_std:
                        problematic.append(sorted_elements[i + 1])

        return problematic

    def _check_hierarchical_alignment_from_json(self, hierarchy: Dict, layout_data: Dict, ai_responses: Dict) -> List[
        Issue]:
        """JSON hierarchy를 활용한 계층 구조 정렬 검사 """
        issues = []
        elements_dict = {elem['id']: elem for elem in layout_data.get('skeleton', {}).get('elements', [])}

        for parent_id, child_ids in hierarchy.items():
            if len(child_ids) >= 3:
                child_elements = [elements_dict[cid] for cid in child_ids if cid in elements_dict]

                if len(child_elements) >= 3:
                    problematic = self._check_left_alignment_enhanced(child_elements)
                    for elem in problematic:
                        issue = self.parent._create_issue(elem, 5)
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
        """그리드 정렬 검증"""
        x_coords = sorted(set(round(elem['bbox'][0], 2) for elem in candidates))
        y_coords = sorted(set(round(elem['bbox'][1], 2) for elem in candidates))

        if len(x_coords) < 2 or len(y_coords) < 2:
            return False

        grid_positions = set((x, y) for x in x_coords for y in y_coords)
        element_positions = set((round(elem['bbox'][0], 2), round(elem['bbox'][1], 2))
                                for elem in candidates)

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

        interactive_elements = [e for e in elements
                                if e.get('interactivity') or e.get('type', '').lower() in ['button', 'imagebutton']]

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

        return max(0.005, min(0.05, dynamic_threshold))

    def _check_contextual_alignment(self, elements: List[Dict], threshold: float) -> List[Dict]:
        """컨텍스트 고려한 정렬 검사"""
        if len(elements) < 3:
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
                if np.std(sibling_positions) < 0.01:
                    return True

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

        parent_groups = defaultdict(list)
        for element in elements:
            parent_id = element.get('parent_id', 'root')
            parent_groups[parent_id].append(element)

        for parent_id, group in parent_groups.items():
            if len(group) >= 3:
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

    def detect_issues(self, elements: List[Dict], image: np.ndarray) -> List[Issue]:
        # XMLParser 초기화
        issues = []

        for element in elements:
            elem_type = element.get('type', '').lower()

            # 이슈 6: 텍스트 잘림 검출
            if elem_type in ['textview', 'text'] and element.get('content'):
                if self._detect_actual_text_truncation(element, image):
                    issue = self.parent._create_issue(element, 6)
                    issues.append(issue)

            # 이슈 7: 아이콘 잘림 검출
            if elem_type in ['imageview', 'icon', 'image']:
                if self._is_icon_cropped(element, image):
                    issue = self.parent._create_issue(element, 7)
                    issues.append(issue)

        return issues

    def _detect_actual_text_truncation(self, element: Dict, image: np.ndarray) -> bool:
        """실제 텍스트 렌더링 통한 잘림 검출"""
        try:
            content = element.get('content', '')
            if not content:
                return False

            bbox = element['bbox']
            visual_features = element.get('visual_features', {})

            # 방법 1: JSON aspect_ratio 활용
            aspect_ratio = visual_features.get('aspect_ratio')
            if aspect_ratio:
                container_width = bbox[2] - bbox[0]
                container_height = bbox[3] - bbox[1]

                expected_ratio = len(content) * 0.6
                actual_ratio = container_width / container_height if container_height > 0 else 0

                if actual_ratio < expected_ratio * 0.7:
                    return True

            # 방법 2: 말줄임표 검출
            truncation_indicators = ['...', '…', '..', '.', 'more', '더보기']
            content_lower = content.lower().strip()
            if any(indicator in content_lower for indicator in truncation_indicators):
                return True

            # 방법 3: 이미지에서 텍스트 가장자리 검사
            return self._check_text_at_edges(element, image)

        except Exception:
            return False

    def _check_text_at_edges(self, element: Dict, image: np.ndarray) -> bool:
        """텍스트 영역의 가장자리에 텍스트 존재 여부 확인"""
        try:
            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 >= x2 or y1 >= y2:
                return False

            text_region = image[y1:y2, x1:x2]
            if text_region.size == 0:
                return False

            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

            region_h, region_w = binary.shape
            if region_h < 8 or region_w < 8:
                return False

            margin = 3
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
            visual_features = element.get('visual_features', {})

            # 방법 1: 화면 경계와의 거리 확인
            margins = [bbox[0], bbox[1], 1.0 - bbox[2], 1.0 - bbox[3]]
            min_margin = min(margins)

            # 화면 가장자리에 가까우면 잘림으로 판단
            if min_margin < self.crop_threshold:
                return True

            # 방법 2: JSON aspect_ratio 활용
            aspect_ratio = visual_features.get('aspect_ratio', 1.0)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                return True

            # 방법 3: JSON edge_density 활용
            edge_density = visual_features.get('edge_density', 0.5)
            if edge_density > 0.8:
                return self._calculate_icon_completeness(element, image) < 0.7

            return False

        except Exception:
            return False

    def _calculate_icon_completeness(self, element: Dict, image: np.ndarray) -> float:
        """아이콘 완전성 점수 계산"""
        try:
            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 >= x2 or y1 >= y2:
                return 1.0

            icon_region = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(icon_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            icon_h, icon_w = edges.shape
            if icon_h < 10 or icon_w < 10:
                return 1.0

            edge_margin = 2

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
    def __init__(self, output_dir: str):

        super().__init__()
        self.contrast_threshold = 4.5
        self.visibility_threshold = 0.6
        self.alignment_threshold = 0.02
        self.crop_threshold = 0.02
        self.max_issues_per_type = 3

        self.debug = True
        self.output_dir = output_dir

        if self.debug:
            os.makedirs(self.output_dir, exist_ok=True)

        # 검출기들 초기화
        self.visibility_detector = VisibilityDetector(self)
        self.alignment_detector = AlignmentDetector(self)
        self.cutoff_detector = CutoffDetector(self)

    def analyze_layout(self, image_path: str, json_path: str) -> List[Issue]:
        try:
            # 파일 유효성 검사
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON 파일이 없습니다: {json_path}")

            # JSON 데이터 로드
            layout_data = self._load_json(json_path)

            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

            print(f"=== JSON 기반 레이아웃 분석 시작 ===")
            print(f"이미지: {os.path.basename(image_path)}")
            print(f"JSON: {os.path.basename(json_path)}")

            # JSON에서 요소 추출
            elements = layout_data.get('skeleton', {}).get('elements', [])
            if not elements:
                print("분석할 UI 요소가 없습니다.")
                return []
            print(f"총 {len(elements)}개 요소 분석")


            # 화면 특성 분석 (JSON 기반)
            screen_info = self._analyze_screen_characteristics_from_json(layout_data, image)
            print(f"화면 복잡도: {screen_info['complexity']:.2f}")

            all_issues = []

            # 1. 가시성 이슈 검출 (0, 1, 2)
            visibility_issues = self.visibility_detector.detect_issues(elements, image)
            all_issues.extend(visibility_issues)
            print(f"Visibility 이슈 {len(visibility_issues)}개 검출")

            # 2. 정렬 이슈 검출 (3, 4, 5) - layout_data 전달
            alignment_issues = self.alignment_detector.detect_issues(elements, layout_data)
            all_issues.extend(alignment_issues)
            print(f"Alignment 이슈 {len(alignment_issues)}개 검출")

            # 3. 잘림 이슈 검출 (6, 7)
            cutoff_issues = self.cutoff_detector.detect_issues(elements, image)
            all_issues.extend(cutoff_issues)
            print(f"Cut Off 이슈 {len(cutoff_issues)}개 검출")

            # 4. Gemini로 각 후보 이슈 검증
            # verified_issues = self._verify_issues_with_gemini(all_issues, image_path, layout_data)

            # 5. 이슈 우선순위 및 필터링
            final_issues = self._prioritize_and_filter_issues(all_issues, screen_info)
            print(f"최종 {len(final_issues)}개의 이슈 선별")

            # 6. 디버그 시각화
            if self.debug:
                self._save_debug_info(layout_data, final_issues, image_path)
                self.visualize_bboxes(image_path, final_issues, suffix="issues")

            return final_issues

        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _verify_issues_with_gemini(self, candidate_issues: List[Issue], image_path: str, layout_data: Dict) -> List[Issue]:
        """Gemini로 후보 이슈를 검증하고 최종 판단"""
        verified_issues = []
        issue_descriptions = IssuePrompt()

        # 이슈 타입별로 그룹화
        issues_by_type = defaultdict(list)
        for issue in candidate_issues:
            issues_by_type[issue.issue_type].append(issue)

        for issue_type, type_issues in issues_by_type.items():
            print(f"이슈 타입 {issue_type}: {len(type_issues)}개 후보 검증 중...")

            try:
                # 해당 이슈 타입의 프롬프트 가져오기
                prompt = issue_descriptions.get(issue_type, "이 이슈를 분석해주세요.")

                # JSON 컨텍스트 추가
                context_text = f"""
                Layout Data Context:
                {json.dumps(layout_data, indent=2, ensure_ascii=False)}

                Candidate Issues for Type {issue_type}:
                {json.dumps([{
                    'component_id': issue.component_id,
                    'component_type': issue.component_type,
                    'bbox': issue.bbox,
                    'location': issue.location_type
                } for issue in type_issues], indent=2)}
                """

                # Gemini API 호출
                gemini_response = self.generate_response(
                    prompt=prompt,
                    image=image_path,
                    text=context_text
                )

                # Gemini 응답 처리
                if hasattr(gemini_response, 'issues') and gemini_response.issues:
                    # Gemini가 실제 이슈라고 판단한 경우만 추가
                    for gemini_issue in gemini_response.issues:
                        # 원본 후보 이슈와 매칭
                        matched_candidate = self._match_candidate_issue(gemini_issue, type_issues)
                        if matched_candidate:
                            # Gemini 결과로 업데이트
                            verified_issue = self._update_issue_with_gemini_result(matched_candidate, gemini_issue)
                            verified_issues.append(verified_issue)
                            print(f"이슈 확인: {verified_issue.component_id}")
                else:
                    print(f"이슈 타입 {issue_type}: Gemini가 이슈 없음으로 판단")

            except Exception as e:
                print(f"이슈 타입 {issue_type} Gemini 검증 실패: {e}")
                # 실패한 경우 원본 후보들을 기본 설명과 함께 추가
                for candidate in type_issues:
                    candidate.ai_description = f"Gemini 검증 실패 - 기본 분석: {candidate.description_type}"
                    verified_issues.extend(type_issues)

        return verified_issues

    def _match_candidate_issue(self, gemini_issue, candidate_issues: List[Issue]) -> Optional[Issue]:
        """Gemini 결과와 후보 이슈를 매칭"""
        for candidate in candidate_issues:
            # component_id가 일치하거나 bbox가 유사한 경우
            if (hasattr(gemini_issue, 'component_id') and
                    gemini_issue.component_id == candidate.component_id):
                return candidate

            # bbox 기반 매칭 (좌표가 유사한 경우)
            if (hasattr(gemini_issue, 'bbox') and
                    self._bbox_similarity(gemini_issue.bbox, candidate.bbox) > 0.8):
                return candidate

        return None

    def _bbox_similarity(self, bbox1, bbox2) -> float:
        """두 bbox의 유사도 계산 (0~1)"""
        try:
            if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
                return 0.0

            # 중심점과 크기 비교
            center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
            center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

            size1 = [(bbox1[2] - bbox1[0]), (bbox1[3] - bbox1[1])]
            size2 = [(bbox2[2] - bbox2[0]), (bbox2[3] - bbox2[1])]

            # 중심점 거리
            center_dist = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

            # 크기 유사도
            size_similarity = min(size1[0] / size2[0], size2[0] / size1[0]) * min(size1[1] / size2[1], size2[1] / size1[1])

            # 전체 유사도
            similarity = max(0, 1 - center_dist) * size_similarity
            return min(1.0, similarity)

        except:
            return 0.0

    def _update_issue_with_gemini_result(self, candidate_issue: Issue, gemini_issue) -> Issue:
        """후보 이슈를 Gemini 결과로 업데이트"""
        updated_issue = candidate_issue

        # Gemini 결과로 업데이트
        if hasattr(gemini_issue, 'severity'):
            updated_issue.severity = self._map_severity(gemini_issue.severity)

        if hasattr(gemini_issue, 'ai_description'):
            updated_issue.ai_description = gemini_issue.ai_description
        elif hasattr(gemini_issue, 'description'):
            updated_issue.ai_description = gemini_issue.description
        else:
            updated_issue.ai_description = f"Gemini 검증 완료 - 이슈 타입 {candidate_issue.issue_type} 확인됨"

        # 추가 정보 업데이트 가능
        if hasattr(gemini_issue, 'component_type') and gemini_issue.component_type:
            updated_issue.component_type = gemini_issue.component_type

        return updated_issue

    def _analyze_screen_characteristics_from_json(self, layout_data: Dict, image: np.ndarray) -> Dict:
        """JSON 데이터로 화면 특성 분석 """
        elements = layout_data.get('skeleton', {}).get('elements', [])
        layout_regions = layout_data.get('layout_regions', {})

        # 요소 밀도
        active_regions = [name for name, info in layout_regions.items() if info.get('elements')]
        region_density = len(active_regions) / 10.0

        # 타입 다양성
        element_types = set(elem.get('type', 'unknown') for elem in elements)
        type_diversity = len(element_types) / 8.0

        # 계층 복잡도
        hierarchy = layout_data.get('skeleton', {}).get('hierarchy', {})
        hierarchy_complexity = len(hierarchy) / 20.0

        # 종합 복잡도
        complexity = min(1.0, (region_density + type_diversity + hierarchy_complexity) / 3)

        return {
            'complexity': complexity,
            'element_count': len(elements),
            'active_regions': len(active_regions),
            'type_diversity': len(element_types)
        }

    def _load_json(self, json_path: str) -> Dict:
        """JSON 파일 로드"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파일 형식 오류: {json_path}, {str(e)}")

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
                    -self._calculate_visibility_weight(x.bbox),
                    x.bbox[0]
                ),
                reverse=True
            )

            filtered_issues = self._remove_duplicate_issues(sorted_issues[:max_issues_per_type])
            final_issues.extend(filtered_issues)

        return final_issues

    def _calculate_visibility_weight(self, bbox: List[float]) -> float:
        """요소 위치의 시각적 중요도 계산"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # 중앙에 가까울수록, 상단에 가까울수록 높은 가중치
        x_weight = 1.0 - abs(center_x - 0.5) * 2
        y_weight = 1.0 - center_y * 0.5

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

    def _save_debug_info(self, layout_data: Dict, issues: List[Issue], image_path: str):
        """디버그 정보 저장"""
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # filename 필드를 각 이슈에 추가
        for issue in issues:
            if not hasattr(issue, 'filename') or issue.filename is None:
                issue.filename = os.path.basename(image_path)

        debug_info = {
            'image_file': os.path.basename(image_path),
            'total_elements': len(layout_data.get('skeleton', {}).get('elements', [])),
            'active_regions': [name for name, info in layout_data.get('layout_regions', {}).items()
                               if info.get('elements')],
            'detected_issues': [
                # Issue 객체의 모든 필드를 포함하도록 수정
                {
                    'filename': issue.filename,
                    'issue_type': issue.issue_type,
                    'component_id': issue.component_id,
                    'component_type': issue.component_type,
                    'ui_component_id': issue.ui_component_id,
                    'ui_component_type': issue.ui_component_type,
                    'severity': issue.severity,
                    'location_id': issue.location_id,
                    'location_type': issue.location_type,
                    'bbox': issue.bbox,
                    'description_id': issue.description_id,
                    'description_type': issue.description_type,
                    'ai_description': issue.ai_description
                }
                for issue in issues
            ],
            'issue_summary': {
                'total': len(issues),
                'by_type': {issue_type: len([i for i in issues if i.issue_type == issue_type])
                            for issue_type in set(i.issue_type for i in issues)}
            }
        }

        debug_path = os.path.join(self.output_dir, f"debug_{filename}.json")
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        print(f"디버그 정보 저장: {debug_path}")

    def verify_bboxes(self, issues: List[Issue], xml_elements: List[Dict], json_elements: List[Dict]):
        """Verify that issue bounding boxes match XML or JSON bounding boxes"""
        if not self.debug:
            return

        epsilon = 0.001
        for issue in issues:
            issue_bbox = issue.bbox
            component_id = issue.component_id
            matched = False

            # Check JSON elements
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
                thickness = 3 if issue.severity == 'high' else 2
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                label = f"Issue {issue.issue_type} ({issue.severity})"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            output_path = os.path.join(self.output_dir, f"debug_{suffix}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, image)
            print(f"Debug image saved: {output_path}")

        except Exception as e:
            print(f"Error visualizing bounding boxes: {str(e)}")

    def _create_issue(self, element: Dict, issue_type: int) -> Issue:
        """이슈 객체 생성 헬퍼 함수"""
        elem_type = element.get('type', 'unknown')
        bbox = element.get('bbox', [0, 0, 0, 0])
        component_id = element.get('id', 'unknown')

        ui_component_id, ui_component_type = self._calc_ui_component(elem_type)
        location_id, location_type = self._calc_location(bbox)
        description_id, description_type = self._get_description(str(issue_type))
        severity = 'medium'
        ai_description = description_type

        return Issue(
            filename=None,
            issue_type=str(issue_type),
            component_id=component_id,
            component_type=elem_type,
            ui_component_id=ui_component_id,
            ui_component_type=ui_component_type,
            severity=severity,
            location_id=location_id,
            location_type=location_type,
            bbox=bbox,
            description_id=description_id,
            description_type=description_type,
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

        location_type = EvalKPI.LOCATION.get(location_id, 'MC')
        return location_id, location_type

    def _calc_ui_component(self, element_type: str) -> Tuple[str, str]:
        """UI 구성 요소 분류 using EvalKPI.UI_COMPONENT"""
        type_mapping = {
            'text': '5',  # TextView
            'textview': '5',
            'button': '1',  # Button
            'imagebutton': '1',
            'image': '4',  # ImageView
            'imageview': '4',
            'input': '6',  # EditText
            'edittext': '6'
        }

        ui_component_id = type_mapping.get(element_type.lower(), '5')
        ui_component_type = EvalKPI.UI_COMPONENT.get(ui_component_id, 'TextView')
        return ui_component_id, ui_component_type

    def _get_description(self, issue_type: str) -> Tuple[str, str]:
        """이슈 유형(issue_type)에 해당하는 설명(description)의 ID와 타입(type)을 반환"""
        description_id = issue_type
        description_type = EvalKPI.DESCRIPTION.get(issue_type, "Unknown issue")
        return description_id, description_type

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
            # Issue 객체의 모든 필드를 dict로 변환
            if hasattr(issue, '__dict__'):
                issue_dict = issue.__dict__.copy()
            else:
                issue_dict = asdict(issue)

            if 'filename' not in issue_dict or issue_dict['filename'] is None:
                issue_dict['filename'] = filename

            results.append(issue_dict)

        df = pd.DataFrame(results)

        # 컬럼 정의
        columns = [
            'filename', 'issue_type', 'component_id', 'component_type',
            'ui_component_id', 'ui_component_type', 'severity',
            'location_id', 'location_type', 'bbox',
            'description_id', 'description_type', 'ai_description'
        ]

        # 존재하는 컬럼만 선택
        available_columns = [col for col in columns if col in df.columns]
        df = df[available_columns]

        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to CSV: {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {str(e)}")


def batch_analyze(file_triplets: List[Tuple[str, str, str]]) -> Dict[str, List[Issue]]:
    """배치 분석 함수"""
    results = {}
    for i, (image_path, json_path, xml_path) in enumerate(file_triplets, 1):
        print(f"\n=== 배치 분석 {i}/{len(file_triplets)} ===")
        print(f"파일명: {os.path.basename(image_path)}")
        try:
            detector = LayoutDetector()
            # JSON 우선, XML은 선택적
            issues = detector.analyze_layout(image_path, json_path, xml_path)
            file_key = os.path.basename(image_path).split('.')[0]

            # filename 설정
            filename = os.path.basename(image_path)
            for issue in issues:
                issue.filename = filename

            results[file_key] = issues
            print(f"완료: {len(issues)}개 이슈 검출")
        except Exception as e:
            print(f"오류: {str(e)}")
            results[f"error_{i}"] = []
    return results


def batch_analyze_json_only(image_dir: str, json_dir: str, output_dir: str) -> Dict[str, List[Issue]]:
    """JSON만 사용하는 배치 분석 함수"""
    detector = LayoutDetector(output_dir)
    results = {}

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    print(f"총 {len(image_files)}개 이미지 발견")

    for image_path in image_files:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(json_dir, f"{filename}.json")

        if not os.path.exists(json_path):
            print(f"JSON 파일 없음: {json_path}")
            continue

        print(f"\n=== 분석 중: {filename} ===")

        try:
            issues = detector.analyze_layout(image_path, json_path)

            # filename 설정
            image_filename = os.path.basename(image_path)
            for issue in issues:
                issue.filename = image_filename

            results[filename] = issues

            print(f"완료: {len(issues)}개 이슈 검출")

            # 개별 결과 저장
            if issues:
                individual_csv = os.path.join(output_dir, f"{filename}_issues.csv")
                save_results_to_csv(image_filename, issues, individual_csv)

        except Exception as e:
            print(f"오류: {str(e)}")
            results[filename] = []

    return results


if __name__ == "__main__":
    # 기존 방식 (XML + JSON)
    image_paths = glob.glob("D:/hnryu/Themes/resource/image/*.png")

    print(f"총 {len(image_paths)}개 이미지 처리 시작...")
    all_issues = []

    for image_path in image_paths[:5]:

        filename = os.path.splitext(os.path.basename(image_path))[0]
        xml_path = f"D:/hnryu/Themes/resource/xml/{filename}.xml"
        json_path = f"D:/hnryu/Themes/output/json/{filename}.json"

        print(f"\n=== 처리 중: {filename} ===")
        print(f"  이미지: {image_path}")
        print(f"  JSON: {json_path}")
        print(f"  XML: {xml_path}")

        output_dir = "D:/hnryu/Themes/output/result/20250608"
        # 이슈 검출 실행
        detector = LayoutDetector(output_dir=output_dir)
        issues = detector.analyze_layout(image_path, json_path)

        # filename 설정
        image_filename = os.path.basename(image_path)
        for issue in issues:
            issue.filename = image_filename

        if issues:
            all_issues.extend(issues)

    if all_issues:
        output_csv = f"{output_dir}/total_issue_results.csv"
        save_results_to_csv("integrated", all_issues, output_csv)
        print(f"\n통합 결과 저장 완료: {output_csv}")
        print(f"총 {len(all_issues)}개 이슈 발견")
    else:
        print("\n발견된 이슈가 없습니다.")

    # JSON만 사용하는 새로운 방식 예시
    # results = batch_analyze_json_only(
    #     image_dir="./resource/image",
    #     json_dir="./output/json",
    #     output_dir="./output/enhanced_json_only_results"
    # )