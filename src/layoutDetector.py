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
from common.logger import timefn
from src.gemini import Gemini


class DuplicateDetector:
    """중복 아이콘 이슈(8) 검출 클래스"""

    def __init__(self, parent: 'LayoutDetector'):
        self.parent = parent
        self.similarity_threshold = 0.95
        self.default_image_cache = {}

    @timefn
    def detect_duplicate_icons(self, elements: List[Dict], image: np.ndarray,
                               default_image: Optional[np.ndarray] = None) -> List[Issue]:
        """중복 아이콘 검출"""
        issues = []

        # 아이콘 크기 요소만 필터링
        icon_elements = self._filter_icon_elements(elements)
        if len(icon_elements) < 2:
            return issues

        print(f"아이콘 요소 {len(icon_elements)}개 검사 시작")

        # 각 아이콘의 이미지 데이터 추출
        icon_images = []
        for i, icon in enumerate(icon_elements):
            bbox = icon['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                icon_img = image[y1:y2, x1:x2]
                if icon_img.size > 0:
                    icon_images.append({
                        'index': i,
                        'element': icon,
                        'image': icon_img
                    })

        # 중복 그룹 찾기
        duplicate_groups = self._find_duplicate_groups(icon_images)

        for group in duplicate_groups:
            if len(group) > 1:
                # 디폴트 이미지와 비교하여 정상 중복인지 확인
                is_normal_duplicate = self._verify_with_default_image(
                    group, icon_images, default_image
                )

                if not is_normal_duplicate:
                    # 중복 이슈 생성
                    for icon_idx in group:
                        element = icon_images[icon_idx]['element']

                        other_positions = [
                            str(icon_images[other_idx]['element']['bbox'])
                            for other_idx in group if other_idx != icon_idx
                        ]

                        issue = self.parent._create_issue(element, 8)
                        issue.score = self.parent._map_score(2)  # "Fail with issue"
                        issue.ai_description = (
                            f"중복 아이콘 탐지: {len(group)}개의 동일한 아이콘. "
                            f"현재 위치: {element['bbox']}, "
                            f"동일 아이콘 위치: {', '.join(other_positions)} "
                            f"(디폴트와 다름)"
                        )
                        issues.append(issue)

                        print(f"중복 아이콘 이슈: {element.get('id', 'unknown')}")

        return issues

    def _filter_icon_elements(self, elements: List[Dict]) -> List[Dict]:
        """아이콘 크기 요소 필터링"""
        icon_elements = []

        for element in elements:
            elem_type = element.get('type', '').lower()

            # 이미지/아이콘 타입 확인
            if elem_type in ['imageview', 'image', 'icon']:
                bbox = element['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                # 아이콘 크기 범위 (상대좌표 기준)
                if 0.02 <= width <= 0.15 and 0.02 <= height <= 0.15:
                    # 정사각형에 가까운 비율
                    aspect_ratio = width / height if height > 0 else 1.0
                    if 0.5 <= aspect_ratio <= 2.0:
                        icon_elements.append(element)

        return icon_elements

    def _find_duplicate_groups(self, icon_images: List[Dict]) -> List[List[int]]:
        """아이콘 유사도 기반 중복 그룹 찾기"""
        if len(icon_images) < 2:
            return []

        # 유사도 매트릭스 계산
        similarity_matrix = np.zeros((len(icon_images), len(icon_images)))

        for i in range(len(icon_images)):
            for j in range(i + 1, len(icon_images)):
                similarity = self._calculate_icon_similarity(
                    icon_images[i]['image'],
                    icon_images[j]['image']
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity

        # 중복 그룹 찾기
        duplicate_groups = []
        processed = set()

        for i in range(len(icon_images)):
            if i in processed:
                continue

            group = [i]
            processed.add(i)

            for j in range(i + 1, len(icon_images)):
                if j not in processed and similarity_matrix[i][j] >= self.similarity_threshold:
                    group.append(j)
                    processed.add(j)

            if len(group) > 1:
                duplicate_groups.append(group)

        return duplicate_groups

    def _calculate_icon_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """아이콘 유사도 계산"""
        try:
            # 크기 정규화
            target_size = (32, 32)
            img1_resized = cv2.resize(img1, target_size)
            img2_resized = cv2.resize(img2, target_size)

            # 그레이스케일 변환
            if len(img1_resized.shape) == 3:
                img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1_resized

            if len(img2_resized.shape) == 3:
                img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2_resized

            # 1. 히스토그램 비교
            hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
            hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # 2. 픽셀 단위 비교
            diff = cv2.absdiff(img1_gray, img2_gray)
            pixel_similarity = 1.0 - (np.mean(diff) / 255.0)

            # 3. 템플릿 매칭
            result = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
            template_similarity = np.max(result)

            # 가중 평균
            final_similarity = (
                    hist_similarity * 0.3 +
                    pixel_similarity * 0.4 +
                    template_similarity * 0.3
            )

            return max(0.0, min(1.0, final_similarity))

        except Exception as e:
            print(f"유사도 계산 오류: {e}")
            return 0.0

    def _verify_with_default_image(self, group: List[int], icon_images: List[Dict],
                                   default_image: Optional[np.ndarray]) -> bool:
        """디폴트 이미지와 비교 검증"""
        if default_image is None:
            return False

        try:
            # default_image가 딕셔너리인 경우 numpy 배열로 변환
            if isinstance(default_image, dict):
                # 딕셔너리에서 실제 이미지 데이터 추출
                if 'image' in default_image:
                    default_image = default_image['image']
                elif 'data' in default_image:
                    default_image = default_image['data']
                else:
                    print("디폴트 이미지 딕셔너리에서 이미지 데이터를 찾을 수 없습니다")
                    return False

            # numpy 배열인지 확인
            if not isinstance(default_image, np.ndarray):
                print(f"디폴트 이미지가 numpy 배열이 아닙니다: {type(default_image)}")
                return False

            # 이미지가 비어있는지 확인
            if default_image.size == 0:
                print("디폴트 이미지가 비어있습니다")
                return False

            # 각 그룹 멤버의 좌표에서 디폴트 이미지 추출
            default_crops = []

            for icon_idx in group:
                element = icon_images[icon_idx]['element']
                bbox = element['bbox']

                # 디폴트 이미지에서 같은 좌표로 추출
                h, w = default_image.shape[:2]
                x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

                if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                    default_crop = default_image[y1:y2, x1:x2]
                    if default_crop.size > 0:
                        default_crops.append(default_crop)

            if len(default_crops) < 2:
                return False

            # 디폴트 이미지에서 추출한 이미지들이 모두 동일한지 확인
            base_image = default_crops[0]

            for compare_image in default_crops[1:]:
                # 크기 맞추기
                if base_image.shape != compare_image.shape:
                    compare_image = cv2.resize(compare_image,
                                               (base_image.shape[1], base_image.shape[0]))

                # 유사도 계산
                similarity = self._calculate_icon_similarity(base_image, compare_image)

                # 디폴트에서 다르면 이슈
                if similarity < 0.99:
                    return False

            return True  # 디폴트에서도 동일하면 정상

        except Exception as e:
            print(f"디폴트 검증 오류: {e}")
            print(f"default_image 타입: {type(default_image)}")
            if hasattr(default_image, 'keys'):
                print(f"default_image 키들: {list(default_image.keys())}")
            return False


class AestheticDetector:
    """심미적 이슈(9, 10, 11) 검출 클래스"""

    def __init__(self, parent: 'LayoutDetector'):
        self.parent = parent

    @timefn
    def detect_aesthetic_issues(self, elements: List[Dict], image: np.ndarray) -> List[Issue]:
        issues = []

        # 1. 색상 조화 검사
        color_issues = self._check_color_harmony(elements, image)
        issues.extend(color_issues)

        # 2. 공간 활용 검사
        spacing_issues = self._check_spacing_consistency(elements)
        issues.extend(spacing_issues)

        # 3. 비례 조화 검사
        proportion_issues = self._check_proportional_harmony(elements)
        issues.extend(proportion_issues)

        return issues

    def _check_color_harmony(self, elements: List[Dict], image: np.ndarray) -> List[Issue]:
        """색상 조화 검사"""
        issues = []

        # 전체 화면의 주요 색상 추출
        screen_colors = self._extract_screen_colors(image)

        # 각 요소의 색상이 전체 조화를 해치는지 확인
        for element in elements:
            elem_type = element.get('type', '').lower()

            if elem_type in ['button', 'imagebutton']:
                element_colors = self._get_element_colors(element, image)

                if element_colors and self._is_color_discordant(element_colors, screen_colors):
                    issue = self.parent._create_issue(element, 9)
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
                    issue.ai_description = f"색상 조화 불일치: 요소 색상이 전체 테마와 부조화"
                    issues.append(issue)

        return issues

    def _extract_screen_colors(self, image: np.ndarray) -> List[List[int]]:
        """화면 전체 주요 색상 추출"""
        try:
            # 이미지를 작게 리사이즈하여 처리 속도 향상
            small_img = cv2.resize(image, (100, 100))
            pixels = small_img.reshape(-1, 3)

            # 간단한 K-means 대신 대표 색상 추출
            colors = []
            for i in range(0, len(pixels), len(pixels) // 5):
                chunk = pixels[i:i + len(pixels) // 5]
                if len(chunk) > 0:
                    avg_color = np.mean(chunk, axis=0)
                    colors.append(avg_color.astype(int).tolist())

            return colors[:5]  # 최대 5개 색상
        except:
            return [[128, 128, 128]]  # 기본 회색

    def _get_element_colors(self, element: Dict, image: np.ndarray) -> List[List[int]]:
        """요소의 주요 색상 추출"""
        try:
            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return []

            element_region = image[y1:y2, x1:x2]
            if element_region.size == 0:
                return []

            # 평균 색상 계산
            avg_color = np.mean(element_region.reshape(-1, 3), axis=0)
            return [avg_color.astype(int).tolist()]

        except Exception:
            return []

    def _is_color_discordant(self, element_colors: List[List[int]], screen_colors: List[List[int]]) -> bool:
        """색상 부조화 판별"""
        if not element_colors or not screen_colors:
            return False

        element_color = element_colors[0]

        # 모든 화면 색상과의 거리 계산
        min_distance = float('inf')
        for screen_color in screen_colors:
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(element_color, screen_color)))
            min_distance = min(min_distance, distance)

        # 거리가 너무 크면 부조화로 판정
        return min_distance > 100  # 임계값

    def _check_spacing_consistency(self, elements: List[Dict]) -> List[Issue]:
        """공간 활용 일관성 검사"""
        issues = []

        # 요소 간 간격 계산
        spacings = []
        sorted_elements = sorted(elements, key=lambda e: (e['bbox'][1], e['bbox'][0]))

        for i in range(len(sorted_elements) - 1):
            current = sorted_elements[i]
            next_elem = sorted_elements[i + 1]

            # 수직 간격 계산
            vertical_gap = next_elem['bbox'][1] - current['bbox'][3]
            if vertical_gap > 0:
                spacings.append(vertical_gap)

        if spacings:
            spacing_std = np.std(spacings)
            mean_spacing = np.mean(spacings)

            # 간격 편차가 큰 경우
            if spacing_std > mean_spacing * 0.5:
                # 가장 편차가 큰 요소 찾기
                for i, spacing in enumerate(spacings):
                    if abs(spacing - mean_spacing) > spacing_std:
                        problematic_element = sorted_elements[i + 1]

                        issue = self.parent._create_issue(problematic_element, 10)
                        issue.score = self.parent._map_score(2)  # "Fail with issue"
                        issue.ai_description = f"공간 활용 불일치: 간격 {spacing:.3f} (평균: {mean_spacing:.3f})"
                        issues.append(issue)
                        break

        return issues

    def _check_proportional_harmony(self, elements: List[Dict]) -> List[Issue]:
        """비례 조화 검사"""
        issues = []

        # 황금비 (1.618) 기반 검사
        golden_ratio = 1.618

        for element in elements:
            bbox = element['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if height > 0:
                aspect_ratio = width / height

                # 황금비와의 차이가 클 때, 그리고 정사각형도 아닐 때
                golden_diff = abs(aspect_ratio - golden_ratio)
                square_diff = abs(aspect_ratio - 1.0)

                if golden_diff > 0.5 and square_diff > 0.3 and (aspect_ratio > 3.0 or aspect_ratio < 0.3):
                    issue = self.parent._create_issue(element, 11)
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
                    issue.ai_description = f"비례 부조화: 종횡비 {aspect_ratio:.2f} (극단적 비율)"
                    issues.append(issue)

        return issues


class VisibilityDetector:
    """ Visibility 이슈(0, 1, 2) 검출 클래스"""

    def __init__(self, parent: 'LayoutDetector'):
        self.parent = parent
        self.base_visibility_threshold = parent.visibility_threshold
        self.base_affordance_score = parent.affordance_threshold
        self.base_overlap_threshold = parent.overlap_threshold

    @timefn
    def detect_visibility_issues(self, elements: List[Dict], image: np.ndarray) -> List[Issue]:
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
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
                    issues.append(issue)

            # 이슈 1: 하이라이트 요소 대비 문제
            if self._is_highlighted_element(element):
                highlight_ratio = self._calculate_highlight_contrast_ratio(element, image)
                if highlight_ratio < adaptive_contrast_threshold:
                    issue = self.parent._create_issue(element, 1)
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
                    issues.append(issue)

            # 이슈 2: 상호작용 요소 시각적 구분성 (겹치는 요소 고려)
            if element.get('interactivity') and elem_type in ['button', 'imagebutton']:
                affordance_score = self._calculate_button_affordance(element, image, overlapping_elements)
                if affordance_score < self.base_affordance_score:
                    issue = self.parent._create_issue(element, 2)
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
                    issues.append(issue)

        return issues

    def _find_overlapping_elements(self, target_element: Dict, all_elements: List[Dict]) -> List[Dict]:
        overlapping = []
        target_bbox = target_element['bbox']

        for element in all_elements:
            if element['id'] == target_element['id']:
                continue

            overlap_ratio = self._calculate_overlap_ratio(target_bbox, element['bbox'])
            if overlap_ratio >= 0.5:
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
            adaptive_threshold = max(2.5, min(5.0, overall_contrast * 8))
            return adaptive_threshold
        except:
            return 4.5

    def _calculate_wcag_contrast_ratio(self, element: Dict, image: np.ndarray) -> float:
        """WCAG 표준에 따른 정확한 대비 비율 계산"""
        try:
            # 1. 요소 영역 추출
            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return 1.0

            element_region = image[y1:y2, x1:x2]
            if element_region.size == 0:
                return 1.0

            # 색상 추출 로직만 사용
            colors = self._extract_colors_with_visibility_logic(element_region)

            if len(colors) < 2:
                return 1.0

            # 3. 추출된 색상들 간의 최대 대비 계산
            max_contrast = 0
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    luminance1 = self._calculate_relative_luminance(colors[i])
                    luminance2 = self._calculate_relative_luminance(colors[j])

                    lighter = max(luminance1, luminance2)
                    darker = min(luminance1, luminance2)

                    contrast_ratio = (lighter + 0.05) / (darker + 0.05)
                    max_contrast = max(max_contrast, contrast_ratio)

            return max_contrast

        except Exception as e:
            print(f"WCAG 대비 계산 오류: {e}")
            return 1.0

    def _extract_colors_with_visibility_logic(self, img_region: np.ndarray, num_colors: int = 3) -> List[List[int]]:
        try:
            if len(img_region.shape) == 3:
                img_rgb = cv2.cvtColor(img_region, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img_region

            # 이미지 특성 분석 (visibility.py와 동일)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(img_hsv)

            # 채도와 명도의 분산 계산
            saturation_var = np.var(s)
            value_var = np.var(v)

            # 수정된 고채도 픽셀 비율 - 명도가 충분히 높은 픽셀만 고려
            bright_pixels_mask = v > 80
            if np.any(bright_pixels_mask):
                bright_s = s[bright_pixels_mask]
                high_saturation_ratio = np.sum(bright_s > 30) / bright_s.size
            else:
                high_saturation_ratio = 0.0

            # 명도 대비 분석 (검정-흰색 조합 감지)
            low_value_ratio = np.sum(v < 80) / v.size
            high_value_ratio = np.sum(v > 180) / v.size
            has_high_value_contrast = (low_value_ratio > 0.2 and high_value_ratio > 0.2) or value_var > 3000

            # RGB 기반 색상 분산도 체크
            rgb_std = np.std(img_rgb.reshape(-1, 3), axis=0)
            is_grayscale = (rgb_std < 15).all()

            # 색상 추출 방법 선택
            if (has_high_value_contrast and high_saturation_ratio <= 0.3) or is_grayscale:
                colors = self._simple_color_extraction(img_rgb, num_colors)
            elif high_saturation_ratio <= 0.4:
                colors = self._lab_color_extraction(img_rgb, num_colors)
            else:
                colors = self._hsv_color_extraction(img_rgb, num_colors)

            return colors

        except Exception as e:
            print(f"색상 추출 오류: {e}")
            return [[128, 128, 128], [64, 64, 64]]

    def _simple_color_extraction(self, img_rgb: np.ndarray, num_colors: int) -> List[List[int]]:
        """검정-흰색 같은 무채색 고대비 조합을 위한 간단한 색상 추출"""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        colors = []

        # 매우 어두운 영역 (0~60)
        very_dark_mask = gray < 60
        if np.any(very_dark_mask):
            dark_pixels = img_rgb[very_dark_mask]
            dark_color = np.mean(dark_pixels, axis=0).astype(int)
            colors.append(dark_color.tolist())

        # 매우 밝은 영역 (200~255)
        very_bright_mask = gray > 200
        if np.any(very_bright_mask):
            bright_pixels = img_rgb[very_bright_mask]
            bright_color = np.mean(bright_pixels, axis=0).astype(int)
            colors.append(bright_color.tolist())

        # 중간 어두운 영역 (60~120)
        if len(colors) < num_colors:
            mid_dark_mask = (gray >= 60) & (gray <= 120)
            if np.any(mid_dark_mask):
                mid_dark_pixels = img_rgb[mid_dark_mask]
                mid_dark_color = np.mean(mid_dark_pixels, axis=0).astype(int)
                colors.append(mid_dark_color.tolist())

        # 중간 밝은 영역 (150~200)
        if len(colors) < num_colors:
            mid_bright_mask = (gray >= 150) & (gray <= 200)
            if np.any(mid_bright_mask):
                mid_bright_pixels = img_rgb[mid_bright_mask]
                mid_bright_color = np.mean(mid_bright_pixels, axis=0).astype(int)
                colors.append(mid_bright_color.tolist())

        # KMeans 사용
        if len(colors) < 2:
            try:
                from sklearn.cluster import KMeans
                pixels_reshaped = img_rgb.reshape(-1, 3)
                kmeans = KMeans(n_clusters=max(2, num_colors), random_state=42, n_init=10)
                kmeans.fit(pixels_reshaped)

                colors = []
                for center in kmeans.cluster_centers_:
                    colors.append(center.astype(int).tolist())
            except ImportError:
                # sklearn이 없는 경우 기본 색상 사용
                colors = [[0, 0, 0], [255, 255, 255]]

        return colors[:num_colors]

    def _hsv_color_extraction(self, img_rgb: np.ndarray, num_colors: int) -> List[List[int]]:
        """HSV 색공간 기반 색상 추출 """
        try:
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            h_channel = img_hsv[:, :, 0]
            hist_h = cv2.calcHist([h_channel], [0], None, [180], [0, 180])

            # 피크 찾기
            peaks = []
            for i in range(1, len(hist_h) - 1):
                if hist_h[i] > hist_h[i - 1] and hist_h[i] > hist_h[i + 1] and hist_h[i] > np.max(hist_h) * 0.1:
                    peaks.append(i)

            peaks = sorted(peaks, key=lambda x: hist_h[x], reverse=True)[:num_colors]

            colors = []
            for peak_h in peaks:
                mask = np.abs(h_channel - peak_h) < 10
                if np.any(mask):
                    masked_pixels = img_rgb[mask]
                    if len(masked_pixels) > 0:
                        avg_color = np.mean(masked_pixels, axis=0).astype(int)
                        colors.append(avg_color.tolist())

            # 충분한 색상을 찾지 못한 경우 KMeans로 보완
            if len(colors) < num_colors:
                try:
                    from sklearn.cluster import KMeans
                    pixels_reshaped = img_rgb.reshape(-1, 3)
                    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
                    kmeans.fit(pixels_reshaped)

                    for center in kmeans.cluster_centers_:
                        if len(colors) < num_colors:
                            colors.append(center.astype(int).tolist())
                except ImportError:
                    colors.extend([[128, 128, 128]] * (num_colors - len(colors)))

            return colors[:num_colors]

        except Exception:
            return [[128, 128, 128], [64, 64, 64]]

    def _lab_color_extraction(self, img_rgb: np.ndarray, num_colors: int) -> List[List[int]]:
        """LAB 색공간 기반 색상 추출"""
        try:
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            a_channel = img_lab[:, :, 1]
            b_channel = img_lab[:, :, 2]

            # histogram
            hist_ab = cv2.calcHist([a_channel, b_channel], [0, 1], None, [32, 32], [0, 256, 0, 256])

            # 피크 찾기
            peaks = []
            threshold = np.max(hist_ab) * 0.05

            for i in range(1, hist_ab.shape[0] - 1):
                for j in range(1, hist_ab.shape[1] - 1):
                    if (hist_ab[i, j] > threshold and
                            hist_ab[i, j] > hist_ab[i - 1, j] and hist_ab[i, j] > hist_ab[i + 1, j] and
                            hist_ab[i, j] > hist_ab[i, j - 1] and hist_ab[i, j] > hist_ab[i, j + 1]):
                        peaks.append((i, j, hist_ab[i, j]))

            peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:num_colors]

            colors = []
            for a_bin, b_bin, _ in peaks:
                a_center = a_bin * 8
                b_center = b_bin * 8

                a_mask = np.abs(a_channel - a_center) < 12
                b_mask = np.abs(b_channel - b_center) < 12
                region_mask = a_mask & b_mask

                if np.any(region_mask):
                    region_pixels = img_rgb[region_mask]
                    avg_color = np.mean(region_pixels, axis=0).astype(int)
                    colors.append(avg_color.tolist())

            # LAB에서 충분한 색상을 찾지 못한 경우 KMeans로 보완
            if len(colors) < num_colors:
                try:
                    from sklearn.cluster import KMeans
                    pixels_reshaped = img_rgb.reshape(-1, 3)
                    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
                    kmeans.fit(pixels_reshaped)

                    for center in kmeans.cluster_centers_:
                        if len(colors) < num_colors:
                            colors.append(center.astype(int).tolist())
                except ImportError:
                    colors.extend([[128, 128, 128]] * (num_colors - len(colors)))

            return colors[:num_colors]

        except Exception:
            return [[128, 128, 128], [64, 64, 64]]

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
        """겹치는 요소들과의 시각적 구별성 페널티 계산"""
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

                if color_diff < 30 and edge_diff < 0.2:
                    penalty = 1.0 - (color_diff / 30 + edge_diff / 0.2) / 2
                    max_penalty = max(max_penalty, penalty)

        return min(0.5, max_penalty)

    def _detect_border_affordance(self, element_region: np.ndarray) -> float:
        """테두리 검출"""
        try:
            gray = cv2.cvtColor(element_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            h, w = edges.shape
            if h < 4 or w < 4:
                return 0.0

            border_pixels = np.concatenate([
                edges[0, :], edges[-1, :], edges[:, 0], edges[:, -1]
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

    @timefn
    def detect_alignment_issues(self, elements: List[Dict], layout_data: Dict = None) -> List[Issue]:
        issues = []

        if layout_data:
            layout_regions = layout_data.get('layout_regions', {})
            hierarchy = layout_data.get('skeleton', {}).get('hierarchy', {})

            # 영역별 정렬 검사
            for region_name, region_info in layout_regions.items():
                region_elements = region_info.get('elements', [])
                if len(region_elements) < 3:
                    continue

                region_issues = self._check_region_specific_alignment(region_elements, region_name)
                issues.extend(region_issues)

            # 이슈 5: 계층 구조 기반 정렬 검사
            hierarchical_issues = self._check_hierarchical_alignment_from_json(hierarchy, layout_data)
            issues.extend(hierarchical_issues)
        else:

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
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
                    issues.append(issue)

                # 이슈 4: 수직/수평 정렬 불일치
                vertical_issues = self._check_vertical_alignment(group_elements, dynamic_threshold)
                for elem in vertical_issues:
                    issue = self.parent._create_issue(elem, 4)
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
                    issues.append(issue)

                # 이슈 5: 동일 계층 요소 정렬 기준 불일치
                reference_issues = self._check_hierarchical_alignment(group_elements, dynamic_threshold)
                for elem in reference_issues:
                    issue = self.parent._create_issue(elem, 5)
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
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
                issue.score = self.parent._map_score(2)  # "Fail with issue"
                issues.append(issue)

        elif region_name in ['content', 'main_content']:
            # 컨텐츠 영역은 좌측 정렬 중시
            problematic = self._check_left_alignment_enhanced(elements)
            for elem in problematic:
                issue = self.parent._create_issue(elem, 3)
                issue.score = self.parent._map_score(2)  # "Fail with issue"
                issues.append(issue)

        elif region_name in ['bottom_navigation']:
            # 하단 네비게이션은 균등 분배 검사
            problematic = self._check_even_distribution(elements)
            for elem in problematic:
                issue = self.parent._create_issue(elem, 5)
                issue.score = self.parent._map_score(2)  # "Fail with issue"
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

    def _check_hierarchical_alignment_from_json(self, hierarchy: Dict, layout_data: Dict) -> List[
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
                        issue.score = self.parent._map_score(2)  # "Fail with issue"
                        issues.append(issue)

        return issues

    def _detect_layout_patterns(self, elements: List[Dict]) -> Dict[str, List[List[Dict]]]:
        """레이아웃 패턴 감지"""
        patterns = {
            'vertical_groups': [],
            'grid': [],
            'list': [],
            'navigation': []
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
        size_based_threshold = min(avg_width, avg_height) * 0.1

        # 요소 개수에 따른 조정
        count_factor = max(0.5, 1.0 - len(elements) * 0.05)
        dynamic_threshold = size_based_threshold * count_factor

        return max(0.01, min(0.1, dynamic_threshold))

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
        self.base_crop_threshold = parent.crop_threshold

    @timefn
    def detect_cutoff_issues(self, elements: List[Dict], image: np.ndarray) -> List[Issue]:
        # XMLParser 초기화
        issues = []

        for element in elements:
            elem_type = element.get('type', '').lower()

            # 이슈 6: 텍스트 잘림 검출
            if elem_type in ['textview', 'text'] and element.get('content'):
                if self._detect_actual_text_truncation(element, image):
                    issue = self.parent._create_issue(element, 6)
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
                    issues.append(issue)

            # 이슈 7: 아이콘 잘림 검출
            if elem_type in ['imageview', 'icon', 'image']:
                if self._is_icon_cropped(element, image):
                    issue = self.parent._create_issue(element, 7)
                    issue.score = self.parent._map_score(2)  # "Fail with issue"
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
            threshold = 0.3  # 30% 이상
            return left_text_ratio > threshold or right_text_ratio > threshold

        except Exception:
            return False

    def _is_icon_cropped(self, element: Dict, image: np.ndarray) -> bool:
        """아이콘 잘림 검출"""
        try:
            bbox = element['bbox']
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]

            crop_img = image[y1:y2, x1:x2]
            if crop_img.size == 0:
                return False

            # cutoff.py의 로직 적용:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=2, maxLineGap=0.5)

            if lines is not None:
                crop_h, crop_w = gray.shape
                min_line_length_ratio = 0.05

                for line in lines:
                    x1_line, y1_line, x2_line, y2_line = line[0]
                    line_length = np.sqrt((x2_line - x1_line) ** 2 + (y2_line - y1_line) ** 2)

                    if line_length > min_line_length_ratio * max(crop_h, crop_w):
                        return True

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


class ScoreConstants:
    FAIL_CRITICAL = "Fail with Critical issue"  # 1등
    FAIL_ISSUE = "Fail with issue"  # 2등
    CONDITIONAL_PASS = "Conditional Pass"  # 3등
    PASS_MINOR = "Pass with minor concern"  # 4등
    PASS_NO_ISSUE = "Pass with no issue"  # 5등

    SCORE_MAP = {
        1: FAIL_CRITICAL,
        2: FAIL_ISSUE,
        3: CONDITIONAL_PASS,
        4: PASS_MINOR,
        5: PASS_NO_ISSUE
    }


class LayoutDetector(Gemini):
    """ layoutParser.py -> 레이아웃 이슈 검출기"""
    def __init__(self, output_dir: str):

        super().__init__()
        self.affordance_threshold = 0.3
        self.visibility_threshold = 0.3
        self.alignment_threshold = 0.05
        self.crop_threshold = 0.05

        self.overlap_threshold = 0.7
        self.remove_overlapping = True

        self.max_issues_per_type = 10

        self.debug = True
        self.output_dir = output_dir

        if self.debug:
            os.makedirs(self.output_dir, exist_ok=True)

        self.visibility_detector = VisibilityDetector(self)
        self.alignment_detector = AlignmentDetector(self)
        self.cutoff_detector = CutoffDetector(self)
        self.duplicate_detector = DuplicateDetector(self)
        self.aesthetic_detector = AestheticDetector(self)

    @timefn
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

            # JSON에서 요소 추출
            elements = layout_data.get('skeleton', {}).get('elements', [])

            if not elements:
                print("분석할 UI 요소가 없습니다.")
                return []

            if self.remove_overlapping:
                filtered_elements = self._filter_overlapping_elements(elements)
                print(f"중복 제거 후: {len(filtered_elements)}개 요소")
            else:
                filtered_elements = elements
            print(f"총 {len(filtered_elements)}개 요소 분석")

            # 화면 특성 분석 (JSON 기반)
            screen_info = self._analyze_screen_characteristics_from_json(layout_data, image)
            print(f"화면 복잡도: {screen_info['complexity']:.2f}")

            all_issues = []

            # 1. 가시성 이슈 검출 (0, 1, 2)
            visibility_issues = self.visibility_detector.detect_visibility_issues(filtered_elements, image)
            all_issues.extend(visibility_issues)
            print(f"Visibility 이슈 {len(visibility_issues)}개 검출")

            # 2. 정렬 이슈 검출 (3, 4, 5)
            alignment_issues = self.alignment_detector.detect_alignment_issues(filtered_elements, layout_data)
            all_issues.extend(alignment_issues)
            print(f"Alignment 이슈 {len(alignment_issues)}개 검출")

            # 3. 잘림 이슈 검출 (6, 7)
            cutoff_issues = self.cutoff_detector.detect_cutoff_issues(filtered_elements, image)
            all_issues.extend(cutoff_issues)
            print(f"Cut Off 이슈 {len(cutoff_issues)}개 검출")

            # 4. 중복 아이콘 검출 (8)
            duplicate_issues = self.duplicate_detector.detect_duplicate_icons(filtered_elements, image, layout_data)
            all_issues.extend(duplicate_issues)
            print(f"Duplicate Icon 이슈 {len(duplicate_issues)}개 검출")

            # 5. 심미적 이슈 검출 (9, 10, 11)
            aesthetic_issues = self.aesthetic_detector.detect_aesthetic_issues(filtered_elements, image)
            all_issues.extend(aesthetic_issues)
            print(f"Aesthetic 이슈 {len(aesthetic_issues)}개 검출")

            print(f"총 후보 이슈: {len(all_issues)}개")

            # 6. Gemini로 각 후보 이슈 검증
            verified_issues = self._verify_issues_with_gemini(all_issues, image_path, layout_data)
            print(f"Gemini 검증 완료: {len(verified_issues)}type 이슈 확인")

            # 7. 이슈 우선순위 및 필터링
            final_issues = self._prioritize_and_filter_issues(verified_issues, screen_info)
            print(f"최종 CV+Gemini {len(final_issues)}type 이슈 선별")

            # 8. 디버그 시각화
            if self.debug:
                self._save_debug_info(layout_data, final_issues, image_path)
                self.visualize_bboxes(image_path, final_issues)

            return final_issues

        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _filter_overlapping_elements(self, elements: List[Dict]) -> List[Dict]:
        """overlapping 요소 제거"""
        if len(elements) <= 1:
            return elements

        filtered = []
        used_indices = set()

        for i, element in enumerate(elements):
            if i in used_indices:
                continue

            filtered.append(element)
            used_indices.add(i)

            # 현재 요소와 겹치는 모든 요소들 제거 대상으로 표시
            for j, other_element in enumerate(elements[i + 1:], i + 1):
                if j in used_indices:
                    continue

                overlap_ratio = self._calculate_overlap_ratio(
                    element['bbox'], other_element['bbox']
                )

                if overlap_ratio >= self.overlap_threshold:
                    used_indices.add(j)  # 겹치는 요소는 제외

        return filtered

    def _calculate_overlap_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """두 bbox 겹침 비율 계산"""
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

        # 작은 영역 기준으로 겹침 비율 계산
        min_area = min(area1, area2)
        return overlap_area / min_area if min_area > 0 else 0.0

    @timefn
    def _verify_issues_with_gemini(self, candidate_issues: List[Issue], image_path: str, layout_data: Dict) -> List[Issue]:
        """Gemini로 후보 이슈를 검증하고 최종 판단"""
        verified_issues = []
        issue_descriptions = IssuePrompt()

        issues_by_type = defaultdict(list)
        for issue in candidate_issues:
            issues_by_type[issue.issue_type].append(issue)

        for issue_type, type_issues in issues_by_type.items():
            print(f"이슈 타입 {issue_type}: {len(type_issues)}개 후보 검증 중...")

            try:
                prompt = issue_descriptions.get(issue_type, "해당 이슈를 분석해주세요.")

                enhanced_prompt = f"""
                {prompt}

                **중요**: 다음 중 하나에 해당하면 반드시 명시해주세요:
                1. 이슈가 발견되지 않은 경우: "No issue found" 또는 "문제가 발견되지 않았습니다"
                2. 정상적으로 표시된 경우: "Appears normal" 또는 "정상적으로 보입니다"  
                3. 기준을 충족하는 경우: "Meets requirements" 또는 "기준을 충족합니다"

                """

                # JSON 컨텍스트 추가
                context_text = f"""
                Layout Data Context:
                {json.dumps(layout_data, indent=2, ensure_ascii=False)}

                Candidate Issues for Type {issue_type}:
                {json.dumps([{
                    'component_id': issue.component_id,
                    'component_type': issue.component_type,
                    'bbox': issue.bbox,
                } for issue in type_issues], indent=2)}
                """

                # Gemini API 호출
                gemini_response = self.generate_response(
                    prompt=enhanced_prompt,
                    image=image_path,
                    text=context_text
                )

                print(f"Gemini 응답 타입: {type(gemini_response)}")
                print(f"Gemini 응답 내용: {gemini_response}")

                # Gemini 응답 처리
                # gemini_response가 단일 Issue 객체인 경우
                if isinstance(gemini_response, Issue):
                    print(f"단일 Issue 객체 수신: {gemini_response.component_id}")

                    # 원본 후보 이슈와 매칭
                    matched_candidate = self._match_candidate_issue(gemini_response, type_issues)
                    if matched_candidate:
                        # Gemini 결과로 업데이트
                        verified_issue = self._update_issue_with_gemini_result(matched_candidate, gemini_response)
                        verified_issues.append(verified_issue)
                        print(f"이슈 확인: {verified_issue.component_id}")
                    else:
                        # 매칭되지 않아도 Gemini가 확인한 이슈이므로 추가
                        verified_issues.append(gemini_response)
                        print(f"새로운 이슈 추가: {gemini_response.component_id}")

                # gemini_response가 Issue 리스트인 경우
                elif isinstance(gemini_response, (list, tuple)):
                    print(f"Issue 리스트 수신: {len(gemini_response)}개")

                    for gemini_issue in gemini_response:
                        if isinstance(gemini_issue, Issue):
                            # 원본 후보 이슈와 매칭
                            matched_candidate = self._match_candidate_issue(gemini_issue, type_issues)
                            if matched_candidate:
                                # Gemini 결과로 업데이트
                                verified_issue = self._update_issue_with_gemini_result(matched_candidate, gemini_issue)
                                verified_issues.append(verified_issue)
                                print(f"이슈 확인: {verified_issue.component_id}")
                            else:
                                # 매칭되지 않아도 Gemini가 확인한 이슈이므로 추가
                                verified_issues.append(gemini_issue)
                                print(f"새로운 이슈 추가: {gemini_issue.component_id}")

                # gemini_response에 issues 속성이 있는 경우
                elif hasattr(gemini_response, 'issues'):
                    issues = getattr(gemini_response, 'issues', [])
                    print(f"issues 속성에서 {len(issues)}개 이슈 수신")

                    for gemini_issue in issues:
                        # 원본 후보 이슈와 매칭
                        matched_candidate = self._match_candidate_issue(gemini_issue, type_issues)
                        if matched_candidate:
                            # Gemini 결과로 업데이트
                            verified_issue = self._update_issue_with_gemini_result(matched_candidate, gemini_issue)
                            verified_issues.append(verified_issue)
                            print(f"이슈 확인: {verified_issue.component_id}")

                else:
                    print(f"이슈 타입 {issue_type}: Gemini가 이슈 없음으로 판단")
                    print(f"응답 타입: {type(gemini_response)}, 응답: {gemini_response}")

            except Exception as e:
                print(f"이슈 타입 {issue_type} Gemini 검증 실패: {e}")
                import traceback
                traceback.print_exc()

                # 실패한 경우 원본 후보들을 기본 설명과 함께 추가
                for candidate in type_issues:
                    candidate.ai_description = f"Gemini 검증 실패 - 기본 분석: {candidate.description_type}"
                    verified_issues.append(candidate)

        print(f"=== 최종 검증 결과 ===")
        print(f"총 검증된 이슈 개수: {len(verified_issues)}")
        return verified_issues

    def _match_candidate_issue(self, gemini_issue, candidate_issues: List[Issue]) -> Optional[Issue]:
        """Gemini 결과와 후보 이슈를 매칭"""

        if isinstance(gemini_issue, Issue):
            for candidate in candidate_issues:
                if gemini_issue.component_id == candidate.component_id:
                    print(f"component_id 매칭: {gemini_issue.component_id}")
                    return candidate

                # bbox 기반 매칭 (좌표가 유사한 경우)
                if self._bbox_similarity(gemini_issue.bbox, candidate.bbox) > 0.8:
                    print(f"bbox 매칭: {gemini_issue.component_id} <-> {candidate.component_id}")
                    return candidate

        else:
            for candidate in candidate_issues:
                # component_id가 일치하거나 bbox가 유사한 경우
                if (hasattr(gemini_issue, 'component_id') and
                        gemini_issue.component_id == candidate.component_id):
                    return candidate

                # bbox 기반 매칭 (좌표가 유사한 경우)
                if (hasattr(gemini_issue, 'bbox') and
                        self._bbox_similarity(gemini_issue.bbox, candidate.bbox) > 0.8):
                    return candidate

        print(f"매칭 실패: {getattr(gemini_issue, 'component_id', 'unknown')}")
        return None

    def _bbox_similarity(self, bbox1, bbox2) -> float:
        """두 bbox의 유사도 계산 (0~1)"""
        try:
            if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
                return 0.0

            # bbox 형식: [x1, y1, x2, y2]
            x1_inter = max(bbox1[0], bbox2[0])
            y1_inter = max(bbox1[1], bbox2[1])
            x2_inter = min(bbox1[2], bbox2[2])
            y2_inter = min(bbox1[3], bbox2[3])

            # 교집합 영역 계산
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                intersection = 0.0
            else:
                intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

            # 각 bbox의 영역 계산
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

            # 합집합 영역 계산
            union = area1 + area2 - intersection

            # IoU 계산
            if union == 0:
                return 0.0

            iou = intersection / union
            return min(1.0, max(0.0, iou))

        except Exception as e:
            print(f"bbox 유사도 계산 오류: {e}")
            return 0.0

    def _update_issue_with_gemini_result(self, candidate_issue: Issue, gemini_issue) -> Issue:
        """후보 이슈를 Gemini 결과로 업데이트"""

        updated_issue = Issue(
            filename=candidate_issue.filename,
            issue_type=candidate_issue.issue_type,
            component_id=candidate_issue.component_id,
            component_type=candidate_issue.component_type,
            ui_component_id=candidate_issue.ui_component_id,
            ui_component_type=candidate_issue.ui_component_type,
            score=candidate_issue.score,
            location_id=candidate_issue.location_id,
            location_type=candidate_issue.location_type,
            bbox=candidate_issue.bbox,
            description_id=candidate_issue.description_id,
            description_type=candidate_issue.description_type,
            ai_description=candidate_issue.ai_description
        )

        if isinstance(gemini_issue, Issue):
            # Gemini에서 더 상세한 정보가 있으면 업데이트
            if gemini_issue.ai_description:
                updated_issue.ai_description = gemini_issue.ai_description

            if gemini_issue.score:
                updated_issue.score = gemini_issue.score

            if gemini_issue.component_type:
                updated_issue.component_type = gemini_issue.component_type

        else:
            # Gemini 결과로 업데이트
            if hasattr(gemini_issue, 'score'):
                updated_issue.score = self._map_score(gemini_issue.score)

            if hasattr(gemini_issue, 'ai_description'):
                updated_issue.ai_description = gemini_issue.ai_description
            elif hasattr(gemini_issue, 'description'):
                updated_issue.ai_description = gemini_issue.description
            else:
                updated_issue.ai_description = f"Gemini 검증 완료 - 이슈 타입 {candidate_issue.issue_type} 확인됨"

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
        """화면 특성을 고려한 이슈 우선순위 및 필터링 (EvalKPI 활용)"""
        try:
            complexity = screen_info.get('complexity', 0.5)

            # 복잡도에 따른 이슈 개수 조정
            if complexity > 0.7:
                max_issues_per_type = 2
            elif complexity < 0.3:
                max_issues_per_type = 4
            else:
                max_issues_per_type = getattr(self, 'max_issues_per_type', 3)

            valid_issues = EvalKPI.filter_valid_issues(issues)
            print(f"유효한 이슈 필터링 후: {len(valid_issues)}개")

            final_issues = EvalKPI.prioritize_issues(valid_issues, max_issues_per_type)
            print(f"최종 우선순위 적용 후: {len(final_issues)}개")

            final_issues = self._remove_duplicate_issues(final_issues)

            return final_issues

        except Exception as e:
            print(f"이슈 필터링 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

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
                    'score': issue.score,
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

        debug_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        print(f"JSON 저장: {debug_path}")

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

    def visualize_bboxes(self, image_path: str, issues: List[Issue]):
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
                '8': (255, 128, 0),  # 주황 - 중복 아이콘
                '9': (128, 255, 0),  # 연두 - 색상 조화
                '10': (0, 128, 255),  # 하늘 - 공간 활용
                '11': (255, 0, 128)  # 분홍 - 비례 조화
            }

            valid_issues_count = 0

            for i, issue in enumerate(issues):
                try:
                    # bbox 검증
                    bbox = getattr(issue, 'bbox', None)
                    if not bbox or len(bbox) < 4:
                        print(f"이슈 {i}: bbox 데이터 없음 또는 불완전 - {bbox}")
                        continue

                    # 좌표 변환 및 검증
                    try:
                        x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox[:4], [w, h, w, h])]
                    except (ValueError, TypeError) as e:
                        print(f"이슈 {i}: 좌표 변환 실패 - bbox: {bbox}, 에러: {e}")
                        continue

                    # 좌표 유효성 검사
                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        print(f"이슈 {i}: 유효하지 않은 좌표 - ({x1}, {y1}, {x2}, {y2}), 이미지 크기: ({w}, {h})")
                        continue

                    # 이슈 타입별 색상 선택
                    issue_type = str(getattr(issue, 'issue_type', 'unknown'))
                    color = colors.get(issue_type, (128, 128, 128))  # 기본 회색

                    # 점수에 따른 두께 설정
                    score = getattr(issue, 'score', 'Conditional Pass')
                    if score in ['Fail with Critical issue', 'Fail with issue']:
                        thickness = 3
                    else:
                        thickness = 2

                    # 사각형 그리기
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                    # 라벨 텍스트
                    component_id = getattr(issue, 'component_id', 'unknown')
                    label = f"T{issue_type}:{component_id[:8]}"

                    # 텍스트 배경 사각형 그리기 (가독성 향상)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    font_thickness = 1

                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                    # 텍스트 위치 계산 (bbox 위쪽)
                    text_x = x1
                    text_y = max(text_height + 5, y1 - 5)  # 이미지 상단을 벗어나지 않도록

                    # 텍스트 배경
                    cv2.rectangle(image,
                                  (text_x - 2, text_y - text_height - 2),
                                  (text_x + text_width + 2, text_y + baseline + 2),
                                  color, -1)

                    # 텍스트 (흰색)
                    cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

                    valid_issues_count += 1

                except Exception as issue_error:
                    print(f"이슈 {i} 시각화 중 에러: {str(issue_error)}")
                    continue

            print(f"총 {len(issues)}개 이슈 중 {valid_issues_count}개 시각화 완료")

            # 결과 이미지 저장
            output_path = os.path.join(self.output_dir, f"{os.path.basename(image_path)}")
            success = cv2.imwrite(output_path, image)

            if success:
                print(f"Debug image saved: {output_path}")
            else:
                print(f"Failed to save debug image: {output_path}")

        except Exception as e:
            print(f"Error visualizing bounding boxes: {str(e)}")
            import traceback
            traceback.print_exc()

    def _safe_get_bbox_coordinates(self, bbox, image_width: int, image_height: int) -> Optional[
        Tuple[int, int, int, int]]:
        """안전하게 bbox 좌표를 추출하고 검증"""
        try:
            if not bbox or len(bbox) < 4:
                return None

            # 상대 좌표를 절대 좌표로 변환
            x1 = max(0, min(int(bbox[0] * image_width), image_width - 1))
            y1 = max(0, min(int(bbox[1] * image_height), image_height - 1))
            x2 = max(x1 + 1, min(int(bbox[2] * image_width), image_width))
            y2 = max(y1 + 1, min(int(bbox[3] * image_height), image_height))

            # 유효성 검사
            if x1 >= x2 or y1 >= y2:
                return None

            return (x1, y1, x2, y2)

        except (ValueError, TypeError, IndexError):
            return None


    def _create_issue(self, element: Dict, issue_type: int) -> Issue:
        """이슈 객체 생성 헬퍼 함수"""
        elem_type = element.get('type', 'unknown')
        bbox = element.get('bbox', [0, 0, 0, 0])
        component_id = element.get('id', 'unknown')

        ui_component_id, ui_component_type = self._calc_ui_component(elem_type)
        location_id, location_type = self._calc_location(bbox)
        description_id, description_type = self._get_description(str(issue_type))
        score = '3'
        ai_description = description_type

        return Issue(
            filename=None,
            issue_type=str(issue_type),
            component_id=component_id,
            component_type=elem_type,
            ui_component_id=ui_component_id,
            ui_component_type=ui_component_type,
            score=score,
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

    def _map_score(self, score_input) -> str:
        """점수를 표준 문자열로 변환"""
        return EvalKPI.map_score_to_string(score_input)


def save_results_to_csv(issues: List[Issue], output_path: str):
    try:
        results = []
        for issue in issues:
            # Issue 객체의 모든 필드를 dict로 변환
            if hasattr(issue, '__dict__'):
                issue_dict = issue.__dict__.copy()
            else:
                issue_dict = asdict(issue)
            results.append(issue_dict)

        df = pd.DataFrame(results)

        # 컬럼 정의
        columns = [
            'filename', 'issue_type', 'component_id', 'component_type',
            'ui_component_id', 'ui_component_type', 'score',
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


def save_tuple_data_to_csv(tuple_data, output_path: str):
    """
    튜플 형태의 데이터를 CSV로 저장하는 함수
    """
    try:
        # 데이터를 이슈별로 분리
        issues = []
        current_issue = {}

        print(f"총 튜플 개수: {len(tuple_data)}")

        for key, value in tuple_data:
            # filename이 나오면 새로운 이슈 시작 (첫 번째가 아닌 경우)
            if key == 'filename' and current_issue and 'filename' in current_issue:
                issues.append(current_issue.copy())
                current_issue = {}

            current_issue[key] = value

        # 마지막 이슈 추가
        if current_issue:
            issues.append(current_issue)

        print(f"분리된 이슈 개수: {len(issues)}")

        # 데이터 정리 및 표준화
        cleaned_issues = []
        for i, issue in enumerate(issues):
            cleaned_issue = {}

            # 필수 컬럼들 처리
            cleaned_issue['filename'] = issue.get('filename', f'unknown_{i}')
            cleaned_issue['issue_type'] = str(issue.get('issue_type', ''))
            cleaned_issue['component_id'] = str(issue.get('component_id', ''))
            cleaned_issue['component_type'] = str(issue.get('component_type', ''))
            cleaned_issue['ui_component_id'] = str(issue.get('ui_component_id', ''))
            cleaned_issue['ui_component_type'] = str(issue.get('ui_component_type', ''))
            cleaned_issue['score'] = str(issue.get('score', ''))
            cleaned_issue['location_id'] = str(issue.get('location_id', ''))
            cleaned_issue['location_type'] = str(issue.get('location_type', ''))

            # bbox 처리 - 리스트 형태로 유지
            bbox = issue.get('bbox', [])
            if isinstance(bbox, (list, tuple)):
                cleaned_issue['bbox'] = str(bbox)  # 리스트를 문자열로 변환
            else:
                cleaned_issue['bbox'] = str(bbox)
            cleaned_issue['description_id'] = str(issue.get('description_id', ''))
            cleaned_issue['description_type'] = str(issue.get('description_type', ''))
            cleaned_issue['ai_description'] = str(issue.get('ai_description', ''))

            cleaned_issues.append(cleaned_issue)

        # DataFrame 생성
        df = pd.DataFrame(cleaned_issues)

        # 컬럼 순서 정의
        column_order = [
            'filename', 'issue_type', 'component_id', 'component_type',
            'ui_component_id', 'ui_component_type', 'score',
            'location_id', 'location_type', 'bbox',
            'description_id', 'description_type', 'ai_description'
        ]

        # 존재하는 컬럼만 선택하여 순서 맞추기
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]

        print(f"DataFrame 생성: {len(df)}행 x {len(df.columns)}열")
        print(f"컬럼: {list(df.columns)}")

        # CSV 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"CSV 저장 완료: {output_path}")

        return df

    except Exception as e:
        print(f"CSV 저장 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 기존 방식 (XML + JSON)
    image_paths = glob.glob("../resource/image/*.png")

    print(f"총 {len(image_paths)}개 이미지 처리 시작...")
    all_issues = []
    first_issues = []

    for image_path in image_paths:

        filename = os.path.splitext(os.path.basename(image_path))[0]
        xml_path = f"../resource/xml/{filename}.xml"
        json_path = f"../output/json/{filename}.json"

        print(f"\n=== 처리 중: {filename} ===")
        print(f"  이미지: {image_path}")
        print(f"  JSON: {json_path}")
        print(f"  XML: {xml_path}")

        output_dir = "../output/result/gemini-2.5-flash"
        # 이슈 검출 실행
        detector = LayoutDetector(output_dir=output_dir)
        issues = detector.analyze_layout(image_path, json_path)

        # filename 설정
        image_filename = os.path.basename(image_path)
        for issue in issues:
            issue.filename = image_filename

        if issues:
            all_issues.extend(issues)

        priority = EvalKPI.select_final_priority_issue(issues, image_path)
        first_issues.extend(priority)

    if all_issues:
        output_csv = f"{output_dir}/total_issue_results.csv"
        save_results_to_csv(all_issues, output_csv)
        print(f"\n통합 결과 저장 완료: {output_csv}")
        print(f"총 {len(all_issues)}개 이슈 발견")
    else:
        print("\n발견된 이슈가 없습니다.")

    if first_issues:
        first_output_csv = f"{output_dir}/first_issue_results.csv"
        save_tuple_data_to_csv(first_issues, first_output_csv)
        print(f"\n검수 결과 저장 완료: {first_output_csv}")
        print(f"총 {len(first_issues)}개 검수 완료")
    else:
        print("\n최종 검수 중 발견된 이슈가 없습니다.")

    # JSON만 사용하는 새로운 방식 예시
    # results = batch_analyze_json_only(
    #     image_dir="./resource/image",
    #     json_dir="./output/json",
    #     output_dir="./output/enhanced_json_only_results"
    # )