import json

from typing import Dict, List, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils.schemas import UIElement, Issue
from common.prompt import UI_ISSUE, _issue_prompts
from src.gemini import Gemini


@dataclass
class UIElement:
    """UI 요소 데이터 클래스"""
    id: str
    type: str
    bbox: List[float]
    content: str
    confidence: float
    interactivity: bool
    parent_id: str
    children: List[str]
    layout_role: str
    visual_features: Dict

@dataclass
class LayoutElement:
    """레이아웃 구조 데이터 클래스"""
    skeleton: Dict
    layout_regions: Dict
    parsed_regions: Dict
    forms: List[Dict]
    navigation: Dict
    grid_structure: Dict
    interaction_map: Dict
    accessibility: Dict
    statistics: Dict


def json_parser(json_path: str) -> LayoutElement:
    """JSON 파일을 파싱하여 LayoutElement 객체로 변환"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return LayoutElement(**data)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        raise


class LayoutAnalyzer(Gemini):
    def __init__(self):
        super().__init__()
        # Setting Threshold Variables
        self.DEBUG = False
        self.alignment_threshold = 0.05
        self.visibility_threshold = 0.3
        self.icon_crop_threshold = 0.1
        self.highlight_contrast_threshold = 0.4
        self.analysis_prompts = _issue_prompts()

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """IoU 계산"""
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

    def _deduplicate_and_prioritize(self, issues: List[Issue]) -> List[Issue]:
        """이슈를 중복 제거하고 심각도(severity)에 따라 내림차순 정렬.
        tracked_issue_keys는 component_id와 issue_type의 조합 키를 저장해 중복 이슈를 방지."""
        tracked_issue_keys = set()
        unique_issues = []
        for issue in issues:
            key = f"{issue.component_id}_{issue.issue_type}"
            if key not in tracked_issue_keys:
                tracked_issue_keys.add(key)
                unique_issues.append(issue)
        return sorted(unique_issues, key=lambda x: x.severity, reverse=True)

class CompareLayout(LayoutAnalyzer):
    def __init__(self):
        super().__init__()
        self.analysis_prompts = {k: v for k, v in _issue_prompts().items() if k in ['1', '2', '3', '4', '13']}

    def analyze_comparison(self, default_image: str, themed_image: str, default_layout: LayoutElement, themed_layout: LayoutElement) -> List[Issue]:
        """Compare two layouts and detect issues [1, 2, 3, 4, 13]"""
        mapping = self.map_components(default_layout.skeleton.get('elements', []), themed_layout.skeleton.get('elements', []))
        all_issues = []

        # Gemini-based analysis
        responses = {}
        for issue_type, prompt in self.analysis_prompts.items():
            response = dict(self.generate_response(prompt, themed_image))
            responses[issue_type] = response

            if self.DEBUG:
                image = cv2.imread(themed_image)
                x1, y1, x2, y2 = response.get("bbox", [0, 0, 0, 0])
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
                description = response.get("description", "")
                cv2.namedWindow(f"Issue {issue_type}:{description}", cv2.WINDOW_NORMAL)
                cv2.imshow(f"Issue {issue_type}:{description}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Rule-based checks
        for themed_id, match in mapping.items():
            default_elem = match['default']
            themed_elem = match['themed']
            all_issues.extend(self._check_alignment_issues(default_elem, themed_elem, responses.get('1'), responses.get('2'), responses.get('3')))
            all_issues.extend(self._check_text_truncation(themed_elem, responses.get('4')))
            all_issues.extend(self._check_interactive_elements(themed_elem, responses.get('13')))

        return self._deduplicate_and_prioritize(all_issues)

    def map_components(self, default_elements: List[Dict], themed_elements: List[Dict]) -> Dict:
        """Map components between default and themed layouts"""
        mapping = {}
        content_map = {f"{elem['type']}_{elem['content']}": elem for elem in default_elements if elem.get('content')}
        for themed_elem in themed_elements:
            if themed_elem.get('content'):
                content_key = f"{themed_elem['type']}_{themed_elem['content']}"
                if content_key in content_map:
                    mapping[themed_elem['id']] = {'default': content_map[content_key], 'themed': themed_elem}
                    continue
            best_match, best_iou = None, 0
            for default_elem in default_elements:
                if default_elem['type'] == themed_elem['type']:
                    iou = self._calculate_iou(default_elem['bbox'], themed_elem['bbox'])
                    if iou > 0.3 and iou > best_iou:
                        best_match, best_iou = default_elem, iou
            if best_match:
                mapping[themed_elem['id']] = {'default': best_match, 'themed': themed_elem, 'iou': best_iou}
        return mapping

    def _check_alignment_issues(self, default_elem: Dict, themed_elem: Dict, resp1: Dict, resp2: Dict, resp3: Dict) -> List[Issue]:
        """Check alignment-related issues (1, 2, 3)"""
        issues = []
        elem_type = themed_elem.get('type', 'unknown')
        default_bbox = default_elem.get('bbox', [0, 0, 0, 0])
        themed_bbox = themed_elem.get('bbox', [0, 0, 0, 0])

        # Issue 1: Inconsistent alignment
        if elem_type in ['button', 'icon', 'text']:
            default_x_center = (default_bbox[0] + default_bbox[2]) / 2
            themed_x_center = (themed_bbox[0] + themed_bbox[2]) / 2
            if abs(default_x_center - themed_x_center) > self.alignment_threshold:
                issues.append(Issue(
                    issue_type='1',
                    component_id=themed_elem.get('id', 'unknown'),
                    component_type=elem_type,
                    severity=0.7,
                    bbox=themed_bbox,
                    description=UI_ISSUE['1'],
                    suggestion=resp1.get('text', "Align elements consistently with the reference layout.")
                ))

        # Issue 2: Non-uniform vertical/horizontal alignment
        default_y_center = (default_bbox[1] + default_bbox[3]) / 2
        themed_y_center = (themed_bbox[1] + themed_bbox[3]) / 2
        if abs(default_y_center - themed_y_center) > self.alignment_threshold:
            issues.append(Issue(
                issue_type='2',
                component_id=themed_elem.get('id', 'unknown'),
                component_type=elem_type,
                severity=0.7,
                bbox=themed_bbox,
                description=UI_ISSUE['2'],
                suggestion=resp2.get('text', "Ensure uniform vertical/horizontal alignment.")
            ))

        # Issue 3: Different alignment reference points
        if default_elem.get('layout_role') == themed_elem.get('layout_role'):
            if abs(default_bbox[0] - themed_bbox[0]) > self.alignment_threshold:
                issues.append(Issue(
                    issue_type='3',
                    component_id=themed_elem.get('id', 'unknown'),
                    component_type=elem_type,
                    severity=0.7,
                    bbox=themed_bbox,
                    description=UI_ISSUE['3'],
                    suggestion=resp3.get('text', "Use consistent alignment reference points for same-level elements.")
                ))

        return issues

    def _check_text_truncation(self, themed_elem: Dict, response: Dict) -> List[Issue]:
        """ Issue 4: Check text truncation """
        issues = []
        if themed_elem.get('type') == 'text':
            bbox = themed_elem.get('bbox', [0, 0, 0, 0])
            content = themed_elem.get('content', '')
            visual_features = themed_elem.get('visual_features', {})
            text_width = len(content) * visual_features.get('font_size', 0.02)
            bbox_width = bbox[2] - bbox[0]
            if text_width > bbox_width * 1.1:
                issues.append(Issue(
                    issue_type='4',
                    component_id=themed_elem.get('id', 'unknown'),
                    component_type='text',
                    severity=0.6,
                    bbox=bbox,
                    description=UI_ISSUE['4'],
                    suggestion=response.get('text', "Adjust text container size or reduce font size.")
                ))
        return issues

    def _check_interactive_elements(self, themed_elem: Dict, response: Dict) -> List[Issue]:
        """ Issue 13: check unclear interactive elements """
        issues = []
        if themed_elem.get('interactivity') and themed_elem.get('type') in ['button', 'input', 'toggle']:
            visual_features = themed_elem.get('visual_features', {})
            edge_density = visual_features.get('edge_density', 0.5)
            if edge_density < self.visibility_threshold:
                issues.append(Issue(
                    issue_type='13',
                    component_id=themed_elem.get('id', 'unknown'),
                    component_type=themed_elem.get('type', 'unknown'),
                    severity=0.8,
                    bbox=themed_elem.get('bbox', [0, 0, 0, 0]),
                    description=UI_ISSUE['13'],
                    suggestion=response.get('text', "Enhance visual distinction of interactive elements.")
                ))
        return issues

class SingleLayout(LayoutAnalyzer):
    def __init__(self):
        super().__init__()
        self.analysis_prompts = {k: v for k, v in _issue_prompts().items() if k in ['5', '11', '12']}

    def analyze_single_layout(self, image_path: str, layout_data: LayoutElement) -> List[Issue]:
        """ check issues [5, 11, 12] """
        all_issues = []
        responses = {}
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image at {image_path}")
            return []

        for issue_type, prompt in self.analysis_prompts.items():
            response = dict(self.generate_response(prompt, image_path))
            responses[issue_type] = response

            bbox = response.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4 or any(not isinstance(x, (int, float)) or x < 0 for x in bbox):
                print(f"Warning: Invalid bbox for Issue {issue_type}: {bbox}")
                continue

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            description = response.get("description", "")
            if self.DEBUG:
                cv2.namedWindow(f"Issue {issue_type}:{description}", cv2.WINDOW_NORMAL)
                cv2.imshow(f"Issue {issue_type}:{description}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        for elem in layout_data.skeleton.get('elements', []):
            all_issues.extend(self._check_icon_cropping(elem, responses.get('5')))
            all_issues.extend(self._check_contrast_issues(elem, responses.get('11'), responses.get('12')))

        return self._deduplicate_and_prioritize(all_issues)

    def _check_icon_cropping(self, elem: Dict, response: Dict) -> List[Issue]:
        """ Issue 5 :check icon cropping """
        issues = []
        if elem.get('type') == 'icon':
            bbox = elem.get('bbox', [0, 0, 0, 0])
            margin_left, margin_top, margin_right, margin_bottom = bbox[0], bbox[1], 1.0 - bbox[2], 1.0 - bbox[3]
            if min(margin_left, margin_top, margin_right, margin_bottom) < self.icon_crop_threshold:
                issues.append(Issue(
                    issue_type='5',
                    component_id=elem.get('id', 'unknown'),
                    component_type='icon',
                    severity=0.5,
                    bbox=bbox,
                    description=UI_ISSUE['5'],
                    suggestion=response.get('text', "Adjust icon position or add margins.")
                ))
        return issues

    def _check_contrast_issues(self, elem: Dict, resp11: Dict, resp12: Dict) -> List[Issue]:
        """ Issue 11, 12: check contrast issues """
        issues = []
        elem_type = elem.get('type', 'unknown')
        visual_features = elem.get('visual_features', {})
        bbox = elem.get('bbox', [0, 0, 0, 0])

        # Issue 11: Low text/icon contrast
        if elem_type in ['text', 'icon']:
            avg_color = visual_features.get('avg_color', [128, 128, 128])
            if isinstance(avg_color, list) and len(avg_color) >= 3:
                gray_value = 0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
                contrast = 1 - abs(gray_value - 128) / 128
                if contrast < self.highlight_contrast_threshold:
                    issues.append(Issue(
                        issue_type='11',
                        component_id=elem.get('id', 'unknown'),
                        component_type=elem_type,
                        severity=0.7,
                        bbox=bbox,
                        description=UI_ISSUE['11'],
                        suggestion=resp11.get('text', "Adjust colors to improve contrast.")
                    ))

        # Issue 12: Low highlight contrast
        if elem.get('layout_role') == 'main_content':
            avg_color = visual_features.get('avg_color', [128, 128, 128])
            if isinstance(avg_color, list) and len(avg_color) >= 3:
                r, g, b = avg_color[:3]
                saturation = (max(r, g, b) - min(r, g, b)) / max(max(r, g, b), 1)
                brightness = (r + g + b) / 3
                contrast_estimate = visual_features.get('edge_density', 0.5) * (saturation + brightness / 255) / 2
                if contrast_estimate < self.highlight_contrast_threshold:
                    issues.append(Issue(
                        issue_type='12',
                        component_id=elem.get('id', 'unknown'),
                        component_type=elem_type,
                        severity=0.7,
                        bbox=bbox,
                        description=UI_ISSUE['12'],
                        suggestion=resp12.get('text', "Adjust highlight colors to improve contrast.")
                    ))

        return issues


if __name__ == '__main__':

    try:
        # Load data
        theme_image_path = "../resource/sample/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.png"
        theme_json_path = "../output/json/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.json"
        default_image_path = "../resource/sample/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.png"
        default_json_path = "../output/json/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.json"

        # SingleLayout analysis
        single_analyzer = SingleLayout()
        theme_layout_data = json_parser(theme_json_path)
        single_issues = single_analyzer.analyze_single_layout(theme_image_path, theme_layout_data)
        print(f"Single Layout: 총 {len(single_issues)}개의 이슈가 발견되었습니다:")
        for issue in single_issues:
            print(f"- 이슈 {issue.issue_type}: {issue.description} (심각도: {issue.severity:.2f})")

        # CompareLayout analysis
        compare_analyzer = CompareLayout()
        default_layout_data = json_parser(default_json_path)
        compare_issues = compare_analyzer.analyze_comparison(default_image_path, theme_image_path, default_layout_data, theme_layout_data)
        print(f"\nCompare Layout: 총 {len(compare_issues)}개의 이슈가 발견되었습니다:")
        for issue in compare_issues:
            print(f"- 이슈 {issue.issue_type}: {issue.description} (심각도: {issue.severity:.2f})")

    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
