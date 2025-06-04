import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import cv2

from utils.schemas import Issue
from common.eval_kpi import EvalKPI
from common.prompt import single_layout_prompt, compare_layout_prompt
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

        self.max_issues_per_type = 3

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
        """우선순위를 기반으로 이슈를 필터링하고 개수를 제한"""

        # 이슈 타입별로 그룹화
        issues_by_type = {}
        for issue in issues:
            issue_type = issue.issue_type
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)

        # 각 이슈 타입별로 우선순위 정렬 및 개수 제한
        final_issues = []
        for issue_type, type_issues in issues_by_type.items():
            sorted_issues = sorted(
                type_issues,
                key=lambda x: (x.severity, x.confidence),
                reverse=True
            )

            # 해당 이슈 타입의 최대 개수만큼만 선택
            max_count = self.max_issues_per_type  # 모든 타입에 동일하게 적용
            final_issues.extend(sorted_issues[:max_count])

        return final_issues

    def _get_json_elements_with_confidence(self, layout_data: LayoutElement, issue_type: int) -> List[Dict]:

        elements = layout_data.skeleton.get('elements', [])

        # 이슈 타입에 따라 관련 요소만 선별
        relevant_elements = []
        for elem in elements:
            if self._is_relevant_for_issue_type(elem, issue_type):
                relevant_elements.append(elem)

        return relevant_elements

    def _is_relevant_for_issue_type(self, elem: Dict, issue_type: int) -> bool:
        """특정 이슈 타입과 관련된 요소인지 확인"""
        elem_type = elem.get('type', '')

        if issue_type in [0, 1]:  # 대비 문제
            return elem_type in ['text', 'icon', 'button']
        elif issue_type == 2:  # 상호작용 요소
            return elem.get('interactivity', False) and elem_type in ['button', 'input', 'toggle']
        elif issue_type in [3, 4, 5]:  # 정렬 문제
            return elem_type in ['button', 'icon', 'text', 'input']
        elif issue_type == 6:  # 텍스트 잘림
            return elem_type == 'text'
        elif issue_type == 7:  # 아이콘 크롭
            return elem_type == 'icon'

        return True

    def _calc_location(self, bbox: List[float]) -> tuple:
        """Calculate location based on bbox coordinates using EvalKPI.LOCATION
        Returns (location_id, location_type) tuple"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Determine location using 3x3 grid (TL, TC, TR, ML, MC, MR, BL, BC, BR)
        if center_y < 0.33:  # Top
            if center_x < 0.33:
                location_id = '0'  # TL
            elif center_x < 0.67:
                location_id = '1'  # TC
            else:
                location_id = '2'  # TR
        elif center_y < 0.67:  # Middle
            if center_x < 0.33:
                location_id = '3'  # ML
            elif center_x < 0.67:
                location_id = '4'  # MC
            else:
                location_id = '5'  # MR
        else:  # Bottom
            if center_x < 0.33:
                location_id = '6'  # BL
            elif center_x < 0.67:
                location_id = '7'  # BC
            else:
                location_id = '8'  # BR

        location_type = EvalKPI.LOCATION.get(location_id, 'MC')
        return location_id, location_type

    def _calc_ui_component(self, element_type: str) -> tuple:
        """UI 구성 요소 분류 using EvalKPI.UI_COMPONENT
        Returns (ui_component_id, ui_component_type) tuple"""
        # Map element_type to EvalKPI UI_COMPONENT keys
        ui_component_id = EvalKPI.UI_COMPONENT.get(element_type, '5')  # Default to TextView
        ui_component_type = EvalKPI.UI_COMPONENT.get(ui_component_id, 'TextView')
        return ui_component_id, ui_component_type

    def _get_description(self, issue_type: str) -> tuple:
        """Get description id and type for issue
        Returns (description_id, description_type) tuple"""
        description_id = issue_type
        description_type = EvalKPI.DESCRIPTION.get(issue_type, "Unknown issue")
        return description_id, description_type

    def _get_gemini_suggestion_reason(self, response: Dict, elem: Dict, issue_type: int) -> str:
        """gemini가 이슈를 감지한 이유 추출"""
        reason_parts = []

        if 'reason' in response:
            reason_parts.append(f"AI Analysis: {response['reason']}")

        # Gemini 응답의 severity 추출 및 추가
        severity = response.get('severity', 'unknown')
        if severity != 'unknown':
            reason_parts.append(f"AI Severity: {severity}")

        # 요소별 구체적인 판단 근거
        if issue_type in [0, 1]:  # 대비 문제
            visual_features = elem.get('visual_features', {})
            avg_color = visual_features.get('avg_color', [128, 128, 128])
            edge_density = visual_features.get('edge_density', 0.5)
            reason_parts.append(f"Color: {avg_color}, Edge density: {edge_density:.2f}")

        elif issue_type == 2:  # 상호작용 요소
            visual_features = elem.get('visual_features', {})
            edge_density = visual_features.get('edge_density', 0.5)
            reason_parts.append(f"Interactive element visibility: {edge_density:.2f} < {self.visibility_threshold}")

        elif issue_type in [3, 4, 5]:  # 정렬 문제
            bbox = elem.get('bbox', [0, 0, 0, 0])
            reason_parts.append(f"Element position: {bbox}, threshold: {self.alignment_threshold}")

        elif issue_type == 6:  # 텍스트 잘림
            bbox = elem.get('bbox', [0, 0, 0, 0])
            content = elem.get('content', '')
            visual_features = elem.get('visual_features', {})
            font_size = visual_features.get('font_size', 0.02)
            text_width = len(content) * font_size
            bbox_width = bbox[2] - bbox[0]
            reason_parts.append(f"Text width {text_width:.3f} > Container width {bbox_width:.3f}")

        elif issue_type == 7:  # 아이콘 크롭
            bbox = elem.get('bbox', [0, 0, 0, 0])
            margins = [bbox[0], bbox[1], 1.0 - bbox[2], 1.0 - bbox[3]]
            min_margin = min(margins)
            reason_parts.append(f"Min margin {min_margin:.3f} < threshold {self.icon_crop_threshold}")

        return " | ".join(reason_parts)


class CompareLayout(LayoutAnalyzer):
    def __init__(self):
        super().__init__()
        # Updated to use new issue numbers: 3, 4, 5, 6, 2
        self.analysis_prompts = {k: v for k, v in compare_layout_prompt().items() if k in ['2', '3', '4', '5', '6']}

    def analyze_comparison(self, default_image: str, themed_image: str, default_layout: LayoutElement, themed_layout: LayoutElement) -> List[Issue]:
        """Compare two layouts and detect issues [2, 3, 4, 5, 6]"""
        mapping = self.map_components(default_layout.skeleton.get('elements', []), themed_layout.skeleton.get('elements', []))
        all_issues = []

        # Gemini-based analysis
        responses = {}
        for issue_type, prompt in self.analysis_prompts.items():
            response = dict(self.generate_response(prompt, themed_image))
            responses[issue_type] = response

        for issue_type in ['2', '3', '4', '5', '6']:
            relevant_elements = self._get_json_elements_with_confidence(themed_layout, int(issue_type))

            for elem in relevant_elements:
                if issue_type == '2':
                    all_issues.extend(self._check_interactive_elements(elem, responses.get('2')))
                elif issue_type in ['3', '4', '5']:
                    # 매핑된 요소가 있는 경우에만 정렬 검사
                    if elem['id'] in mapping:
                        default_elem = mapping[elem['id']]['default']
                        all_issues.extend(
                            self._check_alignment_issues(default_elem, elem,
                                                         responses.get('3'), responses.get('4'), responses.get('5'))
                        )
                elif issue_type == '6':
                    all_issues.extend(self._check_text_truncation(elem, responses.get('6')))

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

    def _check_alignment_issues(self, default_elem: Dict, themed_elem: Dict, resp3: Dict, resp4: Dict, resp5: Dict) -> List[Issue]:
        """Check alignment-related issues (3, 4, 5)"""
        issues = []
        elem_type = themed_elem.get('type', 'unknown')
        default_bbox = default_elem.get('bbox', [0, 0, 0, 0])
        themed_bbox = themed_elem.get('bbox', [0, 0, 0, 0])

        # Issue 3: Inconsistent alignment
        if elem_type in ['button', 'icon', 'text']:
            default_x_center = (default_bbox[0] + default_bbox[2]) / 2
            themed_x_center = (themed_bbox[0] + themed_bbox[2]) / 2
            if abs(default_x_center - themed_x_center) > self.alignment_threshold:
                severity = resp3.get('severity')
                issue = self._create_issue(themed_elem, issue_type=3, severity=severity, suggestion=resp3.get('text', EvalKPI.DESCRIPTION['3']))
                issues.append(issue)

        # Issue 4: Non-uniform vertical/horizontal alignment
        default_y_center = (default_bbox[1] + default_bbox[3]) / 2
        themed_y_center = (themed_bbox[1] + themed_bbox[3]) / 2
        if abs(default_y_center - themed_y_center) > self.alignment_threshold:
            severity = resp4.get('severity')
            issue = self._create_issue(themed_elem, issue_type=4, severity=severity, suggestion=resp4.get('text', EvalKPI.DESCRIPTION['4']))
            issues.append(issue)

        # Issue 5: Different alignment reference points
        if default_elem.get('layout_role') == themed_elem.get('layout_role'):
            if abs(default_bbox[0] - themed_bbox[0]) > self.alignment_threshold:
                severity = resp5.get('severity')
                issue = self._create_issue(themed_elem, issue_type=5, severity=severity, suggestion=resp5.get('text', EvalKPI.DESCRIPTION['5']))
                issues.append(issue)

        return issues

    def _check_text_truncation(self, themed_elem: Dict, response: Dict) -> List[Issue]:
        """ Issue 6: Check text truncation """
        issues = []
        if themed_elem.get('type') == 'text':
            bbox = themed_elem.get('bbox', [0, 0, 0, 0])
            content = themed_elem.get('content', '')
            visual_features = themed_elem.get('visual_features', {})
            text_width = len(content) * visual_features.get('font_size', 0.02)
            bbox_width = bbox[2] - bbox[0]

            if text_width > bbox_width * 1.1:
                severity = response.get('severity')
                issue = self._create_issue(themed_elem, issue_type=6, severity=severity, suggestion=response.get('text', EvalKPI.DESCRIPTION['6']))
                issues.append(issue)
        return issues

    def _check_interactive_elements(self, themed_elem: Dict, response: Dict) -> List[Issue]:
        """ Issue 2: check unclear interactive elements """
        issues = []
        if themed_elem.get('interactivity') and themed_elem.get('type') in ['button', 'input', 'toggle']:
            visual_features = themed_elem.get('visual_features', {})
            edge_density = visual_features.get('edge_density', 0.5)

            if edge_density < self.visibility_threshold:
                severity = response.get('severity')
                issue = self._create_issue(themed_elem, issue_type=2, severity=severity, suggestion=response.get('text', EvalKPI.DESCRIPTION['2']))
                issues.append(issue)
        return issues

    def _create_issue(self, elem: Dict, issue_type: int, severity: str, suggestion: str) -> Issue:
        """이슈 객체 생성 헬퍼 함수"""
        elem_type = elem.get('type', 'unknown')
        bbox = elem.get('bbox', [0, 0, 0, 0])
        ui_component_id, ui_component_type = self._calc_ui_component(elem_type)
        location_id, location_type = self._calc_location(bbox)
        description_id, description_type = self._get_description(str(issue_type))

        return Issue(
            issue_type=issue_type,
            component_id=elem.get('id', 'unknown'),
            component_type=elem_type,
            ui_component_id=ui_component_id,
            ui_component_type=ui_component_type,
            severity=severity,
            location_id=location_id,
            location_type=location_type,
            bbox=bbox,
            description_id=description_id,
            description_type=description_type,
            suggestion=suggestion
        )


class SingleLayout(LayoutAnalyzer):
    def __init__(self):
        super().__init__()
        # Updated to use new issue numbers: 0, 1, 7
        self.analysis_prompts = {k: v for k, v in single_layout_prompt().items() if k in ['0', '1', '7']}

    def analyze_single_layout(self, image_path: str, layout_data: LayoutElement) -> List[Issue]:
        """ check issues [0, 1, 7] """
        all_issues = []
        responses = {}
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image at {image_path}")
            return []

        for issue_type, prompt in self.analysis_prompts.items():
            response = dict(self.generate_response(prompt, image_path))
            responses[issue_type] = response

        for issue_type in ['0', '1', '7']:
            relevant_elements = self._get_json_elements_with_confidence(layout_data, int(issue_type))

            for elem in relevant_elements:
                if issue_type in ['0', '1']:
                    all_issues.extend(self._check_contrast_issues(elem, responses.get('0'), responses.get('1')))
                elif issue_type == '7':
                    all_issues.extend(self._check_icon_cropping(elem, responses.get('7')))

        return self._deduplicate_and_prioritize(all_issues)

    def _check_icon_cropping(self, elem: Dict, response: Dict) -> List[Issue]:
        """ Issue 7: check icon cropping """
        issues = []
        if elem.get('type') == 'icon':
            bbox = elem.get('bbox', [0, 0, 0, 0])

            margin_left, margin_top, margin_right, margin_bottom = bbox[0], bbox[1], 1.0 - bbox[2], 1.0 - bbox[3]
            if min(margin_left, margin_top, margin_right, margin_bottom) < self.icon_crop_threshold:
                severity = response.get('severity')
                issue = self._create_issue(elem, issue_type=7, severity=severity, suggestion=response.get('text', EvalKPI.DESCRIPTION['7']))
                issues.append(issue)
        return issues

    def _check_contrast_issues(self, elem: Dict, resp0: Dict, resp1: Dict) -> List[Issue]:
        """ Issue 0, 1: check contrast issues """
        issues = []
        elem_type = elem.get('type', 'unknown')
        visual_features = elem.get('visual_features', {})

        # Issue 0: Low text/icon contrast
        if elem_type in ['text', 'icon']:
            avg_color = visual_features.get('avg_color', [128, 128, 128])
            if isinstance(avg_color, list) and len(avg_color) >= 3:
                gray_value = 0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
                contrast = 1 - abs(gray_value - 128) / 128
                if contrast < self.highlight_contrast_threshold:
                    severity = resp0.get('severity')
                    issue = self._create_issue(elem, issue_type=0, severity=severity, suggestion=resp0.get('text', EvalKPI.DESCRIPTION['0']))
                    issues.append(issue)

        # Issue 1: Low highlight contrast
        if elem.get('layout_role') == 'main_content':
            avg_color = visual_features.get('avg_color', [128, 128, 128])
            if isinstance(avg_color, list) and len(avg_color) >= 3:
                r, g, b = avg_color[:3]
                saturation = (max(r, g, b) - min(r, g, b)) / max(max(r, g, b), 1)
                brightness = (r + g + b) / 3
                contrast_estimate = visual_features.get('edge_density', 0.5) * (saturation + brightness / 255) / 2
                if contrast_estimate < self.highlight_contrast_threshold:
                    severity = resp1.get('severity')
                    issue = self._create_issue(elem, issue_type=1, severity=severity, suggestion=resp1.get('text', EvalKPI.DESCRIPTION['1']))
                    issues.append(issue)

        return issues

    def _create_issue(self, elem: Dict, issue_type: int, severity: float, suggestion: str) -> Issue:
        """이슈 객체 생성"""
        elem_type = elem.get('type', 'unknown')
        bbox = elem.get('bbox', [0, 0, 0, 0])
        ui_component_id, ui_component_type = self._calc_ui_component(elem_type)
        location_id, location_type = self._calc_location(bbox)
        description_id, description_type = self._get_description(str(issue_type))

        return Issue(
            issue_type=issue_type,
            component_id=elem.get('id', 'unknown'),
            component_type=elem_type,
            ui_component_id=ui_component_id,
            ui_component_type=ui_component_type,
            severity=severity,
            location_id=location_id,
            location_type=location_type,
            bbox=bbox,
            description_id=description_id,
            description_type=description_type,
            suggestion=suggestion
        )


if __name__ == '__main__':
    try:
        # Load data
        theme_image_path = "../resource/0530_theme_img_xml_labeled/Visibility Issue/Fail_[042]_com.sec.android.app.launcher_LauncherActivity_20250521_173924.png"
        theme_json_path = "../output/json/Fail_[042]_com.sec.android.app.launcher_LauncherActivity_20250521_173924.json"
        default_image_path = "../resource/0530_theme_img_xml_labeled/Visibility Issue/Fail_[042]_com.sec.android.app.launcher_LauncherActivity_20250521_173924.png"
        default_json_path = "../output/json/Fail_[042]_com.sec.android.app.launcher_LauncherActivity_20250521_173924.json"

        # SingleLayout analysis
        single_analyzer = SingleLayout()
        theme_layout_data = json_parser(theme_json_path)
        single_issues = single_analyzer.analyze_single_layout(theme_image_path, theme_layout_data)
        print(f"Single Layout: 총 {len(single_issues)}개의 이슈가 발견 되었습니다:")
        for issue in single_issues:
            print(f"- 이슈 {issue.issue_type}: {issue.description_type} (심각도: {issue.severity:.2f})")

        # CompareLayout analysis
        compare_analyzer = CompareLayout()
        default_layout_data = json_parser(default_json_path)
        compare_issues = compare_analyzer.analyze_comparison(default_image_path, theme_image_path, default_layout_data,
                                                             theme_layout_data)
        print(f"\nCompare Layout: 총 {len(compare_issues)}개의 이슈가 발견 되었습니다:")
        for issue in compare_issues:
            print(f"- 이슈 {issue.issue_type}: {issue.description_type} (심각도: {issue.severity:.2f})")

    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")