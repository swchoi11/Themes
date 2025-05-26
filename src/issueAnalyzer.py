"""
테마 레이아웃 이슈 검출 모듈

해당 모듈은 동일한 레이아웃에 서로 다른 테마가 적용 되었을 때 발생할 수 있는
가시성(visibility), 디자인(design), 짤림(cut-off) 3가지 타입의 이슈를 검출 하는 것을 목표로 한다.
"""

import os
import cv2
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from skimage.metrics import structural_similarity as ssim

from Themes.src.easyParser import SkeletonUIExtractor, LayoutAwareParser, UIElement

@dataclass
class ThemeIssue:
    """테마 이슈 정보를 담는 데이터 클래스"""
    issue_type: str  # 'visibility', 'cut_off', 'design'
    component_id: str
    component_type: str
    severity: float  # 0.0 ~ 1.0
    bbox: List[float] 
    description: str
    screenshot_region: Optional[np.ndarray] = None

class ThemeCompAnalyzer():
    """테마 호환성 분석 클래스"""

    def __init__(self, config: Dict):
        self.config = config
        self.parser = LayoutAwareParser(config)

        self.visibility_threshold = config.get('visibility_threshold', 0.3)
        self.cut_off_threshold = config.get('cut_off_threshold', 0.2)
        self.design_threshold = config.get('design_threshold', 0.4)

    def analyze_themes(self,
                       default_image_path: str,
                       theme_image_path: str) -> Dict:
        """두 테마 이미지 간 호환성 분석 수행"""

        filename = os.path.basename(theme_image_path)

        # 1. Preprocess images
        default_image = Image.open(default_image_path)
        themed_img = Image.open(theme_image_path)

        if default_image.size != themed_img.size:
            themed_img = themed_img.resize(default_image.size)

        # 2. Extract UI elements from both images
        default_result = self.parser.parse_by_layout(default_image)
        themed_result = self.parser.parse_by_layout(themed_img)

        # 3. Mapping components between images
        component_mapping = self._map_components(
            default_result['skeleton']['elements'],
            themed_result['skeleton']['elements']
        )

        # 4. Detect issues
        visibility_issues = self._detect_visibility_issues( default_image, themed_img, component_mapping)
        cut_off_issues = self._detect_cut_off_issues( component_mapping, themed_img)
        design_issues = self._detect_design_issues(default_image, themed_img, component_mapping)

        total_issues = visibility_issues + cut_off_issues + design_issues

        # 5. Visualize issues
        if self.config.get('is_visual'):
            visualized_image = self._visualize_issues(np.array(themed_img), total_issues)
            if visualized_image.shape[2] ==4:
                visualized_image = cv2.cvtColor(visualized_image, cv2.COLOR_RGBA2BGR)

            scale_percent = 50  # 50%로 축소
            width = int(visualized_image.shape[1] * scale_percent / 100)
            height = int(visualized_image.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized_image = cv2.resize(visualized_image, dim, interpolation=cv2.INTER_AREA)
            cv2.namedWindow(f"{filename}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{filename}", resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            visualized_image = None

        summary  = {
            'total_issues_count': len(total_issues),
            'visibility_issues_count': len(visibility_issues),
            'cut_off_issues_count': len(cut_off_issues),
            'design_issues_count': len(design_issues),
            'issues': [asdict(issue) for issue in total_issues],
            'component_mapping': component_mapping,
        }
        return visualized_image, summary

    def _map_components(self,
                        default_elements: List[Dict],
                        themed_elements: List[Dict]) -> Dict:
        """두 이미지 간 동일한 컴포넌트 매핑"""
        mapping = {}

        # 텍스트 기반 매핑 (컨텐츠가 동일하면 같은 컴포넌트로 간주)
        content_map = {}
        for elem in default_elements:
            if elem.get('content'):
                content_key = f"{elem['type']}_{elem['content']}"
                content_map[content_key] = elem

        # 테마 적용된 요소들과 매핑
        for themed_elem in themed_elements:
            if themed_elem.get('content'):
                content_key = f"{themed_elem['type']}_{themed_elem['content']}"
                if content_key in content_map:
                    default_elem = content_map[content_key]
                    mapping[themed_elem['id']] = {
                        'default': default_elem,
                        'themed': themed_elem
                    }

        # 위치 기반 매핑 (컨텐츠가 없는 아이콘, 버튼 등)
        for themed_elem in themed_elements:
            if themed_elem['id'] not in mapping:
                best_match = None
                best_iou = 0

                for default_elem in default_elements:
                    if default_elem['type'] == themed_elem['type']:
                        iou = self._calculate_iou(
                            default_elem['bbox'], themed_elem['bbox']
                        )
                        if iou > 0.5 and iou > best_iou:
                            best_match = default_elem
                            best_iou = iou

                if best_match:
                    mapping[themed_elem['id']] = {
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

    def _detect_visibility_issues(self,
                                  default_image: Image.Image,
                                  themed_img: Image.Image,
                                  component_mapping: Dict) -> List[ThemeIssue]:
        """가시성 문제 검출 (텍스트/아이콘이 배경과 대비가 낮아 보이지 않는 경우)"""
        issues = []
        default_np = np.array(default_image)
        themed_np = np.array(themed_img)

        for component_id, mapping in component_mapping.items():
            themed_elem = mapping['themed']

            # 바운딩 박스 추출
            x1, y1, x2, y2 = themed_elem['bbox']
            w, h = themed_img.size
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            # 컴포넌트 영역 추출
            default_component = default_np[y1:y2, x1:x2]
            themed_component  = themed_np[y1:y2, x1:x2]

            if themed_component.size == 0 or default_component.size == 0:
                continue

            # 대비(Contrast) 분석
            themed_contrast = self._calculate_contrast(themed_component)
            default_contrast = self._calculate_contrast(default_component)

            # 테마 이미지 대비(themed_contrast)가 원본보다 낮거나(60% 미만) visibility_threshold 보다 낮으면 이슈로 판단
            contrast_ratio = themed_contrast / default_contrast if default_contrast > 0 else 1.0
            if contrast_ratio < 0.6 or themed_contrast < self.visibility_threshold:
                contrast_decrease = 1.0 - contrast_ratio
                threshold_ratio = 1.0 - (
                            themed_contrast / self.visibility_threshold) if self.visibility_threshold > 0 else 0.0
                severity = max(contrast_decrease, threshold_ratio)

                issues.append(ThemeIssue(
                    issue_type='visibility',
                    component_id=component_id,
                    component_type=themed_elem['type'],
                    severity=min(severity, 1.0),  # 심각도는 최대 1.0
                    bbox=themed_elem['bbox'],
                    description=f"낮은 가시성 (대비: {themed_contrast:.2f}, 원본: {default_contrast:.2f}, 대비 감소율: {contrast_decrease:.2f})",
                    screenshot_region=themed_component
                ))

        return issues

    def _calculate_contrast(self, img_region: np.ndarray) -> float:
        """이미지 영역의 대비 계산"""
        if img_region.size == 0:
            return 0.0

        # 그레이스케일 변환
        if len(img_region.shape) == 3:
            gray = cv2.cvtColor(img_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_region

        # 픽셀 강도의 히스토그램 계산 (0-255 범위의 256개의 구간)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()

        # 엔트로피 계산: -sum(p * log2(p))
        # 0값은 log 계산에서 제외 (log(0)는 정의되지 않음)
        non_zero_vals = hist_norm[hist_norm > 0]
        entropy = -np.sum(non_zero_vals * np.log2(non_zero_vals))

        # 엔트로피 0~1 범위 대비(Contrast) 값으로 정규화
        # 8은 256개 구간에서 모든 픽셀이 완벽하게 균등 분포일 때의 최대 엔트로피 값
        contrast = entropy / 8.0

        return contrast

    def _detect_cut_off_issues(self,
                               component_mapping: Dict,
                               themed_img: Image.Image) -> List[ThemeIssue]:
        """
        잘림 문제 검출 (컴포넌트가 부분적으로 잘리거나 제대로 표시되지 않는 경우)

        1. 컴포넌트의 크기가 원본보다 20% 이상 줄어든 경우 (width_ratio < 0.8 또는 height_ratio < 0.8)
        2. 컴포넌트가 화면 가장자리에 너무 가까운 경우:
           - 컴포넌트가 화면 가장자리의 1% 이내에 위치 (x1 < 0.01, y1 < 0.01, x2 > 0.99, y2 > 0.99)
           - 원본 위치에서 가장자리 방향으로 5% 이상 이동했을 때 (edge_shift > 0.05)

        """
        issues = []
        themed_np = np.array(themed_img)
        w, h = themed_img.size

        for component_id, mapping in component_mapping.items():
            default_elem = mapping['default']
            themed_elem = mapping['themed']

            # 크기 차이 계산
            orig_width = default_elem['bbox'][2] - default_elem['bbox'][0]
            orig_height = default_elem['bbox'][3] - default_elem['bbox'][1]

            themed_width = themed_elem['bbox'][2] - themed_elem['bbox'][0]
            themed_height = themed_elem['bbox'][3] - themed_elem['bbox'][1]

            # 너비/높이 비율 변화
            width_ratio = themed_width / orig_width if orig_width > 0 else 1.0
            height_ratio = themed_height / orig_height if orig_height > 0 else 1.0

            # 컴포넌트 영역 추출
            x1, y1, x2, y2 = themed_elem['bbox']
            x1_px, y1_px, x2_px, y2_px = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            # 바운딩 박스 유효성 검사
            if x1_px >= x2_px or y1_px >= y2_px or x2_px > w or y2_px > h:
                continue  # 유효하지 않은 바운딩 박스

            # 컴포넌트 영역 추출
            themed_component = themed_np[y1_px:y2_px, x1_px:x2_px]

            # 너무 작은 영역인 경우 건너뛰기 (면적이 50픽셀 미만 또는 한 변의 길이가 5픽셀 미만)
            if themed_component.size == 0 or \
                    themed_component.shape[0] * themed_component.shape[1] < 50 or \
                    themed_component.shape[0] < 5 or themed_component.shape[1] < 5:
                continue

            # 크기가 크게 줄어들면 잘림으로 판단 (너비나 높이가 원본의 80% 미만)
            if width_ratio < 0.8 or height_ratio < 0.8:
                severity = 1.0 - min(width_ratio, height_ratio)
                issues.append(ThemeIssue(
                    issue_type='cut_off',
                    component_id=component_id,
                    component_type=themed_elem['type'],
                    severity=severity,
                    bbox=themed_elem['bbox'],
                    description=f"컴포넌트 잘림 (너비 비율: {width_ratio:.2f}, 높이 비율: {height_ratio:.2f})",
                    screenshot_region=themed_component  # 스크린샷 영역 설정
                ))

            # 경계 영역 검사: 컴포넌트가 화면 테두리의 1% 이내에 위치하는지 확인
            is_near_edge = x1 < 0.01 or y1 < 0.01 or x2 > 0.99 or y2 > 0.99

            if is_near_edge: # 원본과 비교하여 경계에 가까워졌는지 확인
                orig_x1, orig_y1, orig_x2, orig_y2 = default_elem['bbox']

                # 가장자리로 이동 정도 계산(정규화된 좌표 기준)
                edge_shift = max(
                    orig_x1 - x1,  # 왼쪽 가장자리로 이동량
                    orig_y1 - y1,  # 위쪽 가장자리로 이동량
                    x2 - orig_x2,  # 오른쪽 가장자리로 이동량
                    y2 - orig_y2,  # 아래쪽 가장자리로 이동량
                    0
                )

                # 원본보다 5% 이상 가장자리 방향으로 이동한 경우에만 이슈로 판단
                if edge_shift > 0.05:
                    severity = min(edge_shift * 5, 1.0)
                    issues.append(ThemeIssue(
                        issue_type='cut_off',
                        component_id=component_id,
                        component_type=themed_elem['type'],
                        severity=severity,
                        bbox=themed_elem['bbox'],
                        description=f"화면 가장자리 잘림 위험 (원본에서 가장자리 방향으로 {edge_shift * 100:.1f}% 이동)",
                        screenshot_region=themed_component
                    ))

        return issues

    def _detect_design_issues(self,
                              default_image: Image.Image,
                              themed_img: Image.Image,
                              component_mapping: Dict) -> List[ThemeIssue]:
        """디자인 이슈 검출 (테마에 따라 UI 요소의 색상, 모양이 부적절하게 변경되는 경우)
            1. 원본 이미지와 테마 적용 이미지 간의 구조적 유사도(SSIM) 계산 하여 디자인 이슈를 검출 (default: 80% 미만 검출)
            2. Gemini 활용하여 어떻게 고도화 할 것인지 논의 필요!!!

        """
        issues = []
        default_np = np.array(default_image)
        themed_np = np.array(themed_img)

        for component_id, mapping in component_mapping.items():
            default_elem = mapping['default']
            themed_elem = mapping['themed']

            # 바운딩 박스 추출 (정규화된 좌표를 실제 픽셀 좌표로 변환)
            x1, y1, x2, y2 = themed_elem['bbox']
            w, h = themed_img.size
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            # 바운딩 박스 유효성 검사
            if x1 >= x2 or y1 >= y2 or x2 > w or y2 > h:
                continue

            themed_component = themed_np[y1:y2, x1:x2]

            orig_x1, orig_y1, orig_x2, orig_y2 = default_elem['bbox']
            orig_x1, orig_y1, orig_x2, orig_y2 = int(orig_x1 * w), int(orig_y1 * h), int(orig_x2 * w), int(orig_y2 * h)

            # 원본 바운딩 박스 유효성 검사
            if orig_x1 >= orig_x2 or orig_y1 >= orig_y2 or orig_x2 > w or orig_y2 > h:
                continue

            default_component = default_np[orig_y1:orig_y2, orig_x1:orig_x2]

            if themed_component.size == 0 or default_component.size == 0:
                continue

            # 원본 컴포넌트를 테마 컴포넌트 크기에 맞게 리사이징
            if default_component.shape[:2] != themed_component.shape[:2]:
                default_component = cv2.resize(default_component,
                                                (themed_component.shape[1], themed_component.shape[0]))

            # 구조적 유사도 계산
            if len(default_component.shape) == 3 and len(themed_component.shape) == 3:
                # 그레이스케일 변환
                default_gray = cv2.cvtColor(default_component, cv2.COLOR_RGB2GRAY)
                themed_gray = cv2.cvtColor(themed_component, cv2.COLOR_RGB2GRAY)

                # SSIM(Structural Similarity Index) 계산
                # SSIM은 -1~1 범위의 값을 가지며 1에 가까 울수록 유사도가 높음
                similarity, _ = ssim(default_gray, themed_gray, full=True)

                # 유사도가 낮으면 디자인 이슈로 판단
                if similarity < 0.8:  # 80% 이하면 이슈로 간주
                    severity = 1.0 - similarity
                    issues.append(ThemeIssue(
                        issue_type='design',
                        component_id=component_id,
                        component_type=themed_elem['type'],
                        severity=severity,
                        bbox=themed_elem['bbox'],
                        description=f"디자인 변경 이슈 (유사도: {similarity:.2f})",
                        screenshot_region=themed_component
                    ))

        return issues

    def _visualize_issues(self, image: np.ndarray, issues: List[ThemeIssue]) -> np.ndarray:
        """이슈 시각화"""
        vis_img = image.copy()
        h, w = vis_img.shape[:2]

        color_map = {
            'visibility': (255, 0, 0),  # 빨강
            'cut_off': (0, 0, 255),  # 파랑
            'design': (0, 255, 0)  # 초록
        }

        for issue in issues:
            x1, y1, x2, y2 = issue.bbox
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            color = color_map.get(issue.issue_type, (255, 255, 0))

            # 바운딩 박스 그리기
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # 이슈 타입 표시
            text = f"{issue.issue_type} ({issue.severity:.2f})"
            cv2.putText(vis_img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_img


class ThemeIssueAnalyzer:
    """테마 이슈 분석 클래스"""

    def __init__(self, config: Dict):
        self.config = config
        self.analyzer = ThemeCompAnalyzer(config)

    def analyze_images(self, default_image_path: str, theme_image_path: str, output_dir: str = None) -> Dict:
        vis_image, summary = self.analyzer.analyze_themes(default_image_path, theme_image_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            json_result = summary.copy()
            if 'issues' in json_result:
                json_result['issues'] = sorted(
                    json_result['issues'],
                    key=lambda x: x['severity'],
                    reverse=True
                )

                save_screenshots = self.config.get('save_screenshots', False)

                if save_screenshots:
                    screenshots_dir = os.path.join(output_dir, "screenshots")
                    os.makedirs(screenshots_dir, exist_ok=True)

                # 타입별 인덱스 카운터 초기화
                type_counters = {'visibility': 0, 'cut_off': 0, 'design': 0}

                for i, issue in enumerate(json_result['issues']):
                    if 'screenshot_region' in issue and issue['screenshot_region'] is not None:
                        if save_screenshots:
                            issue_type = issue['issue_type']
                            type_counters[issue_type] += 1

                            screenshot_path = os.path.join(
                                screenshots_dir,
                                f"issue_{i + 1}_{issue_type}.png"
                            )
                            cv2.imwrite(screenshot_path, issue['screenshot_region'])
                            issue['screenshot_path'] = os.path.relpath(screenshot_path, output_dir)

                        # 스크린샷 저장 여부와 관계없이 screenshot_region은 항상 제거
                        del issue['screenshot_region']

                json_path = os.path.join(output_dir, f"theme_total_info.json")

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, ensure_ascii=False, indent=2)

                if vis_image is not None:
                    vis_path = os.path.join(output_dir, f"theme_issues_visual.png")
                    cv2.imwrite(vis_path, vis_image)

            return json_result

    def suggest_fixes(self, issues):
        """검출된 이슈에 대한 수정 제안(예시)"""
        suggestions = []

        for issue in issues:
            if issue.issue_type == 'visibility':
                # 대비 향상 제안
                suggestions.append({
                    'issue_id': issue.component_id,
                    'fix_type': 'improve_contrast',
                    'suggestion': '배경과 전경 색상의 대비를 최소 4.5:1로 높이세요(gemini)'
                })
            elif issue.issue_type == 'cut_off':
                # 크기/위치 조정 제안
                suggestions.append({
                    'issue_id': issue.component_id,
                    'fix_type': 'resize_component',
                    'suggestion': '컴포넌트 크기를 원본과 동일하게 조정하고 화면 가장자리에서 최소 8dp 간격을 유지하세요(gemini)'
                })
            elif issue.issue_type == 'design':
                # 디자인 일관성 제안
                suggestions.append({
                    'issue_id': issue.component_id,
                    'fix_type': 'adjust_design',
                    'suggestion': '원본 테마의 디자인 언어와 일관성을 유지하면서 색상만 변경하세요(gemini)'
                })

        return suggestions

    def generate_report(self, result: Dict, output_path: str = None) -> str:
            """분석 결과 리포트 생성"""
            report = []
            report.append("# 테마 호환성 분석 리포트")
            report.append(f"\n## 요약")
            report.append(f"- 총 이슈 수: {result['total_issues_count']}")
            report.append(f"- 가시성 이슈: {result['visibility_issues_count']}")
            report.append(f"- 잘림 이슈: {result['cut_off_issues_count']}")
            report.append(f"- 디자인 이슈: {result['design_issues_count']}")

            if result['total_issues_count'] > 0:
                report.append(f"\n## 상세 이슈")

                # 심각도 순으로 정렬
                sorted_issues = sorted(
                    result['issues'],
                    key=lambda x: x['severity'],
                    reverse=True
                )

                for i, issue in enumerate(sorted_issues):
                    report.append(f"\n### 이슈 {i + 1}: {issue['issue_type']}")
                    report.append(f"- 컴포넌트 타입: {issue['component_type']}")
                    report.append(f"- 심각도: {issue['severity']:.2f}")
                    report.append(f"- 설명: {issue['description']}")
                    report.append(f"- 위치: {issue['bbox']}")
                    if 'screenshot_path' in issue:
                        report.append(f"- 스크린샷: {issue['screenshot_path']}")

            report_text = "\n".join(report)

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)

            return report_text


if __name__ == "__main__":

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # set threshold variable
    config = {
        'som_model_path': os.path.join(BASE_DIR, 'weights/icon_detect/model.pt'),
        'caption_model_name': 'florence2',
        'caption_model_path': os.path.join(BASE_DIR, 'weights/icon_caption_florence'),
        'BOX_TRESHOLD': 0.05,
        'iou_threshold': 0.7,
        'visibility_threshold': 0.4,
        'cut_off_threshold': 0.3,
        'design_threshold': 0.2,
        'save_screenshots': True,
        'is_visual': True
    }

    # image paths
    default_image = "../resource/sample/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.png"
    theme_image = "../resource/sample/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_xuka.png"
    output_dir = os.path.join(BASE_DIR, 'output/analysis', os.path.basename(theme_image))

    # processing...
    analyzer = ThemeIssueAnalyzer(config)
    result = analyzer.analyze_images( default_image, theme_image, output_dir)

    report_path = os.path.join(output_dir, "theme_issue_report.txt")
    analyzer.generate_report(result, report_path)

    print(f"분석 완료: {result['total_issues_count']}개 이슈 발견")