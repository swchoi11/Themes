"""
UI 스켈레톤 결과 시각화 도구:  캡셔닝 이미지와 분석 결과 시각화
"""
import os
from PIL import Image, ImageDraw, ImageFont
import json
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


class Visualizer():
    """결과 시각화 클래스"""
    def __init__(self):
        # 요소 타입별 색상
        self.element_colors = {
            'text': (255, 100, 100),      # 빨강
            'button': (100, 255, 100),    # 초록
            'input': (100, 100, 255),     # 파랑
            'icon': (255, 255, 100),      # 노랑
            'container': (255, 100, 255), # 마젠타
            'image': (100, 255, 255),     # 옥색
        }

        # 레이아웃 역할별 색상
        self.role_colors = {
            'logo': (255, 235, 210),                # 크림색
            'navigation': (200, 255, 200),          # 연한 연두
            'toolbar': (200, 230, 255),             # 연하늘
            'main_content': (240, 230, 255),        # 연보라
            'content': (255, 225, 250),             # 연핑크
            'bottom_navigation': (230, 255, 240),   # 밝은 민트
            'form_elements': (255, 215, 170),       # 연한 오렌지베이지
        }

        # 레이아웃 영역별 색상
        self.region_colors = {
            'header': (255, 220, 220),              # 연한 빨강
            'navigation': (200, 255, 200),          # 연한 연두
            'sidebar': (220, 230, 255),             # 연한 블루그레이
            'sidebar_left': (255, 250, 200),        # 연한 베이지
            'sidebar_right': (200, 255, 255),       # 연한 시안
            'main_content': (240, 230, 255),        # 연보라
            'content': (255, 225, 250),             # 연핑크
            'footer': (200, 255, 200),              # 밝은 연두
            'bottom_navigation': (230, 255, 240),   # 밝은 민트
            'toolbar': (200, 230, 255),             # 연하늘
            'logo': (255, 235, 210),                # 크림색
            'form_elements': (255, 215, 170),       # 연한 오렌지베이지
        }

    def visualize_skeleton_result(self, image_path: str, result_path: str, output_dir: str):
        """전체 스켈레톤 결과를 시각화"""

        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        image = Image.open(image_path)
        os.makedirs(output_dir, exist_ok=True)

        # 1. 전체 스켈레톤 구조 시각화
        self._visualize_all_elements(image.copy(), result, os.path.join(output_dir, '01_skeleton_full.png'))

        # 2. 레이아웃 영역별 시각화
        self._visualize_layout_regions(image.copy(), result, os.path.join(output_dir, '02_layout_regions.png'))

        # 3. 역할별 시각화
        self._visualize_by_roles(image.copy(), result, os.path.join(output_dir, '03_layout_roles.png'))

        # # 4. 폼 구조 시각화
        # self._visualize_forms(image.copy(), result, os.path.join(output_dir, '04_forms_structure.png'))
        #
        # # 5. 네비게이션 시각화
        # self._visualize_navigation(image.copy(), result, os.path.join(output_dir, '05_navigation.png'))
        #
        # # 6. 계층 구조 시각화
        # self._visualize_hierarchy(image.copy(), result, os.path.join(output_dir, '06_hierarchy.png'))
        #
        # # 7. 상호작용 맵 시각화
        # self._visualize_interaction_map(image.copy(), result, os.path.join(output_dir, '07_interaction_map.png'))
        #
        # # 8. 그리드 구조 시각화
        # self._visualize_grid_structure(image.copy(), result, os.path.join(output_dir, '08_grid_structure.png'))
        #
        # # 9. 통계 대시보드
        # self._create_statistics_dashboard(result, os.path.join(output_dir, '09_statistics_dashboard.png'))
        #
        # # 10. 접근성 분석
        # self._visualize_accessibility(image.copy(), result, os.path.join(output_dir, '10_accessibility_analysis.png'))
        #
        # # 11. 상세 분석 리포트
        # self._export_detailed_analysis(result, os.path.join(output_dir, '11_detailed_analysis.txt'))

        print(f"시각화 결과가 {output_dir}에 저장되었습니다.")

    def _visualize_all_elements(self, image: Image.Image, result: Dict, output_path: str):
        """모든 UI 요소를 시각화"""
        draw = ImageDraw.Draw(image)
        w, h = image.size

        # 폰트 설정
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        for element in result['skeleton']['elements']:
            bbox = element['bbox']
            x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

            # 바운딩 박스 그리기
            color = self.element_colors.get(element['type'], 'red')
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 라벨 텍스트
            label = f"{element['type']}({element['id']})"
            if element.get('content'):
                label += f": {element['content'][:10]}..."

            # 텍스트 배경
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, y1 - text_h - 5, x1 + text_w + 5, y1], fill=color)
            draw.text((x1 + 2, y1 - text_h - 3), label, fill=(0, 0, 0), font=font)

        image.save(output_path)

    def _visualize_layout_regions(self, image: Image.Image, result: Dict, output_path: str):
        """레이아웃 영역을 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')

        w, h = image.size

        for region_name, region_info in result.get('layout_regions', {}).items():
            if region_info.get('bbox'):
                bbox = region_info['bbox']
                x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

                # 반투명 색상 오버레이
                color = self.region_colors.get(region_name, (128, 128, 128))
                color_norm = tuple(c / 255 for c in color)

                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=5, edgecolor='black',
                               facecolor=color_norm, alpha=0.2)
                ax.add_patch(rect)

                # 영역 정보 표시
                element_count = len(region_info.get('elements', []))
                label = f"{region_name.upper()}\n({element_count} elements)"
                ax.text(x1, y1, label, fontsize=5, bbox=dict(boxstyle="square", facecolor=color_norm, alpha=1))

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


    def _visualize_by_roles(self, image: Image.Image, result: Dict, output_path: str):
        """레이아웃 역할별 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Elements by Layout Role')
        ax.axis('off')

        w, h = image.size
        role_counts = {}

        for element in result['skeleton']['elements']:
            role = element.get('layout_role', 'unknown')
            role_counts[role] = role_counts.get(role, 0) + 1

            bbox = element['bbox']
            x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

            color = self.role_colors.get(role, (128, 128, 128))
            color_norm = tuple(c / 255 for c in color)

            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor=color_norm,
                           facecolor=color_norm, alpha=0.3)
            ax.add_patch(rect)

        # 범례 추가
        legend_elements = [patches.Patch(color=tuple(c / 255 for c in self.role_colors.get(role, (128, 128, 128))),
                                       label=f'{role} ({count})')
                          for role, count in role_counts.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_forms(self, image: Image.Image, result: Dict, output_path: str):
        """폼 구조 시각화 (확장)"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Forms Structure Analysis')
        ax.axis('off')

        w, h = image.size

        forms = result.get('forms', [])

        for i, form in enumerate(forms):
            # 입력 필드 강조
            for input_elem in form.get('inputs', []):
                bbox = input_elem['bbox']
                x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2, edgecolor='blue',
                               facecolor='blue', alpha=0.2)
                ax.add_patch(rect)

            # 제출 버튼 강조
            if form.get('submit_button'):
                bbox = form['submit_button']['bbox']
                x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=3, edgecolor='red',
                               facecolor='red', alpha=0.3)
                ax.add_patch(rect)
                ax.text(x1, y1 - 10, f'FORM_{i+1}', color='red', weight='bold', fontsize=12)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_navigation(self, image: Image.Image, result: Dict, output_path: str):
        """네비게이션 구조 시각화 (확장)"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Navigation Structure')
        ax.axis('off')

        w, h = image.size

        navigation = result.get('navigation', {})
        nav_type = navigation.get('type', 'unknown')

        ax.text(10, 30, f'Navigation Type: {nav_type}',
                fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))

        # 네비게이션 요소들 강조
        for element in navigation.get('elements', []):
            bbox = element['bbox']
            x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=3, edgecolor='green',
                           facecolor='green', alpha=0.2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, 'NAV', color='green', weight='bold')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_hierarchy(self, image: Image.Image, result: Dict, output_path: str):
        """계층 구조를 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('UI Hierarchy Structure')
        ax.axis('off')

        w, h = image.size
        hierarchy = result['skeleton']['hierarchy']

        # 요소 ID별 위치 매핑
        element_positions = {}
        for elem in result['skeleton']['elements']:
            bbox = elem['bbox']
            center_x = (bbox[0] + bbox[2]) / 2 * w
            center_y = (bbox[1] + bbox[3]) / 2 * h
            element_positions[elem['id']] = (center_x, center_y)

        # 계층 관계를 선으로 표시
        for parent_id, children_ids in hierarchy.items():
            if parent_id in element_positions:
                parent_pos = element_positions[parent_id]

                for child_id in children_ids:
                    if child_id in element_positions:
                        child_pos = element_positions[child_id]

                        ax.plot([parent_pos[0], child_pos[0]],
                               [parent_pos[1], child_pos[1]],
                               'r-', alpha=0.6, linewidth=2)

                        # 화살표 추가
                        ax.annotate('', xy=child_pos, xytext=parent_pos,
                                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_interaction_map(self, image: Image.Image, result: Dict, output_path: str):
        """상호작용 맵 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Interaction Map')
        ax.axis('off')

        w, h = image.size
        interaction_map = result.get('interaction_map', {})

        # 클릭 가능한 영역들 강조
        for area in interaction_map.get('clickable_areas', []):
            bbox = area['bbox']
            x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

            area_type = area['type']
            if area_type == 'button':
                color = 'orange'
            elif area_type == 'input':
                color = 'cyan'
            else:
                color = 'gray'

            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor=color,
                           facecolor=color, alpha=0.3)
            ax.add_patch(rect)

        # 범례
        legend_elements = [
            patches.Patch(color='orange', label='Clickable Buttons'),
            patches.Patch(color='cyan', label='Input Areas'),
            patches.Patch(color='gray', label='Other Interactive')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_grid_structure(self, image: Image.Image, result: Dict, output_path: str):
        """그리드 구조 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Grid Structure Analysis')
        ax.axis('off')

        w, h = image.size
        grid_structure = result.get('grid_structure', {})

        if grid_structure:
            columns = grid_structure.get('columns', 0)
            rows = grid_structure.get('rows', 0)
            cell_size = grid_structure.get('cell_size', {})

            # 그리드 라인 그리기
            if columns > 0 and rows > 0:
                cell_width = cell_size.get('width', 0) * w
                cell_height = cell_size.get('height', 0) * h

                # 세로 선들
                for i in range(columns + 1):
                    x = i * cell_width
                    ax.axvline(x=x, color='red', linestyle='--', alpha=0.7)

                # 가로 선들
                for i in range(rows + 1):
                    y = i * cell_height
                    ax.axhline(y=y, color='red', linestyle='--', alpha=0.7)

                # 그리드 정보 표시
                ax.text(10, 60, f'Grid: {columns}×{rows}',
                       fontsize=14, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _create_statistics_dashboard(self, result: Dict, output_path: str):
        """통계 대시보드 생성"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('UI Analysis Statistics Dashboard', fontsize=16, weight='bold')

        statistics = result.get('statistics', {})

        # 1. 요소 타입별 분포
        elements_by_type = statistics.get('elements_by_type', {})
        ax1.pie(elements_by_type.values(), labels=elements_by_type.keys(), autopct='%1.1f%%')
        ax1.set_title('Elements by Type')

        # 2. 역할별 분포
        elements_by_role = statistics.get('elements_by_role', {})
        ax2.bar(elements_by_role.keys(), elements_by_role.values())
        ax2.set_title('Elements by Role')
        ax2.tick_params(axis='x', rotation=45)

        # 3. 주요 지표들
        metrics = [
            ('Total Elements', statistics.get('total_elements', 0)),
            ('Coverage Ratio', f"{statistics.get('coverage_ratio', 0):.2%}"),
            ('Complexity Score', statistics.get('complexity_score', 0)),
            ('Avg Element Size', f"{statistics.get('average_element_size', 0):.4f}")
        ]

        ax3.axis('off')
        table_data = [[metric[0], str(metric[1])] for metric in metrics]
        table = ax3.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax3.set_title('Key Metrics')

        # 4. 접근성 이슈
        accessibility = result.get('accessibility', {})
        issues = {
            'Text Elements': accessibility.get('text_elements', 0),
            'Interactive Elements': accessibility.get('interactive_elements', 0),
            'Size Issues': len(accessibility.get('size_issues', [])),
            'Contrast Issues': len(accessibility.get('contrast_issues', []))
        }

        colors = ['green' if k != 'Size Issues' and k != 'Contrast Issues' else 'red' for k in issues.keys()]
        ax4.bar(issues.keys(), issues.values(), color=colors)
        ax4.set_title('Accessibility Analysis')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_accessibility(self, image: Image.Image, result: Dict, output_path: str):
        """접근성 분석 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Accessibility Analysis')
        ax.axis('off')

        w, h = image.size
        accessibility = result.get('accessibility', {})

        # 크기 이슈가 있는 요소들 강조
        size_issues = accessibility.get('size_issues', [])
        for issue in size_issues:
            # 해당 요소 찾기
            element_id = issue['id']
            for element in result['skeleton']['elements']:
                if element['id'] == element_id:
                    bbox = element['bbox']
                    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=3, edgecolor='red',
                                   facecolor='red', alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, 'SIZE ISSUE', color='red', weight='bold', fontsize=8)

        # 접근성 점수 표시
        total_elements = accessibility.get('interactive_elements', 0)
        issue_count = len(size_issues) + len(accessibility.get('contrast_issues', []))
        accessibility_score = max(0, (total_elements - issue_count) / max(1, total_elements) * 100)

        ax.text(10, 30, f'Accessibility Score: {accessibility_score:.1f}%',
                fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5",
                         facecolor='green' if accessibility_score > 80 else 'orange' if accessibility_score > 60 else 'red',
                         alpha=0.8))

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _export_detailed_analysis(self, result: Dict, output_path: str):
        """상세 분석 결과를 텍스트로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== UI 스켈레톤 분석 상세 보고서 ===\n\n")

            # 1. 기본 정보
            f.write("1. 기본 정보\n")
            f.write(f"- 구조 타입: {result['skeleton']['structure_type']}\n")
            f.write(f"- 총 요소 수: {len(result['skeleton']['elements'])}\n\n")

            # 2. 통계 정보
            statistics = result.get('statistics', {})
            f.write("2. 통계 분석\n")
            f.write(f"- 총 요소 수: {statistics.get('total_elements', 0)}\n")
            f.write(f"- 평균 요소 크기: {statistics.get('average_element_size', 0):.4f}\n")
            f.write(f"- 커버리지 비율: {statistics.get('coverage_ratio', 0):.2%}\n")
            f.write(f"- 복잡도 점수: {statistics.get('complexity_score', 0)}\n\n")

            # 3. 요소별 통계
            f.write("3. 요소 타입별 분포\n")
            elements_by_type = statistics.get('elements_by_type', {})
            for elem_type, count in elements_by_type.items():
                f.write(f"- {elem_type}: {count}개\n")
            f.write("\n")

            # 4. 역할별 통계
            f.write("4. 레이아웃 역할별 분포\n")
            elements_by_role = statistics.get('elements_by_role', {})
            for role, count in elements_by_role.items():
                f.write(f"- {role}: {count}개\n")
            f.write("\n")

            # 5. 레이아웃 영역 정보 (확장)
            f.write("5. 레이아웃 영역 분석\n")
            for region_name, region_info in result.get('layout_regions', {}).items():
                element_count = len(region_info.get('elements', []))
                f.write(f"- {region_name}: {element_count}개 요소")
                if region_info.get('bbox'):
                    bbox = region_info['bbox']
                    f.write(f" (위치: {bbox})")
                f.write("\n")
            f.write("\n")

            # 6. 폼 구조 분석 (확장)
            forms = result.get('forms', [])
            f.write("6. 폼 구조 분석\n")
            f.write(f"- 총 폼 수: {len(forms)}\n")
            for i, form in enumerate(forms):
                f.write(f"- 폼 {i + 1}:\n")
                f.write(f"  - 입력 필드: {len(form.get('inputs', []))}개\n")
                if form.get('submit_button'):
                    f.write(f"  - 제출 버튼: {form['submit_button']['id']}\n")
                f.write("\n")

            # 7. 네비게이션 구조 (확장)
            navigation = result.get('navigation', {})
            f.write("7. 네비게이션 구조\n")
            f.write(f"- 타입: {navigation.get('type', 'N/A')}\n")
            f.write(f"- 요소 수: {len(navigation.get('elements', []))}\n\n")

            # 8. 그리드 구조
            grid_structure = result.get('grid_structure', {})
            if grid_structure:
                f.write("8. 그리드 구조\n")
                f.write(f"- 타입: {grid_structure.get('type', 'N/A')}\n")
                f.write(f"- 열 수: {grid_structure.get('columns', 0)}\n")
                f.write(f"- 행 수: {grid_structure.get('rows', 0)}\n")
                f.write(f"- 셀 크기: {grid_structure.get('cell_size', {})}\n")
                f.write(f"- 포함 요소: {len(grid_structure.get('elements', []))}개\n\n")

            # 9. 상호작용 맵
            interaction_map = result.get('interaction_map', {})
            f.write("9. 상호작용 분석\n")
            f.write(f"- 클릭 가능 영역: {len(interaction_map.get('clickable_areas', []))}개\n")
            f.write(f"- 입력 영역: {len(interaction_map.get('input_areas', []))}개\n")
            f.write(f"- 스크롤 가능 영역: {len(interaction_map.get('scrollable_areas', []))}개\n\n")

            # 10. 접근성 분석
            accessibility = result.get('accessibility', {})
            f.write("10. 접근성 분석\n")
            f.write(f"- 텍스트 요소: {accessibility.get('text_elements', 0)}개\n")
            f.write(f"- 상호작용 요소: {accessibility.get('interactive_elements', 0)}개\n")
            f.write(f"- 네비게이션 요소: {accessibility.get('navigation_elements', 0)}개\n")
            f.write(f"- 폼 요소: {accessibility.get('form_elements', 0)}개\n")

            size_issues = accessibility.get('size_issues', [])
            contrast_issues = accessibility.get('contrast_issues', [])
            f.write(f"- 크기 이슈: {len(size_issues)}개\n")
            f.write(f"- 대비 이슈: {len(contrast_issues)}개\n")

            if size_issues:
                f.write("  크기 이슈 세부사항:\n")
                for issue in size_issues[:5]:  # 최대 5개만 표시
                    f.write(f"    - {issue['id']}: {issue['size']}\n")
            f.write("\n")

            # 11. 계층 구조 상세
            f.write("11. 계층 구조 상세\n")
            hierarchy = result['skeleton']['hierarchy']
            for parent_id, children_ids in hierarchy.items():
                f.write(f"- {parent_id}\n")
                for child_id in children_ids:
                    f.write(f"  └─ {child_id}\n")
            f.write("\n")

            # 12. 요소별 상세 정보 (개선)
            f.write("12. 요소 상세 정보\n")
            for element in result['skeleton']['elements'][:10]:  # 처음 10개만 표시
                f.write(f"- {element['id']} ({element['type']})\n")
                f.write(f"  위치: {element['bbox']}\n")
                if element.get('content'):
                    content = element['content'][:50] + "..." if len(element['content']) > 50 else element['content']
                    f.write(f"  내용: {content}\n")
                f.write(f"  신뢰도: {element.get('confidence', 'N/A')}\n")
                f.write(f"  상호작용 가능: {element.get('interactivity', 'N/A')}\n")
                if element.get('layout_role'):
                    f.write(f"  레이아웃 역할: {element['layout_role']}\n")
                if element.get('visual_features'):
                    vf = element['visual_features']
                    f.write(f"  평균 색상: {vf.get('avg_color', 'N/A')}\n")
                    f.write(f"  가장자리 밀도: {vf.get('edge_density', 'N/A')}\n")
                    f.write(f"  종횡비: {vf.get('aspect_ratio', 'N/A')}\n")
                f.write("\n")

            # 13. 성능 및 품질 지표
            f.write("13. 품질 지표\n")
            total_interactive = accessibility.get('interactive_elements', 0)
            total_issues = len(size_issues) + len(contrast_issues)

            if total_interactive > 0:
                accessibility_score = max(0, (total_interactive - total_issues) / total_interactive * 100)
                f.write(f"- 접근성 점수: {accessibility_score:.1f}%\n")

            complexity_score = statistics.get('complexity_score', 0)
            if complexity_score < 0.3:
                complexity_level = "낮음"
            elif complexity_score < 0.7:
                complexity_level = "보통"
            else:
                complexity_level = "높음"
            f.write(f"- 복잡도 수준: {complexity_level} ({complexity_score:.2f})\n")

            coverage = statistics.get('coverage_ratio', 0)
            f.write(f"- UI 커버리지: {coverage:.1%}\n")
            f.write("\n")


        print(f"상세 분석 리포트: {output_path} 저장")

def visualize_ui_skeleton_result(image_path: str, result_path: str, output_dir: str, cluster_output_name: str = None):
    """UI 스켈레톤 결과 시각화 편의 함수"""

    if not cluster_output_name is None:
        output_dir = os.path.join(output_dir, cluster_output_name)

    os.makedirs(output_dir, exist_ok=True)

    visualizer = Visualizer()
    visualizer.visualize_skeleton_result(image_path, result_path, output_dir)

    print(f"\n=== Enhanced 시각화 완료 ===")
    print(f"결과 위치: {output_dir}")
    print("\n생성된 파일 목록:")
    print("01_skeleton_full.png: 전체 스켈레톤 구조")
    print("02_layout_regions.png: 레이아웃 영역별 시각화")
    print("03_layout_roles.png: 레이아웃 역할별 시각화")
    # print("04_forms_structure.png: 폼 구조 분석")
    # print("05_navigation.png: 네비게이션 구조")
    # print("06_hierarchy.png: 계층 구조")
    # print("07_interaction_map.png: 상호작용 맵")
    # print("08_grid_structure.png: 그리드 구조")
    # print("19_statistics_dashboard.png: 통계 대시보드")
    # print("10_accessibility_analysis.png: 접근성 분석")
    print("11_detailed_analysis.txt: 상세 분석 리포트")

# if __name__ == "__main__":

#     image_path = "../resource/sample/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.png"
#     result_path = "../output/json/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.json"
#     output_dir = "../output/visualization"

#     visualize_ui_skeleton_result(image_path, result_path, output_dir)