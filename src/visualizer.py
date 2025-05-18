"""
UI 스켈레톤 결과 시각화 도구:  캡셔닝 이미지와 분석 결과 시각화
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


class Visualizer():
    """결과 시각화 클래스"""
    def __init__(self):
        # 색상 매핑 (각 요소 타입별)
        self.color_map = {
            'text': (255, 0, 0),  # 빨강
            'button': (0, 255, 0),  # 초록
            'input': (0, 0, 255),  # 파랑
            'icon': (255, 255, 0),  # 노랑
            'container': (255, 0, 255),  # 마젠타
        }

        # 레이아웃 영역별 색상
        self.region_colors = {
            'header': (255, 200, 200),
            'navigation': (200, 255, 200),
            'sidebar': (200, 200, 255),
            'content': (255, 255, 200),
            'footer': (255, 200, 255)
        }

    def visualize_skeleton_result(self, image_path: str, result_path: str, output_dir: str):
        """스켈레톤 결과를 시각화합니다."""
        # 결과 로드
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        # 이미지 로드
        image = Image.open(image_path)
        original_image = image.copy()

        # 1. 전체 스켈레톤 구조 시각화
        self._visualize_all_elements(image, result, os.path.join(output_dir, 'skeleton_full.png'))

        # 2. 레이아웃 영역별 시각화
        self._visualize_layout_regions(original_image.copy(), result, os.path.join(output_dir, 'layout_regions.png'))

        # 3. 요소 타입별 시각화
        self._visualize_element_types(original_image.copy(), result, os.path.join(output_dir, 'element_types.png'))

        # 4. 계층 구조 시각화
        self._visualize_hierarchy(original_image.copy(), result, os.path.join(output_dir, 'hierarchy.png'))

        # 5. 상세 정보 매트릭스 생성
        self._create_info_matrix(result, os.path.join(output_dir, 'info_matrix.png'))

        # 6. 네비게이션 및 폼 구조 시각화
        self._visualize_navigation_and_forms(original_image.copy(), result, os.path.join(output_dir, 'nav_forms.png'))

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
            color = self.color_map.get(element['type'], (128, 128, 128))
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
        draw = ImageDraw.Draw(image)
        w, h = image.size

        # 반투명 오버레이 생성
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        for region_name, region_info in result['layout_regions'].items():
            if region_info['bbox']:
                bbox = region_info['bbox']
                x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

                # 반투명 색상 오버레이
                color = self.region_colors.get(region_name, (128, 128, 128))
                overlay_draw.rectangle([x1, y1, x2, y2], fill=color + (100,))

                # 영역 라벨
                label = f"{region_name.upper()} ({len(region_info['elements'])} elements)"
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                draw.text((x1 + 5, y1 + 5), label, fill=(0, 0, 0), font=font)

        # 오버레이 합성
        image = Image.alpha_composite(image.convert('RGBA'), overlay)
        image.convert('RGB').save(output_path)

    def _visualize_element_types(self, image: Image.Image, result: Dict, output_path: str):
        """요소 타입별로 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 원본 이미지 표시
        ax1.imshow(image)
        ax1.set_title('UI Elements by Type')
        ax1.axis('off')

        w, h = image.size

        # 각 타입별로 다른 색상으로 표시
        for element in result['skeleton']['elements']:
            bbox = element['bbox']
            x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

            color = self.color_map.get(element['type'], (0.5, 0.5, 0.5))
            color_norm = tuple(c / 255 for c in color)

            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor=color_norm,
                             facecolor='none', alpha=0.7)
            ax1.add_patch(rect)

        # 통계 차트 생성
        element_counts = {}
        for element in result['skeleton']['elements']:
            element_type = element['type']
            element_counts[element_type] = element_counts.get(element_type, 0) + 1

        ax2.bar(element_counts.keys(), element_counts.values(),
                color=[tuple(c / 255 for c in self.color_map.get(k, (128, 128, 128)))
                       for k in element_counts.keys()])
        ax2.set_title('Element Type Distribution')
        ax2.set_ylabel('Count')

        plt.tight_layout()
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

        # 계층 관계를 선으로 표시
        for parent_id, children_ids in hierarchy.items():
            # 부모 요소 찾기
            parent_elem = None
            for elem in result['skeleton']['elements']:
                if elem['id'] == parent_id:
                    parent_elem = elem
                    break

            if not parent_elem:
                continue

            parent_bbox = parent_elem['bbox']
            parent_center = ((parent_bbox[0] + parent_bbox[2]) / 2 * w,
                             (parent_bbox[1] + parent_bbox[3]) / 2 * h)

            # 자식 요소들과 연결선 그리기
            for child_id in children_ids:
                child_elem = None
                for elem in result['skeleton']['elements']:
                    if elem['id'] == child_id:
                        child_elem = elem
                        break

                if child_elem:
                    child_bbox = child_elem['bbox']
                    child_center = ((child_bbox[0] + child_bbox[2]) / 2 * w,
                                    (child_bbox[1] + child_bbox[3]) / 2 * h)

                    ax.plot([parent_center[0], child_center[0]],
                            [parent_center[1], child_center[1]],
                            'r-', alpha=0.6, linewidth=1)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _create_info_matrix(self, result: Dict, output_path: str):
        """상세 정보 매트릭스 생성"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # 정보 수집
        info_data = []

        # 기본 정보
        info_data.append(['Structure Type', result['skeleton']['structure_type']])
        info_data.append(['Total Elements', str(len(result['skeleton']['elements']))])

        # 레이아웃 영역별 정보
        info_data.append(['', ''])  # 빈 줄
        info_data.append(['Layout Regions', ''])
        for region_name, region_info in result['layout_regions'].items():
            count = len(region_info['elements'])
            if count > 0:
                info_data.append([f'  {region_name}', f'{count} elements'])

        # 요소 타입별 통계
        info_data.append(['', ''])  # 빈 줄
        info_data.append(['Element Types', ''])
        type_counts = {}
        for element in result['skeleton']['elements']:
            element_type = element['type']
            type_counts[element_type] = type_counts.get(element_type, 0) + 1

        for elem_type, count in type_counts.items():
            info_data.append([f'  {elem_type}', str(count)])

        # 네비게이션 정보
        if result.get('navigation'):
            info_data.append(['', ''])  # 빈 줄
            info_data.append(['Navigation', ''])
            info_data.append(['  Type', result['navigation'].get('type', 'N/A')])
            info_data.append(['  Elements', str(len(result['navigation'].get('elements', [])))])

        # 폼 정보
        if result.get('forms'):
            info_data.append(['', ''])  # 빈 줄
            info_data.append(['Forms', str(len(result['forms']))])

        # 테이블 생성
        table = ax.table(cellText=info_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # 스타일링
        for i, row in enumerate(info_data):
            if row[0] and not row[0].startswith('  '):
                # 헤더 스타일
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            elif row[0].startswith('  '):
                # 서브 항목 스타일
                table[(i, 0)].set_facecolor('#f5f5f5')
                table[(i, 1)].set_facecolor('#f5f5f5')

        ax.set_title('UI Analysis Summary', fontsize=16, weight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_navigation_and_forms(self, image: Image.Image, result: Dict, output_path: str):
        """네비게이션과 폼 구조 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Navigation and Forms Structure')
        ax.axis('off')

        w, h = image.size

        # 네비게이션 요소 강조
        if result.get('navigation') and result['navigation'].get('elements'):
            for element in result['navigation']['elements']:
                bbox = element['bbox']
                x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=3, edgecolor='green',
                                 facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, 'NAV', color='green', weight='bold')

        # 폼 요소 강조
        if result.get('forms'):
            for i, form in enumerate(result['forms']):
                # 입력 필드 강조
                for input_elem in form.get('inputs', []):
                    bbox = input_elem['bbox']
                    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * w

                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='blue',
                                     facecolor='none', alpha=0.6)
                    ax.add_patch(rect)

                # 제출 버튼 강조
                if form.get('submit_button'):
                    bbox = form['submit_button']['bbox']
                    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=3, edgecolor='red',
                                     facecolor='none', alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f'FORM_{i}', color='red', weight='bold')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def export_detailed_analysis(self, result: Dict, output_path: str):
        """상세 분석 결과를 텍스트로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== UI 스켈레톤 분석 상세 보고서 ===\n\n")

            # 1. 기본 정보
            f.write("1. 기본 정보\n")
            f.write(f"- 구조 타입: {result['skeleton']['structure_type']}\n")
            f.write(f"- 총 요소 수: {len(result['skeleton']['elements'])}\n\n")

            # 2. 레이아웃 영역 분석
            f.write("2. 레이아웃 영역 분석\n")
            for region_name, region_info in result['layout_regions'].items():
                if region_info['elements']:
                    f.write(f"- {region_name}: {len(region_info['elements'])}개 요소\n")
                    if region_info['bbox']:
                        f.write(f"  위치: {region_info['bbox']}\n")
            f.write("\n")

            # 3. 요소별 상세 정보
            f.write("3. 요소 상세 정보\n")
            for element in result['skeleton']['elements']:
                f.write(f"- {element['id']} ({element['type']})\n")
                f.write(f"  위치: {element['bbox']}\n")
                if element.get('content'):
                    f.write(f"  내용: {element['content']}\n")
                f.write(f"  신뢰도: {element['confidence']}\n")
                f.write(f"  상호작용 가능: {element['interactivity']}\n")
                if element.get('layout_role'):
                    f.write(f"  레이아웃 역할: {element['layout_role']}\n")
                f.write("\n")

            # 4. 계층 구조
            f.write("4. 계층 구조\n")
            hierarchy = result['skeleton']['hierarchy']
            for parent_id, children_ids in hierarchy.items():
                f.write(f"- {parent_id}\n")
                for child_id in children_ids:
                    f.write(f"  └─ {child_id}\n")
            f.write("\n")

            # 5. 네비게이션 정보
            if result.get('navigation'):
                f.write("5. 네비게이션 구조\n")
                f.write(f"- 타입: {result['navigation'].get('type', 'N/A')}\n")
                f.write(f"- 요소 수: {len(result['navigation'].get('elements', []))}\n\n")

            # 6. 폼 정보
            if result.get('forms'):
                f.write("6. 폼 구조\n")
                for i, form in enumerate(result['forms']):
                    f.write(f"- 폼 {i + 1}\n")
                    f.write(f"  입력 필드: {len(form.get('inputs', []))}\n")
                    if form.get('submit_button'):
                        f.write(f"  제출 버튼: {form['submit_button']['id']}\n")
                f.write("\n")


# 편의 함수
def visualize_ui_skeleton_result(image_path: str, result_path: str, output_dir: str):
    """UI 스켈레톤 결과 시각화 편의 함수"""
    visualizer = Visualizer()
    visualizer.visualize_skeleton_result(image_path, result_path, output_dir)

    # 상세 분석 리포트 생성
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    report_path = os.path.join(output_dir, 'detailed_analysis.txt')
    visualizer.export_detailed_analysis(result, report_path)

    print(f"시각화 완료! 결과는 {output_dir}에서 확인하세요.")
    print("생성된 파일:")
    print("- skeleton_full.png: 전체 스켈레톤 구조")
    print("- layout_regions.png: 레이아웃 영역별 시각화")
    print("- element_types.png: 요소 타입별 분포")
    print("- hierarchy.png: 계층 구조")
    print("- info_matrix.png: 분석 요약 매트릭스")
    print("- nav_forms.png: 네비게이션 및 폼 구조")
    print("- detailed_analysis.txt: 상세 분석 리포트")


# 사용 예제
if __name__ == "__main__":
    # 예제 사용법
    image_path = "../resource/sample/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.png"
    result_path = "../output/json/com.android.settings_SubSettings_20250509_160428_settings_checkbox_cut_Default_xuka.json"
    output_dir = "../output/visualization"

    os.makedirs(output_dir, exist_ok=True)
    visualize_ui_skeleton_result(image_path, result_path, output_dir)