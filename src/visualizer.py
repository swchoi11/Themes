"""
UI 스켈레톤 결과 시각화 도구:  캡셔닝 이미지와 분석 결과 시각화
"""
import os
from PIL import Image, ImageDraw, ImageFont
import json
from typing import Dict


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

    def visualize_skeleton_result(self, image_path: str, result_path: str, output_dir: str):
        """전체 스켈레톤 결과를 시각화"""

        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        image = Image.open(image_path)

        # 1. 전체 스켈레톤 구조 시각화
        self._visualize_all_elements(image.copy(), result, output_dir)

        # # 2. 상세 분석 리포트
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


    # def _export_detailed_analysis(self, result: Dict, output_path: str):
    #     """상세 분석 결과를 텍스트로 내보내기"""
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         f.write("=== UI 스켈레톤 분석 상세 보고서 ===\n\n")

    #         # 1. 기본 정보
    #         f.write("1. 기본 정보\n")
    #         f.write(f"- 구조 타입: {result['skeleton']['structure_type']}\n")
    #         f.write(f"- 총 요소 수: {len(result['skeleton']['elements'])}\n\n")

    #         # 2. 통계 정보
    #         statistics = result.get('statistics', {})
    #         f.write("2. 통계 분석\n")
    #         f.write(f"- 총 요소 수: {statistics.get('total_elements', 0)}\n")
    #         f.write(f"- 평균 요소 크기: {statistics.get('average_element_size', 0):.4f}\n")
    #         f.write(f"- 커버리지 비율: {statistics.get('coverage_ratio', 0):.2%}\n")
    #         f.write(f"- 복잡도 점수: {statistics.get('complexity_score', 0)}\n\n")

    #         # 3. 요소별 통계
    #         f.write("3. 요소 타입별 분포\n")
    #         elements_by_type = statistics.get('elements_by_type', {})
    #         for elem_type, count in elements_by_type.items():
    #             f.write(f"- {elem_type}: {count}개\n")
    #         f.write("\n")

    #         # 4. 역할별 통계
    #         f.write("4. 레이아웃 역할별 분포\n")
    #         elements_by_role = statistics.get('elements_by_role', {})
    #         for role, count in elements_by_role.items():
    #             f.write(f"- {role}: {count}개\n")
    #         f.write("\n")

    #         # 5. 레이아웃 영역 정보 (확장)
    #         f.write("5. 레이아웃 영역 분석\n")
    #         for region_name, region_info in result.get('layout_regions', {}).items():
    #             element_count = len(region_info.get('elements', []))
    #             f.write(f"- {region_name}: {element_count}개 요소")
    #             if region_info.get('bbox'):
    #                 bbox = region_info['bbox']
    #                 f.write(f" (위치: {bbox})")
    #             f.write("\n")
    #         f.write("\n")

    #         # 6. 폼 구조 분석 (확장)
    #         forms = result.get('forms', [])
    #         f.write("6. 폼 구조 분석\n")
    #         f.write(f"- 총 폼 수: {len(forms)}\n")
    #         for i, form in enumerate(forms):
    #             f.write(f"- 폼 {i + 1}:\n")
    #             f.write(f"  - 입력 필드: {len(form.get('inputs', []))}개\n")
    #             if form.get('submit_button'):
    #                 f.write(f"  - 제출 버튼: {form['submit_button']['id']}\n")
    #             f.write("\n")

    #         # 7. 네비게이션 구조 (확장)
    #         navigation = result.get('navigation', {})
    #         f.write("7. 네비게이션 구조\n")
    #         f.write(f"- 타입: {navigation.get('type', 'N/A')}\n")
    #         f.write(f"- 요소 수: {len(navigation.get('elements', []))}\n\n")

    #         # 8. 그리드 구조
    #         grid_structure = result.get('grid_structure', {})
    #         if grid_structure:
    #             f.write("8. 그리드 구조\n")
    #             f.write(f"- 타입: {grid_structure.get('type', 'N/A')}\n")
    #             f.write(f"- 열 수: {grid_structure.get('columns', 0)}\n")
    #             f.write(f"- 행 수: {grid_structure.get('rows', 0)}\n")
    #             f.write(f"- 셀 크기: {grid_structure.get('cell_size', {})}\n")
    #             f.write(f"- 포함 요소: {len(grid_structure.get('elements', []))}개\n\n")

    #         # 9. 상호작용 맵
    #         interaction_map = result.get('interaction_map', {})
    #         f.write("9. 상호작용 분석\n")
    #         f.write(f"- 클릭 가능 영역: {len(interaction_map.get('clickable_areas', []))}개\n")
    #         f.write(f"- 입력 영역: {len(interaction_map.get('input_areas', []))}개\n")
    #         f.write(f"- 스크롤 가능 영역: {len(interaction_map.get('scrollable_areas', []))}개\n\n")

    #         # 10. 접근성 분석
    #         accessibility = result.get('accessibility', {})
    #         f.write("10. 접근성 분석\n")
    #         f.write(f"- 텍스트 요소: {accessibility.get('text_elements', 0)}개\n")
    #         f.write(f"- 상호작용 요소: {accessibility.get('interactive_elements', 0)}개\n")
    #         f.write(f"- 네비게이션 요소: {accessibility.get('navigation_elements', 0)}개\n")
    #         f.write(f"- 폼 요소: {accessibility.get('form_elements', 0)}개\n")

    #         size_issues = accessibility.get('size_issues', [])
    #         contrast_issues = accessibility.get('contrast_issues', [])
    #         f.write(f"- 크기 이슈: {len(size_issues)}개\n")
    #         f.write(f"- 대비 이슈: {len(contrast_issues)}개\n")

    #         if size_issues:
    #             f.write("  크기 이슈 세부사항:\n")
    #             for issue in size_issues[:5]:  # 최대 5개만 표시
    #                 f.write(f"    - {issue['id']}: {issue['size']}\n")
    #         f.write("\n")

    #         # 11. 계층 구조 상세
    #         f.write("11. 계층 구조 상세\n")
    #         hierarchy = result['skeleton']['hierarchy']
    #         for parent_id, children_ids in hierarchy.items():
    #             f.write(f"- {parent_id}\n")
    #             for child_id in children_ids:
    #                 f.write(f"  └─ {child_id}\n")
    #         f.write("\n")

    #         # 12. 요소별 상세 정보 (개선)
    #         f.write("12. 요소 상세 정보\n")
    #         for element in result['skeleton']['elements'][:10]:  # 처음 10개만 표시
    #             f.write(f"- {element['id']} ({element['type']})\n")
    #             f.write(f"  위치: {element['bbox']}\n")
    #             if element.get('content'):
    #                 content = element['content'][:50] + "..." if len(element['content']) > 50 else element['content']
    #                 f.write(f"  내용: {content}\n")
    #             f.write(f"  신뢰도: {element.get('confidence', 'N/A')}\n")
    #             f.write(f"  상호작용 가능: {element.get('interactivity', 'N/A')}\n")
    #             if element.get('layout_role'):
    #                 f.write(f"  레이아웃 역할: {element['layout_role']}\n")
    #             if element.get('visual_features'):
    #                 vf = element['visual_features']
    #                 f.write(f"  평균 색상: {vf.get('avg_color', 'N/A')}\n")
    #                 f.write(f"  가장자리 밀도: {vf.get('edge_density', 'N/A')}\n")
    #                 f.write(f"  종횡비: {vf.get('aspect_ratio', 'N/A')}\n")
    #             f.write("\n")

    #         # 13. 성능 및 품질 지표
    #         f.write("13. 품질 지표\n")
    #         total_interactive = accessibility.get('interactive_elements', 0)
    #         total_issues = len(size_issues) + len(contrast_issues)

    #         if total_interactive > 0:
    #             accessibility_score = max(0, (total_interactive - total_issues) / total_interactive * 100)
    #             f.write(f"- 접근성 점수: {accessibility_score:.1f}%\n")

    #         complexity_score = statistics.get('complexity_score', 0)
    #         if complexity_score < 0.3:
    #             complexity_level = "낮음"
    #         elif complexity_score < 0.7:
    #             complexity_level = "보통"
    #         else:
    #             complexity_level = "높음"
    #         f.write(f"- 복잡도 수준: {complexity_level} ({complexity_score:.2f})\n")

    #         coverage = statistics.get('coverage_ratio', 0)
    #         f.write(f"- UI 커버리지: {coverage:.1%}\n")
    #         f.write("\n")


    #     print(f"상세 분석 리포트: {output_path} 저장")

def visualize_ui_skeleton_result(image_path: str, result_path: str, output_dir: str, cluster_output_name: str = None):
    """UI 스켈레톤 결과 시각화 편의 함수"""

    if not cluster_output_name is None:
        output_path = os.path.join(output_dir, cluster_output_name)
    else :
        output_path = output_dir

    # os.makedirs(output_dir, exist_ok=True)

    # output_path = os.path.join(output_dir, cluster_output_name)

    visualizer = Visualizer()
    visualizer.visualize_skeleton_result(image_path, result_path, output_path)
    print(f"output_dir : {output_dir}")
    print(f"output_path : {output_path}")

    print(f"\n=== Enhanced 시각화 완료 ===")
    print(f"결과 위치: {output_dir}")
    print("\n생성된 파일 목록:") 
    print("01_skeleton_full.png: 전체 스켈레톤 구조")
    # print("02_detailed_analysis.txt: 상세 분석 리포트")
 