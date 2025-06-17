import os
import json
import ast
import cv2

from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.gemini import Gemini

gemini = Gemini()


class BoundingBoxVisualizer:
    """바운딩 박스 시각화를 위한 클래스"""

    def __init__(self):
        # 이슈 타입별 색상 정의
        self.issue_colors_mpl = {
            'normal': (0.5, 0.5, 0.5),  # 회색
            'alignment': (1.0, 0.0, 0.0),  # 빨강
            'cutoff': (0.0, 1.0, 1.0),  # 시안
            'design': (0.0, 1.0, 0.0),  # 초록
            'default': (1.0, 1.0, 0.0)  # 노랑
        }

    def parse_bbox(self, bbox_str) -> List[float]:
        """바운딩 박스 문자열을 파싱"""
        if pd.isna(bbox_str) or bbox_str == '' or bbox_str == '[]':
            return []

        try:
            if isinstance(bbox_str, str):
                bbox = ast.literal_eval(bbox_str)
            else:
                bbox = bbox_str

            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(x) for x in bbox]
            else:
                return []
        except Exception as e:
            print(f"[WARNING] bbox 파싱 실패: {bbox_str}, 오류: {e}")
            return []

    def create_image_with_bboxes(self, image_path: str, issues: List[dict]) -> str:
        """이미지에 모든 이슈의 바운딩 박스를 그려서 임시 파일로 저장"""

        # 이미지 로드
        if not os.path.exists(image_path):
            print(f"[ERROR] 이미지 파일 없음: {image_path}")
            return image_path

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 이미지 로드 실패: {image_path}")
            return image_path

        # BGR to RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 이미지 크기에 맞춰 figure 크기 조정
        dpi = 100
        fig_w = w / dpi
        fig_h = h / dpi

        # matplotlib 시각화
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(image_rgb)
        ax.axis('off')

        # 여백 제거
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # 각 이슈에 대해 바운딩 박스 그리기
        for idx, issue in enumerate(issues):
            issue_type = issue.get('issue_type', 'default')
            bbox_str = issue.get('bbox', '')

            # 바운딩 박스 파싱
            bbox = self.parse_bbox(bbox_str)

            # 색상 선택
            color = self.issue_colors_mpl.get(issue_type, self.issue_colors_mpl['default'])

            if not bbox:
                # 바운딩 박스가 없으면 전체 이미지 크기로 설정
                rect = Rectangle((2, 2), w - 4, h - 4,
                                 linewidth=3, edgecolor=color,
                                 facecolor='none', alpha=0.8)
                ax.add_patch(rect)

                # 텍스트 추가 (인덱스 포함)
                ax.text(10, 10 + idx * 30, f'BOX {idx + 1:02d}: {issue_type.upper()}',
                        ha='left', va='top',
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle="square,pad=0.3", facecolor=color, alpha=0.9))
            else:
                # 정규화된 좌표인지 확인
                if all(0 <= coord <= 1 for coord in bbox):
                    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
                else:
                    x1, y1, x2, y2 = bbox

                # 좌표 유효성 검사
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=3, edgecolor=color,
                                     facecolor='none', alpha=0.8)
                    ax.add_patch(rect)

                    # 텍스트 추가 (인덱스 포함)
                    label = f"BOX {idx + 1:02d}: {issue_type.upper()}"
                    ax.text(x1, y1, label,
                            ha='left', va='bottom',
                            fontsize=10, fontweight='bold', color='white',
                            bbox=dict(boxstyle="square,pad=0.1", facecolor=color, alpha=0.9))

        # 임시 파일에 저장
        os.makedirs('./temp', exist_ok=True)
        temp_path = f'./temp/{filename}'
        plt.savefig(temp_path, dpi=dpi, bbox_inches='tight',
                    pad_inches=0, facecolor='white', edgecolor='none')
        plt.close()
        return temp_path


def _sort_issues_by_file(image_path: str, issues: List[dict]) -> dict:
    """단일 파일의 이슈들을 정렬 - 2개 이상의 이슈가 있을 때 Gemini가 가장 중요한 것 선택"""

    issues = [issue for issue in issues if issue.get('issue_type') != 'normal']
    # 바운딩 박스 시각화 객체 생성
    visualizer = BoundingBoxVisualizer()

    # 바운딩 박스가 그려진 이미지 생성
    image_with_bbox = visualizer.create_image_with_bboxes(image_path, issues)

    # 이슈 타입 정의
    issue_types = list(set(issue.get('issue_type', 'unknown') for issue in issues))
    issue_descriptions = {
        'cutoff': 'Issue where part of an icon or button is cut off',
        'alignment': 'Issue where UI elements are not properly aligned',
        'design': 'Issue related to design consistency or visual appearance',
        'normal': 'Normal state - no issues detected',
        'default': 'General UI issue'
    }

    # 바운딩 박스 정보 생성
    box_info = []
    for idx, issue in enumerate(issues):
        box_info.append(f"BOX {idx + 1:02d}")

    # 동적 프롬프트 생성
        # 바운딩 박스 정보 생성
        box_info = []
        for idx, issue in enumerate(issues):
            issue_type = issue.get('issue_type', 'unknown')
            box_info.append(f"BOX {idx + 1:02d}: ({issue_type})")

        prompt = f"""
        안드로이드 애플리케이션의 UI 품질 검사를 수행하고 있습니다.
        UI 스크린샷을 분석하여 UI 문제점을 검토하고 보고하는 작업을 담당하고 있습니다.
    
        이미지에서 문제가 예상 되는 부분은 **색상이 있는 테두리 박스**로 표시 되어 있습니다.
        현재 이미지(스크린 샷)에는 총 {len(issues)}개의 바운딩 박스가 있습니다.
        각 바운딩 박스 위에는 "BOX 01: cutoff", "BOX 02: design" 등의 번호가 표시되어 있습니다.
    
        **검토 대상 이슈 유형:**
        {chr(10).join(f"- **{issue_type.upper()}**: {issue_descriptions.get(issue_type, '일반적인 UI 문제')}" for issue_type in issue_types)}
    
        **바운딩 박스별 예상 문제 영역:**
        {chr(10).join(box_info)}
    
        **분석 요청사항:**
        1. 전체 UI의 기능과 목적에 대한 간단한 설명
        2. 각 바운딩 박스 영역의 구체적인 문제점 분석
        3. 가장 심각하다고 판단되는 이슈 1개 선택 및 이유 제시
    
        **출력 형식:**
        ```
        ## UI 전체 설명
        [이 UI 화면의 기능과 목적 설명]
    
        ## 바운딩 박스별 이슈 분석
        {chr(10).join(f"**BOX {idx + 1:02d}**: [위치 설명] - [발견된 문제점]" for idx in range(len(issues)))}
    
        ## 최우선 이슈 선택
        **선택된 BOX**: BOX XX
        **선택 이유**: [구체적인 이유와 사용자 경험에 미치는 영향 설명]
        ```
    
        바운딩 박스 영역 내의 UI 문제만 검토하고, 한국어로 응답해주세요.
        """

    issue_text = json.dumps(issues, ensure_ascii=False, indent=2)

    try:
        # Gemini API 호출 (바운딩 박스가 그려진 이미지 사용)
        result = gemini._call_gemini_image_text(prompt, image_with_bbox, issue_text)

        # 원본 이슈 중에서 가장 우선순위가 높은 것을 선택
        # (Gemini의 응답을 기반으로 최종 이슈 결정)
        # 여기서는 첫 번째 이슈를 기본으로 하고 AI 설명만 업데이트
        selected_issue = dict(result)
        selected_issue['description'] = result.ai_description   ## result.description

        return selected_issue

    except Exception as e:
        print(f"Gemini 검증 중 오류 발생: {e}")
        # 오류 발생 시 첫 번째 이슈 반환
        return issues[0]


if __name__ == "__main__":

    dirpath = '../../resource/image'
    filepath = './resource/dataset.xlsx'
    output_file = f'./output/{gemini.model}.csv'

    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"Excel 파일을 찾을 수 없습니다: {filepath}")
        exit(1)

    df = pd.read_excel(filepath)
    df = df[~df['issue_type'].isin(['not_processed', 'no_xml'])]
    df['filename'] = df.apply(
        lambda row: str(row['label']) + os.path.basename(row['filename'])
        if pd.notna(row['label']) and row['label'] != ''
        else os.path.basename(row['filename']),
        axis=1
    )

    df['issue_type'] = df['issue_type'].apply(
        lambda x: 'normal' if str(x).startswith('normal_') else str(x)
    )

    for filename, group in df.groupby('filename'):
        image_path = os.path.join(dirpath, filename)

        if not os.path.isfile(image_path):
            print(f"not found: {image_path}")
            continue

        issues = group.to_dict('records')
        if (group['issue_type'] == 'normal').all():
            # 모두 normal이면 첫 번째 이슈를 선택
            selected_issue = issues[0]
        else:
            selected_issue = _sort_issues_by_file(image_path, issues)

        if selected_issue:
            # CSV는 mode='a'가 잘 작동함
            pd.DataFrame([selected_issue]).to_csv(
                output_file,
                mode='a',
                header=not os.path.exists(output_file),  # 첫 번째만 헤더 포함
                index=False
            )

        print("=" * 30)
        print(f"파일명:{filename}")
        print(json.dumps(selected_issue, ensure_ascii=False, indent=2))