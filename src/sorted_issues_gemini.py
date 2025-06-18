import os
import json
import ast
import cv2

from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.gemini import Gemini

from google.genai import types
import numpy as np
from tqdm import tqdm

gemini = Gemini()

# prompt_parts = [
#         # # 1. 분석 요청 안내 텍스트
#         # types.Part.from_text(text="Analyze the design applied to the following image data:"),
#         #
#         # # 2. 분석해야 할 대상 이미지
#         # types.Part.from_uri(
#         #     file_uri=image_to_analyze_uri,  # <-- 모델이 분석할 이미지
#         #     mime_type="image/png",
#         # ),
#
#         # 3. 오류 식별 안내 텍스트
#         types.Part.from_text(text="\n\nIdentify any of the following issues in the design:\n\n"),
#
#         # 4. Error 0 정의와 샘플 이미지
#         types.Part.from_text(
#             text="* Error 0): 텍스트, 아이콘과 배경 간 대비가 낮아 가독성이 떨어짐. 다음 이미지를 보면, 'Sound'라는 글자 위에 아이콘이 'Vibrate' 글자 위의 아이콘 대비 잘 안보임. \n** Sample image:"),
#         gemini.get_base_image('error0'),
#
#
#         # 5. Error 1 정의와 샘플 이미지
#         types.Part.from_text(
#             text="\n* Error 1): 하이라이트된 항목, 텍스트와 배경 간 대비가 낮아 가독성이 떨어짐. 다음 이미지를 보면, 'Contacts'라는 글자는 바로 인접한 배경색상과의 차이가 없어 잘 안보임. 'Conversations'는 잘 보임 \n** Sample image:"),
#         gemini.get_base_image('error1'),
#
#         # 5. Error 2 정의와 샘플 이미지
#         types.Part.from_text(
#             text="\n* Error 2): 상호작용 가능한 요소가 시각적으로 명확히 구분되지 않음. 사용자가 'dup'이라고 검색한 키워드가, 검색 결과에 잘 안보임. 나의 입력값이 시각적으로 잘 안보이는 경우는 이상감지 필요함. \n** Sample image:"),
#         gemini.get_base_image('error2'),
#
#         # 5. Error 3 정의와 샘플 이미지
#         types.Part.from_text(
#             text="\n* Error 3): 화면 요소들(아이콘,텍스트, 버튼 등)이 일관된 정렬 기준을 따르지 않음 (전화 버튼 Text 배열이 중앙 정렬이 되지 않을 경우 Defect로 인식) \n"),
#         # types.Part.from_uri(
#         #     file_uri="gs://jc-test-dataset/Sample_1_2.jpg", # Error 1의 예시 이미지
#         #     mime_type="image/png",
#         # ),
#
#         # 5. Error 4 정의와 샘플 이미지
#         types.Part.from_text(text="\n* Error 4): 컴포넌트 내부 요소들의 수직/수평 정렬이 균일하지 않음 \n"),
#         # types.Part.from_uri(
#         #     file_uri="gs://jc-test-dataset/Sample_1_2.jpg", # Error 1의 예시 이미지
#         #     mime_type="image/png",
#         # ),
#         # 5. Error 5 정의와 샘플 이미지
#         types.Part.from_text(text="\n* Error 5): 동일 계층 요소들 간의 정렬 기준점이 서로 다름 \n"),
#         # types.Part.from_uri(
#         #     file_uri="gs://jc-test-dataset/Sample_1_2.jpg", # Error 1의 예시 이미지
#         #     mime_type="image/png",
#         # ),
#         # 5. Error 6 정의와 샘플 이미지
#         types.Part.from_text(text="\n* Error 6): 텍스트가 할당된 영역을 초과하여 영역 외 텍스트 잘림 \n"),
#         # types.Part.from_uri(
#         #     file_uri="gs://jc-test-dataset/Sample_1_2.jpg", # Error 1의 예시 이미지
#         #     mime_type="image/png",
#         # ),
#         # 5. Error 7 정의와 샘플 이미지
#         types.Part.from_text(text="\n* Error 7): 아이콘의 가장자리가 보이지 않음거나 잘려보임(이미지 제외). 각 아이콘이 잘려서 어색하게 생긴 것을 판단해야함 \n"),
#         # types.Part.from_uri(
#         #     file_uri="gs://jc-test-dataset/Sample_1_2.jpg", # Error 1의 예시 이미지
#         #     mime_type="image/png",
#         # ),
#         # 5. Error 8 정의와 샘플 이미지
#         types.Part.from_text(
#             text="\n* Error 8): 역할이 다른 기능 요소에 동일한 아이콘 이미지로 중복 존재. 이미지를 보면, 'store'의 아이콘 모양과 'Play Store'의 아이콘 모양이 비슷하게 겹침. \n** Sample image:"),
#         gemini.get_base_image('error8'),
#
#         # 5. Error 9 정의와 샘플 이미지
#         types.Part.from_text(text="\n* Error 9): 달력 아이콘에서 요일 글자가 테두리를 벗어남. 요일 글자는 반드시 테두리 안쪽에 정확히 들어가야함 \n"),
#         # types.Part.from_uri(
#         #     file_uri="gs://jc-test-dataset/Sample_1_2.jpg", # Error 1의 예시 이미지
#         #     mime_type="image/png",
#         # ),
#         # 5. Error a 정의와 샘플 이미지
#         types.Part.from_text(text="\n* Error a): 앱 내 달력, 시간 아이콘이 현재 날짜, 시각과 매칭되지 않음 \n"),
#         # types.Part.from_uri(
#         #     file_uri="gs://jc-test-dataset/Sample_1_2.jpg", # Error 1의 예시 이미지
#         #     mime_type="image/png",
#         # ),
#         # 5. Error b 정의와 샘플 이미지
#         types.Part.from_text(text="\n* Error b): 콘텐츠와 화면 비율이 맞지 않아 불필요한 여백이 많이 발생함. \n"),
#         # types.Part.from_uri(
#         #     file_uri="gs://jc-test-dataset/Sample_1_2.jpg", # Error 1의 예시 이미지
#         #     mime_type="image/png",
#         # ),
#
#         # Normal 정의와 샘플 이미지
#         types.Part.from_text(
#             text="\n* Normal): 정상인 경우 입니다. 화면의 양 끝에 아이콘이 잘려 있더라도, 이것은 캡쳐할 때의 자연스러운 모습입니다. \n** Sample images:"),
#         gemini.get_base_image('Normal1'),
#         gemini.get_base_image('Normal2'),
#         gemini.get_base_image('Normal3'),
#
#         types.Part.from_text(text="""\n\n
#         Specify the location of the error using the following 3x3 grid system based on the mobile number keypad layout:
#
#         * 1: 좌측 상단
#         * 2: 상단 중앙
#         * 3: 우측 상단
#         * 4: 좌측 중앙
#         * 5: 정중앙
#         * 6: 우측 중앙
#         * 7: 좌측 하단
#         * 8: 하단 중앙
#         * 9: 우측 하단
#
#         If you identify any errors, respond in the following JSON format:
#
#         ```json
#         {
#         "Error#": "[Error number]",
#         "Reason":"Explain your chain of thinking",
#         "Location": "[The location of the error using the 3x3 grid system]"
#         }
#
#         If the design is normal, respond with:
#
#         ```json
#         {
#         "Error#": "Normal",
#         "Reason":"None",
#         "Location": "None"
#         }
#         ```""")
#     ]

# used_sample_links = [
#     "Fail_[100]_com.android.settings_SubSettings_20250521_174909.png",
#     "Fail_[521]_com.samsung.android.messaging_ContactPickerActivity_20250521_180841.png",
#     "Fail_[532]_com.samsung.android.messaging_ConversationComposer_20250521_174806.png",
#     "Fail_[b88]_com.sec.android.app.launcher_LauncherActivity_20250516_174703.png",
#     "com.android.intentresolver_ChooserActivityLauncher_20250513_104716.png",
#     "com.android.intentresolver_ChooserActivityLauncher_20250513_115106",
#     "com.android.settings_SubSettings_20250509_153306.png",
# ]

class BoundingBoxVisualizer:
    """바운딩 박스 시각화를 위한 클래스"""

    def __init__(self):
        # 이슈 타입별 색상 정의
        self.issue_colors_mpl = {
            'normal': (0.5, 0.5, 0.5),  # 회색
            'alignment': (1.0, 0.0, 0.0),  # 빨강
            'cutoff': (0.0, 1.0, 1.0),  # 시안bbox = issue.get('bbox', '')
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
            try:
                issue_type = issue.get('issue_type', 'default')
            except AttributeError:
                issue_type = 'alignment'
            # bbox_str = issue.get('bbox', '')
            #
            # # 바운딩 박스 파싱
            # bbox = self.parse_bbox(bbox_str)

            bbox = issue.get('bbox','')
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
        os.makedirs('testing/temp', exist_ok=True)
        temp_path = f'./temp/{os.path.basename(image_path)}'
        plt.savefig(temp_path, dpi=dpi, bbox_inches='tight',
                    pad_inches=0, facecolor='white', edgecolor='none')
        plt.close()
        return temp_path


def _sort_issues_by_file(image_path: str, issues: List[dict]) -> dict:
    """단일 파일의 이슈들을 정렬 - 2개 이상의 이슈가 있을 때 Gemini가 가장 중요한 것 선택"""

    # issues = [issue for issue in issues if issue.get('issue_type') != 'normal']
    # # 바운딩 박스 시각화 객체 생성
    # visualizer = BoundingBoxVisualizer()
    #
    # # 바운딩 박스가 그려진 이미지 생성
    # image_with_bbox = visualizer.create_image_with_bboxes(image_path, issues)
    #
    # # 이슈 타입 정의
    # issue_types = list(set(issue.get('issue_type', 'unknown') for issue in issues))
    # issue_descriptions = {
    #     'cutoff': 'Issue where part of an icon or button is cut off',
    #     'alignment': 'Issue where UI elements are not properly aligned',
    #     'design': 'Issue related to design consistency or visual appearance',
    #     'normal': 'Normal state - no issues detected',
    #     'default': 'General UI issue'
    # }
    # # 동적 프롬프트 생성
    # # 바운딩 박스 정보 생성
    # box_info = []
    # for idx, issue in enumerate(issues):
    #     issue_type = issue.get('issue_type', 'unknown')
    #     box_info.append(f"BOX {idx + 1:02d}: ({issue_type})")
    #
    # prompt = f"""
    # 안드로이드 애플리케이션의 UI 품질 검사를 수행하고 있습니다.
    # UI 스크린샷을 분석하여 UI 문제점을 검토하고 보고하는 작업을 담당하고 있습니다.
    #
    # 이미지에서 문제가 예상 되는 부분은 **색상이 있는 테두리 박스**로 표시 되어 있습니다.
    # 현재 이미지(스크린 샷)에는 총 {len(issues)}개의 바운딩 박스가 있습니다.
    # 각 바운딩 박스 위에는 "BOX 01: cutoff", "BOX 02: design" 등의 번호가 표시되어 있습니다.
    #
    # **검토 대상 이슈 유형:**
    # {chr(10).join(f"- **{issue_type.upper()}**: {issue_descriptions.get(issue_type, '일반적인 UI 문제')}" for issue_type in issue_types)}
    #
    # **바운딩 박스별 예상 문제 영역:**
    # {chr(10).join(box_info)}
    #
    # **분석 요청사항:**
    # 1. 전체 UI의 기능과 목적에 대한 간단한 설명
    # 2. 각 바운딩 박스 영역의 구체적인 문제점 분석
    # 3. 가장 심각하다고 판단되는 이슈 1개 선택 및 이유 제시
    #
    # **출력 형식:**
    # ```
    # ## UI 전체 설명
    # [이 UI 화면의 기능과 목적 설명]
    #
    # ## 바운딩 박스별 이슈 분석
    # {chr(10).join(f"**BOX {idx + 1:02d}**: [위치 설명] - [발견된 문제점]" for idx in range(len(issues)))}
    #
    # ## 최우선 이슈 선택
    # **선택된 BOX**: BOX XX
    # **선택 이유**: [구체적인 이유와 사용자 경험에 미치는 영향 설명]
    # ```
    #
    # 바운딩 박스 영역 내의 UI 문제만 검토하고, 한국어로 응답해주세요.
    # """

    prompt ="""
        You are an AI assistant that inspects the UI quality of Android applications. Please inspect the UI quality based on the screenshot of the Android application provided.
        The screenshot provided has bounding boxes that can distinguish UI components, and the names of UI components are labeled on the upper left of each bounding box (Text, Button, etc.)
        Please measure the following items for each UI component.
        - visibility: **Text**, **icons**, **imageviews**, **Image**, etc. are similar to the background color, making it difficult for people to see with their eyes.
        **Please exclude bounding boxes and labels on the top of bounding boxes from QA**
        Please also calculate the "score" for visibility issues. It has a value of 0 to 9. 9 is the highest visibility.
        **Please give a score for each UI component**
        **Please output only the 3 most problematic UI components**
        Please output in table format. The table must include the following contents.
        * UI component bound box label
        * Reason
        * Score
        Please answer in Korean.
    """

    issue_text = json.dumps(issues, ensure_ascii=False, indent=2)

    try:
        # Gemini API 호출 (바운딩 박스가 그려진 이미지 사용)
        # result = gemini._call_gemini_comp_image_text(prompt=prompt, image=image_with_bbox, text=issue_text, prompt_parts=prompt_parts)
        result = gemini._call_gemini_image(prompt=prompt, image=image_path)
        # 원본 이슈 중에서 가장 우선순위가 높은 것을 선택
        # (Gemini의 응답을 기반으로 최종 이슈 결정)
        # 여기서는 첫 번째 이슈를 기본으로 하고 AI 설명만 업데이트
        # selected_issue = result
        # selected_issue['description'] = result.ai_description   ## result.description
        # return selected_issue
        return result

    except Exception as e:
        print(f"Gemini 검증 중 오류 발생: {e}")
        # 오류 발생 시 첫 번째 이슈 반환
        return issues[0]

if __name__ == "__main__":

    dirpath = '../output/visualization'
    filepath = '../resource/dataset.xlsx'
    output_file = '../output/report.csv'

    report_filename=[]
    if os.path.exists(output_file):
        report = pd.read_csv(output_file)
        report_filename = np.unique(report['filename'])


    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"Excel 파일을 찾을 수 없습니다: {filepath}")
        exit(1)

    df = pd.read_excel(filepath)
    df = df[~df['issue_type'].isin(['no_xml', 'not_processed'])]
    df['filename'] = df.apply(
        lambda row: str(row['label']) + os.path.basename(row['filename'])
        if pd.notna(row['label']) and row['label'] != ''
        else os.path.basename(row['filename']),
        axis=1
    )

    df['issue_type'] = df['issue_type'].apply(
        lambda x: 'normal' if str(x).startswith('normal_') else str(x)
    )

    df = df[~df['filename'].isin(report_filename)]
    issue_report=[]
    try:
        total_files = len(df.groupby('filename'))
        for filename, group in tqdm(df.groupby('filename'), desc="파일 처리 중", total=total_files):

            if filename in report_filename:
                print("해당 파일은 존재합니다.")
                continue

            image_path = os.path.join(dirpath, filename)

            if not os.path.isfile(image_path):
                print(f"not found: {image_path}")
                continue

            issues = group.to_dict('records')
            # if (group['issue_type'] == 'normal').all():
            #     # 모두 normal이면 첫 번째 이슈를 선택
            #     selected_issue = issues[0]
            # else:
            #     selected_issue = _sort_issues_by_file(image_path, issues)

            selected_issue = _sort_issues_by_file(image_path, issues)
            if selected_issue:
                # CSV는 mode='a'가 잘 작동함
                pd.DataFrame([{
                    "filename":filename,
                    "selected_issue":selected_issue
                }]
                    ).to_csv(
                    output_file,
                    mode='a',
                    header=not os.path.exists(output_file),  # 첫 번째만 헤더 포함
                    index=False
                )

            print("=" * 30)
            print(f"파일명:{filename}")
            print(json.dumps(selected_issue, ensure_ascii=False, indent=2))
    except Exception as e:
        issue_report.append(f'{filename}:{e}')

    report= pd.DataFrame(issue_report)
    report.to_excel('../output/issue_report.xlsx')