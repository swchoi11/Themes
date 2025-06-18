import os
import cv2
import json
import re

import pandas as pd
import numpy as np


def process_score(score):
    """
    스코어 값을 처리하는 함수
    - '/'가 있는 경우: 앞의 숫자를 가져와서 +1
    - 5점 이상인 경우: 5로 제한
    """
    if pd.isna(score):
        return score

    score_str = str(score)

    if '/' in score_str:
        front_num = int(score_str.split('/')[0])
        processed_score = front_num + 1
    else:
        processed_score = int(score_str) + 1

    # 5점 이상인 경우 5로 제한
    if processed_score >= 5:
        processed_score = 5

    return processed_score


def calculate_similarity(str1, str2):
    """
    두 문자열 간의 유사도를 계산 (Levenshtein Distance 기반)
    0~1 사이의 값으로 반환 (1에 가까울수록 유사)
    """

    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0

    distance = levenshtein_distance(str1.lower(), str2.lower())
    return 1 - (distance / max_len)


def extract_ui_component_name(ui_component_string):

    # 미리 정의된 UI 컴포넌트 목록
    ui_components = [
        'Button',  # 클릭 가능한 일반 버튼
        'ImageView',  # 이미지가 표시된 뷰
        'RadioButton',  # 단일 선택 가능한 동그란 선택 버튼
        'CheckBox',  # 다중 선택 가능한 사각형 선택 버튼
        'EditText',  # 텍스트 입력 필드
        'TextView',  # 읽기 전용 텍스트
        'Switch',  # 토글 가능한 스위치
        'ToggleButton',  # 눌러서 상태 전환이 되는 버튼 형태의 스위치
        'SeekBar',  # 수평 슬라이더, 값 조절 가능
        'ProgressBar',  # 로딩 상태 등을 표시하는 진행 바
        'Spinner',  # 클릭 시 목록이 뜨는 드롭다운 선택 박스
        'ImageButton',  # 이미지로 된 버튼
    ]

    if pd.isna(ui_component_string):
        return 'Button'  # 기본값을 가장 일반적인 Button으로 설정

    ui_string = str(ui_component_string)

    # 1단계: 괄호 안의 값 추출 시도
    match = re.search(r'\(([^)]+)\)', ui_string)
    if match:
        extracted_value = match.group(1)
        # '_' 기준으로 나누고 첫 번째 값 반환
        split_values = extracted_value.split('_')
        first_part = split_values[0]

        # 추출된 값이 이미 목록에 있으면 그대로 반환
        if first_part in ui_components:
            return first_part

        # 추출된 값과 가장 유사한 컴포넌트 찾기
        best_match = None
        best_similarity = 0

        for component in ui_components:
            similarity = calculate_similarity(first_part, component)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = component

        return best_match if best_match else 'Button'

    # 2단계: 괄호 추출 실패시 전체 문자열과 가장 유사한 컴포넌트 찾기
    best_match = None
    best_similarity = 0

    for component in ui_components:
        # 부분 문자열 매칭도 고려
        if component.lower() in ui_string.lower():
            return component

        # 유사도 계산
        similarity = calculate_similarity(ui_string, component)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = component

    # 최소 유사도 임계값 (0.3 이상이면 유사하다고 판단)
    if best_similarity >= 0.3:
        return best_match

    # 3단계: 유사도가 낮으면 기본값 반환
    return 'Button'


jsonRoot = '../../output/json'
imageRoot = '../../resource/image'

report = pd.read_csv('./final_report.csv', on_bad_lines='skip')
report = report.replace(r'^\s*$', pd.NA, regex=True)
report = report.dropna(how='all')
report['score'] = report['score'].apply(process_score)
print(np.unique(report['score']))

report['itemName'] = report['ui_component'].apply(extract_ui_component_name)

rows=[]
for idx, group in report.groupby('filename'):
    group = group.sort_values('score')
    first = group.iloc[0]
    gt = 'Fail' in first['filename']
    pred = first['score']<3
    row = {
        'FileName':first['filename'],
        'GroundTruth': gt,
        'Predict': pred,
        'Match': gt==pred,
        'Score': first['score'],
        'itemName': first['itemName'],
        'Location': 'null',
        'Description':first['description'],
        'Reason':first['description'],
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('final_kpi_matrix.csv')


def draw_bbox_on_image(image_path, json_path, output_path=None):
    """
    이미지에 JSON의 UI 컴포넌트 bbox를 그려서 저장하는 함수
    """
    try:
        # JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return False

        height, width = image.shape[:2]

        # skeleton의 elements에서 bbox 정보 추출
        elements = data.get('skeleton', {}).get('elements', [])

        # 각 UI 컴포넌트에 대해 bbox 그리기
        for element in elements:
            if 'bbox' in element and element['bbox']:
                bbox = element['bbox']

                # 정규화된 좌표를 픽셀 좌표로 변환
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)

                # 사각형 그리기 (빨간색, 두께 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 컴포넌트 타입과 ID를 텍스트로 표시
                label = f"{element.get('type', 'unknown')}_{element.get('id', '')}"

                # 텍스트 배경 그리기
                font_scale = 0.8
                font_thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(image, (x1, y1 - text_height - 8), (x1 + text_width + 4, y1), (0, 0, 255), -1)

                # 텍스트 그리기
                cv2.putText(image, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),font_thickness)

        # 결과 이미지 저장
        if output_path is None:
            # 원본 이미지 경로에 '_bbox' 추가
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_bbox.png"

        cv2.imwrite(output_path, image)
        print(f"bbox가 그려진 이미지 저장: {output_path}")
        return True

    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return False
    except Exception as e:
        print(f"오류 발생: {e}")
        return False


for filename, group in report.groupby('filename'):
    base_filename = os.path.splitext(filename)[0]
    json_path = f'{jsonRoot}/{base_filename}.json'
    image_path = f'{imageRoot}/{base_filename}.png'

    print(f"처리 중: {filename} (그룹 크기: {len(group)})")

    # 파일 존재 여부 확인
    if not os.path.exists(json_path):
        print(f"JSON 파일이 존재하지 않습니다: {json_path}")
        continue

    if not os.path.exists(image_path):
        print(f"이미지 파일이 존재하지 않습니다: {image_path}")
        continue

    # bbox 그리기
    success = draw_bbox_on_image(image_path, json_path)
    if success:
        print(f"✓ {filename} 처리 완료")
    else:
        print(f"✗ {filename} 처리 실패")

print("모든 파일 처리 완료!")


