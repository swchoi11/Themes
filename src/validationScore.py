import os
import cv2
import json
import re
import pandas as pd
from tqdm import tqdm

# 9분할 영역 매핑
LOCATION_MAP = {
    '0': 'TL', '1': 'TC', '2': 'TR',
    '3': 'ML', '4': 'MC', '5': 'MR',
    '6': 'BL', '7': 'BC', '8': 'BR'
}


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
        processed_score = int(float(score_str)) + 1

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
    """UI 컴포넌트 이름을 추출하는 함수"""
    # 미리 정의된 UI 컴포넌트 목록
    ui_components = [
        'Button', 'ImageView', 'RadioButton', 'CheckBox', 'EditText', 'TextView',
        'Switch', 'ToggleButton', 'SeekBar', 'ProgressBar', 'Spinner', 'ImageButton',
        'text', 'icon', 'button',
    ]

    if pd.isna(ui_component_string):
        return 'Button'  # 기본값

    ui_string = str(ui_component_string)

    # 1단계: 괄호 안의 값 추출 시도
    match = re.search(r'\(([^()]+)\)[^)]*$', ui_string)
    if match:
        extracted_value = match.group(1)
        split_values = extracted_value.split('_')
        first_part = split_values[0]

        if first_part in ui_components:
            return first_part

        # 가장 유사한 컴포넌트 찾기
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
        if component.lower() in ui_string.lower():
            return component

        similarity = calculate_similarity(ui_string, component)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = component

    if best_similarity >= 0.3:
        return best_match

    return 'Button'


def bbox_to_location_zone(bbox):
    """정규화된 bbox 좌표를 9분할 영역으로 변환"""
    if not bbox or len(bbox) != 4:
        return 'MC'

    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    col = 0 if center_x < 0.33 else (1 if center_x < 0.67 else 2)
    row = 0 if center_y < 0.33 else (1 if center_y < 0.67 else 2)
    zone_index = row * 3 + col

    return LOCATION_MAP.get(str(zone_index), 'MC')


def extract_innermost_component(component_string):
    """컴포넌트 문자열에서 가장 안쪽 괄호의 내용을 추출"""
    if not component_string or pd.isna(component_string):
        return ""

    component_str = str(component_string).strip()
    bracket_contents = re.findall(r'\(([^()]*)\)', component_str)

    if bracket_contents:
        innermost = bracket_contents[-1].strip()
        return innermost if innermost else component_str

    if ':' in component_str:
        before_colon = component_str.split(':')[0].strip()
        return before_colon

    first_word = component_str.split()[0] if component_str.split() else component_str
    return first_word


def find_best_matching_component(target_component, json_elements, similarity_threshold=0.3):
    """JSON elements에서 target_component와 가장 유사한 컴포넌트를 찾아 반환"""
    if not target_component or not json_elements:
        return None

    target_clean = extract_innermost_component(target_component)
    best_match = None
    best_similarity = 0

    for element in json_elements:
        element_id = element.get('id', '')
        element_type = element.get('type', '')
        element_bbox = element.get('bbox', [])

        candidates = [
            element_id,
            element_type,
            f"{element_type}_{element_id}",
        ]

        # 언더스코어로 분리된 각 부분들 추가
        if element_id:
            candidates.extend(element_id.split('_'))
        if element_type:
            candidates.extend(element_type.split('_'))

        for candidate in candidates:
            if not candidate:
                continue

            # 1. 정확한 매칭
            if target_clean.lower() == candidate.lower():
                location_zone = bbox_to_location_zone(element_bbox)
                return {
                    'component_name': f"{element_type}_{element_id}" if element_type and element_id else candidate,
                    'bbox': element_bbox,
                    'location': location_zone,
                    'similarity': 1.0,
                    'match_type': 'exact'
                }

            # 2. 부분 문자열 매칭
            if (target_clean.lower() in candidate.lower() or
                    candidate.lower() in target_clean.lower()):
                substring_similarity = min(len(target_clean), len(candidate)) / max(len(target_clean), len(candidate))
                if substring_similarity > best_similarity:
                    location_zone = bbox_to_location_zone(element_bbox)
                    best_similarity = substring_similarity
                    best_match = {
                        'component_name': f"{element_type}_{element_id}" if element_type and element_id else candidate,
                        'bbox': element_bbox,
                        'location': location_zone,
                        'similarity': substring_similarity,
                        'match_type': 'substring'
                    }

            # 3. 유사도 계산
            similarity = calculate_similarity(target_clean, candidate)
            if similarity > best_similarity and similarity >= similarity_threshold:
                location_zone = bbox_to_location_zone(element_bbox)
                best_similarity = similarity
                best_match = {
                    'component_name': f"{element_type}_{element_id}" if element_type and element_id else candidate,
                    'bbox': element_bbox,
                    'location': location_zone,
                    'similarity': similarity,
                    'match_type': 'similarity'
                }

    return best_match


def find_location_for_component(ui_component, filename, json_dir_path):
    """개별 컴포넌트의 ComponentID, BBOX, Location을 찾는 함수"""
    base_filename = os.path.splitext(filename)[0]
    json_path = os.path.join(json_dir_path, f"{base_filename}.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            elements = json_data.get('skeleton', {}).get('elements', [])
            match_result = find_best_matching_component(ui_component, elements)

            if match_result and match_result['bbox']:
                bbox = match_result['bbox']
                component_name = match_result['component_name']
                location_zone = match_result['location']

                # ComponentID와 BBOX 분리
                component_id = component_name
                bbox_info = f"[{','.join(map(str, bbox))}]"
                location_info = location_zone

                return component_id, bbox_info, location_info

        except Exception as e:
            pass  # 조용히 처리

    return "null", "null", "null"


def draw_bbox_on_image(image_path, json_path, group, output_path=None, DEBUG=False):
    """이미지에 JSON의 UI 컴포넌트 bbox를 그려서 저장하는 함수"""
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
        elements = data.get('skeleton', {}).get('elements', [])
        target_components = set()

        # group에서 UI 컴포넌트 추출
        if hasattr(group, 'iterrows'):
            for idx, row in group.iterrows():
                ui_components = row.get('ui_component', [])
                if isinstance(ui_components, str):
                    ui_components = [comp.strip().replace('`', '') for comp in ui_components.split(',')]
                elif isinstance(ui_components, list):
                    pass
                else:
                    print(f"행 {idx}: ui_component 형태를 인식할 수 없음 ({type(ui_components)})")
                    continue
                target_components.update(ui_components)

        target_components = list(target_components)
        print(f"원본 target_components: {target_components}")

        # 가장 안쪽 괄호의 내용 추출
        def extract_innermost(text):
            matches = re.findall(r'\(([^()]+)\)', text)
            if matches:
                return matches[-1]
            return text

        # 괄호 처리
        cleaned_components = []
        for component in target_components:
            if isinstance(component, str):
                inner_content = extract_innermost(component)
                cleaned_components.append(inner_content)
            else:
                cleaned_components.append(component)

        if not cleaned_components:
            print("target_components가 비어있습니다.")
            return False

        print(f"최종 타겟 컴포넌트: {cleaned_components}")
        print(f"전체 elements 수: {len(elements)}")

        # Score 필터링
        if hasattr(group, 'query'):
            filtered_group = group[group['Score'] < 3]
            print(f"Score < 3인 항목: {len(filtered_group)}개")
        else:
            filtered_group = group

        matched_count = 0

        # 각 UI 컴포넌트에 대해 bbox 그리기
        for element in elements:
            element_type = element.get('type', '')
            element_id = element.get('id', '')

            # 매칭 조건 확인
            is_match = False
            match_reason = ""

            if element_id in cleaned_components:
                is_match = True
                match_reason = f"id '{element_id}' 직접 매칭"
            elif element_type in cleaned_components:
                is_match = True
                match_reason = f"type '{element_type}' 매칭"
            elif f"{element_type}_{element_id}" in cleaned_components:
                is_match = True
                match_reason = f"full_name '{element_type}_{element_id}' 매칭"
            else:
                for target in cleaned_components:
                    if target in element_id or target in element_type:
                        is_match = True
                        match_reason = f"부분 매칭 '{target}'"
                        break

            if is_match and matched_count < 5:
                print(f"매칭성공: {match_reason}")

            # 매칭되는 경우에만 bbox 그리기
            if is_match and 'bbox' in element and element['bbox']:
                bbox = element['bbox']

                if len(bbox) != 4:
                    print(f"잘못된 bbox 형식: {bbox}")
                    continue

                # 정규화된 좌표를 픽셀 좌표로 변환
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)

                if x1 >= x2 or y1 >= y2:
                    continue

                # 좌표를 이미지 범위 내로 클립
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                score_color = (0, 0, 255)  # 기본값: 빨간색
                score_info = ""

                # Score에 따른 색상 결정
                if hasattr(filtered_group, 'iterrows'):
                    matching_scores = []
                    for idx, row in filtered_group.iterrows():
                        row_components = row.get('ui_component', [])
                        if isinstance(row_components, str):
                            row_components = [comp.strip().replace('`', '') for comp in row_components.split(',')]

                        cleaned_row_components = []
                        for comp in row_components:
                            if isinstance(comp, str):
                                inner_content = extract_innermost(comp)
                                cleaned_row_components.append(inner_content)
                            else:
                                cleaned_row_components.append(comp)

                        if (element_id in cleaned_row_components or
                                element_type in cleaned_row_components or
                                f"{element_type}_{element_id}" in cleaned_row_components):
                            matching_scores.append(row['Score'])

                    if matching_scores:
                        current_score = min(matching_scores)
                        if current_score == 1:
                            score_color = (0, 0, 255)  # 빨간색
                            score_info = " (Score: 1)"
                        elif current_score == 2:
                            score_color = (0, 255, 0)  # 초록색
                            score_info = " (Score: 2)"
                        else:
                            score_color = (255, 0, 0)  # 파란색
                            score_info = f" (Score: {current_score})"

                # 사각형 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), score_color, 2)

                # 텍스트 라벨
                label = f"{element_id}{score_info}"

                # 텍스트 배경과 텍스트 그리기
                font_scale = 0.6
                font_thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                               font_thickness)

                text_x = max(0, x1)
                text_y = max(text_height + 8, y1)

                cv2.rectangle(image, (text_x, text_y - text_height - 8),
                              (text_x + text_width + 4, text_y), score_color, -1)
                cv2.putText(image, label, (text_x + 2, text_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

                matched_count += 1
                if matched_count <= 3:
                    print(f"매칭됨: {label} - 좌표: ({x1},{y1})-({x2},{y2})")

            elif is_match and matched_count < 3:
                print(f"매칭되었지만 bbox가 없음: {element_type}_{element_id}")

        print(f"총 {matched_count}개 컴포넌트가 매칭되어 그려졌습니다.")

        if matched_count == 0:
            print("*")

        # 결과 저장
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_bbox.png"

        # 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if DEBUG:
            cv2.namedWindow('outputImage', cv2.WINDOW_NORMAL)
            cv2.imshow('outputImage', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        success = cv2.imwrite(output_path, image)
        if success:
            print(f"bbox가 그려진 이미지 저장: {output_path}")
            return True
        else:
            print(f"이미지 저장 실패: {output_path}")
            return False

    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return False
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    REPORT_DIR_PATH = '../output/report'
    JSON_DIR_PATH = '../output/json'
    IMAGE_DIR_PATH = '../resource/image'

    # 리포트 데이터 로드 및 전처리
    print("리포트 데이터 로드 중...")
    report = pd.read_csv(f'{REPORT_DIR_PATH}/final_report.csv')
    report = report.replace(r'^\s*$', pd.NA, regex=True)
    report = report.dropna(how='all')
    report['score'] = report['score'].apply(process_score)
    report['itemName'] = report['ui_component'].apply(extract_ui_component_name)

    # KPI 매트릭스 생성 (ComponentID, BBOX, Location 포함)
    print("ComponentID, BBOX, Location 정보를 추가하여 KPI 매트릭스 생성 중...")
    rows = []
    for idx, row in tqdm(report.iterrows(), total=len(report), desc="Processing KPI Matrix"):
        gt = 'Fail' in row['filename']
        pred = row['score'] < 3

        # ComponentID, BBOX, Location 정보 찾기
        component_id, bbox_info, location_info = find_location_for_component(
            row['ui_component'],
            row['filename'],
            JSON_DIR_PATH
        )

        row_data = {
            'FileName': row['filename'],
            'GroundTruth': gt,
            'Predict': pred,
            'Match': gt == pred,
            'Score': row['score'],
            'itemName': row['itemName'],
            'ComponentID': component_id,
            'BBOX': bbox_info,
            'Location': location_info,
            'Description': row['description'],
            'Reason': row['description'],
            'ui_component': row['ui_component']
        }
        rows.append(row_data)

    # KPI 매트릭스 저장
    df = pd.DataFrame(rows)
    df.to_csv(f'{REPORT_DIR_PATH}/final_kpi_matrix_total.csv', index=False)

    # 통계 출력
    successful_component_id = len([row for row in rows if row['ComponentID'] != 'null'])
    successful_bbox = len([row for row in rows if row['BBOX'] != 'null'])
    successful_location = len([row for row in rows if row['Location'] != 'null'])

    print(f"\n=== KPI 매트릭스 생성 완료 ===")
    print(f"전체 행 수: {len(rows)}")
    print(f"ComponentID 매칭 성공: {successful_component_id}개 ({successful_component_id / len(rows) * 100:.1f}%)")
    print(f"BBOX 매칭 성공: {successful_bbox}개 ({successful_bbox / len(rows) * 100:.1f}%)")
    print(f"Location 매칭 성공: {successful_location}개 ({successful_location / len(rows) * 100:.1f}%)")

    # 영역별 분포 출력
    location_counts = {}
    for row in rows:
        if row['Location'] != 'null':
            location_counts[row['Location']] = location_counts.get(row['Location'], 0) + 1

    print(f"\n=== 영역별 분포 ===")
    for zone_code, zone_name in LOCATION_MAP.items():
        count = location_counts.get(zone_name, 0)
        print(f"{zone_name}: {count}개")

    print(f"\nKPI 매트릭스 저장 완료: {REPORT_DIR_PATH}/final_kpi_matrix_total.csv")

    final_rows = []
    for filename, group in df.groupby('FileName'):
        filtered_group = group[group['Score'] < 3]         # 점수가 3 미만인 것들만 필터링
        filtered_group['sort_priority'] = filtered_group['itemName'].apply(
            lambda x: 1 if x in ['text', 'icon', 'button'] else 0
        )
        filtered_group = filtered_group.sort_values(['Score', 'sort_priority']).drop('sort_priority', axis=1)

        if filtered_group.empty:
            first_row = group.iloc[0]
            gt = first_row['GroundTruth']
            pred = False
            row_data = {
                'FileName': first_row['FileName'],
                'GroundTruth': gt,
                'Predict': pred,
                'Match': gt == pred,
                'Score': first_row['Score'],
                'itemName': first_row['itemName'],
                'ComponentID': first_row['ComponentID'],
                'BBOX': first_row['BBOX'],
                'Location': first_row['Location'],
                'Description': first_row['Description'],
                'Reason': first_row['Reason'],
                'ui_component': first_row['ui_component']
            }
            final_rows.append(row_data)
            continue

        sorted_group = filtered_group.sort_values('Score')
        first_row = sorted_group.iloc[0]

        if sorted_group['itemName'].isin(['text', 'icon', 'button']).all():
            gt = first_row['GroundTruth']
            pred = False
            row_data = {
                'FileName': first_row['FileName'],
                'GroundTruth': gt,
                'Predict': pred,
                'Match': gt == pred,
                'Score': first_row['Score'],
                'itemName': first_row['itemName'],
                'ComponentID': first_row['ComponentID'],
                'BBOX': first_row['BBOX'],
                'Location': first_row['Location'],
                'Description': first_row['Description'],
                'Reason': first_row['Reason'],
                'ui_component': first_row['ui_component']
            }
            final_rows.append(row_data)
            continue

        final_rows.append(first_row.to_dict())

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(f'{REPORT_DIR_PATH}/final_kpi_matrix.csv', index=False)


    # 이미지에 BBOX 그리기 (기존 기능 유지)
    print("\n이미지에 BBOX 그리기 시작...")
    for filename, group in tqdm(final_df.groupby('FileName'), desc="Drawing BBOX"):
        print(filename)

        if group['Score'].isna().all():
            print(f"건너뛰기: {filename} - Score 열이 없거나 모든 값이 NaN")
            continue

        group = group[(group['Score'] < 3)]

        if group.empty:
            print(f"건너뛰기: {filename} - Score < 3인 항목이 없음")
            continue

        if group['itemName'].isin(['text', 'icon', 'button']).all():
            print(f"건너뛰기: {filename} - 모든 itemName이 ['text', 'icon', 'button'] 중 하나임")
            continue

        base_filename = os.path.splitext(filename)[0]
        json_path = f'{JSON_DIR_PATH}/{base_filename}.json'
        image_path = f'{IMAGE_DIR_PATH}/{base_filename}.png'
        output_path = f'{REPORT_DIR_PATH}/result/{base_filename}.png'

        print(f"처리 중: {filename} (그룹 크기: {len(group)})")

        # 파일 존재 여부 확인
        if not os.path.exists(json_path):
            print(f"JSON 파일이 존재하지 않습니다: {json_path}")
            continue

        if not os.path.exists(image_path):
            print(f"이미지 파일이 존재하지 않습니다: {image_path}")
            continue

        # bbox 그리기
        success = draw_bbox_on_image(image_path, json_path, group, output_path)
        if success:
            print(f"{filename} 처리 완료")
        else:
            print(f"{filename} 처리 실패")

    print("\n전체 처리 완료!")