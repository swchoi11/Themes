import os
import json

from typing import List
import pandas as pd
from src.gemini import Gemini

gemini = Gemini()


def _sort_issues_by_file(image_path: str, issues: List[dict]) -> dict:
    """단일 파일의 이슈들을 정렬 - 2개 이상의 이슈가 있을 때 Gemini가 가장 중요한 것 선택"""
    prompt = """
        다음 이미지는 핸드폰의 실제 사용화면이고, 다음 이미지에서 감지된 이슈는 다음과 같습니다. 
        이미지와 이슈를 확인하고, 중요도 순서대로 정렬해 주세요.
    
        중요도는 다음과 같습니다.
        1. 사용자 경험에 영향을 주는 이슈
        2. 디자인 측면에서 중요한 이슈
        3. 기능 측면에서 중요한 이슈
        4. 버그 측면에서 중요한 이슈
        5. 기타 이슈
        
        응답 형식: 선택한 이슈의 인덱스 번호와 이유를 명확히 제시해주세요.
    """
    issue_text = json.dumps(issues, ensure_ascii=False, indent=2)

    try:
        # Gemini API 호출
        result = gemini._call_gemini_image_text(prompt, image_path, issue_text, model='')

        # 원본 이슈 중에서 가장 우선순위가 높은 것을 선택
        # (Gemini의 응답을 기반으로 최종 이슈 결정)
        # 여기서는 첫 번째 이슈를 기본으로 하고 AI 설명만 업데이트
        selected_issue = issues[0].copy()  # 첫 번째 이슈를 기본으로
        selected_issue['description'] = result.ai_description   ## result.description
        selected_issue['score'] = result.score

        return selected_issue

    except Exception as e:
        print(f"Gemini 검증 중 오류 발생: {e}")
        # 오류 발생 시 첫 번째 이슈 반환
        return issues[0]


if __name__ == "__main__":

    BASE_DIR = '../../eval'
    dirpath = '../../resource/image'
    filepath = f'{BASE_DIR}/vm5/output/excels/all_issues/result-20250613.xlsx'
    output_file = f'{BASE_DIR}/{gemini.model}.csv'

    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"Excel 파일을 찾을 수 없습니다: {filepath}")
        exit(1)

    df = pd.read_excel(filepath)
    df['filename'] = df['filename'].apply(os.path.basename)

    for filename, group in df.groupby('filename'):
        image_path = os.path.join(dirpath, filename)
        if not os.path.isfile(image_path):
            print(f"not found: {image_path}")
            continue
        issues = group.to_dict('records')

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