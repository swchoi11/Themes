import os
import json

import pandas as pd
from src.gemini import Gemini

import numpy as np
from tqdm import tqdm

gemini = Gemini()

def visibility_inspector(image_path: str) -> dict:
    """단일 파일의 이슈들을 정렬 - 2개 이상의 이슈가 있을 때 Gemini가 가장 중요한 것 선택"""

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

    try:
        # Gemini API 호출 (바운딩 박스가 그려진 이미지 사용)
        result = gemini._call_gemini_image(prompt=prompt, image=image_path)
        return result.text

    except Exception as e:
        print(f"Gemini 검증 중 오류 발생: {e}")

if __name__ == "__main__":

    dirpath = '../../output/visualization'
    filepath = './resource/dataset.xlsx'
    output_file = './output/visibility00.csv'
    report_file = './output/report.csv'

    report = pd.read_csv(report_file)
    report_filename = np.unique(report['filename'])

    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"Excel 파일을 찾을 수 없습니다: {filepath}")
        exit(1)

    df = pd.read_excel(filepath)
    df = df[~df['issue_type'].isin(['no_xml'])]
    df['filename'] = df.apply(
        lambda row: str(row['label']) + os.path.basename(row['filename'])
        if pd.notna(row['label']) and row['label'] != ''
        else os.path.basename(row['filename']),
        axis=1
    )

    df['issue_type'] = df['issue_type'].apply(
        lambda x: 'normal' if str(x).startswith('normal_') else str(x)
    )

    df = df[df['issue_type']=='not_processed']

    issue_report=[]
    try:
        total_files = len(df.groupby('filename'))
        for filename, group in tqdm(df.groupby('filename'), desc="파일 처리 중", total=total_files):

            image_path = os.path.join(dirpath, filename)

            if not os.path.isfile(image_path):
                print(f"not found: {image_path}")
                continue

            issues = group.to_dict('records')
            selected_issue = visibility_inspector(image_path, issues)
            if selected_issue:
                # CSV는 mode='a'가 잘 작동함
                pd.DataFrame([filename,selected_issue]
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
    report.to_excel('./output/issue_report.xlsx')