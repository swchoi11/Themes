'''
현재 파일 구조
resource/pass/ : 정상
resource/default/ : 디폴트
resource/defect/ : 이슈

알고리즘
1. 정상, 디폴트 테마 내의 xml을 기준으로 동일한 레이아웃 분류
2. 이슈 내의 xml을 분류 기준에 맞춤
3. 분류된 레이아웃 내 요소를 바탕으로 이슈 확인 - rule base
  # 0. text, 아이콘과 배경간 대비가 낮아 가독성이 떨어짐
  # 1. 하이라이트된 항목, 텍스트와 배경간 대비가 낮아 가독성이 떨어짐
  # 2. 상호작용 가능한 요소가 시각적으로 명확히 구분되지 않음
  # 3. 전화버튼이 정렬 안됨
  # 4. 컴포넌트 내부 요소의 정렬
  # 5. 동일 계층 요소간의 정렬
  # 6. 텍스트가 할당된 영역을 초과
  # 7. 아이콘의 가장자리가 잘려 보이지 않음
  # 8. 완벽하게 동일한 아이콘이 있음
4. 제미나이
  # 9. 달력
  # a. 달력, 시계 아이콘
  # b. 이미지 여백
'''
from src.classification import ImageXMLClassifier
from src.layout import Layout
from src.gemini import Gemini
from tqdm import tqdm
# # classifier = ImageXMLClassifier()
# # classifier.run_classification() 

import os
import json

results = []
for cutoff_image in tqdm(os.listdir('./resource/defect/Design Issue')):
    # PNG 파일만 처리
    if not cutoff_image.endswith('.png'):
        continue
    target_file = './resource/defect/Designn  Issue/' + cutoff_image
        
    layout = Layout(target_file)
    issues = layout.run_layout_check()

    gemini = Gemini(target_file)
    result = gemini.detect_all_issues()

    
    # 각 항목의 타입 확인
    if issues:
        for i, issue in enumerate(issues):
            print(f"issues[{i}] 타입: {type(issue)}")
            if not hasattr(issue, 'model_dump'):
                print(f"  ⚠️ model_dump 메서드가 없음! 내용: {issue}")
    
    if result:
        for i, res in enumerate(result):
            print(f"result[{i}] 타입: {type(res)}")
            if not hasattr(res, 'model_dump'):
                print(f"  ⚠️ model_dump 메서드가 없음! 내용: {res}")

    # 두 결과 합치기 (중복 가능성 있음)
    all_issues = issues + result
    
    # # Gemini로 중복 이슈 정리 및 정렬 (선택사항)
    # if all_issues:
    #     sorted_issues = gemini.sort_issues(all_issues)
    #     results.append(sorted_issues)
    # else:
    #     results.append([])

    with open('./resource/defect/Cut off issue/result.json', 'a') as f:
        # 안전한 JSON 변환
        json_results = []
        for issue in all_issues:
            if hasattr(issue, 'model_dump'):
                json_results.append(issue.model_dump())
            else:
                print(f"⚠️ model_dump 메서드가 없는 항목 발견: {type(issue)} - {issue}")
                # 튜플이나 다른 타입인 경우 건너뛰기
                continue
        
        json.dump(json_results, f, ensure_ascii=False, indent=2)
