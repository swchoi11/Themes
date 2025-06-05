import json
import glob
from tqdm import tqdm

from src.classification import ImageXMLClassifier
from src.layout import Layout
from src.gemini import Gemini


# 1. classification -- src/classification.py
classifier = ImageXMLClassifier()
classifier.run_classification() 

for test_image in tqdm(glob.glob('./resource/defect/*/*.png')):  
    # 2. layout -- src/layout.py, src/utils/detect.py
    layout = Layout(test_image)
    issues = layout.run_layout_check()

    # 3. gemini -- src/gemini.py    
    gemini = Gemini(test_image)
    result = gemini.detect_all_issues()

    all_issues = issues + result

    try:
        all_issues = [issue.model_dump() for issue in all_issues]
    except:
        all_issues = []

    # -- 위에서 검출된 이슈 중 가장 중요한 이슈를 종합해서 판단하기
    # 이 부분은 할루시네이션이 심하고 고도화가 필요해 수행하지 않았습니다.
    # gemini = Gemini(test_image)
    # result = gemini.detect_all_issues()

    with open('./output/result-0605.json', 'a') as f:
        # 안전한 JSON 변환
        json_results = []
        for issue in all_issues:
            json_results.append(issue)
        
        json.dump(json_results, f, ensure_ascii=False, indent=2)


'''
# raw result -- ./output/result-0605.json
# 시간 관계상 visibility 파일의 ./resource/defect/Visibility Issue/com.sec.android.app.launcher_LauncherActivity_20250521_171919.png에서 중단하였습니다.

## 할루시네이션 문제가 해결되지 않았으므로 
## 제미나이가 이미지 별 이슈를 종합하여 검토하는 부분은 result.json파일을 활용하여 진행하는 방향으로 수정해야 할 것 같습니다.

# 이후 진행해야하는 mock code
# 이슈 종합 진단
gemini = Gemini()
issues = gemini.sort_issues('./output/result-0605.json')
# 검출된 이슈에 대한 정답 여부 판정 및 산출물 생성
issuse_to_confusion_matrix(issues)
'''




