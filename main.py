import json
import glob
from tqdm import tqdm

from src.layout import Layout
from src.gemini import Gemini

# # classifier = ImageXMLClassifier()
# # classifier.run_classification() 

for test_image in tqdm(glob.glob('./resource/defect/*/*.png')):  
    layout = Layout(test_image)
    issues = layout.run_layout_check()

    gemini = Gemini(test_image)
    result = gemini.detect_all_issues()

    all_issues = issues + result
    
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
