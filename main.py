import json
import glob
import os
from tqdm import tqdm

from src.classification import ImageXMLClassifier
from src.layout import Layout
from src.gemini import Gemini
from src.utils.utils import check_size, init_process, move_to_not_processed, load_existing_results, save_results
from src.utils.model import ResultModel


# 1. classification
# classifier = ImageXMLClassifier()
# classifier.check_classification()

json_filename = './output/jsons/all_issues/result-0609.json'
all_results = load_existing_results(json_filename)

# for test_image in tqdm(glob.glob('./resource/defect/*/*.png')):
#     init_process()
    
#     # 이미지 크기 확인
#     if not check_size(test_image):
#         move_to_not_processed(test_image)
#         continue
    
#     # 2. layout check
#     layout = Layout(test_image)
#     issues = layout.run_layout_check()
    
#     # 3. gemini check
#     gemini = Gemini(test_image)
#     issues.extend(gemini.layout_issues())
#     issues.extend(gemini.design_issues())

#     # 이슈가 없으면 정상 결과 추가
#     if not issues:
#         issue = ResultModel(
#             filename=test_image,
#             issue_type="normal",
#             component_id=0,
#             ui_component_id="",
#             ui_component_type="",
#             severity="0",
#             location_id="",
#             location_type="",
#             bbox=[],
#             description_id="",
#             description_type="",
#             description="문제가 없습니다.",
#             ai_description=""
#         )
#         issues.append(issue)
    
#     # ResultModel을 딕셔너리로 변환하여 전체 결과에 추가
#     for issue in issues:
#         all_results.append(issue.model_dump())
    
#     # 매 이미지 처리 후 저장 (안전성을 위해)
#     save_results(all_results, json_filename)

gemini = Gemini(json_filename)
output_path = gemini.sort_issues(json_filename)

print(output_path)