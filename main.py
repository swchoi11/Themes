import json
import glob
import os
from tqdm import tqdm

from src.classification import ImageXMLClassifier
from src.layout import Layout
from src.utils.utils import check_size, init_process, move_to_not_processed, save_results, check_valid_issues, unprocessed_issues, to_excel
from src.utils.model import ResultModel
from src.gemini import Gemini, IssueProcessor

# 1. classification
# classifier = ImageXMLClassifier()
# classifier.check_classification()

# result = []
# for test_image in tqdm(glob.glob('./resource/defect/*.png')):
# # # for test_image in ['/home/user/dev/themes/resource/com.android.intentresolver_ChooserActivityLauncher_20250521_161950.png']:
#     json_filename = init_process()
    
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

#     if  not check_valid_issues(issues):
#         issues = [ResultModel(
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
#         )]


#     # ResultModel을 딕셔너리로 변환하여 전체 결과에 추가
#     for issue in issues:
#         result.append(issue.model_dump())
    
#     # 매 이미지 처리 후 저장 (안전성을 위해)
#     json_filename = f'./output/jsons/all_issues/{json_filename}'
#     save_results(result, json_filename)

# # json 파일을 돌면서 제미나이 -> 최종 결과 산출

# json_filename = f'./output/jsons/all_issues/result-20250610.json'
# processor = IssueProcessor()
# output_path = processor.sort_issues(json_filename)
output_path = './output/jsons/final_issue/result-20250610.json'
# unprocessed_issues(output_path)
# print(f"결과 파일: {output_path}")
to_excel(output_path)