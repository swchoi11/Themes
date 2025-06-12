import json
import glob
import os
from tqdm import tqdm

from src.classification import ImageXMLClassifier
from src.layout import Layout
from src.utils.utils import check_size, init_process, move_to_not_processed, save_results, check_valid_issues, unprocessed_issues, to_excel
from src.utils.model import ResultModel
from src.gemini import Gemini, IssueProcessor
from src.utils.bucket import test_image_list, upload_to_bucket, set_api_key

# 0. 데이터 준비
set_api_key()

test_image_list = test_image_list()
     
result = []
for test_image in tqdm(test_image_list):
# # for test_image in tqdm(glob.glob('./resource/*.png')):
    json_filename = init_process()
    
#    이미지 크기 확인
    if not check_size(test_image):
        result.append(move_to_not_processed(test_image).model_dump())
        continue
    
#    2. layout check
    layout = Layout(test_image)
    issues = layout.run_layout_check() 
    
#    3. gemini check
    gemini = Gemini(test_image)
    issues.extend(gemini.layout_issues())



    if not check_valid_issues(issues):
        issues = [ResultModel(
            filename=test_image,
            issue_type="normal",
            component_id=0,
            ui_component_id="",
            ui_component_type="",
            score="5",
            location_id="",
            location_type="",
            bbox=[],
            description_id="",
            description_type="",
            description="문제가 없습니다."
        )]
        result.append(issues[0].model_dump())
    else:
        for issue in issues:
            issue = gemini.issue_score(issue)
            result.append(issue.model_dump())

    
#    매 이미지 처리 후 저장 (안전성을 위해)
    json_filename = f'./output/jsons/all_issues/{json_filename}'

    save_results(result, json_filename)

# json 파일을 돌면서 제미나이 -> 최종 결과 산출
to_excel(json_filename)
processor = IssueProcessor()
output_path = processor.sort_issues(json_filename)
to_excel(output_path)
#json_filename = './output/jsons/final_issue/result-20250612.json'
upload_to_bucket(json_filename)
