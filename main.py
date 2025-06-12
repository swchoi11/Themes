from tqdm import tqdm
import pandas as pd
from datetime import datetime
from src.layout import Layout
from src.utils.utils import check_size, init_process, move_to_not_processed, save_results, check_valid_issues, unprocessed_issues, to_excel, check_all_issues_json, check_xml
from src.utils.model import ResultModel
from src.gemini import Gemini, IssueProcessor
from src.utils.bucket import test_image_list, upload_to_bucket, set_api_key

# 0. 데이터 준비
# set_api_key()

test_image_list = test_image_list()

json_filename = init_process()
json_filename = f'./output/jsons/all_issues/{json_filename}'

valid_test_list = check_all_issues_json(json_filename, test_image_list)

     
result = []
exception = []

for test_image in tqdm(valid_test_list):
# for test_image in tqdm(glob.glob('./resource/test/*.png')):
    
    try:
        if not check_xml(test_image):
            issue = ResultModel(
                filename=test_image,
                issue_type="no xml",
                component_id=0,
                ui_component_id="",
                ui_component_type="",
                score="5",
                location_id="",
                location_type="",
                bbox=[],
                description_id="",
                description_type="",
                description="xml 파일이 없습니다."
            )
            result.append(issue.model_dump())
            continue

        
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
        
        save_results(result, json_filename)
    except Exception as e:
        exception.append(test_image)
        continue

df = pd.DataFrame(exception)
df.to_csv(f'./output/exception-{datetime.now().strftime("%Y%m%d")}.csv', 
          mode = 'a', index=False, header=False, encoding='utf-8-sig')

# json 파일을 돌면서 제미나이 -> 최종 결과 산출
to_excel(json_filename)

processor = IssueProcessor()
output_path = processor.sort_issues(json_filename)

to_excel(output_path)
upload_to_bucket(json_filename)
