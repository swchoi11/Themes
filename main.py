import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from src.layout import Layout
from src.utils.model import ResultModel
from src.gemini import Gemini, IssueProcessor
from src.utils.utils import init_process, save_results, to_excel
from src.utils.bucket import test_image_list, upload_to_bucket, set_api_key
from src.utils.exceptions import move_to_not_processed, check_xml, check_size, check_valid_image, check_all_issues_json, check_valid_issues

import glob

# 0. 데이터 준비
set_api_key()

test_image_list = test_image_list()

json_filename = init_process()

valid_test_list = check_all_issues_json(json_filename, test_image_list)
# valid_test_list = glob.glob('./sample40/*.png')
exception = []

for test_image in tqdm(valid_test_list):
    result = []
    try:
        if not check_xml(test_image):
            issue = ResultModel(
                filename=test_image,
                issue_type="no_xml",
                component_id=0,
                ui_component_id="",
                ui_component_type="",
                score="5",
                location_id="",
                location_type="",
                bbox=[1,1,1,1],
                description_id="",
                description_type="",
                description="xml 파일이 없습니다."
            )
            result.append(issue.model_dump())
            save_results(result, json_filename)
            continue

        if not check_valid_image(test_image):
            issue = ResultModel(
                filename=test_image,
                issue_type="invalid_image",
                component_id=0,
                ui_component_id="",
                ui_component_type="",
                score="5",
                location_id="",
                location_type="",
                bbox=[1,1,1,1],
                description_id="",
                description_type="",
                description="이미지를 읽을 수 없습니다."
            )
            result.append(issue.model_dump())
            save_results(result, json_filename)
            continue

        if not check_size(test_image):
            result.append(move_to_not_processed(test_image).model_dump())
            save_results(result, json_filename)
            continue
        
        # 4. layout check (기본 체크 모두 통과 후)
        layout = Layout(test_image)
        issues = layout.run_layout_check() 
        
        # 5. gemini check
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
        issue = {
            "filename": test_image,
            "exception": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        exception.append(issue)
        continue

    # 예외 데이터가 있는 경우에만 CSV 저장
    if exception:
        csv_filename = f'./output/exception-{datetime.now().strftime("%Y%m%d")}.csv'
        
        # DataFrame 생성
        df = pd.DataFrame(exception)
        
        # 파일이 존재하는지 확인
        file_exists = os.path.isfile(csv_filename)
        
        # CSV 저장 (헤더는 파일이 없을 때만 추가)
        df.to_csv(csv_filename, 
                 mode='a', 
                 index=False, 
                 header=not file_exists,
                 encoding='utf-8-sig')

# json 파일을 돌면서 제미나이 -> 최종 결과 산출
xlsx_filename = to_excel(json_filename)

# sort_issues에서 final_inference 로직이 통합되어 있으므로 별도 호출 불필요
processor = IssueProcessor()
output_path = processor.sort_issues(xlsx_filename)
final_output_path = to_excel(output_path)

upload_to_bucket(final_output_path)
