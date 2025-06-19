import os
import cv2
import glob
import json
import shutil
from src.utils.model import ResultModel



def check_xml(image_path: str):
    xml_path = image_path.replace('.png', '.xml')

    if os.path.isfile(xml_path):
        return True

    return False

def check_size(image_path: str):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    return h/w <= 2.0

def check_valid_issues(issues):
    flag = False
    for issue in issues:
        if issue.score == "":
            issue.score = "5"
        if issue.bbox != []:
            flag = True
            break
    return flag

def check_valid_image(image_path: str):
    try:
        img = cv2.imread(image_path)
        return True
    except Exception as e:
        return False


def check_all_issues_json(json_filename, test_image_list):
    if not os.path.isfile(json_filename):
        return test_image_list
    
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if data == [] or len(data) == 0:
        return test_image_list
    
    for item in data:
        # filename = os.path.basename(item['filename'])
        # print(filename)
        if item['filename'] in test_image_list:
            test_image_list.remove(item['filename'])

    return test_image_list


def move_to_not_processed(test_image: str):
    file_name = os.path.basename(test_image)
    xml_path = test_image.replace('.png', '.xml')
    
    # 이미지와 XML 파일 이동
    shutil.copy2(test_image, f'./output/images/not_processed/{file_name}')
    if os.path.exists(xml_path):
        xml_name = os.path.basename(xml_path)
        shutil.copy2(xml_path, f'./output/images/not_processed/{xml_name}')

    return ResultModel(
        filename=test_image,
        issue_type="not_processed",
        component_id=0,
        ui_component_id="",
        ui_component_type="",
        score="5",
        location_id="",
        location_type="",
        bbox=[],
        description_id="",
        description_type="",
        description="사이즈가 맞지 않아 처리 하지 않습니다. 폴드만 처리됩니다."
    )

def unprocessed_issues(json_filename):
    # 기존 JSON 파일 읽기
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    # 새로운 이슈들 추가
    for filename in glob.glob('./output/images/not_processed/*.png'):
        issue = {
            "filename": filename,
            "issue_type": "normal",
            "component_id": 0,
            "ui_component_id": "",
            "ui_component_type": "",
            "severity": "0",
            "location_id": "",
            "location_type": "",
            "bbox": [],
            "description_id": "0",
            "description_type": "",
            "description": "사이즈가 맞지 않아 처리 하지 않습니다. 폴드만 처리됩니다.",
            "ai_description": ""
        }
        existing_data.append(issue)
    
    # 전체 데이터를 다시 저장
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


