import os
import re
import cv2
import json
import random
import pandas as pd
from typing import Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from src.utils.model import EvalKPI


def normalize_xml_content(xml_path: str) -> str:
    """
    XML 파일을 파싱하여 정규화된 문자열로 변환
    
    Args:
        xml_path: XML 파일 경로
        
    Returns:
        정규화된 XML 문자열
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode')
    except ET.ParseError as e:
        print(f"XML 파싱 오류 ({xml_path}): {e}")
        return ""

def calculate_xml_similarity(xml_pair: Tuple[str, str]) -> Tuple[str, str, float]:
    """
    두 XML 파일 간의 유사도 계산
    
    Args:
        xml_pair: (xml_path1, xml_path2) 튜플
        
    Returns:
        (xml_path1, xml_path2, similarity_percentage) 튜플
    """
    xml_path1, xml_path2 = xml_pair
    xml_content1 = normalize_xml_content(xml_path1)
    xml_content2 = normalize_xml_content(xml_path2)
    
    if not xml_content1 or not xml_content2:
        return (xml_path1, xml_path2, 0.0)
    
    similarity = SequenceMatcher(None, xml_content1, xml_content2).ratio() * 100
    return (xml_path1, xml_path2, similarity)

def get_bounds(bounds_str):
    match = re.search(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
    x1, y1, x2, y2 = map(int, match.groups())
    return x1, y1, x2, y2

def get_nodes_same_level(node, level, result):
    if len(result) <= level:
        result.append([])
    result[level].append(node)
    for child in node.findall('node'):
        get_nodes_same_level(child, level+1, result)

def draw_components(components, image_path, file_name='output.png'):
    img = cv2.imread(image_path)
    for component in components:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (component[0], component[1]), (component[2], component[3]), random_color, 2)
    cv2.imwrite(file_name, img)


def init_process():
    os.makedirs('./output/images/', exist_ok=True)
    os.makedirs('./output/images/not_processed', exist_ok=True)
    os.makedirs('./output/jsons/all_issues', exist_ok=True)
    os.makedirs('./output/jsons/final_issue', exist_ok=True)
    os.makedirs('./output/excels/all_issues', exist_ok=True)
    os.makedirs('./output/excels/final_issue', exist_ok=True)
    os.makedirs('./output/logs', exist_ok=True)

    json_filename = f'result-{datetime.now().strftime("%Y%m%d")}.json'
    return json_filename

def save_results(all_results, filename):
    """결과를 JSON 파일에 저장"""
    # 기존 데이터 읽기
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    # 새로운 데이터 추가
    if isinstance(existing_data, list):
        existing_data.extend(all_results)
    else:
        existing_data = all_results
    
    # 전체 데이터를 다시 저장
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


def to_excel(json_filename):
    df = pd.read_json(json_filename)
    output_path = json_filename.replace('.json', '.xlsx').replace('jsons','excels') 
    df.to_excel(output_path, index=False)


def bbox_to_location(bbox, image_height, image_width):

    top = (image_height // 3, image_width // 3)
    middle = (image_height // 3 * 2, image_width // 3 * 2)
    bottom = (image_height // 3 * 3, image_width // 3 * 3)

    x1, x2, y1, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    location = ""

    if center_x < top[1] :
        location += 'T'
    elif center_x < middle[1] :
        location += 'M'
    else:
        location += 'B'

    if center_y < top[0] :
        location += 'L'
    elif center_y < middle[0] :
        location += 'C'
    else:
        location += 'R'

    # 위치 코드에 해당하는 키를 찾아 반환
    for key, value in EvalKPI.LOCATION.items():
        if value == location:
            return key
    
    # 기본값 반환 (찾지 못한 경우)
    return '4'  # 'MC' (Middle Center)


