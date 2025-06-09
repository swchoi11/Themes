import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Tuple
import cv2
import re
import random
import os
import shutil
import json

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

def check_size(image_path: str):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    return h/w <= 2.0

def init_process():
    os.makedirs('./output/images/', exist_ok=True)
    os.makedirs('./output/images/not_processed', exist_ok=True)
    os.makedirs('./output/jsons/all_issues', exist_ok=True)
    os.makedirs('./output/jsons/final_issue', exist_ok=True)
    os.makedirs('./output/classification', exist_ok=True)

def move_to_not_processed(test_image: str):
    file_name = os.path.basename(test_image)
    xml_path = test_image.replace('.png', '.xml')
    
    # 이미지와 XML 파일 이동
    shutil.move(test_image, f'./output/images/not_processed/{file_name}')
    if os.path.exists(xml_path):
        xml_name = os.path.basename(xml_path)
        shutil.move(xml_path, f'./output/images/not_processed/{xml_name}')

def load_existing_results(filename):
    """기존 JSON 파일 로드"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_results(all_results, filename):
    """결과를 JSON 파일에 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
