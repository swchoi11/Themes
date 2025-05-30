import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Tuple
import cv2
import re

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

