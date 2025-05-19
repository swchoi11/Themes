import os
import cv2
import glob
import numpy as np
from typing import Tuple
import pandas as pd

def _colors():
    return ["#00ff00","#ff0000","#0000ff"]

def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """
    16진수 색상 코드를 BGR 형식으로 변환합니다.
    
    Args:
        hex_color (str): '#RRGGBB' 형식의 16진수 색상 코드
        
    Returns:
        Tuple[int, int, int]: (Blue, Green, Red) 형식의 색상값
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

def _mask_image(image):
    """
    이미지에서 모든 색상의 상자 영역을 검출하고 마스킹된 이미지를 반환합니다.
    
    Args:
        image (numpy.ndarray): 처리할 이미지
        
    Returns:
        Tuple[numpy.ndarray, List[Dict]]: 마스킹된 이미지와 검출된 상자 정보 리스트
    """
    mask_color = (255, 255, 255)  # 하얀색 배경
    masked_image = np.full_like(image, mask_color, dtype=np.uint8)
    all_rows = []
    
    # 모든 색상에 대해 검출 수행
    for hex_color in _colors():
        bgr = _hex_to_bgr(hex_color)
        bgr_array = np.array(bgr, dtype=np.uint8)
        
        # 현재 색상의 마스크 생성
        mask = np.all(image == bgr_array, axis=-1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # 면적이 0보다 큰 컨투어만 필터링
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]
        
        # 현재 색상의 마스크를 3채널로 변환
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 현재 색상의 영역을 마스킹된 이미지에 복사
        np.copyto(masked_image, image, where=(mask_3ch == [255, 255, 255]))
        
        # 현재 색상의 상자 정보 저장
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            all_rows.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "color": hex_color  # 색상 정보도 함께 저장
            })
    
    return masked_image, all_rows

def mask_images(image_dir):

    image_list = glob.glob(f"{image_dir}/*.png")

    for image_path in image_list:
        image = cv2.imread(image_path)
        masked_image, rows = _mask_image(image)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        image_name = os.path.basename(image_path)

        cv2.imwrite(image_path, masked_image)
        # 흑백으로 변환
        cv2.imwrite(f"{image_dir}/gray_{image_name}", gray)
        print(f"마스킹된 이미지 저장됨: {image_path}")

def compare_layouts(img1, img2, threshold=0.95):
    """
    두 이미지의 레이아웃 유사도를 비교합니다.
    
    Args:
        img1 (numpy.ndarray): 첫 번째 이미지
        img2 (numpy.ndarray): 두 번째 이미지
        threshold (float, optional): 동일 레이아웃 판단 임계값. 기본값 0.95
        
    Returns:
        Dict: 비교 결과
            - similarity_score (float): 유사도 점수 (0~1)
            - is_same_layout (bool): 동일 레이아웃 여부
    """
    # 초록색 바운딩 박스 마스크 생성
    green_bgr = _hex_to_bgr("#00ff00")
    mask1 = np.all(img1 == green_bgr, axis=-1).astype(np.uint8) * 255
    mask2 = np.all(img2 == green_bgr, axis=-1).astype(np.uint8) * 255
    
    # 두 마스크의 크기를 동일하게 맞춤
    target_size = (min(mask1.shape[1], mask2.shape[1]), min(mask1.shape[0], mask2.shape[0]))
    mask1 = cv2.resize(mask1, target_size)
    mask2 = cv2.resize(mask2, target_size)
    
    # 두 마스크의 유사도 계산
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        score = 0.0
    else:
        score = intersection / union
    
    results = {
        'similarity_score': float(score),
        'is_same_layout': score >= threshold
    }

    return results

def layout_classification(target_image_dir):

    candidate_image_list, final_ratio = _split_main_images(target_image_dir)
    print(candidate_image_list, final_ratio)
    
    max_score = {'score': 0, 'image_path': ''}

    for compare_image_dir in candidate_image_list:
        target_image = cv2.imread(target_image_dir)
        compare_image = cv2.imread(compare_image_dir)

        target_image = cv2.resize(target_image, (0, 0), fx=final_ratio, fy=final_ratio)
        compare_image = cv2.resize(compare_image, (0, 0), fx=final_ratio, fy=final_ratio)

        result = compare_layouts(target_image, compare_image)

        if result['is_same_layout'] and result['similarity_score'] > max_score['score']:
            max_score['score'] = result['similarity_score']
            compare_image_dir = os.path.basename(compare_image_dir)
            max_score['image_path'] = compare_image_dir.split('_')[0]

    print(max_score)


def _split_main_images(target_image_path):

    target_image = cv2.imread(target_image_path)
    target_ratio = target_image.shape[0] / target_image.shape[1]

    image_list = glob.glob("./output/main/*.png")
    candidate_image_list = []

    for image_path in image_list:
        print(image_path)
        image = cv2.imread(image_path)
        ratio = image.shape[0] / image.shape[1]
        print(ratio, target_ratio)

        if ratio > 1.8 and target_ratio > 1.8:
            final_ratio = 2.0
            candidate_image_list.append(image_path)
        
        if ratio < 1.8 and target_ratio < 1.8:
            final_ratio = 1.2
            candidate_image_list.append(image_path)

    return candidate_image_list, final_ratio

def extract_issues(cluster):
    image_list = glob.glob(f"./output/{cluster}/*.png")

    rows = []

    for image_path in image_list:
        rows.append(os.path.basename(image_path))

    return rows

def relevant_issues(rows):

    prompt = ''


def generate_description(target_image_dir, cluster_main_image, relevant_issues):
    pass




if __name__ == "__main__":
    target_image_dir = './resource/sample/target.png'

    # 메인 프레임 이미지에 대한 마스킹
    mask_images("./output/main")
    
    # 타겟 이미지에 대한 마스킹
    _mask_image(target_image_dir)

    # # 레이아웃 분류
    cluster_result = layout_classification(target_image_dir)

    # 이슈 탐지
    extract_issues(cluster_result)