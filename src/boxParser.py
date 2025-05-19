"""
이미지 레이아웃 분류 모듈

이 모듈은 이미지들의 레이아웃을 분석하고 분류하는 기능을 제공합니다.
주요 기능:
1. 이미지 비율에 따른 분류
2. 특정 색상 영역 검출
3. 레이아웃 유사도 비교 및 분류

입력:
- resource/xml-bbox/image... : 원본 이미지 저장 경로

출력:
- output/xml-bbox/ratio_width_height/layout_1_image... : 비율별 분류된 이미지 저장
- output/xml-bbox/layout_classified.xlsx : 분류 결과 정보
"""

import os
import cv2
import glob
import numpy as np
from typing import Tuple
import pandas as pd

def split_by_ratio(image_dir):
    """
    이미지들을 비율에 따라 분류하여 저장합니다. 동일한 비율의 이미지만 분류할 수 있기 때문입니다.
    
    Args:
        image_dir (str): 이미지가 저장된 디렉토리 경로
        resource/xml-bbox/
        
    Returns:
        None
    
    생성되는 디렉토리 구조:
        {image_dir}/ratio_{width}_{height}/
    """
    image_list = glob.glob(f"{image_dir}/*.png")
    for image_path in image_list:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        os.makedirs(f"{image_dir}/ratio_{width}_{height}", exist_ok=True)
        image_name = os.path.basename(image_path).replace(".png", "")
        os.rename(image_path, f"{image_dir}/ratio_{width}_{height}/{image_name}.png")

    return _layout_list(image_dir)

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

def _layout_list(image_dir):    
    return glob.glob(f"{image_dir}/*/")

def _hex_items_detect(image, hex_color="#00ff00"):
    """
    이미지에서 특정 색상 영역을 검출하고 해당 영역의 바운딩 박스 정보를 반환합니다.
    
    Args:
        image_path (str): 처리할 이미지 경로
        hex_color (str, optional): 검출할 색상의 16진수 코드. 기본값 "#00ff00"
        
    Returns:
        List[Dict]: 검출된 영역들의 위치 정보
            - x (int): 바운딩 박스의 x 좌표
            - y (int): 바운딩 박스의 y 좌표
            - width (int): 바운딩 박스의 너비
            - height (int): 바운딩 박스의 높이
    """
    
    mask_color=(0,0,0) # black
    bgr = _hex_to_bgr(hex_color)
    bgr_array = np.array(bgr, dtype=np.uint8)

    # 원하는 색상 영역의 마스크 생성
    mask = np.all(image == bgr_array, axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"image: {image} - 컨투어를 찾을 수 없습니다")
        return []
    
    # 면적이 0보다 큰 컨투어만 필터링
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]
    if valid_contours:
        # 면적이 가장 큰 컨투어 선택
        valid_contours = [max(valid_contours, key=cv2.contourArea)]
    else:
        valid_contours = []
    
    # 마스킹된 이미지 생성
    masked_image = np.full_like(image, mask_color, dtype=np.uint8)  # 배경을 mask_color로 채움
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 마스크를 3채널로 변환
    
    # 원본 이미지에서 원하는 색상 영역만 마스킹된 이미지에 복사
    np.copyto(masked_image, image, where=(mask_3ch == [255,255,255]))
    
    rows = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rows.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h
            })
        
    return masked_image, rows

def mask_images(image_dir):
    image_list = glob.glob(f"{image_dir}/*.png")
    image_base_dir = os.path.dirname(image_dir).split("/")[-1]
    print(image_base_dir)
    for image_path in image_list:
        image = cv2.imread(image_path)
        masked_image, rows = _hex_items_detect(image, hex_color="#00ff00")
        
        image_name = os.path.basename(image_path).replace(".png", "")
        os.makedirs(f"./output/xml-bbox/{image_base_dir}", exist_ok=True)
        save_path = os.path.join(f"./output/xml-bbox/{image_base_dir}", f"masked_{image_name}.png")
        cv2.imwrite(save_path, masked_image)
        print(f"마스킹된 이미지 저장됨: {save_path}")

def compare_layouts(image1, image2, threshold=0.95):
    """
    두 이미지의 레이아웃 유사도를 비교합니다.
    
    Args:
        image1 (str): 첫 번째 이미지 경로
        image2 (str): 두 번째 이미지 경로
        threshold (float, optional): 동일 레이아웃 판단 임계값. 기본값 0.95
        
    Returns:
        Dict: 비교 결과
            - similarity_score (float): 유사도 점수 (0~1)
            - is_same_layout (bool): 동일 레이아웃 여부
    """
    img1_path = image1
    img2_path = image2
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 초록색 바운딩 박스 마스크 생성
    green_bgr = _hex_to_bgr("#00ff00")
    mask1 = np.all(img1 == green_bgr, axis=-1).astype(np.uint8) * 255
    mask2 = np.all(img2 == green_bgr, axis=-1).astype(np.uint8) * 255
    
    # 구조적 유사도 계산 (SSIM)
    score = cv2.matchTemplate(mask1, mask2, cv2.TM_CCOEFF_NORMED)[0][0]
    
    results={
        'similarity_score': score,
        'is_same_layout': score >= threshold
    }

    return results

def layout_classification(image_dir):
    """
    디렉토리 내의 이미지들을 레이아웃 유사도에 따라 분류합니다.
    모든 이미지 쌍의 유사도를 계산한 후, 가장 유사한 쌍부터 그룹핑합니다.
    
    Args:
        image_path (str): 이미지들이 저장된 디렉토리 경로
        
    Returns:
        Dict[str, List[str]]: 레이아웃 그룹별 이미지 목록
            - key: 그룹 이름
            - value: 해당 그룹에 속한 이미지 파일명 리스트
    """
    image_list = glob.glob(f"{image_dir}/*.png")
    n_images = len(image_list)
    
    # 1. 모든 이미지 쌍의 유사도 계산
    similarity_matrix = np.zeros((n_images, n_images))
    for i in range(n_images):
        for j in range(i+1, n_images):
            result = compare_layouts(image_list[i], image_list[j])
            similarity_matrix[i][j] = similarity_matrix[j][i] = result['similarity_score']
    
    # 2. 유사도가 높은 순서대로 정렬된 이미지 쌍 생성
    pairs = []
    for i in range(n_images):
        for j in range(i+1, n_images):
            pairs.append((similarity_matrix[i][j], i, j))
    pairs.sort(reverse=True)  # 유사도 높은 순으로 정렬
    
    # 3. 그룹핑
    groups = []  # 각 그룹의 대표 이미지 인덱스와 멤버 인덱스 리스트
    group_representatives = {}  # 각 이미지가 속한 그룹의 대표 이미지 인덱스
    
    for similarity, i, j in pairs:
        if similarity < 0.95:  # threshold
            continue
            
        # i와 j가 모두 아직 그룹에 속하지 않은 경우
        if i not in group_representatives and j not in group_representatives:
            groups.append([i, [i, j]])
            group_representatives[i] = i
            group_representatives[j] = i
        # i만 그룹에 속한 경우
        elif i in group_representatives and j not in group_representatives:
            rep_i = group_representatives[i]
            # j와 i의 대표 이미지와의 유사도 확인
            if similarity_matrix[j][rep_i] >= 0.95:
                for group in groups:
                    if group[0] == rep_i:
                        group[1].append(j)
                        group_representatives[j] = rep_i
                        break
        # j만 그룹에 속한 경우
        elif j in group_representatives and i not in group_representatives:
            rep_j = group_representatives[j]
            # i와 j의 대표 이미지와의 유사도 확인
            if similarity_matrix[i][rep_j] >= 0.95:
                for group in groups:
                    if group[0] == rep_j:
                        group[1].append(i)
                        group_representatives[i] = rep_j
                        break
        # i와 j가 다른 그룹에 속한 경우
        elif i in group_representatives and j in group_representatives:
            rep_i = group_representatives[i]
            rep_j = group_representatives[j]
            if rep_i != rep_j:
                # 두 그룹의 대표 이미지 간 유사도 확인
                if similarity_matrix[rep_i][rep_j] >= 0.95:
                    # 그룹 병합
                    for group in groups:
                        if group[0] == rep_j:
                            groups.remove(group)
                            for member in group[1]:
                                group_representatives[member] = rep_i
                            for g in groups:
                                if g[0] == rep_i:
                                    g[1].extend(group[1])
                                    break
                            break
    
    # 4. 그룹에 속하지 않은 이미지들은 각각 새로운 그룹으로
    for i in range(n_images):
        if i not in group_representatives:
            groups.append([i, [i]])
            group_representatives[i] = i
    
    # 5. 파일 이름 변경 및 결과 저장
    layout_groups = {}
    for group_idx, (rep_idx, members) in enumerate(groups):
        group_name = f"layout_{group_idx}"
        for member_idx in members:
            img = image_list[member_idx]
            img_name = os.path.basename(img)
            new_name = f"{group_name}_{img_name}"
            new_path = os.path.join(os.path.dirname(img), new_name)
            os.rename(img, new_path)
        
        layout_groups[group_name] = [os.path.basename(image_list[idx]) for idx in members]
        print(f"레이아웃 {group_name} 생성됨: {len(members)}개 이미지")
    
    return layout_groups


if __name__ == "__main__":
    ## 이미지 비율에 따른 분류
    # split_by_ratio("./resource/xml-bbox/")
    
    # ## box 영역 검출 -> 마스킹 이미지 생성
    # layout_list = _layout_list("./resource/xml-bbox/")
    # for layout in layout_list:
    #     mask_images(layout)
    
    # ## 마스킹 이미지 비교 -> 레이아웃 분류
    # output_layout_list = _layout_list("./output/xml-bbox/")
    # for layout in output_layout_list:
    #     layout_classification(layout)

    result = compare_layouts("./output/xml-bbox/ratio_1080_2340/layout_0_masked_com.samsung.android.messaging_ContactPickerActivity_20250508_173054_Message_selected button_Poor_Kim_boxed.png", "./output/xml-bbox/ratio_748_720/layout_0_masked_com.android.systemui_SubHomeActivity_20250508_173640_Cover_AOD mode_Clock number_poor_Joe_boxed.png")
    print(result)