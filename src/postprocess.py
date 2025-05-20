import os
import cv2
import glob
import numpy as np
from typing import Tuple, List, Dict, Any, Union
import pandas as pd


class Mask:
    def __init__(self):
        self._colors = ["#00ff00","#ff0000","#0000ff"]

    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """
        16진수 색상 코드를 BGR 형식으로 변환합니다.
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

    def mask_image(self, img: Union[str, np.ndarray]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        이미지에서 모든 색상의 상자 영역을 검출하고 마스킹된 이미지를 반환합니다.
        input: 이미지
        output: 마스킹된 이미지, 상자 정보
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        mask_color = (255, 255, 255)  # 하얀색 배경
        masked_image = np.full_like(img, mask_color, dtype=np.uint8)
        all_rows = []
        
        # 모든 색상에 대해 검출 수행
        for hex_color in self._colors:
            bgr = self._hex_to_bgr(hex_color)
            bgr_array = np.array(bgr, dtype=np.uint8)
            
            # 현재 색상의 마스크 생성
            mask = np.all(img == bgr_array, axis=-1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # 면적이 0보다 큰 컨투어만 필터링
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]
            
            # 현재 색상의 마스크를 3채널로 변환
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # 현재 색상의 영역을 마스킹된 이미지에 복사
            np.copyto(masked_image, img, where=(mask_3ch == [255, 255, 255]))
            
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

    def mask_directory_images(self, image_dir: str, is_gray: bool) -> None:
        """
        input: 메인 프레임 이미지 디렉토리, 흑백 변환 여부
        output: None
        """

        image_list = glob.glob(f"{image_dir}/*.png")

        for image_path in image_list:
            image_name = os.path.basename(image_path)
            
            img = cv2.imread(image_path)
            masked_img, rows = self.mask_image(img)
            
            if not is_gray:
                cv2.imwrite(image_path, masked_img)
                print(f"마스킹된 이미지 저장됨: {image_path}")
            # 흑백으로 변환
            else:
                gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(image_path, gray_img)
                print(f"흑백 이미지 저장됨: {image_path}")
            


class Classification:
    def __init__(self, target_image_dir: str):
        self.main_frame_dir = "./output/main"
        self.target_img = cv2.imread(target_image_dir)
        ratio = self.target_img.shape[0] / self.target_img.shape[1]
        if ratio > 1.8:
           self.target_ratio = (1080, 2376) 
        else:
            self.target_ratio = (1856, 2176)

    def _resize_image(self, image_path: str):
        """
        원본 이미지의 비율을 유지하면서 target_ratio 크기의 배경 위에 이미지를 배치합니다.
        output: 리사이즈된 이미지 (target_ratio 크기의 배경 위에 중앙 정렬)
        """
        # 원본 이미지의 비율 계산
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        target_w, target_h = self.target_ratio

        if h/w > 1.8 and self.target_ratio == (1080, 2376):
            same_size = True
        elif h/w < 1.8 and self.target_ratio == (1856, 2176):
            same_size = True
        else:
            same_size = False

        if same_size:
        
            # 비율 유지하면서 리사이즈할 크기 계산
            scale = min(target_w/w, target_h/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 이미지 리사이즈
            resized_img = cv2.resize(img, (new_w, new_h))
            
            # 검은색 배경 생성
            background = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # 이미지를 배경 중앙에 배치
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
            
            return background
        else:
            return None

    def compare_layouts(self, img1: np.ndarray, img2: np.ndarray):
        """
        두 이미지의 레이아웃 유사도를 비교합니다.
        """
        score_map = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        score = score_map[0][0]
        results = {
            'similarity_score': float(score),
        }

        return results

    def layout_classification(self, target_image_dir: str):
        """
        타겟 이미지와 비교 이미지의 레이아웃 유사도를 비교하여 가장 유사한 이미지를 반환합니다.
        input: 타겟 이미지 경로
        output: 가장 유사한 이미지 경로
        """
        candidate_image_list = glob.glob(f"{self.main_frame_dir}/*.png")

        target_img = self._resize_image(target_image_dir)
        
        max_score = {'similarity_score': 0, 'image_path': '', 'cluster': ''}

        for candidate_image_path in candidate_image_list:
            candidate_img = self._resize_image(candidate_image_path)
          
            if candidate_img is None:
                continue

            result = self.compare_layouts(target_img, candidate_img)
            if result['similarity_score'] > max_score['similarity_score']:
                max_score['similarity_score'] = result['similarity_score']
                max_score['image_path'] = candidate_image_path
                max_score['cluster'] = os.path.basename(candidate_image_path).split('_')[0]
    
        return max_score


