import os
import cv2
import glob
import numpy as np
from typing import Tuple, List, Dict, Any, Union


class Mask:
    def __init__(self):
        self._colors = ["#00ff00","#ff0000","#0000ff","#e13232"]

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
            
            # 현재 색상의 마스크를 3채널로 변환
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # 현재 색상의 영역을 마스킹된 이미지에 복사
            np.copyto(masked_image, img, where=(mask_3ch == [255, 255, 255]))
            
            # 현재 색상의 상자 정보 저장
        
        return masked_image

    def mask_directory_images(self, image_dir: str, is_gray: bool) -> None:
        """
        input: 이미지 디렉토리, 흑백 변환 여부
        output: None
        """

        image_list = glob.glob(f"{image_dir}/*.png")

        for image_path in image_list:
            image_name = os.path.basename(image_path)
            
            img = cv2.imread(image_path)
            masked_img = self.mask_image(img)
            
            if not is_gray:
                cv2.imwrite(f"{image_dir}/masked_{image_name}", masked_img)
                print(f"마스킹된 이미지 저장됨: {image_path}")
            # 흑백으로 변환
            else:
                gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"{image_dir}/masked_gray_{image_name}", gray_img)
                print(f"흑백 이미지 저장됨: {image_path}")
            

