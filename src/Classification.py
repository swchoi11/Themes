import os
import cv2
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim


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

    def compare_layouts_match(self, img1: np.ndarray, img2: np.ndarray):
        """
        두 이미지의 레이아웃 유사도를 비교합니다. 픽셀간 비교 방식
        """
        score_map = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
        score = score_map[0][0]
        results = {
            'similarity_score': float(score),
        }

        return results
    
    def compare_layouts_ssim(self, img1: np.ndarray, img2: np.ndarray):
        """
        두 이미지의 레이아웃 유사도를 비교합니다. SSIM 방식
        """
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 그레이스케일로 변환
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        s = ssim(img1_gray, img2_gray)  # 최소 윈도우 크기 사용
        
        return {'similarity_score': s}

    def compare_layouts_orb(self, img1: np.ndarray, img2: np.ndarray):
        """
        두 이미지의 레이아웃 유사도를 비교합니다. ORB 방식
        """
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = [m for m in matches if m.distance < 10]

        return {'similarity_score': len(good_matches) / len(matches)}

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

            result = self.compare_layouts_ssim(target_img, candidate_img)
            if result['similarity_score'] > max_score['similarity_score']:
                max_score['similarity_score'] = result['similarity_score']
                max_score['image_path'] = candidate_image_path
                max_score['cluster'] = os.path.basename(candidate_image_path).split('_')[0]
    
        return max_score


