import os
import cv2
import glob
import shutils
import numpy as np
from skimage.metrics import structural_similarity as ssim
from common.logger import init_logger, timefn

logger = init_logger()

class Classification:
    def __init__(self, is_gray: bool=False):
        self.base_template_dir = "./output/main_frames"
        self.is_gray = is_gray
        
    @timefn
    def _resize_image(self, image_path: str, target_ratio: float=1.8):
        """
        원본 이미지의 비율을 유지하면서 target_ratio 크기의 배경 위에 이미지를 배치합니다.
        output: 리사이즈된 이미지 (target_ratio 크기의 배경 위에 중앙 정렬)
        """
        # 원본 이미지의 비율 계산
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        ratio = h/w
        
        # 1.8 이상이면 1080*2376 크기, 아니면 1856*2176 크기로 리사이즈
        if ratio > 1.8 and target_ratio > 1.8:
            target_w, target_h = 1080, 2376
        elif ratio < 1.8 and target_ratio < 1.8:
            target_w, target_h = 1856, 2176
        else:
            return None
        
        # 비율 유지하면서 리사이즈할 크기 계산
        scale = min(target_w/w, target_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 리사이즈된 이미지 생성
        resized_img = cv2.resize(img, (new_w, new_h))

        # 검은색 배경 생성
        background = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 이미지를 배경 중앙에 배치
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

        # 그레이스케일로 변환
        if self.is_gray:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        logger.info(f"이미지 리사이징 완료: {image_path}")
        return background

    @timefn
    def compare_layouts_match(self, img1: np.ndarray, img2: np.ndarray):
        """
        두 이미지의 레이아웃 유사도를 비교합니다. 픽셀간 비교 방식
        """
        logger.info("레이아웃 비교 시작 (matchTemplate)")
        score_map = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
        score = score_map[0][0]
        results = {
            'similarity_score': float(score),
        }
        logger.info(f"유사도 점수: {score:.4f}")
        return results
    
    @timefn
    def compare_layouts_ssim(self, img1: np.ndarray, img2: np.ndarray):
        """
        두 이미지의 레이아웃 유사도를 비교합니다. SSIM 방식
        """
        logger.info("레이아웃 비교 시작 (SSIM)")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 그레이스케일로 변환
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        s = ssim(img1_gray, img2_gray)
        logger.info(f"SSIM 점수: {s:.4f}")
        return {'similarity_score': s}

    @timefn
    def compare_layouts_orb(self, img1: np.ndarray, img2: np.ndarray):
        """
        두 이미지의 레이아웃 유사도를 비교합니다. ORB 방식
        """
        logger.info("레이아웃 비교 시작 (ORB)")

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = [m for m in matches if m.distance < 10]
        score = len(good_matches) / len(matches)
        logger.info(f"ORB 매칭 점수: {score:.4f}")
        return {'similarity_score': score}

    @timefn
    def most_similar_image(self, target_image_path: str, candidate_image_dir: str, method: str='orb'):
        """
        타겟 이미지와 비교 이미지의 레이아웃 유사도를 비교하여 가장 유사한 이미지를 반환합니다.
        input: 타겟 이미지 경로, 비교 방식
        output: 가장 유사한 이미지 경로
        """
        logger.info(f"레이아웃 분류 시작: {target_image_path}")
        candidate_image_list = glob.glob(f"{candidate_image_dir}/*.png")
        logger.info(f"베이스 템플릿 이미지 개수: {len(candidate_image_list)}")

        target_img = cv2.imread(target_image_path)
        target_ratio = target_img.shape[0]/target_img.shape[1]
        if target_ratio > 1.8:
            ratio = 2.0
        else:
            ratio = 1.2
        logger.info(f"타겟 이미지 비율: {target_ratio}")

        target_img = self._resize_image(target_image_path, ratio)
        
        max_score = {'similarity_score': 0, 'image_path': '', 'cluster': ''}

        if method == 'ssim':
            compare_method = self.compare_layouts_ssim
            logger.info("SSIM 방식 선택")
        elif method == 'orb':
            compare_method = self.compare_layouts_orb
            logger.info("ORB 방식 선택")
        else:
            compare_method = self.compare_layouts_match
            logger.info("matchTemplate 방식 선택")

        for candidate_image_path in candidate_image_list:
            logger.info(f"비교할 베이스 템플릿 이미지: {candidate_image_path}")
            candidate_img = self._resize_image(candidate_image_path, ratio)
          
            if candidate_img is None:
                logger.info(f"베이스 템플릿이 다른 화면 비율을 가져 비교를 건너 뜁니다.")
                continue

            logger.info(f"베이스 템플릿이 동일한 화면 비율을 가져 비교를 시작합니다.")
            result = compare_method(target_img, candidate_img)
            if result['similarity_score'] > max_score['similarity_score']:
                max_score['similarity_score'] = result['similarity_score']
                max_score['image_path'] = candidate_image_path
                logger.info(f"새로운 최고 점수 발견: {max_score['similarity_score']:.4f} ")

        logger.info(f"가장 높은 유사도를 가진 이미지: {max_score['image_path']} (점수: {max_score['similarity_score']:.4f})")
        return max_score
    
    @timefn
    def get_cluster(self, target_image_path: str, method: str='orb'):
        candidate_image_dir = self.base_template_dir
        result = self.most_similar_image(target_image_path, candidate_image_dir, method=method)
        cluster = os.path.basename(result['image_path']).split('_')[0]
        return cluster, result['similarity_score']
    
    @timefn
    def get_default(self, target_image_path: str, cluster_id: str, method: str='orb'):
        candidate_image_dir = f'./output/{cluster_id}/'
        result = self.most_similar_image(target_image_path, candidate_image_dir, method=method)
        return result['image_path'], result['similarity_score']
        
    def run(self, target_image_dir: str):
        image_list = glob.glob(f"{target_image_dir}/*.png")

        index = 0
        for target_image_path in image_list:
            cluster, score = self.get_cluster(target_image_path)
            default_image_path, score = self.get_default(target_image_path, cluster)

            output_path = f"./output/default/default_{index}_{default_image_path}"
            os.makedirs("./output/default", exist_ok=True)

            shutils.copy2(default_image_path, output_path)
            index += 1


            




    
