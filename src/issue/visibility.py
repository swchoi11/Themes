import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from src.utils.model import ResultModel
from src.utils.detect import Detect
from src.utils.logger import init_logger
from src.utils.utils import bbox_to_location
from src.utils.model import EvalKPI

logger = init_logger()

class Visibility():
    def __init__(self, file_path: str, filter_type: str = 'all'):
        
        if file_path.endswith(".xml"):
            self.image_path = file_path.replace(".xml", ".png")
        else:
            self.image_path = file_path
        self.file_name = self.image_path.replace('.png', '')
        
        self.num_colors = 3
        self.output_path = f"./output/images/{os.path.basename(self.image_path)}"
        
        detect = Detect(file_path)
        
        match filter_type:
            case 'text':
                self.components = detect._filter_text()
            case 'button':
                self.components = detect._filter_button()
            case 'all':
                self.components = detect._no_filter()
        
        self.img = cv2.imread(self.image_path)

    def _hsv_color(self, img_rgb: np.ndarray):
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # h channel main color
        h_channel = img_hsv[:, :, 0]
        hist_h = cv2.calcHist([h_channel], [0], None, [180], [0, 180])

        # find peaks
        peaks = []
        for i in range(1, len(hist_h) - 1):
            if hist_h[i] > hist_h[i-1] and hist_h[i] > hist_h[i+1] and hist_h[i] > np.max(hist_h) * 0.1:
                peaks.append(i)
        
        # select peaks
        peaks = sorted(peaks, key=lambda x: hist_h[x], reverse=True)[:self.num_colors]

        # find colors
        colors = []
        for peak_h in peaks:
            mask = np.abs(h_channel - peak_h) < 10
            if np.any(mask):
                masked_pixels = img_rgb[mask]
                if len(masked_pixels) > 0:
                    avg_color = np.mean(masked_pixels, axis=0).astype(int)
                    colors.append(avg_color.tolist())
        
        # HSV에서 충분한 색상을 찾지 못한 경우 KMeans로 보완
        if len(colors) < self.num_colors:
            pixels_reshaped = img_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels_reshaped)

            for center in kmeans.cluster_centers_:
                if len(colors) < self.num_colors:
                    colors.append(center.astype(int).tolist())
        
        return colors[:self.num_colors]

    def _lab_color(self, img_rgb):
        # lab
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        # a, b channel 
        a_channel = img_lab[:, :, 1]
        b_channel = img_lab[:, :, 2]

        # histogram
        hist_ab = cv2.calcHist([a_channel, b_channel], [0, 1], None, [32, 32], [0, 256, 0, 256])

        # find peaks
        peaks = []
        threshold = np.max(hist_ab) * 0.05

        for i in range(1, hist_ab.shape[0] - 1):
            for j in range(1, hist_ab.shape[1] - 1):
                if (hist_ab[i, j] > threshold and 
                    hist_ab[i, j] > hist_ab[i-1, j] and hist_ab[i, j] > hist_ab[i+1, j] and
                    hist_ab[i, j] > hist_ab[i, j-1] and hist_ab[i, j] > hist_ab[i, j+1]):
                    peaks.append((i, j, hist_ab[i, j]))

        peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:self.num_colors]

        #find colors
        colors = []
        for a_bin, b_bin, _ in peaks:
            a_center = a_bin * 8
            b_center = b_bin * 8

            a_mask = np.abs(a_channel - a_center) < 12
            b_mask = np.abs(b_channel - b_center) < 12
            region_mask = a_mask & b_mask

            if np.any(region_mask):
                region_pixels = img_rgb[region_mask]
                avg_color = np.mean(region_pixels, axis=0).astype(int)
                colors.append(avg_color.tolist())

        # LAB에서 충분한 색상을 찾지 못한 경우 KMeans로 보완
        if len(colors) < self.num_colors:
            pixels_reshaped = img_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels_reshaped)

            for center in kmeans.cluster_centers_:
                if len(colors) < self.num_colors:
                    colors.append(center.astype(int).tolist())

        return colors[:self.num_colors]
    
    def _simple_color_extraction(self, img_rgb):
        """검정-흰색 같은 무채색 고대비 조합을 위한 간단한 색상 추출"""
        # 픽셀 밝기 기준으로 클러스터링
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 어두운 픽셀과 밝은 픽셀의 대표 색상 찾기
        colors = []
        
        # 매우 어두운 영역 (0~60)
        very_dark_mask = gray < 60
        if np.any(very_dark_mask):
            dark_pixels = img_rgb[very_dark_mask]
            dark_color = np.mean(dark_pixels, axis=0).astype(int)
            colors.append(dark_color.tolist())
        
        # 매우 밝은 영역 (200~255)
        very_bright_mask = gray > 200
        if np.any(very_bright_mask):
            bright_pixels = img_rgb[very_bright_mask]
            bright_color = np.mean(bright_pixels, axis=0).astype(int)
            colors.append(bright_color.tolist())
        
        # 중간 어두운 영역 (60~120)
        if len(colors) < self.num_colors:
            mid_dark_mask = (gray >= 60) & (gray <= 120)
            if np.any(mid_dark_mask):
                mid_dark_pixels = img_rgb[mid_dark_mask]
                mid_dark_color = np.mean(mid_dark_pixels, axis=0).astype(int)
                colors.append(mid_dark_color.tolist())
        
        # 중간 밝은 영역 (150~200)
        if len(colors) < self.num_colors:
            mid_bright_mask = (gray >= 150) & (gray <= 200)
            if np.any(mid_bright_mask):
                mid_bright_pixels = img_rgb[mid_bright_mask]
                mid_bright_color = np.mean(mid_bright_pixels, axis=0).astype(int)
                colors.append(mid_bright_color.tolist())
        
        # 여전히 부족하면 KMeans 사용
        if len(colors) < 2:  # 최소 2개 색상은 있어야 대비 계산 가능
            #print("KMeans로 색상 보완")
            pixels_reshaped = img_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=max(2, self.num_colors), random_state=42, n_init=10)
            kmeans.fit(pixels_reshaped)
            
            colors = []
            for center in kmeans.cluster_centers_:
                colors.append(center.astype(int).tolist())
        
        return colors[:self.num_colors]
    
    def color_extraction(self, component_img: np.ndarray):
        if len(component_img.shape) == 3:
            img_rgb = cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = component_img
        
        # 이미지 특성 분석
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)

        # 채도와 명도의 분산 계산
        saturation_var = np.var(s)
        value_var = np.var(v)  # 명도 분산 추가

        # 수정된 고채도 픽셀 비율 - 명도가 충분히 높은 픽셀만 고려
        bright_pixels_mask = v > 80  # 명도가 80 이상인 픽셀만
        if np.any(bright_pixels_mask):
            bright_s = s[bright_pixels_mask]
            high_saturation_ratio = np.sum(bright_s > 30) / bright_s.size
        else:
            high_saturation_ratio = 0.0  # 밝은 픽셀이 없으면 채도 0으로 간주
        
        # 명도 대비 분석 (검정-흰색 조합 감지)
        low_value_ratio = np.sum(v < 80) / v.size  # 어두운 픽셀 비율
        high_value_ratio = np.sum(v > 180) / v.size  # 밝은 픽셀 비율
        has_high_value_contrast = (low_value_ratio > 0.2 and high_value_ratio > 0.2) or value_var > 3000
        
        # RGB 기반 색상 분산도 체크 (추가 검증)
        rgb_std = np.std(img_rgb.reshape(-1, 3), axis=0)
        is_grayscale = (rgb_std < 15).all()  # RGB 편차가 작으면 무채색
        
        #print(f"밝은 픽셀 기준 고채도 비율: {high_saturation_ratio:.3f}")
        #print(f"명도 대비 존재: {has_high_value_contrast}")
        #print(f"무채색 판단: {is_grayscale}, RGB 편차: {rgb_std}")
        
        # 검정-흰색 같은 무채색 고대비 조합 우선 감지
        if (has_high_value_contrast and high_saturation_ratio <= 0.3) or is_grayscale:
            #print("무채색 고대비 - 간단한 색상 추출 사용")
            colors = self._simple_color_extraction(img_rgb)
        elif high_saturation_ratio <= 0.4:  # 임계값 조정
            #print("LAB 색공간 사용")
            colors = self._lab_color(img_rgb)
        else:
            #print("HSV 색공간 사용")
            colors = self._hsv_color(img_rgb)
        
        return colors

    def _normalize_color(self, c):
        c = c / 255.0
        if c < 0.03928:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4

    def _get_luminance(self, color):
        r, g, b = color
        r_lum = self._normalize_color(r)
        g_lum = self._normalize_color(g)
        b_lum = self._normalize_color(b)
        return 0.2126 * r_lum + 0.7152 * g_lum + 0.0722 * b_lum
    
    def calculate_contrast(self, colors):
        if len(colors) < 2:
            #print("색상이 2개 미만이므로 대비 계산 불가")
            return False, 0.0  # 대비 계산 불가능한 경우 False와 0.0 반환
                
        max_contrast = 0
        has_sufficient_contrast = False
        
        for i in range(len(colors)):
            for j in range(i+1, len(colors)):
                luminance1 = self._get_luminance(colors[i])
                luminance2 = self._get_luminance(colors[j])

                lighter = max(luminance1, luminance2)
                darker = min(luminance1, luminance2)

                contrast = (lighter + 0.05) / (darker + 0.05)
                max_contrast = max(max_contrast, contrast)
                # WCAG AA 기준 (4.5:1) 또는 더 관대한 기준 (3:1) 사용
                # print(contrast)
                if contrast >= 1.5:
                    # #print(f"충분한 대비 발견! 대비값: {contrast:.2f}")
                    has_sufficient_contrast = True

        return has_sufficient_contrast, max_contrast
    
    def run_visibility_check(self):        
        try:
            logger.info(f"가독성 검사 시작: {self.image_path}")
            issues = []
            print(f"가독성 검사 중 컴포넌트 수: {len(self.components)}")
            for component in tqdm(self.components, desc="가독성 검사 중"):

                x1, y1, x2, y2 = component['bounds']
                index = component['index']

                img_crop = self.img[y1:y2, x1:x2]

                # 컴포넌트 영역의 주된 색상 추출
                colors = self.color_extraction(img_crop)
                
                if not colors:
                    continue
                
                # 대비 검사
                has_sufficient_contrast, max_contrast = self.calculate_contrast(colors)
                
                # 충분한 대비가 없는 경우에만 이슈 생성
                if not has_sufficient_contrast:
                    try:
                        resource_id = component.get('resource_id', 'unknown')
                        location_id = bbox_to_location(component['bounds'], self.img.shape[0], self.img.shape[1])
                        location_type = EvalKPI.LOCATION[location_id]
                        
                        issue = ResultModel(
                            filename=self.image_path,
                            issue_type = "visibility",
                            component_id = int(index),
                            ui_component_id = "5",
                            ui_component_type = "TextView",
                            score = "",
                            location_id = location_id,
                            location_type = location_type,
                            bbox=component['bounds'],
                            description_id = "0",
                            description_type = "텍스트, 아이콘과 배경 간 대비가 낮아 가독성이 떨어짐",
                            description=f"{component.get('type', 'Unknown')}, {resource_id}에서 가독성 이슈 발생 - 색상 대비 부족:{max_contrast:.2f}"
                        )
                        issues.append(issue)
                    except Exception as e:
                        logger.error(f"이슈 생성 중 오류: {e}")
                        logger.error(f"컴포넌트 데이터: {component}")
            
            # 모든 이슈가 있는 컴포넌트들을 이미지에 표시
            if issues:
                try:
                    for issue in issues:
                        x1, y1, x2, y2 = issue.bbox
                        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        contrast_text = f"contrast: {issue.description.split(':')[-1]}"
                        cv2.putText(self.img, contrast_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.imwrite(self.output_path, self.img)
                except Exception as e:
                    logger.error(f"이미지 표시 중 오류: {e}")
            
            return issues
        except Exception as e:
            #print(f"가독성 검사 중 오류 발생: {e}")
            return []
    