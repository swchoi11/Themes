import cv2
import numpy as np
from sklearn.cluster import KMeans
from src.result import ResultModel
from src.detect import Detect


class Visibility():
    def __init__(self, file_path: str, filter_type: str = 'all'):
        if file_path.endswith(".xml"):
            self.image_path = file_path.replace(".xml", ".png")
            self.xml_path = file_path
        else:
            self.image_path = file_path
            self.xml_path = file_path.replace(".png", ".xml")
        self.file_name = self.image_path.replace('.png', '')
        
        self.num_colors = 3
        
        # component_bbox가 제공되지 않으면 자동으로 추출
        detect = Detect(file_path)
        self.component_bbox = detect.get_valid_components(filter_type) or []
        
        self.img = cv2.imread(self.image_path)

    @staticmethod
    def _hsv_color(self, component_img: np.ndarray):
        if len(component_img.shape) == 3:
            img_rgb = cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = component_img

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

    @staticmethod
    def _lab_color(self, component_img: np.ndarray):
        if len(component_img.shape) == 3:
            img_rgb = cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = component_img

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

        # 고채도 픽셀 비율
        high_saturation_ratio = np.sum(s > 50) / s.size

        print(f"이미지 크기: {component_img.shape}")
        
        if high_saturation_ratio < 0.3:
            print("LAB 색공간 사용")
            colors = self._lab_color(component_img)
        else:
            print("HSV 색공간 사용")
            colors = self._hsv_color(component_img)
        
        print(f"추출된 색상 개수: {len(colors)}")
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
            print("색상이 2개 미만이므로 대비 계산 불가")
            return False
        
        print(f"대비 계산 시작 - 색상 개수: {len(colors)}")
        print(f"색상들: {colors}")
        
        max_contrast = 0
        contrast_results = []
        
        for i in range(len(colors)):
            for j in range(i+1, len(colors)):
                luminance1 = self._get_luminance(colors[i])
                luminance2 = self._get_luminance(colors[j])

                lighter = max(luminance1, luminance2)
                darker = min(luminance1, luminance2)

                contrast = (lighter + 0.05) / (darker + 0.05)
                max_contrast = max(max_contrast, contrast)
                
                contrast_results.append({
                    'color1': colors[i],
                    'color2': colors[j],
                    'luminance1': luminance1,
                    'luminance2': luminance2,
                    'contrast': contrast,
                    'sufficient': contrast >= 3.0
                })
                
                print(f"색상 {colors[i]} vs {colors[j]}: 대비 {contrast:.2f} ({'충분' if contrast >= 3.0 else '부족'})")

                # WCAG AA 기준 (4.5:1) 또는 더 관대한 기준 (3:1) 사용
                if contrast >= 3.0:
                    print(f"충분한 대비 발견! 대비값: {contrast:.2f}")
                    return True
        
        print(f"최대 대비값: {max_contrast:.2f} - 모든 색상 쌍이 기준 미달")
        
        # 최대 대비가 3.0 이상이면 최소한의 가독성은 있다고 판단
        return max_contrast >= 3.0
    
    def run_visibility_check(self):
        issues = []
        
        try:
            index = 0
            for component_bbox in self.component_bbox:
                # 컴포넌트 영역 추출
                img_crop = self.img[component_bbox[1]:component_bbox[3], component_bbox[0]:component_bbox[2]]

                # 컴포넌트 영역의 주된 색상 추출                
                colors = self.color_extraction(img_crop)
                print(colors)
                
                if not colors:
                    print(f"색상 추출 실패: {component_bbox}")
                    continue
                
                # 대비 검사
                if not self.calculate_contrast(colors):
                    issue = ResultModel(
                        image_path=self.image_path,
                        issue_type="visibility",
                        issue_location=component_bbox,
                        issue_description=f"컴포넌트 영역 {component_bbox}에서 가독성 이슈 발생 - 색상 대비 부족"
                    )
                    print(index)
                    cv2.imwrite(f"{self.file_name}_issue_{index}.png", img_crop)
                    issues.append(issue)
                    index += 1
                

            return issues
            
        except Exception as e:
            print(f"가독성 검사 중 오류 발생: {e}")
            return []
    