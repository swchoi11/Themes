'''
이미지 내의 컴포넌트 안에서 가독성 이슈 확인하기
'''
from paddleocr import PaddleOCR
import cv2
import numpy as np
from sklearn.cluster import KMeans

class Visibility:
    def __init__(self, file_path: str, num_colors: None):
        if file_path.endswith(".xml"):
            self.image_path = file_path.replace(".xml", ".png")
            self.xml_path = file_path
        else:
            self.image_path = file_path
            self.xml_path = file_path.replace(".png", ".xml")
        self.file_name = self.image_path.replace('.png', '')
        if num_colors is None:
            self.num_colors = 3
        else:
            self.num_colors = num_colors

        self.img = cv2.imread(self.image_path)

    def _hsv_color(self):
        if len(self.img.shape) == 3:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = self.img

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
        return colors

    def _lab_color(self):
        if len(self.img.shape) == 3:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = self.img

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
        for a_bin, b_bin in peaks:
            a_center = a_bin * 8
            b_center = b_bin * 8

            a_mask = np.abs(a_channel - a_center) < 12
            b_mask = np.abs(b_channel - b_center) < 12
            region_mask = a_mask & b_mask

            if np.any(region_mask):
                region_pixels = img_rgb[region_mask]
                avg_color = np.mean(region_pixels, axis=0).astype(int)
                colors.append(avg_color.tolist())

        if len(colors) < self.num_colors:
            pixels_reshaped = img_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels_reshaped)

            for center in kmeans.cluster_centers_:
                if len(colors) < self.num_colors:
                    colors.append(center.astype(int).tolist())

        return colors[:self.num_colors]
    
    def color_extraction(self):
        if len(self.img.shape) == 3:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = self.img
        
        # 이미지 특성 분석
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)

        # 채도와 명도의 분산 계산
        saturation_var = np.var(s)
        value_var = np.var(v)

        # 고채도 픽셀 비율
        high_saturation_ratio = np.sum(s > 50) / s.size

        if high_saturation_ratio < 0.3 or saturation_var < 500:
            return self._lab_color()
        else:
            return self._hsv_color()
        
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
    
    def calculate_contranst(self, colors):
        for i in range(len(colors)):
            for j in range(i+1, len(colors)):
                luminance1 = self._get_luminance(colors[i])
                luminance2 = self._get_luminance(colors[j])

                lighter = max(luminance1, luminance2)
                darker = min(luminance1, luminance2)

                contrast = (lighter + 0.05) / (darker + 0.05)

                if contrast > 3.0:
                    return True
        return False
    
    def run_visibility_check(self):
        colors = self.color_extraction()
        return self.calculate_contranst(colors)