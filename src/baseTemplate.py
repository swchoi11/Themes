# 1. 이미지 겹쳐서 공통 박스만 필터링
# 2. 박스 크기 조건
# 3. 위치 유사도 (IoU)
# 4. 결과는 까만 화면에 박스만 표기

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import DBSCAN
import easyocr

@dataclass
class BaseConfig:
    min_box_size: int = 10
    max_box_size: int = 1000
    iou_threshold: float = 0.5
    max_workers: int = 4
    debug_mode: bool = False

class MakeBaseTemplate:
    def __init__(self, min_box_size: int = 10, max_box_size: int = 1000, iou_threshold: float = 0.5):
        self.MIN_BOX_SIZE = min_box_size
        self.MAX_BOX_SIZE = max_box_size
        self.IOU_THRESHOLD = iou_threshold

    @staticmethod
    def safe_imread(path: str) -> Optional[np.ndarray]:
        try:
            return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ERROR] 이미지 읽기 실패: {path} | 오류: {e}")
            return None

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        target_size = (1080, 2376) if w <= 1800  else (1856, 2176)
        target_w, target_h = target_size

        # 비율 유지 resize
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 검정 배경
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        top = max(0, (target_h - new_h) // 2)
        left = max(0, (target_w - new_w) // 2)
        canvas[top:top+new_h, left:left+new_w] = resized
        return canvas

    def compute_iou(self, box1, box2) -> float:
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xi1, yi1 = max(x1, x1g), max(y1, y1g)
        xi2, yi2 = min(x2, x2g), min(y2, y2g)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 1e-10 else 0.0

    def extract_boxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        image = self.resize_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny 엣지 검출
        edges = cv2.Canny(gray, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 컨투어 검출
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.MIN_BOX_SIZE <= w <= self.MAX_BOX_SIZE and self.MIN_BOX_SIZE <= h <= self.MAX_BOX_SIZE:
                boxes.append((x, y, x + w, y + h))
        return boxes

    # 공통된 bounding box 만 남기는 코드 
    def filter_common_boxes(self, all_boxes: List[List[Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int, int]]:
        base_boxes = all_boxes[0]
        common_boxes = []
        for box in base_boxes:
            matched = True
            for other_boxes in all_boxes[1:]:
                if not any(self.compute_iou(box, other) > self.IOU_THRESHOLD for other in other_boxes):
                    matched = False
                    break
            if matched:
                common_boxes.append(box)
        return common_boxes

    # mask 채운 후 모든 이미지의 AND 연산 or 누적 계산 코드
    def compute_mask_intersection(self, all_boxes: List[List[Tuple[int, int, int, int]]], 
                                  image_shape: Tuple[int, int], mask_save_path: str = None) -> List[Tuple[int, int, int, int]]:
        # 빈 마스크 생성
        intersection_mask = np.ones(image_shape, dtype=np.uint8) * 255

        for box_list in all_boxes:
            temp_mask = np.zeros(image_shape, dtype=np.uint8)
            for x1, y1, x2, y2 in box_list:
                cv2.rectangle(temp_mask, (x1, y1), (x2, y2), 255, thickness=-1)
            # 이미지 간 AND 연산
            intersection_mask = cv2.bitwise_and(intersection_mask, temp_mask)

        if mask_save_path:
            cv2.imwrite(mask_save_path, intersection_mask)

        # contour 찾기
        contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.MIN_BOX_SIZE <= w <= self.MAX_BOX_SIZE and self.MIN_BOX_SIZE <= h <= self.MAX_BOX_SIZE:
                final_boxes.append((x, y, x + w, y + h))
        return final_boxes

    # 겹친 이미지 기준으로 threshold 없이 많이 겹친 영역 추출
    def compute_overlap_heatmap(self, all_boxes: List[List[Tuple[int, int, int, int]]],
                                image_shape: Tuple[int, int],
                                min_overlap_count: int=2,
                                heatmap_save_path: str = None) -> List[Tuple[int, int, int, int]]:

        # 초기화된 누적 히트맵
        heatmap = np.zeros(image_shape, dtype=np.uint16)
        for box_list in all_boxes:
            mask = np.zeros(image_shape, dtype=np.uint8)
            for (x1, y1, x2, y2) in box_list:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
            heatmap += mask
        
        # 특정 겹침 수 이상만 남김
        thresholded = (heatmap >= min_overlap_count).astype(np.uint8) * 255

        # 디버깅용 히트맵 저장
        if heatmap_save_path:
            norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(heatmap_save_path, norm)

        # contour -> box 추출
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.MIN_BOX_SIZE <= w <= self.MAX_BOX_SIZE and self.MIN_BOX_SIZE <= h <= self.MAX_BOX_SIZE:
                final_boxes.append((x, y, x + w, y + h))
        return final_boxes

    def visualize_boxes(self, image_shape: Tuple[int, int], boxes: List[Tuple[int, int, int, int]], save_path: str):
        canvas = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        print(f"공통 박스 수: {len(boxes)}")

        for (x1, y1, x2, y2) in boxes:
            if x1 < 0 or y1 < 0 or x2 > image_shape[1] or y2 > image_shape[0]:
                print(f"유효하지 않은 상자 좌표: ({x1}, {y1}, {x2}, {y2})")
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ext = os.path.splitext(save_path)[-1]
        success, encoded_image = cv2.imencode(ext, canvas)
        if success:
            with open(save_path, mode='wb') as f:
                f.write(encoded_image.tobytes())
            print(f"이미지 저장 성공 : {save_path}")
        else:
            print(f"이미지 저장 실패 : {save_path}")


class BaseTemplateGenerator:
    def __init__(self, min_box_size: int, iou_threshold: float, cluster_dir: str,
                 output_dir: str, condition_output_dir: str, image_extensions: List[str],
                 cluster_prefix: str = "cluster", max_workers: int = 4):
        self.template_maker = MakeBaseTemplate(min_box_size=min_box_size, iou_threshold=iou_threshold)
        self.cluster_dir = cluster_dir
        self.output_dir = output_dir
        self.condition_output_dir = condition_output_dir
        self.image_extensions = image_extensions
        self.cluster_prefix = cluster_prefix
        self.max_workers = max_workers

    def run(self):
        for cluster_folder in sorted(os.listdir(self.cluster_dir)):
            if not cluster_folder.startswith(self.cluster_prefix):
                continue

            cluster_path = os.path.join(self.cluster_dir, cluster_folder)
            print(f"[INFO] cluster_path: {cluster_path}")

            image_paths = [
                os.path.join(cluster_path, f)
                for f in os.listdir(cluster_path)
                if (os.path.isfile(os.path.join(cluster_path, f)) and
                    os.path.splitext(f)[1].lower() in self.image_extensions)
            ]

            if len(image_paths) < 2:
                print(f"[WARN] {cluster_folder}: 이미지 부족")
                continue

            # # 병렬로 상자 추출
            all_boxes  = []
            failed_count = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_single_image, path): path
                    for path in image_paths
                }

                for future in as_completed(future_to_path):
                    image_path = future_to_path[future]
                    try:
                        result = future.result(timeout=30)
                        if result is not None:
                            all_boxes.append(result)
                        else:
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        print(f"[ERROR] 처리 실패 {image_path}: {e}")

            print(f"[INFO] 처리 중: {cluster_folder} ({len(image_paths)}개 이미지)")

            # for p in image_paths:
            #     img = MakeBaseTemplate.safe_imread(p)
            #     if img is None:
            #         print(f"[WARN] 이미지 로드 실패: {p}")
            #         continue
            #     all_images.append(img)
            #
            # all_boxes = [self.template_maker.extract_boxes(img) for img in all_images]
            sample_img = MakeBaseTemplate.safe_imread(image_paths[0])

            # common_boxes = self.template_maker.filter_common_boxes(all_boxes)  # [basic]
            # common_boxes = self.template_maker.compute_mask_intersection(all_boxes,
            #       sample_img.shape[:2], mask_save_path=os.path.join(condition_output_dir, f"{cluster_folder}_mask.png")  # [mask]
            common_boxes = self.template_maker.compute_overlap_heatmap(all_boxes, sample_img.shape[:2], 
                    heatmap_save_path=os.path.join(self.condition_output_dir, f"{cluster_folder}_heatmap.png"))  # [heatmap]

            output_path = os.path.join(self.output_dir, f"{cluster_folder}_base_image.png")
            self.template_maker.visualize_boxes(sample_img.shape[:2], common_boxes, output_path)
            print(f"[INFO] 저장 완료: {output_path}")

if __name__ == "__main__":
    # 기본 설정
    generator = BaseTemplateGenerator(
        min_box_size=10,
        iou_threshold=0.5,
        cluster_dir="./clusters",
        output_dir="./output",
        condition_output_dir="./debug",
        image_extensions=['.png', '.jpg', '.jpeg'],
        max_workers=4
    )