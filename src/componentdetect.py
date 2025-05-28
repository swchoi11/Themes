from paddleocr import PaddleOCR
import os   
from typing import Optional
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import glob

class ComponentDetect:
    def __init__(self, image_path: Optional[str] = None):
        if image_path is not None:
            self.image_path = image_path
            self.image_base_name = os.path.splitext(os.path.basename(image_path))[0]
            self.json_path = f"./resource/ocr_result_{self.image_base_name}.json"
            self.img = cv2.imread(image_path)

    def text_detect(self):
        # paddle ocr 초기화
        ocr = PaddleOCR(
                det_model_dir='./src/weights/en_PP-OCRv3_det_infer',
                rec_model_dir='./src/weights/en_PP-OCRv3_rec_infer',
                cls_model_dir='./src/weights/ch_ppocr_mobile_v2.0_cls_infer',
                lang='en',  # other lang also available
                # vis_font_path='/src/weights/arial.ttf',
                use_angle_cls=False,
                use_gpu=False,  # using cuda will conflict with pytorch in the same process
                show_log=False,
                # max_batch_size=1024,
                # use_dilation=True,  # improves accuracy
                # det_db_score_mode='slow',  # improves accuracy
                # rec_batch_num=1024,
                )

        # 이미지 불러오기
        img = self.img
        h, w = img.shape[:2]

        # ocr 결과
        result = ocr.ocr(self.image_path)[0]

        # json 생성
        result_json = []
        margin = 10
        for line in result:
            box, (text, conf) = line
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            offset_ratio = abs(center_x - w / 2) / w
            clipped = x_min < margin or x_max > w - margin
            result_json.append({
                "type": "text",
                "text": text,
                "confidence": conf,
                "size": {"width": width, "height": height},
                "center": {"x": center_x, "y": center_y},
                "bbox": {"x1": x_min, "y1": y_min, "x2": x_max, "y2": y_max},
                "issues": {
                    "alignment_issue": offset_ratio,
                    "clipped": clipped
                }
            })
            if offset_ratio > 0.15:
                print(f":경고: 정렬 이상 가능성: '{text}' | offset ratio = {offset_ratio:.2f}")
            if clipped:
                print(f":경광등: 텍스트 '{text}' 좌우 잘림 의심")

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f":흰색_확인_표시: OCR 결과가 저장되었습니다: {self.json_path}")
        
        return result_json

    def _mask_image(self, json_data):
        height, width = self.img.shape[:2]
        
        for element in json_data:
            x1 = int(element["bbox"]["x1"])
            y1 = int(element["bbox"]["y1"])
            x2 = int(element["bbox"]["x2"])
            y2 = int(element["bbox"]["y2"])

            # 약간의 패딩 추가
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            # 텍스트 영역 주변의 실제 배경색 계산
            local_bg = np.median(self.img[max(0, y1-5):min(height, y2+5), 
                                    max(0, x1-5):min(width, x2+5)], axis=(0,1)).astype(np.uint8)
            cv2.rectangle(self.img, (x1, y1), (x2, y2), local_bg.tolist(), -1)
        
        return self.img
        
    def ui_detect(self, mask: bool = False, ocr_json: Optional[str] = None):
        # 이미지의 평균 색상 계산 
        if mask:
            with open(ocr_json, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            self.img = self._mask_image(json_data)
        
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ui_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 10 :
                element = {
                    "type": "",
                    "size": {"width": w, "height": h},
                    "center": {"x": x + w / 2, "y": y + h / 2},
                    "bbox": {
                        "x1": x,
                        "y1": y,
                        "x2": x + w,
                        "y2": y + h
                    }
                }
                ui_boxes.append(element)
                
        return ui_boxes
    
    def merge_overlapping_or_close_boxes(self, boxes, threshold_ratio=0.01):
        # threshold_ratio: 예를 들어 0.01이면, 이미지 짧은 변의 1%만큼 떨어져 있으면 병합
        height, width = self.img.shape[:2]
        threshold = int(min(width, height) * threshold_ratio)

        def is_overlap_or_close(box1, box2):
            return not (
                box1["bbox"]["x2"] + threshold < box2["bbox"]["x1"] or
                box1["bbox"]["x1"] > box2["bbox"]["x2"] + threshold or
                box1["bbox"]["y2"] + threshold < box2["bbox"]["y1"] or
                box1["bbox"]["y1"] > box2["bbox"]["y2"] + threshold
            )

        merged = []
        used = [False] * len(boxes)

        for i, box in enumerate(boxes):
            if used[i]:
                continue
            x1, y1, x2, y2 = box["bbox"]["x1"], box["bbox"]["y1"], box["bbox"]["x2"], box["bbox"]["y2"]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(boxes):
                    if used[j]:
                        continue
                    if is_overlap_or_close({"bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}, other):
                        ox1, oy1, ox2, oy2 = other["bbox"]["x1"], other["bbox"]["y1"], other["bbox"]["x2"], other["bbox"]["y2"]
                        x1, y1, x2, y2 = min(x1, ox1), min(y1, oy1), max(x2, ox2), max(y2, oy2)
                        used[j] = True
                        changed = True
            merged.append({
                "type": box["type"],
                "size": box["size"],
                "center": box["center"],
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

        for box in range(len(merged)):
            if abs(int(merged[box]["size"]["width"]) - int(merged[box]["size"]["height"])) < int(merged[box]["size"]["height"])*0.4:
                merged[box]["type"] = "icon"
            elif merged[box]["type"] == "":
                merged[box]["type"] = "ui"

        return merged

    def merge_boxes(self, ocr_boxes, ui_boxes, visualize: bool = False):
        all_boxes = ocr_boxes + ui_boxes
        filtered_boxes = self.merge_overlapping_or_close_boxes(all_boxes)
        for box in filtered_boxes:
            cv2.rectangle(self.img, (int(box["bbox"]["x1"]), int(box["bbox"]["y1"])), (int(box["bbox"]["x2"]), int(box["bbox"]["y2"])), (0, 0, 255), 2)

        if visualize:
            # === 결과 시각화 ===
            plt.figure(figsize=(12, 10))
            plt.imshow(self.img)
            plt.axis("off")
            plt.show()

        cv2.imwrite(f"./resource/merged_all_boxes_{self.image_base_name}.png", self.img)

        with open(f"./resource/merged_boxes_{self.image_base_name}.json", "w", encoding="utf-8") as f:
            json.dump(filtered_boxes, f, ensure_ascii=False, indent=2)

        return filtered_boxes

    def assign_text_to_ui(self, ui_boxes, text_boxes):
        for ui in ui_boxes:
            for text in text_boxes:
                # bbox가 겹치면
                if self.is_overlap(ui["bbox"], text["bbox"]):
                    ui["text"] = text["text"]
        return ui_boxes

if __name__ == "__main__":

    image_dir = "./resource/"
    image_list = glob.glob(os.path.join(image_dir, "*.png"))

    for image_path in image_list:

        component = ComponentDetect(image_path)
        ocr_boxes = component.text_detect()
        ui_boxes = component.ui_detect()
        merged_boxes = component.merge_boxes(ocr_boxes, ui_boxes)

        img = cv2.imread(image_path)
        for box in merged_boxes:
            if box["type"] == "icon":
                cv2.rectangle(img, (int(box["bbox"]["x1"]), int(box["bbox"]["y1"])), (int(box["bbox"]["x2"]), int(box["bbox"]["y2"])), (0, 0, 255), 2)
        cv2.imwrite(f"./resource/merged_boxes_{os.path.splitext(os.path.basename(image_path))[0]}.png", img)
