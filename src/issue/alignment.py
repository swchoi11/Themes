import cv2
import numpy as np
from xml.etree import ElementTree as ET
import json
import re
from typing import List, Tuple, Dict, Optional
from src.match import Match
from src.utils.utils import get_bounds
from src.utils.model import ResultModel
from src.utils.detect import Detect
from src.gemini import Gemini
from src.utils.prompt import Prompt

class Align:
    def __init__(self, file_path: str):
        if file_path.endswith('.png'):
            self.image_path = file_path
            self.xml_path = file_path.replace('.png', '.xml')
        else:
            self.image_path = file_path.replace('.xml', '.png')
            self.xml_path = file_path
        
        # Detect 모듈 초기화
        self.detector = Detect(self.image_path)
            
    def run_alignment_check(self) -> List[ResultModel]:

        groups = self.get_bound_groups()
    

        text = self.groups_to_text(groups)
        print(text)

        gemini = Gemini(self.image_path)

        response = gemini.generate_response(Prompt.alignment_check_prompt(), text=text)
        response = json.loads(response)

        for res in response:
            print(res)
            if not res['aligned']:
                img = cv2.imread(self.image_path)
                h, w = img.shape[:2]  # h: 높이, w: 너비
                
                # bounds가 [y1, y2] 형태라고 가정 (수직 구간)
                y1, y2 = res['bounds'][0], res['bounds'][1]
                
                # 전체 이미지 너비에 걸쳐 수직 구간을 표시
                cv2.rectangle(img, (0, int(y1)), (w, int(y2)), (0, 0, 255), 2)
                                
                cv2.imwrite(f"alignment_issue.png", img)
                print(f"정렬 문제 발견: y축 범위 {y1}-{y2}")
                return [res['bounds']]
        
        return []

    
    def get_bound_groups(self):
        components = self.detector.get_valid_components()
        img = cv2.imread(self.image_path)

        for comp in components:
            x1, y1, x2, y2 = comp['bounds']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite("components.png", img)

        sorted_components = sorted(components, key=lambda x: x['bounds'][1])

        groups = []
        for comp in sorted_components:
            vertical_bound = (comp['bounds'][1], comp['bounds'][3])
            
            if not groups:
                groups.append({"bounds": vertical_bound, "components": [comp['bounds']]})
                continue
            
            overlap_count, overlapping_indices = self._check_overlap(groups, vertical_bound)
            
            if overlap_count == 0:
                # 겹치지 않는 경우 새로운 그룹 생성
                groups.append({"bounds": vertical_bound, "components": [comp['bounds']]})
            
            elif overlap_count == 1:
                # 하나의 그룹과 겹치는 경우 해당 그룹에 추가하고 범위 확장
                idx = overlapping_indices[0]
                existing_start, existing_end = groups[idx]['bounds']
                new_start, new_end = vertical_bound
                
                # 범위 확장 (최소값과 최대값으로)
                groups[idx]['bounds'] = (min(existing_start, new_start), max(existing_end, new_end))
                groups[idx]['components'].append(comp['bounds'])
            
            else:
                # 여러 그룹과 겹치는 경우 - 모든 겹치는 그룹을 병합
                all_components = [comp]
                min_start = vertical_bound[0]
                max_end = vertical_bound[1]
                
                # 역순으로 제거해야 인덱스 꼬임 방지
                for idx in sorted(overlapping_indices, reverse=True):
                    all_components.extend(groups[idx]['components'])
                    existing_start, existing_end = groups[idx]['bounds']
                    min_start = min(min_start, existing_start)
                    max_end = max(max_end, existing_end)
                    groups.pop(idx)
                
                # 새로운 병합된 그룹 추가
                groups.append({"bounds": (min_start, max_end), "components": all_components})
        
        return groups

    def _check_overlap(self, groups, vertical_bound):
        overlapping_groups = []
        
        for i, group in enumerate(groups):
            existing_start, existing_end = group['bounds']
            new_start, new_end = vertical_bound
            
            # 두 범위가 겹치는지 확인
            if max(existing_start, new_start) < min(existing_end, new_end):
                overlapping_groups.append(i)
        
        return len(overlapping_groups), overlapping_groups
    

    def groups_to_text(self, groups):
        text = ""
        for group in groups:
            text += f"{group}\n"
        return text

    
# 다이얼러 전용 정렬 확인 함수 (기존 코드 유지)
def get_dial_alignment(image_path: str):
    """다이얼러 앱 전용 정렬 확인 함수"""
    try:
        from paddleocr import PaddleOCR
        
        ocr = PaddleOCR(
            det_model_dir='./src/weights/en_PP-OCRv3_det_infer',
            rec_model_dir='./src/weights/en_PP-OCRv3_rec_infer',
            cls_model_dir='./src/weights/ch_ppocr_mobile_v2.0_cls_infer',
            lang='en',
            use_angle_cls=False,
            use_gpu=False, 
            show_log=False,
        )

        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        result = ocr.ocr(image_path)[0]
        if not result:
            return []

        # 3등분 기준 중심선
        expected_centers = [
            w * 1/6,  # 1열 (좌)
            w * 3/6,  # 2열 (중앙)
            w * 5/6   # 3열 (우)
        ]

        threshold = 30  # 허용 편차(px)
        misaligned = []

        for line in result:
            box, (text, conf) = line
            x_coords = [pt[0] for pt in box]
            x_center = (min(x_coords) + max(x_coords)) / 2

            # 어떤 열에 해당하는지 추정
            col_idx = np.argmin([abs(x_center - cx) for cx in expected_centers])
            offset = abs(x_center - expected_centers[col_idx])

            if offset > threshold:
                misaligned.append({
                    "text": text,
                    "x_center": x_center,
                    "expected": expected_centers[col_idx],
                    "offset": offset
                })

        return misaligned
        
    except ImportError:
        print("PaddleOCR를 사용할 수 없습니다. 다이얼러 정렬 확인을 건너뜁니다.")
        return []
    except Exception as e:
        print(f"다이얼러 정렬 확인 중 오류: {e}")
        return []