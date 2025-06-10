import os
import cv2
import numpy as np
from xml.etree import ElementTree as ET
import json
from typing import List
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
        file_name = os.path.basename(self.image_path)
        self.output_path = f'./output/images/{file_name}'
            
    def run_alignment_check(self) -> List[ResultModel]:

        groups = self.get_bound_groups()

        if not groups:
            return []

        text = self.groups_to_text(groups)

        gemini = Gemini(self.image_path)

        response = gemini.generate_response(Prompt.alignment_check_prompt(), text=text)
        response = json.loads(response)
        
        results = []

        for res in response:
            # print(res)
            if not res['aligned']:
                img = cv2.imread(self.image_path)
                h, w = img.shape[:2]  # h: 높이, w: 너비
                
                y1, y2 = res['bounds'][0], res['bounds'][1]
                
                # 전체 이미지 너비에 걸쳐 수직 구간을 표시
                cv2.rectangle(img, (0, int(y1)), (w, int(y2)), (52, 195, 235), 2)
                cv2.putText(img, "alignment issue", (0, int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 195, 235), 2)                                
                cv2.imwrite(self.output_path, img)

                result = ResultModel(
                    filename=self.image_path,
                    issue_type='alignment',
                    component_id=0,
                    ui_component_id="",
                    ui_component_type="",
                    severity="high",
                    location_id="",
                    location_type="",
                    bbox=[0, y1, w, y2],
                    description_id="4",
                    description_type="컴포넌트 내부 요소들의 수직/수평 정렬이 균일하지 않음",
                    description=f"정렬 문제 발견: y축 범위 {y1}-{y2}",
                    ai_description=""
                )
                results.append(result)
        
        if not results:
            results = [ResultModel(
                filename=self.image_path,
                issue_type="alignment",
                component_id=0,
                ui_component_id="",
                ui_component_type="",
                severity="high",
                location_id="",
                location_type="",
                bbox=[],
                description_id="4",
                description_type="컴포넌트 내부 요소들의 수직/수평 정렬이 균일하지 않음",
                description="",
                ai_description=""
            )]

        return results

    
    def get_bound_groups(self):
        components = self.detector.get_valid_components()
        
        sorted_components = sorted(components, key=lambda x: x['bounds'][1])

        groups = []
        if not components:
            return []
        
        for i, comp in enumerate(sorted_components):
            vertical_bounds = (comp['bounds'][1], comp['bounds'][3])
            
            self._add_component_to_groups(groups, vertical_bounds, comp['bounds'])
            
        self._validate_grouping(groups)
        
        return groups

    def _add_component_to_groups(self, groups, vertical_bounds, component_bounds):
        """
        컴포넌트를 적절한 그룹에 추가하거나 새 그룹을 생성
        
        포함/겹침 관계 처리 규칙:
        1. 2개 이상의 구간을 포함하는 커다란 컴포넌트 -> 새로운 구간으로 설정
           예: 기존 1~2, 2~3 → 새로 1~4 들어옴 → 결과: 1~2, 2~3, 1~4
        
        2. 단순히 일정 부분이 겹치는 컴포넌트 -> 서로 다른 구간으로 설정  
           예: 기존 1~4, 4~8 → 새로 4~6 들어옴 → 결과: 1~4, 4~8, 4~6
        
        3. 겹치는 구간이 있는 경우 -> 서로 다른 구간으로 설정
           예: 기존 1~4 → 새로 2~6 들어옴 → 결과: 1~4, 2~6
        """
        new_start, new_end = vertical_bounds
        
        # 1. 먼저 정확히 같은 bounds를 가진 그룹이 있는지 확인
        for group in groups:
            if group['bounds'] == vertical_bounds:
                # 같은 구간이면 해당 그룹에 컴포넌트 추가
                group['components'].append(component_bounds)
                return
        
        # 2. 기존 그룹들과의 관계 분석
        overlapping_groups = []  # 겹치는 그룹들의 인덱스
        contained_groups = []    # 새 컴포넌트에 완전히 포함되는 그룹들의 인덱스
        
        for i, group in enumerate(groups):
            existing_start, existing_end = group['bounds']
            
            # 겹침 여부 확인
            if self._has_overlap(existing_start, existing_end, new_start, new_end):
                overlapping_groups.append(i)
                
                # 새 컴포넌트가 기존 그룹을 완전히 포함하는지 확인
                if new_start <= existing_start and new_end >= existing_end:
                    contained_groups.append(i)
        
        # 3. 케이스별 처리
        if len(contained_groups) >= 2:
            # 케이스 1: 2개 이상의 구간을 포함하는 커다란 컴포넌트 
            # -> 기존 그룹들은 유지하고 새로운 구간을 추가
            groups.append({
                "bounds": vertical_bounds,
                "components": [component_bounds]
            })
        elif len(overlapping_groups) > 0:
            # 케이스 2, 3: 겹치는 구간이 있는 경우 
            # -> 기존 그룹들은 유지하고 새로운 구간을 추가
            groups.append({
                "bounds": vertical_bounds,
                "components": [component_bounds]
            })
        else:
            # 겹치지 않는 경우 -> 새로운 그룹 생성
            groups.append({
                "bounds": vertical_bounds,
                "components": [component_bounds]
            })
    
    def _has_overlap(self, start1, end1, start2, end2):
        """두 구간이 겹치는지 확인"""
        return max(start1, start2) < min(end1, end2)

    def groups_to_text(self, groups):
        text = ""
        for group in groups:
            text += f"{group}\n"
        return text

    def _validate_grouping(self, groups):
        """그룹핑 결과를 검증하고 출력"""
        
        for i, group in enumerate(groups):
            bounds = group['bounds']
            components = group['components']
            
            # 같은 bounds를 가진 다른 그룹이 있는지 확인
            duplicate_count = sum(1 for g in groups if g['bounds'] == bounds)

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