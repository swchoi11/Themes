import numpy as np
import cv2
import os
from src.utils.detect import Detect
from src.utils.model import ResultModel, EvalKPI
from src.utils.utils import bbox_to_location
from src.utils.logger import init_logger

logger = init_logger()

class Cutoff:
    def __init__(self, file_path: str):
        if file_path.endswith('.png'):
            self.image_path = file_path
            self.xml_path = file_path.replace('.png', '.xml')
        else:
            self.xml_path = file_path
            self.image_path = file_path.replace('.xml', '.png')

        self.detect = Detect(self.xml_path)
        self.img = cv2.imread(self.image_path)
        self.output_path = f"./output/images/{os.path.basename(self.image_path)}"

    def run_radio_button_check(self):
        components = self.detect.get_class_components("android.widget.RadioButton")
        issues = []
        logger.info(f"라디오 버튼 검사 시작")
        bounds_str = ""
        for component in components:
            logger.info(f"라디오 버튼 검사 중: {component['bounds']}")
            x1, y1, x2, y2 = component['bounds']
            crop_img = self.img[y1:y2, x1:x2]
            # cv2.imwrite(f"./output/radio_button_{x1}_{y1}_{x2}_{y2}.png", crop_img)
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=2, maxLineGap=0.5)
            h, w = gray.shape
            min_line_length_ratio = 0.05

            if lines is not None:
                for line in lines:
                    x1_line, y1_line, x2_line, y2_line = line[0]
                    line_length = np.sqrt((x2_line - x1_line) ** 2 + (y2_line - y1_line) ** 2)

                    if line_length > min_line_length_ratio * max(h, w):
                        # 컷오프 이슈 발견
                        cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(self.img, f"cutoff", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.imwrite(self.output_path, self.img)
                        
                        bounds_str += f"({x1}, {y1}, {x2}, {y2})"
                        break  # 한 컴포넌트에서 이슈를 찾으면 다음 컴포넌트로

            location_id = bbox_to_location([x1, y1, x2, y2], self.img.shape[0], self.img.shape[1])
            location_type = EvalKPI.LOCATION[location_id]

            result = ResultModel(
                filename=self.image_path,
                issue_type='cutoff',
                component_id=int(component['index']),
                ui_component_id="2",
                ui_component_type="RadioButton",
                score="",
                location_id=location_id,
                location_type=location_type,
                bbox=[x1, y1, x2, y2],
                description_id="7",
                description_type="아이콘의 가장자리가 보이지 않음거나 잘려보임(이미지 제외)",
                description=f"컴포넌트 영역 {bounds_str}에서 컷오프 이슈 발생 : {component['type'] or 'Unknown'}",
            )
        issues.append(result)

        return issues



