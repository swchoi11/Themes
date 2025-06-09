import os
import glob
import time
import json
from typing import Optional, List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from collections import defaultdict

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.prompt import Prompt
from src.utils.logger import init_logger
from src.utils.model import ResultModel, Result
from src.utils.detect import Detect
from src.utils.utils import get_bounds
import cv2
import os

logger = init_logger()

class Gemini:
    def __init__(self, target_file):        
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv('API_KEY'))
        self.model = 'gemini-2.0-flash'

        # 항상 PNG 파일을 image_path로, XML 파일을 xml_path로 설정
        if target_file.endswith('.xml'):
            self.xml_path = target_file
            self.image_path = target_file.replace('.xml', '.png')
        else:  # PNG 파일인 경우
            self.image_path = target_file
            self.xml_path = target_file.replace('.png', '.xml')

        self.max_retries = 5  
        self.initial_delay = 1

    def retry_with_delay(func):
        def wrapper(self, *args, **kwargs):
            delay = self.initial_delay
            for attempt in range(self.max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    logger.error(f"gemini 호출 {attempt + 1}번째 실패: {e}")
                    time.sleep(delay)
                    delay *= 2
        return wrapper

    @retry_with_delay
    def _call_gemini_text(self, prompt, text, issue_type: str = "", index: int = 0) -> ResultModel:
        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bounds": {"type": "array", "items": {"type": "number"}},
                    "aligned": {"type": "boolean"}
                }
            }
        }
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, text],
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )
        logger.info(f"gemini 호출 완료")

        return response.text
    
    @retry_with_delay
    def _call_gemini_image(self, prompt, image, issue_type: str = "", description_id: int = 0) -> ResultModel:
        target_image = self.client.files.upload(file=image)
        logger.info(f"image 업로드 완료: {image}")

        logger.info(f"gemini 호출 시작")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                target_image,
                prompt,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Result.model_json_schema(),
            }
        )
        logger.info(f"gemini 호출 완료")

        # JSON 문자열을 파싱하여 Result 객체로 변환
        result = Result.model_validate(json.loads(response.text))
        
        # ResultModel로 변환하면서 추가 정보 채우기
        return ResultModel(
            filename=self.image_path,
            issue_type=issue_type,
            component_id=0,
            ui_component_id="",
            ui_component_type="",
            severity=result.severity,
            location_id="",
            location_type="",
            bbox=result.bbox,
            description_id=description_id,
            description_type="",
            description="상호작용 가능한 요소가 시각적으로 명확히 구분되지 않음",
            ai_description=result.ai_description
        )

    @retry_with_delay
    def _call_gemini_image_text(self, prompt, image, text, issue_type: str = "", description_id: int = 0) -> ResultModel:
        target_image = self.client.files.upload(file=image)
        logger.info(f"image 업로드 완료: {image}")

        logger.info(f"gemini 호출 시작")
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                text,
                target_image,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Result.model_json_schema(),
            }
        )
        logger.info(f"gemini 호출 완료")

        # JSON 문자열을 파싱하여 Result 객체로 변환
        result = Result.model_validate(json.loads(response.text))
        
        # ResultModel로 변환하면서 추가 정보 채우기
        return ResultModel(
            filename=self.image_path,
            issue_type=issue_type,
            component_id=0,
            ui_component_id="",
            ui_component_type="",
            severity=result.severity,
            location_id="",
            location_type="",
            bbox=result.bbox,
            description_id=description_id,
            description_type="",
            description="상호작용 가능한 요소가 시각적으로 명확히 구분되지 않음",
            ai_description=result.ai_description
        )
    
    def _call_gemini_images(self, prompt, image1, image2, issue_type: str = "", index: int = 0) -> ResultModel:
        target_image1 = self.client.files.upload(file=image1)
        logger.info(f"image 업로드 완료: {image1}")
        
        target_image2 = self.client.files.upload(file=image2)
        logger.info(f"image 업로드 완료: {image2}")
        
        logger.info(f"gemini 호출 시작")
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                target_image1,
                target_image2,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Result.model_json_schema(),
            }
        )
        logger.info(f"gemini 호출 완료")
        
        result = Result.model_validate(json.loads(response.text))
        
        return ResultModel(
            image_path=image2,
            index=index,
            issue_type=issue_type,
            issue_location=result.issue_location,
            issue_description=result.issue_description
        )
    
    def generate_response(self, prompt, image1: Optional[str] = None, image2: Optional[str] = None, text: Optional[str] = None, 
                          issue_type: str = "", index: str = "") -> ResultModel:
        if image1 and text:
            return self._call_gemini_image_text(prompt, image1, text, issue_type, index)
        elif text:
            return self._call_gemini_text(prompt, text, issue_type, index)
        elif image1:
            return self._call_gemini_image(prompt, image1, issue_type, index)
        elif image1 and image2:
            return self._call_gemini_images(prompt, image1, image2, issue_type, index)
        else:
            raise ValueError("image 또는 text 중 하나는 반드시 제공되어야 합니다.")
        
    def design_issues(self) -> List[ResultModel]:
        issues = []

        prompt_info = [
            (Prompt.interaction_issue(), "visibility"),
        ]

        for (prompt, issue_type) in prompt_info:
            response = self.generate_response(prompt, self.image_path, 
                                              issue_type=issue_type, 
                                              index="2")
            issues.append(response)
        
        return issues

    def layout_issues(self):
        issues = []

        detect = Detect(self.image_path)
        calender_components = detect.get_calender_components()
        clock_components = detect.get_clock_components()

        if calender_components:
            for comp in calender_components:
                bound = get_bounds(comp.get('bounds'))
                img = cv2.imread(self.image_path)
                calender_img = img[bound[1]:bound[3], bound[0]:bound[2]]
                cv2.imwrite('./output/images/calender_img.png', calender_img)
                result = self.generate_response(Prompt.calender_text_issue(),
                                                './output/images/calender_img.png', 
                                                issue_type="design",index="9")
                issues.append(result)

        
        if clock_components:
            for comp in clock_components:
                bound = get_bounds(comp.get('bounds'))
                img = cv2.imread(self.image_path)
                clock_img = img[bound[1]:bound[3], bound[0]:bound[2]]
                cv2.imwrite('./output/images/clock_img.png', clock_img)
                header_img = img[0:img.shape[0]//10, 0:img.shape[1]]
                cv2.imwrite('./output/images/header_img.png', header_img)
                result = self.generate_response(Prompt.clock_issue(), 
                                                './output/images/header_img.png', './output/images/clock_img.png',
                                                issue_type="design",index="A")
                issues.append(result)
                os.remove('./output/images/clock_img.png')
                os.remove('./output/images/header_img.png')
                
        return issues
        

    def sort_issues(self, json_filename):
        with open(json_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        issues_by_file = defaultdict(list)
        for item in json_data:
            filename = item['filename']
            issues_by_file[filename].append(item)

        final_issues = []
        for filename, file_issues in issues_by_file.items():
            print(f"파일 {filename}의 이슈 {len(file_issues)}개 처리 중...")
            try:
                sorted_issues = self.sort_issues_by_file(filename,file_issues)
                if sorted_issues:
                    final_issues.extend(sorted_issues)
                else:
                    final_issues.extend(
                        filename=filename,
                        issue_type="none"
                    )
            except Exception as e:
                print(f"오류 ({filename}): {e}")
                continue
        
        output_file_name = json_filename.replace('all_issues', 'final_issues')
        with open(output_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_issues, f, ensure_ascii=False, indent=2)

        return output_file_name

    def sort_issues_by_file(self, image_path, issues) -> ResultModel:
        prompt = Prompt.sort_detected_issues_prompt()
        issue_text = json.dumps(issues, ensure_ascii=False, indent=2)
        issue_text = str(issue_text)
        # print(issue_text)
        
        response = self.generate_response(prompt, image_path, text=issue_text)
        
        return response
