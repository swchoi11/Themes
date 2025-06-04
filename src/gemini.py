import os
import glob
import time
import json
from typing import Optional, List
from dotenv import load_dotenv
from google import genai
from google.genai import types

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.prompt import Prompt
from src.utils.logger import init_logger
from src.utils.model import ResultModel, Result

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
    def _call_gemini_image(self, prompt, image, issue_type: str = "", index: int = 0) -> ResultModel:
        target_image = self.client.files.upload(file=image)
        logger.info(f"image 업로드 완료: {image}")

        logger.info(f"gemini 호출 시작")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
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
            image_path=image,
            index=index,
            issue_type=issue_type,
            issue_location=result.issue_location,
            issue_description=result.issue_description
        )

    @retry_with_delay
    def _call_gemini_image_text(self, prompt, image, text, issue_type: str = "", index: int = 0) -> ResultModel:
        target_image = self.client.files.upload(file=image)
        logger.info(f"image 업로드 완료: {image}")

        logger.info(f"gemini 호출 시작")
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                target_image,
                text,
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
            image_path=image,
            index=index,
            issue_type=issue_type,
            issue_location=result.issue_location,
            issue_description=result.issue_description
        )

    def generate_response(self, prompt, image, text: Optional[str] = None, issue_type: str = "", index: int = 0) -> ResultModel:
        if text:
            return self._call_gemini_image_text(prompt, image, text, issue_type, index)
        else:
            return self._call_gemini_image(prompt, image, issue_type, index)
        
    def detect_all_issues(self) -> List[ResultModel]:
        issues = []
        prompts_info = [
            (Prompt.calender_text_issue(), "calendar_text"),
            (Prompt.calender_date_issue(), "calendar_date"),
            (Prompt.clock_issue(), "clock"),
            (Prompt.interaction_issue(), "interaction"),
        ]
        
        for index, (prompt, issue_type) in enumerate(prompts_info):
            response = self.generate_response(prompt, self.image_path, issue_type=issue_type, index=index)
            issues.append(response)
        
        return issues

    def sort_issues(self, issues: List[ResultModel]) -> ResultModel:
        prompt = Prompt.sort_detected_issues_prompt()
        # issues를 문자열로 변환해서 전달
        issues_text = json.dumps([{
            "index": issue.index,
            "issue_type": issue.issue_type,
            "issue_location": issue.issue_location,
            "issue_description": issue.issue_description
        } for issue in issues], ensure_ascii=False, indent=2)
        
        response = self.generate_response(prompt, self.image_path, text=issues_text, issue_type="sorted", index=-1)
        return response
