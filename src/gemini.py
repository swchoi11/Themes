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
from common.prompt import Prompt
from common.logger import init_logger
from utils.schemas import Issue

logger = init_logger()

class Gemini:
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv('API_KEY'))
        self.model = 'gemini-2.0-flash'

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
    def _call_gemini_image(self, prompt, image) -> Issue:
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
                "response_schema": Issue.model_json_schema(),
            }
        )
        logger.info(f"gemini 호출 완료")
        return Issue.model_validate(json.loads(response.text))

    @retry_with_delay
    def _call_gemini_image_text(self, prompt, image, text) -> Issue:
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
                "response_schema": Issue.model_json_schema(),
            }
        )
        logger.info(f"gemini 호출 완료")
        return Issue.model_validate(json.loads(response.text))

    def generate_response(self, prompt, image, text: Optional[str] = None) -> Issue:
        if not text is None:
            return self._call_gemini_image_text(prompt, image, text)
        else:
            return self._call_gemini_image(prompt, image)

    def detect_all_issues(self, image) -> List[Issue]:
        issues = []
        prompts = [
            Prompt.calender_text_issue(),
            Prompt.calender_date_issue(),
            Prompt.clock_issue(),
            Prompt.highlight_issue(),
            Prompt.interaction_issue(),
        ]
        for prompt in prompts:
            response = self.generate_response(prompt, image)
            issues.append(response)

        return issues

    def sort_issues(self, image, text) -> Issue:
        prompt = Prompt.sort_detected_issues_prompt()
        response = self.generate_response(prompt, image, text)
        return response


if __name__ == "__main__":
    gemini = Gemini()
    issues = gemini.detect_all_issues(
        image="./resource/setting icon same back button.jpg",
    )
    print(issues)