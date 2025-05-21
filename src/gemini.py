from google import genai
from google.genai import types
from src.prompt import Prompt
import glob
import os
import time
from dotenv import load_dotenv

class Gemini:
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv('API_KEY'))
        self.model = 'gemini-2.0-flash-001'
        self.max_retries = 5  
        self.initial_delay = 1
        raw_issues = open('./resource/raw_issues.txt', 'r').read()
        self.raw_issues = raw_issues.split('\n')

    def retry_with_delay(func):
        def wrapper(self, *args, **kwargs):
            delay = self.initial_delay
            for attempt in range(self.max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    time.sleep(delay)
                    delay *= 2
        return wrapper
            
    def _extract_issues(self, cluster_id):

        image_list = glob.glob(f"./output/{cluster_id}/*.png")

        rows = ''

        for image_path in image_list:
            rows += os.path.basename(image_path) + ','

        rows = rows.replace('default', '')
        rows = rows.replace('Default', '')
        
        rows = rows.replace('preview', '')
        rows = rows.replace('Preview', '')

        return rows
    
    @retry_with_delay
    def relevant_issues(self, rows):
        prompt = Prompt.relevant_issues_prompt()

        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ['issue_type'],
                "properties": {
                    "issue_type": {"type": "string"},
                }
            }
        }

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                rows
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )

        return response.text

    @retry_with_delay
    def generate_description(self, target_image_dir, cluster_main_image_dir, relevant_issues):
        prompt = Prompt.generate_description_prompt()
        target_image = self.client.files.upload(file=target_image_dir)
        cluster_main_image = self.client.files.upload(file=cluster_main_image_dir)

        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ['application','function', 'issue', 'detail'],
                "properties": {
                    "application": {"type": "string"},
                    "function": {"type": "string"},
                    "issue": {"type": "string"},
                    "detail": {"type": "string"},
                }
            }
        }
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                target_image,
                cluster_main_image,
                relevant_issues
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )

        return response.text
    
    @retry_with_delay
    def only_visibility(self, target_image_dir):
        prompt = Prompt.only_visibility_prompt()
        target_image = self.client.files.upload(file=target_image_dir)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                target_image
            ])
        
        return response.text
