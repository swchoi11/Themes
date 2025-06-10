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

class GeminiClient:
    """Gemini API 클라이언트의 기본 기능을 담당하는 클래스"""
    
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
    def call_gemini_text(self, prompt, text):
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
    def call_gemini_image(self, prompt, image):
        target_image = self.client.files.upload(file=image)
        logger.info(f"image 업로드 완료: {image}")
        logger.info(f"gemini 호출 시작")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[target_image, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": Result.model_json_schema(),
            }
        )
        logger.info(f"gemini 호출 완료")
        return Result.model_validate(json.loads(response.text))

    @retry_with_delay
    def call_gemini_image_text(self, prompt, image, text):
        target_image = self.client.files.upload(file=image)
        logger.info(f"image 업로드 완료: {image}")
        logger.info(f"gemini 호출 시작")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, text, target_image],
            config={
                "response_mime_type": "application/json",
                "response_schema": Result.model_json_schema(),
            }
        )
        logger.info(f"gemini 호출 완료")
        return Result.model_validate(json.loads(response.text))
    
    @retry_with_delay
    def call_gemini_images(self, prompt, image1, image2):
        target_image1 = self.client.files.upload(file=image1)
        logger.info(f"image 업로드 완료: {image1}")
        target_image2 = self.client.files.upload(file=image2)
        logger.info(f"image 업로드 완료: {image2}")
        logger.info(f"gemini 호출 시작")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, target_image1, target_image2],
            config={
                "response_mime_type": "application/json",
                "response_schema": Result.model_json_schema(),
            }
        )
        logger.info(f"gemini 호출 완료")
        return Result.model_validate(json.loads(response.text))


class ImageAnalyzer:
    """이미지 기반 분석을 담당하는 클래스"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.xml_path = image_path.replace('.png', '.xml')
        self.gemini_client = GeminiClient()
    
    def _create_result_model(self, result: Result, issue_type: str, description_id: str) -> ResultModel:
        """Result를 ResultModel로 변환하는 헬퍼 메서드"""
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
    
    def design_issues(self) -> List[ResultModel]:
        """디자인 이슈 분석"""
        issues = []
        prompt_info = [(Prompt.interaction_issue(), "visibility")]

        for (prompt, issue_type) in prompt_info:
            result = self.gemini_client.call_gemini_image(prompt, self.image_path)
            issues.append(self._create_result_model(result, issue_type, "2"))
        
        return issues

    def layout_issues(self) -> List[ResultModel]:
        """레이아웃 이슈 분석"""
        issues = []
        detect = Detect(self.image_path)
        
        # 캘린더 컴포넌트 분석
        calender_components = detect.get_calender_components()
        if calender_components:
            issues.extend(self._analyze_calendar_components(calender_components))
        
        # 시계 컴포넌트 분석
        clock_components = detect.get_clock_components()
        if clock_components:
            issues.extend(self._analyze_clock_components(clock_components))
            
        return issues
    
    def _analyze_calendar_components(self, components) -> List[ResultModel]:
        """캘린더 컴포넌트 분석"""
        issues = []
        for comp in components:
            bound = get_bounds(comp.get('bounds'))
            img = cv2.imread(self.image_path)
            calender_img = img[bound[1]:bound[3], bound[0]:bound[2]]
            cv2.imwrite('./output/images/calender_img.png', calender_img)
            
            result = self.gemini_client.call_gemini_image(
                Prompt.calender_text_issue(), 
                './output/images/calender_img.png'
            )
            issues.append(self._create_result_model(result, "design", "9"))
        return issues
    
    def _analyze_clock_components(self, components) -> List[ResultModel]:
        """시계 컴포넌트 분석"""
        issues = []
        for comp in components:
            bound = get_bounds(comp.get('bounds'))
            img = cv2.imread(self.image_path)
            clock_img = img[bound[1]:bound[3], bound[0]:bound[2]]
            cv2.imwrite('./output/images/clock_img.png', clock_img)
            header_img = img[0:img.shape[0]//10, 0:img.shape[1]]
            cv2.imwrite('./output/images/header_img.png', header_img)
            
            result = self.gemini_client.call_gemini_images(
                Prompt.clock_issue(), 
                './output/images/header_img.png', 
                './output/images/clock_img.png'
            )
            issues.append(self._create_result_model(result, "design", "A"))
            
            # 임시 파일 정리
            os.remove('./output/images/clock_img.png')
            os.remove('./output/images/header_img.png')
        return issues


class IssueProcessor:
    """JSON 기반 이슈 처리를 담당하는 클래스"""
    
    def __init__(self):
        self.gemini_client = GeminiClient()
    
    def sort_issues(self, json_filename: str) -> str:
        """이슈들을 정렬하고 최종 결과 생성"""
        with open(json_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        issues_by_file = defaultdict(list)
        for item in json_data:
            filename = item['filename']
            issues_by_file[filename].append(item)

        final_issues = []
        print(f"총 {len(issues_by_file)}개 파일 처리")
        
        for filename, file_issues in issues_by_file.items():
            print(f"파일 {filename}의 이슈 {len(file_issues)}개 처리 중...")
            try:
                valid_issues = [issue for issue in file_issues if issue['bbox'] != []]
                print(f"유효한 이슈: {len(valid_issues)}개")
                
                if len(valid_issues) == 0:
                    # 유효한 이슈가 없는 경우 -> 정상으로 처리
                    normal_issue = {
                        "filename": filename,
                        "issue_type": "normal",
                        "component_id": 0,
                        "ui_component_id": "",
                        "ui_component_type": "",
                        "severity": "0",
                        "location_id": "",
                        "location_type": "",
                        "bbox": [],
                        "description_id": "",
                        "description_type": "",
                        "description": "문제가 없습니다.",
                        "ai_description": ""
                    }
                    final_issues.append(normal_issue)
                    
                elif len(valid_issues) == 1:
                    # 유효한 이슈가 1개인 경우 -> 바로 추가
                    final_issues.append(valid_issues[0])
                    
                else:
                    # 유효한 이슈가 2개 이상인 경우 -> Gemini에게 검증 요청
                    sorted_issue = self._sort_issues_by_file(filename, valid_issues)
                    if sorted_issue:
                        final_issues.append(sorted_issue)
                    
            except Exception as e:
                print(f"오류 ({filename}): {e}")
                continue
        
        # 결과를 JSON 파일로 저장
        output_file_name = json_filename.replace('all_issues', 'final_issue')
        with open(output_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_issues, f, ensure_ascii=False, indent=2)
        
        print(f"최종 {len(final_issues)}개 이슈가 {output_file_name}에 저장되었습니다.")
        return output_file_name

    def _sort_issues_by_file(self, image_path: str, issues: List[dict]) -> dict:
        """단일 파일의 이슈들을 정렬 - 2개 이상의 이슈가 있을 때 Gemini가 가장 중요한 것 선택"""
        prompt = Prompt.sort_detected_issues_prompt()
        issue_text = json.dumps(issues, ensure_ascii=False, indent=2)
        
        try:
            # Gemini API 호출
            result = self.gemini_client.call_gemini_image_text(prompt, image_path, issue_text)
            
            # 원본 이슈 중에서 가장 우선순위가 높은 것을 선택
            # (Gemini의 응답을 기반으로 최종 이슈 결정)
            # 여기서는 첫 번째 이슈를 기본으로 하고 AI 설명만 업데이트
            selected_issue = issues[0].copy()  # 첫 번째 이슈를 기본으로
            selected_issue['ai_description'] = result.ai_description
            selected_issue['severity'] = result.severity
            
            return selected_issue
            
        except Exception as e:
            print(f"Gemini 검증 중 오류 발생: {e}")
            # 오류 발생 시 첫 번째 이슈 반환
            return issues[0]


# 기존 Gemini 클래스와의 호환성을 위한 래퍼
class Gemini:
    """기존 코드와의 호환성을 위한 래퍼 클래스"""
    
    def __init__(self, target_file: str):
        # 파일 타입에 따라 적절한 분석기 선택
        if target_file.endswith('.json'):
            self.processor = IssueProcessor()
            self.analyzer = None
        else:
            self.analyzer = ImageAnalyzer(target_file)
            self.processor = IssueProcessor()
    
    def design_issues(self) -> List[ResultModel]:
        if self.analyzer is None:
            raise ValueError("이미지 분석기가 초기화되지 않았습니다.")
        return self.analyzer.design_issues()
    
    def layout_issues(self) -> List[ResultModel]:
        if self.analyzer is None:
            raise ValueError("이미지 분석기가 초기화되지 않았습니다.")
        return self.analyzer.layout_issues()
    
    def sort_issues(self, json_filename: str) -> str:
        return self.processor.sort_issues(json_filename) 