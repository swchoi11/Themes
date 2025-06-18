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
from src.utils.model import ResultModel, Result, EvalKPI
from src.utils.detect import Detect
from src.utils.utils import get_bounds, bbox_to_location
import cv2
import os
import pandas as pd
logger = init_logger()

class GeminiClient:
    """Gemini API 클라이언트의 기본 기능을 담당하는 클래스"""
    
    def __init__(self):
        load_dotenv()
        
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.client = self._get_client()
        self.model = 'gemini-2.0-flash' #'gemini-2.5-flash-preview-05-20' | 'gemini-2.5-pro-preview-06-05'
        self.max_retries = 5  
        self.initial_delay = 1

    def _load_api_keys(self):
        """환경변수에서 API 키들을 로드"""
        keys = []
        
        # API_KEY_1, API_KEY_2, API_KEY_3... 형태로 여러 키 지원
        for i in range(1, 4):  # 최대 9개 키 지원
            key = os.getenv(f'API_KEY{i}')
            if key:
                keys.append(key)
                    
        if not keys:
            raise ValueError("API 키가 설정되지 않았습니다. API_KEY 또는 API_KEY_1, API_KEY_2... 환경변수를 설정해주세요.")
            
        # logger.info(f"총 {len(keys)}개의 API 키가 로드되었습니다.")
        return keys
    
    def _get_client(self):
        """현재 API 키로 클라이언트 생성"""
        return genai.Client(api_key=self.api_keys[self.current_key_index])
    
    def _rotate_key(self):
        """다음 API 키로 로테이션"""
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = self._get_client()
        logger.warning(f"API 키 변경: {old_index + 1} -> {self.current_key_index + 1}")

    def retry_with_delay(func):
        def wrapper(self, *args, **kwargs):
            delay = self.initial_delay
            for attempt in range(self.max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    if any(keyword in error_msg for keyword in ['quota', 'rate limit', 'resource exhausted', 'limit exceeded']):
                        logger.warning(f"API 할당량 초과 감지: {e}")
                        if len(self.api_keys) > 1:  # 키가 여러개인 경우만 로테이션
                            self._rotate_key()
                            # 키 로테이션 후 즉시 재시도 (delay 없이)
                            continue
                    
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
        # print(response.text)
        result = Result.model_validate(json.loads(response.text))
        if result.score == "":
            result.score = "5"
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
    
    def _create_result_model(self, result: Result, component_info: dict) -> ResultModel:
        """Result를 ResultModel로 변환하는 헬퍼 메서드"""
        return ResultModel(
            filename=self.image_path,
            issue_type=component_info.get('issue_type', ''),
            component_id=component_info.get('component_id', 0),
            ui_component_id=component_info.get('ui_component_id', ''),
            ui_component_type=component_info.get('ui_component_type', ''),
            score=result.score,
            location_id=component_info.get('location_id', ''),
            location_type=component_info.get('location_type', ''),
            bbox=result.bbox,
            description_id=component_info.get('description_id', ''),
            description_type=component_info.get('description_type', ''),
            description=component_info.get('description', '')
        )
    
    def issue_score(self,  issue: ResultModel) -> ResultModel:
        """이슈 점수 계산"""
        prompt = Prompt.issue_score_prompt()
        issue_text = json.dumps(issue.model_dump(), ensure_ascii=False, indent=2)

        result = self.gemini_client.call_gemini_image_text(prompt, self.image_path, issue_text)
        issue.score = result.score
        issue.description = result.description
        return issue

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

            location_id = bbox_to_location(bound, img.shape[0], img.shape[1])
            location_type = EvalKPI.LOCATION[location_id]

            component_info = {
                "issue_type": "design",
                "component_id": comp.get('index', 0),
                "description_id": "9",
                "description_type": "달력 아이콘에서 요일 글자가 테두리를 벗어남",
                "ui_component_type": "ImageButton",
                "ui_component_id": "B",
                "bbox": bound,
                "location_id": location_id,
                "location_type": location_type,
            }

            issues.append(self._create_result_model(result, component_info))
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

            location_id = bbox_to_location(bound, img.shape[0], img.shape[1])
            location_type = EvalKPI.LOCATION[location_id]

            component_info = {
                "issue_type": "design",
                "component_id": comp.get('index', 0),
                "description_id": "A",
                "description_type": "앱 내 달력, 시간 아이콘이 status bar 등에 보이는 실제 현재 날짜, 시각과 매칭되지 않음",
                "ui_component_type": "ImageButton",
                "ui_component_id": "B",
                "bbox": bound,
                "location_id": location_id,
                "location_type": location_type,
            }
            # 임시 파일 정리
            os.remove('./output/images/clock_img.png')
            os.remove('./output/images/header_img.png')
        
            issues.append(self._create_result_model(result, component_info))
        return issues


class IssueProcessor:
    """JSON 기반 이슈 처리를 담당하는 클래스"""
    
    def __init__(self):
        self.gemini_client = GeminiClient()
    
    def sort_issues(self, json_filename: str) -> str:
        """이슈들을 정렬하고 최종 결과 생성"""
        with open(json_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 테스트 이미지 목록 로드
        test_images = self._load_test_images()
        
        issues_by_file = defaultdict(list)
        for item in json_data:
            filename = item['filename']
            # 테스트 이미지에 있는 파일만 처리
            if filename not in test_images:
                continue
                
            if item['score'] == "":
                item['score'] = "5"
            item['score'] = int(item['score'])
            issues_by_file[filename].append(item)

        final_issues = []
        normal_issues = []
        print(f"총 {len(issues_by_file)}개 파일 처리")
        
        for filename, file_issues in issues_by_file.items():
            print(f"파일 {filename}의 이슈 {len(file_issues)}개 처리 중...")
            try:
                valid_issues = [issue for issue in file_issues if  issue['score'] < 5] #issue['bbox'] != [] and
                print(f"유효한 이슈: {len(valid_issues)}개")
                
                if len(valid_issues) == 0:
                    # 유효한 이슈가 없는 경우 -> final_inference로 추가 검증
                    normal_issues.append({
                        "filename": filename,
                    })
                    
                elif len(valid_issues) == 1:
                    # 유효한 이슈가 1개인 경우 -> 바로 추가
                    if valid_issues[0]['issue_type'] == "not_processed":

                        normal_issues.append({
                            "filename": filename,
                        })

                    else:
                        final_issues.append(valid_issues[0])
                    
                else:
                    # 유효한 이슈가 2개 이상인 경우 -> Gemini에게 검증 요청
                    sorted_issue = self._sort_issues_by_file(valid_issues)
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

        normal_file_name = json_filename.replace('all_issues/', '').replace('jsons/','').replace('.json', '-normal.txt')
        with open(normal_file_name, 'w', encoding='utf-8') as f:
            pd.DataFrame(normal_issues).to_csv(f, 
                                        mode='a',
                                        index=False,
                                        header=not os.path.isfile(normal_file_name),
                                        encoding='utf-8-sig')

        print(f"normal 이슈{len(normal_issues)}개가 {normal_file_name}에 저장되었습니다.")

        return output_file_name

    def _load_test_images(self) -> set:
        """테스트 이미지 목록을 로드합니다."""
        import glob
        
        # 이미지 리스트 파일 찾기
        image_list_files = glob.glob('./util_files/vm*_image_list.csv')
        
        if not image_list_files:
            print("테스트 이미지 리스트 파일을 찾을 수 없습니다.")
            return set()
        
        image_list_file = image_list_files[0]
        test_images = set()
        
        try:
            with open(image_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    filename = line.strip()
                    if filename:
                        # 파일명을 JSON의 filename 형식에 맞게 변환
                        if not filename.startswith('./'):
                            filename = f'./{filename}'
                        test_images.add(filename)
            
            print(f"테스트 이미지 {len(test_images)}개 로드 완료")
            return test_images
            
        except Exception as e:
            print(f"테스트 이미지 로드 중 오류: {e}")
            return set()

    def _sort_issues_by_file(self, issues: List[dict]) -> dict:
        """이슈를 정렬하는 메서드"""
        # 각 이슈의 score를 정수로 변환
        for issue in issues:
            if isinstance(issue['score'], str):
                issue['score'] = int(issue['score'])
            # description_id가 문자열인 경우 16진수로 변환 시도
            if isinstance(issue['description_id'], str):
                try:
                    issue['description_id'] = int(issue['description_id'], base=16)
                except ValueError:
                    # 16진수 변환 실패시 10진수로 변환 시도
                    try:
                        issue['description_id'] = int(issue['description_id'])
                    except ValueError:
                        pass  # 변환 실패시 그대로 유지
        
        sorted_issues = sorted(issues, key=lambda x: (x['score'], x['description_id']), reverse=True)
        return sorted_issues[0]


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
    
    def issue_score(self, issue: ResultModel) -> ResultModel:
        if self.analyzer is None:
            raise ValueError("이미지 분석기가 초기화되지 않았습니다.")
        return self.analyzer.issue_score(issue)
    
    def sort_issues(self, json_filename: str) -> str:
        return self.processor.sort_issues(json_filename) 