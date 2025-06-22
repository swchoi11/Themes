import os
import glob

import json
import ast
import cv2
import csv
import numpy as np

from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.gemini import Gemini
from google import genai
from google.genai import types

from tqdm import tqdm


class Config:
    """Configuration class for file paths and settings"""

    def __init__(self):
        self.SAVE_DIR = '../output/report'
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.VISUALIZATION_DIR = '../output/visualization/'
        self.JSON_DIR = '../output/json/'
        self.DATASET_PATH = '../resource/dataset.xlsx'  # 중간 산출물
        self.OUTPUT_RAW_REPORT_PATH = f'{self.SAVE_DIR}/raw.csv'
        self.OUTPUT_REPORT_PATH = f'{self.SAVE_DIR}/report.csv'
        self.OUTPUT_PARSED_PATH = f'{self.SAVE_DIR}/parse_report.csv'
        self.OUTPUT_RESULT_PATH = f'{self.SAVE_DIR}/final_report.csv'
        self.ERROR_LOG_PATH = f'{self.SAVE_DIR}/error.txt'
        self.ISSUE_REPORT_PATH = f'{self.SAVE_DIR}/issue_report.xlsx'

        # Processing settings
        self.BATCH_SIZE = 6
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5


class BoundingBoxVisualizer:
    """바운딩 박스 시각화를 위한 클래스"""

    def __init__(self):
        # 이슈 타입별 색상 정의
        self.issue_colors_mpl = {
            'normal': (0.5, 0.5, 0.5),  # 회색
            'alignment': (1.0, 0.0, 0.0),  # 빨강
            'cutoff': (0.0, 1.0, 1.0),  # 시안
            'design': (0.0, 1.0, 0.0),  # 초록
            'default': (1.0, 1.0, 0.0)  # 노랑
        }

    def parse_bbox(self, bbox_str) -> List[float]:
        """바운딩 박스 문자열 파싱"""
        if pd.isna(bbox_str) or bbox_str == '' or bbox_str == '[]':
            return []

        try:
            if isinstance(bbox_str, str):
                bbox = ast.literal_eval(bbox_str)
            else:
                bbox = bbox_str

            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(x) for x in bbox]
            else:
                return []
        except Exception as e:
            print(f"[WARNING] bbox 파싱 실패: {bbox_str}, 오류: {e}")
            return []

    def create_image_with_bboxes(self, image_path: str, issues: List[dict], temp_dir='./temp') -> str:
        """이미지에 모든 이슈의 바운딩 박스를 그려서 임시 파일로 저장"""

        if not os.path.exists(image_path):
            print(f"[ERROR] 이미지 파일 없음: {image_path}")
            return image_path

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 이미지 로드 실패: {image_path}")
            return image_path

        # BGR to RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 이미지 크기에 맞춰 figure 크기 조정
        dpi = 100
        fig_w = w / dpi
        fig_h = h / dpi

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(image_rgb)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # 각 이슈에 대해 바운딩 박스 그리기
        for idx, issue in enumerate(issues):
            issue_type = issue.get('issue_type', 'default')
            bbox = issue.get('bbox', '')
            color = self.issue_colors_mpl.get(issue_type, self.issue_colors_mpl['default'])

            if not bbox:
                # 바운딩 박스가 없으면 전체 이미지 크기로 설정
                rect = Rectangle((2, 2), w - 4, h - 4,
                                 linewidth=3, edgecolor=color,
                                 facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                ax.text(10, 10 + idx * 30, f'BOX {idx + 1:02d}: {issue_type.upper()}',
                        ha='left', va='top',
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle="square,pad=0.3", facecolor=color, alpha=0.9))
            else:
                # 정규화된 좌표인지 확인
                if all(0 <= coord <= 1 for coord in bbox):
                    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
                else:
                    x1, y1, x2, y2 = bbox

                # 좌표 유효성 검사
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                if x2 > x1 and y2 > y1:
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=3, edgecolor=color,
                                     facecolor='none', alpha=0.8)
                    ax.add_patch(rect)

                    label = f"BOX {idx + 1:02d}: {issue_type.upper()}"
                    ax.text(x1, y1, label,
                            ha='left', va='bottom',
                            fontsize=10, fontweight='bold', color='white',
                            bbox=dict(boxstyle="square,pad=0.1", facecolor=color, alpha=0.9))

        # 임시 파일 저장
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, os.path.basename(image_path))
        plt.savefig(temp_path, dpi=dpi, bbox_inches='tight',
                    pad_inches=0, facecolor='white', edgecolor='none')
        plt.close()
        return temp_path


class UIQualityInspector:
    """Main class for UI quality inspection using Gemini API"""

    def __init__(self, config: Config):
        self.config = config
        self.gemini = Gemini()
        self.visualizer = BoundingBoxVisualizer()
        self.client = None
        self.gemini = Gemini()

    def _get_visibility_inspection_prompt(self) -> str:

        return """
        You are an AI assistant that inspects the UI quality of Android applications. Please inspect the UI quality based on the screenshot of the Android application provided.
        The screenshot provided has bounding boxes that can distinguish UI components, and the names of UI components are labeled on the upper left of each bounding box (Text, Button, etc.)
        Please measure the following items for each UI component.
        - visibility: **Text**, **icons**, **imageviews**, **Image**, etc. are similar to the background color, making it difficult for people to see with their eyes.
        **Please exclude bounding boxes and labels on the top of bounding boxes from QA**
        Please also calculate the "score" for visibility issues. It has a value of 0 to 9. 9 is the highest visibility.
        
        - duplicate: **TextView**, **button** with different functional roles but using identical icon images, creating confusion for users about their distinct purposes. 
        Icons must be completely identical in all aspects including color, shape, size, and visual appearance to be considered duplicates. 
        **Please exclude bounding boxes and labels on the top of bounding boxes from QA** 
        Please also calculate the "score" for duplicate issues. It has a value of 0 to 3. 0 is the highest duplicate issue.
        
        **Please give a score for each UI component**
        **Please output only the 3 most problematic UI components**
        Please output in table format. The table must include the following contents.
        * UI component bound box label
        * Reason
        * Score
        Please answer in Korean.
        """

    def _get_cutoff_inspection_prompt(self) -> str:
        return """
        You are an AI assistant that inspects the UI quality of Android applications. Please inspect the UI quality based on the screenshot of the Android application provided.
        The screenshot provided has bounding boxes that can distinguish UI components, and the names of UI components are labeled on the upper left of each bounding box (RadioButton, ToggleButton, Switch, etc.)
        Please measure the following items for each UI component.
        - cut-off: **RadioButton**, **ToggleButton**, and **Switch** components for vertical truncation at top and bottom edges that distorts circular shapes into non-circular forms.
        **Please exclude bounding boxes and their labels from quality assessment**        
        **Please exclude bounding boxes and labels on the top of bounding boxes from QA**
        Please also calculate the "score" for cut-off issues. It has a value of 0 to 3. 0 is the highest cut-off issue.
        **Please give a score for each UI component**
        **Please output only the 3 most problematic UI components**
        Please output in table format. The table must include the following contents.
        * UI component bound box label
        * Reason
        * Score
        Please answer in Korean.
        """

    def _get_csv_parsing_prompt(self, csv_data: str) -> str:
         return f"""
         CSV 에서 필요한 정보를 추출해서 Example 형식에 맞춰 출력해 주세요. 
        - CSV는 \"sequence number, filename, description\" 으로 구성되어 있습니다.
        - description은 테이블을 포함한 markdown 입니다. 여기에는 UI 컴포넌트 level, reason, score 정보가 포함되어 있습니다.

        결과물은 코드가 아닌 CSV 입니다.
         * OUTPUT Format Example를 참조하여 csv 형태로 print 해주세요.
         * csv 내용만 출력하고, 그 외의 내용은 출력하지 마세요. 
         * 마크다운 (```csv)블럭은 사용하지 마세요.

        ### 출력 포맷
        filename-01,UI 컴포넌트 label, reason, score
        filename-01,UI 컴포넌트 label, reason, score

        ### CSV 데이터
        {csv_data}
        """

    import os
    import glob
    import json
    import ast
    import cv2
    import csv
    import numpy as np

    from typing import List, Optional
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from src.gemini import Gemini
    from google import genai
    from google.genai import types

    from tqdm import tqdm

    class Config:
        """Configuration class for file paths and settings"""

        def __init__(self):
            self.SAVE_DIR = '../output/report'
            os.makedirs(self.SAVE_DIR, exist_ok=True)
            self.VISUALIZATION_DIR = '../output/visualization'
            self.JSON_DIR = '../resource/json'  # JSON 파일 디렉토리 추가
            self.DATASET_PATH = '../resource/dataset.xlsx'  # 중간 산출물
            self.OUTPUT_RAW_REPORT_PATH = f'{self.SAVE_DIR}/raw.csv'
            self.OUTPUT_REPORT_PATH = f'{self.SAVE_DIR}/report.csv'
            self.OUTPUT_PARSED_PATH = f'{self.SAVE_DIR}/parse_report.csv'
            self.OUTPUT_RESULT_PATH = f'{self.SAVE_DIR}/final_report.csv'
            self.ERROR_LOG_PATH = f'{self.SAVE_DIR}/error.txt'
            self.ISSUE_REPORT_PATH = f'{self.SAVE_DIR}/issue_report.xlsx'

            # Processing settings
            self.BATCH_SIZE = 6
            self.MAX_RETRIES = 3
            self.RETRY_DELAY = 5

    class BoundingBoxVisualizer:
        """바운딩 박스 시각화를 위한 클래스"""

        def __init__(self):
            # 이슈 타입별 색상 정의
            self.issue_colors_mpl = {
                'normal': (0.5, 0.5, 0.5),  # 회색
                'alignment': (1.0, 0.0, 0.0),  # 빨강
                'cutoff': (0.0, 1.0, 1.0),  # 시안
                'design': (0.0, 1.0, 0.0),  # 초록
                'default': (1.0, 1.0, 0.0)  # 노랑
            }

        def parse_bbox(self, bbox_str) -> List[float]:
            """바운딩 박스 문자열 파싱"""
            if pd.isna(bbox_str) or bbox_str == '' or bbox_str == '[]':
                return []

            try:
                if isinstance(bbox_str, str):
                    bbox = ast.literal_eval(bbox_str)
                else:
                    bbox = bbox_str

                if isinstance(bbox, list) and len(bbox) == 4:
                    return [float(x) for x in bbox]
                else:
                    return []
            except Exception as e:
                print(f"[WARNING] bbox 파싱 실패: {bbox_str}, 오류: {e}")
                return []

        def create_image_with_bboxes(self, image_path: str, issues: List[dict], temp_dir='./temp') -> str:
            """이미지에 모든 이슈의 바운딩 박스를 그려서 임시 파일로 저장"""

            if not os.path.exists(image_path):
                print(f"[ERROR] 이미지 파일 없음: {image_path}")
                return image_path

            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] 이미지 로드 실패: {image_path}")
                return image_path

            # BGR to RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # 이미지 크기에 맞춰 figure 크기 조정
            dpi = 100
            fig_w = w / dpi
            fig_h = h / dpi

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.imshow(image_rgb)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # 각 이슈에 대해 바운딩 박스 그리기
            for idx, issue in enumerate(issues):
                issue_type = issue.get('issue_type', 'default')
                bbox = issue.get('bbox', '')
                color = self.issue_colors_mpl.get(issue_type, self.issue_colors_mpl['default'])

                if not bbox:
                    # 바운딩 박스가 없으면 전체 이미지 크기로 설정
                    rect = Rectangle((2, 2), w - 4, h - 4,
                                     linewidth=3, edgecolor=color,
                                     facecolor='none', alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(10, 10 + idx * 30, f'BOX {idx + 1:02d}: {issue_type.upper()}',
                            ha='left', va='top',
                            fontsize=12, fontweight='bold', color='white',
                            bbox=dict(boxstyle="square,pad=0.3", facecolor=color, alpha=0.9))
                else:
                    # 정규화된 좌표인지 확인
                    if all(0 <= coord <= 1 for coord in bbox):
                        x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
                    else:
                        x1, y1, x2, y2 = bbox

                    # 좌표 유효성 검사
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))

                    if x2 > x1 and y2 > y1:
                        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=3, edgecolor=color,
                                         facecolor='none', alpha=0.8)
                        ax.add_patch(rect)

                        label = f"BOX {idx + 1:02d}: {issue_type.upper()}"
                        ax.text(x1, y1, label,
                                ha='left', va='bottom',
                                fontsize=10, fontweight='bold', color='white',
                                bbox=dict(boxstyle="square,pad=0.1", facecolor=color, alpha=0.9))

            # 임시 파일 저장
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, os.path.basename(image_path))
            plt.savefig(temp_path, dpi=dpi, bbox_inches='tight',
                        pad_inches=0, facecolor='white', edgecolor='none')
            plt.close()
            return temp_path

    def _has_cutoff_components(self, image_path: str) -> bool:
        """
        이미지에 해당하는 JSON 파일에서 RadioButton, ToggleButton, Switch 컴포넌트가 있는지 확인
        """
        try:
            # 이미지 파일명에서 JSON 파일 경로 생성
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = f"{self.config.JSON_DIR}/{base_name}.json"

            # JSON 파일이 없는 경우 기본 경로 시도
            if not os.path.exists(json_path):
                json_path = image_path.replace('.png', '.json')

            if not os.path.exists(json_path):
                print(f"JSON 파일을 찾을 수 없습니다: {json_path}")
                return False

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # cutoff 검사 대상 컴포넌트들 (set으로 더 빠른 검색)
            cutoff_components = {'RadioButton', 'ToggleButton', 'Switch'}

            # 1. 특정 JSON 구조에서 빠른 검색: obj['skelton']['elements'][i]['id']
            try:
                if 'skelton' in data and 'elements' in data['skelton']:
                    elements = data['skelton']['elements']
                    if isinstance(elements, list):
                        # 빠른 검색: any()를 사용하여 첫 번째 일치 시 즉시 반환
                        cutoff_found = any(
                            isinstance(element, dict) and
                            'id' in element and
                            element['id'] in cutoff_components
                            for element in elements
                        )

                        if cutoff_found:
                            # 발견된 컴포넌트 정보 출력
                            for i, element in enumerate(elements):
                                if isinstance(element, dict) and 'id' in element and element['id'] in cutoff_components:
                                    print(f"Cutoff 대상 컴포넌트 발견 (skelton.elements[{i}].id): {element['id']}")
                                    break
                            return True
            except (KeyError, TypeError, IndexError) as e:
                print(f"skelton.elements 구조 접근 오류: {e}")

            # 2. 백업: 일반적인 재귀 검색 (기존 로직 유지)
            def search_components(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key in ['component_id', 'class_name', 'type', 'id'] and isinstance(value, str):
                            # 정확한 매칭과 부분 매칭 모두 지원
                            if value in cutoff_components:
                                print(f"Cutoff 대상 컴포넌트 발견 (정확 매칭, {key}): {value}")
                                return True
                            # 부분 매칭 (소문자 변환)
                            value_lower = value.lower()
                            cutoff_lower = {comp.lower() for comp in cutoff_components}
                            if any(component in value_lower for component in cutoff_lower):
                                print(f"Cutoff 대상 컴포넌트 발견 (부분 매칭, {key}): {value}")
                                return True
                        elif isinstance(value, (dict, list)):
                            if search_components(value):
                                return True
                elif isinstance(obj, list):
                    for item in obj:
                        if search_components(item):
                            return True
                return False

            return search_components(data)

        except Exception as e:
            print(f"JSON 파일 읽기 오류: {e}")
            return False

    def inspect_ui_quality(self, image_path: str) -> Optional[str]:
        """Inspect UI quality for a single image with issues"""
        try:
            # Call Gemini API with the visibility prompt
            prompt = self._get_visibility_inspection_prompt()
            result = self.gemini._call_gemini_issue_inspection(prompt=prompt, image=image_path)
            return result

        except Exception as e:
            print(f"Error Inspect UI Quality: {e}")
            return None

    def inspect_cutoff_quality(self, image_path: str) -> Optional[str]:
        """Inspect cutoff quality for RadioButton/ToggleButton components"""
        try:
            # Call Gemini API with the cutoff prompt
            prompt = self._get_cutoff_inspection_prompt()
            result = self.gemini._call_gemini_issue_inspection(prompt=prompt, image=image_path)
            return result

        except Exception as e:
            print(f"Error Inspect Cutoff Quality: {e}")
            return None

    def preproc_dataset(self) -> pd.DataFrame:
        # # 중간 산출물-> 2차 검수(Normal)
        if not os.path.exists(self.config.DATASET_PATH):
            raise FileNotFoundError(f"Excel file not found: {self.config.DATASET_PATH}")

        df = pd.read_excel(self.config.DATASET_PATH)
        df = df[~df['issue_type'].isin(['no_xml', 'not_processed'])]

        df['filename'] = df.apply(
            lambda row: str(row['label']) + os.path.basename(row['filename'])
            if pd.notna(row['label']) and row['label'] != ''
            else os.path.basename(row['filename']),
            axis=1
        )

        # Normalize issue types
        df['issue_type'] = df['issue_type'].apply(
            lambda x: 'normal' if str(x).startswith('normal_') else str(x)
        )

        return df

    def get_processed_files(self) -> set:
        """Get list of already processed files"""
        if os.path.exists(self.config.OUTPUT_REPORT_PATH):
            report = pd.read_csv(self.config.OUTPUT_REPORT_PATH)
            return set(report['filename'].unique())
        return set()

    def process(self) -> None:
        """Main processing function for all files"""
        try:
            df = self.preproc_dataset()
            processed_files = self.get_processed_files()

            # Filter out already processed files
            df = df[~df['filename'].isin(processed_files)]

            issue_report = []
            for filename, group in tqdm(df.groupby('filename')):
                try:
                    if filename in processed_files:
                        print(f"해당 파일은 이미 처리된 파일 입니다.: {filename}")
                        continue
                    image_path = os.path.join(self.config.VISUALIZATION_DIR, filename)

                    if not os.path.isfile(image_path):
                        print(f"해당 이미지 파일이 존재하지 않습니다.: {image_path}")
                        continue

                    # 기본 visibility 검사
                    inspection_result = self.inspect_ui_quality(image_path)

                    # RadioButton, ToggleButton, Switch가 있는 경우 cutoff 검사 추가
                    cutoff_result = None
                    if self._has_cutoff_components(image_path):
                        print(f"Cutoff 대상 컴포넌트 발견, cutoff 검사 수행: {filename}")
                        cutoff_result = self.inspect_cutoff_quality(image_path)

                    # 결과 저장
                    if inspection_result:
                        result_data = [filename, inspection_result]
                        if cutoff_result:
                            result_data.append([filename, cutoff_result])

                        pd.DataFrame([result_data]).to_csv(
                            self.config.OUTPUT_RAW_REPORT_PATH,
                            mode='a',
                            header=not os.path.exists(self.config.OUTPUT_RAW_REPORT_PATH),  # 첫 번째만 헤더 포함
                            index=False
                        )

                        print("=" * 30)
                        print(f"File: {filename}")
                        print("Visibility Inspection:")
                        print(json.dumps(inspection_result, ensure_ascii=False, indent=2))

                        if cutoff_result:
                            print("Cutoff Inspection:")
                            print(json.dumps(cutoff_result, ensure_ascii=False, indent=2))

                except Exception as e:
                    error_msg = f'{filename}: {e}'
                    issue_report.append(error_msg)
                    print(f"Error processing {filename}: {e}")

            # Save error report if any issues occurred
            if issue_report:
                error_df = pd.DataFrame(issue_report, columns=['error'])
                error_df.to_excel(self.config.ISSUE_REPORT_PATH, index=False)

        except Exception as e:
            print(f"Error in process_files: {e}")
            raise

    def total_process(self):

        image_list = [os.path.basename(path) for path in glob.glob(f'{self.config.VISUALIZATION_DIR}/*.png')]

        if os.path.isfile(self.config.OUTPUT_RAW_REPORT_PATH):
            self._convert_report_format()
            processed = self.get_processed_files()
            df = pd.DataFrame({'filename': image_list})
            image_list = df[~df['filename'].isin(processed)]['filename'].tolist()

        issue_report = []
        try:
            for filename in tqdm(image_list):
                image_path = f'{self.config.VISUALIZATION_DIR}/{filename}'
                if not os.path.isfile(image_path):
                    print(f"해당 이미지 파일이 존재하지 않습니다.: {image_path}")
                    continue

                # 기본 visibility 검사
                inspection_result = self.inspect_ui_quality(image_path)

                # RadioButton, ToggleButton, Switch가 있는 경우 cutoff 검사 추가
                cutoff_result = None
                if self._has_cutoff_components(image_path):
                    print(f"Cutoff 대상 컴포넌트 발견, cutoff 검사 수행: {filename}")
                    cutoff_result = self.inspect_cutoff_quality(image_path)

                # 결과 저장
                if inspection_result:
                    result_data = [filename, inspection_result]

                    if cutoff_result:

                        pd.DataFrame([filename, cutoff_result]).to_csv(
                            self.config.OUTPUT_RAW_REPORT_PATH,
                            mode='a',
                            header=not os.path.exists(self.config.OUTPUT_RAW_REPORT_PATH),  # 첫 번째만 헤더 포함
                            index=False
                        )

                    pd.DataFrame(result_data).to_csv(
                        self.config.OUTPUT_RAW_REPORT_PATH,
                        mode='a',
                        header=not os.path.exists(self.config.OUTPUT_RAW_REPORT_PATH),  # 첫 번째만 헤더 포함
                        index=False
                    )

                    print("=" * 30)
                    print(f"File: {filename}")
                    print("Visibility Inspection:")
                    print(json.dumps(inspection_result, ensure_ascii=False, indent=2))

                    if cutoff_result:
                        print("Cutoff Inspection:")
                        print(json.dumps(cutoff_result, ensure_ascii=False, indent=2))

        except Exception as e:
            error_msg = f'{filename}: {e}'
            issue_report.append(error_msg)
            print(f"Error processing {filename}: {e}")

        # Save error report if any issues occurred
        if issue_report:
            error_df = pd.DataFrame(issue_report, columns=['error'])
            error_df.to_excel(self.config.ISSUE_REPORT_PATH, index=False)

    def _convert_report_format(self):

        with open(self.config.OUTPUT_RAW_REPORT_PATH, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        data = []
        current_filename = ""
        current_description = ""

        for line in lines:
            if line.endswith('.png'):
                if current_filename:
                    data.append({'filename': current_filename, 'description': current_description})

                current_filename = line
                current_description = ""
            else:  # description
                if current_description:
                    current_description += " " + line
                else:
                    current_description = line

        if current_filename:
            data.append({'filename': current_filename, 'description': current_description})

        df = pd.DataFrame(data)
        df.to_csv(self.config.OUTPUT_REPORT_PATH, index=True)

    def _generate_chunk_data(self, batch):
        csv_data = ''.join(batch)
        prompt = self._get_csv_parsing_prompt(csv_data)
        output = ""
        response = self.gemini._call_gemini_issue_parsing(prompt=prompt, model='gemini-2.0-flash-001')
        for chunk in response:
            if chunk.text is not None:
                output += chunk.text
        return output

    def get_processed_parsed_files(self) -> set:
        """Get list of already parsed files"""
        print(f"Checking parsed file: {self.config.OUTPUT_PARSED_PATH}")

        if os.path.exists(self.config.OUTPUT_PARSED_PATH):
            try:
                df = pd.read_csv(self.config.OUTPUT_PARSED_PATH,
                                 on_bad_lines='skip',
                                 header=None)
                first_column = df.iloc[:, 0]
                png_mask = first_column.str.endswith('.png', na=False)
                png_values = first_column[png_mask]
                unique_png = png_values.unique()
                return set(unique_png)
            except Exception as e:
                print(f"Error reading parsed file: {e}")
                return set()
        else:
            print("Parsed file does not exist")
            return set()

    def parse_reports_with_gemini(self) -> None:
        if not os.path.exists(self.config.OUTPUT_REPORT_PATH):
            print(f"Report file not found: {self.config.OUTPUT_REPORT_PATH}")
            return

        try:
            df = pd.read_csv(self.config.OUTPUT_REPORT_PATH, header=None)

            if df.shape[1] > 0:
                filenames = df.iloc[:, 1].tolist()
                descriptions = df.iloc[:, 2].tolist() if df.shape[1] > 1 else [''] * len(filenames)
            else:
                print("CSV 파일이 비어있습니다.")
                return

            processed_parsed_files = self.get_processed_parsed_files()
            print(f"이미 처리된 파일 개수: {len(processed_parsed_files)}개")

            # 처리할 파일들 필터링
            unprocessed_groups = []
            processed_count = 0
            skipped_count = 0

            for i, filename in enumerate(filenames):
                if isinstance(filename, str) and filename.strip().endswith('.png'):
                    filename = filename.strip()
                    description = descriptions[i] if i < len(descriptions) else ""

                    if filename not in processed_parsed_files:
                        unprocessed_groups.append(filename + '\n')
                        if description and str(description).strip():
                            unprocessed_groups.append(str(description).strip() + '\n')
                        processed_count += 1
                    else:
                        skipped_count += 1

            print(f"\n처리 결과:")
            print(f"   처리할 파일: {processed_count}개")
            print(f"   건너뛴 파일: {skipped_count}개")
            print(f"   unprocessed_groups 길이: {len(unprocessed_groups)}")

            if not unprocessed_groups:
                print("\n이미 모든 파일이 처리되었습니다.")

            print(f"Files to process: {len(unprocessed_groups) // 2}")

            total_batches = (len(unprocessed_groups) + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
            file_mode = "a" if os.path.exists(self.config.OUTPUT_PARSED_PATH) else "w"

            with open(self.config.OUTPUT_PARSED_PATH, file_mode, encoding="utf-8") as out_f:
                with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
                    for batch_start in range(0, len(unprocessed_groups), self.config.BATCH_SIZE):
                        batch = unprocessed_groups[batch_start:batch_start + self.config.BATCH_SIZE]
                        batch_end = min(batch_start + self.config.BATCH_SIZE, len(unprocessed_groups))

                        success = False
                        for attempt in range(self.config.MAX_RETRIES):
                            try:
                                result = self._generate_chunk_data(batch)
                                out_f.write(result)
                                out_f.write("\n")
                                out_f.flush()  # Ensure immediate write
                                success = True

                                current_batch = batch_start // self.config.BATCH_SIZE + 1
                                pbar.set_description(f"Processing batch {current_batch}/{total_batches}")

                                # Display result
                                print("=" * 50)
                                print(f"Batch {current_batch}/{total_batches} - Lines {batch_start + 1}-{batch_end}")
                                print("Result:")
                                try:
                                    result_json = json.loads(result.strip())
                                    print(json.dumps(result_json, ensure_ascii=False, indent=2))
                                except (json.JSONDecodeError, ValueError):
                                    print(result.strip())
                                print("=" * 50)
                                break

                            except Exception as e:
                                print(f"Batch {current_batch} - Attempt {attempt + 1} failed: {e}")

                        if success:
                            pbar.update(1)
                        else:
                            print(f"Failed to process batch starting at {batch_start}")
                            self._save_error_log(unprocessed_groups)
                            pbar.update(1)

        except Exception as e:
            print(f"Error in parse_reports_with_gemini: {e}")
            self._save_error_log([])

    def _save_error_log(self, lines: List[str]) -> None:
        """Save error log to file"""
        try:
            with open(self.config.ERROR_LOG_PATH, "w", encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Failed to save error log: {e}")

    def fix_csv_columns(self):
        """
        CSV 파일을 4열 구조로 단순하게 변환
        - 첫 번째 열: 이미지명 (filename)
        - 두 번째 열: UI component
        - 세 번째 열: description (나머지 모든 중간 열들을 합침)
        - 네 번째 열: 스코어 (숫자인 열을 찾아서)

        패턴 매칭 없이 최대한 단순하게 처리
        """

        fixed_rows = []
        skipped_rows = 0

        # CSV 파일 읽기
        with open(self.config.OUTPUT_PARSED_PATH, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)

            for i, row in enumerate(csv_reader):
                # 완전히 빈 행 건너뛰기
                if not row or all(cell.strip() == '' for cell in row):
                    print(f"빈 행 건너뜀: {i + 1}행")
                    skipped_rows += 1
                    continue

                # 빈 셀들을 빈 문자열로 정규화
                normalized_row = [cell.strip() if cell else '' for cell in row]

                print(f"행 {i + 1}: {normalized_row}")

                # 기본값 설정
                filename = ""
                ui_component = ""
                score = ""

                # 1. 첫 번째 열: 항상 이미지명
                if len(normalized_row) >= 1:
                    filename = normalized_row[0]

                # 2. 스코어 찾기 (숫자인 열)
                score_index = None
                for idx, cell in enumerate(normalized_row):
                    if self._is_score(cell):
                        score_index = idx
                        score = self._extract_score_value(cell)  # / 앞의 숫자만 추출
                        break

                # 3. 두 번째 열: UI component (스코어가 아닌 경우)
                if len(normalized_row) >= 2 and score_index != 1:
                    ui_component = normalized_row[1]

                # 4. 나머지 모든 열: description으로 합치기
                description_parts = []
                for idx, cell in enumerate(normalized_row[1:], 1):  # 첫 번째 열(이미지명) 제외
                    if idx != score_index and cell.strip():  # 스코어 열이 아니고 빈 값이 아닌 경우
                        if idx == 1:
                            # 두 번째 열이 UI component로 사용된 경우는 제외
                            if score_index == 1 or not ui_component:
                                description_parts.append(cell.strip())
                        else:
                            description_parts.append(cell.strip())

                # description 합치기
                combined_description = ' '.join(description_parts)

                # UI component와 description이 합쳐진 경우 ... 기준으로 분리
                ui_component, description = self._split_ui_and_description(ui_component, combined_description)

                # 결과 출력 (디버깅용)
                print(f"  → filename: {filename}")
                print(f"  → ui_component: {ui_component}")
                print(f"  → description: {description}")
                print(f"  → score: {score}")
                print()

                # 결과 추가
                fixed_rows.append([filename, ui_component, description, score])

        # DataFrame 생성 및 저장
        dataframe = pd.DataFrame(fixed_rows, columns=['filename', 'ui_component', 'description', 'score'])

        # 빈 이미지명 행 제거
        dataframe = dataframe[dataframe['filename'].str.strip() != '']

        # CSV 저장 (인덱스 제외)
        dataframe.to_csv(self.config.OUTPUT_RESULT_PATH, index=False, encoding='utf-8')

        print(f"처리 완료: {len(fixed_rows)}개 행 처리, {skipped_rows}개 행 건너뜀")
        print(f"최종 저장: {len(dataframe)}개 행")
        print(f"save final report: {self.config.OUTPUT_RESULT_PATH}")

        return dataframe

    def _split_ui_and_description(self, ui_component, combined_description):
        """
        UI component와 description을 ... 기준으로 분리

        예시:
        - "input(EditText_13): Category n..." → UI: "input(EditText_13): Category n", DESC: ""
        - "" + "input(EditText_13): Category n...'Category name' 텍스트..."
          → UI: "input(EditText_13): Category n", DESC: "'Category name' 텍스트..."
        """
        # UI component가 비어있고 combined_description에 ... 가 있는 경우
        if not ui_component.strip() and combined_description and '...' in combined_description:
            parts = combined_description.split('...', 1)
            if len(parts) == 2:
                ui_part = parts[0].strip() + '...'  # ... 포함해서 UI component로
                desc_part = parts[1].strip()
                return ui_part, desc_part

        # UI component에 ... 가 있는 경우
        if ui_component and '...' in ui_component:
            parts = ui_component.split('...', 1)
            if len(parts) == 2:
                ui_part = parts[0].strip() + '...'  # ... 포함해서 UI component로
                desc_part = parts[1].strip()
                # combined_description이 있으면 합치기
                if combined_description:
                    desc_part = (desc_part + ' ' + combined_description).strip()
                return ui_part, desc_part

        # combined_description에만 ... 가 있는 경우
        if combined_description and '...' in combined_description:
            parts = combined_description.split('...', 1)
            if len(parts) == 2:
                # 기존 ui_component가 있으면 유지, 없으면 앞부분을 사용
                if ui_component.strip():
                    desc_part = parts[1].strip()
                    return ui_component, desc_part
                else:
                    ui_part = parts[0].strip() + '...'
                    desc_part = parts[1].strip()
                    return ui_part, desc_part

        # ... 가 없거나 분리할 수 없는 경우 원본 반환
        return ui_component, combined_description

    def _extract_score_value(self, cell):
        """
        셀에서 실제 스코어 값만 추출
        1/9 -> "1", 3.5 -> "3", -1 -> "0", 4 -> "4"
        소수점은 정수 부분만, 음수는 0으로 처리
        """
        cell = cell.strip()

        # 1. / 가 있는 경우 앞의 숫자만 추출
        if '/' in cell:
            parts = cell.split('/')
            if len(parts) >= 2:
                front_part = parts[0].strip()
                try:
                    num = float(front_part)
                    # 음수는 0으로, 소수점은 정수 부분만
                    result = max(0, int(num))
                    return str(result)
                except ValueError:
                    return "0"

        # 2. 순수한 정수인 경우
        if cell.isdigit():
            return cell

        # 3. 음수 또는 소수점 처리
        try:
            num = float(cell)
            # 음수는 0으로, 소수점은 정수 부분만
            result = max(0, int(num))
            return str(result)
        except ValueError:
            return "0"

    def _is_score(self, cell):
        """
        셀이 스코어(숫자)인지 간단하게 확인
        1/9, 2/10 같은 형태도 처리 (/ 앞의 숫자만 사용)
        스코어 범위: 0-10
        소수점은 정수 부분만 사용 (3.5 → 3)
        음수는 0으로 처리 (-1 → 0)
        """
        cell = cell.strip()

        # 1. / 가 있는 경우 앞의 숫자만 확인
        if '/' in cell:
            parts = cell.split('/')
            if len(parts) >= 2:
                front_part = parts[0].strip()
                try:
                    num = float(front_part)
                    # 음수는 0으로, 소수점은 정수 부분만
                    num = max(0, int(num))
                    return 0 <= num <= 10
                except ValueError:
                    return False
            return False  # / 가 있지만 올바른 형태가 아닌 경우

        # 2. 순수한 숫자인지 확인
        if cell.isdigit():
            num = int(cell)
            return 0 <= num <= 10  # 0-10 범위의 점수

        # 3. 음수 또는 소수점 숫자인지 확인
        try:
            num = float(cell)
            # 음수는 0으로, 소수점은 정수 부분만
            num = max(0, int(num))
            return 0 <= num <= 10
        except ValueError:
            return False


if __name__ == "__main__":

    try:
        config = Config()
        inspector = UIQualityInspector(config)

        # Process all files
        print("Starting file processing...")
        inspector.total_process()

        # Parse reports with Gemini
        print("Starting report parsing...")
        inspector.parse_reports_with_gemini()

        print("Final report processing...")
        inspector.fix_csv_columns()

        print("Processing completed successfully!")

    except Exception as e:
        print(f"Fatal error in main: {e}")
        raise