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
        self.DATASET_PATH = '../resource/dataset.xlsx'  # 중간 산출물
        self.OUTPUT_RAW_REPORT_PATH = f'{self.SAVE_DIR}/raw.csv'
        self.OUTPUT_REPORT_PATH = f'{self.SAVE_DIR}/report.csv'
        self.OUTPUT_PARSED_PATH = f'{self.SAVE_DIR}/parse_report.csv'
        self.OUTPUT_RESULT_PATH = f'{self.SAVE_DIR}/final_report.csv'
        self.ERROR_LOG_PATH = f'{self.SAVE_DIR}/error.txt'
        self.ISSUE_REPORT_PATH = f'{self.SAVE_DIR}/issue_report.xlsx'

        # Processing settings
        self.BATCH_SIZE = 5
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

    def inspect_ui_quality(self, image_path: str) -> Optional[str]:
        """Inspect UI quality for a single image with issues"""
        try:
            # # Create image with bounding boxes
            # temp_image_path = self.visualizer.create_image_with_bboxes(
            #     image_path, issues, self.config.TEMP_DIR
            # )

            # Call Gemini API with the prompt
            prompt = self._get_visibility_inspection_prompt()
            result = self.gemini._call_gemini_issue_inspection(prompt=prompt, image=image_path)
            return result

        except Exception as e:
            print(f"Error Inspect UI Quality: {e}")
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

                    inspection_result = self.inspect_ui_quality(image_path)

                    if inspection_result:
                        # Save result to CSV
                        pd.DataFrame([filename, inspection_result]
                                     ).to_csv(
                            self.config.OUTPUT_RAW_REPORT_PATH,
                            mode='a',
                            header=not os.path.exists(self.config.OUTPUT_RAW_REPORT_PATH),  # 첫 번째만 헤더 포함
                            index=False
                        )

                        print("=" * 30)
                        print(f"File: {filename}")
                        print(json.dumps(inspection_result, ensure_ascii=False, indent=2))

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

                inspection_result = self.inspect_ui_quality(image_path)

                if inspection_result:
                    # Save result to CSV
                    pd.DataFrame([filename, inspection_result]
                                 ).to_csv(
                        self.config.OUTPUT_RAW_REPORT_PATH,
                        mode='a',
                        header=not os.path.exists(self.config.OUTPUT_RAW_REPORT_PATH),  # 첫 번째만 헤더 포함
                        index=False
                    )

                    print("=" * 30)
                    print(f"File: {filename}")
                    print(json.dumps(inspection_result, ensure_ascii=False, indent=2))

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
        CSV 파일의 중간 열들을 하나로 합치는 함수 (개선된 버전)
        - 첫 번째 열: 이미지명 (그대로 유지)
        - 마지막 열: 스코어 (그대로 유지)
        - 중간 열들: 모두 합쳐서 하나의 텍스트 열로 만들기

        빈 행, 길이가 다른 행, 부분적으로 빈 셀들을 모두 처리합니다.
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

                # 길이에 따른 처리
                if len(normalized_row) == 1:
                    # 1열만 있는 경우: 이미지명만 있고 설명과 스코어는 빈값
                    image_name = normalized_row[0]
                    description = ""
                    score = ""
                    print(f"경고: {i + 1}행 - 1열만 존재 (이미지명만): {image_name}")

                elif len(normalized_row) == 2:
                    # 2열인 경우: 첫 번째는 이미지명, 두 번째는 스코어로 처리
                    image_name = normalized_row[0]
                    description = ""
                    score = normalized_row[1]

                else:
                    # 3열 이상인 경우: 정상 처리
                    image_name = normalized_row[0]
                    score = normalized_row[-1]

                    # 중간 열들 합치기 (빈 셀들도 포함하되 연속된 공백은 제거)
                    middle_cols = normalized_row[1:-1]
                    description = ' '.join(middle_cols).strip()
                    # 연속된 공백들을 하나로 정리
                    description = ' '.join(description.split())

                # 결과 추가 (모든 경우를 포함)
                fixed_rows.append([image_name, description, score])

        dataframe = pd.DataFrame(fixed_rows, columns=['filename', 'description', 'score'])
        dataframe.to_csv(self.config.OUTPUT_RESULT_PATH)

        return print(f"save final report: {self.config.OUTPUT_RESULT_PATH}")


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
