import os
import ast

import cv2
from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import pandas as pd


class IssueVisualizer:
    """이슈 바운딩 박스 시각화 클래스"""

    def __init__(self):
        # 이슈 타입별 색상 정의
        self.issue_colors = {
            'normal': (128, 128, 128),  # 회색
            'alignment': (255, 0, 0),  # 빨강
            'cutoff': (0, 255, 255),  # 시안
            'design': (0, 255, 0),  # 초록
            'default': (255, 255, 0)  # 노랑 (기본값: 이상치 값)
        }

        # matplotlib용 색상 (0-1 범위)
        self.issue_colors_mpl = {
            key: tuple(c / 255 for c in color)
            for key, color in self.issue_colors.items()
        }

    def parse_bbox(self, bbox_str) -> List[float]:
        """바운딩 박스 문자열을 파싱"""
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

    def visualize_single_image(self, image_path: str, bbox: List[float],
                               issue_type: str, output_path: str):
        """단일 이미지에 바운딩 박스 시각화"""

        # 이미지 로드
        if not os.path.exists(image_path):
            print(f"[SKIP] 이미지 파일 없음: {image_path}")
            return False

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 이미지 로드 실패: {image_path}")
            return False

        # BGR to RGB 변환 (matplotlib 사용을 위해)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 이미지 크기에 맞춰 figure 크기 조정
        dpi = 100
        fig_w = w / dpi
        fig_h = h / dpi

        # matplotlib 시각화 - 이미지 원본 크기 유지
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(image_rgb)
        ax.axis('off')

        # 여백 제거
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # 색상 선택
        color = self.issue_colors_mpl.get(issue_type, self.issue_colors_mpl['default'])

        # 바운딩 박스가 비어 있으면 전체 이미지 크기로 설정 (Normal)
        if not bbox:
            # 전체 이미지에 대한 바운딩 박스
            rect = Rectangle((2, 2), w - 4, h - 4,
                             linewidth=5, edgecolor=color,
                             facecolor='none')
            ax.add_patch(rect)

            # 텍스트 추가
            ax.text(10, 10, f'{issue_type.upper()}',
                    ha='left', va='top',
                    fontsize=14, fontweight='bold', color='white',
                    bbox=dict(boxstyle="square,pad=0.3", facecolor=color, alpha=0.9, edgecolor='none'))
        else:
            # 정규화된 좌표인지 확인 (값이 0-1 사이인지)
            if all(0 <= coord <= 1 for coord in bbox):
                # 정규화된 좌표를 픽셀 좌표로 변환
                x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
            else:
                # 이미 픽셀 좌표인 경우
                x1, y1, x2, y2 = bbox

            # 좌표 유효성 검사
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            # 바운딩 박스 크기 검사
            if x2 <= x1 or y2 <= y1:
                print(f"[WARNING] 유효하지 않은 바운딩 박스: {bbox}")
                return False

            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=4, edgecolor=color,
                             facecolor='none')
            ax.add_patch(rect)

            label = f"{issue_type.upper()}"

            text_x = x1
            text_y = y1

            ax.text(text_x, text_y, label,
                    ha='left', va='bottom',
                    fontsize=13, fontweight='bold', color='white',
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=color, alpha=0.9, edgecolor='none'))

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 저장 (여백 없이, 원본 이미지 크기 유지)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    pad_inches=0, facecolor='none', edgecolor='none')
        plt.close()

        return True

    def process_excel_data(self, excel_path: str, resource_dir: str, output_dir: str):
        """엑셀 데이터를 읽어서 모든 이미지 시각화"""

        # 엑셀 파일 읽기
        if not os.path.exists(excel_path):
            print(f"[ERROR] 엑셀 파일 없음: {excel_path}")
            return

        try:
            df = pd.read_excel(excel_path)
            print(f"엑셀 파일 로드 성공: {len(df)}개 행")
        except Exception as e:
            print(f"[ERROR] 엑셀 파일 읽기 실패: {e}")
            return

        # 필수 컬럼 확인
        required_columns = ['filename', 'issue_type', 'bbox']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"[ERROR] 필수 컬럼 누락: {missing_columns}")
            print(f"사용 가능한 컬럼: {list(df.columns)}")
            return

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 이슈 타입별 통계
        issue_stats = {}
        processed_count = 0
        skipped_count = 0
        error_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            try:
                filename = row['filename']
                issue_type = row['issue_type']
                bbox_str = row['bbox']

                # 이미지 경로 구성
                image_path = os.path.join(resource_dir, filename)

                # 바운딩 박스 파싱
                bbox = self.parse_bbox(bbox_str)

                # 출력 파일명 생성
                output_path = os.path.join(output_dir, filename)

                # 시각화 실행
                success = self.visualize_single_image(
                    image_path=image_path,
                    bbox=bbox,
                    issue_type=issue_type,
                    output_path=output_path,
                    filename=filename
                )

                if success:
                    processed_count += 1
                    issue_stats[issue_type] = issue_stats.get(issue_type, 0) + 1
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"[ERROR] 행 {idx} 처리 중 오류: {e}")
                print(f"       파일명: {row.get('filename', 'N/A')}")
                error_count += 1
                continue

        # 결과 요약 출력
        print(f"\n=== 처리 완료 ===")
        print(f"처리된 이미지: {processed_count}개")
        print(f"건너뛴 이미지: {skipped_count}개")
        print(f"오류 발생: {error_count}개")
        print(f"결과 저장 위치: {output_dir}")

        print(f"\n=== 이슈 타입별 통계 ===")
        for issue_type, count in sorted(issue_stats.items()):
            print(f"- {issue_type}: {count}개")


def run_image_dump(excel_filename: str, resource_dir: str, output_dir: str):
    """메인 실행 함수"""
    # 경로 존재 확인
    if not os.path.exists(excel_filename):
        print(f"[ERROR] 엑셀 파일이 존재하지 않습니다: {excel_filename}")
        return

    if not os.path.exists(resource_dir):
        print(f"[ERROR] 리소스 디렉토리가 존재하지 않습니다: {resource_dir}")
        return

    # 시각화 객체 생성
    visualizer = IssueVisualizer()

    # 색상 범례 먼저 생성
    os.makedirs(output_dir, exist_ok=True)

    # 엑셀 데이터 처리
    visualizer.process_excel_data(
        excel_path=excel_filename,
        resource_dir=resource_dir,
        output_dir=output_dir
    )

