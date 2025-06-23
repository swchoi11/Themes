import os
import glob
import pandas as pd
import shutil
from pathlib import Path


ROOT_DIR  = '../output/merge'
os.makedirs(ROOT_DIR, exist_ok=True)

FINAL_KPI_MATRIX_PATH = 'D:/hnryu/Themes */output/report/final_kpi_matrix.csv'
FINAL_KPI_MATRIX_TOTAL_PATH = 'D:/hnryu/Themes */output/report/final_kpi_matrix_total.csv'
FAIL_IMAGE_PATH = 'D:/hnryu/Themes */output/report/result/*.png'

#1.  final_kpi_matrix.csv 파일들 병합
kpi_files = glob.glob(FINAL_KPI_MATRIX_PATH)
print(f"찾은 final_kpi_matrix.csv 파일 수: {len(kpi_files)}")

if kpi_files:
    kpi_dataframes = []
    for file in kpi_files:
        print(f"읽는 중: {file}")
        df = pd.read_csv(file)
        kpi_dataframes.append(df)

    # 병합 후 저장
    merged_kpi = pd.concat(kpi_dataframes, ignore_index=True)
    output_kpi_path = os.path.join(ROOT_DIR, 'final_kpi_matrix.csv')
    merged_kpi.to_csv(output_kpi_path, index=False)
    print(f"병합 완료: {output_kpi_path} (총 {len(merged_kpi)}행)")
else:
    print("final_kpi_matrix.csv 파일을 찾을 수 없습니다.")

#2.  final_kpi_matrix_total.csv 파일들 병합
total_files = glob.glob(FINAL_KPI_MATRIX_TOTAL_PATH)
print(f"\n찾은 final_kpi_matrix_total.csv 파일 수: {len(total_files)}")

if total_files:
    # 모든 CSV 파일 읽어서 병합
    total_dataframes = []
    for file in total_files:
        print(f"읽는 중: {file}")
        df = pd.read_csv(file)
        total_dataframes.append(df)

    # 병합 후 저장
    merged_total = pd.concat(total_dataframes, ignore_index=True)
    output_total_path = os.path.join(ROOT_DIR, 'final_kpi_matrix_total.csv')
    merged_total.to_csv(output_total_path, index=False)
    print(f"병합 완료: {output_total_path} (총 {len(merged_total)}행)")
else:
    print("final_kpi_matrix_total.csv 파일을 찾을 수 없습니다.")


#3.  result 폴더의 png 파일들 복사
image_files = glob.glob(FAIL_IMAGE_PATH)
print(f"\n찾은 PNG 파일 수: {len(image_files)}")

if image_files:
    # result 폴더 생성
    result_dir = os.path.join(ROOT_DIR, 'result')
    os.makedirs(result_dir, exist_ok=True)

    # 파일들 복사
    copied_count = 0
    for image_file in image_files:
        filename = os.path.basename(image_file)
        destination = os.path.join(result_dir, filename)

        try:
            shutil.copy2(image_file, destination)
            print(f"복사 완료: {filename}")
            copied_count += 1
        except Exception as e:
            print(f"복사 실패 {filename}: {e}")

    print(f"총 {copied_count}개 이미지 파일 복사 완료")
else:
    print("PNG 파일을 찾을 수 없습니다.")

print("\n전체 병합 및 복사 작업 완료!")