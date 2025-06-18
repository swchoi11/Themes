import os
import glob
import pandas as pd
from pathlib import Path


FOLDER_PATH = "D:/hnryu/Themes/resource"
SPLIT_NUMBER = 30
OUTPUT_PREFIX = 'vm'
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

OUTPUT_DIR = './valid_list'
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_folder = glob.glob(os.path.join(FOLDER_PATH, "**", "*.png"), recursive=True)
image_files = [
    os.path.basename(file)
    for file in image_folder
    if Path(file).suffix.lower() in VALID_EXTENSIONS
]

if not image_files:
    print("이미지 파일을 찾을 수 없습니다.")
    exit()

print(f"총 {len(image_files)}개의 이미지 파일을 발견했습니다.")
print(f"{SPLIT_NUMBER}개로 분할하여 CSV 파일을 생성합니다.")

# 분할 계산
chunk_size = len(image_files) // SPLIT_NUMBER
remainder = len(image_files) % SPLIT_NUMBER

for i in range(SPLIT_NUMBER):
    start = i * chunk_size
    end = start + chunk_size
    chunk = image_files[start:end]

    if chunk:  # 빈 청크가 아닌 경우에만 CSV 생성
        df = pd.DataFrame(chunk, columns=["FileName"])
        filename = f'{OUTPUT_DIR}/{OUTPUT_PREFIX}{i}_image_list.csv'
        df.to_csv(filename, index=False, header=False, encoding='utf-8-sig')
        print(f'{filename} 생성 완료')

if remainder > 0:
    remaining_files = image_files[-remainder:]
    df_remainder = pd.DataFrame(remaining_files, columns=["FileName"])
    filename = f'{OUTPUT_DIR}/{OUTPUT_PREFIX}{SPLIT_NUMBER+1}_image_list.csv'
    df_remainder.to_csv(f'{filename}', index=False, header=False, encoding='utf-8-sig')
    print(f'최종 파일 +{filename} 생성 완료')

print("작업 완료!")