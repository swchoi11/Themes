"""
csv 내 클러스터 별 파일명을 기준으로 폴더 구분
"""
import os
import pandas as pd
import shutil

CSV_PATH = "cluster-result.csv"
IMAGE_DIR = "./cluster-data"
OUTPUT_ROOT = "./sorted_clusters-3"

df = pd.read_csv(CSV_PATH)

# 열 -> 하나의 클러스터터
for cluster_name in df.columns:
    cluster_folder = os.path.join(OUTPUT_ROOT, cluster_name)
    os.makedirs(cluster_folder, exist_ok=True)

    for file_name in df[cluster_name].dropna():
        src_path = os.path.join(IMAGE_DIR, str(file_name).strip())
        dst_path = os.path.join(cluster_folder, os.path.basename(str(file_name).strip()))

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"파일 없음: {src_path}")
