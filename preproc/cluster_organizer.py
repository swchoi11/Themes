# 1. 클러스터별 열(Column) 형식으로 변환
    ## 동일한 cluster_label을 하나의 열로 정리
    ## 가장 긴 클러스터에 맞춰 None 값으로 padding
    ## 변환된 결과를 같은 경로 혹은 지정된 output_csv 경로에 저장
# 2. 클러스터 디렉토리 생성 및 이미지 분류
    ## 열마다 cluster00, cluster01, ... 식으로 폴더 생성
    ## 해당 클러스터에 속한 이미지들을 지정된 이름으로 복사 (e.g., cluster00_filename.png)

import os
import pandas as pd
import shutil


class ClusterOrganizer:
    def __init__(self, input_csv: str, image_dir: str, output_root: str):
        """
        input_csv: image_name, cluster_label 형식의 csv 파일 경로
        image_dir: 원본 이미지 디렉토리
        output_root: 클러스터 별 이미지 저장될 디렉토리 
        """
        self.input_csv = input_csv
        self.image_dir = image_dir
        self.output_root = output_root

    def convert_cluster_csv_to_column_format(self, output_csv: str = None):
        """
        cluster_label을 열로 피벗한 csv로 변환
        """
        df = pd.read_csv(self.input_csv)
        grouped = df.groupby("cluster_label")["image_name"].apply(list)
        max_len = grouped.apply(len).max()

        df_wide = pd.DataFrame({
            str(label): imgs + [None] * (max_len - len(imgs))
            for label, imgs in grouped.items()
        })

        # input_csv 에 output_csv 덮어쓰기
        if output_csv is None:
            output_csv = self.input_csv
        
        df_wide.to_csv(output_csv, index=False)
        print(f"[INFO] 변환 완료 및 저장: {output_csv}")

    def organize_clusters_by_csv(self, csv_path: str = None):
        """
        클러스터 라벨 기반으로 이미지를 클러스터 폴더별로 정렬
        """
        
        if csv_path is None:
            csv_path = self.input_csv

        df = pd.read_csv(csv_path)

        # 열 -> 하나의 클러스터
        for cluster_name in df.columns:
            cluster_folder = os.path.join(self.output_root, f"cluster{int(cluster_name):02d}")
            os.makedirs(cluster_folder, exist_ok=True)

            for file_name in df[cluster_name].dropna():
                org_name = str(file_name).strip()
                src_path = os.path.join(self.image_dir, org_name)
                new_name = f"cluster{int(cluster_name):02d}_{os.path.basename(org_name)}"
                dst_path = os.path.join(cluster_folder, new_name)

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"[WARN] 파일 없음: {src_path}")

        print(f"[INFO] 클러스터 폴더 정리 완료: {self.output_root}")

    def run(self):
        """
        전체 실행
        """
        updated_csv = self.convert_cluster_csv_to_column_format()
        self.organize_clusters_by_csv(csv_path=updated_csv)