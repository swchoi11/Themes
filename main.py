from src.Mask import Mask
from src.Classification import Classification
from src.Layout import json_parser, Compare
# 1. raw 이미지에 대한 클러스터링

# 2. 클러스터링 결과에 대한 컴포넌트 추출

# 3. 추출된 컴포넌트를 바탕으로 베이스 템플릿 추출

## output 구조
'''
 .
 ├── resource
 │   ├── image_1.png
 │   ├── image_2.png
 │   ├── image_3.png
 │   └── ...
 └── output
     ├── main_frames
     │   ├── cluster01_image_1.png
     │   ├── cluster02_image_3.png
     │   └── ...
     ├── visualization
     │   ├── cluster01
     │   │   ├── component_image_1.png
     │   │   ├── component_image_2.png
     │   │   └── ...
     │   ├── cluster02
     │   │   ├── component_image_3.png
     │   │   ├── component_image_4.png
     │   │   └── ...
     │   └── ...
     ├── json
     │   ├── cluster01
     │   │   ├── component_image_1.json
     │   │   ├── component_image_2.json
     │   │   └── ...
     │   ├── cluster02
     │   │   ├── component_image_3.json
     │   │   ├── component_image_4.json
     │   │   └── ...
     └── debug
'''
if __name__ == "__main__":
# def run(target_image_dir: str):
    # target_image_path = './resource/test.png'
    # mask = Mask()
    # classify = Classification(is_gray=True)

    # # 새로운 이미지에 대한 전처리
    # ## 이지 파서

    ## 타겟 이미지 경로 하위의 이미지들이 전부 흑백 또는 컬러 이미지로 마스킹되어 덮어씌워집니다.
    # mask.mask_directory_images(target_image_dir, is_gray=False)

    # 레이아웃 분류
    ## 마스킹된 타겟 이미지와 메인 프레임 경로 하위의 베이스 템플릿을 비교하여 클러스터 아이디를 추출합니다. 
    # cluster, score = classify.get_cluster(target_image_path, method='orb')
    # print(f"클러스터: {cluster}")
    # print(f"점수: {score}")

    # 레이아웃과 이미지를 비교해 cut off, visibility 이슈 확인
    ## 타겟 이미지와 베이스 템플릿을 비교 
    # map components , calculate iou
    theme_layout = json_parser(json_path='./resource/test.json')
    default_layout = json_parser(json_path='./resource/test.json')
    # print(theme_layout.skeleton.elements)
    
    theme_elements = theme_layout.skeleton.elements
    default_elements = default_layout.skeleton.elements

    compare = Compare()
    result = compare._map_components(default_elements, theme_elements)
    for box in result.values():
        iou = compare._calculate_iou(box['default'].bbox, box['themed'].bbox)
        print(f"iou: {iou}")

    ## 타겟 이미지와 디폴트 이미지를 비교
    # detect_cut_off_issue, calculate_contrast
    # default_image_path, default_score = classify.get_default(target_image_path, cluster, method='orb')
    # print(f"디폴트 이미지: {default_image_path}")
    # print(f"디폴트 점수: {default_score}")


    # Gemini를 통한 visibility 및 디자인 이슈 확인
    ## 타겟 이미지와 이슈 목록을 제공하여 이슈를 확인합니다.


    # 결과 산출



# import os
# import cv2
# import glob
# import json
# from tqdm import tqdm
# from PIL import Image

# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# from sklearn.cluster import KMeans
# import pandas as pd
# import numpy as np

# from matplotlib import pyplot as plt
# from typing import List, Tuple

# from preproc.dca import DCAClustering
# from preproc.cluster_organizer import ClusterOrganizer
# from src.easyParser import EasyParserRunner
# from src.BaseTemplate import BaseTemplateGenerator


# if __name__ == "__main__":

#     # 설정
#     BASE_DIR = os.path.abspath(os.path.dirname(__file__))
#     print(f"[INFO] BASE_DIR: {BASE_DIR}")

#     raw_image_dir = os.path.join(BASE_DIR, "resource", "raw")
#     cluster_base_dir = os.path.join(BASE_DIR, "resource", "cluster_data")
#     dca_csv_path = os.path.join(BASE_DIR, "output", "dca_output", "dca_image_clusters.csv")
#     json_output_dir = os.path.join(BASE_DIR, "output", "json")
#     visual_output_dir = os.path.join(BASE_DIR, "output", "visualization")
#     template_output_dir = os.path.join(BASE_DIR, "output", "main_frames")
#     mask_ouput_dir = os.path.join(BASE_DIR, "output", "mask")
#     heatmap_output_dir = os.path.join(BASE_DIR, "output", "heatmap")

#     os.makedirs(json_output_dir, exist_ok=True)
#     os.makedirs(visual_output_dir, exist_ok=True)
#     os.makedirs(os.path.dirname(dca_csv_path), exist_ok=True)
#     os.makedirs(template_output_dir, exist_ok=True)
#     os.makedirs(mask_ouput_dir, exist_ok=True)
#     os.makedirs(heatmap_output_dir, exist_ok=True)

#     # parameter
#     num_cluster = 15
#     min_width = 1800
#     min_box_size = 10
#     iou_threshold = 0.5

#     # 인스턴스 생성
#     dca = DCAClustering(raw_image_dir, dca_csv_path, num_cluster, min_width, 
#                       image_extensions=["*.png", "*.jpg", "*.jpeg"])
#     organizer = ClusterOrganizer(dca_csv_path, raw_image_dir, cluster_base_dir)
#     parser = EasyParserRunner(BASE_DIR, cluster_base_dir, json_output_dir, visual_output_dir, num_cluster)
#     template = BaseTemplateGenerator(
#         min_box_size=min_box_size, 
#         iou_threshold=iou_threshold, 
#         cluster_dir = cluster_base_dir,
#         output_dir = template_output_dir,
#         condition_output_dir = heatmap_output_dir,
#         image_extensions=[".png", ".jpg", ".jpeg"],
#         cluster_prefix = "cluster")

#     # 실행
#     dca.run()
#     organizer.run()
#     parser.run()
#     template.run()

#     print("\n 전체 처리 완료")


