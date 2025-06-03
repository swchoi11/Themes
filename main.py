import os
import cv2
import glob
import json
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from typing import List, Tuple

from preproc.clustering import KMeansClusterer
from preproc.cluster_organizer import ClusterOrganizer
from src.easyParser import EasyParserRunner
from src.baseTemplate import BaseTemplateGenerator
from src.Mask import Mask
from src.Classification import Classification
# from src.layoutAnalyzer import run
# from src.gemini import Gemini

import time
start = time.time()

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
    # 설정
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    print(f"[INFO] BASE_DIR: {BASE_DIR}")

    raw_image_dir = os.path.join(BASE_DIR, "resource", "dataset")
    # raw_image_dir = os.path.join(BASE_DIR, "dataset", "normal")
    cluster_base_dir = os.path.join(BASE_DIR, "resource", "cluster_data")
    cluster_csv_path = os.path.join(BASE_DIR, "output", "clustering_output", "image_clusters.csv")
    json_output_dir = os.path.join(BASE_DIR, "output", "json")
    visual_output_dir = os.path.join(BASE_DIR, "output", "visualization")
    template_output_dir = os.path.join(BASE_DIR, "output", "main_frames")
    # mask_ouput_dir = os.path.join(BASE_DIR, "output", "mask")
    # heatmap_output_dir = os.path.join(BASE_DIR, "output", "heatmap")

    target_image_dir = os.path.join(BASE_DIR, "resource", "target_data")
    target_json_dir = os.path.join(BASE_DIR, "output", "loop_json")
    target_visual_dir = os.path.join(BASE_DIR, "output", "loop_visualization")

    issue_output_dir = os.path.join(BASE_DIR, "output", "issue")
    gemini_image_dir = os.path.join(BASE_DIR, "output", "gemini_image")
    gemini_json_dir = os.path.join(BASE_DIR, "output", "gemini_json")

    result_path = os.path.join(BASE_DIR, "output", "result.json")

    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(visual_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cluster_csv_path), exist_ok=True)
    os.makedirs(template_output_dir, exist_ok=True)
    # os.makedirs(mask_ouput_dir, exist_ok=True)
    # os.makedirs(heatmap_output_dir, exist_ok=True)

    os.makedirs(target_json_dir, exist_ok=True)
    os.makedirs(target_visual_dir, exist_ok=True)
    os.makedirs(issue_output_dir, exist_ok=True)
    os.makedirs(gemini_image_dir, exist_ok=True)
    os.makedirs(gemini_json_dir, exist_ok=True)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # parameter
    num_cluster = 15
    min_width = 1800
    min_box_size = 10
    iou_threshold = 0.5
    size_filter_mode = "f"
    size_threshold = 1800
    pca_components = 50
    # min_cluster_size = 10
    # n_components = 20
    # n_neighbors = 15
    # min_dist = 0.1
    # min_samples = 5
    max_k = 20


    # 인스턴스 생성 및 실행

    # 사전 처리
    ## 1. 클러스터링
    """
    desc. raw 이미지를 clustering 한 후 각 cluster 폴더 내 분류
    input: resource > dataset (3000개 정상 이미지)
    output: 
        1. output > clustering_output (이미지명 + clsuter 가 적혀 있는 csv 파일)
            (e.g., > image_clusters.csv)
        2. resource > cluster_data (각 cluster 폴더와 이미지)
            (e.g., > cluster00 > cluster00_image_name)
    """
    # clustering = KMeansClusterer(raw_image_dir, cluster_csv_path, size_filter_mode, size_threshold,
    #                      pca_components, max_k)
    # organizer = ClusterOrganizer(cluster_csv_path, raw_image_dir, cluster_base_dir)

    # clustering.run()
    # organizer.run()

    print("finished clustering")

    ## 2. 컴포넌트 디텍터 (easyParser)
    """
    desc. cluster 폴더 내 분류되어 있는 이미지를 json과 bbox 가 포함된 이미지로 추출
    input: resource > cluster_data (각 cluster 폴더와 이미지)
    output: 
        1. output > json (각 이미지에 대한 영역, 상호관계 등을 담은 정보 json 파일)
            (e.g., > cluster00 > cluster00_image_name.json)
        2. output > visualization (각 이미지에 대해 스켈레톤 구조가 그려져 있는 이미지 파일)
            (e.g., > cluster00 > cluster00_image_name.png)
    """
    # parser = EasyParserRunner(BASE_DIR, cluster_base_dir, json_output_dir, visual_output_dir, num_cluster)
    # parser.run()

    # print("finished easyParser")

    ## 3. 베이스 템플릿 추출
    """
    dsec. cluster 폴더 내 이미지 기준으로 base template (heatmap 기준) 추출
    input: output > visualization (2번에 추출된 스켈레톤 구조가 그려져 있는 이미지 파일들)
            (e.g., > cluster00 > cluster00_image_name.png)
    output: output > main_frames (target data와 비교할 수 있는 cluster별 base template 이미지)
            (e.g., > cluster00_base_image.png)
    """
    template = BaseTemplateGenerator(
        min_box_size=min_box_size, 
        iou_threshold=iou_threshold, 
        cluster_dir = visual_output_dir,
        output_dir = template_output_dir,
        image_extensions=[".png", ".jpg", ".jpeg"],
        cluster_prefix = "cluster")
    template.run()

    print("finished template")



    # # 반복 처리 (새로운 이미지 = target 이미지)
    # # 1. 컴포넌트 디텍터 (easyParser)
    # """
    # desc. target 이미지에 대해 컴포넌트 추출
    # input: resource > target_data (target 이미지)
    # output: 
    #     1. output > loop_json
    #     2. output > loop_visualization
    # """
    # parser = EasyParserRunner(BASE_DIR, target_image_dir, target_json_dir, target_visual_dir, num_cluster)
    # parser.run()

    # print("finished target easyParser")

    # # 2. 마스킹
    # """
    # desc. target 이미지 마스킹
    # input: output > loop_visualization (bbox 그려져 있는 target 이미지)
    # output: output > loop_visualization (마스킹된 target 이미지)
    # """
    # mask = Mask()
    # mask.mask_directory_images(target_visual_dir, is_gray=False)

    # print("finished mask")
    
    # # 3. 클러스터 & 대표 이미지 추출
    # """                         
    # desc. 마스킹된 target 이미지와 base template 비교 후 해당 클러스터 추출 및
    #       추출된 clsuter 내 가장 유사한 default 이미지 추출
    # input: output > loop_visualization (마스킹된 target 이미지)
    # output: output > default (유사한 default 이미지)
    # """
    # classify = Classification(is_gray=True)
    # classify.run(target_visual_dir, gemini_image_dir, gemini_json_dir)

    # # 4. 이슈 파악 1
    # # 4-1. 새로운 이미지와 디폴트 이미지의 비교를 통한 이슈 파악 & 새로운 이미지 자체에 있는 이슈 파악
    # """                         
    # desc. 새로운 이미지와 default 이미지의 비교를 통한 이슈 파악 및
    #       새로운 이미지 자체에 있는 이슈 파악 (디자인 이슈 제외)
    # input: 
    #     1. resource > target_data (target 원본 이미지)
    #     2. output > loop_json (easyParser 코드의 output인 json 파일)
    # output: output > issue.json (출력된 issue가 적혀있는 json 파일)
    # """
    # '''
    # 타겟 이슈:
    #     - 요소 정렬 기준 불일치
    #     - 상호작용 가능한 요소가 시각적으로 명확하게 구분되지 않음
    #     - 텍스트가 할당된 영역을 초과하여 텍스트 잘림
    #     - 아이콘의 가장자리가 보이지 않거나 잘려보임
    #     - 달력 아이콘에서 요일 글자가 테두리를 벗어남
    #     - 앱 내 달력, 시간 아이콘이 현재 날짜 시각과 매칭되지 않음
    #     - 하이라이트된 항목, 텍스트와 배경간 대비가 낮아 가독성이 떨어짐
    # '''
    # run(
    #     theme_image_path=target_image_dir,
    #     theme_json_path=target_json_dir,
    #     default_image_path=gemini_image_dir,
    #     default_json_path=gemini_json_dir,
    #     issue_output_path=issue_output_dir
    # )

    # print("finished gemini 1")

    # # 4-2. 이슈 파악 2
    # """                         
    # desc. 새로운 이미지 자체에 있는 디자인 이슈 파악
    # input: resource > target_data (target 원본 이미지)
    # output: output > issue.json (출력된 issue가 적혀있는 json 파일)
    # """
    # '''
    # 타겟 이슈:
    #     - 다른 기능 요소에 동일한 아이콘이 적용됨
    #     - 테마가 적용되지 않은 아이콘 등이 있음
    # '''
    # gemini = Gemini()
    # design_issues = gemini.detect_all_issues(target_image_dir)

    # # 4-3. 결과 출력
    # """                         
    # desc. 새로운 이미지에서 찾은 모든 이슈에 대한 심각도 판정
    # input: 
    #     1. resource > target_data (target 원본 이미지)
    #     2. 4-2. 에서 출력된 issues
    #     3. 4-1. 에서 출력된 issue json 파일
    # output: output > issue.json (출력된 issue가 적혀있는 json 파일)
    # """
    # gemini.sort_issues(target_image_dir, design_issues, issue_output_dir, result_path)

    # # end = time.time()
    # # print(f"실행 시간: {end - start:.4f}초") 
