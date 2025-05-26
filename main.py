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

from preproc.dca import DCAClustering
from preproc.cluster_organizer import ClusterOrganizer
from src.easyParser import EasyParserRunner
from src.BaseTemplate import BaseTemplateGenerator
from src.Mask import Mask
from src.Classification import Classification
from src.Layout import json_parser, Compare, Single
from src.Gemini import Gemini

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

    raw_image_dir = os.path.join(BASE_DIR, "resource", "raw")
    cluster_base_dir = os.path.join(BASE_DIR, "resource", "cluster_data")
    dca_csv_path = os.path.join(BASE_DIR, "output", "dca_output", "dca_image_clusters.csv")
    json_output_dir = os.path.join(BASE_DIR, "output", "json")
    visual_output_dir = os.path.join(BASE_DIR, "output", "visualization")
    template_output_dir = os.path.join(BASE_DIR, "output", "main_frames")
    mask_ouput_dir = os.path.join(BASE_DIR, "output", "mask")
    heatmap_output_dir = os.path.join(BASE_DIR, "output", "heatmap")

    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(visual_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dca_csv_path), exist_ok=True)
    os.makedirs(template_output_dir, exist_ok=True)
    os.makedirs(mask_ouput_dir, exist_ok=True)
    os.makedirs(heatmap_output_dir, exist_ok=True)

    # parameter
    num_cluster = 15
    min_width = 1800
    min_box_size = 10
    iou_threshold = 0.5

    # 인스턴스 생성
    dca = DCAClustering(raw_image_dir, dca_csv_path, num_cluster, min_width, 
                      image_extensions=["*.png", "*.jpg", "*.jpeg"])
    organizer = ClusterOrganizer(dca_csv_path, raw_image_dir, cluster_base_dir)
    parser = EasyParserRunner(BASE_DIR, cluster_base_dir, json_output_dir, visual_output_dir, num_cluster)
    template = BaseTemplateGenerator(
        min_box_size=min_box_size, 
        iou_threshold=iou_threshold, 
        cluster_dir = cluster_base_dir,
        output_dir = template_output_dir,
        condition_output_dir = heatmap_output_dir,
        image_extensions=[".png", ".jpg", ".jpeg"],
        cluster_prefix = "cluster")
    mask = Mask()
    classify = Classification(is_gray=True)

    # 1. 클러스터링
    dca.run()
    organizer.run()

    # 2. 컴포넌트 디텍터


    # 3. 베이스 템플릿 추출
    parser.run()
    template.run()

    # 4. 새로운 이미지에 대한 클러스터링
    # 4-1. 새로운 이미지에 대한 마스킹
    ## 타겟 이미지 전체에 대한 마스킹 일괄 진행
    # target_image_dir = './resource/'
    # mask.mask_directory_images(target_image_dir, is_gray=False)
    
    ## 타겟 이미지 한개에 대한 마스킹 진행
    target_image_path = './resource/test.png'
    mask.mask_image(target_image_path, is_gray=False)

    # 4-2. 마스킹된 새로운 이미지와 베이스 템플릿 이미지 비교 -> 클러스터 선택
    cluster, score = classify.get_cluster(target_image_path, method='orb')
    print(f"클러스터: {cluster}")
    print(f"점수: {score}")

    # 4-3. 선택된 클러스터 중 가장 유사한 디폴트 이미지 선택
    default_image_path, default_score = classify.get_default(target_image_path, cluster, method='orb')
    print(f"디폴트 이미지: {default_image_path}")
    print(f"디폴트 점수: {default_score}")

    # 5. 이슈 파악
    ## 선택된 디폴트 이미지, 분석중인 새로운 이미지에 대한 컴포넌트 추출을 통한
    ## json파일이 인풋으로 필요합니다. 
    theme_layout = json_parser(json_path='./resource/test.json')
    default_layout = json_parser(json_path='./resource/test-default.json')

    # 5-1. 새로운 이미지와 디폴트 이미지의 비교를 통한 이슈 파악
    '''
    타겟 이슈:
    - 요소 정렬 기준 불일치
    - 상호작용 가능한 요소가 시각적으로 명확하게 구분되지 않음
    - 텍스트가 할당된 영역을 초과하여 텍스트 잘림
    '''
    compare = Compare()
    compare.analyze_layout(theme_layout, default_layout)

    # 5-2. 새로운 이미지 자체에 있는 이슈 파악
    '''
    타겟 이슈:
    - 아이콘의 가장자리가 보이지 않거나 잘려보임
    - 달력 아이콘에서 요일 글자가 테두리를 벗어남
    - 앱 내 달력, 시간 아이콘이 현재 날짜 시각과 매칭되지 않음
    - 하이라이트된 항목, 텍스트와 배경간 대비가 낮아 가독성이 떨어짐
    '''
    single = Single()
    single.analyze_layout(theme_layout)
    
    # 5-3. 새로운 이미지 자체에 있는 디자인 이슈 파악
    '''
    타겟 이슈:
    - 다른 기능 요소에 동일한 아이콘이 적용됨
    - 테마가 적용되지 않은 아이콘 등이 있음
    등
    '''
    gemini = Gemini()
    gemini.analyze_layout(theme_layout)

    # 5-4. 새로운 이미지에서 찾은 모든 이슈에 대한 심각도 판정
    gemini.analyze_issues(theme_layout)

    # 5-5. 결과물 출력


