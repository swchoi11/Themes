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

    # 실행
    dca.run()
    organizer.run()
    parser.run()
    template.run()

    print("\n 전체 처리 완료")


