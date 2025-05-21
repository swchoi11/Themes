# 1. 이미지 전처리
# 2. 모델
# 3. 특징 추출 함수 정의
    ## RGB 이미지 기준 -> ResNet-50 CNN -> 학습된 필터로 특징 추출 -> numpy로 반환
# 4. 이미지 필터링 및 특징 추출
# 5. KMeans 클러스터링

import os
import glob
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm


class DCAClustering:

    def __init__(self, image_dir: str, output_csv: str, num_cluster: int,
                 min_width: int=1800,
                 image_extensions = ["*.png", "*.jpg", "*.jpeg"]):
        
        self.image_dir = image_dir
        self.output_csv = output_csv
        self.num_cluster = num_cluster
        self.min_width = min_width
        self.image_extensions = image_extensions

        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device).eval()

    def extract_features(self, image_path):
        """
        특징 추출
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze().cpu().numpy()

    def run(self):
        """
        클러스터링 실행
        """

        embeddings, valid_images = [], []

        # 이미지 경로
        image_paths = []
        for ext in self.image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
        image_paths.sort()

        print(f"[INFO] 총 {len(image_paths)}개의 이미지 파일이 수집되었습니다.")

        # default 파일만
        # image_paths = [
        #     f for f in os.listdir(IMAGE_DIR)
        #     if f.lower().endswith((".jpg", ".png", ".jpeg")) and "default" in f.lower()
        # ]

        for img_path in tqdm(image_paths, desc="DCA 특징 추출 중"):
            try:
                if "default" in os.path.basename(img_path).lower():
                    with Image.open(img_path) as img:
                        width, _ = img.size
                        if width >= self.min_width:
                            feature = self.extract_features(img_path)
                            embeddings.append(feature)
                            valid_images.append(os.path.basename(img_path))
            except Exception as e:
                print(f"[ERROR] {img_path} 처리 중 오류 발생 : {e}")

        print(f"[INFO] 조건에 맞는 이미지 수: {len(valid_images)}")

        if embeddings:
            kmeans = KMeans(n_clusters=self.num_cluster, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            df = pd.DataFrame({
                "image_name": valid_images,
                "cluster_label": cluster_labels
            })
            df.to_csv(self.output_csv, index=False)
            print(f"[INFO] DCA 결과 : {self.output_csv}")
        else:
            print("[WARN] 유효한 이미지가 없어 클러스터링이 실행되지 않았습니다.")
