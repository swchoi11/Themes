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

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

# from sklearn.cluster import KMeans
import  clip
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
import os
import umap
import hdbscan

import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(BASE_DIR)

class HDBSCAN:

    def __init__(self, image_dir: str, output_csv: str, size_filter_mode: str,
                 size_threshold: int, pca_components: int, min_cluster_size: int,
                 n_neighbors: int, min_dist: float, min_samples: int, 
                 n_components: int, image_extensions = ["*.png", "*.jpg", "*.jpeg"]):
        
        # 경로 지정
        self.ocr = PaddleOCR(
            det_model_dir='../src/weights/en_PP-OCRv3_det_infer',
            rec_model_dir='../src/weights/en_PP-OCRv3_rec_infer',
            cls_model_dir='../src/weights/ch_ppocr_mobile_v2.0_cls_infer',
            lang='en',  # other lang also available
            use_angle_cls=False,
            use_gpu=False,  # using cuda will conflict with pytorch in the same process
            show_log=False,
            max_batch_size=1024,
            use_dilation=True,  # improves accuracy
            det_db_score_mode='slow',  # improves accuracy
            rec_batch_num=1024,
            )
        
        self.image_dir = image_dir
        self.output_csv = output_csv

        # self.num_cluster = num_cluster
        # self.min_width = min_width

        self.size_filter_mode = size_filter_mode.lower()
        self.size_threshold = size_threshold

        self.image_extensions = image_extensions
        self.pca_components = pca_components # 현재 CLIP+UMAP 이어서 사용하지 않음
        self.min_cluster_size = min_cluster_size
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.min_samples = min_samples
        self.n_components = n_components # 30-50
 
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
        # resnet = models.resnet50(pretrained=True)
        # self.model = torch.nn.Sequential(*list(resnet.children())[:-2])
        # self.model.to(self.device).eval()

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_model= SentenceTransformer("paraphrase-MiniLM-L6-v2")
        # self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def extract_features(self, image_path):   # default
        """
        ResNet-50 특징 추출
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1,1)) #resnet에서[:-2]까지 사용할 경우 필요
        # return features.squeeze().cpu().numpy()
        return pooled.squeeze().cpu().numpy()
    
    def extract_clip_features(self, image_path):
        """
        CLIP 특징 추출
        """
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
        return features.cpu().numpy().squeeze()
    
    def extract_clip_ocr_features(self, image_path):
        """
        CLIP + OCR 텍스트 임베딩 결합
        """
        # CLIP
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(image).cpu().numpy().squeeze()
        
        # OCR
        result = self.ocr.ocr(image_path, cls=True)
        texts = []
        for line in result[0]:
            text = line[1][0]
            texts.append(text)
        text_joined = " ".join(texts) if texts else "none"

        # 텍스트 임베딩
        text_embedding = self.text_model.encode(text_joined)

        # 결합
        combined = np.concatenate([clip_features, text_embedding])
        return combined
    
    def subcluster_left_bottom(self, reduced, features, valid_images):
        """
        UMAP 결과에서 왼쪽 하단 영역만 추출하여 재클러스터링 수행
        """
        # 관심 영역 조건 (필요시 조정 가능)
        mask = (reduced[:, 0] < 0.5) & (reduced[:, 1] < 7.0)
        subset_features = features[mask]
        subset_image_names = np.array(valid_images)[mask]

        if len(subset_features) < 5:
            print("[WARN] 하위 영역에 클러스터링 가능한 데이터가 충분하지 않습니다.")
            return

        print(f"[INFO] 하위 영역 클러스터링 대상 수: {len(subset_features)}")

        # --- UMAP 다시 ---
        subset_umap = umap.UMAP(
            metric="cosine", n_components=2, n_neighbors=15, min_dist=0.05, random_state=42
        ).fit_transform(subset_features)

        # --- HDBSCAN 다시 ---
        subset_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3, min_samples=1, cluster_selection_method="leaf"
        )
        subset_labels = subset_clusterer.fit_predict(subset_umap)

        # --- 시각화 ---
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=subset_umap[:, 0], y=subset_umap[:, 1], hue=subset_labels, palette="tab10", s=40)
        plt.title("Subcluster in Left-Bottom UMAP Region")
        plt.tight_layout()
        plt.show()

        # --- 저장 ---
        pd.DataFrame({
            "image_name": subset_image_names,
            "cluster_label": subset_labels
        }).to_csv("subcluster_result.csv", index=False)
        print("[INFO] subcluster_result.csv 저장 완료")

    def run(self):
        """
        클러스터링 실행
        """
        # embeddings, valid_images = [], []
        features, valid_images = [], []

        # 이미지 경로
        image_paths = []
        for ext in self.image_extensions:
            image_paths.extend(
                [p for p in glob.glob(os.path.join(self.image_dir, ext)) if "default" in os.path.basename(p).lower()]
            )
        image_paths.sort()

        print(f"[INFO] 총 {len(image_paths)}개의 이미지 파일이 수집되었습니다.")

        # for img_path in tqdm(image_paths, desc="HDBSCAN 특징 추출 중"):
        #     try:
        #         if "default" in os.path.basename(img_path).lower():
        #             with Image.open(img_path) as img:
        #                 width, _ = img.size
        #                 if width >= self.min_width:
        #                     feature = self.extract_features(img_path)
        #                     embeddings.append(feature)
        #                     valid_images.append(os.path.basename(img_path))
        #     except Exception as e:
        #         print(f"[ERROR] {img_path} 처리 중 오류 발생 : {e}")

        for path in tqdm(image_paths, desc="ResNet 특징 추출"):
            try:
                with Image.open(path) as img:
                    width, _ = img.size
                    
                    if (
                        (self.size_filter_mode == "f" and width <= self.size_threshold) or
                        (self.size_filter_mode == "s" and width > self.size_threshold)
                    ):
                        continue
                
                # feature = self.extract_features(path) # ResNet
                # feature = self.extract_clip_features(path) # clip
                feature = self.extract_clip_ocr_features(path)
                features.append(feature)
                valid_images.append(os.path.basename(path))
            except:
                continue
        
        features = np.array(features).reshape(len(features), -1)
        # features = np.array(features) # clip

        print(f"[INFO] 조건에 맞는 이미지 수: {len(valid_images)}")

        # if embeddings:
            # K-Means
            # kmeans = KMeans(n_clusters=self.num_cluster, random_state=42)
            # cluster_labels = kmeans.fit_predict(embeddings)
            
        # PCA 차원 축소
        pca = PCA(n_components=self.pca_components, random_state=42)
        features_pca = pca.fit_transform(features)
    
        # UMAP 차원 축소
        reduced = umap.UMAP(metric="cosine", n_components=self.n_components, n_neighbors=self.n_neighbors,
                            min_dist=self.min_dist, random_state=42).fit_transform(features_pca)

        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                    min_samples=self.min_samples, cluster_selection_method="leaf",
                                    cluster_selection_epsilon=0.00)
        cluster_labels = clusterer.fit_predict(reduced)

        # 추가 clustering
        # self.subcluster_left_bottom(reduced, features, valid_images)
        
        # 전체 시각화
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=cluster_labels, palette="tab10", s=30)
        plt.title("UMAP Reduced Space with HDBSCAN Clusters")
        plt.legend(title="Cluster", loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

        unique_clusters = set(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = sum(cluster_labels == -1)

        print(f"[INFO] 유효 클러스터 수: {n_clusters}")
        print(f"[INFO] 노이즈로 분류된 이미지 수: {n_noise}")

        for cluster_id in sorted(unique_clusters):
            count = sum(cluster_labels == cluster_id)
            if cluster_id == -1:
                print(f"[Cluster -1] (Noise): {count}개")
            else:
                print(f"[Cluster {cluster_id}] 이미지 수: {count}")
                
            df = pd.DataFrame({
                "image_name": valid_images,
                "cluster_label": cluster_labels
            })

        df.to_csv(self.output_csv, index=False)
        print(f"[INFO] HDBSCAN 결과 저장 완료 : {self.output_csv}")

        # print(f"[INFO] 유효 클러스터 수: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
        # print(f"[INFO] 노이즈로 분류된 이미지 수: {(cluster_labels == -1).sum()}")
        # else:
        #     print("[WARN] 유효한 이미지가 없어 클러스터링이 실행되지 않았습니다.")


if __name__ == "__main__":
    # 사용자 정의 경로 및 파라미터 설정
    image_dir = "./resource/raw"
    output_csv = "./cluster_result.csv"

    # 클러스터링 파라미터 설정
    clustering_params = {
        "size_filter_mode": "f",   
        "size_threshold": 1800,
        "pca_components": 50, 
        "n_components": 40,
        "n_neighbors": 30,
        "min_dist": 0.1,
        "min_cluster_size": 5,
        "min_samples": 1
    }

    # 인스턴스 생성 및 실행
    clusterer = HDBSCAN(
        image_dir=image_dir,
        output_csv=output_csv,
        **clustering_params
    )

    clusterer.run()
