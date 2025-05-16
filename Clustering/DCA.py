# 1. 이미지 전처리
# 2. 모델
# 3. 특징 추출 함수 정의
    ## RGB 이미지 기준 -> ResNet-50 CNN -> 학습된 필터로 특징 추출 -> numpy로 반환
# 4. 이미지 필터링 및 특징 추출
# 5. KMeans 클러스터링

import os
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.cluster import KMeans
import pandas as pd

IMAGE_DIR = "./cluster-data"  
OUTPUT_CSV = "dca_image_clusters.csv"
NUM_CLUSTERS = 15 #10  
MIN_WIDTH = 1800

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().cpu().numpy()

valid_images = []
embeddings = []

# image_paths = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
# default 파일만
image_paths = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg")) and "default" in f.lower()
]

for img_name in tqdm(image_paths, desc="Filtering + Extracting features"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            if width >= MIN_WIDTH:
                feature = extract_features(img_path)
                embeddings.append(feature)
                valid_images.append(img_name)
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print(f"총 {len(valid_images)}개의 이미지가 width ≥ {MIN_WIDTH} 조건을 만족합니다.")

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

df = pd.DataFrame({
    "image_name": valid_images,
    "cluster_label": cluster_labels
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved clustered results to {OUTPUT_CSV}")
