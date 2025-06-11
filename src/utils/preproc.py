import os
from google.cloud import storage
from typing import List, Optional
from dotenv import load_dotenv
from google.cloud import compute_v1
import subprocess
import socket
import json

load_dotenv()
BUCKET_NAME = os.getenv('BUCKET_NAME')

def get_instance_info():
    instances = subprocess.run([
        'gcloud', 'compute', 'instances', 'list',
        '--filter', f'name={socket.gethostname()}',
        '--format', 'json'], capture_output=True, text=True, check=True)

    instances = json.loads(instances.stdout)

    instance_name = instances[0]['disks'][0]['deviceName']

    return instance_name

# INSTANCE_NAME = get_instance_info()
# INSTANCE_NUM = int(INSTANCE_NAME.split('-')[-1])
INSTANCE_NUM = 0
image_list_file_path = './resource/image_list.csv'


def from_bucket_image_list(prefix: str = "image_list") :
    try:
        # Storage 클라이언트 초기화
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{prefix}/vm{INSTANCE_NUM}_image_list.csv")        
        
        blob.download_to_filename(image_list_file_path)
    
    except Exception as e:
        print(f"버킷에서 파일 목록을 가져오는 중 오류 발생: {e}")
        return []

def download_file_from_bucket(source_blob_name: str, destination_file_name: str) -> bool:
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
                
        blob.download_to_filename(destination_file_name)
        return True
        
    except Exception as e:
        print(f"파일 다운로드 중 오류 발생: {e}")
        return False
