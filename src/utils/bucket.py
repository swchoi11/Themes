import os
import glob
from google.cloud import storage
from typing import List, Optional
from dotenv import load_dotenv
from google.cloud import compute_v1
import subprocess
import socket
import json
from src.utils.logger import init_logger

logger = init_logger()

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
        client = storage.Client.from_service_account_json('./service_account.json')
        
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{prefix}/vm{INSTANCE_NUM}_image_list.csv")        
        
        blob.download_to_filename(image_list_file_path)
        return image_list_file_path
    
    except Exception as e:
        print(f"버킷에서 파일 목록을 가져오는 중 오류 발생: {e}")
        return []

def test_image_list():
    image_list_file_path = from_bucket_image_list()

    with open(image_list_file_path, 'r', encoding='utf-8') as file:
        image_list = file.readlines()
        clean_image_list = [image.strip() for image in image_list if image.strip()]
        clean_image_list = [f'./mnt/resource/{image}' for image in clean_image_list]
        return clean_image_list



def download_file_from_bucket(source_blob_name: str, destination_file_name: str) -> bool:
    try:
        client = storage.Client.from_service_account_json('./service_account.json')
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
                
        blob.download_to_filename(destination_file_name)
        return True
        
    except Exception as e:
        print(f"파일 다운로드 중 오류 발생: {e}")
        return False

def upload_to_bucket(json_filename: str):
    try:
        client = storage.Client.from_service_account_json('./service_account.json')
        bucket = client.bucket(BUCKET_NAME)

        filename = os.path.basename(json_filename).replace('.json', '')
    
        result_files = [
            f'output/jsons/all_issues/{filename}.json',
            f'output/jsons/final_issue/{filename}.json',
            f'output/excels/all_issues/{filename}.xlsx',
            f'output/excels/final_issue/{filename}.xlsx',
            ]        
        
        result_files.extend(glob.glob('output/images/*.png'))
        result_files.extend(glob.glob('output/images/not_processed/*.png'))


        for result_file in result_files:
            input_file = result_file.replace('output/', '')
            blob = bucket.blob(f"result/vm{INSTANCE_NUM}/{input_file}")
            blob.upload_from_filename(result_file)

    except Exception as e:
        print(f"파일 업로드 중 오류 발생: {e}")
        return False
    
