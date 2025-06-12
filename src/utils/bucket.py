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


def test_image_list():
    image_list_file_path = glob.glob('./Themes/vm*_image_list.csv')

    if len(image_list_file_path) > 0:
        logger.info("이미지 파일이 유일하지 않습니다.")
        return []
    
    image_list_file_path = image_list_file_path[0]
    
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


        image_list_file_path = glob.glob('./Themes/vm*_image_list.csv')

        if len(image_list_file_path) > 0:
            logger.info("이미지 파일이 유일하지 않습니다.")
            return []
    
        image_list_file_path = image_list_file_path[0]
        INSTANCE_NUM = image_list_file_path.split('/')[-1].split('_')[1]
        INSTANCE_NUM = int(INSTANCE_NUM)
    

        for result_file in result_files:
            input_file = result_file.replace('output/', '')
            blob = bucket.blob(f"result/vm{INSTANCE_NUM}/{input_file}")
            blob.upload_from_filename(result_file)

    except Exception as e:
        print(f"파일 업로드 중 오류 발생: {e}")
        return False
    
