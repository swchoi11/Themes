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
    image_list_file_path = glob.glob('./util_files/vm*_image_list.csv')

    if len(image_list_file_path) > 1:
        logger.info("이미지 파일이 유일하지 않습니다.")
        return []
    
    image_list_file_path = image_list_file_path[0]
    
    with open(image_list_file_path, 'r', encoding='utf-8') as file:
        image_list = file.readlines()
        image_list = [image.replace('\ufeff', '') for image in image_list]
        clean_image_list = [image.strip() for image in image_list if image.strip()]
        clean_image_list = [f'./mnt/resource/{image}' for image in clean_image_list]
        clean_image_list = sorted(clean_image_list, reverse=True)
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
            f'output/excels/all_issues/{filename}.xlsx',
            f'output/excels/final_issue/{filename}.xlsx',
            f'output/{filename}_normal.txt'
            ]        
        
        # result_files.extend(glob.glob('output/images/*.png'))
        # result_files.extend(glob.glob('output/images/not_processed/*.png'))

        INSTANCE_NUM = get_instance_num()

        for result_file in result_files:
            input_file = result_file.replace('output/', '')
            blob = bucket.blob(f"output/themes/vm_{INSTANCE_NUM}/{input_file}")
            blob.upload_from_filename(result_file)

    except Exception as e:
        print(f"파일 업로드 중 오류 발생: {e}")
        return False
    
def get_instance_num():
    image_list_file_path = glob.glob('./resource/vm*_image_list.csv')

    if len(image_list_file_path) > 1:
        logger.info("이미지 파일이 유일하지 않습니다.")
        return []

    image_list_file_path = image_list_file_path[0]
    INSTANCE_NUM = image_list_file_path.split('/')[-1].split('_')[0].replace('vm', '')
    INSTANCE_NUM = int(INSTANCE_NUM)

    return INSTANCE_NUM

def set_api_key():
    api_list_path = './util_files/api_keys.txt'

    with open(api_list_path, 'r', encoding='utf-8') as file:
        api_key_list = file.readlines()
        api_key_list = [api_key.strip() for api_key in api_key_list if api_key.strip()]

    instance_num = get_instance_num()
    
    # 인스턴스 번호에 따라 시작 인덱스 계산 (각 인스턴스마다 3개씩 할당)
    # 할당량 문제가 현재는 없으므로 1개만 가져오고 3개를 복사하겟음
    start_index = (int(instance_num)-10) * 3
    end_index = int(start_index) + 3  # 3개를 가져오기 위해 +3
    
    # 할당된 3개의 키 추출
    instance_keys = api_key_list[start_index:end_index]
    # .env 파일에 API 키 저장
    env_content = []
    
    # 기존 .env 파일이 있다면 읽어오기
    env_path = './.env'
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            env_content = f.readlines()
        
    # API 키 관련 라인들을 제거 (기존 API_KEY들 삭제)
    env_content = [line for line in env_content if not line.strip().startswith('API_KEY')]
    
    # 마지막 라인에 줄바꿈이 없다면 추가
    if env_content and not env_content[-1].endswith('\n'):
        env_content[-1] += '\n'
    
    # 새로운 API 키들 추가
    for i, key in enumerate(instance_keys, 1):
        env_content.append(f'API_KEY{i} = "{key}"\n')
    
    # .env 파일에 쓰기
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(env_content)
    
    logger.info(f"인스턴스 {instance_num}에 API 키 {len(instance_keys)}개 설정 완료")

    download_file_from_bucket(f'image_list/vm{instance_num}_image_list.csv', './util_files/vm{instance_num}_image_list.csv')
    
    return instance_keys