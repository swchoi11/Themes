import os
import pandas as pd
from pathlib import Path
from google.cloud import storage

# 구글 클라우드 스토리지 설정
BUCKET_NAME = "theme-qa-poc-gs-theme"  # 실제 버킷 이름으로 변경하세요
SOURCE_FOLDER = "resource"  # 버킷 내 소스 폴더
SPLIT_NUMBER = 10
OUTPUT_PREFIX = 'vm'
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

OUTPUT_DIR = './valid_list'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_bucket_files(bucket_name, folder_path):
    """버킷에서 특정 폴더의 파일 목록을 가져옵니다."""
    try:
        storage_client = storage.Client.from_service_account_json('./service_account.json')
        bucket = storage_client.bucket(bucket_name)
        
        blobs = bucket.list_blobs(prefix=folder_path)
        files = []
        
        for blob in blobs:
            if blob.name.endswith('/'):  # 폴더는 건너뛰기
                continue
            file_path = blob.name
            if Path(file_path).suffix.lower() in VALID_EXTENSIONS:
                files.append(os.path.basename(file_path))
        
        print(f"버킷에서 {len(files)}개의 이미지 파일을 찾았습니다.")
        return files
    except Exception as e:
        print(f"버킷에서 파일 목록을 가져오는 중 오류 발생: {e}")
        return []

def upload_to_bucket(bucket_name, local_file_path, bucket_file_path):
    """로컬 파일을 버킷에 업로드합니다."""
    try:
        storage_client = storage.Client.from_service_account_json('./service_account.json')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(bucket_file_path)
        
        blob.upload_from_filename(local_file_path)
        print(f"업로드 완료: {bucket_file_path}")
        return True
    except Exception as e:
        print(f"파일 업로드 중 오류 발생: {e}")
        return False

def main():
    print("구글 클라우드 스토리지에서 이미지 파일 목록을 가져오는 중...")
    
    # 버킷에서 파일 목록 가져오기
    image_files = list_bucket_files(BUCKET_NAME, SOURCE_FOLDER)
    
    if not image_files:
        print("이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 발견했습니다.")
    print(f"{SPLIT_NUMBER}개로 분할하여 CSV 파일을 생성합니다.")
    
    # 분할 계산
    chunk_size = len(image_files) // SPLIT_NUMBER
    remainder = len(image_files) % SPLIT_NUMBER
    
    uploaded_files = []
    failed_uploads = []
    
    for i in range(SPLIT_NUMBER):
        start = i * chunk_size
        end = start + chunk_size
        
        # 마지막 청크(10번째)에 나머지 파일들을 추가
        if i == SPLIT_NUMBER - 1 and remainder > 0:
            end += remainder
        
        chunk = image_files[start:end]
        
        if chunk:  # 빈 청크가 아닌 경우에만 CSV 생성
            df = pd.DataFrame(chunk, columns=["FileName"])
            local_filename = f'{OUTPUT_DIR}/{OUTPUT_PREFIX}{i}_image_list.csv'
            bucket_filename = f'image_list/{OUTPUT_PREFIX}{i}_image_list.csv'
            
            try:
                df.to_csv(local_filename, index=False, header=False, encoding='utf-8-sig')
                print(f'{local_filename} 생성 완료 (파일 수: {len(chunk)}개)')
                
                # 버킷에 업로드
                if upload_to_bucket(BUCKET_NAME, local_filename, bucket_filename):
                    uploaded_files.append(bucket_filename)
                else:
                    failed_uploads.append(bucket_filename)
            except Exception as e:
                print(f"파일 생성 중 오류 발생: {e}")
                failed_uploads.append(bucket_filename)
    
    print(f"\n작업 완료! 총 {len(uploaded_files)}개의 파일이 버킷에 업로드되었습니다.")
    
    # if uploaded_files:
    #     print("업로드된 파일들:")
    #     for file in uploaded_files:
    #         print(f"  - {file}")
    
    # if failed_uploads:
    #     #logger.warning(f"업로드 실패한 파일들 ({len(failed_uploads)}개):")
    #     for file in failed_uploads:
    #         #logger.warning(f"  - {file}")

if __name__ == "__main__":
    main()