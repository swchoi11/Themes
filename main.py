from src.Mask import Mask
from src.Classification import Classification

# 1. raw 이미지에 대한 클러스터링

# 2. 클러스터링 결과에 대한 컴포넌트 추출

# 3. 추출된 컴포넌트를 바탕으로 베이스 템플릿 추출

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
# def run(target_image_dir: str):
    target_image_path = './resource/test.png'
    mask = Mask()
    classify = Classification(is_gray=True)

    # # 새로운 이미지에 대한 전처리
    # ## 이지 파서

    ## 타겟 이미지 경로 하위의 이미지들이 전부 흑백 또는 컬러 이미지로 마스킹되어 덮어씌워집니다.
    # mask.mask_directory_images(target_image_dir, is_gray=False)

    # 레이아웃 분류
    ## 마스킹된 타겟 이미지와 메인 프레임 경로 하위의 베이스 템플릿을 비교하여 클러스터 아이디를 추출합니다. 
    cluster, score = classify.get_cluster(target_image_path, method='orb')
    print(f"클러스터: {cluster}")
    print(f"점수: {score}")

    # 레이아웃과 이미지를 비교해 cut off, visibility 이슈 확인
    ## 타겟 이미지와 베이스 템플릿을 비교 
    # map components , calculate iou
    
    ## 타겟 이미지와 디폴트 이미지를 비교
    # detect_cut_off_issue, calculate_contrast
    default_image_path, default_score = classify.get_default(target_image_path, cluster, method='orb')
    print(f"디폴트 이미지: {default_image_path}")
    print(f"디폴트 점수: {default_score}")


    # Gemini를 통한 visibility 및 디자인 이슈 확인
    ## 타겟 이미지와 이슈 목록을 제공하여 이슈를 확인합니다.


    # 결과 산출


