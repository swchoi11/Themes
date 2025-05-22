from src.Mask import Mask
from src.Classification import Classification
from src.Gemini import Gemini

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

# def run(target_image_dir: str):
    # mask = Mask()
    # classify = Classification(target_image_dir)

    # # 새로운 이미지에 대한 전처리
    # ## 타겟 이미지 경로 하위의 이미지들이 전부 흑백 또는 컬러 이미지로 마스킹되어 덮어씌워집니다.
    # mask.mask_directory_images(target_image_dir, is_gray=False)

    # # 레이아웃 분류
    # ## 마스킹된 타겟 이미지와 메인 프레임 경로 하위의 베이스 템플릿을 비교하여 클러스터 아이디를 추출합니다. 
    # classify.layout_classification(target_image_dir)

    # 레이아웃과 이미지를 비교해 cut off, visibility 이슈 확인
    ## 타겟 이미지와 베이스 템플릿을 비교 
    # map components , calculate iou
    
    ## 타겟 이미지와 디폴트 이미지를 비교
    # detect_cut_off_issue, calculate_contrast


    # Gemini를 통한 visibility 및 디자인 이슈 확인
    ## 타겟 이미지와 이슈 목록을 제공하여 이슈를 확인합니다.


    # 결과 산출


import cv2
if __name__ == "__main__":
    # run('./resource/')
    classify = Classification('./resource/wrong/default.png')
    mask = Mask()

    # 1-1 : 원본 이미지끼리 비교
    image_path1 = './resource/wrong/theme.png'
    image_path2 = './resource/wrong/default.png'
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    print(classify.compare_layouts_match(img1, img2))
    print(classify.compare_layouts_ssim(img1, img2))
    print(classify.compare_layouts_orb(img1, img2))

    # 1-2 : 원본 위에 컴포넌트 이미지끼리 비교
    image_path1 = './resource/wrong/theme-red.png'
    image_path2 = './resource/wrong/default-red.png'
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    print("----")
    print(classify.compare_layouts_match(img1, img2))
    print(classify.compare_layouts_ssim(img1, img2))
    print(classify.compare_layouts_orb(img1, img2))

    # 1-3 : 마스킹 만
    image_path1 = './resource/wrong/theme-red.png'
    image_path2 = './resource/wrong/default-red.png'
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    img1, _ = mask.mask_image(img1)
    img2, _ = mask.mask_image(img2)
    print("test------")
    print(classify.compare_layouts_match(img1, img2))
    print(classify.compare_layouts_ssim(img1, img2))
    print(classify.compare_layouts_orb(img1, img2))

