'''
pass 테마 파일에서 레이아웃 그려서 템플릿 확인
새로운 이미지가 들어오면 그 레이아웃을 올려보고 템플릿 확인- 매치 레이아웃
그리고 나서 해당 템플릿에 존재할 수 있는 이슈 - 파싱
추출
'''
from radiobutton import xml_visualize, xml_class_crop, viewgroup_children
import glob
import os

# 1. xml 파일에 있는 레이아웃 이미지 위에 올려서 확인
# image_list = glob.glob("./resource/*.png")

# for image_path in image_list:
#     image_name = os.path.splitext(os.path.basename(image_path))[0]
    # print(image_name)
    # xml_visualize(image_name)
    # xml_class_crop(image_name)

# 2. xml 파일의 레이아웃을 투명 배경 위에 올려서 저장 -> 새로운 테마 이미지 위에 올릴 수 있도록

image_name = "com.samsung.android.dialer_DialtactsActivity_20250514_152102"
# xml_class_crop(image_name)


viewgroup_children(image_name)
