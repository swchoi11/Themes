import os
import cv2
import pandas as pd
from src.boxParser import split_by_ratio, hex_items_detect, compare_layouts, layout_classification  

# split_by_ratio("./resource/xml-bbox/")

# for image_path in os.listdir("./resource/xml-bbox/split_1080_2340/"):
#     print(image_path)
#     image_path = os.path.join("./resource/xml-bbox/split_1080_2340/", image_path)
#     image_name = os.path.basename(image_path).split(".")[0]
#     split_path = '1080_2340'
#     hex_items_detect(image_path, image_name, split_path)

RESULT = compare_layouts("./resource/xml-bbox/detected_contours/1080_2340/masked_9.png", "./resource/xml-bbox/detected_contours/1080_2340/masked_45.png")
print(RESULT)
# layout_classification("./resource/xml-bbox/detected_contours/1080_2340/")
   