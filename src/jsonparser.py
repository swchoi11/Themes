import cv2
import json

json_path = "./resource/ocr_result_20250416_084238_Clock_Resume_poor_Kim.json"
image_path = "./resource/20250416_084238_Clock_Resume_poor_Kim.png"

img = cv2.imread(image_path)

with open(json_path, "r") as f:
    json_data = json.load(f)

# 추출된 json 요소의 바운딩 박스 이미지에 표시
for element in json_data:
    x1 = int(element["bbox"]["x1"])
    y1 = int(element["bbox"]["y1"])
    x2 = int(element["bbox"]["x2"])
    y2 = int(element["bbox"]["y2"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("./resource/result.jpg", img)