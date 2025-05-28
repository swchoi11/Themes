import cv2
import json
import numpy as np

def same_icon_detect(image_path, elements):
    img = cv2.imread(image_path)
    icon_list = []
    for element in elements:
        if element["type"] == "icon":
            x1 = int(element["bbox"]["x1"])
            y1 = int(element["bbox"]["y1"])
            x2 = int(element["bbox"]["x2"])
            y2 = int(element["bbox"]["y2"])
            icon_list.append(img[y1:y2, x1:x2])

    for i in range(len(icon_list)):
        for j in range(i+1, len(icon_list)):
            icon = icon_list[i]
            other_icon = icon_list[j]
            # 크기 맞추기
            if icon.shape == other_icon.shape:
                res = cv2.matchTemplate(icon, other_icon, cv2.TM_CCOEFF_NORMED)
                print(res.max())
                if res.max() > 0.9:
                    print(f"동일한 아이콘 발견: {i}와 {j}")
                    cv2.imshow("icon", icon)
                    cv2.imshow("other_icon", other_icon)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                # 크기가 다르면 스킵 또는 리사이즈
                continue

if __name__ == "__main__":
    image_path = "./resource/com.samsung.android.app.contacts_PeopleActivity_20250515_171302.png"
    json_path = "./resource/merged_boxes_com.samsung.android.app.contacts_PeopleActivity_20250515_171302.json"
    elements = json.load(open(json_path, "r"))
    same_icon_detect(image_path, elements)