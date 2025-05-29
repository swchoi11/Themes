




import glob
import os
import re
import pandas as pd
import shutil

def clean(filename):
    filename = re.sub(r'\d+', '', filename)
    filename = re.sub('__','',filename)
    return filename


def theme_issues(theme_dir):
    theme_id = os.path.dirname(theme_dir).split('/')[-1]
    image_list = os.listdir(theme_dir)
    issue_list = set()
    for image_path in image_list:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_name = clean(image_name)
        issue_list.add(image_name)
    return theme_id, list(issue_list)

def move_image(resource_dirs, output_dir):
    issue_dict = {}
    for resource_dir in resource_dirs:
        for theme_id in os.listdir(resource_dir):
            theme_path = os.path.join(resource_dir, theme_id)
            if not os.path.isdir(theme_path):
                continue
            for img_file in os.listdir(theme_path):
                img_path = os.path.join(theme_path, img_file)
                issue = clean(img_file)
                if issue not in issue_dict:
                    issue_dict[issue] = []
                save_name = f"{theme_id}_{img_file}"
                issue_dict[issue].append((img_path, save_name))

    for issue, files in issue_dict.items():
        issue_folder = os.path.join(output_dir, issue)
        os.makedirs(issue_folder, exist_ok=True)
        for src_path, save_name in files:
            dst_path = os.path.join(issue_folder, save_name)
            shutil.copy2(src_path, dst_path)
        

if __name__ == "__main__":
    resource_dirs = ["./resource/pass", "./resource/default"]
    output_dir = "./output"
    move_image(resource_dirs, output_dir)

import re
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
def xml_visualize(image_base_name):
    image_path = f"./resource/{image_base_name}.png"
    xml_path = f"./resource/{image_base_name}.xml"
    if not os.path.exists(xml_path):
        print(f"xml 파일이 존재하지 않습니다. {xml_path}")
        return 
    image = cv2.imread(image_path)
    # 이미지가 없을 수도 있으니, 크기만 참고하고 흰 배경 생성
    height, width = image.shape[:2] if image is not None else (2340, 1080)
    background = np.ones((height, width, 3), dtype=np.uint8) * 255

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for elem in root.iter('node'):
        bounds = elem.attrib.get('bounds')
        if bounds:
            match = re.search(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                cv2.putText(background, elem.attrib.get('class'), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(background, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, elem.attrib.get('class'), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite(f"./output/{image_base_name}_mask.png", background)
    cv2.imwrite(f"./output/{image_base_name}_visualized.png", image)
    
def xml_class_crop(image_base_name):
    image_path = f"./resource/{image_base_name}.png"
    xml_path = f"./resource/{image_base_name}.xml"
    if not os.path.exists(xml_path):
        print(f"xml 파일이 존재하지 않습니다. {xml_path}")
        return 
    image = cv2.imread(image_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    i = 0
    for elem in root.iter('node'):
        bounds = elem.attrib.get('bounds')
        if bounds:
            match = re.search(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                image_crop = image[y1:y2, x1:x2]
                os.makedirs(f"./output/{elem.attrib.get('class')}", exist_ok=True)
                cv2.imwrite(f"./output/{elem.attrib.get('class')}/{image_base_name}_crop_{i}.png", image_crop)
                i += 1

def draw_box(image, node, parent=None):
    class_name = node.attrib.get("class","")
    bounds = node.attrib.get("bounds","")
    if "ViewGroup" in class_name or "Layout" in class_name:
        if bounds:
            x1,y1,x2,y2 = map(int, re.findall(r'\d+', bounds))
            cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)        
            for child in node:
                draw_box(image, child, class_name)
    else:
        if bounds:
            x1,y1,x2,y2 = map(int, re.findall(r'\d+', bounds))
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
    

def viewgroup_children(image_base_name):
    image_path = f"./resource/{image_base_name}.png"
    xml_path = f"./resource/{image_base_name}.xml"
    image = cv2.imread(image_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for node in root.iter("node"):
        class_name = node.attrib.get("class","")
        if "ViewGroup" in class_name or "Layout" in class_name:
            draw_box(image, node)

    cv2.imwrite(f"./output/{image_base_name}_viewgroup_children.png", image)






def find_radio_button(image_base_name):
    image_path = f"./resource/{image_base_name}.png"
    xml_path = f"./resource/{image_base_name}.xml"
    image = cv2.imread(image_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    radio_buttons = []

    for elem in root.iter('node'):
        if elem.attrib.get('class') == 'android.widget.RadioButton':
            bounds = elem.attrib.get('bounds')
            if bounds:
                match = re.search(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    radio_buttons.append((x1, y1, x2, y2))

    cv2.imwrite(f"./output/{image_base_name}_visualized.png", image)

    return radio_buttons

def crop_radio_button(image_path, radio_buttons):
    
    image = cv2.imread(image_path)
    for idx, (x1, y1, x2, y2) in enumerate(radio_buttons):
        cropped = image[y1:y2, x1:x2]
        cv2.imwrite(f"./output/radio_button_{idx}.png", cropped)

def radio_button_defect(radio_button_image):
    image = cv2.imread(radio_button_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=2, maxLineGap=0.5)
    h, w = gray.shape
    min_line_length_ratio = 0.05
    print(lines)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if line_length > min_line_length_ratio * max(h, w):
                return True
    return False


if __name__ == "__main__":
    # xml_class_crop("Fail_V2_com.sec.android.app.launcher_LauncherActivity_20250522_164450")
    xml_visualize("Fail_V2_com.sec.android.app.launcher_LauncherActivity_20250522_164450")

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

