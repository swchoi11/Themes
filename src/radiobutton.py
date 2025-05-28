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
    image_name = "com.android.settings_SubSettings_20250520_074420"
    radio_buttons = find_radio_button(image_name)
    print(radio_buttons)
    crop_radio_button(f"./resource/{image_name}.png", radio_buttons)
    for i in range(8):
        result = radio_button_defect(f"./output/radio_button_{i}.png")
        print(result)