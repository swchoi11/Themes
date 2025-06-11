import cv2
import re
import xml.etree.ElementTree as ET
import os
import random
from src.utils.utils import get_bounds
import numpy as np


class Visual:
    def __init__(self, file_path: str):
        if file_path.endswith(".xml"):
            self.image_path = file_path.replace(".xml", ".png")
            self.xml_path = file_path
        else:
            self.image_path = file_path
            self.xml_path = file_path.replace(".png", ".xml")
        self.file_name = self.image_path.replace('.png', '')

    def find_component(self, class_name: str):
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"이미지 파일을 찾을 수 없습니다: {self.image_path}")
            return None

        tree = ET.parse(self.xml_path)
        if tree is None:
            print(f"xml 파일을 찾을 수 없습니다: {self.xml_path}")
            return None
        root = tree.getroot()
        components = []
        j = 0
        for elem in root.iter('node'):
            original_image = image.copy()
            if elem.attrib.get('class') == class_name:
                bounds = elem.attrib.get('bounds')
                if bounds:
                    match = re.search(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
                    if match:
                        x1, y1, x2, y2 = map(int, match.groups())
                        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        components.append((x1, y1, x2, y2))
                        j += 1
                        cv2.imwrite(f"./output/{class_name}_visualized_{j}.png", original_image)

        return components
    
    def xml_visualize(self, output_path: str=None):
        if not os.path.exists(self.xml_path):
            print(f"xml 파일이 존재하지 않습니다. {self.xml_path}")
            return 
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"이미지 파일을 찾을 수 없습니다: {self.image_path}")
            return 

        height, width = image.shape[:2] if image is not None else (2340, 1080)
        background = np.ones((height, width, 3), dtype=np.uint8) * 255

        tree = ET.parse(self.xml_path)
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

        if output_path is None:
            cv2.imwrite(f"{self.file_name}_mask.png", background)
            cv2.imwrite(f"{self.file_name}_visualized.png", image)
        else:
            cv2.imwrite(f"{output_path}_mask.png", background)
            cv2.imwrite(f"{output_path}_visualized.png", image)

    def draw_all_nodes(self, nodes, image_path, file_name='output.png'):
        img = cv2.imread(image_path)
        for node in nodes:
            bounds = node.attrib.get('bounds')
            if bounds:
                x1, y1, x2, y2 = get_bounds(bounds)
                random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), random_color, 2)
                cv2.putText(img, node.attrib.get('class'), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, random_color, 2)
        cv2.imwrite(file_name, img)    
    
    

        