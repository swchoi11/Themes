import os
import glob
from tqdm import tqdm
import pandas as pd

from xml.etree import ElementTree as ET
from src.xmlParser import XMLParser

xml_list = glob.glob("D:/hnryu/Themes/resource/xml/*.xml")

parsing_class_name = []
parsing_content_name = []
for xml_file_path in tqdm(xml_list):

    filename = os.path.splitext(os.path.basename(xml_file_path))[0]
    image_file_path = f"D:/hnryu/Themes/resource/image/{filename}.png"

    try:
        parser = XMLParser(image_path=image_file_path, xml_path=xml_file_path)
        all_node_names = parser.get_all_context()
        all_class_names = parser.get_all_classes()

        if all_node_names:
            for node_name in all_node_names:
                parsing_content_name.append(node_name)
        if all_class_names:
            for class_name in sorted(all_class_names):
                parsing_class_name.append(class_name)
        else:
            print("추출된 클래스가 없습니다.")

    except (FileNotFoundError, ValueError, ET.ParseError) as e:
        print(f"오류가 발생했습니다: {e}")

## UI Component
class_df = pd.DataFrame(parsing_class_name)
class_df.to_csv('D:/hnryu/Themes/output/class_xml.csv')

## UI Component content
content_df = pd.DataFrame(parsing_content_name)
content_df.to_csv('D:/hnryu/Themes/output/content_xml.csv')

print("*")