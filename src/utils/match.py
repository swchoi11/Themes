import cv2
import glob
from tqdm import tqdm
import os
from src.utils.utils import calculate_xml_similarity

class Match:
    def __init__(self, file_path: str):
        if file_path.endswith(".xml"):
            self.image_path = file_path.replace(".xml", ".png")
            self.xml_path = file_path
        else:
            self.image_path = file_path
            self.xml_path = file_path.replace(".png", ".xml")
        
    def _included(self):
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"이미지 파일을 찾을 수 없습니다: {self.image_path}")
        ratio =  (img.shape[0]/ img.shape[1])
        if ratio > 2.0:
            return False
        return True

    def select_group(self):
    # 폴드인 경우 가장 유사한 xml그룹 선택
        if not self._included():
            return None
        
        max_similarity = 0

        groups = glob.glob("./mnt/output/classification/*/*/", recursive=True)
        for group in tqdm(groups, desc="그룹 대표 이미지와 비교중"):
            repr_xml_path = [xml for xml in os.listdir(group) if xml.endswith(".xml")][0]
            repr_xml_path = os.path.join(group, repr_xml_path)
            _, _, similarity = calculate_xml_similarity((self.xml_path, repr_xml_path))
            if similarity > max_similarity:
                max_similarity = similarity
                selected_group = group
        return max_similarity, selected_group
    
    def selct_default_image(self, selected_group: str):
        # 98% 이상 유사한 디폴트 이미지 선택
        max_similarity = 0.99
        selected_image = ""

        xml_list = glob.glob(f"{selected_group}/*.xml")
        for xml in tqdm(xml_list, desc="디폴트 이미지 비교중"):
            _, _, similarity = calculate_xml_similarity((self.xml_path, xml))
            if similarity > max_similarity:
                max_similarity = similarity
                selected_xml = xml
        return selected_xml


