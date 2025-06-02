'''
classification 후 그룹별로 해당 그룹에서 확인해야할 이슈 확인 및 컴포넌트 선정
'''




from xml.etree import ElementTree as ET
import re

from src.utils import get_bounds

class Detect:
    def __init__(self, file_path: str):
        if file_path.endswith(".xml"):
            self.image_path = file_path.replace(".xml", ".png")
            self.xml_path = file_path
        else:
            self.image_path = file_path
            self.xml_path = file_path.replace(".png", ".xml")
        self.file_name = self.image_path.replace('.png', '')

        tree = ET.parse(self.xml_path)
        self.root = tree.getroot().find('node')
        
    def get_nodes_same_level(self, level=0, result=[]):
        if len(result) <= level:
            result.append([])
        result[level].append(self.root)
        for child in self.root.findall('node'):
            self.get_nodes_same_level(child, level+1, result)
        return result
    
    def get_valid_components(self):
        components = [get_bounds(node.get('bounds')) for node in self.root]
        valid_components = []
        if len(components) == 1:
            return None
        else:
            for component in components:
                if abs(component[0] - component[2]) < 1812 * 0.5 and abs(component[1] - component[3]) < 2176 * 0.5:
                    valid_components.append(component)
            return valid_components
    
    def filter_components(self, components):
        valid_components = []
        for i, comp in enumerate(components):
            contained = False
            for j, comp_b in enumerate(components):
                if i != j and self.is_contained(comp, comp_b):
                    contained = True
                    break
            if not contained:
                valid_components.append(comp)
        return valid_components
    
    def is_contained(self, comp, comp_b):
        return comp[0] >= comp_b[0] and comp[1] >= comp_b[1] and comp[2] <= comp_b[2] and comp[3] <= comp_b[3]