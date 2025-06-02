from xml.etree import ElementTree as ET

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
    
    def get_valid_components(self, filter_type="all"):
        # 모든 노드에서 bounds 추출
        all_nodes = self._get_all_nodes(self.root)
        components = []
        
        for node in all_nodes:
            bounds_str = node.get('bounds')
            class_name = node.get('class', '')
            text_content = node.get('text', '')
            
            if bounds_str:
                bounds = get_bounds(bounds_str)
                if bounds:
                    # 필터링 타입에 따른 컴포넌트 선별
                    if self._should_include_component(class_name, text_content, filter_type):
                        components.append(bounds)
        
        if len(components) <= 1:
            return None
        
        valid_components = []
        for component in components:
            width = abs(component[2] - component[0])
            height = abs(component[3] - component[1])
            
            # 화면 크기의 절반보다 작은 컴포넌트만 유효한 것으로 간주
            if width < 1812 * 0.5 and height < 2176 * 0.5:
                valid_components.append(component)
        
        # 겹치는 컴포넌트들 필터링
        if valid_components:
            valid_components = self.filter_components(valid_components)
        
        return valid_components if valid_components else None
    
    def _is_valid_text_content(self, text_content):
        """
        텍스트 내용이 유효한지 검증
        
        Args:
            text_content (str): 검증할 텍스트 내용
            
        Returns:
            bool: 유효한 텍스트이면 True, 아니면 False
        """
        if not text_content:
            return False
        
        return True
    
    def _should_include_component(self, class_name, text_content, filter_type):
        if filter_type == "all":
            return True
        
        # 실제 XML에서 사용되는 텍스트 관련 클래스명들
        text_classes = [
            'android.widget.TextView',
        ]
        
        # 실제 XML에서 사용되는 버튼 관련 클래스명들
        button_classes = [
            'android.widget.Button',
            'android.widget.ImageButton', 
            'android.widget.RadioButton',
            'android.widget.CheckBox'
        ]
        
        # 유효한 텍스트 내용이 있는지 확인 (더 엄격한 검증)
        has_valid_text = self._is_valid_text_content(text_content)
        is_text_class = any(text_class in class_name for text_class in text_classes)
        is_button_class = any(button_class in class_name for button_class in button_classes)
        
        if filter_type == "text":
            # 텍스트 필터의 경우: TextView 클래스이면서 유효한 텍스트가 있거나, 
            # 다른 클래스라도 유효한 텍스트가 있으면 포함
            return (is_text_class and has_valid_text) or (not is_text_class and has_valid_text)
        elif filter_type == "button":
            return is_button_class
        elif filter_type == "text_and_button":
            # 텍스트와 버튼 필터의 경우: 버튼 클래스이거나, 유효한 텍스트가 있는 경우
            return is_button_class or has_valid_text
        
        return False
    
    def get_all_components(self):
        all_nodes = self._get_all_nodes(self.root)
        components = []
        
        for node in all_nodes:
            bounds_str = node.get('bounds')
            if bounds_str:
                bounds = get_bounds(bounds_str)
                if bounds:
                    components.append(bounds)
        
        return components if components else None
    
    def _get_all_nodes(self, node):
        nodes = [node]
        for child in node.findall('node'):
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
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