from xml.etree import ElementTree as ET
import cv2
from src.utils.utils import get_bounds


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

        self.image = cv2.imread(self.image_path)
        self.image_width, self.image_height = self.image.shape[:2]

    def _all_nodes(self, node):
        nodes = [node]
        for child in node.findall('node'):
            nodes.extend(self._all_nodes(child))
        return nodes
    
    def get_class_components(self, class_names):
        all_nodes = self._all_nodes(self.root)
        components = []

        for idx, node in enumerate(all_nodes):
            if node.get('class') in class_names:
                bounds_str = node.get('bounds')
                if bounds_str:
                    bounds = get_bounds(bounds_str)
                    if bounds:
                        component = {
                            'index': idx,
                            'type': node.get('class'),
                            'content': node.get('text'),
                            'resource_id': node.get('resource-id'),
                            'bounds': bounds
                        }
                        components.append(component)

        return components if components else None
                
    def _filter_text(self):
        text_classes = [
        'android.widget.TextView',
        'android.widget.EditText'
        ]
        components = self.get_class_components(text_classes)
        valid_components = self._text_content_filter(components)
        return valid_components
    
    def _filter_button(self):
        button_classes = [
        'android.widget.Button',
        'android.widget.ImageButton', 
        'android.widget.RadioButton',
        'android.widget.CheckBox'
        ]
        components = self.get_class_components(button_classes)
        return components
    
    def _no_filter(self):
        classes = [
        'android.widget.TextView',
        'android.widget.EditText',
        'android.widget.Button',
        'android.widget.ImageButton', 
        ]
        components = self.get_class_components(classes)
        return components

    def _is_contained_node(self, inner_node, outer_node):
        """노드 간 포함관계 확인"""
        inner_bounds_str = inner_node.get('bounds')
        outer_bounds_str = outer_node.get('bounds')
        
        if not inner_bounds_str or not outer_bounds_str:
            return False
            
        inner_bounds = get_bounds(inner_bounds_str)
        outer_bounds = get_bounds(outer_bounds_str)
        
        if not inner_bounds or not outer_bounds:
            return False
        
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_bounds
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_bounds

        return (inner_x1 >= outer_x1 and inner_y1 >= outer_y1 and
                inner_x2 <= outer_x2 and inner_y2 <= outer_y2)

    def _is_contained(self, inner, outer):
        """컴포넌트 딕셔너리 간 포함관계 확인"""
        inner_x1, inner_y1, inner_x2, inner_y2 = inner['bounds']
        outer_x1, outer_y1, outer_x2, outer_y2 = outer['bounds']

        return (inner_x1 >= outer_x1 and inner_y1 >= outer_y1 and
                inner_x2 <= outer_x2 and inner_y2 <= outer_y2)

    def _text_content_filter(self, components):
        valid_components = []
        if not components:
            return valid_components
            
        for component in components:
            content = component.get('content')
            # None이 아니고, 빈 문자열이 아니고, 공백만 있는 것도 아닌 경우
            if content and content.strip():
                valid_components.append(component)
        return valid_components

    def get_valid_components(self):
        all_components = self._all_nodes(self.root)

        size_filtered_components = self._size_filter(all_components)
        valid_components = self._include_filter(size_filtered_components)
        return valid_components

    def _size_filter(self, components):
        valid_components = []
        for idx, node in enumerate(components):
            if node.get('bounds'):
                bounds = get_bounds(node.get('bounds'))
                if bounds:
                    width = abs(bounds[2] - bounds[0])
                    height = abs(bounds[3] - bounds[1])
                    if width < self.image_width * 0.5 and height < self.image_height * 0.5:
                        valid_component = {
                            'index': idx,
                            'type': node.get('class'),
                            'content': node.get('text'),
                            'resource_id': node.get('resource-id'),
                            'bounds': bounds
                        }
                        valid_components.append(valid_component)
        return valid_components
    
    def _include_filter(self, components):
        valid_components = []
        for i, comp in enumerate(components):
            contained = False
            for j, comp_b in enumerate(components):
                if i != j and self._is_contained(comp, comp_b):
                    contained = True
                    break
            if not contained:
                valid_components.append(comp)
        return valid_components
    
    def get_icon_components(self):
        all_nodes = self._all_nodes(self.root)
        
        # 아이콘 관련 클래스와 리소스 ID 패턴 정의
        icon_classes = ['android.widget.ImageView']
        icon_resource_patterns = [
            'icon', 'Icon', 'ICON',
            'badge', 'Badge', 'BADGE', 
            'ic_', 'Ic_', 'IC_',
        ]
        
        icon_components = []
        for idx, node in enumerate(all_nodes):
            bounds = get_bounds(node.get('bounds'))
            if bounds:
                width = abs(bounds[2] - bounds[0])
                height = abs(bounds[3] - bounds[1])
                node_class = node.get('class')
                resource_id = node.get('resource-id', '')
                
                # 조건 1: ImageView이면서 적절한 크기
                is_imageview_icon = (node_class == 'android.widget.ImageView' and 
                                   width <= 200 and height <= 200 and 
                                   width >= 10 and height >= 10)
                
                # 조건 2: 리소스 ID에 아이콘 관련 패턴이 포함된 경우
                has_icon_resource = any(pattern in resource_id for pattern in icon_resource_patterns)
                
                # 조건 3: 기존 크기 기반 조건 (백업용)
                size_based_icon = (width <= 100 and height <= 100 and 
                                 width >= 20 and height >= 20)
                
                if is_imageview_icon or has_icon_resource or size_based_icon:
                    icon_component = {
                        'index': idx,
                        'type': node_class,
                        'content': node.get('text'),
                        'resource_id': resource_id,
                        'bounds': bounds
                    }
                    if icon_component not in icon_components:
                        icon_components.append(icon_component)

        # 바운딩박스가 동일한 경우에만 중복으로 처리하여 제거
        unique_components = []
        seen_bounds = set()
        
        for comp in icon_components:
            bbox = tuple(comp['bounds'])  # 바운딩박스를 튜플로 변환
            if bbox not in seen_bounds:
                unique_components.append(comp)
                seen_bounds.add(bbox)
        
        return unique_components
    
    def all_node_classes(self):
        all_nodes = self._all_nodes(self.root)
        classes = []
        for node in all_nodes:
            classes.append(node.get('class'))
        return list(set(classes))