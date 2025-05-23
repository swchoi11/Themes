# 디폴트 이미지 & 이미지의 json / 테마 이미지 & 이미지의 json 비교
import json
from model import Layout






class LayoutAwareParser:
    def json_parser(self, json_path: str) -> Layout:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return Layout(**data)


if __name__ == "__main__":
    parser = LayoutAwareParser()
    layout = parser.json_parser('./resource/test.json')
    
    # 스켈레톤 요소 접근
    print("First element:", layout.skeleton.elements[0])
    
    # 파싱된 영역 접근
    print("\nHeader elements count:", layout.parsed_regions.header.elements_count)
    print("Header OCR text:", layout.parsed_regions.header.cropped_ocr.text)
    
    # 레이아웃 영역 접근
    header_elements = layout.layout_regions.header
    print("\nHeader region elements:", len(header_elements))