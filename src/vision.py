import io
import os
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont # 이미지에 경계 상자 및 텍스트를 그리기 위해 PIL 사용

def detect_objects_in_screenshot(image_path):
    """스크린샷 이미지에서 객체를 감지하고 결과를 출력합니다."""

    # Vision API 클라이언트 인스턴스화
    client = vision.ImageAnnotatorClient()

    # 이미지 파일 로드
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # 객체 감지 수행
    print(f"'{os.path.basename(image_path)}'에서 객체 감지를 수행합니다...")
    response = client.object_localization(image=image)
    localized_object_annotations = response.localized_object_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    print(f"'{len(localized_object_annotations)}'개의 객체를 감지했습니다.")

    # 감지된 객체 정보 출력 및 이미지에 표시 (선택 사항)
    img_to_draw = Image.open(image_path)
    draw = ImageDraw.Draw(img_to_draw)
    img_width, img_height = img_to_draw.size

    try:
        # 시스템에 따라 적절한 폰트를 지정하세요. 없으면 기본 폰트 사용.
        font = ImageFont.truetype("NanumGothic.ttf", 15) # 예: 나눔고딕
    except IOError:
        font = ImageFont.load_default()

    for obj in localized_object_annotations:
        print(f"\n객체 이름: {obj.name}")
        print(f"신뢰도 점수: {obj.score:.2f}")
        print("경계 상자 꼭짓점 (정규화):")

        # 경계 상자 그리기
        box = []
        for vertex in obj.bounding_poly.normalized_vertices:
            # 정규화된 좌표를 실제 픽셀 좌표로 변환
            abs_x = vertex.x * img_width
            abs_y = vertex.y * img_height
            print(f" - ({vertex.x:.4f}, {vertex.y:.4f}) -> ({abs_x:.0f}, {abs_y:.0f})")
            box.append((abs_x, abs_y))
        
        # 객체 이름과 신뢰도 표시
        # PIL의 polygon은 (x1, y1, x2, y2, ...) 형태의 시퀀스를 받거나,
        # ( (x1,y1), (x2,y2), ... ) 형태의 튜플 시퀀스를 받습니다.
        # 여기서는 왼쪽 위 꼭짓점을 기준으로 텍스트를 그립니다.
        draw.polygon(box, outline='lime', width=3)
        text_position_x = box[0][0] + 5 # 약간 안쪽으로
        text_position_y = box[0][1] + 5 # 약간 안쪽으로
        
        # 텍스트 배경 추가 (선택 사항)
        text = f"{obj.name} ({obj.score:.2f})"
        text_bbox = draw.textbbox((text_position_x, text_position_y), text, font=font)
        draw.rectangle(text_bbox, fill="rgba(0,0,0,0.5)") # 반투명 검은색 배경
        draw.text((text_position_x, text_position_y), text, fill='white', font=font)


    # 결과 이미지 저장 또는 표시
    output_image_path = "output_" + os.path.basename(image_path)
    img_to_draw.save(output_image_path)
    print(f"\n결과 이미지가 '{output_image_path}'에 저장되었습니다.")
    # img_to_draw.show() # 이미지를 바로 보고 싶을 때 주석 해제

if __name__ == '__main__':
    # 여기에 스크린샷 이미지 파일 경로를 지정하세요.
    # 예: screenshot_path = "my_screenshot.png"
    #     screenshot_path = "/Users/username/Desktop/test_image.jpg"
    
    screenshot_path = "./resource/images.jpeg" # !!! 경로를 실제 파일 경로로 변경해주세요 !!!

    if not os.path.exists(screenshot_path):
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {screenshot_path}")
        print("스크린샷 이미지 경로를 'screenshot_path' 변수에 올바르게 지정해주세요.")
    else:
        detect_objects_in_screenshot(screenshot_path)