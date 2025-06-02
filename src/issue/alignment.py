def is_centered(line):
    centers = [ ((x1 + x2) // 2) for (x1, y1, x2, y2) in line]
    gaps = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
    return all(gap == gaps[0] for gap in gaps)

def is_left(line):
    lefts = [x1 for (x1, y1, x2, y2) in line]
    gaps = [lefts[i+1] - lefts[i] for i in range(len(lefts)-1)]
    return all(gap == gaps[0] for gap in gaps)

def is_right(line):
    rights = [x2 for (x1, y1, x2, y2) in line]
    gaps = [rights[i+1] - rights[i] for i in range(len(rights)-1)]
    return all(gap == gaps[0] for gap in gaps)

def get_grid(components):
    grid = []
    lines = {}
    for component in components:
        horizontal = [component[1], component[3]]
        if horizontal in grid:
            index = grid.index(horizontal)
            lines[index].append(component)
        else:
            grid.append(horizontal)
            lines[len(grid)-1] = [component]

    for i in range(len(lines)):
        line = lines[i]
        if line[i] is None or len(line) <= 2:
            return True
        elif is_centered(line) or is_left(line) or is_right(line):
            return True
        else:
            return False


# # 정렬탐지1 (전화번호부 중심)

# """
# #alignment-1
# #1. Paddle OCR로 텍스트 감지
# #2. 각 텍스트의 bbox 중심 좌표 계산
# #3. 버튼 그리드 기준 가상 열 중심 좌표 생성
# #4. 텍스트 중심 vs 가상 중심의 편차 비교 
# """

# import cv2
# from paddleocr import PaddleOCR
# import numpy as np

# ocr = PaddleOCR(
#             det_model_dir='./src/weights/en_PP-OCRv3_det_infer',
#             rec_model_dir='./src/weights/en_PP-OCRv3_rec_infer',
#             cls_model_dir='./src/weights/ch_ppocr_mobile_v2.0_cls_infer',
#             lang='en',
#             use_angle_cls=False,
#             use_gpu=False, 
#             show_log=False,
#             )

# image_path = "./resource/sample/default_image_006.jpg"
# image = cv2.imread(image_path)
# h, w = image.shape[:2]

# result = ocr.ocr(image_path)[0]

# # 3등분 기준 중심선
# expected_centers = [
#     w * 1/6,  # 1열 (좌)
#     w * 3/6,  # 2열 (중앙)
#     w * 5/6   # 3열 (우)
# ]

# threshold = 30  # 허용 편차(px)

# misaligned = []

# for line in result:
#     box, (text, conf) = line
#     x_coords = [pt[0] for pt in box]
#     x_center = (min(x_coords) + max(x_coords)) / 2

#     # 어떤 열에 해당하는지 추정
#     col_idx = np.argmin([abs(x_center - cx) for cx in expected_centers])
#     offset = abs(x_center - expected_centers[col_idx])

#     if offset > threshold:
#         misaligned.append({
#             "text": text,
#             "x_center": x_center,
#             "expected": expected_centers[col_idx],
#             "offset": offset
#         })

# # 결과 출력
# for m in misaligned:
#     print(f"'{m['text']}' is misaligned: offset {m['offset']:.1f}px from center {m['expected']:.1f}")

# # 정렬 상태 요약
# if misaligned:
#     print("Alignment-1")  # 하나라도 틀린 게 있다면
# else:
#     print("Pass")  # 모두 정렬 OK
