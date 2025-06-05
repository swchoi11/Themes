import json
import glob
from tqdm import tqdm

from src.layout import Layout
from src.gemini import Gemini

# # classifier = ImageXMLClassifier()
# # classifier.run_classification() 

# for test_image in tqdm(glob.glob('./resource/defect/*/*.png')):  
#     layout = Layout(test_image)
#     issues = layout.run_layout_check()

#     gemini = Gemini(test_image)
#     result = gemini.detect_all_issues()

#     all_issues = issues + result

#     print(all_issues)
    
    # with open('./output/result-0605.json', 'a') as f:
    #     # 안전한 JSON 변환
    #     json_results = []
    #     for issue in all_issues:
    #         if hasattr(issue, 'model_dump'):
    #             json_results.append(issue.model_dump())
    #         else:
    #             print(f"⚠️ model_dump 메서드가 없는 항목 발견: {type(issue)} - {issue}")
    #             # 튜플이나 다른 타입인 경우 건너뛰기
    #             continue
        
    #     json.dump(json_results, f, ensure_ascii=False, indent=2)

from src.issue.visibility import Visibility
from src.utils.detect import Detect
from src.utils.logger import init_logger
from src.issue.alignment import get_dial_alignment
from src.issue.cutoff import Cutoff
from src.issue.icon import Icon

logger = init_logger()

image_path = './resource/test/fail1.png'
detector = Detect(image_path)
classes = detector.all_node_classes()
issues = []

# if 'android.widget.DialerButton' in classes:
#     logger.info(f"dial 정렬 검증 시작")
#     # 만약 전화 버튼 컴포넌트가 있는 경우
#         # 3. 전화버튼이 정렬 안됨
#     get_dial_alignment(image_path)
#     logger.info(f"dial 정렬 검증 완료")


# logger.info("dial: 해당 없음")

if 'android.widget.RadioButton' in classes:
    logger.info(f"radio 검증 시작")
    # 만약 라디오버튼 컴포넌트가 있는 경우
        # 7. 아이콘의 가장자리가 잘려 보이지 않음
    cutoff = Cutoff(image_path)
    issues.extend(cutoff.run_radio_button_check())
    print(issues)
    logger.info(f"radio 검증 완료")
logger.info("radio: 해당 없음")

# icon = Icon(image_path)
# icon_issues = icon.run_icon_check()
# issues.extend(icon_issues)
# 아이콘 컴포넌트를 감지한 경우
    # 4. 컴포넌트 내부 요소의 정렬
    # 5. 동일 계층 요소간의 정렬
    # 6. 텍스트가 할당된 영역을 초과
    # 8. 완벽하게 동일한 아이콘이 있음
# print(issues)
# print("icon 검증 완료")

# align = Align(image_path)
# issues.extend(align.run_alignment_check())
# print("align 검증 완료")

