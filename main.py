'''
현재 파일 구조
resource/pass/ : 정상
resource/default/ : 디폴트
resource/defect/ : 이슈

알고리즘
1. 정상, 디폴트 테마 내의 xml을 기준으로 동일한 레이아웃 분류
2. 이슈 내의 xml을 분류 기준에 맞춤
3. 분류된 레이아웃 내 요소를 바탕으로 이슈 확인 - rule base
  a. 라디오 버튼 이슈 확인
  b. 아이콘 정렬 확인
  c. 전화버튼 텍스트 중앙 정렬 확인
  d. 완벽하게 동일한 아이콘 확인
  e. 가독성 확인
'''
from src.classification import ImageXMLClassifier
from src.match import Match
from src.issue.visibility import Visibility
from src.detect import Detect
from src.visualize import Visual
# classifier = ImageXMLClassifier()
# classifier.run_classification() 

target_file = './resource/test/fail2.png'
# match = Match(target_file)
# print(match.select_group())
# result =(29.305655517254444, './output/com.samsung.android.app.dressroom_EditMainActivity/group_0/')

# # 컴포넌트 분리 및 이슈 할당

from src.utils import draw_components

detect = Detect(target_file)
# 정렬 확인
from src.issue.alignment import get_grid
print(get_grid(detect.get_valid_components(filter_type="all")))


# 가시성 확인
# visibility = Visibility(target_file, filter_type="text")
# print(visibility.run_visibility_check())


# 다이얼러 텍스트 확인

# 제미나이
