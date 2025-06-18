import os
from src.issue.icon import Icon
from src.issue.cutoff import Cutoff
from src.utils.detect import Detect
from src.utils.logger import init_logger
from src.issue.visibility import Visibility
from src.issue.alignment import Align

logger = init_logger()

class Layout:
    def __init__(self, file_path: str):
        if file_path.endswith('.png'):
            self.image_path = file_path
            self.xml_path = file_path.replace('.png', '.xml')
        else:
            self.image_path = file_path.replace('.xml', '.png')
            self.xml_path = file_path

        self.detector = Detect(self.image_path)

    def run_layout_check(self):
        classes = self.detector.all_node_classes()

        issues = []

        # 3-1. 가독성 검증 --> src/issue/visibility.py
        if 'android.widget.TextView' in classes:
            logger.info(f"visibility 검증 시작")
            # 만약 텍스트, 아이콘 컴포넌트가 있는 경우
                # 0. text, 아이콘과 배경간 대비가 낮아 가독성이 떨어짐
                # 1. 하이라이트된 항목, 텍스트와 배경간 대비가 낮아 가독성이 떨어짐
                # 2. 상호작용 가능한 요소가 시각적으로 명확히 구분되지 않음
            visibility = Visibility(self.image_path, filter_type="text")
            visibility_issues = visibility.run_visibility_check()
            issues.extend(visibility_issues)
            logger.info(f"visibility 검증 완료")
            

        # 3-2. 아이콘 잘림 이슈 검증 --> src/issue/cutoff.py
        if 'android.widget.RadioButton' in classes:
            # 만약 라디오버튼 컴포넌트가 있는 경우
                # 7. 아이콘의 가장자리가 잘려 보이지 않음
            cutoff = Cutoff(self.image_path)
            issues.extend(cutoff.run_radio_button_check())
            logger.info(f"radio 검증 완료")
            

        # 3-3. 동일 아이콘 검증 --> src/issue/icon.py(고도화 필요)
        icon = Icon(self.image_path)
        icon_issues = icon.run_icon_check()
        issues.extend(icon_issues)
        # 아이콘 컴포넌트를 감지한 경우
            # 4. 컴포넌트 내부 요소의 정렬
            # 5. 동일 계층 요소간의 정렬
            # 6. 텍스트가 할당된 영역을 초과
            # 8. 완벽하게 동일한 아이콘이 있음
        logger.info("icon 검증 완료")

        # 3-4. 정렬 이슈 검증 --> src/issue/alignment.py(고도화 필요)
        align = Align(self.image_path)
        issues.extend(align.run_alignment_check())
        logger.info("alignment 검증 완료")

        return issues

