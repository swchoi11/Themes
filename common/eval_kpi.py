from typing import List
from utils.schemas import Issue


class EvalKPI:
    """ PoC 평가 기준 정의"""

    #1. UI 구성 요소 12가지
    UI_COMPONENT = {
        '0': 'Button',          # 클릭 가능한 일반 버튼
        '1': 'ImageView',       # 이미지가 표시된 뷰
        '2': 'RadioButton',     # 단일 선택 가능한 동그란 선택 버튼
        '3': 'CheckBox',        # 다중 선택 가능한 사각형 선택 버튼
        '4': 'EditText',        # 텍스트 입력 필드
        '5': 'TextView',        # 읽기 전용 텍스트
        '6': 'Switch',          # 토글 가능한 스위치
        '7': 'ToggleButton',    # 눌러서 상태 전환이 되는 버튼 형태의 스위치
        '8': 'SeekBar',         # 수평 슬라이더, 값 조절 가능
        '9': 'ProgressBar',     # 로딩 상태 등을 표시하는 진행 바
        'A': 'Spinner',         # 클릭 시 목록이 뜨는 드롭다운 선택 박스
        'B': 'ImageButton'      # 이미지로 된 버튼
    }

    #2. 9 분위 영역
    LOCATION = {
        '0': 'TL',  # Top Left
        '1': 'TC',  # Top Center
        '2': 'TR',  # Top Right
        '3': 'ML',  # Middle Left
        '4': 'MC',  # Middle Center
        '5': 'MR',  # Middle Right
        '6': 'BL',  # Bottom Left
        '7': 'BC',  # Bottom Center
        '8': 'BR'   # Bottom Right
    }

    # 3. Issue Representative Description
    DESCRIPTION = {
        '0': "텍스트, 아이콘과 배경 간 대비가 낮아 가독성이 떨어짐",
        '1': "하이라이트된 항목, 텍스트와 배경 간 대비가 낮아 가독성이 떨어짐",
        '2': "상호작용 가능한 요소가 시각적으로 명확히 구분되지 않음",
        '3': "화면 요소들(아이콘,텍스트, 버튼 등)이 일관된 정렬 기준을 따르지 않음(전화 버튼 Text 배열이 중앙 정렬이 되지 않을 경우 Defect로 인식)",
        '4': "컴포넌트 내부 요소들의 수직/수평 정렬이 균일하지 않음",
        '5': "동일 계층 요소들 간의 정렬 기준점이 서로 다름",
        '6': "텍스트가 할당된 영역을 초과하여 영역 외 텍스트 잘림",
        '7': "아이콘의 가장자리가 보이지 않음거나 잘려보임(이미지 제외)",
        '8': "역할이 다른 기능 요소에 동일한 아이콘 이미지로 중복 존재",
        '9': "달력 아이콘에서 요일 글자가 테두리를 벗어남",
        'A': "앱 내 달력, 시간 아이콘이 status bar 등에 보이는 실제 현재 날짜, 시각과 매칭되지 않음",
        'B': "콘텐츠와 화면 비율이 맞지 않아 불필요한 여백이 많이 발생함"
    }

    # 4. 점수 매핑
    SCORE_MAPPING = {
        1: "Fail with Critical issue",
        2: "Fail with issue",
        3: "Conditional Pass",
        4: "Pass with minor concern",
        5: "Pass with no issue"
    }

    # 5. 점수 우선순위 (낮을수록 심각)
    SCORE_PRIORITY = {
        'Fail with Critical issue': 1,
        'Fail with issue': 2,
        'Conditional Pass': 3,
        'Pass with minor concern': 4,
        'Pass with no issue': 5,
    }

    @classmethod
    def filter_valid_issues(cls, issues: List[Issue]) -> List[Issue]:
        """유효한 이슈만 필터링"""
        actual_issues = []

        for issue in issues:
            if cls.is_normal_judgment(issue):
                print(
                    f"정상 판정 제외: {getattr(issue, 'issue_type', 'unknown')} - {getattr(issue, 'component_id', 'unknown')}")
                continue

            if not cls.is_valid_issue_type(issue):
                print(f"유효하지 않은 이슈 타입 제외: {getattr(issue, 'issue_type', 'unknown')}")
                continue

            if not cls.is_actual_issue(issue):
                print(f"실제 이슈 아님 제외: {getattr(issue, 'component_id', 'unknown')}")
                continue

            actual_issues.append(issue)

        return actual_issues

    @classmethod
    def is_normal_judgment(cls, issue: Issue) -> bool:
        """정상 판정인지 확인"""
        try:
            # issue_type 체크
            issue_type = str(getattr(issue, 'issue_type', '')).lower()
            if issue_type in ['no issue', 'null', 'none', 'n/a', '']:
                return True

            # component_id 체크
            component_id = str(getattr(issue, 'component_id', '')).lower()
            if component_id in ['null', 'none', 'n/a', '']:
                return True

            # score 체크 (점수 5 또는 "Pass with no issue")
            score = getattr(issue, 'score', '')
            if isinstance(score, str):
                if score.lower() in ['5', 'pass with no issue']:
                    return True
            elif isinstance(score, (int, float)):
                if score >= 5:
                    return True

            # ai_description에서 정상 판정 키워드 확인
            ai_description = str(getattr(issue, 'ai_description', '')).lower()
            normal_keywords = [
                'no issue', 'not found', 'no problem', 'no cropping',
                'not detected', 'confirmed that none', 'no calendar icon found',
                'no text truncation', 'no icon cropping', 'were not found',
                '문제가 발견되지 않았습니다', '없습니다', '확인되지 않았습니다',
                'adhering to the specified criteria', 'appear fully visible',
                'no alignment issues', 'properly aligned'
            ]

            if any(keyword in ai_description for keyword in normal_keywords):
                return True

            # bbox가 비어있거나 [0,0,0,0]인 경우
            bbox = getattr(issue, 'bbox', [])
            if not bbox or bbox == [0, 0, 0, 0] or all(coord == 0 for coord in bbox):
                return True

            return False

        except Exception as e:
            print(f"정상 판정 확인 중 오류: {e}")
            return False

    @classmethod
    def is_valid_issue_type(cls, issue: Issue) -> bool:
        """유효한 이슈 타입인지 확인"""
        issue_type = str(getattr(issue, 'issue_type', '')).lower()

        # 무효한 이슈 타입들
        invalid_types = [
            'none', 'null', 'n/a', 'no issue', '',
            'no_issue', 'normal', 'pass', 'ok'
        ]

        return issue_type not in invalid_types

    @classmethod
    def is_actual_issue(cls, issue: Issue) -> bool:
        """실제 이슈인지 추가 검증"""
        try:
            # 심각도가 높은 이슈는 통과
            score = getattr(issue, 'score', '')
            if isinstance(score, str):
                if score in ['Fail with Critical issue', 'Fail with issue']:
                    return True
            elif isinstance(score, (int, float)):
                if score <= 2:
                    return True

            # AI 설명에서 실제 문제인지 확인
            ai_description = str(getattr(issue, 'ai_description', '')).lower()

            # 문제를 나타내는 키워드
            issue_keywords = [
                'fail', 'error', 'problem', 'issue', 'incorrect', 'wrong',
                'misalignment', 'cropped', 'truncated', 'overlap', 'contrast',
                'defect', 'violation', 'inconsistent', 'poor', 'insufficient',
                '문제', '오류', '잘못', '부족', '불일치', '잘림', '결함'
            ]

            # 정상을 나타내는 키워드 (더 강한 가중치)
            normal_keywords = [
                'no issue', 'not found', 'no problem', 'confirmed that none',
                'appear fully visible', 'adhering to', 'not detected',
                'properly aligned', 'no alignment issues', 'no violations',
                '발견되지 않았습니다', '없습니다', '정상', '적절', '올바른'
            ]

            # 정상 키워드가 있으면 제외
            if any(keyword in ai_description for keyword in normal_keywords):
                return False

            # 문제 키워드가 있으면 포함
            if any(keyword in ai_description for keyword in issue_keywords):
                return True

            # 모호한 경우 component_id와 bbox 확인
            component_id = str(getattr(issue, 'component_id', '')).lower()
            bbox = getattr(issue, 'bbox', [])

            if component_id in ['null', 'none', 'n/a'] or not bbox:
                return False

            return True

        except Exception as e:
            print(f"실제 이슈 검증 중 오류: {e}")
            return False

    @classmethod
    def prioritize_issues(cls, issues: List[Issue], max_per_type: int = 3) -> List[Issue]:
        """이슈 우선순위 정렬 및 제한"""
        from collections import defaultdict

        # 타입별 그룹화
        issues_by_type = defaultdict(list)
        for issue in issues:
            issue_type = getattr(issue, 'issue_type', None)
            if issue_type:
                issues_by_type[issue_type].append(issue)

        final_issues = []

        for issue_type, type_issues in issues_by_type.items():
            # 심각도 순으로 정렬
            sorted_issues = sorted(
                type_issues,
                key=lambda x: (
                    cls._get_score_priority(x),
                    -cls._get_visibility_weight(x),
                    cls._get_position_priority(x)
                )
            )

            # 상위 이슈만 선택
            top_issues = sorted_issues[:max_per_type]
            final_issues.extend(top_issues)

        return final_issues

    @classmethod
    def _get_score_priority(cls, issue: Issue) -> int:
        """점수 우선순위 반환 (낮을수록 우선)"""
        try:
            score = getattr(issue, 'score', 'Conditional Pass')
            return cls.SCORE_PRIORITY.get(score, 3)
        except:
            return 3

    @classmethod
    def _get_visibility_weight(cls, issue: Issue) -> float:
        """가시성 가중치 계산"""
        try:
            bbox = getattr(issue, 'bbox', [0, 0, 1, 1])
            if len(bbox) < 4:
                return 0.5

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # 중앙에 가까울수록, 상단에 가까울수록 높은 가중치
            x_weight = 1.0 - abs(center_x - 0.5) * 2
            y_weight = 1.0 - center_y * 0.5

            return (x_weight + y_weight) / 2
        except:
            return 0.5

    @classmethod
    def _get_position_priority(cls, issue: Issue) -> float:
        """위치 우선순위 (좌측 상단부터)"""
        try:
            bbox = getattr(issue, 'bbox', [0, 0, 1, 1])
            if len(bbox) < 4:
                return 1.0
            return bbox[0] + bbox[1] * 0.1  # Y가 더 중요하지만 X도 고려
        except:
            return 1.0

    @classmethod
    def map_score_to_string(cls, score_input) -> str:
        """점수를 표준 문자열로 변환"""
        valid_scores = list(cls.SCORE_MAPPING.values())

        if isinstance(score_input, str) and score_input in valid_scores:
            return score_input

        # 숫자를 문자열로 변환
        if isinstance(score_input, (int, float)):
            if score_input <= 1:
                return cls.SCORE_MAPPING[1]
            elif score_input <= 2:
                return cls.SCORE_MAPPING[2]
            elif score_input <= 3:
                return cls.SCORE_MAPPING[3]
            elif score_input <= 4:
                return cls.SCORE_MAPPING[4]
            else:
                return cls.SCORE_MAPPING[5]

        return cls.SCORE_MAPPING[3]  # 기본값

    @classmethod
    def get_issue_severity_by_type(cls, issue_type: int) -> str:
        """이슈 타입별 기본 심각도 반환"""
        severity_map = {
            0: cls.SCORE_MAPPING[2],  # 텍스트 대비 문제
            1: cls.SCORE_MAPPING[2],  # 하이라이트 대비 문제  
            2: cls.SCORE_MAPPING[3],  # 버튼 시각성
            3: cls.SCORE_MAPPING[4],  # 정렬 일관성
            4: cls.SCORE_MAPPING[4],  # 수직 정렬
            5: cls.SCORE_MAPPING[4],  # 계층 정렬
            6: cls.SCORE_MAPPING[2],  # 텍스트 잘림
            7: cls.SCORE_MAPPING[3],  # 아이콘 잘림
            8: cls.SCORE_MAPPING[3],  # 중복 아이콘
            9: cls.SCORE_MAPPING[4],  # 색상 조화
            10: cls.SCORE_MAPPING[4],  # 공간 활용
            11: cls.SCORE_MAPPING[4]  # 비례 조화
        }

        return severity_map.get(issue_type, cls.SCORE_MAPPING[3])

    @classmethod
    def validate_issue_structure(cls, issue: Issue) -> bool:
        """이슈 구조의 유효성 검증"""
        required_fields = ['issue_type', 'component_id', 'bbox']

        for field in required_fields:
            if not hasattr(issue, field) or getattr(issue, field) is None:
                return False

        # bbox 검증
        bbox = getattr(issue, 'bbox', [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False

        return True