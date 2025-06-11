from typing import List, Dict
from utils.schemas import Issue
import os
from src.gemini import Gemini


class EvalKPI:
    """ PoC 평가 기준 정의"""

    # 1. UI 구성 요소 12가지
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

    # 2. 9 분위 영역
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
        'B': "콘텐츠와 화면 비율이 맞지 않아 불필요한 여백이 많이 발생함",
        'C': '화면에서 UI / UX 이슈가 발견되지 않음(정상)'
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
    def select_final_priority_issue(cls, issues: List[Issue], image_path: str, layout_data: Dict = None) -> Issue:
        """
        Gemini를 사용하여 여러 이슈 중 최우선순위 1개 선별
        정상으로 판단되면 정상 이슈 반환

        Args:
            issues: 검출된 이슈 리스트
            image_path: 이미지 파일 경로
            layout_data: 레이아웃 JSON 데이터

        Returns:
            최우선 이슈 1개 또는 정상 이슈
        """
        if not issues:
            # 이슈가 없으면 정상 이슈 생성
            return cls._create_normal_issue(image_path, layout_data)

        if len(issues) == 1:
            # 이슈가 1개면 해당 이슈가 정상인지 검증
            single_issue = issues[0]
            if cls._verify_single_issue_with_gemini(single_issue, image_path):
                return single_issue
            else:
                # Gemini가 정상으로 판단하면 정상 이슈 반환
                return cls._create_normal_issue(image_path, layout_data)

        # 여러 이슈 중 최우선 선별
        try:
            gemini = Gemini()

            # 이슈 컨텍스트 구성
            issues_context = cls._build_priority_context(issues, layout_data)

            prompt = f"""
            다음은 UI 화면에서 검출된 {len(issues)}개의 이슈들입니다.
            이 중에서 사용자 경험에 가장 심각한 영향을 미치는 **단 1개의 최우선 이슈**를 선별하거나, 
            모든 이슈가 실제로는 문제가 없다면 **"정상"**으로 판단해주세요.
            
            **중요한 판단 기준:**
            1. 실제로 사용자가 기능을 사용하기 어려운가?
            2. 중요한 정보를 읽기 어려운가?
            3. 접근성에 문제가 있는가?
            4. 전체적인 사용성에 영향을 미치는가?
            
            검출된 이슈들:
            {issues_context}
            
            **응답 형식:**
            만약 실제 문제가 있다면:
            "선별된 이슈: [component_id]
            선별 이유: [구체적인 이유]"
            
            만약 모든 이슈가 실제로는 문제없다면:
            "선별된 이슈: NORMAL
            선별 이유: 검출된 이슈들이 실제로는 사용성에 문제가 없어 정상으로 판단됩니다."
            """

            # Gemini API 호출
            gemini_response = gemini.generate_response(
                prompt=prompt,
                image=image_path,
                text="",
                model="gemini-2.5-flash-preview-05-20"
            )

            print(f"Gemini 최우선 이슈 선별 응답: {gemini_response}")

            # 응답 파싱
            selected_issue = cls._parse_gemini_priority_response(gemini_response, issues, image_path, layout_data)

            return selected_issue

        except Exception as e:
            print(f"Gemini 최우선 이슈 선별 중 오류: {e}")
            # 오류 시 첫 번째 이슈 반환
            if issues:
                return issues[0]
            else:
                return cls._create_normal_issue(image_path, layout_data)

    @classmethod
    def _verify_single_issue_with_gemini(cls, issue: Issue, image_path: str,) -> bool:
        """단일 이슈가 실제 문제인지 Gemini로 검증"""
        try:
            gemini = Gemini()

            issue_description = cls.DESCRIPTION.get(issue.issue_type, "Unknown issue")

            prompt = f"""
            다음 UI 화면에서 검출된 이슈가 실제로 사용자 경험에 문제가 되는지 판단해주세요.
            
            검출된 이슈:
            {{
                "issues": [
                    {{
                        "issue_type": "{issue.issue_type}",
                        "component_id": "{issue.component_id}",
                        "component_type": "{issue.component_type}",
                        "ui_component_id": "{issue.ui_component_id}",
                        "ui_component_type": "{issue.ui_component_type}",
                        "score": "{issue.score}",
                        "location_id": "{issue.location_id}",
                        "location_type": "{issue.location_type}",
                        "bbox": {issue.bbox},
                        "description_id": "{issue.description_id}",
                        "description_type": "{issue_description}",
                        "ai_description": "{issue.ai_description}"
                    }}
                ]
            }}
            
            **질문: 이 이슈가 실제로 사용자에게 문제가 됩니까?**
            
            다음 중 하나로 응답해주세요:
            1. "YES" - 실제 문제가 있어 수정이 필요합니다
            2. "NO" - 검출되었지만 실제로는 문제가 없습니다
            
            응답: 
            """

            gemini_response = gemini.generate_response(
                prompt=prompt,
                image=image_path,
                text=""
            )

            response_text = str(gemini_response).upper()

            # YES/NO 판단
            if "YES" in response_text and "NO" not in response_text:
                return True  # 실제 이슈
            elif "NO" in response_text:
                return False  # 정상
            else:
                # 모호한 경우 이슈로 간주
                return True

        except Exception as e:
            print(f"단일 이슈 검증 중 오류: {e}")
            return True  # 오류 시 이슈로 간주

    @classmethod
    def _build_priority_context(cls, issues: List[Issue], layout_data: Dict = None) -> str:
        """이슈들을 Gemini 우선순위 판단용 JSON 컨텍스트로 구성"""

        # 화면 정보
        screen_info = ""
        if layout_data:
            elements_count = len(layout_data.get('skeleton', {}).get('elements', []))
            screen_info = f'"screen_info": "총 {elements_count}개 UI 요소",'

        # 이슈들을 JSON 형태로 구성
        issues_json = []

        for i, issue in enumerate(issues):
            # 이슈 설명
            issue_description = cls.DESCRIPTION.get(issue.issue_type, "Unknown issue")

            issue_json = f'''{{
            "issue_number": {i + 1},
            "issue_type": "{issue.issue_type}",
            "component_id": "{issue.component_id}",
            "component_type": "{issue.component_type}",
            "ui_component_id": "{issue.ui_component_id}",
            "ui_component_type": "{issue.ui_component_type}",
            "score": "{issue.score}",
            "location_id": "{issue.location_id}",
            "location_type": "{issue.location_type}",
            "bbox": {issue.bbox},
            "description_id": "{issue.description_id}",
            "description_type": "{issue_description}",
            "ai_description": "{issue.ai_description}"
        }}'''

            issues_json.append(issue_json)

        # 최종 JSON 구조
        context = f'''{{
            {screen_info}
            "issues": [
                {",".join(issues_json)}
            ]
        }}'''

        return context

    @classmethod
    def _parse_gemini_priority_response(cls, gemini_response, issues: List[Issue],
                                        image_path: str, layout_data: Dict = None) -> Issue:
        """Gemini 응답에서 선별된 이슈 또는 정상 판정 추출"""

        # Gemini가 Issue 객체를 직접 반환한 경우
        if isinstance(gemini_response, Issue):
            print(f"Gemini가 Issue 객체 직접 반환: {gemini_response.component_id}")

            # 반환된 Issue가 원본 issues 리스트에 있는지 확인
            for issue in issues:
                if issue.component_id == gemini_response.component_id:
                    # 원본 이슈에 Gemini 분석 결과 추가
                    issue.ai_description = f"[최우선 이슈] Gemini가 선별한 최우선 이슈입니다."
                    return issue

            # 원본에서 찾지 못한 경우 Gemini가 반환한 이슈 그대로 사용
            gemini_response.ai_description = f"[최우선 이슈] {gemini_response.ai_description}"
            return gemini_response

        # Gemini가 문자열 응답을 한 경우
        if not gemini_response:
            return issues[0] if issues else cls._create_normal_issue(image_path, layout_data)

        response_text = str(gemini_response).upper()

        # 정상 판정 확인
        normal_keywords = ['NORMAL', '정상', 'NO ISSUE', 'NO PROBLEM', 'ALL NORMAL']
        if any(keyword in response_text for keyword in normal_keywords):
            normal_issue = cls._create_normal_issue(image_path, layout_data)
            normal_issue.ai_description = f"[Gemini 정상 판정] {gemini_response}"
            return normal_issue

        # component_id 직접 매칭
        response_lower = str(gemini_response).lower()
        for issue in issues:
            component_id = str(issue.component_id).lower()
            if component_id in response_lower:
                # Gemini 분석 결과 추가
                issue.ai_description = f"[최우선 이슈] {gemini_response}"
                return issue

        # 순서 번호 매칭
        import re

        # "이슈 N" 패턴
        issue_pattern = r'이슈\s*(\d+)'
        matches = re.findall(issue_pattern, response_text)
        if matches:
            try:
                issue_num = int(matches[0]) - 1
                if 0 <= issue_num < len(issues):
                    selected = issues[issue_num]
                    selected.ai_description = f"[최우선 이슈] {gemini_response}"
                    return selected
            except (ValueError, IndexError):
                pass

        # "N번" 패턴
        num_pattern = r'(\d+)번'
        matches = re.findall(num_pattern, response_text)
        if matches:
            try:
                issue_num = int(matches[0]) - 1
                if 0 <= issue_num < len(issues):
                    selected = issues[issue_num]
                    selected.ai_description = f"[최우선 이슈] {gemini_response}"
                    return selected
            except (ValueError, IndexError):
                pass

        # 매칭 실패시 첫 번째 이슈 반환
        if issues:
            first_issue = issues[0]
            first_issue.ai_description = f"[기본 선택] {gemini_response}"
            return first_issue
        else:
            return cls._create_normal_issue(image_path, layout_data)

    @classmethod
    def _create_normal_issue(cls, image_path: str, layout_data: Dict = None) -> Issue:
        """정상 이슈 생성"""

        filename = os.path.basename(image_path) if image_path else "unknown.png"

        # 대표 요소 또는 전체 화면 정보
        if layout_data:
            elements = layout_data.get('skeleton', {}).get('elements', [])
            if elements:
                # 화면 중앙에 가장 가까운 요소 선택
                center_element = cls._get_center_element(elements)
                component_id = center_element.get('id', 'screen_center')
                component_type = center_element.get('type', 'screen')
                bbox = center_element.get('bbox', [0.0, 0.0, 1.0, 1.0])
            else:
                component_id = 'screen_overall'
                component_type = 'screen'
                bbox = [0.0, 0.0, 1.0, 1.0]
        else:
            component_id = 'screen_overall'
            component_type = 'screen'
            bbox = [0.0, 0.0, 1.0, 1.0]

        # 위치 계산
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        if center_y < 0.33:
            if center_x < 0.33:
                location_id = '0'  # TL
            elif center_x < 0.67:
                location_id = '1'  # TC
            else:
                location_id = '2'  # TR
        elif center_y < 0.67:
            if center_x < 0.33:
                location_id = '3'  # ML
            elif center_x < 0.67:
                location_id = '4'  # MC
            else:
                location_id = '5'  # MR
        else:
            if center_x < 0.33:
                location_id = '6'  # BL
            elif center_x < 0.67:
                location_id = '7'  # BC
            else:
                location_id = '8'  # BR

        location_type = cls.LOCATION.get(location_id, 'MC')

        # 정상 이슈 생성
        normal_issue = Issue(
            filename=filename,
            issue_type='C',
            component_id=component_id,
            component_type=component_type,
            ui_component_id='5',  # TextView (기본)
            ui_component_type='TextView',
            score='Pass with no issue',
            location_id=location_id,
            location_type=location_type,
            bbox=bbox,
            description_id='C',
            description_type=cls.DESCRIPTION['C'],
            ai_description='Gemini 분석 결과 해당 화면에서 가시성, 정렬, 잘림, 중복, 심미성 관련 이슈가 발견되지 않았습니다.'
        )

        return normal_issue

    @classmethod
    def _get_center_element(cls, elements: List[Dict]) -> Dict:
        """화면 중앙에 가장 가까운 요소 반환"""
        if not elements:
            return {'id': 'screen_center', 'type': 'screen', 'bbox': [0.0, 0.0, 1.0, 1.0]}

        center_x, center_y = 0.5, 0.5
        min_distance = float('inf')
        center_element = elements[0]

        for element in elements:
            bbox = element.get('bbox', [0, 0, 1, 1])
            if len(bbox) >= 4:
                elem_center_x = (bbox[0] + bbox[2]) / 2
                elem_center_y = (bbox[1] + bbox[3]) / 2

                distance = ((elem_center_x - center_x) ** 2 + (elem_center_y - center_y) ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    center_element = element

        return center_element

    @classmethod
    def filter_valid_issues(cls, issues: List[Issue]) -> List[Issue]:
        """유효한 이슈만 필터링"""
        actual_issues = []

        for issue in issues:
            # 정상 이슈 타입(C)은 필터링하지 않음
            if issue.issue_type == 'C':
                actual_issues.append(issue)
                continue

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
        """정상 판정인지 확인 - Gemini 정상 판정 패턴 강화"""
        try:
            # 정상 이슈 타입(C)은 정상 판정하지 않음 (이미 정상 이슈이므로)
            issue_type = str(getattr(issue, 'issue_type', ''))
            if issue_type == 'C':
                return False

            # issue_type 체크
            if issue_type.lower() in ['no issue', 'null', 'none', 'n/a', '', 'normal', 'pass']:
                return True

            # component_id 체크
            component_id = str(getattr(issue, 'component_id', '')).lower()
            if component_id in ['null', 'none', 'n/a', '', 'no_issue', 'normal']:
                return True

            # score 체크 (점수 5 또는 "Pass with no issue")
            score = getattr(issue, 'score', '')
            if isinstance(score, str):
                if score.lower() in ['5', 'pass with no issue', 'pass']:
                    return True
            elif isinstance(score, (int, float)):
                if score >= 5:
                    return True

            # ai_description에서 정상 판정 키워드 확인 (Gemini 패턴 추가)
            ai_description = str(getattr(issue, 'ai_description', '')).lower()

            # 강화된 정상 키워드들
            normal_keywords = [
                'no issue', 'not found', 'no problem', 'no cropping',
                'not detected', 'confirmed that none', 'appears normal',
                'looks normal', 'seems normal', 'is normal', 'are normal',
                'no issues detected', 'no issues found', 'properly displayed',
                'correctly aligned', 'sufficient contrast', 'meets requirements',
                'working correctly', 'functioning properly', 'displayed correctly',
                'visible and clear', 'clearly visible', 'fully visible',
                'no cropping detected', 'not cropped', 'complete and visible',
                'within acceptable range', 'adequate', 'sufficient',
                '문제가 발견되지 않았습니다', '정상적으로 보입니다', '문제없습니다',
                '정상입니다', '적절합니다', '기준을 충족합니다', '양호합니다',
                '충분합니다', '만족스럽습니다', '괜찮습니다', '이상이 없습니다'
            ]

            # 정상을 나타내는 강한 패턴들
            strong_normal_patterns = [
                'no.*issue', 'no.*problem', 'no.*cropping', 'no.*truncation',
                'not.*cropped', 'not.*truncated', 'not.*found', 'not.*detected',
                'appears.*normal', 'looks.*normal', 'seems.*normal',
                'properly.*aligned', 'correctly.*aligned', 'well.*aligned',
                'sufficient.*contrast', 'adequate.*contrast', 'good.*contrast',
                'fully.*visible', 'clearly.*visible', 'completely.*visible',
                '정상.*입니다', '문제.*없습니다', '이상.*없습니다',
                '정상적.*표시', '정상적.*정렬', '적절.*대비'
            ]

            # 정규표현식 패턴 매칭
            import re
            for pattern in strong_normal_patterns:
                if re.search(pattern, ai_description):
                    return True

            # 기본 키워드 매칭
            if any(keyword in ai_description for keyword in normal_keywords):
                return True

            # Gemini 특별 패턴: "I don't see any..." 또는 "I cannot find..."
            gemini_negative_patterns = [
                "i don't see any", "i cannot see any", "i don't find any",
                "i cannot find any", "i don't detect any", "i cannot detect any",
                "there are no", "there is no", "i see no", "i find no"
            ]

            if any(pattern in ai_description for pattern in gemini_negative_patterns):
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
            # 정상 이슈 타입(C)은 실제 이슈로 처리
            if issue.issue_type == 'C':
                return True

            # 먼저 정상 판정인지 확인
            if cls.is_normal_judgment(issue):
                return False

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
                '문제', '오류', '잘못', '부족', '불일치', '잘림', '결함',
                'appears to be cropped', 'seems to be cropped', 'looks cropped',
                'appears cropped', 'is cropped', 'are cropped',
                'poor contrast', 'low contrast', 'insufficient contrast',
                'hard to read', 'difficult to read', 'not clearly visible',
                'alignment issue', 'alignment problem', 'misaligned',
                'not aligned', 'poorly aligned', 'inconsistent alignment'
            ]

            # 기본 키워드 매칭
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

        # 정상 이슈는 별도 처리
        normal_issues = [issue for issue in issues if issue.issue_type == 'C']
        problem_issues = [issue for issue in issues if issue.issue_type != 'C']

        if not problem_issues:
            return normal_issues

        # 타입별 그룹화
        issues_by_type = defaultdict(list)
        for issue in problem_issues:
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

        # 정상 이슈도 포함
        final_issues.extend(normal_issues)

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
            11: cls.SCORE_MAPPING[4],  # 비례 조화
            12: cls.SCORE_MAPPING[5]  # 정상
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