import cv2
import numpy as np
from xml.etree import ElementTree as ET
import re
from typing import List, Tuple, Dict, Optional
from src.match import Match
from src.utils import get_bounds
from src.result import ResultModel
from src.detect import Detect


class Align:
    def __init__(self, file_path: str):
        if file_path.endswith('.png'):
            self.image_path = file_path
            self.xml_path = file_path.replace('.png', '.xml')
        else:
            self.image_path = file_path.replace('.xml', '.png')
            self.xml_path = file_path
        
        # Detect 모듈 초기화
        self.detector = Detect(self.image_path)
            
    def run_alignment_check(self) -> List[ResultModel]:
        
        issues = self._self_alignment_check()
        return issues
    
    
    def _self_alignment_check(self) -> List[ResultModel]:
        issues = []
        
        try:
            # Detect 모듈의 get_valid_components 사용하여 유효한 컴포넌트 추출
            components = self.detector.get_valid_components()
            # print(components)
            if not components:
                return issues
            
            # 수평 그리드 생성 및 정렬 분석
            horizontal_groups = self._group_components_by_horizontal_position(components)
            # print(horizontal_groups)
            
            for group in horizontal_groups:
                if len(group) < 2:  # 2개 미만의 컴포넌트는 정렬 분석 불가
                    continue
                
                alignment_result = self._analyze_group_alignment(group)
                if not alignment_result['is_aligned']:
                    # 정렬 이슈 생성
                    for comp in group:
                        issue = ResultModel(
                            image_path=self.image_path,
                            index=comp.get('index', 0),
                            issue_type='alignment',
                            issue_location=list(comp['bounds']),
                            issue_description=f"수평 그룹 정렬 이슈: {alignment_result['description']}"
                        )
                        issues.append(issue)
        
        except Exception as e:
            print(f"자체 정렬 분석 중 오류: {e}")
        
        return issues
    
    
    def _group_components_by_horizontal_position(self, components: List[Dict]) -> List[List[Dict]]:
        # y 좌표 기준으로 정렬
        sorted_components = sorted(components, key=lambda c: c['bounds'][1])
        # print(sorted_components)
        
        groups = []
        current_group = []
        threshold = 20
        
        for comp in sorted_components:
            if not current_group:
                current_group.append(comp)
            else:
                # 현재 그룹의 평균 y 좌표와 비교
                current_y = (comp['bounds'][1] + comp['bounds'][3]) / 2
                group_y_avg = sum((c['bounds'][1] + c['bounds'][3]) / 2 for c in current_group) / len(current_group)
                
                if abs(current_y - group_y_avg) <= threshold:
                    current_group.append(comp)
                else:
                    if len(current_group) >= 2:
                        groups.append(current_group)
                    current_group = [comp]
        
        # 마지막 그룹 추가
        if len(current_group) >= 2:
            groups.append(current_group)
        
        return groups
    
    def _analyze_group_alignment(self, group: List[Dict]) -> Dict:
        
        if len(group) < 2:
            return {'is_aligned': True, 'description': '충분한 컴포넌트 없음'}
        
        # x 좌표 기준으로 정렬
        sorted_group = sorted(group, key=lambda c: c['bounds'][0])
        
        # 수평 그룹에서는 요소들 간의 간격이 일정한지 확인
        spacing_result = self._check_horizontal_spacing(sorted_group)
        
        if spacing_result['is_evenly_spaced']:
            return {'is_aligned': True, 'type': 'even_spacing', 'description': '균등 간격 배치'}
        else:
            return {
                'is_aligned': False,
                'description': f'간격 불균등 - {spacing_result["description"]}'
            }
    
    def _check_horizontal_spacing(self, group: List[Dict]) -> Dict:
        """수평 그룹 내 요소들의 간격이 균등한지 확인"""
        if len(group) < 2:
            return {'is_evenly_spaced': True, 'description': '충분한 컴포넌트 없음'}
        
        threshold = 20  # 허용 편차 (픽셀)
        
        # 요소들 간의 간격 계산 (요소의 중심점 기준)
        centers = [(comp['bounds'][0] + comp['bounds'][2]) / 2 for comp in group]
        gaps = []
        
        for i in range(len(centers) - 1):
            gap = centers[i + 1] - centers[i]
            gaps.append(gap)
        
        if len(gaps) < 2:
            return {'is_evenly_spaced': True, 'description': '간격 확인 불가 (요소 2개)'}
        
        # 평균 간격 계산
        avg_gap = sum(gaps) / len(gaps)
        
        # 각 간격이 평균에서 얼마나 벗어나는지 확인
        max_deviation = max(abs(gap - avg_gap) for gap in gaps)
        
        if max_deviation <= threshold:
            return {
                'is_evenly_spaced': True,
                'description': f'균등 간격 (평균: {avg_gap:.1f}px, 최대편차: {max_deviation:.1f}px)'
            }
        else:
            gap_info = ', '.join([f'{gap:.1f}px' for gap in gaps])
            return {
                'is_evenly_spaced': False,
                'description': f'간격 불균등 (간격들: {gap_info}, 평균: {avg_gap:.1f}px, 최대편차: {max_deviation:.1f}px)'
            }
    
    def _find_best_matching_component(self, target_comp: Dict, default_components: List[Dict]) -> Optional[Dict]:
        """
        타겟 컴포넌트와 가장 유사한 디폴트 컴포넌트 찾기
        
        Args:
            target_comp: 타겟 컴포넌트
            default_components: 디폴트 컴포넌트 리스트
            
        Returns:
            가장 유사한 디폴트 컴포넌트 또는 None
        """
        best_match = None
        best_score = 0
        
        target_x1, target_y1, target_x2, target_y2 = target_comp['bounds']
        target_center_x = (target_x1 + target_x2) / 2
        target_center_y = (target_y1 + target_y2) / 2
        
        for default_comp in default_components:
            # 위치 유사도 계산
            default_x1, default_y1, default_x2, default_y2 = default_comp['bounds']
            default_center_x = (default_x1 + default_x2) / 2
            default_center_y = (default_y1 + default_y2) / 2
            
            distance = np.sqrt((target_center_x - default_center_x)**2 + 
                             (target_center_y - default_center_y)**2)
            
            # 클래스 유사도
            class_match = 1.0 if target_comp['class'] == default_comp['class'] else 0.5
            
            # 종합 점수 (거리가 가까울수록, 클래스가 같을수록 높은 점수)
            score = class_match / (1 + distance / 100)
            
            if score > best_score:
                best_score = score
                best_match = default_comp
        
        return best_match if best_score > 0.1 else None
    
    def _compare_component_alignment(self, target_comp: Dict, default_comp: Dict) -> Optional[ResultModel]:
        """
        타겟 컴포넌트와 디폴트 컴포넌트의 정렬 비교
        
        Args:
            target_comp: 타겟 컴포넌트
            default_comp: 디폴트 컴포넌트
            
        Returns:
            정렬 이슈 또는 None
        """
        threshold = 15  # 허용 편차 (픽셀)
        
        target_x1, target_y1, target_x2, target_y2 = target_comp['bounds']
        default_x1, default_y1, default_x2, default_y2 = default_comp['bounds']
        
        # 중심점 차이 계산
        target_center_x = (target_x1 + target_x2) / 2
        default_center_x = (default_x1 + default_x2) / 2
        
        x_deviation = abs(target_center_x - default_center_x)
        
        if x_deviation > threshold:
            return ResultModel(
                image_path=self.image_path,
                index=target_comp.get('index', 0),
                issue_type='alignment',
                issue_location=list(target_comp['bounds']),
                issue_description=f"디폴트 대비 정렬 이슈: x축 편차 {x_deviation:.1f}px (허용: {threshold}px)"
            )
        
        return None


# 다이얼러 전용 정렬 확인 함수 (기존 코드 유지)
def get_dial_alignment(image_path: str):
    """다이얼러 앱 전용 정렬 확인 함수"""
    try:
        from paddleocr import PaddleOCR
        
        ocr = PaddleOCR(
            det_model_dir='./src/weights/en_PP-OCRv3_det_infer',
            rec_model_dir='./src/weights/en_PP-OCRv3_rec_infer',
            cls_model_dir='./src/weights/ch_ppocr_mobile_v2.0_cls_infer',
            lang='en',
            use_angle_cls=False,
            use_gpu=False, 
            show_log=False,
        )

        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        result = ocr.ocr(image_path)[0]
        if not result:
            return []

        # 3등분 기준 중심선
        expected_centers = [
            w * 1/6,  # 1열 (좌)
            w * 3/6,  # 2열 (중앙)
            w * 5/6   # 3열 (우)
        ]

        threshold = 30  # 허용 편차(px)
        misaligned = []

        for line in result:
            box, (text, conf) = line
            x_coords = [pt[0] for pt in box]
            x_center = (min(x_coords) + max(x_coords)) / 2

            # 어떤 열에 해당하는지 추정
            col_idx = np.argmin([abs(x_center - cx) for cx in expected_centers])
            offset = abs(x_center - expected_centers[col_idx])

            if offset > threshold:
                misaligned.append({
                    "text": text,
                    "x_center": x_center,
                    "expected": expected_centers[col_idx],
                    "offset": offset
                })

        return misaligned
        
    except ImportError:
        print("PaddleOCR를 사용할 수 없습니다. 다이얼러 정렬 확인을 건너뜁니다.")
        return []
    except Exception as e:
        print(f"다이얼러 정렬 확인 중 오류: {e}")
        return []
