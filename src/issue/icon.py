import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from src.utils.detect import Detect
from src.utils.model import ResultModel
from src.match import Match
from src.utils.logger import init_logger

logger = init_logger()

class Icon:
    def __init__(self, file_path: str):
        if file_path.endswith('.png'):
            self.image_path = file_path
            self.xml_path = file_path.replace('.png', '.xml')
        else:
            self.image_path = file_path.replace('.xml', '.png')
            self.xml_path = file_path
        
        # Detect 모듈 초기화
        self.detector = Detect(self.image_path)
        
        # Match 모듈 초기화
        self.match = Match(self.image_path)
        
        # 이미지 로드
        self.image = cv2.imread(self.image_path)

    def run_icon_check(self) -> List[ResultModel]:
        issues = []
        logger.info(f"중복 아이콘에 대한 검사 시작")
        
        # 디폴트 이미지를 한 번만 선택 (효율성을 위해)
        default_image = None
        try:
            group_result = self.match.select_group()
            if group_result is not None:
                try:
                    _, selected_group = group_result
                    default_xml_path = self.match.selct_default_image(selected_group)
                    if default_xml_path != "":
                        default_image_path = default_xml_path.replace('.xml', '.png')
                        default_image = cv2.imread(default_image_path)
                        if default_image is not None:
                            logger.info(f"디폴트 이미지 로드 완료: {default_image_path}")
                        else:
                            logger.warning(f"디폴트 이미지 로드 실패: {default_image_path}")
                    else:
                        logger.warning("디폴트 이미지를 찾을 수 없습니다.")
                except (TypeError, ValueError):
                    logger.error("그룹 선택 결과를 언패킹할 수 없습니다.")
            else:
                logger.warning("그룹 선택 결과가 None입니다.")
        except Exception as e:
            logger.error(f"디폴트 이미지 선택 중 오류: {e}")
        
        try:
            # 아이콘 크기 컴포넌트만 추출
            components = self.detector.get_icon_components()
            if len(components) < 2:
                return issues
                        
            # 각 아이콘의 이미지 데이터 추출
            icon_images = []
            
            for i, icon in enumerate(components):
                x1, y1, x2, y2 = icon['bounds']
                icon_img = self.image[y1:y2, x1:x2]
                
                if icon_img.size > 0:
                    icon_images.append({
                        'index': i,
                        'component': icon,
                        'image': icon_img
                    })
            
            # 아이콘 간 유사도 비교
            duplicates = self._find_duplicate_icons(icon_images)

            # 중복 아이콘 이슈 생성
            for duplicate_group in duplicates:
                if len(duplicate_group) > 1:
                    # 중복 그룹의 바운딩박스 정보 수집
                    group_bounds = []
                    group_info = []
                    for icon_idx in duplicate_group:
                        icon = components[icon_idx]
                        bounds = icon['bounds']
                        group_bounds.append(bounds)
                        group_info.append(f"{icon['type']} at {bounds}")
                    
                    # logger.info(f"중복 아이콘 그룹 발견 ({len(duplicate_group)}개):")
                    
                    # 중복된 아이콘들을 imshow로 표시
                    for i, icon_idx in enumerate(duplicate_group):
                        icon = components[icon_idx]
                        x1, y1, x2, y2 = icon['bounds']
                        icon_img = self.image[y1:y2, x1:x2]
                        
                        # if icon_img.size > 0:
                        #     # 아이콘 이미지를 창에 표시
                        #     window_name = f"중복 아이콘 {i+1}/{len(duplicate_group)} - 위치: {icon['bounds']}"
                        #     # cv2.imshow(window_name, icon_img)
                        #     print(f"  - 아이콘 {i+1}: 위치 {icon['bounds']}")
                    
                    # 키 입력 대기 (아무 키나 누르면 다음으로)
                    # print("아무 키나 누르면 계속...")
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    # 전체 그룹에 대해 디폴트 검증 (디폴트 이미지를 매개변수로 전달)
                    all_bounds_str = [str(bounds) for bounds in group_bounds]
                    is_normal_duplicate = self.check_default_duplicate(all_bounds_str, default_image)
                    # logger.info(f"디폴트 검증 결과: {is_normal_duplicate}")
                    
                    # 검증 결과에 따라 이슈 생성
                    if not is_normal_duplicate:
                        for icon_idx in duplicate_group:
                            icon = components[icon_idx]
                                                    
                            # 그룹 내 다른 아이콘들의 위치 정보 포함
                            other_bounds = [str(bounds) for i, bounds in enumerate(group_bounds) if i != duplicate_group.index(icon_idx)]
                            bounds_list_str = ", ".join(other_bounds)
                            
                            issue = ResultModel(
                                filename = self.image_path,
                                issue_type='design',
                                component_id=0,
                                ui_component_id="",
                                ui_component_type="horizontal_section",
                                severity="high",
                                location_id="",
                                location_type="",
                                bbox=icon['bounds'],
                                description_id="8",
                                description_type="역할이 다른 기능 요소에 동일한 아이콘 이미지로 중복 존재",
                                description=f"수평 구간에서 중복 아이콘 탐지: {len(duplicate_group)}개의 동일한 아이콘이 해당 수평 구간에서 발견됨. 중복 아이콘 위치들: [{bounds_list_str}] (디폴트와 다름)",
                                ai_description=""
                            )

                            issues.append(issue)
                            cv2.rectangle(self.image, (icon['bounds'][0], icon['bounds'][1]), (icon['bounds'][2], icon['bounds'][3]), (0, 255, 255), 2)
                            cv2.putText(self.image, f"duplicate", (icon['bounds'][0], icon['bounds'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                        file_name = os.path.basename(self.image_path)
                        cv2.imwrite(f"./output/images/{file_name}", self.image)
                    else:
                        logger.info(f"디폴트 이미지에서도 동일한 중복이 발견되어 정상으로 판정됨")
                        issue = ResultModel(
                            filename = self.image_path,
                            issue_type='design',
                            component_id=0,
                            ui_component_id="",
                            ui_component_type="",
                            severity="high",
                            location_id="",
                            location_type="",
                            bbox=[],
                            description_id="8",
                            description_type="역할이 다른 기능 요소에 동일한 아이콘 이미지로 중복 존재",
                            description="",
                            ai_description=""
                        )
                        issues.append(issue)
        except Exception as e:
            logger.error(f"아이콘 중복 탐지 중 오류: {e}")
        
        return issues
    
    def check_default_duplicate(self, duplicate_bounds: List[str], default_img: Optional[np.ndarray]) -> bool:
        """
        디폴트 이미지의 같은 좌표에서 이미지들이 동일한지 확인
        
        Args:
            duplicate_bounds: 타겟 이미지에서 중복된 아이콘들의 바운딩박스 리스트
            default_img: 미리 로드된 디폴트 이미지 (None일 수 있음)
            
        Returns:
            True: 디폴트에서도 같은 위치의 이미지들이 동일함 (정상)
            False: 디폴트에서는 다른 이미지 (이슈)
        """
        try:
            if default_img is None:
                logger.warning("디폴트 이미지가 제공되지 않았습니다.")
                return False
            
            # print(f"디폴트 이미지에서 같은 좌표의 이미지 비교:")
            
            # 각 바운딩박스에서 디폴트 이미지 추출
            default_crops = []
            for bounds_str in duplicate_bounds:
                # 문자열에서 좌표 추출: "(89, 950, 184, 1045)" → [89, 950, 184, 1045]
                bounds_str = bounds_str.strip("()").replace(" ", "")
                coords = [int(x) for x in bounds_str.split(',')]
                x1, y1, x2, y2 = coords
                
                # 디폴트 이미지에서 같은 좌표로 이미지 추출
                default_crop = default_img[y1:y2, x1:x2]
                
                if default_crop.size > 0:
                    default_crops.append(default_crop)
                    # print(f"  - 좌표 {coords}에서 이미지 추출: {default_crop.shape}")
                else:
                    # print(f"  - 좌표 {coords}에서 이미지 추출 실패")
                    return False
            
            # 추출된 이미지들이 모두 동일한지 확인
            if len(default_crops) < 2:
                return False
            
            # 첫 번째 이미지와 나머지들을 비교
            base_image = default_crops[0]
            
            for i, compare_image in enumerate(default_crops[1:], 1):
                # 이미지 크기가 다르면 리사이즈
                if base_image.shape != compare_image.shape:
                    compare_image = cv2.resize(compare_image, (base_image.shape[1], base_image.shape[0]))
                
                # 픽셀 단위로 비교
                diff = cv2.absdiff(base_image, compare_image)
                similarity = 1.0 - (np.mean(diff) / 255.0)
                
                # print(f"  이미지 0 vs 이미지 {i}: 유사도 {similarity:.3f}")
                
                # 유사도가 0.95 이상이면 동일한 것으로 판정
                if similarity < 0.99:
                    # print(f"  → 디폴트에서 다른 이미지 발견 (이슈)")
                    return False
            
            # print(f"  → 디폴트에서도 모든 이미지가 동일함 (정상)")
            return True
            
        except Exception as e:
            logger.error(f"디폴트 중복 확인 중 오류: {e}")
            return False
    
    def _find_duplicate_icons(self, icon_images: List[Dict]) -> List[List[int]]:
        """
        아이콘 이미지들 간의 유사도를 비교하여 중복을 찾음
        
        Args:
            icon_images: 아이콘 이미지 데이터 리스트
            
        Returns:
            중복 그룹들의 리스트 (각 그룹은 인덱스 리스트)
        """
        if len(icon_images) < 2:
            return []
        
        # 유사도 매트릭스 계산
        similarity_matrix = np.zeros((len(icon_images), len(icon_images)))
        
        for i in range(len(icon_images)):
            for j in range(i + 1, len(icon_images)):
                similarity = self._calculate_icon_similarity(
                    icon_images[i]['image'], 
                    icon_images[j]['image']
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # 유사도 임계값 (0.8 이상이면 중복으로 판정)
        threshold = 0.95
        
        # 중복 그룹 찾기
        duplicate_groups = []
        processed = set()
        
        for i in range(len(icon_images)):
            if i in processed:
                continue
                
            group = [i]
            processed.add(i)
            
            for j in range(i + 1, len(icon_images)):
                if j not in processed and similarity_matrix[i][j] >= threshold:
                    group.append(j)
                    processed.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    def _calculate_icon_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        두 아이콘 이미지 간의 유사도 계산
        
        Args:
            img1: 첫 번째 아이콘 이미지
            img2: 두 번째 아이콘 이미지
            
        Returns:
            유사도 (0.0 ~ 1.0)
        """
        try:
            # 이미지 크기 정규화 (32x32)
            target_size = (32, 32)
            img1_resized = cv2.resize(img1, target_size)
            img2_resized = cv2.resize(img2, target_size)
            
            # 그레이스케일 변환
            if len(img1_resized.shape) == 3:
                img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1_resized
                
            if len(img2_resized.shape) == 3:
                img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2_resized
            
            # 히스토그램 비교
            hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
            hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 구조적 유사도 (SSIM-like 계산)
            # 간단한 픽셀 단위 비교
            diff = cv2.absdiff(img1_gray, img2_gray)
            pixel_similarity = 1.0 - (np.mean(diff) / 255.0)
            
            # 템플릿 매칭
            result = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
            template_similarity = np.max(result)
            
            # 가중 평균으로 최종 유사도 계산
            final_similarity = (
                hist_similarity * 0.3 + 
                pixel_similarity * 0.4 + 
                template_similarity * 0.3
            )
            
            return max(0.0, min(1.0, final_similarity))
            
        except Exception as e:
            print(f"유사도 계산 중 오류: {e}")
            return 0.0

    