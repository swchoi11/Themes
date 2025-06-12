import os
import re
import glob
import shutil
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple, Dict

from src.utils.utils import calculate_xml_similarity

class ImageXMLClassifier:
    """이미지-XML 파일 쌍을 자동으로 분류하는 클래스"""
    
    def __init__(self, resource_dirs: List[str] = None, output_dir: str = None):
        """
        분류기 초기화
        
        Args:
            resource_dirs: 소스 디렉토리 리스트 (기본값: ['./resource/pass', './resource/default'])
            output_dir: 출력 디렉토리 (기본값: './output')
        """
        self.resource_dirs = resource_dirs or ['./pass', './default']
        self.output_dir = output_dir or './output/classification'
        self.similarity_threshold = 90.0  # XML 유사도 임계값 (%)
    
    def _clean_filename(self, filename: str) -> str:
        """
        파일명에서 숫자와 불필요한 문자를 제거하여 정규화
        
        Args:
            filename: 원본 파일명
            
        Returns:
            정규화된 파일명 (확장자 제외)
        """
        # 숫자 제거
        cleaned = re.sub(r'\d+', '', filename)
        # 연속된 언더스코어 제거
        cleaned = re.sub(r'__+', '_', cleaned)
        # 확장자 제거
        cleaned = os.path.splitext(cleaned)[0]
        # 앞뒤 언더스코어 제거
        cleaned = cleaned.strip('_')
        return cleaned
    
    
    def classify_by_filename(self) -> None:
        """
        1차 분류: 파일명 기반으로 이미지-XML 쌍을 그룹화하여 output 디렉토리로 복사
        
        처리 과정:
        1. resource 디렉토리들을 순회
        2. 파일명을 정규화하여 그룹 생성
        3. theme_id를 접두사로 하여 파일명 중복 방지
        4. output/{cleaned_filename}/ 디렉토리에 파일 복사
        5. 정규화 실패 파일들은 uncategorized 그룹에 수집
        """
        print("1차 분류 시작: 파일명 기반 그룹화")
        
        # 파일명별 그룹 딕셔너리 {cleaned_filename: [(src_path, dst_filename), ...]}
        filename_groups: Dict[str, List[Tuple[str, str]]] = {}
        uncategorized_files: List[Tuple[str, str]] = []  # 정규화 실패 파일들
        failed_copies: List[Tuple[str, str, str]] = []  # 복사 실패한 파일들 (src, dst, error)
        
        # 모든 resource 디렉토리 순회
        for resource_dir in self.resource_dirs:
            if not os.path.exists(resource_dir):
                print(f"경고: 디렉토리가 존재하지 않습니다 - {resource_dir}")
                continue
                
            for theme_id in os.listdir(resource_dir):
                theme_path = os.path.join(resource_dir, theme_id)
                
                if not os.path.isdir(theme_path):
                    continue
                
                # 테마 디렉토리 내 모든 파일 처리
                for file_name in os.listdir(theme_path):
                    file_path = os.path.join(theme_path, file_name)
                    
                    if not os.path.isfile(file_path):
                        continue
                    
                    # 파일명 정규화
                    cleaned_name = self._clean_filename(file_name)
                    
                    # theme_id를 접두사로 하여 파일명 중복 방지
                    dst_filename = f"{theme_id}_{file_name}"
                    
                    if not cleaned_name:  # 정규화 후 빈 문자열인 경우 uncategorized에 추가
                        uncategorized_files.append((file_path, dst_filename))
                        continue
                    
                    # 그룹에 추가
                    if cleaned_name not in filename_groups:
                        filename_groups[cleaned_name] = []
                    
                    filename_groups[cleaned_name].append((file_path, dst_filename))
        
        # 그룹별로 파일 복사
        os.makedirs(self.output_dir, exist_ok=True)
        
        for group_name, file_list in tqdm(filename_groups.items(), desc="파일 복사 중"):
            group_dir = os.path.join(self.output_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
            
            for src_path, dst_filename in file_list:
                dst_path = os.path.join(group_dir, dst_filename)
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    error_msg = f"파일 복사 실패 ({src_path} -> {dst_path}): {e}"
                    print(error_msg)
                    failed_copies.append((src_path, dst_path, str(e)))
        
        # uncategorized 파일들 처리
        if uncategorized_files:
            uncategorized_dir = os.path.join(self.output_dir, "uncategorized")
            os.makedirs(uncategorized_dir, exist_ok=True)
            
            print(f"정규화 실패 파일 {len(uncategorized_files)}개를 uncategorized 그룹에 추가")
            for src_path, dst_filename in tqdm(uncategorized_files, desc="uncategorized 파일 복사 중"):
                dst_path = os.path.join(uncategorized_dir, dst_filename)
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    error_msg = f"파일 복사 실패 ({src_path} -> {dst_path}): {e}"
                    print(error_msg)
                    failed_copies.append((src_path, dst_path, str(e)))
        
        total_groups = len(filename_groups) + (1 if uncategorized_files else 0)
        print(f"1차 분류 완료: {total_groups}개 그룹 생성 (일반: {len(filename_groups)}, uncategorized: {1 if uncategorized_files else 0})")
        
        # 복사 실패한 파일들이 있으면 보고
        if failed_copies:
            print(f"⚠️ 복사 실패한 파일 {len(failed_copies)}개:")
            for src, dst, error in failed_copies:
                print(f"  {os.path.basename(src)} - {error}")
    
    def _group_by_xml_similarity(self, xml_files: List[str], threshold: float) -> List[List[str]]:
        """
        XML 파일들을 유사도 기반으로 그룹화
        
        Args:
            xml_files: XML 파일 경로 리스트
            threshold: 유사도 임계값 (%)
            
        Returns:
            그룹화된 XML 파일 경로 리스트의 리스트
        """
        if not xml_files:
            return []
        
        groups = []
        assigned = [False] * len(xml_files)
        
        for i, xml_file1 in enumerate(tqdm(xml_files, desc="XML 유사도 분석 중")):
            if assigned[i]:
                continue
            
            # 새 그룹 시작
            current_group = [xml_files[i]]
            assigned[i] = True
            
            # 나머지 파일들과 비교
            comparison_pairs = [
                (xml_file1, xml_files[j]) 
                for j in range(i + 1, len(xml_files)) 
                if not assigned[j]
            ]
            
            if comparison_pairs:
                # 병렬 처리로 유사도 계산
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    similarity_results = list(executor.map(calculate_xml_similarity, comparison_pairs))
                
                # 임계값 이상의 유사도를 가진 파일들을 같은 그룹에 추가
                for _, xml_file2, similarity in similarity_results:
                    if similarity >= threshold:
                        j = xml_files.index(xml_file2)
                        if not assigned[j]:  # 이미 할당되지 않은 경우만
                            current_group.append(xml_files[j])
                            assigned[j] = True
            
            groups.append(current_group)
        
        return groups
    
    def classify_by_xml_similarity(self) -> None:
        """
        2차 분류: 각 1차 그룹 내에서 XML 유사도 기반으로 세부 그룹화
        
        처리 과정:
        1. output 디렉토리의 각 그룹 순회
        2. XML 파일들의 유사도 계산
        3. 유사도 임계값 이상인 파일들을 같은 서브그룹으로 분류
        4. 이미지-XML 쌍을 group_{idx} 디렉토리로 이동
        """
        print("2차 분류 시작: XML 유사도 기반 세부 그룹화")
        
        if not os.path.exists(self.output_dir):
            print("출력 디렉토리가 존재하지 않습니다. 먼저 1차 분류를 실행하세요.")
            return
        
        primary_groups = [d for d in os.listdir(self.output_dir) 
                         if os.path.isdir(os.path.join(self.output_dir, d))]
        
        for group_name in tqdm(primary_groups, desc="그룹별 2차 분류 진행"):
            group_dir = os.path.join(self.output_dir, group_name)
            
            # 그룹 내 XML 파일 찾기
            xml_files = [
                os.path.join(group_dir, f) 
                for f in os.listdir(group_dir) 
                if f.endswith('.xml') and os.path.isfile(os.path.join(group_dir, f))
            ]
            
            if len(xml_files) <= 1:
                continue  # XML 파일이 1개 이하면 추가 분류 불필요
            
            # XML 유사도 기반 그룹화
            similarity_groups = self._group_by_xml_similarity(xml_files, self.similarity_threshold)
            
            # 서브그룹별로 파일 이동
            for group_idx, xml_group in enumerate(similarity_groups):
                if len(xml_group) == 1 and len(similarity_groups) == 1:
                    continue  # 단일 그룹인 경우 이동하지 않음
                
                # 서브그룹 디렉토리 생성
                subgroup_dir = os.path.join(group_dir, f"group_{group_idx}")
                os.makedirs(subgroup_dir, exist_ok=True)
                
                # XML과 대응하는 이미지 파일 이동
                for xml_path in xml_group:
                    xml_filename = os.path.basename(xml_path)
                    
                    # XML 파일 이동
                    new_xml_path = os.path.join(subgroup_dir, xml_filename)
                    shutil.move(xml_path, new_xml_path)
                    
                    # 대응하는 이미지 파일 찾기 및 이동
                    base_name = os.path.splitext(xml_filename)[0]
                    for ext in ['.png', '.jpg', '.jpeg']:
                        image_filename = base_name + ext
                        image_path = os.path.join(group_dir, image_filename)
                        
                        if os.path.exists(image_path):
                            new_image_path = os.path.join(subgroup_dir, image_filename)
                            shutil.move(image_path, new_image_path)
                            break
        
        print("2차 분류 완료")
    
    def run_classification(self) -> None:
        """
        전체 분류 프로세스 실행
        
        1. 파일명 기반 1차 분류
        2. XML 유사도 기반 2차 분류
        """
        print("=== 이미지-XML 파일 자동 분류 시작 ===")
        print(f"소스 디렉토리: {self.resource_dirs}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"XML 유사도 임계값: {self.similarity_threshold}%")
        print()
        
        # 1차 분류: 파일명 기반
        self.classify_by_filename()
        print()
        
        # 2차 분류: XML 유사도 기반
        self.classify_by_xml_similarity()
        print()
        
        print("=== 분류 완료 ===")

    def check_classification(self):
        total_pics_count, total_xml_count = 0, 0

        # 각 resource 디렉토리별로 파일 수 계산
        for dir in self.resource_dirs:
            if not os.path.exists(dir):
                print(f"경고: 디렉토리가 존재하지 않습니다 - {dir}")
                continue
                
            dir_pics = glob.glob(os.path.join(dir, '*/*.png'))
            dir_xmls = glob.glob(os.path.join(dir, '*/*.xml'))
            
            total_pics_count += len(dir_pics)
            total_xml_count += len(dir_xmls)
            
            print(f"{dir} 폴더에서 이미지 파일 {len(dir_pics)}개, XML 파일 {len(dir_xmls)}개 확인")

        print(f"전체 원본 파일: 이미지 {total_pics_count}개, XML {total_xml_count}개")
        print()

        # 출력 디렉토리가 존재하지 않으면 종료
        if not os.path.exists(self.output_dir):
            print(f"출력 디렉토리가 존재하지 않습니다: {self.output_dir}")
            return

        # 1차 분류된 파일들 (2단계 깊이: output_dir/group_name/파일명)
        moved_pics_level2 = glob.glob(os.path.join(self.output_dir, '*/*.png'))
        moved_xmls_level2 = glob.glob(os.path.join(self.output_dir, '*/*.xml'))
        
        # 2차 분류된 파일들 (3단계 깊이: output_dir/group_name/group_idx/파일명)
        moved_pics_level3 = glob.glob(os.path.join(self.output_dir, '*/*/*.png'))
        moved_xmls_level3 = glob.glob(os.path.join(self.output_dir, '*/*/*.xml'))
        
        # 전체 이동된 파일 수
        total_moved_pics = len(moved_pics_level2) + len(moved_pics_level3)
        total_moved_xmls = len(moved_xmls_level2) + len(moved_xmls_level3)
        
        print(f"이동된 이미지 파일: {len(moved_pics_level2)}개(1차분류) + {len(moved_pics_level3)}개(2차분류) = {total_moved_pics}개")
        print(f"이동된 XML 파일: {len(moved_xmls_level2)}개(1차분류) + {len(moved_xmls_level3)}개(2차분류) = {total_moved_xmls}개")
        
        # 누락된 파일 확인
        missing_pics = total_pics_count - total_moved_pics
        missing_xmls = total_xml_count - total_moved_xmls
        
        if missing_pics > 0:
            print(f"⚠️ 누락된 이미지 파일: {missing_pics}개")
        if missing_xmls > 0:
            print(f"⚠️ 누락된 XML 파일: {missing_xmls}개")
        else:
            print("✅ 모든 파일이 성공적으로 분류되었습니다!")
        
        # 누락된 파일이 있으면 상세 분석 실행
        if missing_pics > 0 or missing_xmls > 0:
            print()
            self.find_missing_files()

    def find_missing_files(self):
        """
        누락된 파일들을 찾아서 출력하는 디버깅 메서드
        """
        print("=== 누락된 파일 분석 ===")
        
        # 원본 파일들 수집
        all_original_files = set()
        for dir in self.resource_dirs:
            if not os.path.exists(dir):
                continue
            original_files = glob.glob(os.path.join(dir, '*/*'))
            for file_path in original_files:
                if os.path.isfile(file_path) and (file_path.endswith('.png') or file_path.endswith('.xml')):
                    # 상대 경로를 기준으로 파일명 생성 (theme_id_원본파일명)
                    parts = file_path.split(os.sep)
                    theme_id = parts[-2]  # 테마 ID
                    filename = parts[-1]   # 파일명
                    expected_moved_name = f"{theme_id}_{filename}"
                    all_original_files.add(expected_moved_name)
        
        # 이동된 파일들 수집
        moved_files = set()
        if os.path.exists(self.output_dir):
            # 1차 분류된 파일들
            for file_path in glob.glob(os.path.join(self.output_dir, '*/*')):
                if os.path.isfile(file_path):
                    moved_files.add(os.path.basename(file_path))
            
            # 2차 분류된 파일들
            for file_path in glob.glob(os.path.join(self.output_dir, '*/*/*')):
                if os.path.isfile(file_path):
                    moved_files.add(os.path.basename(file_path))
        
        # 누락된 파일들 찾기
        missing_files = all_original_files - moved_files
        
        if missing_files:
            print(f"누락된 파일 {len(missing_files)}개:")
            for file in sorted(missing_files):
                if file.endswith('.png'):
                    print(f"  🖼️  {file}")
                elif file.endswith('.xml'):
                    print(f"  📄 {file}")
        else:
            print("✅ 모든 파일이 성공적으로 이동되었습니다!")
        
        print()



# 사용 예시
if __name__ == "__main__":
    classifier = ImageXMLClassifier()
    classifier.run_classification()











