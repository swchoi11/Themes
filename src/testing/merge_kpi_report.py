import os
import re
import pandas as pd
import glob


def check_image_processing_status(root_dir='./output'):
    """
    CSV 파일에 등록된 이미지들의 실제 처리 상태를 확인하는 함수
    """

    print("\n" + "=" * 70)
    print("CSV 이미지 목록 vs 실제 처리 상태 확인")

    # 경로 설정
    images_dir = os.path.join(root_dir, 'images')
    not_processed_dir = os.path.join(images_dir, 'not_processed')
    resource_dir = os.path.join(os.path.dirname(root_dir), 'resource')
    excel_dir = os.path.join(root_dir, 'excels', 'final_issue')

    print(f"작업 디렉토리: {os.path.abspath(root_dir)}")
    print(f"   - 처리된 이미지 폴더: {images_dir}")
    print(f"   - 미처리 이미지 폴더: {not_processed_dir}")
    print(f"   - 리소스 폴더: {resource_dir}")
    print(f"   - Excel 결과 폴더: {excel_dir}")

    # CSV 파일 찾기 (vm*_image_list.csv 패턴)
    csv_pattern = os.path.join(resource_dir, '*_image_list.csv')
    csv_file = glob.glob(csv_pattern)[0]

    # 처리된 이미지들 (images 폴더의 직접 하위 .png 파일들)
    processed_images = set()
    if os.path.exists(images_dir):
        png_files = [f for f in os.listdir(images_dir)
                     if f.lower().endswith('.png') and os.path.isfile(os.path.join(images_dir, f))]
        processed_images = set(png_files)

    # 미처리된 이미지들 (not_processed 폴더의 .png 파일들)
    not_processed_images = set()
    if os.path.exists(not_processed_dir):
        png_files = [f for f in os.listdir(not_processed_dir)
                     if f.lower().endswith('.png') and os.path.isfile(os.path.join(not_processed_dir, f))]
        not_processed_images = set(png_files)

    excel_processed_images = set()
    if os.path.exists(excel_dir):
        excel_files = glob.glob(os.path.join(excel_dir, '*.xlsx'))
        print(f"VM리스트 결과 파일:{len(pd.read_csv(csv_file))}개,   Excel 결과 파일: {len(excel_files)}개")

        for excel_file in excel_files:
            try:
                df_excel = pd.read_excel(excel_file)
                if 'filename' in df_excel.columns:
                    filenames = df_excel['filename'].dropna().astype(str).tolist()
                    for filename in filenames:
                        basename = os.path.basename(filename)
                        if basename.lower().endswith('.png'):
                            excel_processed_images.add(basename)
                        elif not basename.lower().endswith('.png'):
                            excel_processed_images.add(basename + '.png')
            except Exception as e:
                print(f"   Excel 파일 읽기 오류 ({os.path.basename(excel_file)}): {e}")

        print(f"   Excel에서 처리된 이미지: {len(excel_processed_images)}개")
    else:
        print(f"   Excel 폴더 없음: {excel_dir}")

    print(f"   전체 실제 파일: {len(processed_images) + len(not_processed_images)}개")

    # 결과를 저장할 리스트
    all_results = []
    overall_stats = {
        'total_in_csv': 0,
        'found_processed': 0,
        'found_not_processed': 0,
        'has_intermediate_result': 0,
        'no_intermediate_result': 0,
        'not_uploaded': 0
    }

    # 각 CSV 파일 처리
    csv_filename = os.path.basename(csv_file)
    vm_name = csv_filename.replace('_image_list.csv', '')

    print(f"\n" + "=" * 50)
    print(f"처리 중: {csv_filename}")
    print(f"VM 이름: {vm_name}")

    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file)
        print(f"CSV 파일 정보:")
        print(f"   - 총 행 수: {len(df)}")
        print(f"   - 컬럼: {list(df.columns)}")

        # 이미지 파일명 컬럼 찾기
        image_column = None
        possible_columns = ['filename', 'image_name', 'image', 'file', 'name', 'image_file']

        for col in possible_columns:
            if col in df.columns:
                image_column = col
                break

        if image_column is None:
            image_column = df.columns[0]
            print(f"   - 이미지 컬럼 자동선택: '{image_column}'")
        else:
            print(f"   - 이미지 컬럼: '{image_column}'")

        # 이미지 목록 추출 및 정규화
        raw_images = df[image_column].dropna().astype(str).tolist()

        # 이미지명 정규화 (공백제거, .png 확장자 추가)
        normalized_images = []
        for img in raw_images:
            img = img.strip()
            if img and img != 'nan':  # 빈 값이나 NaN 제외
                if not img.lower().endswith('.png'):
                    img = img + '.png'
                normalized_images.append(img)

        csv_image_count = len(normalized_images)
        overall_stats['total_in_csv'] += csv_image_count

        print(f"   - CSV 등록 이미지: {csv_image_count}개")

        if csv_image_count == 0:
            print(f"   CSV에 처리할 이미지가 없습니다.")

        # 각 이미지의 처리 상태 확인
        vm_stats = {
            'processed': 0,
            'not_processed': 0,
            'has_intermediate': 0,
            'no_intermediate': 0,
            'not_uploaded': 0
        }

        print(f"\n각 이미지 처리 상태 확인:")

        for i, image_name in enumerate(normalized_images, 1):
            # 처리 상태 판단
            is_processed = 0
            is_not_processed = 0
            status_code = 0  # 0: 업로드안됨, 1: 처리완료, 2: 미처리, 3: 중간산출물있음, 4: 중간산출물없음

            if image_name in not_processed_images:
                # not_processed 폴더에 있음 = 미처리
                is_not_processed = 1
                vm_stats['not_processed'] += 1
                status_msg = "미처리"
                status_code = 2
            elif image_name in excel_processed_images:
                # Excel에 처리 결과가 있음 = processed=1
                is_processed = 1
                vm_stats['processed'] += 1

                if image_name in processed_images:
                    # Excel + images 폴더 둘 다 있음 = 중간산출물 있음
                    status_msg = "중간산출물 있음"
                    status_code = 3
                    vm_stats['has_intermediate'] += 1
                else:
                    # Excel에만 있고 images 폴더에 없음 = 중간산출물 없음
                    status_msg = "중간산출물 없음"
                    status_code = 4
                    vm_stats['no_intermediate'] += 1
            else:
                # Excel에도 없음 = VM에 업로드 안됨
                vm_stats['not_uploaded'] += 1
                status_msg = "VM 업로드 안됨"
                status_code = 0

            # 결과에 추가
            all_results.append({
                'vm': vm_name,
                'image': image_name,
                'processed': is_processed,
                'not_processed': is_not_processed,
                'status_code': status_code,
                'status': status_msg
            })

        # VM별 요약 통계
        print(f"\n{vm_name} 처리 결과:")
        print(f"   - CSV 등록: {csv_image_count}개")
        print(f"   - 처리완료: {vm_stats['processed']}개 ({vm_stats['processed'] / csv_image_count * 100:.1f}%)")
        print(f"   - 미처리: {vm_stats['not_processed']}개 ({vm_stats['not_processed'] / csv_image_count * 100:.1f}%)")
        print(
            f"   - 중간산출물 있음: {vm_stats['has_intermediate']}개 ({vm_stats['has_intermediate'] / csv_image_count * 100:.1f}%)")
        print(
            f"   - 중간산출물 없음: {vm_stats['no_intermediate']}개 ({vm_stats['no_intermediate'] / csv_image_count * 100:.1f}%)")
        print(
            f"   - VM 업로드 안됨: {vm_stats['not_uploaded']}개 ({vm_stats['not_uploaded'] / csv_image_count * 100:.1f}%)")

    except Exception as e:
        print(f"CSV 파일 처리 오류 ({csv_filename}): {e}")
        import traceback
        print(f"   상세 오류: {traceback.format_exc()}")

    # 전체 결과 요약
    print(f"\n" + "=" * 70)
    print(f"전체 처리 결과 요약")
    print(f"=" * 70)
    print(f"CSV 등록 총 이미지: {overall_stats['total_in_csv']}개")
    print(
        f"처리완료: {overall_stats['found_processed']}개 ({overall_stats['found_processed'] / max(1, overall_stats['total_in_csv']) * 100:.1f}%)")
    print(
        f"미처리: {overall_stats['found_not_processed']}개 ({overall_stats['found_not_processed'] / max(1, overall_stats['total_in_csv']) * 100:.1f}%)")
    print(
        f"중간산출물 있음: {overall_stats['has_intermediate_result']}개 ({overall_stats['has_intermediate_result'] / max(1, overall_stats['total_in_csv']) * 100:.1f}%)")
    print(
        f"중간산출물 없음: {overall_stats['no_intermediate_result']}개 ({overall_stats['no_intermediate_result'] / max(1, overall_stats['total_in_csv']) * 100:.1f}%)")
    print(
        f"VM 업로드 안됨: {overall_stats['not_uploaded']}개 ({overall_stats['not_uploaded'] / max(1, overall_stats['total_in_csv']) * 100:.1f}%)")

    # 데이터 검증
    found_total = overall_stats['found_processed'] + overall_stats['found_not_processed']
    print(f"\n데이터 검증:")
    print(f"   - CSV 목록에서 실제 찾은 파일: {found_total}개")
    print(f"   - 실제 폴더의 총 파일: {len(processed_images) + len(not_processed_images)}개")
    print(f"   - Excel에서 처리된 파일: {len(excel_processed_images)}개")

    # 결과 DataFrame 생성
    if all_results:
        result_df = pd.DataFrame(all_results)
        print(f"\n총 {len(all_results)}개 이미지의 처리 상태를 확인했습니다.")
        return result_df
    else:
        print("\n처리할 데이터가 없습니다.")
        return None


def save_results_to_csv(result_df, output_file=None):
    """결과를 CSV 파일로 저장"""
    try:
        # 컬럼 순서 정렬
        column_order = ['vm', 'image', 'processed', 'not_processed', 'status_code', 'status']
        result_df = result_df[column_order]

        # CSV 파일로 저장
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n결과 CSV 저장 완료: {output_file}")

        return True

    except Exception as e:
        print(f"CSV 저장 오류: {e}")
        return False


def inspect_result(rootdir='../../eval'):
    """메인 실행 함수"""
    print("이미지 처리 상태 확인 프로그램 시작")
    print("=" * 70)

    # vm5~vm14 폴더 찾기
    all_results = []
    for i in range(5, 15):  # vm5부터 vm14까지
        vm_name = f"vm{i}"
        vm_path = os.path.join(rootdir, vm_name)
        output_path = os.path.join(vm_path, 'output')

        print(f"\n" + "=" * 70)
        print(f"경로: {output_path}")

        # 각 VM의 output 폴더에 대해 처리 상태 확인
        result_df = check_image_processing_status(output_path)

        if result_df is not None and len(result_df) > 0:
            all_results.append(result_df)
            print(f"{vm_name} 처리 완료: {len(result_df)}개 이미지 확인됨")
        else:
            print(f"{vm_name} 처리 결과 없음")

    # 전체 결과 통합
    if all_results:
        # 모든 결과를 하나로 합치기
        combined_df = pd.concat(all_results, ignore_index=True)

        print(f"\n" + "=" * 70)
        print(f"전체 통합 결과")
        print("=" * 70)
        print(f"총 처리된 이미지: {len(combined_df)}개")
        print(f"총 VM 수: {len(combined_df['vm'].unique())}")

        # VM별 요약
        vm_summary = combined_df.groupby('vm').agg({
            'processed': 'sum',
            'not_processed': 'sum',
            'image': 'count'
        }).rename(columns={'image': 'total'})

        print(f"\nVM별 처리 현황:")
        for vm_name, stats in vm_summary.iterrows():
            total = stats['total']
            processed = stats['processed']
            not_processed = stats['not_processed']
            missing = total - processed - not_processed
            rate = (processed / total * 100) if total > 0 else 0

            print(f"   {vm_name}: 총 {total}개 | 완료 {processed}개({rate:.1f}%) | 미처리 {not_processed}개 | 기타 {missing}개")

        # 결과 저장
        output_filename = f'{BASE_ROOTDIR}/데이터처리 현황.csv'
        save_results_to_csv(combined_df, output_filename)

    else:
        print(f"\n전체 처리 실패 또는 데이터 없음")


def kpi_result(rootdir='../../eval', filename='result-20250613.xlsx'):
    # vm5~vm14 폴더 찾기
    all_results = []
    for i in range(5, 15):  # vm5부터 vm14까지
        vm_name = f"vm{i}"
        vm_path = os.path.join(rootdir, vm_name)
        output_path = os.path.join(vm_path, 'output/excels/final_issue')
        excel_file = os.path.join(output_path, filename)

        # 파일이 존재하는지 확인
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
            df.insert(0, 'vm_name', vm_name)

            all_results.append(df)
        else:
            print(f"{excel_file} 파일을 찾을 수 없습니다")

    # 모든 데이터프레임 병합
    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        print(f"\n총 {len(merged_df)}개 행이 병합 되었습니다")

        # 각 vm별 데이터 개수 확인
        print("\nvm별 데이터 개수:")
        print(merged_df['vm_name'].value_counts().sort_index())

        return merged_df
    else:
        print("병합할 데이터가 없습니다")
        return None


def find_fail_version(result_filename):

    matching_rows = merged_data[merged_data['filename'].str.contains(result_filename, regex=False, na=False)]

    if not matching_rows.empty:
        # 매칭되는 항목이 있으면 정답 데이터의 filename 반환 (Fail_ 포함)
        return matching_rows.iloc[0]['filename']
    else:
        # 매칭되는 항목이 없으면 원본 그대로
        return result_filename


def extract_fail_codes(filename):
    if pd.isna(filename):
        return pd.Series([None, None, None])

    # 대괄호 안의 내용을 모두 찾기 (숫자와 문자 모두 포함)
    matches = re.findall(r'\[([^\]]+)\]', str(filename))

    if not matches:
        return pd.Series([None, None, None])

    # 각 매치된 값을 처리
    results = []
    for match in matches:
        # 숫자만 있는 경우
        if match.isdigit():
            code = match.zfill(3)  # 3자리로 패딩
            itemname = int(code[0])
            location = int(code[1])
            desc = int(code[2])
            results.append([itemname, location, desc])

        # 문자가 포함된 경우 - 각 문자/숫자를 개별 처리
        else:
            # 문자열을 문자 단위로 분리하여 처리
            chars = list(match)

            # 3개 값으로 만들기 (부족하면 0으로 채우고, 넘치면 처음 3개만)
            while len(chars) < 3:
                chars.append('0')
            chars = chars[:3]

            # 각 문자를 숫자로 변환 (문자는 ASCII 값 또는 특별한 규칙 적용)
            processed_chars = []
            for char in chars:
                if char.isdigit():
                    processed_chars.append(int(char))
                else:
                    processed_chars.append(ord(char.lower()) - ord('a') + 1)

            results.append(processed_chars)

    # 첫 번째 매치된 결과 반환
    if results:
        return pd.Series(results[0])
    else:
        return pd.Series([None, None, None])


def compare_desc_match(row):
    gt_desc = row['GT_Desc']
    desc_id = row['description_id']

    # 둘 다 NaN인 경우 (정상)
    if pd.isna(gt_desc) and pd.isna(desc_id):
        return True

    # 하나만 NaN인 경우
    if pd.isna(gt_desc) or pd.isna(desc_id):
        return False

    # 둘 다 값이 있는 경우 - 문자열로 변환해서 비교
    try:
        return str(gt_desc).strip() == str(desc_id).strip()
    except:
        return False


def compare_location_match(row):
    gt_location = row['GT_Location']
    location_id = row['location_id']

    # 둘 다 NaN인 경우 (정상)
    if pd.isna(gt_location) and pd.isna(location_id):
        return True

    # 하나만 NaN인 경우
    if pd.isna(gt_location) or pd.isna(location_id):
        return False

    # 둘 다 값이 있는 경우 - 문자열로 변환해서 비교
    try:
        return str(gt_location).strip() == str(location_id).strip()
    except:
        return False


if __name__ == "__main__":

    BASE_ROOTDIR = '../../eval'

    # inspect_result(rootdir=BASE_ROOTDIR)
    # merged_result = kpi_result(rootdir=BASE_ROOTDIR)
    # merged_result.to_excel(f'{BASE_ROOTDIR}/merged_raw_results.xlsx', index=False)

    merged_result = pd.read_excel(f'{BASE_ROOTDIR}/merged_raw_results.xlsx')
    merged_data = pd.read_excel(f'{BASE_ROOTDIR}/merged_raw_data.xlsx')
    merged_result['filename'] = merged_result['filename'].apply(os.path.basename)
    merged_result['bbox'] = merged_result['bbox'].replace('[1.0, 1.0, 1.0, 1.0]', '[]')

    merged_result['filename'] = merged_result['filename'].apply(find_fail_version)
    merged_result = merged_result[~merged_result['issue_type'].isin(['no_xml', 'not_processed'])]
    merged_result.to_excel(f'{BASE_ROOTDIR}/merged_results.xlsx')

    merged_result[['GT_ItemName', 'GT_Location', 'GT_Desc']] = merged_result['filename'].apply(extract_fail_codes)
    merged_result['GroundTrue'] = merged_result['GT_Desc'].notna()
    merged_result['Predict'] = merged_result['description_id'].notna()

    merged_result['MATCH'] = merged_result['GroundTrue'] == merged_result['Predict']

    merged_result['DescMatch'] = merged_result.apply(compare_desc_match, axis=1)
    merged_result['LocationMatch'] = merged_result.apply(compare_location_match, axis=1)

    print(f"\nGroundTrue: {merged_result['GroundTrue'].sum()}/{len(merged_result)}")
    print(f"Predict: {merged_result['Predict'].sum()}/{len(merged_result)}")
    print(f"DescMatch: {merged_result['DescMatch'].sum()}/{len(merged_result)}")
    print(f"LocationMatch: {merged_result['LocationMatch'].sum()}/{len(merged_result)}")
    # 4. FinalResult: DescMatch AND LocationMatch
    merged_result['PredResult'] = merged_result['DescMatch'] & merged_result['LocationMatch']

    merged_result.to_excel(f'{BASE_ROOTDIR}/merged_final_raw_results.xlsx')

    col_simple = {
        'FileName': 'filename',
        'GroundTruth': 'GroundTrue',
        'Predict': 'Predict',
        'MATCH': 'MATCH',
        'Score': 'score',
        'ItemName': 'ui_component_id',
        'Location': 'location_id',
        'Desc': 'description_type',
        'Reason': 'description'
    }

    existing_columns = [col for col in col_simple.values() if col in merged_result.columns]
    result_df = merged_result[existing_columns].copy()
    rename_dict = {v: k for k, v in col_simple.items() if v in existing_columns}
    result_df = result_df.rename(columns=rename_dict)
    result_df.to_excel(f'{BASE_ROOTDIR}/merged_final_simple_results.xlsx', index=False)

    col_detail = {
        'FileName': 'filename',
        'GroundTruth': 'GroundTrue',
        'Predict': 'Predict',
        'MATCH': 'MATCH',
        'DetailMatch': 'PredResult',
        'Score': 'score',
        'ItemName': 'ui_component_id',
        'Location': 'location_id',
        'Desc': 'description_id',
        'Reason': 'description',
        'GT_ItemName':'GT_ItemName',
        'GT_Location': 'GT_Location',
        'GT_Desc': 'GT_Desc',
        'DescMatch': 'DescMatch',
        'LocationMatch': 'LocationMatch',
    }

    existing_columns = [col for col in col_detail.values() if col in merged_result.columns]
    result_df = merged_result[existing_columns].copy()
    rename_dict = {v: k for k, v in col_detail.items() if v in existing_columns}
    result_df = result_df.rename(columns=rename_dict)
    result_df.to_excel(f'{BASE_ROOTDIR}/merged_final_detail_results.xlsx', index=False)