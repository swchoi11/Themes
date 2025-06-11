import os
import pandas as pd



def compare_xml_image_files(xml_folder, image_folder, output_csv='inspection_dataset.csv'):
    """
    xml 폴더와 image 폴더의 파일명을 비교하여 유효한 평가데이터 셋인지 확인하는 함수
    """

    if not os.path.exists(xml_folder):
        print(f"XML 폴더가 존재하지 않습니다: {xml_folder}")
        return None

    if not os.path.exists(image_folder):
        print(f"Image 폴더가 존재하지 않습니다: {image_folder}")
        return None

    # XML 파일들의 확장자를 제외한 파일명 수집
    xml_files = set()
    for file in os.listdir(xml_folder):
        if file.lower().endswith('.xml'):
            filename_without_ext = os.path.splitext(file)[0]
            xml_files.add(filename_without_ext)

    # Image 파일들의 확장자를 제외한 파일명 수집
    image_files = set()
    image_extensions = {'.jpg', '.jpeg', '.png'}
    for file in os.listdir(image_folder):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in image_extensions:
            filename_without_ext = os.path.splitext(file)[0]
            image_files.add(filename_without_ext)

    all_filenames = xml_files.union(image_files)

    results = []

    for filename in sorted(all_filenames):
        xml_exists = filename in xml_files
        image_exists = filename in image_files

        # 상태 결정
        if xml_exists and image_exists:
            status = True
        elif xml_exists and not image_exists:
            status = False
        elif not xml_exists and image_exists:
            status = False
        else:
            status = "Error"

        results.append({
            'filename': filename,
            'XML': xml_exists,
            'Image': image_exists,
            'Match': status
        })


    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 통계 정보 출력
    print(f"\n=== 파일 매칭 결과 ===")
    print(f"총 고유 파일명: {len(all_filenames)}")
    print(f"XML 파일 수: {len(xml_files)}")
    print(f"Image 파일 수: {len(image_files)}")
    print(f"\n=== 매칭 상태별 통계 ===")
    status_counts = df['Match'].value_counts()
    for status, count in status_counts.items():
        print(f"{status}: {count}개")

    print(f"\n결과가 '{output_csv}' 파일로 저장되었습니다.")

    return df


if __name__ == "__main__":

    xml_folder_path = "../../resource/xml"
    image_folder_path = "../../resource/image"

    compare_xml_image_files(xml_folder_path, image_folder_path)
