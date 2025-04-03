"""import os

# 대상 디렉토리 경로
directory = "F:\didimdol2_label"

def rename_files(directory):
    # 디렉토리 내 파일 목록 가져오기
    files = sorted(os.listdir(directory))  # 정렬하여 파일 순서를 유지
    
    # 파일명 변경
    for index, filename in enumerate(files):
        # 기존 파일 경로
        old_path = os.path.join(directory, filename)
        
        # 새로운 파일명 생성 (cam01_XXXXXX.json 형식)
        new_filename = f"cam_1_{index:03d}.png"  # 6자리 숫자로 패딩
        new_path = os.path.join(directory, new_filename)
        
        # 파일명 변경
        try:
            os.rename(old_path, new_path)
            print(f"변경 완료: {filename} -> {new_filename}")
        except Exception as e:
            print(f"오류 발생 ({filename}): {str(e)}")

# 함수 실행
rename_files(directory)"""

"""import os
import re

# 파일이 있는 디렉토리 경로를 지정하세요.
directory = r"F:\didim_data1_label"

# 날짜 패턴 (YYYY-MM-DD 형식으로 시작하는 파일명)
date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}\s+(.*)$')

# video ID 패턴을 수정 (.mp4에 의존하지 않도록)
video_pattern = re.compile(r'(video_\d{8}_\d{6}_\d{3})')  # 수정된 코드

for filename in os.listdir(directory):
    old_path = os.path.join(directory, filename)
    
    # 디렉토리가 아닌 실제 파일만 처리
    if os.path.isfile(old_path):
        # 1단계: 날짜 부분 제거
        date_match = date_pattern.match(filename)
        if date_match:
            # 날짜 이후 부분 (예: "video_20250225_111616_250.mp4.json")
            rest_part = date_match.group(1)
            
            # 2단계: video ID 추출
            video_match = video_pattern.search(rest_part)
            if video_match:
                # video로 시작하여 .mp4 전까지 (예: "video_20250225_111616_250")
                video_id = video_match.group(1)
                
                # 파일 확장자 가져오기 (원본 확장자 유지)
                _, ext = os.path.splitext(filename)
                
                # 새 파일명: video ID + 원본 확장자
                new_filename = f"{video_id}{ext}"
                new_path = os.path.join(directory, new_filename)
                
                # 파일명 변경
                os.rename(old_path, new_path)
                print(f"Renamed '{filename}' -> '{new_filename}'")
            else:
                print(f"Video pattern not found in: {rest_part}")
        else:
            # 날짜 패턴 없이 직접 video ID 추출 시도
            video_match = video_pattern.search(filename)
            if video_match:
                video_id = video_match.group(1)
                _, ext = os.path.splitext(filename)
                new_filename = f"{video_id}{ext}"
                new_path = os.path.join(directory, new_filename)
                
                os.rename(old_path, new_path)
                print(f"Renamed '{filename}' -> '{new_filename}'")
            else:
                print(f"Pattern not matched: {filename}")"""

"""import os
import glob
import json

# 입력 및 출력 디렉토리 설정
input_dir = r'F:\didim_data1_label'
output_dir = r'F:\yolo\data\json2'

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 입력 디렉토리 내의 모든 JSON 파일에 대해 반복
for file_path in glob.glob(os.path.join(input_dir, '*.json')):
    # JSON 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 변환된 데이터를 저장할 리스트
    transformed_data = []
    
    # 데이터 구조 확인 및 처리
    if isinstance(data, list):
        # 데이터가 이미 리스트 형태인 경우
        for item in data:
            # 데이터의 구조에 따라 적절히 변환
            if isinstance(item, dict):
                if "frameRange" in item and "name" in item:
                    transformed_data.append({
                        "frameRange": item.get("frameRange"),
                        "activity": item.get("name")
                    })
                elif "frameRange" in item and "activity" in item:
                    transformed_data.append({
                        "frameRange": item.get("frameRange"),
                        "activity": item.get("activity")
                    })
    elif isinstance(data, dict):
        # 데이터가 딕셔너리 형태인 경우
        if "tags" in data and isinstance(data["tags"], list):
            # tags 배열에서 frameRange와 name(=activity) 추출
            for tag in data["tags"]:
                if isinstance(tag, dict):
                    transformed_data.append({
                        "frameRange": tag.get("frameRange"),
                        "activity": tag.get("name")
                    })
    
    # 원본 파일명 유지하여 출력 파일 경로 생성
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, file_name)
    
    # 변환된 데이터를 새로운 JSON 파일로 저장
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)
    
    print(f"처리 완료: {file_path} -> {output_file_path}")

print("JSON 변환 완료!")"""



"""import os
import pandas as pd

# 원본 CSV 파일들이 있는 폴더
src_folder = r'F:\yolo\result'
# 결과 CSV 파일을 저장할 폴더
dest_folder = r'F:\doldidim\data\csv\result_prepro2'

# 결과 저장 폴더가 없으면 생성
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# src_folder 내의 모든 CSV 파일에 대해 처리
for filename in os.listdir(src_folder):
    if filename.lower().endswith('.csv'):
        file_path = os.path.join(src_folder, filename)
        df = pd.read_csv(file_path, header=None)

        # 프레임(첫 번째 열)과 사람 검출여부(두 번째 열)을 제외한 나머지를 3개씩 묶어서 관절 정보로 가정
        num_joints = (df.shape[1] - 2) // 3

        # 결과를 저장할 DataFrame 생성 (프레임 번호 추가)
        result = pd.DataFrame()
        result['frame'] = df.iloc[:, 0]

        # 각 관절의 x, y 좌표 추출하여 새로운 컬럼으로 추가
        for i in range(num_joints):
            x_index = 2 + i * 3  # 관절의 x 좌표 인덱스
            y_index = 3 + i * 3  # 관절의 y 좌표 인덱스
            result[f'joint{i+1}_x'] = df.iloc[:, x_index]
            result[f'joint{i+1}_y'] = df.iloc[:, y_index]

        # 결과를 저장할 경로 지정 (파일명은 원본과 동일)
        save_path = os.path.join(dest_folder, filename)
        result.to_csv(save_path, index=False)
        print("저장 완료:", save_path)"""

"""import os
import shutil

# 경로 설정 (역슬래시 앞에 r을 붙여 raw string으로 사용)
source_dir = r"F:\didimdol2"
label_dir = r"F:\didimdol2_label"
target_dir = r"F:\yolo\data\2d video"

# 대상 디렉토리가 없으면 생성
os.makedirs(target_dir, exist_ok=True)

# source_dir을 재귀적으로 순회
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # label 디렉토리에 동일한 파일명이 있는지 확인
        label_path = os.path.join(label_dir, file)
        if os.path.exists(label_path):
            source_file = os.path.join(root, file)
            # 파일 복사 (메타데이터까지 복사하고 싶으면 copy2 사용)
            shutil.copy2(source_file, target_dir)
            print(f"Copied: {source_file} -> {target_dir}")"""
"""import os
import shutil

# 삭제 기준이 되는 파일들이 들어 있는 디렉토리
csv_dir = r"F:\yolo\data\csv"

# 순회하며 삭제할 대상 디렉토리
video_dir = r"F:\didimdol_videos0"

# 백업 디렉토리 (선택적)
backup_dir = r"F:\backup_before_delete"

def main():
    # csv_dir 내 파일명(확장자 제외)을 모두 읽어 집합(set)에 저장
    csv_filenames = {os.path.splitext(filename)[0].lower() for filename in os.listdir(csv_dir)}
    
    # 삭제 전 확인
    total_files = 0
    files_to_delete = []
    
    for root, dirs, files in os.walk(video_dir):
        for filename in files:
            base_name = os.path.splitext(filename)[0].lower()
            if base_name in csv_filenames:
                file_path = os.path.join(root, filename)
                files_to_delete.append(file_path)
                total_files += 1
    
    if total_files > 0:
        print(f"총 {total_files}개의 파일이 삭제 대상입니다.")
        confirm = input("정말로 삭제하시겠습니까? (y/n): ")
        
        if confirm.lower() == 'y':
            # 백업 디렉토리 생성 (선택적)
            os.makedirs(backup_dir, exist_ok=True)
            
            for file_path in files_to_delete:
                try:
                    # 백업 (선택적)
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, backup_path)
                    
                    # 삭제
                    os.remove(file_path)
                    print(f"삭제 완료: {file_path}")
                except Exception as e:
                    print(f"삭제 실패 {file_path}: {e}")
        else:
            print("삭제 작업이 취소되었습니다.")
    else:
        print("삭제할 파일이 없습니다.")

if __name__ == "__main__":
    main()"""

"""import os
import shutil

# 대상 디렉토리
main_dir = r"F:\didimdol_videos0"

def extract_files_and_remove_folders():
    # 메인 디렉토리의 모든 항목을 순회
    for item in os.listdir(main_dir):
        item_path = os.path.join(main_dir, item)
        
        # 디렉토리인 경우만 처리
        if os.path.isdir(item_path):
            print(f"폴더 처리 중: {item}")
            
            # 폴더 내의 모든 파일 이동
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                
                # 파일인 경우만 이동
                if os.path.isfile(file_path):
                    # 대상 경로에 같은 이름의 파일이 있는지 확인
                    dest_path = os.path.join(main_dir, file)
                    
                    if os.path.exists(dest_path):
                        # 중복 파일 처리 (이름 변경)
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dest_path):
                            new_name = f"{base}_{counter}{ext}"
                            dest_path = os.path.join(main_dir, new_name)
                            counter += 1
                    
                    # 파일 이동
                    try:
                        shutil.move(file_path, dest_path)
                        print(f"  파일 이동: {file} -> {os.path.basename(dest_path)}")
                    except Exception as e:
                        print(f"  파일 이동 실패 {file}: {e}")
            
            # 폴더 삭제
            try:
                os.rmdir(item_path)  # rmdir은 빈 폴더만 삭제함
                print(f"폴더 삭제 완료: {item}")
            except Exception as e:
                print(f"폴더 삭제 실패 {item}: {e}")

if __name__ == "__main__":
    # 실행 전 확인
    confirm = input(f"{main_dir} 디렉토리의 모든 하위 폴더에서 파일을 추출하고 폴더를 삭제합니다. 계속하시겠습니까? (y/n): ")
    
    if confirm.lower() == 'y':
        extract_files_and_remove_folders()
        print("작업 완료!")
    else:
        print("작업이 취소되었습니다.")

 """

import os
import shutil

def get_file_names(directory):
    """디렉토리에서 파일명만 추출 (확장자 제외)"""
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            # 확장자 제외한 파일명만 저장
            base_name = os.path.splitext(filename)[0]
            files.append(base_name)
    return set(files)  # 중복 제거를 위해 set 사용

def find_and_delete_unique_files():
    # 디렉토리 경로
    json2_dir = r"F:\yolo\data\json2"
    didim_dir = r"F:\didim_data1"
    
    # 각 디렉토리의 파일명 목록 가져오기
    print("파일 목록 읽는 중...")
    json2_files = get_file_names(json2_dir)
    didim_files = get_file_names(didim_dir)
    
    # didim_data1에만 있는 파일 찾기
    unique_to_didim = didim_files - json2_files
    
    print(f"\njson2 디렉토리 파일 수: {len(json2_files)}")
    print(f"didim 디렉토리 파일 수: {len(didim_files)}")
    print(f"didim에만 있는 파일 수: {len(unique_to_didim)}")
    
    if len(unique_to_didim) == 0:
        print("\n삭제할 파일이 없습니다.")
        return
    
    # 삭제할 파일 목록 출력
    print("\n삭제 예정인 파일 목록:")
    for filename in sorted(unique_to_didim):
        print(f"- {filename}")
    
    # 사용자 확인
    confirmation = input("\n위 파일들을 삭제하시겠습니까? (yes/no): ")
    if confirmation.lower() != 'yes':
        print("삭제가 취소되었습니다.")
        return
    
    # 삭제 진행
    deleted_count = 0
    error_count = 0
    
    print("\n파일 삭제 중...")
    for base_name in unique_to_didim:
        # 원본 파일명과 동일한 모든 파일 찾기 (확장자 상관없이)
        for filename in os.listdir(didim_dir):
            current_base_name = os.path.splitext(filename)[0]
            if current_base_name == base_name:
                try:
                    file_path = os.path.join(didim_dir, filename)
                    os.remove(file_path)
                    print(f"삭제됨: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"오류 발생 ({filename}): {str(e)}")
                    error_count += 1
    
    # 결과 보고
    print(f"\n작업 완료:")
    print(f"- 성공적으로 삭제된 파일: {deleted_count}개")
    if error_count > 0:
        print(f"- 삭제 실패한 파일: {error_count}개")

if __name__ == "__main__":
    try:
        find_and_delete_unique_files()
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {str(e)}")