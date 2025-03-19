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
directory = r"F:\didimdol2_label"

# 날짜 패턴 (YYYY-MM-DD 형식으로 시작하는 파일명)
date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}\s+(.*)$')

# video ID 패턴 (video로 시작하여 .mp4 전까지의 부분)
video_pattern = re.compile(r'(video[^.]*?)\.mp4')

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
input_dir = r'F:\label_didimdol'
output_dir = r'F:\label_didimdol_prepro'

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 입력 디렉토리 내의 모든 JSON 파일에 대해 반복
for file_path in glob.glob(os.path.join(input_dir, '*.json')):
    # JSON 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # tags 배열에서 frameRange와 name(=activity) 추출
    transformed_data = []
    for tag in data.get("tags", []):
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
    
    print(f"Processed {file_path} -> {output_file_path}")

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

import os
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
            print(f"Copied: {source_file} -> {target_dir}")
