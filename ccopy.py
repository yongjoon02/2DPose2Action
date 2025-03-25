import os
import shutil

source_dir = r"F:\didimdol_videos"
label_dir = r"F:\yolo\result\tcn_result11\combined_results\output_json"
target_dir = r"F:\didim_testdata"

os.makedirs(target_dir, exist_ok=True)

def extract_key(filename):
    # 예시: 'video_20250227_161225_946.mp4'에서 '20250227_161225'를 추출
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[1] + "_" + parts[2]
    return filename

# 라벨 디렉토리 내 파일 목록 출력 (디버깅용)
print("라벨 디렉토리 내 파일 목록:")
for file in os.listdir(label_dir):
    print(f"  라벨 파일: {file}")

# label_dir 내의 파일들의 키를 미리 추출하여 세트로 만듦
label_keys = set()
for file in os.listdir(label_dir):
    key = extract_key(file)
    label_keys.add(key)
    print(f"  라벨 키 추출: {file} -> {key}")

print(f"추출된 라벨 키 개수: {len(label_keys)}")
matched_files = 0

for root, dirs, files in os.walk(source_dir):
    print(f"디렉토리 확인: {root}")
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mov')):  # 비디오 파일만 처리
            key = extract_key(file)
            print(f"  파일 발견: {file} (키: {key})")
            if key in label_keys:
                source_file = os.path.join(root, file)
                shutil.copy2(source_file, target_dir)
                print(f"  복사됨: {source_file} -> {target_dir}")
                matched_files += 1
            else:
                print(f"  매칭 실패: {key} (사용 가능한 키: {', '.join(list(label_keys)[:5])}...)")

if matched_files == 0:
    print("라벨 디렉토리와 키가 일치하는 파일을 찾지 못했습니다.")
else:
    print(f"총 {matched_files}개의 파일이 복사되었습니다.")
