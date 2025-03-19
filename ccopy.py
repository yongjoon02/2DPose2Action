import os
import shutil

source_dir = r"F:\didimdol2"
label_dir = r"F:\didimdol2_label"
target_dir = r"F:\yolo\data\2d video"

os.makedirs(target_dir, exist_ok=True)

def extract_key(filename):
    # 예시: 'video_20250227_161225_946.mp4'에서 '20250227_161225'를 추출
    # 파일 이름 형식에 맞게 수정 필요
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[1] + "_" + parts[2]
    return filename

# label_dir 내의 파일들의 키를 미리 추출하여 세트로 만듦
label_keys = set()
for file in os.listdir(label_dir):
    label_keys.add(extract_key(file))

matched_files = 0

for root, dirs, files in os.walk(source_dir):
    print(f"디렉토리 확인: {root}")
    for file in files:
        key = extract_key(file)
        print(f"  파일 발견: {file} (키: {key})")
        if key in label_keys:
            source_file = os.path.join(root, file)
            shutil.copy2(source_file, target_dir)
            print(f"  복사됨: {source_file} -> {target_dir}")
            matched_files += 1

if matched_files == 0:
    print("라벨 디렉토리와 키가 일치하는 파일을 찾지 못했습니다.")
else:
    print(f"총 {matched_files}개의 파일이 복사되었습니다.")
