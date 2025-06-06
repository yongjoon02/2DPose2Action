# breakfast.py
"""
BreakfastII_15fps_qvga_sync → Breakfast/
 ├─ videos/              # *.avi
 │   ├─ P03_coffee.avi
 │   ├─ P03_friedegg.avi
 │   └─ …
 ├─ labels/              # *.labels (모든 라벨 파일)
 │   ├─ P03_coffee.labels
 │   ├─ P03_friedegg.labels
 │   └─ …
 └─ pose_json/           # YOLO-Pose 결과 (자동 생성)

각 참가자 폴더에서 cam01만 유지하고 나머지 폴더(stereo, webcam01, webcam02)는 삭제합니다.
모든 라벨 파일은 labels 폴더에 직접 저장됩니다.
"""

from pathlib import Path
import shutil

# ────────────────── 경로 설정 ──────────────────
SRC_ROOT = Path(r"F:\BreakfastII_15fps_qvga_sync")   # 원본 데이터 경로
DST_ROOT = SRC_ROOT.parent / "Breakfast"             # 목적지 경로

def setup_directories():
    """Breakfast 디렉토리 구조 생성"""
    print("디렉토리 구조 생성 중...")
    
    # 기존 Breakfast 디렉토리가 있으면 삭제
    if DST_ROOT.exists():
        print(f"기존 {DST_ROOT} 디렉토리 삭제 중...")
        shutil.rmtree(DST_ROOT)
    
    # 새 디렉토리 구조 생성
    (DST_ROOT / "videos").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "labels").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "pose_json").mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 디렉토리 구조 생성 완료: {DST_ROOT}")

def clean_participant_directory(participant_dir):
    """참가자 디렉토리에서 cam01을 제외한 모든 하위 디렉토리 삭제"""
    print(f"  {participant_dir.name} 폴더 정리 중...")
    
    # cam01이 아닌 모든 하위 디렉토리 삭제
    deleted_folders = []
    for item in participant_dir.iterdir():
        if item.is_dir() and item.name != "cam01":
            deleted_folders.append(item.name)
            shutil.rmtree(item, ignore_errors=True)
    
    if deleted_folders:
        print(f"    삭제된 폴더: {', '.join(deleted_folders)}")

def move_videos_and_labels(participant_dir):
    """비디오 파일과 라벨 파일을 적절한 위치로 이동"""
    participant_name = participant_dir.name
    
    # cam01 폴더에서 비디오 파일과 라벨 파일 이동
    cam01_dir = participant_dir / "cam01"
    if cam01_dir.exists():
        video_count = 0
        for avi_file in cam01_dir.glob("*.avi"):
            dst_video_path = DST_ROOT / "videos" / avi_file.name
            shutil.move(str(avi_file), str(dst_video_path))
            video_count += 1
        
        if video_count > 0:
            print(f"    비디오 파일 {video_count}개 이동 완료")
        
        # cam01 폴더에서 라벨 파일 이동
        label_count = 0
        for label_file in cam01_dir.glob("*.labels"):
            # 파일명에서 .avi. 부분 제거 (예: P03_coffee.avi.labels → P03_coffee.labels)
            new_filename = label_file.name.replace(".avi.", ".")
            dst_label_path = DST_ROOT / "labels" / new_filename
            
            # 라벨 파일 이동
            shutil.move(str(label_file), str(dst_label_path))
            label_count += 1
        
        if label_count > 0:
            print(f"    라벨 파일 {label_count}개 이동 완료 (cam01에서)")
        
        # cam01 폴더 삭제 (비어있으면)
        try:
            cam01_dir.rmdir()
        except OSError:
            pass  # 폴더가 비어있지 않으면 무시
    
    # 참가자 디렉토리에서 라벨 파일 이동 (혹시 다른 위치에 있을 수도 있음)
    label_count = 0
    for label_file in participant_dir.glob("*.labels"):
        # 파일명에서 .avi. 부분 제거 (예: P03_coffee.avi.labels → P03_coffee.labels)
        new_filename = label_file.name.replace(".avi.", ".")
        dst_label_path = DST_ROOT / "labels" / new_filename
        
        # 라벨 파일 이동
        shutil.move(str(label_file), str(dst_label_path))
        label_count += 1
    
    if label_count > 0:
        print(f"    라벨 파일 {label_count}개 이동 완료 (참가자 폴더에서)")

def cleanup_empty_participant_directory(participant_dir):
    """참가자 디렉토리가 비어있으면 삭제"""
    try:
        # 디렉토리가 비어있으면 삭제
        participant_dir.rmdir()
        print(f"    빈 디렉토리 삭제: {participant_dir.name}")
    except OSError:
        # 디렉토리가 비어있지 않으면 무시
        remaining_items = list(participant_dir.iterdir())
        if remaining_items:
            print(f"    경고: {participant_dir.name}에 남은 파일들이 있습니다: {[item.name for item in remaining_items]}")

def process_all_participants():
    """모든 참가자 디렉토리 처리"""
    if not SRC_ROOT.exists():
        print(f"❌ 오류: 원본 디렉토리를 찾을 수 없습니다: {SRC_ROOT}")
        return
    
    # 참가자 디렉토리 찾기 (P로 시작하고 숫자가 있는 디렉토리)
    participant_dirs = []
    for item in SRC_ROOT.iterdir():
        if (item.is_dir() and 
            item.name.lower().startswith("p") and 
            len(item.name) > 1 and 
            item.name[1:].lstrip("0").isdigit()):  # P01, P03 등의 형태
            participant_dirs.append(item)
    
    participant_dirs.sort()  # 정렬
    
    if not participant_dirs:
        print(f"❌ 오류: {SRC_ROOT}에서 참가자 디렉토리(P##)를 찾을 수 없습니다.")
        return
    
    print(f"발견된 참가자 디렉토리: {[d.name for d in participant_dirs]}")
    
    # 각 참가자 디렉토리 처리
    for participant_dir in participant_dirs:
        print(f"\n처리 중: {participant_dir.name}")
        
        # 1. 불필요한 폴더 삭제 (cam01 제외)
        clean_participant_directory(participant_dir)
        
        # 2. 비디오 파일과 라벨 파일 이동
        move_videos_and_labels(participant_dir)
        
        # 3. 빈 참가자 디렉토리 삭제
        cleanup_empty_participant_directory(participant_dir)

def print_summary():
    """처리 결과 요약 출력"""
    print("\n" + "="*60)
    print("📊 처리 결과 요약")
    print("="*60)
    
    # 비디오 파일 수
    video_files = list((DST_ROOT / "videos").glob("*.avi"))
    print(f"비디오 파일: {len(video_files)}개")
    
    # 라벨 파일 수
    label_files = list((DST_ROOT / "labels").glob("*.labels"))
    print(f"라벨 파일: {len(label_files)}개")
    
    print(f"\n✅ 모든 처리가 완료되었습니다!")
    print(f"결과 디렉토리: {DST_ROOT}")
    
    # 디렉토리 구조 출력
    print(f"\n📁 생성된 디렉토리 구조:")
    print(f"{DST_ROOT.name}/")
    print(f"├─ videos/ ({len(video_files)}개 파일)")
    print(f"├─ labels/ ({len(label_files)}개 파일)")
    print(f"└─ pose_json/ (비어있음 - YOLO-Pose 결과 저장용)")

def main():
    """메인 함수"""
    print("🍳 Breakfast 데이터셋 재구성 시작...")
    print(f"원본 경로: {SRC_ROOT}")
    print(f"목적지 경로: {DST_ROOT}")
    
    try:
        # 1. 디렉토리 구조 생성
        setup_directories()
        
        # 2. 모든 참가자 디렉토리 처리
        process_all_participants()
        
        # 3. 결과 요약 출력
        print_summary()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
