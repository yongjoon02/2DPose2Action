#!/usr/bin/env python
"""
fine(48) 라벨 → coarse(10) 라벨 변환 + 검증

Usage:
    python convert_labels.py \
        --label_dir  Breakfast/labels/fine \
        --mapping    mapping_fine2coarse.txt \
        --out_dir    Breakfast/labels/coarse
"""

import argparse, re
from pathlib import Path

def load_mapping(path):
    """fine → coarse dict 생성"""
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            fine, coarse = parts[0], parts[1]
            mapping[fine] = coarse
    return mapping

def convert_file(fine_path: Path, map_dict, dst_dir: Path):
    classes_seen = set()
    lines_out = []
    with fine_path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 2:  # 최소 "start-end label" 형태여야 함
                continue
            
            # start-end와 label 분리
            time_range, fine = parts[0], parts[1]
            
            # start-end에서 start와 end 추출
            if '-' in time_range:
                s, e = time_range.split('-', 1)
            else:
                continue  # 올바른 형식이 아니면 건너뛰기
                
            coarse = map_dict.get(fine)
            if coarse is None:
                raise ValueError(f"[{fine_path.name}] 매핑 누락: {fine}")
            lines_out.append(f"{s}-{e} {coarse}\n")
            classes_seen.add(fine)

    # coarse 파일 저장
    dst = dst_dir / fine_path.name  # 파일명 동일
    with dst.open("w") as g:
        g.writelines(lines_out)
    return classes_seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", required=True, help="fine 라벨 폴더")
    ap.add_argument("--mapping", required=True, help="fine→coarse 매핑 txt")
    ap.add_argument("--out_dir", required=True, help="coarse 라벨 저장 폴더")
    args = ap.parse_args()

    map_dict = load_mapping(args.mapping)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    total_missing = set()
    for lbl in sorted(Path(args.label_dir).glob("*.labels")):
        try:
            classes = convert_file(lbl, map_dict, out_dir)
        except ValueError as e:
            print("❌", e)
            total_missing.add(str(e))
            continue
        if len(classes) > 12:
            print(f"⚠️  {lbl.name}: coarse 변환 후에도 클래스가 12개 초과?")
        else:
            print(f"✅ {lbl.name} → {len(classes)} fine classes mapped")

    if total_missing:
        print("\n[!] 매핑 누락된 fine 클래스가 있습니다:")
        for m in total_missing:
            print("   ", m)

if __name__ == "__main__":
    main()
