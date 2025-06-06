# check_labels.py  (수정 버전)
from pathlib import Path
from collections import Counter          # ← 여기!

LABEL_DIR = Path("data/Breakfast/labels/coarse")   # coarse 라벨 폴더 경로

cls_counter = Counter()
for lbl in LABEL_DIR.glob("*.labels"):
    with lbl.open() as f:
        cls_counter.update(
            ln.split()[-1] for ln in f if ln.strip()
        )

print("Coarse classes:", sorted(cls_counter))
for cls, cnt in cls_counter.items():
    print(f"{cls:<10s}: {cnt:6d} frames")
