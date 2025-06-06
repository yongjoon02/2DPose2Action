# fix_mapping.py  ─────────────────────────────────────────────
from pathlib import Path
import re, json, collections

MAPPING_TXT = "mapping_fine2coarse.txt"             # 기존 파일
FINE_DIR    = Path("data/Breakfast/labels/fine")    # fine 라벨 폴더
OUT_TXT     = "mapping_fine2coarse_fixed.txt"

# ─── 규칙 1: coarse = fine.split('_')[0] ─────────────────────
# 예외 사전(필요하면 더 추가)
EXCEPT = {
    "walk_in":  "sil",   # 배경으로 간주
    "walk_out": "sil",
    "stirfry_egg": "fry",  # fry로 합침
    "pour_egg2pan": "pour",
    "pour_flour":   "pour",
    "pour_sugar":   "pour",
    "stir_milk":    "stir",
    "stir_coffee":  "stir",
    "stir_dough":   "stir",
    "take_squeezer":"take",
    "take_glass":   "take",
    "take_topping": "take",
    "take_eggs":    "take",
    "put_egg2plate":"put",
    "put_fruit2bowl":"put",
    "put_toppingOnTop":"put",
    "peel_fruit":   "cut",
    "squeeze_orange": "pour",
}

# ─── 1) 기존 매핑 읽기 ───────────────────────────────────────
mapping = {}
with open(MAPPING_TXT) as f:
    for ln in f:
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        fine, coarse = re.split(r"\s+", ln)[:2]
        mapping[fine] = coarse

# ─── 2) fine 폴더 돌며 누락 라벨 자동 매핑 ─────────────────
missing = collections.Counter()
for lbl in FINE_DIR.glob("*.labels"):
    for ln in lbl.read_text().splitlines():
        if not ln.strip(): continue
        fine = ln.split()[-1]
        if fine in mapping: continue
        if fine in EXCEPT:
            mapping[fine] = EXCEPT[fine]
        else:
            mapping[fine] = fine.split("_")[0]   # verb 규칙
            missing[mapping[fine]] += 1

# ─── 3) 새 매핑 파일 저장 ──────────────────────────────────
with open(OUT_TXT, "w") as g:
    g.write("# fine  →  coarse (auto-fixed)\n")
    for fine, coarse in sorted(mapping.items()):
        g.write(f"{fine:<20s} {coarse}\n")

print(f"✅  fixed mapping saved to {OUT_TXT}")
print(f"   auto-added {sum(missing.values())} labels, "
      f"{len(missing)} new coarse verbs = {list(missing.keys())}")
