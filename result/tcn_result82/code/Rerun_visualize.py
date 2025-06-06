# src/Rerun_visualize.py

from pathlib import Path
import rerun as rr
import pandas as pd
import numpy as np
import cv2 as cv
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rerun_visualize(csv_dir, pred_dir, video_dir, gt_dir=None, title="skeleton_label_compare"):
    # Step 1: 공통 base_name 자동 추출
    csv_files = {f.stem for f in csv_dir.glob("*.csv")}
    pred_files = {f.stem.replace("_prediction", "") for f in pred_dir.glob("*_prediction.json")}
    video_files = {f.stem for f in video_dir.glob("*.mp4")}
    
    if gt_dir and Path(gt_dir).exists():
        gt_files = {f.stem for f in Path(gt_dir).glob("*.json")}
        common = csv_files & pred_files & video_files & gt_files
        gt_available = True
    else:
        common = csv_files & pred_files & video_files
        gt_available = False

    if not common:
        raise FileNotFoundError("공통으로 존재하는 파일 prefix를 찾을 수 없습니다.")

    base_name = sorted(list(common))[0]
    print(f" base_name 자동 선택됨: {base_name}")
    
    # Step 2: 경로 조립
    csv_path = csv_dir / f"{base_name}.csv"
    pred_json_path = pred_dir / f"{base_name}_prediction.json"
    video_path = video_dir / f"{base_name}.mp4"
    gt_json_path = Path(gt_dir) / f"{base_name}.json" if gt_available else None

    # Step 3: 데이터 로드
    df = pd.read_csv(csv_path)
    if isinstance(df.iloc[0, 1], str) and "nose" in df.iloc[0, 1].lower():
        df = df.drop(index=0).reset_index(drop=True)
    frame_numbers = df["frame"].astype(int).tolist()
    df = df.drop(columns=["frame"])

    # GT 로드 (옵션)
    frame_gt = {}
    if gt_available:
        with open(gt_json_path, "r") as f:
            gt_json = json.load(f)
        frame_gt = {
            frame: entry["activity"]
            for entry in gt_json
            for frame in range(entry["frameRange"][0], entry["frameRange"][1] + 1)
        }

    # Pred 로드
    with open(pred_json_path, "r") as f:
        pred_json = json.load(f)
    frame_pred = {
        frame: entry["activity"]
        for entry in pred_json
        for frame in range(entry["frameRange"][0], entry["frameRange"][1] + 1)
    }

    EDGES = [
        (0, 1), (1, 3), (0, 2), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    cap = cv.VideoCapture(str(video_path))
    rr.init(title, spawn=True)

    for idx, row in df.iterrows():
        frame = frame_numbers[idx]
        rr.set_time_sequence("frame", frame)

        ret, frame_img = cap.read()
        if not ret:
            continue

        frame_rgb = cv.cvtColor(frame_img, cv.COLOR_BGR2RGB)
        rr.log("video/frame", rr.Image(frame_rgb))

        coords = row.to_numpy(dtype=np.float32).reshape(17, 2)
        rr.log("skeleton/joints", rr.Points2D(coords, radii=3.0))
        lines = np.array([[coords[s], coords[e]] for s, e in EDGES], dtype=np.float32)
        rr.log("skeleton/edges", rr.LineStrips2D(lines))

        # 시각화 텍스트 설정
        pred_raw = frame_pred.get(frame)
        gt_raw = frame_gt.get(frame) if gt_available else None

        if pred_raw is None:
            continue

        if gt_available and gt_raw is not None:
            gt = gt_raw.strip().lower()
            pred = pred_raw.strip().lower()
            text = f"GT: {gt_raw}    |    Pred: {pred_raw}"
            is_diff = gt != pred
        else:
            text = f"Pred: {pred_raw}"
            is_diff = False

        color = (255, 0, 0) if is_diff else (255, 255, 255)
        rr.log("skeleton/label_compare", rr.TextLog(text, color=color))
        rr.log("skeleton/joints", rr.Points2D(coords, radii=4.0, colors=color))

    rr.spawn()