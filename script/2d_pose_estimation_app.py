import sys
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 현재 파일(2d_pose_estimation_app.py)의 부모 폴더(= YOLO)를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yolo_pose import main

if __name__ == "__main__":
    main()
