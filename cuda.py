import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 버전 (PyTorch): {torch.version.cuda}")

# NVIDIA GPU 목록 출력 시도
try:
    print(f"GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"GPU 정보 확인 오류: {e}")