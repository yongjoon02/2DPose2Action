import numpy as np

def time_warp(x, sigma=0.1, knot=4):
    orig_steps = np.arange(x.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
    warp_steps = np.linspace(0, x.shape[0]-1, num=knot+2)
    warper = np.interp(orig_steps, warp_steps, random_warps)
    warper = np.maximum(warper, 0.5)
    warper = np.cumsum(warper)
    warper = np.interp(orig_steps, np.arange(x.shape[0])*x.shape[0]/warper[-1], warper)
    new_frames = np.clip(warper, 0, x.shape[0]-1).astype(np.int32)
    return x[new_frames]

def add_noise(x, noise_level=0.02):
    return x + np.random.normal(loc=0.0, scale=noise_level, size=x.shape)

def random_scaling(x, scale_range=(0.95, 1.05)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return x * scale

def augment_skeleton_data(coords, frames, augment_probability=0.7):
    if np.random.random() > augment_probability:
        return coords
    augmented = np.copy(coords)
    aug_type = np.random.choice([
        'jitter', 'scale', 'rotate', 'mirror', 'time_warp', 
        'gaussian_noise', 'drop_joints'
    ])
    
    if aug_type == 'jitter':
        noise_level = np.random.uniform(0.01, 0.05)
        augmented += np.random.normal(0, noise_level, size=augmented.shape)
    elif aug_type == 'scale':
        scale_factor = np.random.uniform(0.8, 1.2)
        augmented *= scale_factor
    elif aug_type == 'rotate':
        if augmented.shape[1] >= 2:
            angle = np.random.uniform(-30, 30) * np.pi / 180
            c, s = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[c, -s], [s, c]])
            for i in range(0, augmented.shape[1], 2):
                if i+1 < augmented.shape[1]:
                    xy = augmented[:, i:i+2]
                    augmented[:, i:i+2] = np.dot(xy, rotation_matrix)
    elif aug_type == 'mirror':
        if augmented.shape[1] >= 1:
            for i in range(0, augmented.shape[1], 2):
                augmented[:, i] = -augmented[:, i]
    elif aug_type == 'time_warp':
        timesteps = augmented.shape[0]
        knot = 4
        orig_steps = np.arange(timesteps)
        random_warps = np.random.normal(loc=1.0, scale=0.1, size=(knot+2))
        warp_steps = np.linspace(0, timesteps-1, num=knot+2)
        warper = np.interp(orig_steps, warp_steps, random_warps)
        new_frames = np.linspace(0, timesteps-1, timesteps)
        for i in range(augmented.shape[1]):
            augmented[:, i] = np.interp(new_frames, orig_steps*warper, augmented[:, i])
    elif aug_type == 'gaussian_noise':
        noise_level = np.random.uniform(0.01, 0.03)
        augmented += np.random.normal(0, noise_level, size=augmented.shape)
    elif aug_type == 'drop_joints':
        joint_drop_prob = 0.1
        joint_mask = np.random.random(augmented.shape[1]) > joint_drop_prob
        for i in range(augmented.shape[1]):
            if not joint_mask[i]:
                augmented[:, i] = 0
    return augmented

# 강화된 증강 함수 추가: standing(0)과 sitting(1) 클래스를 위한 증강
def augment_skeleton_data_enhanced(coords, frames, label, augment_probability=0.8):
    """
    standing(0)과 sitting(1) 클래스에 대해 증강 확률을 높이고 더 강력한 증강 적용
    """
    # standing과 sitting 클래스에 대해서는 높은 확률로 증강 적용
    if label in [0, 1]:  # standing(0) 또는 sitting(1)
        # 증강 확률 증가 (0.9 -> 0.95)
        if np.random.random() > 0.95:
            return coords
            
        # 중첩 증강 적용 (2-4개의 증강 기법을 연속 적용)
        num_augs = np.random.randint(2, 5)  # 최대 4개까지 증강 적용
        augmented = np.copy(coords)
        
        for _ in range(num_augs):
            aug_type = np.random.choice([
                'jitter', 'scale', 'rotate', 'mirror', 'time_warp', 
                'gaussian_noise', 'drop_joints', 'enhanced_rotate', 'enhanced_scale'
            ])
            
            if aug_type == 'jitter':
                # 노이즈 수준 증가
                noise_level = np.random.uniform(0.03, 0.08)  # 노이즈 범위 증가
                augmented += np.random.normal(0, noise_level, size=augmented.shape)
            elif aug_type == 'scale':
                # 스케일 범위 확대
                scale_factor = np.random.uniform(0.65, 1.35)  # 스케일 범위 확대
                augmented *= scale_factor
            elif aug_type == 'enhanced_scale':
                # x, y 방향으로 독립적인 스케일링
                scale_x = np.random.uniform(0.7, 1.3)
                scale_y = np.random.uniform(0.7, 1.3)
                for i in range(0, augmented.shape[1], 2):
                    if i+1 < augmented.shape[1]:
                        augmented[:, i] *= scale_x
                        augmented[:, i+1] *= scale_y
            elif aug_type == 'rotate':
                if augmented.shape[1] >= 2:
                    # 회전각 범위 확대
                    angle = np.random.uniform(-45, 45) * np.pi / 180
                    c, s = np.cos(angle), np.sin(angle)
                    rotation_matrix = np.array([[c, -s], [s, c]])
                    for i in range(0, augmented.shape[1], 2):
                        if i+1 < augmented.shape[1]:
                            xy = augmented[:, i:i+2]
                            augmented[:, i:i+2] = np.dot(xy, rotation_matrix)
            elif aug_type == 'enhanced_rotate':
                # 몸통과 팔, 다리를 독립적으로 회전 (골격 구조에 따라 인덱스 조정 필요)
                if augmented.shape[1] >= 2:
                    # 회전 각도 범위 확대
                    upper_angle = np.random.uniform(-60, 60) * np.pi / 180  # -50~50에서 -60~60으로
                    lower_angle = np.random.uniform(-40, 40) * np.pi / 180  # -30~30에서 -40~40으로
                    
                    upper_rotation = np.array([[np.cos(upper_angle), -np.sin(upper_angle)], 
                                             [np.sin(upper_angle), np.cos(upper_angle)]])
                    lower_rotation = np.array([[np.cos(lower_angle), -np.sin(lower_angle)], 
                                             [np.sin(lower_angle), np.cos(lower_angle)]])
                    
                    # 상체 부분 (예: 0-16 인덱스가 상체라 가정)
                    for i in range(0, min(16, augmented.shape[1]), 2):
                        if i+1 < augmented.shape[1]:
                            xy = augmented[:, i:i+2]
                            augmented[:, i:i+2] = np.dot(xy, upper_rotation)
                    
                    # 하체 부분 (예: 16 이상 인덱스가 하체라 가정)
                    for i in range(16, augmented.shape[1], 2):
                        if i+1 < augmented.shape[1]:
                            xy = augmented[:, i:i+2]
                            augmented[:, i:i+2] = np.dot(xy, lower_rotation)
            elif aug_type == 'mirror':
                if augmented.shape[1] >= 1:
                    for i in range(0, augmented.shape[1], 2):
                        augmented[:, i] = -augmented[:, i]
            elif aug_type == 'time_warp':
                # 더 강한 시간 왜곡
                timesteps = augmented.shape[0]
                knot = 6  # 더 많은 변화점
                orig_steps = np.arange(timesteps)
                random_warps = np.random.normal(loc=1.0, scale=0.15, size=(knot+2))  # 더 큰 왜곡
                warp_steps = np.linspace(0, timesteps-1, num=knot+2)
                warper = np.interp(orig_steps, warp_steps, random_warps)
                warper = np.maximum(warper, 0.4)  # 더 작은 최소값 -> 더 강한 압축
                warper = np.cumsum(warper)
                warper = np.interp(orig_steps, np.arange(timesteps)*timesteps/warper[-1], warper)
                new_frames = np.clip(warper, 0, timesteps-1).astype(np.int32)
                for i in range(augmented.shape[1]):
                    orig_data = augmented[:, i].copy()
                    augmented[:, i] = orig_data[new_frames]
            elif aug_type == 'gaussian_noise':
                # 노이즈 수준 증가
                noise_level = np.random.uniform(0.02, 0.06)
                augmented += np.random.normal(0, noise_level, size=augmented.shape)
            elif aug_type == 'drop_joints':
                # 드롭 확률 증가 및 시간 차원에서도 드롭
                joint_drop_prob = 0.15
                joint_mask = np.random.random(augmented.shape[1]) > joint_drop_prob
                
                # 시간적 드롭아웃 (특정 프레임의 모든 관절 정보를 드롭)
                time_drop_prob = 0.05
                time_mask = np.random.random(augmented.shape[0]) > time_drop_prob
                
                for i in range(augmented.shape[1]):
                    if not joint_mask[i]:
                        augmented[:, i] = 0
                
                for t in range(augmented.shape[0]):
                    if not time_mask[t]:
                        augmented[t, :] = 0
        
        return augmented
    else:
        # 다른 클래스는 기존 증강 적용
        return augment_skeleton_data(coords, frames, augment_probability)
