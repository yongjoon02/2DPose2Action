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
        # 증강 확률 증가
        if np.random.random() > 0.98:  # 거의 항상 증강 적용
            return coords
            
        # 중첩 증강 적용 (3-5개의 증강 기법을 연속 적용)
        num_augs = np.random.randint(3, 6)  # 최소 3개 적용
        augmented = np.copy(coords)
        
        # standing과 sitting에 특화된 증강 기법 추가
        special_augs = ['vertical_shift', 'horizontal_shift', 'enhanced_rotate']
        
        for _ in range(num_augs):
            if np.random.random() < 0.7:  # 70% 확률로 특화된 증강 사용
                aug_type = np.random.choice(special_augs)
            else:
                aug_type = np.random.choice([
                    'jitter', 'scale', 'rotate', 'mirror', 'time_warp', 
                    'gaussian_noise', 'drop_joints'
                ])
            
            if aug_type == 'vertical_shift':
                # 수직 방향 이동 (y 좌표만 조정)
                shift = np.random.uniform(-0.2, 0.2)
                for i in range(1, augmented.shape[1], 2):
                    augmented[:, i] += shift
                    
            elif aug_type == 'horizontal_shift':
                # 수평 방향 이동 (x 좌표만 조정)
                shift = np.random.uniform(-0.15, 0.15)
                for i in range(0, augmented.shape[1], 2):
                    augmented[:, i] += shift
                    
            elif aug_type == 'jitter':
                noise_level = np.random.uniform(0.03, 0.08)
                augmented += np.random.normal(0, noise_level, size=augmented.shape)
                
            elif aug_type == 'scale':
                scale_factor = np.random.uniform(0.6, 1.4)
                augmented *= scale_factor
                
            elif aug_type == 'enhanced_rotate':
                if augmented.shape[1] >= 2:
                    angle = np.random.uniform(-70, 70) * np.pi / 180
                    c, s = np.cos(angle), np.sin(angle)
                    rotation_matrix = np.array([[c, -s], [s, c]])
                    
                    # 상체와 하체를 독립적으로 회전
                    upper_angle = angle * np.random.uniform(0.8, 1.2)
                    lower_angle = angle * np.random.uniform(0.6, 1.4)
                    
                    upper_rotation = np.array([[np.cos(upper_angle), -np.sin(upper_angle)],
                                             [np.sin(upper_angle), np.cos(upper_angle)]])
                    lower_rotation = np.array([[np.cos(lower_angle), -np.sin(lower_angle)],
                                             [np.sin(lower_angle), np.cos(lower_angle)]])
                    
                    # 상체 회전 (0-16 인덱스)
                    for i in range(0, min(16, augmented.shape[1]), 2):
                        if i+1 < augmented.shape[1]:
                            xy = augmented[:, i:i+2]
                            augmented[:, i:i+2] = np.dot(xy, upper_rotation)
                    
                    # 하체 회전 (16 이상 인덱스)
                    for i in range(16, augmented.shape[1], 2):
                        if i+1 < augmented.shape[1]:
                            xy = augmented[:, i:i+2]
                            augmented[:, i:i+2] = np.dot(xy, lower_rotation)
        
        return augmented
    else:
        # 다른 클래스는 기존 증강 적용
        return augment_skeleton_data(coords, frames, augment_probability)
