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

def augment_skeleton_data_enhanced(coords, frames, label, augment_probability=0.8):
    """
    기본 증강 함수를 호출하여 단순화된 증강 적용
    """
    # 강화된 증강 없이 기본 증강만 적용
    return augment_skeleton_data(coords, frames, augment_probability)
