# %%
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Literal, Union
import numpy as np
import cv2
import pandas as pd
from loguru import logger
from tqdm.contrib.concurrent import process_map
from tqdm.rich import tqdm
from skimage.metrics import structural_similarity as ssim

# %%
dataset_path = Path('dataset/Celeb-DF-pair-videos')
fake_path = dataset_path / 'fake'
real_path = dataset_path / 'real'

# %%
labels_path = 'dataset/Celeb-DF-pair-videos/labels.csv'
with open(labels_path, 'r', encoding='utf-8') as f:
    label = pd.read_csv(f)

# %%
def get_video_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"错误：无法打开视频 {video_path}")
        return -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# %%
def video2image(video, frame):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f'cannot open video: {video}')
    target_idx = frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_idx < 0 or target_idx >= total_frames:
        cap.release()
        raise KeyError(f'video has no frame: {frame} is out of {total_frames}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# %%
def is_homologous(real_frame: np.ndarray, fake_frame: np.ndarray) -> bool:
    if not isinstance(real_frame, np.ndarray) or not isinstance(fake_frame, np.ndarray):
        raise TypeError("输入必须是np.ndarray格式的图像帧")
    def to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img[..., 0]
        elif len(img.shape) == 2:
            return img
        else:
            raise ValueError("不支持的图像维度")
    real = to_gray(real_frame)
    fake = to_gray(fake_frame)
    if real.shape != fake.shape:
        fake = cv2.resize(fake, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_LINEAR)
    real = real.astype(np.uint8)
    fake = fake.astype(np.uint8)
    ssim_score = ssim(real, fake, full=True)[0]
    mse = np.mean((real.astype(np.float32) - fake.astype(np.float32)) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse) if mse != 0 else 100
    orb = cv2.ORB_create(nfeatures=1500) # type: ignore
    kp1, des1 = orb.detectAndCompute(real, None)
    kp2, des2 = orb.detectAndCompute(fake, None)
    match_rate = 0.0
    if len(kp1) > 0 and len(kp2) > 0:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        good_matches = [m for m in matches if m.distance < 50]
        match_rate = len(good_matches) / len(kp1)
    def get_noise(img):
        blur = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.5)
        return img.astype(np.float32) - blur
    real_noise = get_noise(real)
    fake_noise = get_noise(fake)
    real_hist, _ = np.histogram(real_noise, bins=50, range=(-50, 50), density=True)
    fake_hist, _ = np.histogram(fake_noise, bins=50, range=(-50, 50), density=True)
    noise_similarity = np.sum(np.sqrt(real_hist * fake_hist))
    conditions = [
        ssim_score > 0.75,
        psnr > 28.0,
        match_rate > 0.25,
        noise_similarity > 0.80
    ]
    return sum(conditions) >= 3

# %%
def frame_to_png(frame: np.ndarray, save_path: Optional[Union[str, Path]] = None) -> bool:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise TypeError("错误：输入不是有效的 OpenCV 帧数据")
    if save_path is None:
        save_path = Path("./output.png")
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(save_path), frame)
    if not success:
        raise KeyError(f"错误：无法保存图片到 {save_path}")
    return success

# %%
def deal_pair(pair_label_info, save_path: Path, root_path: Path, real_video: Path, fake_video: Path, step: int):
    real_video = root_path / real_video
    fake_video = root_path / fake_video
    total_frames = get_video_total_frames(real_video)
    if total_frames != get_video_total_frames(fake_video):
        raise AssertionError(f'{real_video} total frame ({get_video_total_frames(real_video)}) do not equal with {fake_video} ({get_video_total_frames(fake_video)})')
    frame_list = list(range(0, total_frames, step))
    for frame_idx in frame_list:
        real_filename = f"{real_video.stem}_{frame_idx}.png"
        real_save_path = save_path / 'real' / real_filename
        fake_filename = f"{fake_video.stem}_{frame_idx}.png"
        fake_save_path = save_path / 'fake' / fake_filename
        real_frame = video2image(real_video, frame_idx)
        fake_frame = video2image(fake_video, frame_idx)
        if not is_homologous(real_frame=real_frame, fake_frame=fake_frame): # type: ignore
            logger.error('real_frame and fake frame might not from the same source')
            continue
        frame_to_png(real_frame, save_path=real_save_path) # type: ignore
        frame_to_png(fake_frame, save_path=fake_save_path) # type: ignore
        info = {'real': real_save_path, 'fake': fake_save_path}
        pair_label_info.append(info)

# %%
frame_step = 15
save_path = Path('dataset/Celeb-DF-pair')
pair_label_info = list()

# %%
import time

# 核心逻辑：替换 tqdm 为日志输出
total_rows = len(label)
start_time = time.time()  # 记录开始时间

for idx, row in label.iterrows():
    current_row = int(idx) + 1  # type: ignore 
    
    try:
        deal_pair(
            pair_label_info=pair_label_info,
            root_path=dataset_path,
            save_path=save_path,
            real_video=row['real'],
            fake_video=row['fake'],
            step=frame_step
        )
        
        log_interval = 1  # 每处理 10 行输出一次进度
        if current_row % log_interval == 0 or current_row == total_rows:
            # 计算已用时间和剩余时间
            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / current_row  # 每行平均耗时
            remaining_rows = total_rows - current_row
            remaining_time = remaining_rows * avg_time_per_row  # 剩余时间（秒）
            
            # 格式化剩余时间（转换为 时:分:秒，更易读）
            remaining_h = int(remaining_time // 3600)
            remaining_m = int((remaining_time % 3600) // 60)
            remaining_s = int(remaining_time % 60)
            remaining_str = f"{remaining_h:02d}:{remaining_m:02d}:{remaining_s:02d}"
            
            # 输出进度日志
            print(
                ">>> "
                f"处理进度：{current_row}/{total_rows}  "
                f"| 已用时间：{elapsed_time:.1f}s "
                f"| 剩余时间预估：{remaining_str}"
            )
    
    except Exception as e:
        # 异常时记录具体行数和错误信息
        logger.error(f"处理第 {current_row} 行失败（real={row['real']}, fake={row['fake']}）：{str(e)}", exc_info=True)

# %%
print(f'success build {len(pair_label_info)} samples')
labels = pd.DataFrame(pair_label_info)
labels.to_csv(save_path / 'labels.csv')