"""
The poselifter saved the 3D coordinates in the format as an array of PlayerPose objects. 
Rather, they should just be saved as an array of keypoint coordinates.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Labels path
labels_path = os.path.abspath(
    os.path.join(
        __file__, os.pardir, os.pardir, os.pardir, "data", "tenniset", "labels"
    )
)

# Load 3d pose files
pose_3d_files = np.sort(glob.glob(os.path.join(labels_path, "*3d*.npy")))

n_fixed = 0
for file_path in tqdm(pose_3d_files):
    poses_3d = np.load(file_path, allow_pickle=True)
    if len(poses_3d.shape) != 3:
        poses_3d = np.asarray([pose.item().pose for pose in poses_3d])
        np.save(file_path, poses_3d)
        n_fixed += 1
print(f"Fixed {n_fixed} pose files.")
