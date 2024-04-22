# visualize
from mmpose.apis import visualize
import os
import numpy as np


class MapVisualizer:
    def __init__(self, video_segments_dir, pose_dir):
        self.video_segments_dir = video_segments_dir
        self.pose_dir = pose_dir

    def visualize():
        # Load the .npy file
        sample_file = os.path.join(write_path, 'V009_0061_player_top_pose_3d.npy')
        # sample_file = os.path.join(labels_path, 'V010_0071_player_top_bbox.npy')
        data = np.load(sample_file, allow_pickle=True)

        # extract first frame from segment
        img_path = os.path.join(write_path, '000000.jpg')
        keypoints = np.load(sample_file, allow_pickle=True)
        keypoint_scores = None



        metainfo = 'config/_base_/datasets/coco.py'

        visualize(
            img_path,
            keypoints,
            keypoint_scores,
            metainfo=metainfo,
            show=True)