# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch
from torch import nn
import os
import errno

from common.camera import *
from common.model import *
from common.loss import *

from common.utils import deterministic_random
from common.custom_dataset import CustomDataset
from common.visualization import render_animation
from common.generators import UnchunkedGenerator

from model_utils import load_position_model, get_model_metadata
from eval_utils import run_evaluation, get_out_poses_2d

INPUT_DATASET_PATH = 'data/data_2d_custom_keypoints.npz'

def create_checkpoint_dir() -> None:
    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

def load_dataset() -> CustomDataset:
    return CustomDataset(INPUT_DATASET_PATH)

def prepare_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d
    return dataset

def load_2d_detections(dataset_skeleton):
    keypoints = np.load(INPUT_DATASET_PATH, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset_skeleton.joints_left()), list(dataset_skeleton.joints_right())
    keypoints = keypoints['positions_2d'].item()
    return kps_left, kps_right, keypoints, joints_left, joints_right

def postprocess_keypoints(keypoints, dataset):
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue
                
            for cam_idx in range(len(keypoints[subject][action])):
                
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps


def fetch(subjects, keypoints, dataset, subset=1, stride=1):
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            poses_2d: list = keypoints[subject][action]
            num_cameras = len(poses_2d)
            for i in range(num_cameras): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == num_cameras, 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
                
    if len(out_camera_params) == 0:
        out_camera_params = None
    
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
    

    return out_camera_params, out_poses_2d


def get_unique_actions(subjects, dataset) -> dict:
    all_actions = {}
    for subject in subjects:
        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            all_actions[action_name].append((subject, action))
    return all_actions

def render(prediction, args, keypoints, dataset):
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)
    
    if args.viz_output is not None:
        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        # Since the 3D ground truth is not available, take the camera extrinsic params from a random subject.
        # They are almost the same, and anyway, we only need this for visualization purposes.
        for subject in dataset.cameras():
            if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                break

        prediction = camera_to_world(prediction, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        
        anim_output = {'Reconstruction': prediction}
        
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
        
        render_animation(input_keypoints,
                         keypoints_metadata,
                         anim_output,
                         dataset.skeleton(),
                         dataset.fps(),
                         args.viz_bitrate,
                         cam['azimuth'],
                         args.viz_output,
                         limit=args.viz_limit, 
                         downsample=args.viz_downsample, 
                         size=args.viz_size,
                         input_video_path=args.viz_video, 
                         viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip
                        )


def main():
    print('Starting up eval script...')
    print('Parsing arguments...')
    args = parse_args()
    print(args)

    print('Loading and preparing source data...')
    create_checkpoint_dir()
    dataset = load_dataset()
    dataset = prepare_data(dataset)

    print('Loading 2D keypoints...')
    kps_left, kps_right, keypoints, joints_left, joints_right = load_2d_detections(dataset_skeleton=dataset.skeleton())
    keypoints = postprocess_keypoints(dataset)
        
    print('Prepping data for 3D extraction...')
    out_camera_params, out_poses_2d = fetch(subjects_test, keypoints, dataset, stride=args.downsample)

    print('Loading models...')
    model_pos = load_position_model(args=args, num_joints=dataset.skeleton().num_joints())
    # model_traj = load_trajectory_model(args)
    receptive_field, causal_shift, pad = get_model_metadata(use_causal_convolutions=args.causal, model=model_pos)

    print('Executing 3D extraction...')
    actions = get_unique_actions(subjects=subjects_test, dataset=dataset)
    poses_2d_actual = get_out_poses_2d(actions, keypoints, stride=args.downsample)
    loader = UnchunkedGenerator(None,
                                None,
                                poses_2d_actual,
                                pad=pad,
                                causal_shift=causal_shift,
                                augment=args.test_time_augmentation,
                                kps_left=kps_left,
                                kps_right=kps_right,
                                joints_left=joints_left,
                                joints_right=joints_right)
    predictions_3d = run_evaluation(actions, keypoints, loader)
    print('Saving prerdictions...')
    # TODO

    print('Rendering predictions...')
    render(prediction=predictions_3d,
           args=args,
           keypoints=keypoints,
           dataset=dataset)

if __name__ == main:
    main()