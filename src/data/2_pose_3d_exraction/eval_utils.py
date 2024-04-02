import numpy as np
import torch
from torch import nn


def run_evaluation(actions, keypoints, loader):
    action_predictions = {}
    for action_key in actions.keys():
        predictions = evaluate(gen, action_key)
        action_predictions[action_key] = predictions
    return action_predictions


def get_out_poses_2d(actions, keypoints, stride):
    out_poses_2d = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        num_cameras = len(poses_2d)
        for i in range(num_cameras):
            out_poses_2d.append(poses_2d[i])

    # stride = args.downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
    
    return out_poses_2d


def evaluate(model: nn.module, test_generator, return_predictions=False):
    with torch.no_grad():
        model.eval()
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            predicted_3d_pos = model(inputs_2d)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
                
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]
            
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])