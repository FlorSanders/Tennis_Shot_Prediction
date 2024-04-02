import torch
from torch import nn
import os

def load_checkpoint(args):
    chk_filename = os.path.join(args.checkpoint, args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    return checkpoint


def load_position_model(args, num_joints) -> nn.module:
    filter_widths = [int(x) for x in args.architecture.split(',')]
    model_pos = TemporalModel(poses_valid_2d[0].shape[-2],
                              poses_valid_2d[0].shape[-1],
                              num_joints,
                              filter_widths=filter_widths,
                              causal=args.causal,
                              dropout=args.dropout,
                              channels=args.channels,
                              dense=args.dense)
    
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    checkpoint = load_checkpoint(args)
    model_pos.load_state_dict(checkpoint['model_pos'])

    return model_pos


def load_trajectory_model(args) -> nn.module | None:
    filter_widths = [int(x) for x in args.architecture.split(',')]
    
    checkpoint = load_checkpoint(args)

    if 'model_traj' in checkpoint:
        # Load trajectory model if it contained in the checkpoint (e.g. for inference in the wild)
        model_traj = TemporalModel(poses_valid_2d[0].shape[-2],
                                   poses_valid_2d[0].shape[-1],
                                   1,
                                   filter_widths=filter_widths,
                                   causal=args.causal,
                                   dropout=args.dropout, 
                                   channels=args.channels,
                                   dense=args.dense)
        if torch.cuda.is_available():
            model_traj = model_traj.cuda()
        model_traj.load_state_dict(checkpoint['model_traj'])
    else:
        model_traj = None
    return model_traj


def get_model_metadata(use_causal_convolutions: bool, model: nn.module):
    receptive_field = model.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2 # Padding on each side
    if use_causal_convolutions:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    return receptive_field, causal_shift, pad