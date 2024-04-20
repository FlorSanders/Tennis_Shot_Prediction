from pose_dataclasses import PlayerPose
import numpy as np

def keep_largest_volume_3D_pose_heuristic(player_poses: list[PlayerPose]):
    # Get the volume of each player
    player_volumes = [pose.get_volume() for pose in player_poses]

    # Find the player with the largest volume
    largest_volume_idx = np.argmax(player_volumes)

    # Return the pose of the player with the largest volume
    return player_poses[largest_volume_idx]


def keep_largest_xspan_3D_pose_heuristic(player_poses: list[PlayerPose]):
    player_xspans = [pose.get_x_span() for pose in player_poses]
    largest_volume_idx = np.argmax(player_xspans)
    return player_poses[largest_volume_idx]


def keep_largest_yspan_3D_pose_heuristic(player_poses: list[PlayerPose]):
    player_yspans = [pose.get_volume() for pose in player_poses]
    largest_volume_idx = np.argmax(player_yspans)
    return player_poses[largest_volume_idx]
