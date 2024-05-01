import os
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2


def read_segment_frames(
    segment_path,
    labels_path=None,
    load_valid_frames_only=True,
):
    """
    Read video frames for segment
    ---
    Args:
    - segment_path: Path to the segment to read frames from
    - labels_path (defalut = None): Directory where label files are stored
    - load_valid_frames_only (default = True): Whether to load only valid frames
    ---
    Returns:
    - frames: Video frames for the segment
    - fps: Frames per second of the segment
    """

    # Parse segment name
    segment_dir, segment_filename = os.path.split(segment_path)
    segment_name, segment_ext = os.path.splitext(segment_filename)
    if labels_path is None:
        labels_path = os.path.join(segment_dir, os.pardir, "labels")

    # Open capture
    capture = cv2.VideoCapture(segment_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    # Read frames
    frames = []
    while True:
        ret, frame = capture.read()
        if ret:
            frames.append(frame)
        else:
            break

    # Release capture
    capture.release()

    if load_valid_frames_only:
        # Load frame validity
        frame_validity_path = os.path.join(
            labels_path, f"{segment_name}_frame_validity.npy"
        )
        frame_validity = np.load(frame_validity_path, allow_pickle=True)

        # Filter frames
        frames = [frame for i, frame in enumerate(frames) if frame_validity[i]]

    # Return frames & fps
    return frames, fps


def read_segment_info(
    segment_path,
    labels_path=None,
):
    """
    Read info for segment
    ---
    Args:
    - segment_path: Path to the segment to read info from
    - labels_path (defalut = None): Directory where label files are stored
    ---
    Returns:
    - segment_info: Segment info dictionary
    """

    # Parse segment name
    segment_dir, segment_filename = os.path.split(segment_path)
    segment_name, segment_ext = os.path.splitext(segment_filename)
    if labels_path is None:
        labels_path = os.path.join(segment_dir, os.pardir, "labels")

    with open(os.path.join(labels_path, f"{segment_name}_info.json"), "r") as f:
        segment_info = json.load(f)

    return segment_info


def read_segment_2d_annotations(
    segment_path,
    labels_path=None,
    use_pose_bbox=True,
):
    """
    Read annotations for segment
    ---
    Args:
    - segment_path: Path to the segment to read annotations from
    - labels_path (defalut = None): Directory where label files are stored
    ---
    Returns:
    - court_sequence: Court labels for the segment
    - ball_sequence: Ball labels for the segment
    - player_btm_bbox_sequence: Player bottom bounding box labels for the segment
    - player_top_bbox_sequence: Player top bounding box labels for the segment
    - player_btm_pose_sequence: Player bottom 2D pose labels for the segment
    - player_top_pose_sequence: Player top 2D pose labels for the segment
    """

    # Parse segment name
    segment_dir, segment_filename = os.path.split(segment_path)
    segment_name, segment_ext = os.path.splitext(segment_filename)
    if labels_path is None:
        labels_path = os.path.join(segment_dir, os.pardir, "labels")

    # Load court lines
    court_path = os.path.join(labels_path, f"{segment_name}_court.npy")
    court_sequence = np.load(court_path, allow_pickle=True)

    # Load ball
    ball_path = os.path.join(labels_path, f"{segment_name}_ball.npy")
    ball_sequence = np.load(ball_path, allow_pickle=True)

    # Load bounding boxes
    _pose = "_pose" if use_pose_bbox else ""
    player_btm_bbox_path = os.path.join(
        labels_path, f"{segment_name}_player_btm_bbox{_pose}.npy"
    )
    player_btm_bbox_sequence = np.load(player_btm_bbox_path, allow_pickle=True)
    player_top_bbox_path = os.path.join(
        labels_path, f"{segment_name}_player_top_bbox{_pose}.npy"
    )
    player_top_bbox_sequence = np.load(player_top_bbox_path, allow_pickle=True)

    # Load 2D bounding boxes
    player_btm_pose_path = os.path.join(
        labels_path, f"{segment_name}_player_btm_pose.npy"
    )
    player_btm_pose_sequence = np.load(player_btm_pose_path, allow_pickle=True)
    player_top_pose_path = os.path.join(
        labels_path, f"{segment_name}_player_top_pose.npy"
    )
    player_top_pose_sequence = np.load(player_top_pose_path, allow_pickle=True)

    # Return annotations
    return (
        court_sequence,
        ball_sequence,
        player_btm_bbox_sequence,
        player_top_bbox_sequence,
        player_btm_pose_sequence,
        player_top_pose_sequence,
    )


def read_segment_3d_annotations(
    segment_path,
    labels_path=None,
    perform_scaling=True,
    use_rotated=True,
):
    """
    Read annotations for segment
    ---
    Args:
    - segment_path: Path to the segment to read annotations from
    - labels_path (defalut = None): Directory where label files are stored
    - perform_scaling (default = True): Whether to perform scaling on the 3D poses and 2D positions
    - use_rotated (default = True): Whether to use the rotated versions of the 3D poses.
    ---
    Returns:
    - player_btm_2D_position:  Player bottom 2D court position labels for the segment
    - player_top_2D_position: Player top 2D court position labels for the segment
    - player_btm_pose_sequence: Player bottom 3D pose labels for the segment
    - player_top_pose_sequence: Player top 3D pose labels for the segment
    """

    # Parse segment name
    segment_dir, segment_filename = os.path.split(segment_path)
    segment_name, segment_ext = os.path.splitext(segment_filename)
    if labels_path is None:
        labels_path = os.path.join(segment_dir, os.pardir, "labels")

    # Load 2D court position
    player_btm_position_path = os.path.join(
        labels_path, f"{segment_name}_player_btm_position.npy"
    )
    player_btm_position_sequence = np.load(player_btm_position_path, allow_pickle=True)
    player_top_position_path = os.path.join(
        labels_path, f"{segment_name}_player_top_position.npy"
    )
    player_top_position_sequence = np.load(player_top_position_path, allow_pickle=True)

    # Load 3D poses
    _rot = "_rot" if use_rotated else ""
    player_btm_pose_3d_path = os.path.join(
        labels_path, f"{segment_name}_player_btm_pose_3d{_rot}.npy"
    )
    player_btm_pose_3d_sequence = np.load(player_btm_pose_3d_path, allow_pickle=True)
    player_top_pose_3d_path = os.path.join(
        labels_path, f"{segment_name}_player_top_pose_3d{_rot}.npy"
    )
    player_top_pose_3d_sequence = np.load(player_top_pose_3d_path, allow_pickle=True)

    # Perform scaling
    if perform_scaling:
        player_btm_position_sequence = scale_2d_positions(player_btm_position_sequence)
        player_top_position_sequence = scale_2d_positions(player_top_position_sequence)
        player_btm_pose_3d_sequence = scale_3d_poses(player_btm_pose_3d_sequence)
        player_top_pose_3d_sequence = scale_3d_poses(player_top_pose_3d_sequence)

    # Return annotations
    return (
        player_btm_position_sequence,
        player_top_position_sequence,
        player_btm_pose_3d_sequence,
        player_top_pose_3d_sequence,
    )


def rotate_pose_3d_to_match_2d(pose_3d, pose_2d, visualize=False):
    """
    Rotate 3D pose to match 2D pose
    ---
    Args:
    - pose_3d: 3D pose to rotate
    - pose_2d: 2D pose to match
    - visualize: Whether to visualize rotation
    ---
    Returns:
    - pose_3d: Rotated 3D pose
    """

    if pose_3d is None or np.any(pose_3d == None):
        print("Warning: invalid 3d pose")
        return pose_3d

    if pose_2d is None or np.any(pose_2d == None):
        print("Warning: invalid 2d pose")
        return pose_3d

    # Match elbows, knees, shoulders, hips & feet
    pose_2d_indices = np.array([10, 9, 8, 7, 14, 13, 5, 6, 11, 12, 15, 16], dtype=int)
    pose_3d_indices = np.array([13, 16, 12, 15, 5, 2, 14, 11, 1, 4, 3, 6], dtype=int)

    # Extract keypoints
    pose_2d_keypoints = np.copy(pose_2d[pose_2d_indices]).astype(
        np.float32
    )  # N by (X, Z)
    pose_3d_keypoints = np.copy(pose_3d[pose_3d_indices]).astype(
        np.float32
    )  # N by (X', Y', Z')

    # Define 2D keypoint targets in 3D
    pose_3d_target = np.transpose(
        np.stack(
            (
                pose_2d_keypoints[:, 0],  # X
                np.zeros(pose_2d_keypoints.shape[0], dtype=np.float32),  # Y
                -pose_2d_keypoints[:, 1],  # Z
            )
        )
    )

    # Find centroids
    centroid_s = np.expand_dims(np.mean(pose_3d_keypoints, axis=0), axis=0)
    centroid_t = np.expand_dims(np.mean(pose_3d_target, axis=0), axis=0)

    # Center
    pose_3d_keypoints = pose_3d_keypoints - centroid_s
    pose_3d_target = pose_3d_target - centroid_t

    # Find scales
    scale_s = np.mean(np.linalg.norm(pose_3d_keypoints, axis=1))
    try:
        scale_t = np.mean(np.linalg.norm(pose_3d_target, axis=1))
    except Exception as e:
        print(pose_3d_keypoints)
        print(pose_3d_target)
        raise e

    # Normalize
    pose_3d_keypoints = pose_3d_keypoints / scale_s
    pose_3d_target = pose_3d_target / scale_t

    # Find rotation matrix
    H = np.matmul(
        np.transpose(pose_3d_keypoints),
        (pose_3d_target),
    )
    U, S, V_transpose = np.linalg.svd(H)
    R = np.matmul(np.transpose(V_transpose), np.transpose(U))

    # Handle special case
    if np.linalg.det(R) < 0:
        V_transpose[2, :] *= -1
        R = np.matmul(np.transpose(V_transpose), np.transpose(U))

    # Visualize transformation
    if visualize:
        # Create figure &  axis
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        # Visualize source & target
        ax.scatter(
            pose_3d_keypoints[:, 0],
            pose_3d_keypoints[:, 1],
            pose_3d_keypoints[:, 2],
            marker=".",
            color="red",
        )
        ax.scatter(
            pose_3d_target[:, 0],
            pose_3d_target[:, 1],
            pose_3d_target[:, 2],
            marker="o",
            color="green",
        )

        # Visualize source after rotation
        keypoints_transformed = np.matmul(pose_3d_keypoints, np.transpose(R))
        ax.scatter(
            keypoints_transformed[:, 0],
            keypoints_transformed[:, 1],
            keypoints_transformed[:, 2],
            marker="x",
            color="blue",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_aspect("equal", adjustable="box")
        plt.show()

    # Apply rotation to source points
    pose_3d_transformed = np.copy(pose_3d)
    pose_3d_transformed -= centroid_s
    # pose_3d_transformed /= scale_s -> Keep orignal scale
    pose_3d_transformed = np.matmul(pose_3d_transformed, np.transpose(R))
    z_min = np.min(pose_3d_transformed[:, 2], axis=0)

    # Make sure player is not below ground
    pose_3d_transformed[:, 2] -= z_min

    return pose_3d_transformed


def rotate_3d_poses(
    poses_3d,
    poses_2d,
):
    """
    Rotate 3D poses to match 2D poses
    ---
    Args:
    - poses_3d: 3D poses to rotate
    - poses_2d: 2D poses to match
    ---
    Returns:
    - poses_3d: Rotated 3D poses
    """

    # Array of transformed poses
    poses_transformed = np.zeros_like(poses_3d)

    # Transform pose by pose
    for i, (pose_3d, pose_2d) in enumerate(zip(poses_3d, poses_2d)):
        # Rotate 3D pose to match 2D pose
        pose_transformed = rotate_pose_3d_to_match_2d(pose_3d, pose_2d)

        # Store pose
        poses_transformed[i] = pose_transformed

    # Return transformed poses
    return poses_transformed


def scale_3d_poses(
    poses_3d,
    torso_target_height=0.66,
):
    """
    Scale 3D poses
    ---
    Args:
    - poses_3d: 3D poses to scale
    - torso_target_height (default = 0.41): Target height of the player's torso (unit of choice, default = m)
    ---
    Returns:
    - poses_3d: Scaled 3D poses
    """
    # Create a copy
    poses_3d = np.copy(poses_3d)

    # Calculate current height of torso
    torso_height = np.sqrt(
        np.sum((poses_3d[:, 0, :] - poses_3d[:, 7, :]) ** 2, axis=1)
    )  # hips to belly
    torso_height += np.sqrt(
        np.sum((poses_3d[:, 7, :] - poses_3d[:, 8, :]) ** 2, axis=1)
    )  # belly to neck

    # Scale 3D poses
    scale_factor = torso_target_height / torso_height
    poses_3d *= np.expand_dims(scale_factor, (1, 2))

    return poses_3d


def scale_2d_positions(
    positions_2d,
    court_width=10.97,
    court_length=23.77,
):
    """
    Scale 2D positions
    ---
    Args:
    - positions_2d: 2D positions to scale
    - court_width (default = 10.97): Width of the court (unit of choice, default = m)
    - court_length (default = 23.77): Length of the court (unit of choice, default = m)
    ---
    Returns:
    - positions_2d: Scaled 2D positions
    """

    # Create a copy
    positions_2d = np.copy(positions_2d)

    # Scale 2D positions
    positions_2d[:, 0] *= court_width
    positions_2d[:, 1] *= court_length

    return positions_2d
