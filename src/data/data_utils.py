from __init__ import data_path
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging

logger = logging.getLogger(__name__)


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
    print(segment_path)
    assert os.path.exists(segment_path), "Segment path does not exist"

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
        # Release capture
        frame_validity, _, _, _, _, _, _ = read_segment_labels(
            segment_path,
            labels_path,
            load_frame_validity=True,
            load_court=False,
            load_ball=False,
            load_player_bbox=False,
            load_player_pose=False,
        )

        # Filter frames
        frames = [frame for i, frame in enumerate(frames) if frame_validity[i]]

    # Return frames & fps
    return frames, fps


def read_segment_labels(
    segment_path,
    labels_path=None,
    load_frame_validity=True,
    load_court=True,
    load_ball=True,
    load_player_bbox=True,
    load_player_pose=True,
    use_pose_bbox=True,
):
    """
    Read labels for given segments
    ---
    Args:
    - segment_path: Path to the segment to read labels from
    - labels_path (defalut = None): Directory where label files are stored
    - load_frame_validity (default = True): Whether to load frame validity labels
    - load_court (default = True): Whether to load court labels
    - load_ball (default = True): Whether to load ball labels
    - load_player_bbox (default = True): Whether to load player bounding box labels
    ---
    Returns:
    - frame_validity: Frame validity labels for the segment
    - court_sequence: Court labels for the segment
    - ball_sequence: Ball labels for the segment
    - player_btm_bbox_sequence: Player bottom bounding box labels for the segment
    - player_top_bbox_sequence: Player top bounding box labels for the segment
    """

    # Parse segment name
    segment_dir, segment_filename = os.path.split(segment_path)
    segment_name, segment_ext = os.path.splitext(segment_filename)
    if labels_path is None:
        labels_path = os.path.join(segment_dir, os.pardir, "labels")

    # Read labels
    frame_validity = None
    court_sequence = None
    ball_sequence = None
    player_btm_bbox_sequence = None
    player_top_bbox_sequence = None
    player_btm_pose_sequence = None
    player_top_pose_sequence = None
    # Load frame validity
    if load_frame_validity:
        frame_validity_path = os.path.join(
            labels_path, f"{segment_name}_frame_validity.npy"
        )
        frame_validity = np.load(frame_validity_path, allow_pickle=True)
    # Load court
    if load_court:
        court_path = os.path.join(labels_path, f"{segment_name}_court.npy")
        court_sequence = np.load(court_path, allow_pickle=True)
    # Load ball
    if load_ball:
        ball_path = os.path.join(labels_path, f"{segment_name}_ball.npy")
        ball_sequence = np.load(ball_path, allow_pickle=True)
    # Load player bounding boxes
    if load_player_bbox:
        player_btm_bbox_path = os.path.join(
            labels_path, f"{segment_name}_player_btm_bbox_pose.npy"
        )
        if not os.path.exists(player_btm_bbox_path) or not use_pose_bbox:
            player_btm_bbox_path = os.path.join(
                labels_path, f"{segment_name}_player_btm_bbox.npy"
            )
        player_btm_bbox_sequence = np.load(player_btm_bbox_path, allow_pickle=True)
        player_top_bbox_path = os.path.join(
            labels_path, f"{segment_name}_player_top_bbox_pose.npy"
        )
        if not os.path.exists(player_top_bbox_path) or not use_pose_bbox:
            player_top_bbox_path = os.path.join(
                labels_path, f"{segment_name}_player_top_bbox.npy"
            )
        player_top_bbox_sequence = np.load(player_top_bbox_path, allow_pickle=True)
    # Load player poses
    if load_player_pose:
        player_btm_pose_path = os.path.join(
            labels_path, f"{segment_name}_player_btm_pose.npy"
        )
        player_btm_pose_sequence = np.load(player_btm_pose_path, allow_pickle=True)
        player_top_pose_path = os.path.join(
            labels_path, f"{segment_name}_player_top_pose.npy"
        )
        player_top_pose_sequence = np.load(player_top_pose_path, allow_pickle=True)

    # Return labels
    return (
        frame_validity,
        court_sequence,
        ball_sequence,
        player_btm_bbox_sequence,
        player_top_bbox_sequence,
        player_btm_pose_sequence,
        player_top_pose_sequence,
    )


def crop_frame(frame,
              bbox,
              crop_padding=50,
            crop_img_width=256):
    # Frame size
    frame_height, frame_width = frame.shape[:2]

    # Parse bounding box coords
    # if np.any(bbox == None):
    #     return best_keypoints, best_bbox

    x1, y1, x2, y2 = bbox
    xc, yc =  (x1 + x2) / 2, (y1 + y2) / 2
    w, h = abs(x2 - x1), abs(y2 - y1)
    d = max(w, h) + crop_padding * 2

    # Define cropping indices
    x_crop1, x_crop2 = int(xc - d/2), int(xc + d/2)
    y_crop1, y_crop2 = int(yc - d/2), int(yc + d/2)

    # Make sure we don't crop past the edges of the frame
    x_crop_offset = min(frame_width - x_crop2, max(-x_crop1, 0))
    y_crop_offset = min(frame_height - y_crop2, max(-y_crop1, 0))
    x_crop1 += x_crop_offset
    x_crop2 += x_crop_offset
    y_crop1 += y_crop_offset
    y_crop2 += y_crop_offset
    
    # Crop image
    img = frame[y_crop1:y_crop2,  x_crop1:x_crop2].copy()

    # Resize img
    scale = d / crop_img_width
    img = cv2.resize(img, (crop_img_width, crop_img_width))
    return img


def clean_bbox_sequence(
    bbox_sequence, 
    court_sequence, 
    is_btm,
    derivative_threshold=5000,
    make_plot=False,
):
    # Look at center points to gather inconsistencies
    center_points = np.zeros((len(bbox_sequence), 2))
    bbox_areas = np.zeros(len(bbox_sequence))
    bbox_sequence_clean = np.copy(bbox_sequence)
    missing_points = np.zeros(len(center_points), dtype=int)

    # Extract center points wrt court from 
    for i, (bbox, court_points) in enumerate(zip(bbox_sequence, court_sequence)):
        # Skip no bounding box detected
        if np.any(bbox == None):
            center_points[i, :] = np.inf
            continue
        xb1, yb1, xb2, yb2 = bbox
        bbox_areas[i] = np.abs((xb2 - xb1) * (yb2 - yb1))

        # Skip no court outline detected
        court_outline = court_points[:4]
        if np.any(court_outline == None):
            center_points[i, :] = np.inf
            continue
        
        # Get relevant center point of the court
        (xtl, ytl), (xtr, ytr), (xbl, ybl), (xbr, ybr) = court_outline
        x_ref = (xbl + xbr) / 2 if is_btm else (xtl + xtr) / 2
        y_ref = (ybl + ybr) / 2 if is_btm else (ytl + ytr) / 2

        # Get center point of the player's feet
        x_player = (xb1 + xb2) / 2
        y_player = yb2

        # Save player center point referenced to court point
        center_points[i, 0] = x_player - x_ref
        center_points[i, 1] = y_player - y_ref

    # Compute first derivative
    center_points_derivative = np.vstack(([[0, 0]], center_points[:-1] - center_points[1:]))
    center_points_derivative = center_points_derivative[:,0]**2 + center_points_derivative[:,1]**2
    bbox_areas_derivative = np.abs(np.concatenate(([0], bbox_areas[:-1] - bbox_areas[1:])))

    # Area jumps
    bbox_area_jumps = np.sort(np.argwhere(bbox_areas_derivative > derivative_threshold).reshape(-1))
    if len(bbox_area_jumps):
        # print("JUMPS DETECTED")
        # print(bbox_area_jumps)
        mean_area = np.mean(bbox_areas[:bbox_area_jumps[0]])
    else:
        mean_area = np.mean(bbox_areas)

    # Determine jump points
    jump_points = np.argwhere(np.logical_or(
        center_points_derivative > derivative_threshold,
        bbox_areas < mean_area / 2,
    )).reshape(-1)
    
    # Return if no cleaning needs to be done
    if len(jump_points) == 0:
        return missing_points.astype(bool), bbox_sequence_clean

    # Process jump points
    indx_last = None
    missing_start = False
    for indx in jump_points:
        #print(indx_last, indx)
        if indx_last is None:
            # First missing point
            #print("FIRST MISSING POINT")
            missing_points[indx] = 1
            missing_start = True
        elif np.any(bbox_sequence[indx] == None):
            # Missing point
            #print("MISSING POINT", indx)
            missing_points[indx] = 1
            missing_start = False
        elif indx_last == indx - 1:
            # Subsequent problematic points
            #print("SUBSEQUENT MISSING POINT")
            missing_points[indx] = 1
            missing_start = False
        else:
            # Distance between missing points
            #print("DISTANCE BETWEEN MISSING POINTS")
            if missing_start:
                # End point (hopefully)
                missing_points[indx_last:indx+1] = 1
                missing_start = False
            else:
                # Start point (hopefully)
                missing_points[indx] = 1
                missing_start = True

        # Update last indx
        indx_last = indx

    # Fill gaps in missing points by linear interpolation
    filled_center_points = np.copy(center_points)
    missing_starts = np.argwhere((missing_points[1:] - missing_points[:-1]) == 1).reshape(-1)
    missing_ends = np.argwhere((missing_points[1:] - missing_points[:-1]) == -1).reshape(-1)
    for i, missing_start in enumerate(missing_starts):
        # Get start value
        if missing_start != 0:
            # Previous value
            cp_start_value = filled_center_points[missing_start-1]
            bbox_start_value = bbox_sequence_clean[missing_start-1]
        else:
            # First valid value (TODO: fix if none is valid???)
            cp_start_value = filled_center_points[not missing_points.astype(bool)][0]
            bbox_start_value = bbox_sequence_clean[not missing_points.astype(bool)][0]

        # Get missing end
        if len(missing_ends) <= i:
            # No matched endpoint - constant from startpoint onward
            missing_end = len(filled_center_points) - 1
            cp_end_value = cp_start_value
            bbox_end_value = bbox_start_value
        else:
            # Get endpoint
            missing_end = missing_ends[i]
            cp_end_value = filled_center_points[missing_end]
            bbox_end_value = bbox_sequence_clean[missing_end]
            
        # Linearly interpolate
        n_points = missing_end - missing_start + 1
        filled_center_points[missing_start:missing_end+1] = np.linspace(cp_start_value, cp_end_value, n_points)
        bbox_sequence_clean[missing_start:missing_end+1] = np.linspace(bbox_start_value, bbox_end_value, n_points)

    return missing_points.astype(bool), bbox_sequence_clean


def visualize_frame_annotations(
    frame,
    court_points,
    ball_point,
    player_btm_bbox,
    player_top_bbox,
    player_btm_pose,
    player_top_pose,
    show_court_numbers=True,
    show_img=False,
):
    """
    Visualize frame annotations
    ---
    Args:
    - frame: Frame to visualize annotations on
    - court_points: Court points to visualize
    - ball_point: Ball point to visualize
    - player_btm_bbox: Player bottom bounding box to visualize
    - player_top_bbox: Player top bounding box to visualize
    - show_court_numbers (default = True): Whether to show court point numbers
    - show_img (default = True): Whether to show image using matplotlib
    ---
    Returns:
    - img: Visualized image
    """

    # Copy frame
    img = frame.copy()

    # Show court points
    if court_points is not None:
        for i, (x, y) in enumerate(court_points):
            if x is None or y is None:
                continue
            img = cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 5)
            if show_court_numbers:
                img = cv2.putText(
                    img,
                    str(i),
                    (int(x) + 15, int(y) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    # Show ball
    if ball_point is not None:
        x, y = ball_point
        if x is not None and y is not None:
            img = cv2.circle(img, (int(x), int(y)), 5, (0, 180, 255), 5)

    # Show player bounding boxes
    colors = [(255, 0, 0), (0, 255, 0)]
    for i, bbox in enumerate([player_btm_bbox, player_top_bbox]):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue
            img = cv2.rectangle(
                img, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2
            )

    # Show player poses
    pose_lines = [
        (0, 1),  # nose
        (0, 2),  # nose
        (1, 2),  # eyes
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (5, 6),  # shoulders
        (5, 11),
        (6, 12),
        (11, 12),  # hips
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]
    for i, pose in enumerate([player_btm_pose, player_top_pose]):
        # Skip invalid poses
        if pose is None or np.any(pose == None):
            continue

        for k, l in pose_lines:
            x1, y1 = pose[k]
            x2, y2 = pose[l]
            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue
            img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)

    #  Show image
    if show_img:
        fig, ax = plt.subplots()
        ax.imshow(img[:, :, ::-1])
        plt.show()

    return img


def visualize_segment_labels(
    segment_path,
    out_path=None,
    labels_path=None,
    overwrite=False,
    show_first_frame=False,
):
    """
    Visualize segment labels
    ---
    Args:
    - segment_path: Path to the segment to visualize labels for
    - out_path (default = None): Path to save the visualized video to
    - labels_path (default = data_path): Directory where label files are stored
    - overwrite (default = False): Whether to overwrite existing output video
    - show_first_frame (default = False): Whether to show the first frame of the video using matplotlib
    ---
    Returns:
    - out_path: Path to the visualized video
    """

    # Parse segment name
    segment_dir, segment_filename = os.path.split(segment_path)
    segment_name, segment_ext = os.path.splitext(segment_filename)
    if labels_path is None:
        labels_path = os.path.join(segment_dir, os.pardir, "labels")

    # Determine output path
    default_out_path = os.path.join(labels_path, f"{segment_name}_annotated.mp4")
    out_path = out_path if out_path is not None else default_out_path
    if os.path.exists(out_path) and not overwrite:
        return out_path

    # Read frames
    frames, fps = read_segment_frames(segment_path, load_valid_frames_only=False)
    if frames is None or len(frames) == 0:
        return None
    frame_height, frame_width, _ = frames[0].shape
    if not len(frames):
        return None

    # Read labels
    (
        frame_validity,
        court_sequence,
        ball_sequence,
        player_btm_bbox_sequence,
        player_top_bbox_sequence,
        player_btm_pose_sequence,
        player_top_pose_sequence,
    ) = read_segment_labels(segment_path, labels_path=labels_path)

    if np.count_nonzero(frame_validity) == 0:
        return None
    elif np.count_nonzero(frame_validity) == 1:
        court_sequence = np.expand_dims(court_sequence, axis=0)
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    # Visualize & write video
    frame_index = 0
    for i, frame in enumerate(frames):
        # Skip invalid frames
        if not frame_validity[i]:
            continue
        frame = visualize_frame_annotations(
            frame,
            court_sequence[frame_index],
            ball_sequence[frame_index],
            player_btm_bbox_sequence[frame_index],
            player_top_bbox_sequence[frame_index],
            player_btm_pose_sequence[frame_index],
            player_top_pose_sequence[frame_index],
            show_court_numbers=True,
            show_img=frame_index == 0 and show_first_frame,
        )
        writer.write(frame)
        frame_index += 1

    # Release video
    writer.release()

    return out_path
