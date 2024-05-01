import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import json
import cv2
from data import (
    read_segment_frames,
    read_segment_info,
    read_segment_2d_annotations,
    read_segment_3d_annotations,
    scale_2d_positions,
    scale_3d_poses,
)


def plot_img(img):
    """
    Plot image
    ---
    Args:
    - img: Image to plot
    ---
    Returns:
    - fig: Figure
    - ax: Axis
    """

    fig, ax = plt.subplots()
    ax.imshow(img[:, :, ::-1])
    ax.set_axis_off()
    return fig, ax


def fig_to_img(fig, dpi=1000):
    """
    Convert matplotlib figure to image array
    ---
    Args:
    - fig: Figure to convert
    ---
    Returns:
    - img: Image
    """
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3][:, :, ::-1]
    return img


def crop_img_to_bbox(
    img,
    bbox,
    padding=10,
    square=True,
    resize_to=None,
    show_img=False,
):
    """
    Crop image to bounding box
    ---
    Args:
    - img: Image to crop
    - bbox: Bounding box to crop to
    - padding (default = 10): Padding around bounding box
    - square (default = True): Whether to crop to square
    - resize_to (default = None): Resize crop to this width
    ---
    Returns:
    - cropped_img: Cropped image
    """

    # Parse bounding box coordinates
    frame_h, frame_w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = abs(x2 - x1), abs(y2 - y1)
    if square:
        d = max(w, h)
        w, h = d, d

    # Crop image
    cropped_img = np.copy(
        img[
            max(int(y_c - h // 2 - padding), 0) : min(
                int(y_c + h // 2 + padding), frame_h
            ),
            max(int(x_c - w // 2 - padding), 0) : min(
                int(x_c + w // 2 + padding), frame_w
            ),
        ]
    )

    # Resize image to square
    if resize_to is not None:
        scale_factor = resize_to / (w + 2 * padding) * (h + 2 * padding)
        cropped_img = cv2.resize(
            cropped_img,
            (resize_to, int(round(scale_factor))),
        )

    # Show image
    if show_img:
        fig, ax = plot_img(cropped_img)
        plt.show()

    return cropped_img


def crop_img_to_content(img):
    """
    Crop image to content
    ---
    Args:
    - img: Image to crop
    ---
    Returns:
    - img: Cropped image
    """

    # Copy original image
    img = np.copy(img)

    # Average over color channels
    img_mean = np.mean(img, axis=2)

    # Crop h-axis
    img_h_mean = np.mean(img_mean, axis=1)
    content = np.argwhere(img_h_mean != 255).reshape(-1)
    start, end = content[0], content[-1] + 1
    img = img[start:end, :, :]

    # Crop w-axis
    img_w_mean = np.mean(img_mean, axis=0)
    content = np.argwhere(img_w_mean != 255).reshape(-1)
    start, end = content[0], content[-1] + 1
    img = img[:, start:end, :]

    return img


def add_img_border(img, color=[0, 0, 0], thickness=2):
    """
    Add border to image
    ---
    Args:
    - img: Image to add border to
    - color (default = np.array([0, 0, 0])): Color of border
    - thickness (default = 2): Thickness of border
    ---
    Returns:
    - img: Image with border added
    """

    # Add border
    img[:thickness, :] = color
    img[-thickness:, :] = color
    img[:, :thickness, :] = color
    img[:, -thickness:, :] = color

    # return img
    return img


def plot_2d_pose(
    pose,
    fig=None,
    ax=None,
    color="tab:blue",
    marker=".",
):
    """
    Plot 2D pose
    ---
    Args:
    - pose: 2D pose to plot
    - fig (default = None): Figure to plot on
    - ax (default = None): Axis to plot on
    - color (default = "tab:blue"): Color of pose
    - marker (default = "."): Marker type of pose
    ---
    Returns:
    - fig: Figure
    - ax: Axis
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Define pose lines
    pose_lines = [
        np.array([0, 1], dtype=int),  # nose
        np.array([0, 2], dtype=int),  # nose
        np.array([1, 2], dtype=int),  # eyes
        np.array([1, 3], dtype=int),
        np.array([2, 4], dtype=int),
        np.array([3, 5], dtype=int),
        np.array([4, 6], dtype=int),
        np.array([5, 7], dtype=int),
        np.array([6, 8], dtype=int),
        np.array([7, 9], dtype=int),
        np.array([8, 10], dtype=int),
        np.array([5, 6], dtype=int),  # shoulders
        np.array([5, 11], dtype=int),
        np.array([6, 12], dtype=int),
        np.array([11, 12], dtype=int),  # hips
        np.array([11, 13], dtype=int),
        np.array([12, 14], dtype=int),
        np.array([13, 15], dtype=int),
        np.array([14, 16], dtype=int),
    ]

    # Plot pose
    for line in pose_lines:
        ax.plot(pose[line, 0], -pose[line, 1], color=color, marker=marker)

    # Fix scaling
    ax.set_aspect("equal", adjustable="box")

    # Return figure and axis
    return fig, ax


def visualize_frame_2d_annotations(
    frame,
    court_points,
    ball_point,
    player_btm_bbox,
    player_top_bbox,
    player_btm_pose,
    player_top_pose,
    show_court=True,
    show_court_numbers=False,
    show_ball=True,
    show_bbox=True,
    show_pose=True,
    show_picture_in_picture=True,
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
    - show_picture_in_picture: Whether to visualize annotations in picture in picture mode
    - show_court_numbers (default = True): Whether to show court point numbers
    - show_img (default = True): Whether to show image using matplotlib
    ---
    Returns:
    - img: Visualized image
    """

    # Copy frame
    img = frame.copy()
    height, width, _ = img.shape

    # Show court points
    if court_points is not None and show_court:
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
    colors = [(0, 255, 0), (255, 0, 0)]
    for i, pose in enumerate([player_top_pose, player_btm_pose]):
        # Skip invalid poses
        if pose is None or np.any(pose == None) or not show_pose:
            continue

        for k, l in pose_lines:
            x1, y1 = pose[k]
            x2, y2 = pose[l]
            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue
            img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)

    # Show ball
    if ball_point is not None:
        x, y = ball_point
        if x is not None and y is not None:
            if show_ball:
                img = cv2.circle(img, (int(x), int(y)), 5, (0, 180, 255), 5)
            if show_picture_in_picture:
                frame_crop = crop_img_to_bbox(
                    frame,
                    (x, y, x, y),
                    padding=25,
                    resize_to=height // 3,
                    show_img=False,
                )
                frame_crop = add_img_border(frame_crop, thickness=5)
                img[height // 3 * 1 : height // 3 * 2, -height // 3 :] = frame_crop

    # Show player bounding boxes
    for i, bbox in enumerate([player_top_bbox, player_btm_bbox]):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue
            if show_bbox:
                img = cv2.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2
                )

            if show_picture_in_picture:
                frame_crop = crop_img_to_bbox(
                    frame,
                    (x1, y1, x2, y2),
                    padding=25,
                    resize_to=height // 3,
                    show_img=False,
                )
                frame_crop = add_img_border(frame_crop, thickness=5)
                img[
                    height // 3 * (2 * i) : height // 3 * (2 * i + 1), -height // 3 :
                ] = frame_crop

    #  Show image
    if show_img:
        fig, ax = plot_img(img)
        plt.show()

    return img


def visualize_segment_2d_annotations(
    segment_path,
    save_path,
    labels_path=None,
    **kwargs,
):
    """
    Visualize segment annotations
    ---
    Args:
    - segment_path: Path to the segment to visualize annotations for
    - save_path: Path to save the visualizations to
    - labels_path (default = None): Directory where label files are stored
    - **kwargs: Additional keyword arguments to pass to visualize_frame_2d_annotations
    """

    # Load segment frames
    frames, fps = read_segment_frames(
        segment_path,
        labels_path=labels_path,
        load_valid_frames_only=True,
    )
    frame_height, frame_width = frames[0].shape[:2]

    # load segment annotations
    (
        court_sequence,
        ball_sequence,
        player_btm_bbox_sequence,
        player_top_bbox_sequence,
        player_btm_pose_sequence,
        player_top_pose_sequence,
    ) = read_segment_2d_annotations(segment_path, labels_path=labels_path)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

    # Visualize and write frames
    for i, frame in enumerate(frames):
        img = visualize_frame_2d_annotations(
            frame,
            court_sequence[i],
            ball_sequence[i],
            player_btm_bbox_sequence[i],
            player_top_bbox_sequence[i],
            player_btm_pose_sequence[i],
            player_top_pose_sequence[i],
            **kwargs,
        )
        writer.write(img)

    # Release writer
    writer.release()


def make_3d_figax():
    """
    Create 3D figure and axis
    ---
    Returns:
    - fig: 3D figure
    - ax: 3D axis
    """
    # Create figure
    fig = plt.figure(figsize=(8, 8))

    # create axis
    ax = fig.add_subplot(projection="3d")

    # Set axis properties
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=12, azim=-90)
    ax.set_axis_off()

    # Return figure and axis
    return fig, ax


def plot_3d_court(
    fig=None,
    ax=None,
    court_width=10.97,
    court_length=23.77,
    half=False,
):
    """
    Plot 3D court
    ---
    Args:
    - fig (default = None): 3D figure
    - ax (default = None): 3D axis
    - court_width (default = 10.97): Width of the court (unit of choice, default = m)
    - court_length (default = 23.77): Length of the court (unit of choice, default = m)
    - half (default = False): Whether to plot half court
    """

    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = make_3d_figax()

    # Plot tennis court
    ylim = 0 if half else court_length / 2
    court_lines = [
        [
            (-court_width / 2, -court_length / 2),
            (-court_width / 2, min(court_length / 2, ylim)),
        ],  # left vertical
        [
            (court_width / 2, -court_length / 2),
            (court_width / 2, min(court_length / 2, ylim)),
        ],  # right vertical
        [
            (-court_width / 2, -court_length / 2),
            (court_width / 2, -court_length / 2),
        ],  # near horizontal
        [
            (-court_width / 2, min(court_length / 2, ylim)),
            (court_width / 2, min(court_length / 2, ylim)),
        ],  # far horizontal
        [
            (-court_width * 8.23 / 10.97 / 2, -court_length / 2),
            (-court_width * 8.23 / 10.97 / 2, min(court_length / 2, ylim)),
        ],  # left inner vertical
        [
            (court_width * 8.23 / 10.97 / 2, -court_length / 2),
            (court_width * 8.23 / 10.97 / 2, min(court_length / 2, ylim)),
        ],  # left inner vertical
        [
            (-court_width * 8.23 / 10.97 / 2, -court_length * 12.8 / 23.77 / 2),
            (court_width * 8.23 / 10.97 / 2, -court_length * 12.8 / 23.77 / 2),
        ],  # near inner horizontal
        [
            (
                -court_width * 8.23 / 10.97 / 2,
                min(court_length * 12.8 / 23.77 / 2, ylim),
            ),
            (
                court_width * 8.23 / 10.97 / 2,
                min(court_length * 12.8 / 23.77 / 2, ylim),
            ),
        ],  # far inner horizontal
        [
            (0, -court_length * 12.8 / 23.77 / 2),
            (0, min(court_length * 12.8 / 23.77 / 2, ylim)),
        ],  # center line
    ]
    for (x1, y1), (x2, y2) in court_lines:
        ax.plot((x1, x2), (y1, y2), (0, 0), color="red")

    # Create net points
    if not half:
        scale_factor = court_width / 10.97
        net_width = 10.04 * scale_factor
        net_height = 1.067 * scale_factor
        x_start = np.ones(2) * net_width / 2
        y = np.zeros(2)
        z_start = np.ones(2) * net_height / 2
        w = np.linspace(-net_width / 2, net_width / 2, 2, endpoint=True)
        h = np.linspace(0, net_height / 2, 2, endpoint=True)
        ax.plot(x_start, y, h, color="black")
        ax.plot(-x_start, y, h, color="black")
        ax.plot(w, y, z_start, color="black")
        ax.plot(w, y, y, color="black")

    # Return figure and axis
    return fig, ax


def plot_3d_pose(
    pose_3d,
    x_global=0,
    y_global=0,
    fig=None,
    ax=None,
    color="tab:blue",
    marker=".",
):
    """
    Plot 3D pose
    ---
    Args:
    - pose_3d: 3D pose to plot
    - x_global (default = 0): Global x coordinate
    - y_global (default = 0): Global y coordinate
    - fig (default = None): 3D figure
    - ax (default = None): 3D axis
    - color (default = "tab:blue"): Color of the pose
    """

    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = make_3d_figax()

    # Lines making up the body components of the pose
    pose_3d_lines = [
        np.array([1, 2, 3], dtype=int),  # Left leg
        np.array([4, 5, 6], dtype=int),  # Right leg
        np.array([1, 0, 4], dtype=int),  # Hips
        np.array([0, 7, 8], dtype=int),  # Torso
        np.array([8, 9, 10], dtype=int),  # Head
        np.array([8, 11, 12, 13], dtype=int),  # Left arm
        np.array([8, 14, 15, 16], dtype=int),  # Right arm
    ]

    # Plot pose
    for line in pose_3d_lines:
        ax.plot(
            pose_3d[line, 0] + x_global,
            pose_3d[line, 1] + y_global,
            pose_3d[line, 2],
            color=color,
            marker=marker,
        )

    # Return figure and axis
    return fig, ax


def animate_3d_pose(
    pose_3d_sequence,
    position_2d_sequence,
    interval=40,
    plot_court=False,
    save_path=None,
    **kwargs,
):
    """
    Animate 3D pose
    ---
    Args:
    - pose_3d_sequence: 3D pose sequence to animate
    - position_2d_sequence: 2D position sequence to animate
    - interval (default = 40): Interval between frames in milliseconds
    - plot_court (default = False): Whether to plot court
    - save_path (default = None): Path to save animation
    - **kwargs: Additional keyword arguments to pass to plot_3d_pose
    """

    # Create figure and axis
    fig, ax = make_3d_figax()
    z_max = np.max(np.abs(pose_3d_sequence[:, :, 2]))
    if plot_court:
        fig, ax = plot_3d_court(fig=fig, ax=ax, half=True)

    # Plot update function
    def update_plot(frame_number, fig=fig, ax=ax):
        # Plot 3D pose
        ax.clear()
        if plot_court:
            fig, ax = plot_3d_court(fig=fig, ax=ax, half=True)
        fig, ax = plot_3d_pose(
            pose_3d_sequence[frame_number],
            position_2d_sequence[frame_number, 0] if plot_court else 0,
            position_2d_sequence[frame_number, 1] if plot_court else 0,
            fig=fig,
            ax=ax,
            marker=None if plot_court else "o",
            **kwargs,
        )
        ax.set_aspect("equal", adjustable="box")
        if plot_court:
            ax.set_xlim(-10.97 / 2, 10.97 / 2)
            ax.set_ylim(-23.77 / 2, 0)
        ax.set_zlim(0, z_max)
        ax.set_axis_off()
        ax.view_init(elev=12, azim=-90)

    # Make animation
    ani = animation.FuncAnimation(
        fig, update_plot, len(pose_3d_sequence), interval=interval
    )

    # Save animation
    if save_path is not None:
        writer = animation.FFMpegWriter(fps=round(1000 / interval))
        ani.save(save_path, writer=writer, dpi=300)

    return ani


def visualize_segment_3d_annotations(
    segment_path,
    save_path,
    labels_path=None,
    interval=40,
    **kwargs,
):
    """
    Visualize 3D annotations for a segment
    ---
    Args:
    - segment_path: Path to the segment to visualize
    - save_path: Path to save the visualization
    - labels_path (default = None): Path to the labels file
    - **kwargs: Additional keyword arguments to pass to animate_3d_pose
    """

    # Load segment annotations
    (
        player_btm_position_scaled,
        player_top_position_scaled,
        player_btm_pose_3d_scaled,
        player_top_pose_3d_scaled,
    ) = read_segment_3d_annotations(
        segment_path,
        labels_path=labels_path,
        perform_scaling=True,
    )

    # Create figure and axis
    fig, ax = make_3d_figax()
    fig, ax = plot_3d_court(fig=fig, ax=ax, half=False)
    z_max_btm = np.max(np.abs(player_btm_pose_3d_scaled[:, :, 2]))
    z_max_top = np.max(np.abs(player_top_pose_3d_scaled[:, :, 2]))
    z_max = max(z_max_btm, z_max_top)

    # Plot update function
    def update_plot(frame_number, fig=fig, ax=ax):
        # Plot 3D pose
        ax.clear()
        fig, ax = plot_3d_court(fig=fig, ax=ax, half=False)
        fig, ax = plot_3d_pose(
            player_btm_pose_3d_scaled[frame_number],
            player_btm_position_scaled[frame_number, 0],
            player_btm_position_scaled[frame_number, 1],
            fig=fig,
            ax=ax,
            color="blue",
            marker=None,
            **kwargs,
        )
        fig, ax = plot_3d_pose(
            player_top_pose_3d_scaled[frame_number],
            player_top_position_scaled[frame_number, 0],
            player_top_position_scaled[frame_number, 1],
            fig=fig,
            ax=ax,
            color="green",
            marker=None,
            **kwargs,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-10.97 / 2, 10.97 / 2)
        ax.set_ylim(-23.77 / 2, 0)
        ax.set_zlim(0, z_max)
        ax.set_axis_off()
        ax.view_init(elev=12, azim=-90)

    # Make animation
    ani = animation.FuncAnimation(
        fig, update_plot, len(player_top_pose_3d_scaled), interval=interval
    )

    # Save animation
    if save_path is not None:
        writer = animation.FFMpegWriter(fps=round(1000 / interval))
        ani.save(save_path, writer=writer, dpi=300)

    return ani
