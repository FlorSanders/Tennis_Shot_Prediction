import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_pose_wireframe(poses_3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x_amp = np.max(np.abs(poses_3d[:, :, 0]))
    y_amp = np.max(np.abs(poses_3d[:, :, 1]))
    z_amp = np.max(np.abs(poses_3d[:, :, 2]))

    pose_3d_lines = [
        np.array([1, 2, 3], dtype=int),  # Left leg
        np.array([4, 5, 6], dtype=int),  # Right leg
        np.array([1, 0, 4], dtype=int),  # Hips
        np.array([0, 7, 8], dtype=int),  # Torso
        np.array([8, 9, 10], dtype=int),  # Head
        np.array([8, 11, 12, 13], dtype=int),  # Left arm
        np.array([8, 14, 15, 16], dtype=int),  # Right arm
    ]

    def update_plot(num, poses_3d):
        ax.clear()
        for line in pose_3d_lines:
            ax.plot(
                poses_3d[num][line, 0],
                poses_3d[num][line, 1],
                poses_3d[num][line, 2],
                color="tab:blue",
                marker="o",
            )
        # ax.scatter(poses_3d[num][:, 0], poses_3d[num][:, 1], poses_3d[num][:, 2])
        ax.set_xlabel("X")
        ax.set_xlim(-x_amp, x_amp)
        ax.set_ylabel("Y")
        ax.set_ylim(-y_amp, y_amp)
        ax.set_zlabel("Z")
        ax.set_zlim(0, z_amp)
        ax.set_aspect("equal", adjustable="box")
        # scatter.set_data(poses_3d[num][:, 0], poses_3d[num][:, 1])
        # scatter.set_3d_properties(poses_3d[num][:, 2])

    ani = animation.FuncAnimation(
        fig, update_plot, len(poses_3d), fargs=(poses_3d,), interval=40
    )
    return ani
