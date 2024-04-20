from dataclasses import dataclass

@dataclass
class Joint:
    # Example shape:  (3,)
    x: float
    y: float
    z: float

    @classmethod
    def from_tuple(joint):
        return Joint(x=joint[0], y=joint[1], z=joint[2])

@dataclass
class PlayerPose:
    # Example shape: (17, 3)
    pose: list[Joint]

    def from_npy(pose):
        return PlayerPose(pose=[Joint.from_tuple(j) for j in pose])

@dataclass
class PlayerPoseOverTime:
    # Example shape: (165, 17, 3)
    poses: list[PlayerPose]

    @classmethod
    def from_npy_file(npy_data):
        return PlayerPoseOverTime(poses=[PlayerPose.from_npy(pose) for pose in npy_data])


def keep_largest_width_3D_pose_heuristic(player_poses: list[PlayerPose]):
    pass
    # Parse bbox
    # xb1, yb1, xb2, yb2 = pose_bbox[0]
    # center_x = (xb1 + xb2) / 2
    # center_y = (yb1 + yb2) / 2
    # center_distance = ((crop_img_width / 2 - center_x)**2  + (crop_img_width / 2 - center_y)**2)**(1/2)

    # # Keep track of best prediction
    # if center_distance < min_center_distance:
    #     min_center_distance = center_distance
    #     best_keypoints = keypoints
    #     best_bbox = pose_bbox[0]