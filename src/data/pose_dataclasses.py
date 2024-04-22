from dataclasses import dataclass

from numpy import sort

X = 0
Y = 1
Z = 2

@dataclass
class PlayerPose:
    # Example shape: (17, 3)
    pose: list[tuple[float]]

    @classmethod
    def from_npy(cls, pose):
        # joints = map(lambda j: Joint.from_tuple(j), pose)
        return PlayerPose(pose=pose)
    
    def get_x_span(self):
        joint_x_vals = [joint[X] for joint in self.pose]
        soted_x_vals = sort(joint_x_vals)
        x_span = soted_x_vals[-1] - soted_x_vals[0]
        return x_span
    
    def get_y_span(self):
        joint_y_vals = [joint[Y] for joint in self.pose]
        sorted_y_vals = sort(joint_y_vals)
        y_span = sorted_y_vals[-1] - sorted_y_vals[0]
        return y_span
    
    def get_z_span(self):
        joint_z_vals = [joint[Z] for joint in self.pose]
        soted_z_vals = sort(joint_z_vals)
        z_span = soted_z_vals[-1] - soted_z_vals[0]
        return z_span

    def get_volume(self):
        x_span = self.get_x_span()
        y_span = self.get_y_span()
        z_span = self.get_z_span()
        return x_span * y_span * z_span

@dataclass
class PlayerPoseOverTime:
    # Example shape: (165, 17, 3)
    poses: list[PlayerPose]

    @classmethod
    def from_npy_file(cls, npy_data):
        return PlayerPoseOverTime(poses=[PlayerPose.from_npy(pose) for pose in npy_data])
