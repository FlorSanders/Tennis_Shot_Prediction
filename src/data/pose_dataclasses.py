from dataclasses import dataclass

from numpy import sort

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
    
    def get_x_span(self):
        joint_x_vals = [joint.x for joint in self.pose]
        soted_x_vals = sort(joint_x_vals)
        x_span = soted_x_vals[-1] - soted_x_vals[0]
        return x_span
    
    def get_y_span(self):
        joint_y_vals = [joint.y for joint in self.pose]
        soted_y_vals = sort(joint_y_vals)
        y_span = soted_y_vals[-1] - soted_y_vals[0]
        return y_span

    def get_volume(self):
        joint_x_vals = [joint.x for joint in self.pose]
        joint_y_vals = [joint.y for joint in self.pose]
        joint_z_vals = [joint.z for joint in self.pose]
        x_span = sort(joint_x_vals)[-1] - sort(joint_x_vals)[0]
        y_span = sort(joint_y_vals)[-1] - sort(joint_y_vals)[0]
        z_span = sort(joint_z_vals)[-1] - sort(joint_z_vals)[0]
        return x_span * y_span * z_span

@dataclass
class PlayerPoseOverTime:
    # Example shape: (165, 17, 3)
    poses: list[PlayerPose]

    @classmethod
    def from_npy_file(npy_data):
        return PlayerPoseOverTime(poses=[PlayerPose.from_npy(pose) for pose in npy_data])
