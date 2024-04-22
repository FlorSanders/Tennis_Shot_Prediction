# Import libraries
import os
import glob
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import animate_pose_wireframe


# Constants
data_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "data"))
dataset = "tenniset"
dataset_path = os.path.join(data_path, dataset)
labels_path = os.path.join(dataset_path, "shot_labels")
videos_path = os.path.join(dataset_path, "videos")


class TennisDataset(torch.utils.data.Dataset):
    """
    Tennis Player 2D Position & 3D Pose Dataset
    ---
    """

    def __init__(
        self,
        labels_path=labels_path,
        videos=None,
        load_serve=True,
        load_hit=True,
        load_near=True,
        load_far=True,
        in_memory=True,
    ):
        """
        Initialize TennisDataset
        ---
        Args:
        - labels_path: Directory where the data annotations are saved
        - videos: List of videos to load (None = All videos)
        - load_serve: Whether to load serve segments
        - load_hit: Whether to load hit segments
        - load_near: Whether to load near player segments
        - load_far: Whether to load far player segments
        - in_memory: Whether to load the data in memory during initialization
        """

        # Initialize superclass
        super().__init__()

        # Save arguments
        self.labels_path = labels_path
        self.videos = videos
        self.load_serve = load_serve
        self.load_hit = load_hit
        self.load_near = load_near
        self.load_far = load_far
        self.in_memory = in_memory

        # Read videos from file system if None (default argument)
        if self.videos is None:
            self.videos = [
                os.path.splitext(video)[0]
                for video in os.listdir(videos_path)
                if os.path.splitext(video)[-1] == ".mp4"
            ]

        # Read annotation files for selected videos
        annotation_files = []
        for video in sorted(self.videos):
            annotation_files.extend(
                sorted(glob.glob(os.path.join(labels_path, f"{video}_*_info.json")))
            )

        # Initialize items & annotations
        self.items = []
        self.annotations = []
        if self.in_memory:
            self.poses_3d = []
            self.positions_2d = []

        # Read items from annotation files
        for annotation_file in annotation_files:
            # Parse annotation name
            item_name = os.path.basename(annotation_file).replace("_info.json", "")

            # Load annotation
            with open(annotation_file, "r") as f:
                annotation = json.load(f)

            # Decide whether to keep annotation
            keep = True
            keep = keep and annotation["is_valid"]
            keep = keep and (
                (self.load_serve and annotation["is_serve"])
                or (self.load_hit and not annotation["is_serve"])
            )
            keep = keep and (
                (self.load_near and annotation["player_is_near"])
                or (self.load_far and not annotation["player_is_near"])
            )
            if not keep:
                continue

            # Add item
            self.items.append(item_name)
            self.annotations.append(annotation)
            if self.in_memory:
                # Load item to memory
                poses_3d, positions_2d = self.__loaditem__(item_name, annotation)
                self.poses_3d.append(poses_3d)
                self.positions_2d.append(positions_2d)

    def __len__(self):
        return len(self.items)

    def __loaditem__(self, item, annotation):
        """
        Load segment annotations from file system
        ---
        Args:
        - item: Name of the segment
        - annotation: Annotation for the segment

        Returns:
        - poses_3d: 3D Poses for the segment
        - positions_2d: 2D Positions for the segment
        """

        # Determine which player to select from
        is_near = annotation["player_is_near"]
        player_id = "btm" if is_near else "top"

        #  Load Data From File System
        positions_2d = np.load(
            os.path.join(self.labels_path, f"{item}_player_{player_id}_position.npy"),
            allow_pickle=True,
        )
        poses_3d = np.load(
            os.path.join(self.labels_path, f"{item}_player_{player_id}_pose_3d.npy"),
            allow_pickle=True,
        )

        # Rotate 2D positions 180° around center of court -> same point of reference
        # TODO: Apply scaling to 2D Poses
        if is_near:
            mask = np.all(positions_2d != None, axis=1)
            positions_2d[mask] *= -1

        # TODO: Transform & Scale 3D Poses

        # Return annotations
        return poses_3d, positions_2d

    def __getitem__(self, index):
        """
        Get item from dataset
        ---
        Args:
        - index: Index of the item to get

        Returns:
        - poses_3d: 3D Poses for the segment
        - positions_2d: 2D Positions for the segment
        """

        # Obtain the segment 3D Pose & 2D Position Annotations
        if self.in_memory:
            # From Memory
            poses_3d = self.poses_3d[index]
            positions_2d = self.positions_2d[index]
        else:
            # From File System
            poses_3d, positions_2d = self.__loaditem__(
                self.items[index], self.annotations[index]
            )

        # Return annotations
        return poses_3d, positions_2d


class ServeDataset(TennisDataset):
    """
    Tennis Serve Dataset (downstream classiciation task)
    ---
    """

    def __init__(
        self,
        videos=None,
        load_near=True,
        load_far=True,
        in_memory=True,
    ):
        # Load data
        super().__init__(
            labels_path=labels_path,
            videos=videos,
            load_serve=True,
            load_hit=False,
            load_near=load_near,
            load_far=load_far,
            in_memory=in_memory,
        )

        # Determine classes
        self.classes = np.sort(
            np.unique([annotation["info"]["Result"] for annotation in self.annotations])
        )
        self.class_map = {}
        for i, c in enumerate(self.classes):
            self.class_map[c] = i

    def __getitem__(self, index):
        poses_3d, positions_2d = super().__getitem__(index)
        label = self.class_map[self.annotations[index]["info"]["Result"]]
        return poses_3d, positions_2d, label


class HitDataset(TennisDataset):
    """
    Tennis Hit Dataset (downstream classiciation task)
    ---
    """

    def __init__(
        self,
        videos=None,
        load_near=True,
        load_far=True,
        in_memory=True,
    ):
        # Load data
        super().__init__(
            labels_path=labels_path,
            videos=videos,
            load_serve=False,
            load_hit=True,
            load_near=load_near,
            load_far=load_far,
            in_memory=in_memory,
        )

        # Determine classes
        self.classes = np.sort(
            np.unique(
                [
                    f'{annotation["info"]["Side"]}_{annotation["info"]["Type"]}'
                    for annotation in self.annotations
                ]
            )
        )
        self.class_map = {}
        for i, c in enumerate(self.classes):
            self.class_map[c] = i

    def __getitem__(self, index):
        """
        Get item from dataset
        ---
        Args:
        - index: Index of the item to get

        Returns:
        - poses_3d: 3D Poses for the segment
        - positions_2d: 2D Positions for the segment
        - class_label: Label for the segment
        """

        poses_3d, positions_2d = super().__getitem__(index)
        class_name = f'{self.annotations[index]["info"]["Side"]}_{self.annotations[index]["info"]["Type"]}'
        class_label = self.class_map[class_name]
        return poses_3d, positions_2d, class_label


if __name__ == "__main__":
    dataset = TennisDataset(
        load_hit=True,
        load_serve=True,
        load_far=True,
        load_near=True,
        in_memory=True,
    )
    print(len(dataset))
    indx = np.random.randint(len(dataset))
    print(dataset.items[indx])
    poses_3d, positions_2d = dataset.__getitem__(indx)
    print(poses_3d.shape)
    print(positions_2d.shape)

    ani = animate_pose_wireframe(poses_3d)
    plt.show()
