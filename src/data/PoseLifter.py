from mmpose.apis import MMPoseInferencer
import os
import numpy as np
from data_utils import read_segment_frames, read_segment_labels, clean_bbox_sequence
from tqdm import tqdm
from pose_dataclasses import PlayerPose
# import time

class PoseLifter:
    def __init__(self, 
                 crop_fn,
                 dedup_heuristic_fn,
                 dataset_path: str, 
                 write_path: str,
                 duplicate_work: bool = False
                ):
        """
        Parameters:
        - crop_fn:            A function which accepts 
        - dedup_heuristic_fn: A function which, given multiple 3D poses, uses a heuristic to select the active tennis player
                           These heuristics use the idea that in our context, players will be larger in the video frame than
                           ball boys and other peripheral humans. Examples include: max_x, max_y, max_volume...
        """
        # modules / flags
        self.inferencer = MMPoseInferencer(pose3d="human3d", device="cuda")
        self.crop_fn = crop_fn
        self.dedup_heuristic = dedup_heuristic_fn
        self.duplicate_work = duplicate_work

        # Dataset path (source)
        self.dataset_path = dataset_path
        self.segments_path = os.path.join(dataset_path, "segments")
        self.labels_path = os.path.join(dataset_path,"labels")
        self.segment_files = [f for f in os.listdir(self.segments_path) if f.endswith('.mp4')]

        # Write path (destination)
        self.write_path = write_path

    def extract_3d_poses(self):
        for i, segment_file in tqdm(enumerate(self.segment_files)):
            segment_path = os.path.join(self.segments_path, segment_file)
            
            should_process_segment = not os.path.exists(segment_path) or self.duplicate_work 
            if should_process_segment:
                # start = time.time()
                self.__process_segment(segment_path)
                # end = time.time()
                # print(f"segment {i} processing time: ", end-start)

    def __process_segment(
        self,
        segment_path,
        crop_padding=50,
        crop_width=224,
    ):
        # Load frames
        _, segment_filename = os.path.split(segment_path)
        segment_name, _ = os.path.splitext(segment_filename)
        frames, _ = read_segment_frames(
            segment_path,
            labels_path=self.labels_path,
            load_valid_frames_only=True
        )
        if not len(frames):
            return False

        # Load labels
        (
            _,
            court_sequence,
            _,
            player_btm_bbox_sequence,
            player_top_bbox_sequence,
            _,
            _,
        ) = read_segment_labels(
            segment_path,
            labels_path=self.labels_path,
            load_frame_validity=True,
            load_court=True,
            load_ball=False,
            load_player_bbox=True,
            load_player_pose=False,
            use_pose_bbox=True,
        )

        btm_missing_points, btm_bbox_clean = clean_bbox_sequence(
            player_btm_bbox_sequence,
            court_sequence,
            is_btm=True,
            make_plot=True,
        )
        top_missing_points, top_bbox_clean = clean_bbox_sequence(
            player_top_bbox_sequence,
            court_sequence,
            is_btm=False,
            make_plot=True,
        )

        # Process frames
        players_pose_sequences = [[None] *  len(frames) , [None] * len(frames)]
        for frame_index, frame in tqdm(enumerate(frames)):
            # Get frame labels
            players_bbox = [player_top_bbox_sequence[frame_index], player_btm_bbox_sequence[frame_index]]
            players_bbox_clean = [top_bbox_clean[frame_index], btm_bbox_clean[frame_index]]
            players_missing = [top_missing_points[frame_index], btm_missing_points[frame_index]]

            # Perform pose detection
            for is_btm, bbox in enumerate(players_bbox):
                if players_missing[is_btm]:
                    # Try to recover player pose from best knowledge
                    for _, bbox_candidate in enumerate(players_bbox_clean[is_btm], players_bbox[is_btm]):
                        # Skip invalid bboxes
                        if bbox_candidate is None:
                            continue
                        
                        # Detect pose
                        pose_keypoints = self.__detect_3D_pose_single_frame(
                            frame, 
                            bbox_candidate, 
                            crop_padding=crop_padding, 
                            crop_img_width=crop_width,
                        )

                        # Break if result is valid
                        if not np.any(pose_keypoints == None):
                            break
                else:
                    # Detect pose
                    pose_keypoints = self.__detect_3D_pose_single_frame(
                        frame, 
                        bbox, 
                        crop_padding=crop_padding, 
                        crop_img_width=crop_width
                    )

                # Save pose
                players_pose_sequences[is_btm][frame_index] = pose_keypoints
                        

        # Export labels
        for is_btm in range(2):
            player_name = "btm" if is_btm else "top"
            player_3d_pose_file = os.path.join(self.write_path, f"{segment_name}_player_{player_name}_pose_3d.npy")
            np.save(player_3d_pose_file, players_pose_sequences[is_btm])

        return True


    def __detect_3D_pose_single_frame(self,
        frame,
        bbox,
        crop_padding=50,
        crop_img_width=256
    ):

        # Crop image
        crop_padding=10
        cropped_frame = self.crop_fn(frame, bbox, crop_padding, crop_img_width)

        # Detect pose
        result_generator = self.inferencer(
            cropped_frame, 
            return_datasamples=True,
            vis_out_dir="/home/georgetamer/3d_poses", 
            show=False,  
            return_vis=False)

        results = [result for result in result_generator]

        for result in results:
            predictions = result["predictions"]
            if len(predictions) >= 1:
                prediction_keypoints = [PlayerPose.from_npy(pose=pred.pred_instances.keypoints[0]) for pred in predictions]
                keypoints = self.dedup_heuristic(prediction_keypoints)
            else:
                keypoints = predictions[0].pred_instances.keypoints

        return np.array(keypoints)