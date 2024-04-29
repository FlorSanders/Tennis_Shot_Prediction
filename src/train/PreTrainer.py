import sys
path_to_model_directory = '../model'

# Add this path to sys.path
if path_to_model_directory not in sys.path:
    sys.path.append(path_to_model_directory)

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.data import Data
from tqdm import tqdm
import glob
import json 
import numpy as np 
from data import TennisDataset
from model_builder import build_tennis_embedder


class PreTrainer:
    def __init__(self, model_config_path, train_path, val_path, batch_size, lr, epochs, model_save_path = 'trained_models'):
        self.model = build_tennis_embedder(model_config_path)
        self.train_loader = self.get_data_loader(train_path, batch_size, train = True)
        self.val_loader = self.get_data_loader(val_path, batch_size, train = False)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')  # Mean Squared Error Loss
        self.model_save_path = model_save_path
        self.best_val_loss = float('inf')
        print('Using device:', self.device)
    def train(self):
        
        for epoch in range(self.epochs):

            train_loader = self.train_loader
            val_loader = self.val_loader
            self.model.train()
            running_loss = 0.0
            for pose3d, position2d, pose_graph, targets, mask in tqdm(train_loader): 
                global_positions = position2d
                graphs = pose_graph
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 17, 3)

                #Move the data to the device
                graphs = [graph.to(self.device) for graph in graphs]
                global_positions = global_positions.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                #Optimize the model

                # Forward Pass
                self.optimizer.zero_grad()
                outputs = self.model(graphs, global_positions)

                # Compute Loss and Mask it
                loss = self.criterion(outputs, targets)
                masked_loss = loss * mask
                final_loss = masked_loss.sum() / mask.sum() # Reduce the loss; sum and then divide by the number of unmasked elements

                # Backward Pass
                final_loss.backward()
                self.optimizer.step()

                running_loss += final_loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
            
            self.evaluate(val_loader)

            
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for pose3d, position2d, pose_graph, targets, mask in data_loader:
                global_positions = position2d
                graphs = pose_graph
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 17, 3)

                #Move the data to the device
                graphs = [graph.to(self.device) for graph in graphs]
                global_positions = global_positions.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                
                outputs = self.model(graphs, global_positions)
                loss = self.criterion(outputs, targets)
                masked_loss = loss * mask
                final_loss = masked_loss.sum() / mask.sum() # Reduce the loss; sum and then divide by the number of unmasked elements
                total_loss += final_loss.item()
        
        avg_val_loss = total_loss / len(data_loader)
        print(f'Validation Loss: {avg_val_loss}')

        # Check directory and save model if this is the best validation loss so far
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
            print(f'Saving model at {self.model_save_path} with validation loss of {avg_val_loss}')
    
    def get_data_loader(self, data_path, batch_size, train = True):
        dataset = TennisDataset(labels_path = data_path)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, collate_fn=my_collate_fn)
        return data_loader



#### Things to do with the collate function 
#- How do we want to handle np.object_ instances of pose3d and position2d? --> At the moment they are being ignored
#- How do we want to do padding of sequence graph to same length? --> At the moment the last instance is repeated
#### Things to do for loading data
#- What do we want to do with the values being skipped? 

def pad_graphs(graphs, max_frames):
    padded_graphs = []
    for graph_list in graphs:
        num_graphs = len(graph_list)
        if num_graphs < max_frames:
            last_graph = graph_list[-1]
            additional_graphs = [last_graph] * (max_frames - num_graphs)  # Replicate last graph
            padded_graph_list = graph_list + additional_graphs
        else:
            padded_graph_list = graph_list[:max_frames]
        
        padded_graphs.append(padded_graph_list)
        
    return padded_graphs

def my_collate_fn(batch):
    pose3d = []
    position2d = []
    targets = []
    all_graphs = []
    graph_counts = []  # To count graphs per item in the batch
    masks = []


    for item in batch:
        pose3d_item, position2d_item, pose_graph_items, target_item = item
      
        # Check for None values and correct shapes
        if pose3d_item is not None and position2d_item is not None and pose_graph_items is not None: 
            if pose3d_item.dtype != np.object_ and position2d_item.dtype != np.object_ and len(pose_graph_items) > 0:
                
                sequence_graphs = pose_graph_items

                # Ensure numpy arrays are of type float32, convert object arrays if necessary
                # if pose3d_item.dtype == np.object_:
                #     print('NP Object!!')
                #     print("Pose3d: ", pose3d_item)
                #     pose3d_item = np.vstack(pose3d_item).astype(np.float32)
                #     target_item = np.vstack(target_item).astype(np.float32)
                # if position2d_item.dtype == np.object_:
                #     print('NP Object!!')
                #     print("Position2d: ", position2d_item)
                #     position2d_item = np.vstack(position2d_item).astype(np.float32)
                    
                if pose3d_item.ndim == 3 and position2d_item.ndim == 2:  # Ensure the correct dimensionality
                    
                    graph_count = 0
                    if isinstance(sequence_graphs, list):
                        all_graphs.append(sequence_graphs)
                        graph_count = len(sequence_graphs)  # Count graphs for this item
                    else:
                        print("Skipping a graph item due to incorrect type.")
                    graph_counts.append(graph_count)   

                    pose3d.append(torch.tensor(pose3d_item, dtype=torch.float32))
                    position2d.append(torch.tensor(position2d_item, dtype=torch.float32))
                    targets.append(torch.tensor(target_item, dtype=torch.float32))
                    masks.append(torch.ones(len(pose_graph_items), dtype=torch.bool))  # Mask of ones where data is valid

                # else:
                #     print(f"Skipping due to incorrect dimensions - Pose3D: {pose3d_item.shape}, Position2D: {position2d_item.shape}")
            # else: 
            #     print("Skipping a batch item due to object dtype or graph being empty.")
        # else:
        #     print("Skipping a batch item due to None values.")

    #print("Graphs per item in batch:", graph_counts)  # For Debugging

    # Pad pose3d and position2d sequences if not empty
    pose3d_padded = pad_sequence(pose3d, batch_first=True, padding_value=0.0) if pose3d else torch.Tensor()
    position2d_padded = pad_sequence(position2d, batch_first=True, padding_value=0.0) if position2d else torch.Tensor()
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0.0) if targets else torch.Tensor()
    mask_padded = pad_sequence(masks, batch_first=True, padding_value=0)  # Pad mask with zeros

    #print("Number of Graphs:", len(all_graphs))
    
    # Create a list of Batch objects for each item in the batch
    max_frames = max(len(graphs) for graphs in all_graphs)  # Maximum number of frames in the batch
    #print("Max Frames:", max_frames)
    if len(all_graphs) > 0:
        all_graphs = pad_graphs(all_graphs, max_frames)
        batched_graphs = [Batch.from_data_list(graph_list) for graph_list in all_graphs]
    else:
        batched_graphs = []

    return pose3d_padded, position2d_padded, batched_graphs, targets_padded, mask_padded





def validate_data_format(labels_path):
    annotation_files = glob.glob(os.path.join(labels_path, "*_info.json"))
    modified_files = []

    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as file:
            annotation = json.load(file)
        
        # Run checks for annotations that are true
        if annotation['is_valid'] == True:
            is_valid = True

            # Data file paths
            item_name = os.path.basename(annotation_file).replace("_info.json", "")
            positions_2d_path = os.path.join(labels_path, f"{item_name}_player_btm_position.npy")
            poses_3d_path = os.path.join(labels_path, f"{item_name}_player_btm_pose_3d.npy")
            
            # Check if data files exist
            if not os.path.exists(positions_2d_path) or not os.path.exists(poses_3d_path):
                is_valid = False
            else:
                # Load data files
                positions_2d = np.load(positions_2d_path, allow_pickle=True)
                poses_3d = np.load(poses_3d_path, allow_pickle=True)

                # if item_name in ["V006_0066", "V006_0178", "V009_1046", "V009_1639"]:
                #     print("Position_2d_shape: ", positions_2d.shape)
                #     print("Poses_3d_shape: ", poses_3d.shape)
                
                # Check data dimensions
                if positions_2d.ndim != 2 or positions_2d.shape[1] != 2:
                    is_valid = False
                if poses_3d.ndim != 3 or poses_3d.shape[1:] != (17, 3):
                    is_valid = False
                
                # Check for None values and data types
                if positions_2d.dtype == np.object_ or poses_3d.dtype == np.object_:
                    is_valid = False
                if positions_2d is None or poses_3d is None:
                    is_valid = False

                # Additional checks like empty pose graphs can be added here if they are saved and loaded similarly
                
            # Update annotation if invalid
            if not is_valid:
                annotation['is_valid'] = False
                modified_files.append(annotation_file)
                with open(annotation_file, 'w') as file:
                    json.dump(annotation, file)
            
    print(f"Total files: {len(annotation_files)}")
    print(f"Invalid Files: {len(modified_files)}")
    print(f"Valid Files: {len(annotation_files) - len(modified_files)}")
    print(f"Percentage of valid files: {(len(annotation_files) - len(modified_files)) / len(annotation_files) * 100}%")

    return modified_files

