import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
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

# Import model functions
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not base_path in sys.path:
    sys.path.append(base_path)
from model.data import TennisDataset, my_collate_fn
from model.model_builder import build_tennis_embedder


class PreTrainer:
    def __init__(
        self,
        model_config_path,
        train_path,
        val_path,
        batch_size,
        lr,
        epochs,
        model_save_path="trained_models",
    ):
        self.model = build_tennis_embedder(model_config_path)
        self.train_loader = self.get_data_loader(train_path, batch_size, train=True)
        self.val_loader = self.get_data_loader(val_path, batch_size, train=False)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction="none")  # Mean Squared Error Loss
        self.model_save_path = model_save_path
        self.best_val_loss = float("inf")
        print("Using device:", self.device)
    

    def train(self):
        # Keep track of training & validation loss history
        train_loss_history = np.zeros(self.epochs)
        val_loss_history = np.zeros(self.epochs)

        # Run training
        train_loader = self.train_loader
        val_loader = self.val_loader
        for epoch in range(self.epochs):
            # Train step
            self.model.train()
            running_loss = 0.0
            for pose3d, position2d, pose_graph, targets, mask in tqdm(train_loader):
                global_positions = position2d
                graphs = pose_graph
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 17, 3)

                # Move the data to the device
                graphs = [graph.to(self.device) for graph in graphs]
                global_positions = global_positions.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                # Optimize the model

                # Forward Pass
                self.optimizer.zero_grad()
                outputs = self.model(graphs, global_positions)

                # Compute Loss and Mask it
                loss = self.criterion(outputs, targets)
                masked_loss = loss * mask
                final_loss = (
                    masked_loss.sum() / mask.sum()
                )  # Reduce the loss; sum and then divide by the number of unmasked elements

                # Backward Pass
                final_loss.backward()
                self.optimizer.step()

                running_loss += final_loss.item()

            train_loss = running_loss / len(train_loader)
            train_loss_history[epoch] = train_loss
            print(f"Epoch {epoch+1}, Loss: {train_loss}")

            val_loss = self.evaluate(val_loader)
            val_loss_history[epoch] = val_loss

        return train_loss_history, val_loss_history

    def evaluate(self, data_loader, test = False):
        
        self.model.eval()
        total_loss = 0.0
        if test:
            test_loss_history = []
        with torch.no_grad():
            for pose3d, position2d, pose_graph, targets, mask in data_loader:
                global_positions = position2d
                graphs = pose_graph
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 17, 3)

                # Move the data to the device
                graphs = [graph.to(self.device) for graph in graphs]
                global_positions = global_positions.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(graphs, global_positions)
                loss = self.criterion(outputs, targets)
                masked_loss = loss * mask
                final_loss = (
                    masked_loss.sum() / mask.sum()
                )  # Reduce the loss; sum and then divide by the number of unmasked elements
                if test:
                    test_loss_history.append(final_loss.item())
                total_loss += final_loss.item()

        avg_loss = total_loss / len(data_loader)

        if test:
            print(f"Test Loss: {avg_loss}")
        else:
            print(f"Validation Loss: {avg_loss}")

        # Check directory and save model if this is the best validation loss so far
        if avg_loss < self.best_val_loss and not test:
            self.best_val_loss = avg_loss
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            torch.save(
                self.model.state_dict(),
                os.path.join(self.model_save_path, "best_model.pth"),
            )
            print(
                f"Saving model at {self.model_save_path} with validation loss of {avg_loss}"
            )

        if test:
            return test_loss_history
        else: 
            return avg_loss

    def get_data_loader(self, data_path, batch_size, train=True):
        dataset = TennisDataset(labels_path=data_path)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=train, collate_fn=my_collate_fn
        )
        return data_loader
    

    def load_pretrained_model(self, model_weights_path):
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.to(self.device)
        print(f"Model loaded from {model_weights_path}")
    
    def test_trained_model(self, test_path, save_path = None):
        test_loader = self.get_data_loader(test_path, self.batch_size, train=False)
        test_loss_history = self.evaluate(test_loader, test = True)
        self.plot_loss(test_loss_history, save_path)

    
    def plot_loss(self, test_loss_history, save_path = None):
        plt.figure(figsize=(10, 6))  # Set the figure size for better visibility
        plt.plot(test_loss_history, label="Test Loss", color='blue', linewidth=2.0)  # Make line blue and thicker
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title("Test Loss per Batch")  # Add a title for clarity
        plt.xticks(range(len(test_loss_history)), range(1, len(test_loss_history)+1))  # Set x-ticks to be integer values of epochs
        plt.grid(True)  # Enable grid for easier visualization
        avg_loss = np.mean(test_loss_history)  # Calculate average loss
        plt.axhline(y=avg_loss, color='r', linestyle='--', label=f"Average Loss: {avg_loss:.4f}")  # Add a horizontal line for average loss
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        

