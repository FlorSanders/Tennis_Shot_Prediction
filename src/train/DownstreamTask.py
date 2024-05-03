import os
import sys

# Add this path to sys.path
path_to_model_directory = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "model"))
if path_to_model_directory not in sys.path:
    sys.path.append(path_to_model_directory)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import json
import numpy as np
from data import TennisDataset, my_collate_fn, ServeDataset, HitDataset, downstream_task_collate_fn
from model_builder import build_tennis_embedder
from sklearn.metrics import confusion_matrix
import seaborn as sns


#IDEA For Improvement: DownstreamClassificationModel could use some form of attention instead of just the mean over the sequences hidden state to compute the final task. 

class DownstreamClassificationTaskTrainer():
    def __init__(self, 
                 model_config_path, 
                 model_weights_path, 
                 train_path, 
                 val_path, 
                 test_path,
                 batch_size, 
                 lr, 
                 epochs, 
                 task = 'serve', 
                 model_save_path="downstream_trained_models"):
        
        '''
        Args:
        - model_config_path: Path to the pretrained model configuration file
        - model_weights_path: Path to the pretrained model weights file
        - train_path: Path to the training data
        - val_path: Path to the validation data
        - test_path: Path to the test data
        - batch_size: Batch size for training
        - lr: Learning rate for training
        - epochs: Number of epochs for training
        - task: The downstream task to train on (serve or hit)
        - model_save_path: Path to save the trained model

        '''
        
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.task = task
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        
        self.model = DownstreamClassificationModel(model_config_path, model_weights_path, task)
        self.train_loader = self.get_data_loader(train_path, batch_size, train = True)
        self.val_loader = self.get_data_loader(val_path, batch_size, train = False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.model_save_path = task + "_" + model_save_path
        self.best_val_loss = float("inf")
        print("Using device:", self.device)


    def get_data_loader(self, data_path, batch_size, train=True):
        if self.task == 'serve':
            dataset = ServeDataset(labels_path=data_path)
        elif self.task == 'hit':
            dataset = HitDataset(labels_path=data_path)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=train, collate_fn=downstream_task_collate_fn  #TODO: I think the collate function will cause issues
        )
        return data_loader
    
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
            train_loss = 0
            for pose3d, position2d, poseGraph, label in tqdm(train_loader):
                global_positions = position2d
                graphs = poseGraph
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 17, 3)

                # Move the data to the device
                graphs = [graph.to(self.device) for graph in graphs]
                global_positions = global_positions.to(self.device)
                label = label.to(self.device)

                # Optimize the model

                #Forward Pass
                self.optimizer.zero_grad()
                outputs = self.model(graphs, global_positions)

                # Compute loss and Mask it
                loss = self.criterion(outputs, label)

                # Backward Pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_loss_history[epoch] = train_loss
            print(f"Epoch {epoch+1}, Loss: {train_loss}")

            val_loss = self.evaluate(val_loader)
            val_loss_history[epoch] = val_loss

        test_loss_history = self.test_trained_model(self.test_path)

        return train_loss_history, val_loss_history, test_loss_history
    
    def evaluate(self, data_loader, test = False):
        
        self.model.eval()
        total_loss = 0.0
        total_pred = 0
        correct_pred = 0
        all_labels = []
        all_predictions = []

        if test:
            test_loss_history = []
        with torch.no_grad():
            for pose3d, position2d, poseGraph, label in data_loader:
                global_positions = position2d
                graphs = poseGraph

                # Move the data to the device
                graphs = [graph.to(self.device) for graph in graphs]
                global_positions = global_positions.to(self.device)
                label = label.to(self.device)

                outputs = self.model(graphs, global_positions)
                loss = self.criterion(outputs, label)
                if test:
                    test_loss_history.append(loss.item())
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_pred += label.size(0)
                correct_pred += (predicted == label).sum().item()

                all_labels.extend(label.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())


        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct_pred / total_pred

        if test:
            print(f"Test Loss: {avg_loss}, Test Accuracy: {accuracy}")
        else:
            print(f"Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}")

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
            return test_loss_history, all_labels, all_predictions
        else: 
            return avg_loss
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f'Model loaded from {model_path}')
        
    
    def test_trained_model(self, test_path, save_path = None):
        test_loader = self.get_data_loader(test_path, self.batch_size, train=False)
        test_loss_history, labels, preds = self.evaluate(test_loader, test = True)
        self.plot_loss(test_loss_history, save_path)
        self.plot_confusion_matrix(labels, preds, save_path)

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
    
    def plot_confusion_matrix(self, labels, preds, save_path=None):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()




class DownstreamClassificationModel(nn.Module):
    '''
    Args:
    - model_config_path: Path to the pretrained model configuration file
    - model_weights_path: Path to the pretrained model weights file
    - task: The downstream task to train on (serve or hit)
    '''
    def __init__(self, 
                 model_config_path, 
                 model_weights_path, 
                 task = 'serve'):

        super(DownstreamClassificationModel, self).__init__()
        self.base_model = build_tennis_embedder(model_config_path)
        self.base_model.load_state_dict(torch.load(model_weights_path))

        in_features = self.base_model.output_module.output_projection.in_features
        self.base_model.output_module = nn.Identity()

        if task == 'serve':
            self.output_module = ClassificationHead(in_features, 64, 2, 1)
        elif task == 'hit':
            self.output_module = ClassificationHead(in_features, 512, 11, 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    
    def forward(self, graphs, global_positions):

        # Handle Graph Processing

        sequence_embedding = self.base_model(graphs, global_positions, just_embeddings = True)
        sequence_embedding = sequence_embedding.mean(dim=1) #Take mean across the sequence dimension
        output = self.output_module(sequence_embedding)
        return output
    


class ClassificationHead(nn.Module):
    '''
    Args:
    - input_size: The input size of the head
    - hidden_dim: The hidden dimension of the head
    - num_classes: The number of classes to predict
    - num_layers: The number of layers in the head
    '''

    def __init__(self, input_size, hidden_dim, num_classes, num_layers):
        super(ClassificationHead, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)
            ])
        
        self.output_projection = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_projection(x)
        return x


    