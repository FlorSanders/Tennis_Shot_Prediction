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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
                 model_save_path="downstream_trained_models", 
                 freeze_feature_extractor = False, 
                 keep_labels = None):
        
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
        self.keep_labels = keep_labels
        
        self.model = DownstreamClassificationModel(model_config_path, model_weights_path, task, freeze_feature_extractor, keep_labels)
        self.train_loader = self.get_data_loader(train_path, batch_size, train = True)
        self.val_loader = self.get_data_loader(val_path, batch_size, train = False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.model_save_path = task + "_" + model_save_path
        self.best_val_loss = float("inf")
        self.label_dict = self.get_label_dict(task)
        print("Using device:", self.device)


    def get_data_loader(self, data_path, batch_size, train=True):
        if self.task == 'serve':
            if self.keep_labels:
                dataset = ServeDataset(labels_path=data_path, keep_labels=self.keep_labels)
            else:
                dataset = ServeDataset(labels_path=data_path)
        elif self.task == 'hit':
            if self.keep_labels:
                dataset = HitDataset(labels_path=data_path, keep_labels=self.keep_labels)
            else:
                dataset = HitDataset(labels_path=data_path)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=train, collate_fn=downstream_task_collate_fn  #TODO: I think the collate function will cause issues
        )
        return data_loader
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
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

        self.test_trained_model(self.test_path)

        return train_loss_history
    
    def evaluate(self, data_loader, test = False):
        
        self.model.eval()
        total_loss = 0.0
        total_pred = 0
        correct_pred = 0
        all_labels = []
        all_predictions = []


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
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_pred += label.size(0)
                correct_pred += (predicted == label).sum().item()

                all_labels.extend(label.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        if test:
             print(f"Test Accuracy: {accuracy}, Test Precision: {precision}, Test Recall: {recall}, Test F1 Score: {f1}")
        else:
            print(f"Validation Accuracy: {accuracy}, Validation Precision: {precision}, Validation Recall: {recall}, Validation F1 Score: {f1}")

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
            return all_labels, all_predictions
        else: 
            return avg_loss
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f'Model loaded from {model_path}')

    def get_label_dict(self, task):
        
        if task == 'hit':
            label_dict = {
                0: 'Backhand_Flat',
                1: 'Backhand_Slice',
                2: 'Backhand_Topspin',
                3: 'Backhand_Unsure',
                4: 'Backhand_Volley',
                5: 'Forehand_Flat',
                6: 'Forehand_Slice',
                7: 'Forehand_Smash',
                8: 'Forehand_Topspin',
                9: 'Forehand_Unsure',
                10: 'Forehand_Volley'
            }
            if self.keep_labels:
                filtered_label_dict = {}
                count = 0
                for key, value in label_dict.items():
                    if value in self.keep_labels:
                        filtered_label_dict[count] = value
                        count += 1
                label_dict = filtered_label_dict
            return label_dict
        elif task == 'serve':
            return {
                0: 'Fault', 
                1: 'In'
            }  
        
    
    def test_trained_model(self, test_path, save_path = None):
        test_loader = self.get_data_loader(test_path, self.batch_size, train=False)
        labels, preds = self.evaluate(test_loader, test = True)
        self.plot_confusion_matrix(labels, preds, save_path)

    
    def plot_confusion_matrix(self, labels, preds, save_path=None):
        labels_text = [self.label_dict[label] for label in labels] if self.label_dict else labels
        preds_text = [self.label_dict[pred] for pred in preds] if self.label_dict else preds
        cm = confusion_matrix(labels_text, preds_text)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.label_dict.values(), yticklabels=self.label_dict.values())
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
                 task = 'serve', 
                 freeze_feature_extractor = False, 
                 keep_labels = None):

        super(DownstreamClassificationModel, self).__init__()
        self.base_model = build_tennis_embedder(model_config_path)
        self.base_model.load_state_dict(torch.load(model_weights_path))

        in_features = self.base_model.output_module.output_projection.in_features
        self.base_model.output_module = nn.Identity()

        if freeze_feature_extractor:
            for param in self.base_model.parameters():
                param.requires_grad = False

        if keep_labels:
            out_features = len(keep_labels)
        else:
            out_features = 11 if task == 'hit' else 2
        if task == 'serve':
            self.output_module = ClassificationHead(in_features, 64, out_features, 2)
        elif task == 'hit':
            self.output_module = ClassificationHead(in_features, 512, out_features, 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    
    def forward(self, graphs, global_positions):

        # Handle Graph Processing

        sequence_embedding = self.base_model(graphs, global_positions, just_embeddings = True)
        sequence_embedding = sequence_embedding[:, -1, :] #Take the last hidden state
        # sequence_embedding = sequence_embedding.mean(dim=1) #Take mean across the sequence dimension
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


    