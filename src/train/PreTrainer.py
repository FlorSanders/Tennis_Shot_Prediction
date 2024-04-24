import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class PreTrainer:
    def __init__(self, model, train_loader, val_loader, batch_size, lr, epochs):
        self.model = model
        self.mlp = mlp 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss
        
    def train(self):
        
        for epoch in range(self.epochs):

            train_loader = self.train_loader
            val_loader = self.val_loader
            self.model.train()
            running_loss = 0.0

            for graphs, global_positions in tqdm(train_loader): # Extract Targets

                #For the graph variable: include all sequences except the last
                graphs = graphs[:-1, :, :]

                #For the global_positions variable: include all sequences except the last
                global_positions = global_positions[:-1, :]

                #For the targets variable: include all sequences except the first
                targets = graphs[1:, :, :]

                #Move the data to the device
                graphs = graphs.to(self.device)
                global_positions = global_positions.to(self.device)
                targets = targets.to(self.device)

                #Optimize the model
                self.optimizer.zero_grad()
                outputs = self.model(graphs, global_positions)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
            
            self.evaluate(val_loader)
            
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for graphs, global_positions, targets in data_loader:
                graphs = graphs[:-1, :, :]
                global_positions = global_positions[:-1, :]
                targets = graphs[1:, :, :]

                graphs = graphs.to(self.device)
                global_positions = global_positions.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(graphs, global_positions)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        print(f'Test Loss: {total_loss/len(data_loader)}')