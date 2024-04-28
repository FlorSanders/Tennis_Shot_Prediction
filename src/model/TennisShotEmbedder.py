import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from typing import Optional
import torch_geometric


class TennisShotEmbedder(nn.Module):
    """
    Generate tennis frame shot embeddings from player poses & global 2D positions.
    ---
    """
    
    def __init__(
        self, 
        graph_module: nn.Module,
        positional_encoding_module: nn.Module,
        sequence_module: nn.Module,
        output_module: nn.Module,
        use_positional_encoding: bool = True,
    ):
        """
        Initialize Tennis Shot Embedder
        ---
        Args:
        - graph_module: GNN Network to process player pose graph
        - positional_encoding_module: MLP that encodes global 2D player position 
        - sequence_module: RNN responsible for temporal sequence modelling
        - output_module: MLP that processes the output of the sequence module
        - use_positional_encoding: Boolean flag to indicate whether to use the positional encoding module (for ablation study)       
        """

        super(TennisShotEmbedder, self).__init__()
        self.graph_module = graph_module
        self.positional_encoding_module = positional_encoding_module
        self.sequence_module = sequence_module
        self.output_module = output_module
        self.use_positional_encoding = use_positional_encoding

    def forward(self, graphs, global_positions):
        """
        Tennis Shot Embedder Forward Function
        ---
        Args:
        - graphs: Sequence of local player pose graphs (N_seq x D_graph_embedding)
        - global_positions: Sequence of global player positions on minimap (N_seq x 2)

        Returns;
        - frame_embeddings: Sequence of tennis frame embeddings (features for downstream tasks) (N_seq x D_embedding)
        """

        # Handle Graph Processing
        batch_size = len(graphs)
        num_frames = graphs[0].num_graphs
        graph_embeddings = self.compute_graph_embeddings(graphs)

        # Handle global positions
        global_pos_embeddings = self.positional_encoding_module(global_positions.view(-1, 2))  # (batch * frame, h_dim)
        global_pos_embeddings = global_pos_embeddings.view(batch_size, num_frames, -1)  # Reshape to (batch, frame, h_dim)

        assert global_pos_embeddings.shape == graph_embeddings.shape, "Global Position Embeddings and Graph Embeddings should have the same shape"


        # Combine embeddings
        if self.use_positional_encoding:
            combined_embeddings = graph_embeddings + global_pos_embeddings
        else:
            combined_embeddings = graph_embeddings

        assert combined_embeddings.shape == graph_embeddings.shape, "Combined Embeddings shape should "

        # Process sequence
        sequence_output = self.sequence_module(combined_embeddings)  # (batch, frame, dim)

        # Process final output
        output = self.output_module(sequence_output.reshape(-1, sequence_output.size(2)))  # Flatten to (batch * frame, dim)
        output = output.view(-1, sequence_output.size(1), 51)  # Reshape back to (batch, frame, 51)
        output = output.view(batch_size, num_frames, 17, -1) # Reshape to (batch, frame, 17, 3)
        assert output.shape == (batch_size, num_frames, 17, 3), "Output shape should be (batch, frame, 17, 3)"

        return output

    def compute_graph_embeddings(self, graphs):
        """
        Compute graph embeddings from Sequence of pose graphs, and then average (over what??)
        ---
        Args:
        - graphs: Sequence of local player pose graphs (batch_size, N_seq, 17, 3)

        Returns:
        - graph_embeddings: Sequence of graph embeddings (batch_size, N_seq, D_graph_embedding)
        """
        
        graph_embeddings_for_batch = []
        batch_size = len(graphs)
        num_frames = graphs[0].num_graphs
        for seq_graph in graphs:
            x = seq_graph['x']
            edge_index = seq_graph['edge_index']
            batch = seq_graph['batch']
            graph_embeddings = self.graph_module(x, edge_index, batch)
            dense_embeddings, _ = torch_geometric.utils.to_dense_batch(graph_embeddings, batch, max_num_nodes=17)
            graph_embeddings_for_batch.append(dense_embeddings)
        
        graph_embeddings = torch.stack(graph_embeddings_for_batch, dim=0)
        graph_embeddings = graph_embeddings.mean(dim = 2)
        return graph_embeddings
            
    
class GraphModule(nn.Module):
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAT.html
    """
    GNN Network to process player pose graph
    ---
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: Optional[int] = None):
        """
        
        """
        super(GraphModule, self).__init__()
        self.graph_attention_network = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels
        )  # Might want to consider using dropout here, should concat be = False? what about heads? 

    def forward(self, x, edge_index, batch):
        """
        
        """
        return self.graph_attention_network(x, edge_index, batch)

        

class SequenceModule(nn.Module):
    """
    SequenceModule = sequence processing module that processes GNN output and generates tennis frame embeddings.
    ---
    Args:
    sequence_input = input sequence of embeddings [Added Positional Embedding and Graph Embedding]
    
    Returns: 
    frame_embeddings = output sequence of embeddings
    """

    def __init__(self, in_channels, hidden_channels, num_layers=1, bidirectional=False):
        """
        Initialize 
        """

        # Is the input_size the incoming dimensions size? or the sequence length?
        # I think the input_size should just be fixed as 1 

        super(SequenceModule, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )


    def forward(self, x):
        
        # x: (batch, frame, h_dim)
        # Initialize hidden and cell states
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), 
                         batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), 
                         batch_size, self.lstm.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))

        return out



class PositionalEncodingModule(nn.Module):
    """
    Just a simple MLP for learned positional embeddings
    ---
    Args: 
    global_pos  = x and y coordinates of the player on the court
    
    Returns: 
    positional_embedding = learned positional embedding
    """
    def __init__(self, hidden_dim, output_dim):
        super(PositionalEncodingModule, self).__init__()
        input_dim = 2 # (x,y)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_projection(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_projection(x)
        return x
    

class OutputModule(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers):
        super(OutputModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)
            ])
        
        self.output_projection = nn.Linear(hidden_dim, output_size)


    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_projection(x)
        return x