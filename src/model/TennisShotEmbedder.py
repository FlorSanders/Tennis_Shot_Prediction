import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from typing import Optional

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
        - position_al_enciding_module: MLP that encodes global 2D player position 
        -
        sequence_module: RNN responsible for temporal sequence modelling
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

        outputs = []
        h_t = None
        c_t = None 

        # graphs = (#Frames, d_h) & global_positions = (2,)
        for graph, global_pos in zip(graphs, global_positions):
            # Process graph to get graph embedding
            sequence_input = self.graph_module(graph)
            
            # Add global positional encoding
            if self.use_positional_encoding:
                positional_embedding = self.positional_encoding_module(global_pos)
                sequence_input = torch.add(sequence_input, positional_embedding)

            # Process sequence input through sequence module
            y, h_t, c_t = self.sequence_module(sequence_input, h_t, c_t)

            # Process output through output module
            y = self.output_module(y)

            outputs.append(y)

        y = torch.stack(outputs, dim=0)  # Are we stacking the right dimension?
        
        # Tennis Frame Embeddings
        return y
    
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
        )

    def forward(self, graph):
        """
        
        """
        return self.graph_attention_network(graph)

        

class SequenceModule(nn.Module):
    """
    SequenceModule = sequence processing module that processes GNN output and generates tennis frame embeddings.
    ---
    Args:
    sequence_input = input sequence of embeddings [Added Positional Embedding and Graph Embedding]
    
    Returns: 
    frame_embeddings = output sequence of embeddings
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        """
        Initialize 
        """

        # Is the input_size the incoming dimensions size? or the sequence length?
        # I think the input_size should just be fixed as 1 

        super(SequenceModule, self).__init__()
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size


    def forward(self, x, h_t, c_t):

        #x_shape = (1, embedding_size (input_size))
        #h_t shape = (1, self.hidden_size)
        #c_t shape = (1, self.hidden_size)
        #outputs = (1, self.hidden_size)


        if h_t is None and c_t is None:
            h_t = [torch.zeros(1, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(1, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]

        for layer in range(self.num_layers):
            h_t[layer], c_t[layer] = self.lstm_layers[layer](input_t, (h_t[layer], c_t[layer]))
            input_t = h_t[layer]

        return input_t, h_t, c_t



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
        x = self.output_projection
        return x