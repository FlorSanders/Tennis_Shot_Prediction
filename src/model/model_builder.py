import os
import sys
import yaml

# Import data functions
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not base_path in sys.path:
    sys.path.append(base_path)
from model.TennisShotEmbedder import (
    TennisShotEmbedder,
    GraphModule,
    SequenceModule,
    PositionalEncodingModule,
    OutputModule,
)


SKELETON_SIZE = 17


def load_config(config_path: str):
    with open(config_path, "r") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.SafeLoader)
    return cfg


def build_tennis_embedder(config_path: str) -> TennisShotEmbedder:
    cfg = load_config(config_path)["model"]

    graph_cfg = cfg["graph_module"]
    graph_module = GraphModule(
        in_channels=graph_cfg["in_channels"],
        hidden_channels=graph_cfg["hidden_channels"],
        num_layers=graph_cfg["num_layers"],
        out_channels=graph_cfg["out_channels"],
    )

    pos_cfg = cfg["positional_encoder_module"]
    pos_encoding_module = PositionalEncodingModule(
        hidden_dim=pos_cfg["hidden_dim"], output_dim=graph_cfg["out_channels"]
    )

    seq_cfg = cfg["sequence_module"]
    sequence_module = SequenceModule(
        in_channels=seq_cfg["in_channels"],
        hidden_channels=seq_cfg["hidden_channels"],
        num_layers=seq_cfg["num_layers"],
        bidirectional=seq_cfg["bidirectional"],
    )

    out_cfg = cfg["output_module"]
    final_output_size = graph_cfg["in_channels"] * SKELETON_SIZE
    output_module = OutputModule(
        input_size=seq_cfg["hidden_channels"],
        output_size=final_output_size,
        hidden_dim=out_cfg["hidden_dim"],
        num_layers=out_cfg["num_layers"],
    )

    return TennisShotEmbedder(
        graph_module=graph_module,
        positional_encoding_module=pos_encoding_module,
        sequence_module=sequence_module,
        output_module=output_module,
    )
