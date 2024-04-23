from TennisShotEmbedder import TennisShotEmbedder, GraphModule, SequenceModule, PositionalEncodingModule, OutputModule
import yaml


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
        hidden_dim=pos_cfg["hidden_dim"],
        output_dim=pos_cfg["output_dim"]
    )

    seq_cfg = cfg["sequence_module"]
    sequence_module = SequenceModule(
        in_channels=seq_cfg["in_channels"],
        hidden_channels=seq_cfg["hidden_channels"],
        num_layers=seq_cfg["num_layers"],
        bidirectional=seq_cfg["bidirectional"],
    )

    out_cfg = cfg["output_module"]
    output_module = OutputModule(
        input_size=seq_cfg["hidden_channels"],
        output_size=out_cfg["output_dim"],
    )

    return TennisShotEmbedder(graph_module=graph_module, 
                              positional_encoding_module=pos_encoding_module, 
                              sequence_module=sequence_module,
                              output_module=output_module)

