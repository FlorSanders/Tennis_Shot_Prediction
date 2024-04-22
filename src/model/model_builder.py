from TennisShotEmbedder import TennisShotEmbedder, GraphModule, SequenceModule, PositionalEncodingModule
import yaml
import jinja2

def load_config(config_path: str):
    with open(config_path, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg

def build_tennis_embedder(config_path: str) -> TennisShotEmbedder:
    cfg = load_config(config_path)
    # TODO: extract vars
    graph_module = GraphModule()
    pos_encoding_module = PositionalEncodingModule()
    sequence_module = SequenceModule()

    return TennisShotEmbedder(graph_module, pos_encoding_module, sequence_module)

