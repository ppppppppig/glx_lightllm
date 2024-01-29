from .base_layer_infer import BaseLayerInfer


class TransformerLayerInfer(BaseLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode, pp_rank = None, pp_size = None, tp_size = None):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.network_config_ = network_config
        self.mode = mode
        self.pp_rank_ = pp_rank
        self.pp_size_ = pp_size
        self.tp_size_ = tp_size
        return
