from .base_layer_weight import BaseLayerWeight


class TransformerLayerWeight(BaseLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode, pp_rank = None, pp_size = None, tp_size = None):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.pp_rank_ = pp_rank
        self.pp_size_ = pp_size
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.tp_size_ = tp_size
        self.init_static_params()
        return