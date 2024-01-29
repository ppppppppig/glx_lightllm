from .base_layer_weight import BaseLayerWeight


class PreAndPostLayerWeight(BaseLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode, pp_rank = None, pp_size = None, tp_size = None):
        self.tp_rank_ = tp_rank
        self.pp_rank_ = pp_rank
        self.pp_size_ = pp_size 
        self.tp_size_ = tp_size
        self.data_type_ = data_type
        self.world_size_ = world_size
        self.network_config_ = network_config
        self.mode = mode
        self.init_static_params()
        return
    



