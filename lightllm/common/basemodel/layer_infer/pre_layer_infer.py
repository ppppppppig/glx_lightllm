from .base_layer_infer import BaseLayerInfer


class PreLayerInfer(BaseLayerInfer):
    """
    """
    def __init__(self, tp_rank, world_size, network_config, mode, pp_rank = None, pp_size = None, tp_size = None, pre_rank = None):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.network_config_ = network_config
        self.mode = mode
        self.pp_rank_ = pp_rank
        self.pp_size_ = pp_size
        self.pre_rank_ = pre_rank
        self.tp_size_ = tp_size
        return
