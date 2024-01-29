import torch
import numpy as np


class BaseLayerWeight:
    def __init__(self):
        self.tp_rank_ = None
        self.pp_rank_ = None
        self.pp_size_ = None

    def load_hf_weights(self, weights):
        """
        load weights
        """
        pass

    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def verify_load(self):
        """
        verify all load is ok
        """
        raise Exception("must verify weights load ok")
        pass

    def _cuda(self, cpu_tensor):
        if self.tp_rank_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).cuda()
        elif self.pp_size_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).cuda(self.tp_rank_)
        else:
            new_rank = self.tp_rank_ * self.pp_size_ + self.pp_rank_
            return cpu_tensor.contiguous().to(self.data_type_).cuda(new_rank)
