import torch
from .base import FeaturesEmbedding, SparseFeaturesEmbedding, l2_norm_sq


class Linear(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net_u = SparseFeaturesEmbedding(self.config.Du, 1)
        self.net_v = SparseFeaturesEmbedding(self.config.Dv, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.net_u.weight[1:, :])
        torch.nn.init.xavier_uniform_(self.net_v.weight[1:, :])

    def _cal_output(self, inp, inp_placeholder):
        """
        :param inp: Long tensor of size ``(batch_size, nnz)``
        :param inp_placeholder: torch.nn.Module class``
        :output: Float tensor of size ``(batch_size, 1)``
        """
        if inp is not None:
            output_mat = inp_placeholder(inp)
            if self.config.isl2norm:
                inp_norm_sq = l2_norm_sq(inp, keepdim=False)
            else:
                inp_norm_sq = None
            return output_mat, inp_norm_sq
        else:
            return None, None

    def forward(self, U, V):
        '''
        :param U: Long tensor of size ``(batch_size, nnz)``
        :param V: Long tensor of size ``(batch_size, nnz)``
        :output:
            P: Float tensor of size ``(batch_size, 1)``
            Q: Float tensor of size ``(batch_size, 1)``
        '''
        P, Unorm_sq = self._cal_output(U, self.net_u)
        Q, Vnorm_sq = self._cal_output(V, self.net_v)
        return P, Unorm_sq, Q, Vnorm_sq

