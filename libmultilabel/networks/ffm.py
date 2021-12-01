import torch
from .base import FeaturesEmbedding, SparseFeaturesEmbedding, l2_norm_sq


class FM2Tower(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net_u = SparseFeaturesEmbedding(self.config.Du, self.config.k)
        self.net_v = SparseFeaturesEmbedding(self.config.Dv, self.config.k)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #torch.nn.init.xavier_uniform_(self.net_u.weight[1:, :])
        #torch.nn.init.xavier_uniform_(self.net_v.weight[1:, :])
        torch.nn.init.constant_(self.net_u.weight[1:, :], 0.001)
        torch.nn.init.constant_(self.net_v.weight[1:, :], 0.001)

    def _cal_output(self, inp, inp_placeholder):
        """
        :param inp: Long tensor of size ``(batch_size, nnz)``
        :param inp_placeholder: torch.nn.Module class``
        :param isl2norm: Bool, whether need to l2-normalize features``
        :output: Float tensor of size ``(batch_size, k)``
        """
        if inp is not None:
            output_mat = inp_placeholder(inp)
            if self.config.isl2norm:
                inp_norm = l2_norm_sq(inp, keepdim=True) ** 0.5
                output_mat = torch.div(output_mat, inp_norm)
            return output_mat
        else:
            return None

    def forward(self, U, V):
        '''
        :param U: Long tensor of size ``(batch_size, nnz)``
        :param V: Long tensor of size ``(batch_size, nnz)``
        :output:
            P: Float tensor of size ``(batch_size, k)``
            Q: Float tensor of size ``(batch_size, k)``
        '''
        P = self._cal_output(U, self.net_u)
        Q = self._cal_output(V, self.net_v)
        return P, Q

