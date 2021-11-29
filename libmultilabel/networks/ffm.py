import torch


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.embedding.weight)

    def forward(self, x, x_val=None):
        """
        :param x: Long sparse tensor of size ``(batch_size, nnz)``
        :param x_val: Float tensor of size ``(batch_size, nnz)``
        :output: Float tensor of size ``(batch_size, nnz, embed_dim)``
        """
        x = self.embedding(x)
        if x_val is not None:
            x = torch.mul(x_val.unsqueeze(-1), x)
        return x

#class SparseFeaturesEmbedding(torch.nn.Module):
#    def __init__(self, input_dim, embed_dim):
#        super().__init__()
#        self.embedding = torch.nn.Parameter(torch.empty(input_dim, embed_dim))
#        self.reset_parameters()
#
#    def reset_parameters(self) -> None:
#        torch.nn.init.kaiming_uniform_(self.embedding.data)
#
#    def forward(self, x):
#        """
#        :param x: Long sparse tensor of size ``(batch_size, input_dim)``
#        :output: Float tensor of size ``(batch_size, embed_dim)``
#        """
#        return torch.sparse.mm(x, self.embedding)

class FM2Tower(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net_u = FeaturesEmbedding(self.config.Du, self.config.k)
        self.net_v = FeaturesEmbedding(self.config.Dv, self.config.k)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.net_u.embedding.weight)
        torch.nn.init.xavier_uniform_(self.net_v.embedding.weight)
        #torch.nn.init.constant_(self.net_u.embedding.weight, 0.001)
        #torch.nn.init.constant_(self.net_v.embedding.weight, 0.001)

    def _cal_output(self, inp, inp_val, inp_placeholder):
        """
        :param inp: Long tensor of size ``(batch_size, nnz)``
        :param inp_placeholder: torch.nn.Module class``
        :output: Float tensor of size ``(batch_size, k)``
        """
        output_mat = inp_placeholder(inp, inp_val).sum(dim=-2) if inp is not None else None
        return output_mat

    def forward(self, U, V, U_val=None, V_val=None):
     '''
     :param U: Long tensor of size ``(batch_size, nnz)``
     :param V: Long tensor of size ``(batch_size, nnz)``
     :output:
         P: Float tensor of size ``(batch_size, k)``
         Q: Float tensor of size ``(batch_size, k)``
     '''
     P = self._cal_output(U, U_val, self.net_u)
     Q = self._cal_output(V, V_val, self.net_v)
     return P, Q

