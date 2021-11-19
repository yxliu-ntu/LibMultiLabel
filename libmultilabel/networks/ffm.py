import torch


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.empty(input_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.embedding.data)

    def forward(self, x):
        """
        :param x: Long sparse tensor of size ``(batch_size, input_dim)``
        :output: Float tensor of size ``(batch_size, embed_dim)``
        """
        return torch.sparse.mm(x, self.embedding)

class FM2Tower(torch.nn.Module):
    def __init__(self, config, D_u=6038, D_v=3514):
        super().__init__()
        self.config = config
        self.net_u = torch.nn.Embedding(D_u, self.config.k, padding_idx=0)
        self.net_v = torch.nn.Embedding(D_v, self.config.k, padding_idx=0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.net_u.weight)
        torch.nn.init.kaiming_uniform_(self.net_v.weight)

    def _cal_output(self, inp, inp_placeholder):
        """
        :param inp: Long tensor of size ``(batch_size, nnz of feature)``
        :param inp_placeholder: torch.nn.Module class``
        :output: Float tensor of size ``(batch_size, k)``
        """
        output_mat = inp_placeholder(inp)
        return output_mat

    def forward(self, U, V):
     '''
     :param U: Long tensor of size ``(batch_size, nnz of context feature)``
     :param V: Long tensor of size ``(action_num, nnz of action feature)``
     :output:
         P: Float tensor of size ``(batch_size, k)``
         Q: Float tensor of size ``(action_num, k)``
     '''
     P = self._cal_output(U, self.net_u).sum(dim=-2) if U is not None else None
     Q = self._cal_output(V, self.net_v).sum(dim=-2) if V is not None else None
     return P, Q

