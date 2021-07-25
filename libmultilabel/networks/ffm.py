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
    def __init__(self, D_u, D_v, k):
        super().__init__()
        self.k = k
        self.net_u = FeaturesEmbedding(D_u, self.k)
        self.net_v = FeaturesEmbedding(D_v, self.k)

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
     P = self._cal_output(U, self.net_u)
     Q = self._cal_output(V, self.net_v)

     return P, Q

