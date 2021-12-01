import torch
import torch.nn.functional as F


class BaseModel(torch.nn.Module):
    """Base Model for process different inputs

    Args:
        config (AttrbuteDict): config of the experiment
        embed_vecs (FloatTensor): embedding vectors for initialization
    """

    def __init__(self, config, embed_vecs):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(len(embed_vecs), embed_vecs.shape[1], padding_idx=0)
        self.embedding.weight.data = embed_vecs.clone()
        self.embed_drop = torch.nn.Dropout(p=config.dropout)
        # TODO Put the activation function to model files: https://github.com/ASUS-AICS/LibMultiLabel/issues/42
        self.activation = getattr(F, config.activation)

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embed_dim, padding_idx=0)

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

class SparseFeaturesEmbedding(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(input_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.weight[0, :], 0.0)
        torch.nn.init.xavier_uniform_(self.weight[1:, :])

    def forward(self, x):
        """
        :param x: Long sparse tensor of size ``(batch_size, input_dim)``
        :output: Float tensor of size ``(batch_size, embed_dim)``
        """
        return torch.sparse.mm(x, self.weight) if x is not None else None

def l2_norm_sq(x, keepdim=False):
    if x.is_sparse:
        x_norm_sq = torch.sparse.sum(x.detach()**2, dim=-1).to_dense()
    else:
        x_norm_sq = torch.norm(x.detach(), p=2, dim=-1) ** 2
    return x_norm_sq.unsqueeze(dim=-1) if keepdim else x_norm_sq

