import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..networks.base import BaseModel


class KimCNN2TowerReal(BaseModel):
    def __init__(self, config, embed_vecs):
        super(KimCNN2TowerReal, self).__init__(config, embed_vecs)

        self.filter_sizes = config.filter_sizes
        emb_dim = embed_vecs.shape[1]
        num_filter_per_size = config.num_filter_per_size

        self.convs = nn.ModuleList()

        for filter_size in self.filter_sizes:
            conv = nn.Conv1d(
                in_channels=emb_dim,
                out_channels=num_filter_per_size,
                kernel_size=filter_size)
            self.convs.append(conv)
        conv_output_size = num_filter_per_size * len(self.filter_sizes)

        #self.Q = nn.Parameter(torch.Tensor(config.num_classes, conv_output_size))
        self.Q_embedding = nn.Embedding(config.num_classes, conv_output_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.Q_embedding.weight, a=math.sqrt(5))

    def forward(self, text, label_data=None):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.transpose(1, 2) # (batch_size, embed_dim, length)

        h_list = []
        for conv in self.convs:
            h_sub = conv(h) # (batch_size, num_filter, length)
            h_sub = F.max_pool1d(h_sub, kernel_size=h_sub.size()[2]) # (batch_size, num_filter, 1)
            h_sub = h_sub.view(h_sub.shape[0], -1) # (batch_size, num_filter)
            h_list.append(h_sub)

        # Max-pooling and monotonely increasing non-linearities commute. Here
        # we apply the activation function after max-pooling for better
        # efficiency.
        if len(self.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        P = self.activation(h) # (batch_size, total_num_filter)

        if label_data is None:
            Q = self.Q_embedding.weight.data
        else:
            Q = torch.squeeze(self.Q_embedding(label_data))

        return P, Q
