import torch.nn as nn

from .caml import CAML
from .kim_cnn import KimCNN
from .kim_cnn_2tower import KimCNN2Tower
from .xml_cnn import XMLCNN


def get_init_weight_func(config):
    def init_weight(m):
        #print(m)
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            getattr(nn.init, config.init_weight+ '_')(m.weight)
            #print(m.weight.data.sum())
        if isinstance(m, KimCNN2Tower):
            getattr(nn.init, config.init_weight+ '_')(m.Q)
            #print(m.Q.data.sum())
    return init_weight
