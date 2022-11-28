import torch.nn as nn

from .ffm import FM2Tower
from .linear import Linear

def get_init_weight_func(config):
    raise NotImplementedError
    #def init_weight(m):
    #    #print(m)
    #    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
    #        getattr(nn.init, config.init_weight+ '_')(m.weight)
    #        #print(m.weight.data.sum())
    #    try:
    #        getattr(nn.init, config.init_weight+ '_')(m.Q.weight)
    #        #print(m.Q.data.sum())
    #    except:
    #        pass
    #return init_weight
