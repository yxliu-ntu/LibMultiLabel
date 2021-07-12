from torch import nn
from .models.hf_models import get_bert_biencoder_componets
import logging
import torch
from torch import Tensor as T

class Bert(nn.Module):
    def __init__(self, config):
#config requires bert model path, pad_id
        self.config = config
        self.pad_id = config.pad_id
        logging.info("***** Initializing components for training *****")
        self.bert = get_bert_biencoder_componets(config)
    
    def forward(self, question_ids:T, context_ids:T):
        question_segments = torch.zeros_like(question_ids)
        context_segments = torch.zeros_like(context_ids)
        question_attn_mask = self.__get_attn_mask(question_ids)
        context_attn_mask = self.__get_attn_mask(context_ids)
        if self.bert.training:
            model_out = self.bert(question_ids, question_segments, question_attn_mask,
                                context_ids, context_segments, context_attn_mask)
        P, Q = model_out
        return P,Q

    def __get_attn_mask(self, token_tensor:T)->T:
        return token_tensor != self.pad_id


    
