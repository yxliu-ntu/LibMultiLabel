from torch import Tensor as T
from typing import Tuple
from transformers import BertModel, BertConfig
from torch import nn
from models.biencoder import BiEncoder

def get_bert_biencoder_componets(config,  **kwargs):
    dropout = config.dropout if hasattr(config, 'dropout') else 0.0
    bert_path = config.bert_path
    question_encoder = HFBertEncoder.init_encoder(bert_path=bert_path,
                                                    cfg_name=config.pretrained_model_cfg,
                                                    projection_dim=config.projection_dim,
                                                    dropout = dropout, **kwargs)
    ctx_encoder = HFBertEncoder.init_encoder(bert_path=bert_path,
                                            cfg_name=config.pretrain_model_cfg, 
                                            projection_dim=config.projection_dim,
                                            dropout=dropout,**kwargs)
    fix_ctx_encoder = config.fix_ctx_encoder if hasattr(config, 'fix_ctx_encoder') else False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)
    return biencoder


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        super().__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, bert_path :str, cfg_name: str, projection_dim: int =0, dropout:float =0.1 **kwargs) -> BertModel:
        cfg = BertConfig.from_pretrained(bert_path)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(bert_path, config=cfg, projection_dim=projection_dim, **kwargs)
    
    def forward(self, input_ids:T, token_type_ids:T, attention_mask:T) -> Tuple[T,...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            token_type_ids= token_type_ids,
                                                                            attention_mask = attention_mask)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids = input_ids,
                                                            token_type_ids = token_type_ids,
                                                            attention_mask = attention_mask)
        pooled_output = sequence_output[:,0,:]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
            sequence_output = self.encode_proj(sequence_output)
        return sequence_output, pooled_output, hidden_states
    
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size
