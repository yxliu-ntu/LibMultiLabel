from typing import Tuple
from torch import nn
from torch import Tensor as T
import torch

class BiEncoder(nn.Module):
    def __init__(self,
            question_model:nn.Module,
            ctx_model:nn.Module,
            fix_q_encoder:bool=False,
            fix_ctx_encoder:bool=False,
            poly_m:int=4
            ):
        super().__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        self.poly_m = poly_m
        if self.fix_q_encoder:
            self._freeze_model(question_model)
        if self.fix_ctx_encoder:
            self._freeze_model(ctx_model)

    def _freeze_model(self, model):
        for n, param in model.named_parameters():
            if 'encode_proj' not in n:
                param.requires_grad = False

    @staticmethod
    def get_representation(
            sub_model:nn.Module,
            ids: T,
            segments: T,
            attn_mask: T,
            fix_encoder:bool=False
            ):
        sequence_output = None
        pooled_output = None
        hidden_states = None

        if ids is not None:
            sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask
                    )
        return sequence_output, pooled_output, hidden_states
    
    def forward(self,
            question_ids:T,
            question_segments:T,
            question_attn_mask:T,
            context_ids:T,
            ctx_segments:T,
            ctx_attn_mask:T
            ) -> Tuple[T,T]:
        _q_seq, q_pooled_out, q_hidden = self.get_representation(
                self.question_model,
                question_ids,
                question_segments,
                question_attn_mask,
                self.fix_q_encoder
                )
        _ctx_seq, ctx_pooled_out, ctx_hidden = self.get_representation(
                self.ctx_model,
                context_ids,
                ctx_segments,
                ctx_attn_mask,
                self.fix_ctx_encoder)
        return q_pooled_out, ctx_pooled_out

            
