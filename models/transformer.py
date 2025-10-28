# ----------------------------------------------------------------------------------------------
# Ov-SGR Official Code
# Modified from OvGSR and OpenSU Official Code
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Transformer Architectures in OvGSR
"""
import copy
import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor
import sys
#from transformers import Blip2Processor, Blip2Model
import numpy as np

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_glance_enc_layers=3, num_gaze_s1_dec_layers=3,
                 num_gaze_s1_enc_layers=3,
                 num_tr_dec_layers=1, dim_feedforward=2048, dropout=0.15, activation="gelu"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_verb_classes = 504

        # Glacne Transformer
        glance_enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.glance_enc = TransformerEncoder(glance_enc_layer, num_glance_enc_layers)

        # Gaze-Step1 Transformer
        gaze_s1_dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.gaze_s1_dec = TransformerDecoder(gaze_s1_dec_layer, num_gaze_s1_dec_layers)
        gaze_s1_enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.gaze_s1_enc = TransformerEncoder(gaze_s1_enc_layer, num_gaze_s1_enc_layers)

        # Gaze-Step2 Transformer
        tr_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.tr_dec = TransformerDecoder(tr_layer, num_tr_dec_layers)

        hoi_token_length = 10
        scale = d_model ** -0.5
        self.hoi_token_length = hoi_token_length
        self.hoi_token_embed = nn.Parameter(scale * torch.randn(hoi_token_length, d_model))
        self.hoi_pos_embed = nn.Parameter(scale * torch.randn(hoi_token_length, d_model))

        # classifer (for verb prediction)
        self.verb_classifier = nn.Sequential(nn.Linear(d_model * 2, d_model * 2),
                                             nn.GELU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(d_model * 2, self.num_verb_classes)) #self.num_verb_classes

        self.linear = nn.Linear(768, d_model)

        scale = d_model ** -0.5
        self.proj = nn.Parameter(torch.randn(d_model, d_model)) #
        self.proj_eval = nn.Parameter(torch.randn(d_model * 2, 51))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model * 2)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)

        self.linear_blip = nn.Sequential(
            nn.Linear(1408, 512),
            nn.Dropout(p=0.5)  # Specify your dropout rate here
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    #action_tensor
    def forward(self, src, flattend_src, image_queries, IL_token_embed, RL_token_embed, verb_token_embed, role_token_embed, vidx_ridx, text_features, targets=None, inference=False):
        device = IL_token_embed.device

        bs, d = src.shape
        if not inference:
            selected_verb_token = verb_token_embed[targets['verbs']].view(1, -1)
            selected_roles = targets['roles']
        else:
            top1_verb = torch.topk(src, k=1, dim=1)[1].item()
            selected_verb_token = verb_token_embed[top1_verb].view(1, -1)
            selected_roles = vidx_ridx[top1_verb]
        selected_role_tokens = role_token_embed[selected_roles]
        frame_role_queries = selected_role_tokens + selected_verb_token
        frame_role_queries = frame_role_queries.unsqueeze(1).repeat(1, bs, 1)
        role_tgt = torch.zeros_like(frame_role_queries)
        final_rhs = self.tr_dec(frame_role_queries, self.ln3(flattend_src.permute(1, 0, 2)), memory_key_padding_mask=None,
                                     pos=None, query_pos=role_tgt)
        final_rhs = self.ln4(final_rhs)
        final_rhs = final_rhs.transpose(1, 2)

        return final_rhs, selected_roles


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                num_zeros=None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, num_zeros=num_zeros)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor], num_zeros=None):
        if num_zeros is not None:
            return tensor if pos is None else torch.cat([tensor[:num_zeros], (tensor[num_zeros:] + pos)], dim=0)
        else:
            return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                num_zeros=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos, num_zeros=num_zeros)
        # q = self.with_pos_embed(src2, pos, num_zeros=num_zeros)
        # k = self.with_pos_embed(src2, pos, num_zeros=num_zeros)
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Q TYPE: {type(q)} SHAPE: {q.shape}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! K TYPE: {type(k)} SHAPE: {k.shape}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!src2 TYPE: {type(src2)} SHAPE {src.shape}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Q IS K', q is k)
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! K IS V', k is src2)
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Q equal K', torch.equal(q,k))
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! K equal VALUE', torch.equal(k,src2))
        # sys.exit()
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_transformer(args):
    return Transformer(d_model=args.hidden_dim,
                       dropout=args.dropout,
                       nhead=args.nheads,
                       num_glance_enc_layers=args.num_glance_enc_layers,
                       num_gaze_s1_dec_layers=args.num_gaze_s1_dec_layers,
                       num_gaze_s1_enc_layers=args.num_gaze_s1_enc_layers,
                       num_tr_dec_layers=args.num_tr_dec_layers,
                       dim_feedforward=args.dim_feedforward)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
