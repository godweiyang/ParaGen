from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn

from paragen.modules.decoders.layers import AbstractDecoderLayer
from paragen.modules.layers.feed_forward import FFN


class TransformerDecoderLayer(AbstractDecoderLayer):
    """
    TransformerDecoderLayer performs one layer of time-masked transformer operation,
    namely self-attention and feed-forward network.

    Args:
        d_model: feature dimension
        nhead: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation="relu",
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        # Implementation of Feedforward model
        self.ffn = FFN(d_model, dim_feedforward=dim_feedforward, activation=activation)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, prevs_layer,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        if self._mode == 'infer':
            tgt = tgt[-1:]
            tgt_mask, tgt_key_padding_mask = None, None
        residual = tgt
        if self.normalize_before:
            tgt = self.self_attn_norm(tgt)
        cache = tgt.clone()
        prevs = self._cal_prevs(tgt, prevs_layer) if self._mode == 'infer' else tgt
        tgt = self.self_attn(tgt, prevs, prevs, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.dropout1(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.self_attn_norm(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.multihead_attn_norm(tgt)
        tgt = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.dropout2(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.multihead_attn_norm(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.ffn_norm(tgt)
        tgt = self.ffn(tgt)
        tgt = self.dropout3(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.ffn_norm(tgt)
        return tgt, cache

    def _update_cache(self, cur):
        """
        Update cache with current states

        Args:
            cur: current state
        """
        prev = torch.cat([self._cache['prev'], cur], dim=0) if 'prev' in self._cache else cur
        self._cache['prev'] = prev
        return prev

    def _cal_prevs(self, cur, prevs):
        prev_sequence_length = prevs.size(0)
        batch_size = cur.size(1)
        feature_dim = cur.size(2)
        expanded_prev = prevs.expand(prev_sequence_length, batch_size, feature_dim)
        prev = torch.cat([expanded_prev, cur], dim=0)
        prev = prev[1:]
        return prev