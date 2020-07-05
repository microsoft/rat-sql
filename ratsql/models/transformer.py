import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import entmax

# Adapted from
# https://github.com/tensorflow/tensor2tensor/blob/0b156ac533ab53f65f44966381f6e147c7371eee/tensor2tensor/layers/common_attention.py
def relative_attention_logits(query, key, relation):
    # We can't reuse the same logic as tensor2tensor because we don't share relation vectors across the batch.
    # In this version, relation vectors are shared across heads.
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # qk_matmul is [batch, heads, num queries, num kvs]
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))

    # q_t is [batch, num queries, heads, depth]
    q_t = query.permute(0, 2, 1, 3)

    # r_t is [batch, num queries, depth, num kvs]
    r_t = relation.transpose(-2, -1)

    #   [batch, num queries, heads, depth]
    # * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.
    q_tr_t_matmul = torch.matmul(q_t, r_t)

    # qtr_t_matmul_t is [batch, heads, num queries, num kvs]
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)

    # [batch, heads, num queries, num kvs]
    return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])

    # Sharing relation vectors across batch and heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [num queries, num kvs, depth].
    #
    # Then take
    # key reshaped
    #   [num queries, batch * heads, depth]
    # relation.transpose(-2, -1)
    #   [num queries, depth, num kvs]
    # and multiply them together.
    #
    # Without sharing relation vectors across heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, heads, num queries, num kvs, depth].
    #
    # Then take
    # key.unsqueeze(3)
    #   [batch, heads, num queries, 1, depth]
    # relation.transpose(-2, -1)
    #   [batch, heads, num queries, depth, num kvs]
    # and multiply them together:
    #   [batch, heads, num queries, 1, depth]
    # * [batch, heads, num queries, depth, num kvs]
    # = [batch, heads, num queries, 1, num kvs]
    # and squeeze
    # [batch, heads, num queries, num kvs]

def relative_attention_values(weight, value, relation):
    # In this version, relation vectors are shared across heads.
    # weight: [batch, heads, num queries, num kvs].
    # value: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # wv_matmul is [batch, heads, num queries, depth]
    wv_matmul = torch.matmul(weight, value)

    # w_t is [batch, num queries, heads, num kvs]
    w_t = weight.permute(0, 2, 1, 3)

    #   [batch, num queries, heads, num kvs]
    # * [batch, num queries, num kvs, depth]
    # = [batch, num queries, heads, depth]
    w_tr_matmul = torch.matmul(w_t, relation)

    # w_tr_matmul_t is [batch, heads, num queries, depth]
    w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

    return wv_matmul + w_tr_matmul_t


# Adapted from The Annotated Transformer
def clones(module_fn, N):
    return nn.ModuleList([module_fn() for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return torch.matmul(p_attn, value), p_attn

def sparse_attention(query, key, value, alpha, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    if alpha == 2:
        p_attn = entmax.sparsemax(scores, -1)
    elif alpha == 1.5:
        p_attn = entmax.entmax15(scores, -1)
    else:
        raise NotImplementedError
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return torch.matmul(p_attn, value), p_attn

# Adapted from The Annotated Transformers
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        if query.dim() == 3:
            x = x.squeeze(1)
        return self.linears[-1](x)


# Adapted from The Annotated Transformer
def attention_with_relations(query, key, value, relation_k, relation_v, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = relative_attention_logits(query, key, relation_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn_orig = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn_orig)
    return relative_attention_values(p_attn, value, relation_v), p_attn_orig


class PointerWithRelations(nn.Module):
    def __init__(self, hidden_size, num_relation_kinds, dropout=0.2):
        super(PointerWithRelations, self).__init__()
        self.hidden_size = hidden_size
        self.linears = clones(lambda: nn.Linear(hidden_size, hidden_size), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.relation_k_emb = nn.Embedding(num_relation_kinds, self.hidden_size)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, self.hidden_size)

    def forward(self, query, key, value, relation, mask=None):
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)

        if mask is not None:
            mask = mask.unsqueeze(0)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, 1, self.hidden_size).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        _, self.attn = attention_with_relations(
            query,
            key,
            value,
            relation_k,
            relation_v,
            mask=mask,
            dropout=self.dropout)

        return self.attn[0,0]

# Adapted from The Annotated Transformer
class MultiHeadedAttentionWithRelations(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithRelations, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, relation_k, relation_v, mask=None):
        # query shape: [batch, num queries, d_model]
        # key shape: [batch, num kv, d_model]
        # value shape: [batch, num kv, d_model]
        # relations_k shape: [batch, num queries, num kv, (d_model // h)]
        # relations_v shape: [batch, num queries, num kv, (d_model // h)]
        # mask shape: [batch, num queries, num kv]
        if mask is not None:
            # Same mask applied to all h heads.
            # mask shape: [batch, 1, num queries, num kv]
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: [batch, heads, num queries, depth]
        x, self.attn = attention_with_relations(
            query,
            key,
            value,
            relation_k,
            relation_v,
            mask=mask,
            dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# Adapted from The Annotated Transformer
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, layer_size, N, tie_layers=False):
        super(Encoder, self).__init__()
        if tie_layers:
            self.layer = layer()
            self.layers = [self.layer for _ in range(N)]
        else:
            self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer_size)
         
         # TODO initialize using xavier
        
    def forward(self, x, relation, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, relation, mask)
        return self.norm(x)


# Adapted from The Annotated Transformer
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# Adapted from The Annotated Transformer
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, num_relation_kinds, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(lambda: SublayerConnection(size, dropout), 2)
        self.size = size

        self.relation_k_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)

    def forward(self, x, relation, mask):
        "Follow Figure 1 (left) for connections."
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask))
        return self.sublayer[1](x, self.feed_forward)


# Adapted from The Annotated Transformer
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

