#
# Time Series - Encoder Only Transformer
# based on Nouf code
#
import math
import torch
import torch.nn as nn
from ... import nn as nnx
from .tstran import positional_encoding


# ---------------------------------------------------------------------------
# Transformer Implementation (Nouf)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''

    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # dimension

        self.wq = nn.Linear(D, D * H)
        self.wk = nn.Linear(D, D * H)
        self.wv = nn.Linear(D, D * H)

        self.dense = nn.Linear(D * H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))  # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)  # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):
        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q)  # (B, H, S, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)

        # add the mask to the scaled tensor.
        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)   # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention)  # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, attention_weights


class TransformerLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerLayer, self).__init__()
        self.dropout = dropout
        self.mlp_hidden = nn.Linear(d_model, dim_feedforward)
        self.mlp_out = nn.Linear(dim_feedforward, d_model)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mha = MultiHeadAttention(d_model, nhead)

    def forward(self, x, look_ahead_mask):
        attn, attn_weights = self.mha(x, look_ahead_mask)  # (B, S, D)
        attn = self.dropout1(attn)  # (B,S,D)
        attn = self.layernorm1(attn + x)  # (B,S,D)

        mlp_act = torch.relu(self.mlp_hidden(attn))
        mlp_act = self.mlp_out(mlp_act)
        mlp_act = self.dropout2(mlp_act)

        output = self.layernorm2(mlp_act + attn)  # (B, S, D)

        return output, attn_weights


class Transformer(nn.Module):
    '''
    Transformer Decoder Implementating several Decoder Layers.
    '''

    def __init__(self, in_features, out_features,
                 d_model=32, nhead=1,
                 dim_feedforward=0,
                 num_encoder_layers=1, dropout=0.1):
        super().__init__()
        self.sqrt_D = torch.tensor(math.sqrt(d_model))
        self.num_encoder_layers = num_encoder_layers

        if dim_feedforward in [0, None]:
            dim_feedforward = d_model

        self.input_adapter = nn.Linear(in_features, d_model)  # multivariate input
        self.output_adapter = nn.Linear(d_model, out_features)  # multivariate output
        self.pos_encoding = positional_encoding(128, d_model)
        self.dec_layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward,
                             dropout=dropout
                             ) for _ in range(num_encoder_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self, x, mask):
        B, S, D = x.shape
        attention_weights = {}
        x = self.input_adapter(x)
        x *= self.sqrt_D
        x += self.pos_encoding[:, :S, :D]
        x = self.dropout(x)

        for i in range(self.num_encoder_layers):
            x, block = self.dec_layers[i](x=x, look_ahead_mask=mask)
            attention_weights['decoder_layer{}'.format(i + 1)] = block

        x = self.output_adapter(x)
        return x, attention_weights  # (B,S,S)


# ---------------------------------------------------------------------------
# TSNoufTransformer (ex TSTransformerV4)
# ---------------------------------------------------------------------------

class TSNoufTransformer(Transformer):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__(
            in_features=input_shape[1],
            out_features=output_shape[1],
            **kwargs
        )

    def forward(self, x, mask=None):
        y, _ = super().forward(x, mask)
        return y
# end


# ---------------------------------------------------------------------------
# CNNTransformer
# ---------------------------------------------------------------------------

# class CNNTransformerLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout):
#         super(CNNTransformerLayer, self).__init__()
#         # self.dropout = dropout
#         self.mlp_hidden = nn.Linear(d_model, dim_feedforward)
#         self.mlp_out = nn.Linear(dim_feedforward, d_model)
#         self.layernorm1 = nn.LayerNorm(d_model, eps=1e-9)
#         self.layernorm2 = nn.LayerNorm(d_model, eps=1e-9)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.conv1 = nnx.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#         self.conv2 = nnx.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#         self.mha = MultiHeadAttention(d_model, nhead)
#
#     def forward(self, x, look_ahead_mask=None):
#         attn = self.layernorm1(x)  # (B,S,D)
#         attn, attn_weights = self.mha(attn, look_ahead_mask)
#         attn = self.dropout1(attn)
#         res = x + attn
#
#         mlp_act = self.layernorm2(res)
#         mlp_act = self.conv1(mlp_act)
#         mlp_act = self.dropout2(mlp_act)
#         mlp_act = self.conv2(mlp_act)
#         y = mlp_act + res
#
#         return y, attn_weights
#
#
# class TSCNNTransformer(nn.Module):
#     '''Transformer Decoder Implementating several Decoder Layers.
#     '''
#
#     def __init__(self, input_shape, output_shape,
#                  d_model=32, nhead=1,
#                  dim_feedforward=0,
#                  num_encoder_layers=1,
#                  positional_encode=False,
#                  dropout=0.1):
#         super().__init__()
#         input_seqlen, in_features = input_shape
#         output_seqlen, out_features = output_shape
#
#         self.sqrt_D = torch.tensor(math.sqrt(d_model))
#         self.num_encoder_layers = num_encoder_layers
#
#         if dim_feedforward in [0, None]:
#             dim_feedforward = d_model
#
#         self.input_adapter = nn.Linear(in_features, d_model)  # multivariate input
#         self.output_adapter = nn.Linear(d_model, out_features)  # multivariate output
#
#         self.dec_layers = nn.ModuleList([
#             CNNTransformerLayer(d_model, nhead, dim_feedforward,
#                              dropout=dropout
#                              ) for _ in range(num_encoder_layers)
#         ])
#
#         self.dropout = nn.Dropout(dropout)
#         # self.pos_encoding = positional_encoding(128, d_model)
#         pass
#
#     def forward(self, x, mask=None):
#         attention_weights = {}
#         x = self.input_adapter(x)
#         x *= self.sqrt_D
#         # B, S, D = x.shape
#         # x += self.pos_encoding[:, :S, :D]
#         x = self.dropout(x)
#
#         for i in range(self.num_encoder_layers):
#             x, block = self.dec_layers[i](x=x, look_ahead_mask=mask)
#             attention_weights['decoder_layer{}'.format(i + 1)] = block
#
#         x = self.output_adapter(x)
#         return x, attention_weights  # (B,S,S)


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
