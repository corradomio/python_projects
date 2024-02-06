# #
# # Transformer composed only by the decoder part
# # Article reference:
# #
#
# # This implementation is almost equivalent to Pytorch 'TransformerEncoderLayer'
# # with 'norm_first=False'
# # The oly difference is a missing Dropout after the ReLU
#
# import torch
# import torch.nn as nn
# import math
# import numpy as np
#
#
# # ---------------------------------------------------------------------------
# # Utilities
# # ---------------------------------------------------------------------------
#
# DEVICE = torch.device("cpu")
#
#
# # Positional encodings
# def get_angles(pos, i, D) -> np.ndarray:
#     angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
#     return pos * angle_rates
# # end
#
#
# def positional_encoding(D, position=12, dim=3, device=DEVICE) -> torch.Tensor:
#     angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                             np.arange(D)[np.newaxis, :],
#                             D)
#     # apply sin to even indices in the array; 2i
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#     # apply cos to odd indices in the array; 2i+1
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#     if dim == 3:
#         pos_encoding = angle_rads[np.newaxis, ...]
#     elif dim == 4:
#         pos_encoding = angle_rads[np.newaxis,np.newaxis,  ...]
#     return torch.tensor(pos_encoding, device=device)
# # end
#
#
# # function that implement the look_ahead mask for masking future time steps.
# def create_look_ahead_mask(size, device=DEVICE):
#     mask = torch.ones((size, size), device=device)
#     mask = torch.triu(mask, diagonal=1)
#     return mask  # (size, size)
# # end
#
#
# # ---------------------------------------------------------------------------
# # EncoderOnlyTransformer
# #   MultiHeadAttention
# #   TransformerLayer
# # ---------------------------------------------------------------------------
#
# class MultiHeadAttention(nn.Module):
#     '''Multi-head self-attention module'''
#     def __init__(self, D, H):
#         super(MultiHeadAttention, self).__init__()
#         self.H = H # number of heads
#         self.D = D # dimension
#
#         self.wq = nn.Linear(D, D*H)
#         self.wk = nn.Linear(D, D*H)
#         self.wv = nn.Linear(D, D*H)
#
#         self.dense = nn.Linear(D*H, D)
#
#     def concat_heads(self, x):
#         '''(B, H, S, D) => (B, S, D*H)'''
#         B, H, S, D = x.shape
#         x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
#         x = x.reshape((B, S, H*D))   # (B, S, D*H)
#         return x
#
#     def split_heads(self, x):
#         '''(B, S, D*H) => (B, H, S, D)'''
#         B, S, D_H = x.shape
#         x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
#         x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
#         return x
#
#     def forward(self, x, mask):
#
#         q = self.wq(x)  # (B, S, D*H)
#         k = self.wk(x)  # (B, S, D*H)
#         v = self.wv(x)  # (B, S, D*H)
#
#         q = self.split_heads(q)  # (B, H, S, D)
#         k = self.split_heads(k)  # (B, H, S, D)
#         v = self.split_heads(v)  # (B, H, S, D)
#
#         attention_scores = torch.matmul(q, k.transpose(-1, -2)) #(B,H,S,S)
#         attention_scores = attention_scores / math.sqrt(self.D)
#
#         # add the mask to the scaled tensor.
#         if mask is not None:
#             attention_scores += (mask * -1e9)
#
#         attention_weights = nn.Softmax(dim=-1)(attention_scores)
#         scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
#         concat_attention = self.concat_heads(scaled_attention) # (B, S, D*H)
#         output = self.dense(concat_attention)  # (B, S, D)
#
#         return output, attention_weights
# # end
#
#
# class TransformerLayer(nn.Module):
#     def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
#         super(TransformerLayer, self).__init__()
#         self.dropout_rate = dropout_rate
#         self.mlp_hidden = nn.Linear(D, hidden_mlp_dim)
#         self.mlp_out = nn.Linear(hidden_mlp_dim, D)
#         self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
#         self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.dropout2 = nn.Dropout(dropout_rate)
#
#         self.mha = MultiHeadAttention(D, H)
#
#
#     def forward(self, x, look_ahead_mask):
#
#         attn, attn_weights = self.mha(x, look_ahead_mask)  # (B, S, D)
#         attn = self.dropout1(attn)
#         attn = attn + x                     # (B,S,D)
#         attn = self.layernorm1(attn)        # (B,S,D)
#         mlp_act = self.mlp_hidden(attn)
#         mlp_act = torch.relu(mlp_act)
#         # HERE: MISSING Dropout respect the Pytorch TransformerEncoderLayer
#         mlp_act = self.mlp_out(mlp_act)
#         mlp_act = self.dropout2(mlp_act)
#         mlp_act = mlp_act + attn
#         output = self.layernorm2(mlp_act)   # (B, S, D)
#
#         return output, attn_weights
# # end
#
#
# class EncoderOnlyTransformerV2(nn.Module):
#     '''
#     Transformer Decoder Implementating several Decoder Layers.
#     '''
#     def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate):
#         super().__init__()
#         self.sqrt_D = torch.tensor(math.sqrt(D))
#         self.num_layers = num_layers
#         self.input_projection = nn.Linear(inp_features, D) # multivariate input
#         self.output_projection = nn.Linear(D, out_features) # multivariate output
#         self.pos_encoding = positional_encoding(D)
#         self.enc_layers = nn.ModuleList([TransformerLayer(D, H, hidden_mlp_dim,
#                                                           dropout_rate=dropout_rate
#                                                           ) for _ in range(num_layers)])
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(self, x, mask):
#         B, S, D = x.shape
#         attention_weights = {}
#         x = self.input_projection(x)
#         x *= self.sqrt_D
#
#         x += self.pos_encoding[:, :S, :]
#
#         x = self.dropout(x)
#
#         for i in range(self.num_layers):
#             x, block = self.enc_layers[i](x=x, look_ahead_mask=mask)
#             attention_weights['decoder_layer{}'.format(i + 1)] = block
#
#         x = self.output_projection(x)
#
#         return x, attention_weights # (B,S,S)
# # end
#
#
# # ---------------------------------------------------------------------------
# # End
# # ---------------------------------------------------------------------------
