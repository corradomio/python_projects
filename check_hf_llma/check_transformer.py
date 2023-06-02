from pprint import pprint
from transformers import AutoTokenizer, AutoModel
# from bertviz.transformers_neuron_view import BertModel
# from bertviz.neuron_view import show
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

# model = BertModel.from_pretrained(model_ckpt)
text = "time flies like an arrow"
# show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)

inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
pprint(inputs.input_ids)

from torch import nn
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
pprint(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
pprint(inputs_embeds.size())

import torch
import torch.nn.functional as F
from math import sqrt
query = key = value = inputs_embeds
# dim_k = key.size(-1)
# scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
# pprint(scores.size())
#
# weights = F.softmax(scores, dim=-1)
# pprint(weights.sum(dim=-1))
#
# attn_outputs = torch.bmm(weights, value)
# pprint(attn_outputs.shape)

def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

# attn_outputs = scaled_dot_product_attention(query, key, value)
# pprint(attn_outputs.shape)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
        self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


multihead_attn = MultiHeadAttention(config)
attn_outputs = multihead_attn(inputs_embeds)
pprint(attn_outputs.size())


# from bertviz import head_view
# from transformers import AutoModel
# model = AutoModel.from_pretrained(model_ckpt, output_attentions=True)
# sentence_a = "time flies like an arrow"
# sentence_b = "fruit flies like a banana"
# viz_inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')
# attention = model(**viz_inputs).attentions
# sentence_b_start = (viz_inputs.token_type_ids == 0).sum(dim=1)
# tokens = tokenizer.convert_ids_to_tokens(viz_inputs.input_ids[0])
# head_view(attention, tokens, sentence_b_start, heads=[8])

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_outputs)
pprint(ff_outputs.size())


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


encoder_layer = TransformerEncoderLayer(config)
pprint((inputs_embeds.shape, encoder_layer(inputs_embeds).size()))


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


embedding_layer = Embeddings(config)
print(embedding_layer(inputs.input_ids).size())


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config)
        for _ in range(config.num_hidden_layers)])
    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


encoder = TransformerEncoder(config)
pprint(encoder(inputs.input_ids).size())


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, x):
        x = self.encoder(x)[:, 0, :] # select hidden state of [CLS] token
        x = self.dropout(x)
        x = self.classifier(x)
        return x


config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
pprint(encoder_classifier(inputs.input_ids).size())



