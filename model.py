import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


class BertSumClassifier(nn.Module):
    def __init__(self, pretrain_model, device, max_length):
        super(BertSumClassifier, self).__init__()

        self.dim = 768
        self.device = device
        self.max_length = max_length

        self.config = AutoConfig.from_pretrained(pretrain_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        self.bert = AutoModel.from_pretrained(pretrain_model)
        self.classifier = nn.Sequential(
            nn.Linear(self.dim, 2),
            nn.Sigmoid()
        )

    def forward(self, ids, clss):
        input_ids = torch.tensor(ids, dtype=torch.long).to(self.device)
        out = self.bert(input_ids.unsqueeze(0))[0].squeeze(0)
        output = torch.zeros((len(clss), 2))
        for i in range(len(clss)):
            output[i, :] = self.classifier(out[clss[i], :])
        return output


class BertSumTransformer(nn.Module):
    def __init__(self, pretrain_model, device, max_length):
        super(BertSumTransformer, self).__init__()

        self.dim = 768
        self.h = 12
        self.d_model = 12
        self.dropout = 0.1
        self.max_length = max_length
        self.device = device

        self.config = AutoConfig.from_pretrained(pretrain_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        self.bert = AutoModel.from_pretrained(pretrain_model)
        self.fnn = nn.Sequential(
            nn.Linear(self.dim + 1, self.dim + 1),
            nn.Tanh()
        )
        self.ln = nn.LayerNorm([self.max_length, self.dim])
        self.classifier = nn.Sequential(
            nn.Linear(self.dim + 1, 2)
        )
        self.multiheadattention = MultiHeadAttention(self.h, self.d_model, self.dropout)

    def forward(self):
        pass


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self):
        pass
