import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        b, t, c = x.shape
        q = self.wq(x).view(b, t, self.nhead, self.dk).transpose(1, 2)
        k = self.wk(x).view(b, t, self.nhead, self.dk).transpose(1, 2)
        v = self.wv(x).view(b, t, self.nhead, self.dk).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dk ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.fc(out)
        out = self.dropout(out)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, src, mask=None):
        a = self.self_attn(src, mask=mask)
        x = self.norm1(src + a)
        f = self.ff(x)
        x = self.norm2(x + f)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        b, t, d = x.shape
        x = x + self.pe[:, :t]
        return self.dropout(x)

class EEGTransformer(nn.Module):
    """
    同时支持三维 (B, seq_len, input_dim) 的标准 Transformer 输入，
    也支持二维 (B, F) 的输入：若F可被input_dim整除，则 seq_len=F//input_dim，
    否则报错。
    """
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=5,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, mask=None):
        """
        x形状可为:
          - (B, seq_len, input_dim): 正常三维
          - (B, F): 二维特征输入, 若F可被input_dim整除, seq_len=F//input_dim, 否则报错
        """
        if x.dim() == 2:
            b, f = x.shape
            if f % self.input_dim != 0:
                raise ValueError(
                    f"Cannot reshape from (B={b}, F={f}) to (B, seq_len, input_dim={self.input_dim})."
                )
            seq_len = f // self.input_dim
            x = x.view(b, seq_len, self.input_dim)

        b, t, _ = x.shape
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask=mask)
        out = x[:, -1, :]
        out = self.classifier(out)
        return out

def test_forward():
    b = 8
    seq_len = 16
    inp_dim = 32
    nclass = 3
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGTransformer(
        input_dim=inp_dim,
        d_model=64,
        nhead=4,
        num_layers=3,
        num_classes=nclass,
        dim_feedforward=256,
        dropout=0.1
    ).to(d)

    # 测试1: 标准3D输入 (B, seq_len, input_dim)
    x_seq = torch.randn(b, seq_len, inp_dim, device=d)
    y_seq = model(x_seq)
    print("3D input -> Output shape:", y_seq.shape)

    # 测试2: 二维特征 (B,F) F需被 input_dim整除, 例如F= inp_dim*seq_len=32*16=512
    F = inp_dim * seq_len
    x_feat = torch.randn(b, F, device=d)
    y_feat = model(x_feat)
    print("2D input -> Output shape:", y_feat.shape)

if __name__ == "__main__":
    test_forward()
