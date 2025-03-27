import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        b, t, _ = q.shape
        q = self.w_q(q).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        out, _ = ScaledDotProductAttention(self.d_k, self.dropout.p)(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, sublayer):
        return self.norm(x + sublayer)

class LSTMClassifier(nn.Module):
    """
    LSTM + 多头注意力模块的分类器。
    同时支持 (B, seq_len, input_dim) 正常三维输入，
    也允许 (B, F) 二维输入时自动 reshape:
      - 若F可被input_size整除, 则seq_len=F/input_size
      - 否则报错。
    """
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=4,
        num_classes=5,
        bidirectional=True,
        dropout=0.2,
        attention_heads=4,
        attn_dim=128
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        dfac = 2 if bidirectional else 1
        self.proj = nn.Linear(hidden_size * dfac, attn_dim)
        self.attn = MultiHeadAttention(d_model=attn_dim, num_heads=attention_heads, dropout=dropout)
        self.res_ln = ResidualLayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, attn_dim)
        )
        self.res_ln2 = ResidualLayerNorm(attn_dim)
        self.fc_out = nn.Linear(attn_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x形状可为:
          - (B, seq_len, input_size): 正常3D输入
          - (B, F):  仅有二维特征输入, 本模块自动将F拆成 (seq_len, input_size)
                     其中 seq_len = F // input_size, 要求能整除, 否则报错
        """
        if x.dim() == 2:
            b, f = x.shape
            if f % self.input_size != 0:
                raise ValueError(f"Cannot reshape from (B={b}, F={f}) to (B, seq_len, input_size={self.input_size}).")
            seq_len = f // self.input_size
            x = x.view(b, seq_len, self.input_size)

        b, t, _ = x.shape
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            b,
            self.hidden_size,
            device=x.device
        )
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            b,
            self.hidden_size,
            device=x.device
        )
        out, _ = self.lstm(x, (h0, c0))
        z = self.proj(out)
        a = self.attn(z, z, z)
        a = self.res_ln(z, a)
        f = self.ff(a)
        f = self.res_ln2(a, f)
        o = f.mean(dim=1)
        return self.fc_out(o)

def test_lstm():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = 8
    seq_len = 12
    inp_dim = 16
    nclass = 3
    model = LSTMClassifier(
        input_size=inp_dim,
        hidden_size=32,
        num_layers=4,
        num_classes=nclass,
        bidirectional=True,
        dropout=0.2,
        attention_heads=4,
        attn_dim=64
    ).to(d)

    # 测试1: 正常3D输入 (B, seq_len, input_size)
    x_seq = torch.randn(b, seq_len, inp_dim, device=d)
    y_seq = model(x_seq)
    print("3D input -> Output shape:", y_seq.shape)

    # 测试2: 二维特征输入 (B, F) 其中F要是 input_size的整数倍
    # 例如 F= inp_dim * seq_len = 16 * 12 = 192
    F = inp_dim * seq_len
    x_feat = torch.randn(b, F, device=d)
    y_feat = model(x_feat)
    print("2D input -> Output shape:", y_feat.shape)

if __name__ == "__main__":
    test_lstm()
