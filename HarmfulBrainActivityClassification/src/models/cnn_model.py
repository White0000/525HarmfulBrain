import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        c = torch.cat([avg_out, max_out], dim=1)
        s = torch.sigmoid(self.conv(c))
        return x * s

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_ca=True, use_sa=True, drop=0.0):
        super().__init__()
        self.dropout = nn.Dropout2d(p=drop)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels) if use_ca else nn.Identity()
        self.sa = SpatialAttention(kernel_size=7) if use_sa else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        r = x
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        y = self.ca(y)
        y = self.sa(y)
        r = self.shortcut(r)
        y += r
        return F.relu(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, drop=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(drop)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        r = x
        y, _ = self.self_attn(x, x, x)
        y = r + y
        y = self.norm1(y)
        r2 = y
        y = self.ff(y)
        y = r2 + y
        y = self.norm2(y)
        return y

class SimpleCNN(nn.Module):
    """
    带残差 + 注意力 + 小型 Transformer 的 CNN。
    同时支持在输入为 (B, F) 仅有两个维度的情况下，
    自动将 F reshape 成 (1, sqrt(F), sqrt(F))，
    以兼容场景仅有特征向量数据却想用CNN的情况。
    """
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.block1 = ResidualBlock(32, 64, stride=2, use_ca=True, use_sa=True, drop=0.2)
        self.block2 = ResidualBlock(64, 64, stride=1, use_ca=True, use_sa=True, drop=0.2)
        self.block3 = ResidualBlock(64, 128, stride=2, use_ca=True, use_sa=True, drop=0.3)
        self.block4 = ResidualBlock(128, 128, stride=1, use_ca=True, use_sa=True, drop=0.3)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.conv_head = nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(128)
        self.transformer = TransformerBlock(d_model=128, nhead=4, dim_feedforward=256, drop=0.1)
        self.fc = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        输入形状:
          - (B, C, H, W): 正常的CNN图像输入;
          - (B, F): 如果只有特征维度F, 本模块会尝试将F开根reshape为 (1, sqrt(F), sqrt(F))。
            若F不是perfect square, 会抛出错误.
        """
        if x.dim() == 2:
            b, f = x.shape
            s = int(math.sqrt(f))
            if s*s != f:
                raise ValueError(f"Cannot reshape input (B,F) with F={f} to a square. Provide a valid shape or modify code.")
            # 把 (B,F) -> (B,1,s,s)
            x = x.view(b, 1, s, s)

        # 这里 x 应该是 (B, C, H, W)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = F.relu(self.bn_head(self.conv_head(x)))
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(0,2,1)   # (B,HW,C)
        x = self.transformer(x)
        x = x.mean(dim=1)                     # (B,C)
        return self.fc(x)

def test_cnn():
    # 测试1: CNN图像输入 (B,C,H,W)
    b = 8
    c = 1
    h, w = 64, 64
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(in_channels=c, num_classes=5).to(d)
    x_img = torch.randn(b, c, h, w, device=d)
    y_img = model(x_img)
    print("Image input -> Output shape:", y_img.shape)

    # 测试2: 仅有 (B,F) 特征输入
    # 假设 F=64 => sqrt(64)=8 => 8x8
    x_feat = torch.randn(b, 64, device=d)  # (B,F)
    y_feat = model(x_feat)
    print("Feature input -> Output shape:", y_feat.shape)

if __name__ == "__main__":
    test_cnn()
