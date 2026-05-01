"""Attention U-Net с предобученным ResNet50 encoder."""
import torch
import torch.nn as nn
import torchvision.models as models


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg  = nn.Conv2d(g_ch, inter_ch, 1, bias=False)
        self.Wx  = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_up = nn.functional.interpolate(self.Wg(g), size=x.shape[2:])
        att  = self.psi(self.relu(g_up + self.Wx(x)))
        return x * att


class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.att  = AttentionGate(in_ch // 2, skip_ch, skip_ch // 2)
        self.conv = ConvBNReLU(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:])
        skip = self.att(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.mha  = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens     = x.flatten(2).transpose(1, 2)
        out, _     = self.mha(tokens, tokens, tokens)
        out        = self.norm(out + tokens)
        return out.transpose(1, 2).reshape(B, C, H, W)


class AttentionUNetPretrained(nn.Module):
    """Attention U-Net с ResNet50 предобученным на ImageNet."""
    def __init__(self, num_classes=6, pretrained=True, num_heads=8):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = models.resnet50(weights=weights)

        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.bottleneck = nn.Sequential(
            ConvBNReLU(2048, 1024),
            MultiHeadSelfAttention(1024, num_heads),
            ConvBNReLU(1024, 1024),
        )

        self.dec4 = AttentionDecoderBlock(1024, 1024, 512)
        self.dec3 = AttentionDecoderBlock(512,  512,  256)
        self.dec2 = AttentionDecoderBlock(256,  256,  128)
        self.dec1 = AttentionDecoderBlock(128,  64,   64)

        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        p  = self.pool(e0)
        e1 = self.enc1(p)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b,  e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        return self.head(self.up_final(d1))
