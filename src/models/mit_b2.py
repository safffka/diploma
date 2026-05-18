import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layer_norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.layer_norm = nn.LayerNorm(dim)

        self.sr_ratio = sr_ratio

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            kv = x.permute(0, 2, 1).reshape(B, C, H, W)
            kv = self.sr(kv).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.layer_norm(kv)
        else:
            kv = x

        k = self.key(kv).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(kv).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden = dim * expansion_ratio
        self.dense1 = nn.Linear(dim, hidden)
        self.dwconv = DWConv(hidden)
        self.dense2 = nn.Linear(hidden, dim)

    def forward(self, x, H, W):
        x = self.dense1(x)
        x = F.gelu(x)
        x = self.dwconv(x, H, W)
        x = F.gelu(x)
        x = self.dense2(x)
        return x


class AttentionOutput(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)

    def forward(self, x):
        return self.dense(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.self = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.output = AttentionOutput(dim)

    def forward(self, x, H, W):
        x = self.self(x, H, W)
        x = self.output(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, num_heads, sr_ratio)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, mlp_ratio)

    def forward(self, x, H, W):
        x = x + self.attention(self.layer_norm_1(x), H, W)
        x = x + self.mlp(self.layer_norm_2(x), H, W)
        return x


class MiTB2Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        embed_dims = [64, 128, 320, 512]
        depths = [3, 4, 6, 3]
        num_heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        mlp_ratios = [4, 4, 4, 4]
        patch_strides = [4, 2, 2, 2]
        patch_kernels = [7, 3, 3, 3]

        self.patch_embeddings = nn.ModuleList()
        self.block = nn.ModuleList()
        self.layer_norm = nn.ModuleList()

        for i in range(4):
            ch_in = in_channels if i == 0 else embed_dims[i - 1]
            self.patch_embeddings.append(
                OverlapPatchEmbedding(ch_in, embed_dims[i], patch_kernels[i], patch_strides[i])
            )
            blocks = nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i], sr_ratios[i], mlp_ratios[i])
                for _ in range(depths[i])
            ])
            self.block.append(blocks)
            self.layer_norm.append(nn.LayerNorm(embed_dims[i]))

    def forward(self, x):
        features = []
        for i in range(4):
            x, H, W = self.patch_embeddings[i](x)
            for blk in self.block[i]:
                x = blk(x, H, W)
            x = self.layer_norm[i](x)
            B, N, C = x.shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            features.append(x)
        return features

    def load_pretrained(self, weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]

        new_state = {}
        prefix = "segformer.encoder."
        for k, v in state.items():
            if k.startswith(prefix):
                new_state[k[len(prefix):]] = v

        result = self.load_state_dict(new_state, strict=False)
        loaded = len(new_state) - len(result.unexpected_keys)
        total = len(self.state_dict())
        print(f"Loaded {loaded}/{total} keys from {weights_path}")
        if result.missing_keys:
            print(f"  Missing: {len(result.missing_keys)} keys")
        if result.unexpected_keys:
            print(f"  Unexpected: {len(result.unexpected_keys)} keys")
        return result
