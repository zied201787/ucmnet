import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from xnor_layers import XNORConv2d, XNORLinear

__all__ = ['UCM_NetV2BNN']


class LayerNorm(nn.Module):
    """From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class AttributeGate(nn.Module):
    def __init__(self, channels):
        super(AttributeGate, self).__init__()
        self.gate = nn.Sequential(
            XNORConv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = XNORConv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.layer_norm(x, [H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DWConv1(nn.Module):
    def __init__(self, dim=768):
        super(DWConv1, self).__init__()
        self.dwconv = XNORConv2d(2 * dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.layer_norm(x, [H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = XNORConv2d(in_chans, embed_dim, kernel_size=1, stride=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class UCMBlock1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = XNORLinear(dim, mlp_hidden_dim)
        self.dwconv = DWConv1(mlp_hidden_dim)
        self.act = act_layer()
        self.fc2 = XNORLinear(mlp_hidden_dim, dim)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.norm2(x)

        B, N, C = x.shape
        x1 = x.clone()

        x = x.reshape(B * N, C).contiguous()
        x2 = x.clone()

        x = self.fc1(x)
        x = x.reshape(B, N, -1).contiguous()
        x += x1

        x2[[0, B * N - 1], :] = x2[[B * N - 1, 0], :]
        x2 = self.fc2(x2)
        x2[[0, B * N - 1], :] = x2[[B * N - 1, 0], :]
        x2 = x2.reshape(B, N, -1).contiguous()
        x2 += x1
        x = torch.cat((x, x2), dim=2)

        x = self.dwconv(x, H, W)
        x += x1
        x = x + self.drop_path(x)

        return x


class UCM_NetV2BNN(nn.Module):
    """Conv 3 + MLP 2 + shifted MLP with less parameters"""

    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 img_size=256, patch_size=16, in_chans=3, embed_dims=[8, 16, 24, 32, 48, 64, 3],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(embed_dims[-1], embed_dims[0], 3, stride=1, padding=1)
        self.ebn1 = nn.GroupNorm(4, embed_dims[0])
        self.ebn2 = nn.GroupNorm(4, embed_dims[1])
        self.ebn3 = nn.GroupNorm(4, embed_dims[2])

        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder blocks
        self.block_0_1 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.block0 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.block1 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.block3 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[5], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        # Decoder blocks
        self.dblock0 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock3 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock4 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 2, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[3], embed_dim=embed_dims[4])

        self.patch_embed5 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2,
            in_chans=embed_dims[4], embed_dim=embed_dims[5])

        # Decoders
        self.decoder0 = XNORConv2d(embed_dims[5], embed_dims[4], 1, stride=1, padding=0)
        self.decoder1 = XNORConv2d(embed_dims[4], embed_dims[3], 1, stride=1, padding=0)
        self.decoder2 = XNORConv2d(embed_dims[3], embed_dims[2], 1, stride=1, padding=0)
        self.decoder3 = XNORConv2d(embed_dims[2], embed_dims[1], 1, stride=1, padding=0)
        self.decoder4 = XNORConv2d(embed_dims[1], embed_dims[0], 1, stride=1, padding=0)
        self.decoder5 = XNORConv2d(embed_dims[0], embed_dims[-1], 1, stride=1, padding=0)

        # Normalization layers
        self.dbn0 = nn.GroupNorm(4, embed_dims[4])
        self.dbn1 = nn.GroupNorm(4, embed_dims[3])
        self.dbn2 = nn.GroupNorm(4, embed_dims[2])
        self.dbn3 = nn.GroupNorm(4, embed_dims[1])
        self.dbn4 = nn.GroupNorm(4, embed_dims[0])

        # Output layers
        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, kernel_size=1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        self.final = nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)

    def forward(self, x, inference_mode=False):
        B = x.shape[0]

        ### Encoder ###
        # Stage 1
        out = self.encoder1(x)
        out = F.relu(F.max_pool2d(self.ebn1(out), 2, 2))
        t1 = out

        # Stage 2
        out, H, W = self.patch_embed1(out)
        for blk in self.block_0_1:
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out

        # Stage 3
        out, H, W = self.patch_embed2(out)
        for blk in self.block0:
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out

        # Stage 4
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # Bottleneck 1
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out

        # Bottleneck 2
        out, H, W = self.patch_embed5(out)
        for blk in self.block3:
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Decoder ###
        # Stage 4
        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t5)

        if not inference_mode:
            outtpre0 = F.interpolate(out, scale_factor=32, mode='bilinear', align_corners=True)
            outtpre0 = self.finalpre0(outtpre0)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock0:
            out = blk(out, H, W)

        # Stage 3
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)

        if not inference_mode:
            outtpre1 = F.interpolate(out, scale_factor=16, mode='bilinear', align_corners=True)
            outtpre1 = self.finalpre1(outtpre1)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        # Stage 2
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)

        if not inference_mode:
            outtpre2 = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
            outtpre2 = self.finalpre2(outtpre2)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        # Stage 1
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)

        if not inference_mode:
            outtpre3 = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
            outtpre3 = self.finalpre3(outtpre3)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock3:
            out = blk(out, H, W)

        # Final stage
        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)

        if not inference_mode:
            outtpre4 = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
            outtpre4 = self.finalpre4(outtpre4)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock4:
            out = blk(out, H, W)

        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear')
        out = self.final(out)

        if not inference_mode:
            return (outtpre0, outtpre1, outtpre2, outtpre3, outtpre4), out
        else:
        return out


class InferenceModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(InferenceModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, inference_mode=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_gflops(model, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input = torch.randn(1, 3, input_size, input_size).to(device)

    wrapped_model = InferenceModelWrapper(model)

    with torch.no_grad():
        macs, params = profile(wrapped_model, inputs=(input,), verbose=False)

    return macs / (10 ** 9)


if __name__ == "__main__":
    num_classes = 1
    input_channels = 3
    model = UCM_NetV2BNN(num_classes=num_classes, input_channels=input_channels)
    model.cuda()

    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    input_size = 256
    gflops = compute_gflops(model, input_size)
    print(f"GFLOPS: {gflops:.4f}")