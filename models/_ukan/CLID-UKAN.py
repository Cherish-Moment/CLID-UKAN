import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from einops import rearrange
from .kan import KANLinear
from .BiPathResBlock import ResBlock, BiPathResBlock


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out

class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels * 3, out_channels)

    def forward(self, x1, x2, x4):
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        self.fc1 = KANLinear(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        self.fc2 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        self.fc3 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)
        return x


class KANBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.layer = KANLayer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
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
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



# class ConvLayer(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(ConvLayer, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, input):
#         return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


    def forward(self, input):
        return self.conv(input)


class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CLID_UKAN(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels=3,
        img_size=224,
        embed_dims=[256, 512, 1024],
        num_heads=[1, 2, 4, 8],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[1, 1, 1],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    ):
        super().__init__()

        kan_input_dim = embed_dims[0]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.encoder1 = BiPathResBlock(input_channels, kan_input_dim // 4, kan_input_dim // 4)
        self.encoder2 = BiPathResBlock(kan_input_dim // 4, kan_input_dim // 2, kan_input_dim // 2)
        self.encoder3 = BiPathResBlock(kan_input_dim // 2, kan_input_dim, kan_input_dim)

        self.up_sample2to1 = nn.Conv2d(128, 64, kernel_size=1)
        self.up_sample3to2 = nn.Conv2d(256, 128, kernel_size=1)
        self.down_sample1to2 = nn.Conv2d(64, 128, kernel_size=1)
        self.down_sample2to3 = nn.Conv2d(128, 256, kernel_size=1)
        self.msaa1 = MSAA(kan_input_dim // 4, kan_input_dim // 4)
        self.msaa2 = MSAA(kan_input_dim // 2, kan_input_dim // 2)
        self.msaa3 = MSAA(kan_input_dim, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])



        self.block1 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.block2 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[2],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.dblock1 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.dblock2 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 2)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 2, embed_dims[0] // 4)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 4)

        self.final = nn.Conv2d(embed_dims[0] // 4, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        B, _, H, W = x.shape
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        # 将各个纬度的特征图做特征融合
        tmp_t2_to_t1 = F.interpolate(self.up_sample2to1(t2), size=(112, 112), mode='bilinear', align_corners=False)
        tmp_t3_to_t1 = F.interpolate(self.up_sample2to1(F.interpolate(self.up_sample3to2(t3), size=(56, 56), mode='bilinear', align_corners=False)), size=(112, 112), mode='bilinear', align_corners=False)
        tmp_t1_to_t2 = F.interpolate(self.down_sample1to2(t1), size=(56, 56), mode='bilinear', align_corners=False)
        tmp_t3_to_t2 = F.interpolate(self.up_sample3to2(t3), size=(56, 56), mode='bilinear', align_corners=False)
        tmp_t2_to_t3 = F.interpolate(self.down_sample2to3(t2), size=(28, 28), mode='bilinear', align_corners=False)
        tmp_t1_to_t3 = F.interpolate(self.down_sample2to3(F.interpolate(self.down_sample1to2(t1), size=(56, 56), mode='bilinear', align_corners=False)), size=(28, 28), mode='bilinear', align_corners=False)
        t1 = self.msaa1(t1,tmp_t2_to_t1, tmp_t3_to_t1)
        t2 = self.msaa2(t2,tmp_t1_to_t2, tmp_t3_to_t2)
        t3 = self.msaa3(t3,tmp_t2_to_t3,tmp_t1_to_t3)

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        out = F.relu(
            F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode="bilinear")
        )

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(
            F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode="bilinear")
        )
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(
            F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode="bilinear")
        )
        out = torch.add(out, t2)
        out = F.relu(
            F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode="bilinear")
        )
        out = torch.add(out, t1)
        out = F.relu(
            F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode="bilinear")
        )

        return self.final(out)
