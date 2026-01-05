import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


# -------------------- [Segmentation Head] -------------------- #

def get_padding_shape(filter_shape, stride): 
    def _pad_top_bottom(filter_dim, stride_val): 
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes): 
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init 


class Unit3Dpy(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        activation="relu",
        padding="SAME",
        use_bias=False,
        use_bn=True,
    ):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == "VALID":
            padding_shape = 0
        else:
            raise ValueError(
                "padding should be in [VALID|SAME] but got {}".format(padding)
            )

        if padding == "SAME":
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride=stride, bias=use_bias
                )
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias,
                )
        elif padding == "VALID":
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias,
            )
        else:
            raise ValueError(
                "padding should be in [VALID|SAME] but got {}".format(padding)
            )

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == "relu":
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == "SAME" and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class SegmentationHead(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1, target_size=(64, 64, 64)):
        super().__init__()
        self.target_size = target_size
        
        self.block1 = Unit3Dpy(
            in_channels=in_channels,
            out_channels=832,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )
        
        self.block2 = Unit3Dpy(
            in_channels=832,
            out_channels=480,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )
        
        self.block3 = Unit3Dpy(
            in_channels=480,
            out_channels=192,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )
        
        self.block4 = Unit3Dpy(
            in_channels=192,
            out_channels=64,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )
        
        self.block5 = Unit3Dpy(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="SAME",
            use_bias=False,
            use_bn=True
        )

        self.last_layer = nn.Conv3d(64, out_channels, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        
        
    def forward(self, features, return_bottleneck=False):
        out0 = self.block1(features[0])
        out = F.interpolate(out0, size=(16, 8, 8), mode='trilinear', align_corners=False) 
        
        out1 = self.block2(out + features[1])
        out = F.interpolate(out1, size=(32, 16, 16), mode='trilinear', align_corners=False)
        
        out2 = self.block3(out + features[2])
        out = F.interpolate(out2, size=(32, 32, 32), mode='trilinear', align_corners=False)
        
        out3 = self.block4(out + features[3])
        out = F.interpolate(out3, size=(32, 64, 64), mode='trilinear', align_corners=False)
        
        out4 = self.block5(out + features[4])
        out = F.interpolate(out4, size=self.target_size, mode='trilinear', align_corners=False)
        
        out = self.last_layer(out)

        if return_bottleneck:
            return out, out4, out3, out2, out1, out0
        return out
    

# -------------------- [Classifier Head] -------------------- #

class AttentionPool3d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C = x.shape[:2]

        attn = self.attn(x)                  # [B, 1, T, H, W]
        attn = attn.view(B, -1)              # [B, THW]
        attn = torch.softmax(attn, dim=1)

        x_flat = x.view(B, C, -1)             # [B, C, THW]
        pooled = torch.sum(x_flat * attn.unsqueeze(1), dim=2)
        return pooled                         # [B, C]
    

class MultiPoolFusion(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.attn_pool = AttentionPool3d(in_channels)

        self.proj = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels),
            nn.GELU()
        )

    def forward(self, x):
        f_avg = self.avg_pool(x).flatten(1)   # [B, C]
        f_max = self.max_pool(x).flatten(1)   # [B, C]
        f_att = self.attn_pool(x)              # [B, C]

        f = torch.cat([f_avg, f_max, f_att], dim=1)  # [B, 3C]
        return self.proj(f)                            # [B, C]


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        head_type: str = "avg_head",
        tail_type: str = "linear"
    ):
        super().__init__()
        self.head_type = head_type
        self.tail_type = tail_type
        self.in_channels = in_channels

        if head_type == "avg_head":
            self.head = self._build_avg_head()
        elif head_type == "max_head":
            self.head = self._build_max_head()
        elif head_type == "atten_head":
            self.head = self._build_atten_head()
        elif head_type == "multi_head":
            self.head = self._build_multi_head()
        else:
            raise ValueError(f"Unknown classifier head type: {head_type}")
        
    def _build_tail(self, in_dim: int, tail_type: str):
        if tail_type == "linear":
            return nn.Linear(in_dim, 1)

        elif tail_type == "nonlinear":
            return nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(in_dim // 2, 1)
            )

        else:
            raise ValueError(f"Unknown tail type: {tail_type}")

    # ---------- Average Pool ----------    
    def _build_avg_head(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            self._build_tail(self.in_channels, self.tail_type)
        )
    
    # ---------- Max Pool ---------- 
    def _build_max_head(self):
        return nn.Sequential(
            nn.AdaptiveMaxPool3d((1, 1, 1)),
            nn.Flatten(),
            self._build_tail(self.in_channels, self.tail_type)
        )
    
    # ---------- Attention Pool ---------- 
    def _build_atten_head(self):
        return nn.Sequential(
            AttentionPool3d(self.in_channels),
            nn.LayerNorm(self.in_channels),
            self._build_tail(self.in_channels, self.tail_type)
        )

    # ---------- Multi-head Pool Fusion ---------- 
    def _build_multi_head(self):
        return nn.Sequential(
            MultiPoolFusion(self.in_channels),   # output [B, C]
            nn.LayerNorm(self.in_channels),
            self._build_tail(self.in_channels, self.tail_type)
        )

    def forward(self, x):
        return self.head(x)
    
    
# -------------------- [Multi Task FiLM Model] -------------------- #

class MultiTaskFiLMModel(nn.Module):
    def __init__(
        self, 
        feature_extractor, 
        aux_model: Optional[nn.Module] = None, 
        aux_task: str = "none",
        use_aux_model: bool = False,
        use_seg_gate: bool = False,
        use_clinical_gate=False,
        clinical_dim=2,   # age + gender
        cls_head_type = "avg_head",
        cls_tail_type = "linear"
    ):
        super().__init__()
        self.backbone = feature_extractor

        self.use_aux_model = use_aux_model
        self.use_seg_gate = use_seg_gate
        self.use_clinical_gate = use_clinical_gate

        # ---------- Auxiliary Model ----------
        if self.use_aux_model:
            self.aux_model = aux_model
            self.aux_task = aux_task

        C_last = 832
        C = 1024 # I3D output channels

        # ---------- Segmentation Gate ----------
        if self.use_seg_gate:
            self.seg_gate = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(C_last, C),
                nn.Sigmoid()
            )

        # ---------- Clinical Gate ----------
        if self.use_clinical_gate:
            self.clinical_gate = nn.Sequential(
                nn.Linear(clinical_dim, C),
                nn.Sigmoid()
            )

        # ---------- Classifier ----------
        self.classifier_head = ClassifierHead(
            in_channels=C,
            head_type=cls_head_type,
            tail_type=cls_tail_type
        )


    def extract_feature(self, images):
        return self.backbone(images) 
    
    def forward(
        self,
        features_main,
        age=None,
        gender=None,
        validate=False
    ):
        """
        features_main: list of feature maps from backbone
        age: (B, 1)
        gender: (B,) or (B, 1)
        """

        # 1. Base classification feature
        F_cls = features_main[0]   # (B, 1024, D, H, W)

        # 2. Auxiliary task (only for training)
        aux_outputs = None
        if not validate and self.use_aux_model:

            # 3. Seg-aware gating (optional)
            if self.use_seg_gate:
                aux_outputs, _, _, _, _, seg_bottleneck = self.aux_model(
                    features_main, return_bottleneck=True
                )

                F_seg = seg_bottleneck   # deepest feature
                alpha = self.seg_gate(F_seg)  # (B, 1024)
                alpha = alpha.view(alpha.size(0), alpha.size(1), 1, 1, 1)
                F_cls = F_cls * (1.0 + 0.5 * alpha)

            else:
                aux_outputs = self.aux_model(features_main)

        # 4. Clinical gating (optional)
        if self.use_clinical_gate:
            assert age is not None and gender is not None

            if age.dim() == 1:
                age = age.unsqueeze(1)

            if gender.dim() == 1:
                gender = gender.unsqueeze(1)

            clinical = torch.cat([age, gender], dim=1)  # (B, 2)
            beta = self.clinical_gate(clinical)         # (B, 1024)
            beta = beta.view(beta.size(0), beta.size(1), 1, 1, 1)

            F_cls = F_cls * (1.0 + beta)

        # 5. Classification
        main_logits = self.classifier_head(F_cls)

        return main_logits, aux_outputs
    

class MultiTaskFiLMModel_baseline(nn.Module):
    def __init__(
        self, 
        feature_extractor, 
        aux_model: Optional[nn.Module] = None, 
        aux_task: str = "none",
        use_aux_model: bool = False,
        use_seg_gate: bool = False,
        use_clinical_gate=False,
        clinical_dim=2,   # age + gender
    ):
        super().__init__()
        self.backbone = feature_extractor

        self.use_aux_model = use_aux_model
        self.use_seg_gate = use_seg_gate
        self.use_clinical_gate = use_clinical_gate

        # ---------- Auxiliary Model ----------
        if self.use_aux_model:
            self.aux_model = aux_model
            self.aux_task = aux_task

        C_last = 832
        C = 1024 # I3D output channels

        # ---------- Segmentation Gate ----------
        if self.use_seg_gate:
            self.seg_gate = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(C_last, C),
                nn.Sigmoid()
            )

        # ---------- Clinical Gate ----------
        if self.use_clinical_gate:
            self.clinical_gate = nn.Sequential(
                nn.Linear(clinical_dim, C),
                nn.Sigmoid()
            )

        # ---------- Classifier ----------
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)), 
            nn.Flatten(), 
            nn.Linear(1024, 1),
        )


    def extract_feature(self, images):
        return self.backbone(images) 
    
    def forward(
        self,
        features_main,
        age=None,
        gender=None,
        validate=False
    ):
        """
        features_main: list of feature maps from backbone
        age: (B, 1)
        gender: (B,) or (B, 1)
        """

        # 1. Base classification feature
        F_cls = features_main[0]   # (B, 1024, D, H, W)

        # 2. Auxiliary task (only for training)
        aux_outputs = None
        if not validate and self.use_aux_model:

            # 3. Seg-aware gating (optional)
            if self.use_seg_gate:
                aux_outputs, _, _, _, _, seg_bottleneck = self.aux_model(
                    features_main, return_bottleneck=True
                )

                F_seg = seg_bottleneck   # deepest feature
                alpha = self.seg_gate(F_seg)  # (B, 1024)
                alpha = alpha.view(alpha.size(0), alpha.size(1), 1, 1, 1)
                F_cls = F_cls * (1.0 + 0.5 * alpha)

            else:
                aux_outputs = self.aux_model(features_main)

        # 4. Clinical gating (optional)
        if self.use_clinical_gate:
            assert age is not None and gender is not None

            if age.dim() == 1:
                age = age.unsqueeze(1)

            if gender.dim() == 1:
                gender = gender.unsqueeze(1)

            clinical = torch.cat([age, gender], dim=1)  # (B, 2)
            beta = self.clinical_gate(clinical)         # (B, 1024)
            beta = beta.view(beta.size(0), beta.size(1), 1, 1, 1)

            F_cls = F_cls * (1.0 + beta)

        # 5. Classification
        main_logits = self.classifier_head(F_cls)

        return main_logits, aux_outputs