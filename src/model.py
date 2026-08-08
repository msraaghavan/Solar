"""A U-Net decoder over a ``timm`` encoder, specialised for thin dark features.

Written directly against ``timm`` rather than pulled from
``segmentation_models_pytorch`` for two reasons: the Kaggle runtime ships timm
but not smp, and the decoder needs two departures from the stock design.

*Full-resolution output.*  Standard timm encoders stride the input by 32 and the
usual U-Net stops decoding at stride 2, upsampling the logits at the end.  Solar
filament barbs are a few pixels wide, so the decoder here runs all the way back
to stride 1 with a learned block, and a stride-1 stem skip is carried across.

*Squeeze-excite in the decoder.*  Filament contrast varies with radius and
seeing; a cheap channel gate lets each block rescale features rather than
relying on the encoder to have normalised them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(w))))
        return x * w


class DecoderBlock(nn.Module):
    """Upsample, concatenate the skip, then two convolutions with a channel gate."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)
        self.gate = SqueezeExcite(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            # Guard against odd input sizes, where the skip can be a pixel larger.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.gate(self.conv2(self.conv1(x)))


class FilamentNet(nn.Module):
    """Encoder-decoder producing per-pixel filament logits at input resolution.

    ``deep_supervision`` additionally returns logits from the stride-2 and
    stride-4 decoder stages during training; supervising them stabilises the
    early epochs when the positive class covers only ~0.3% of pixels.
    """

    def __init__(
        self,
        encoder_name: str = "tf_efficientnet_b4",
        in_channels: int = 3,
        pretrained: bool = True,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        deep_supervision: bool = True,
        out_channels: int = 1,
    ):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
        )
        encoder_channels = list(self.encoder.feature_info.channels())
        reductions = list(self.encoder.feature_info.reduction())
        if reductions[-1] != 32 or len(encoder_channels) != 5:
            raise ValueError(
                f"expected a 5-stage /32 encoder, got reductions={reductions}"
            )
        self.deep_supervision = deep_supervision

        # Encoder features run [/2, /4, /8, /16, /32]; decode back to /1.
        skips = [encoder_channels[3], encoder_channels[2], encoder_channels[1], encoder_channels[0], 0]
        blocks = []
        in_ch = encoder_channels[4]
        for out_ch, skip_ch in zip(decoder_channels, skips):
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # Channel 0 is always the filament mask.  A second channel, when
        # enabled, predicts the filament spine as an auxiliary task; it is never
        # used at inference, only to shape the representation during training.
        self.out_channels = out_channels
        self.head = nn.Conv2d(decoder_channels[-1], out_channels, 1)
        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [
                    nn.Conv2d(decoder_channels[-2], out_channels, 1),
                    nn.Conv2d(decoder_channels[-3], out_channels, 1),
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.encoder(x)          # [/2, /4, /8, /16, /32]
        skips = [features[3], features[2], features[1], features[0], None]

        y = features[4]
        intermediates = []
        for block, skip in zip(self.blocks, skips):
            y = block(y, skip)
            intermediates.append(y)

        logits = self.head(y)
        if self.training and self.deep_supervision:
            aux = [
                self.aux_heads[0](intermediates[-2]),
                self.aux_heads[1](intermediates[-3]),
            ]
            return logits, aux
        return logits
