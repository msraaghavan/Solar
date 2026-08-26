"""A U-Net decoder over a ``timm`` encoder, specialised for thin dark features.

Written directly against ``timm`` rather than pulled from
``segmentation_models_pytorch`` for two reasons: the Kaggle runtime ships timm
but not smp, and the decoder needs two departures from the stock design.

*Full-resolution output.*  Standard timm encoders stride the input by 32 and the
usual U-Net stops decoding at stride 2, upsampling the logits at the end.  Solar
filament barbs are a few pixels wide, so the decoder here runs all the way back
to stride 1 with a learned block.

*Stride-1 stem skip* (``stem_skip``, default off).  A timm ``features_only``
encoder emits nothing above stride 2, so without this the final stride-2 ->
stride-1 block has no skip at all: it sees a nearest-neighbour upsample of the
/2 features through 16 channels and nothing else, and therefore cannot locate an
edge more precisely than the /2 grid allows.  That is a candidate explanation for
mean matched IoU sitting at 0.67 from epoch 6 onwards regardless of encoder,
training length or fold - see the SQ section of HANDOVER.md.  Switching it on
adds one full-resolution convolution over the input and hands its output to that
last block as a skip.

(An earlier version of this docstring asserted the stem skip was already carried
across.  It was not - the last entry of ``skips`` was a literal 0 - so any
reasoning that relied on the model having full-resolution detail available was
reasoning about a model that did not exist.)

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
        stem_skip: bool = False,
        stem_channels: int = 16,
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

        # Encoder features run [/2, /4, /8, /16, /32]; decode back to /1.  The
        # final block's skip is the stem when enabled and nothing otherwise -
        # there is no encoder feature at stride 1 to use instead.
        self.stem = ConvBNAct(in_channels, stem_channels) if stem_skip else None
        skips = [
            encoder_channels[3],
            encoder_channels[2],
            encoder_channels[1],
            encoder_channels[0],
            stem_channels if stem_skip else 0,
        ]
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

    @staticmethod
    def tiling_for(checkpoint: dict) -> tuple[int, int]:
        """``(tile_size, stride)`` for inference on a model, from how it trained.

        ``predict_full`` defaults to 512/384 regardless of the checkpoint, which
        has been correct only because every model so far trained on 512 tiles.  A
        model trained at another tile size and evaluated at 512 sees a different
        amount of context than it ever saw in training, and scores worse for a
        reason unrelated to whatever was being tested - a false negative
        indistinguishable from a real one.

        The stride keeps the 0.75 overlap of the 512/384 pair, so the
        cosine-blended reconstruction behaves the same way at any tile size.
        """
        tile = int(checkpoint.get("args", {}).get("tile_size") or 512)
        return tile, int(round(tile * 0.75))

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device: str = "cpu") -> "FilamentNet":
        """Rebuild the architecture a checkpoint was trained with, from its weights.

        Four call sites used to reconstruct this by hand, each reading a slightly
        different subset of the saved state - and each therefore able to drift.
        Two architecture choices are invisible in ``args`` for older checkpoints
        and have to be read off the weights themselves:

        * ``out_channels``, which is 2 when the auxiliary spine head was on;
        * ``stem_skip``, which adds ``stem.*`` parameters.

        Reading the weights rather than ``args`` also means a checkpoint trained
        before a flag existed still loads, which is what keeps every earlier
        result reproducible.
        """
        state = checkpoint["model"]
        args = checkpoint.get("args", {})
        return cls(
            encoder_name=args.get("encoder", "tf_efficientnet_b4"),
            pretrained=False,
            out_channels=state["head.weight"].shape[0],
            stem_skip=any(k.startswith("stem.") for k in state),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.encoder(x)          # [/2, /4, /8, /16, /32]
        skips = [
            features[3],
            features[2],
            features[1],
            features[0],
            self.stem(x) if self.stem is not None else None,
        ]

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
