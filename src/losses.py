"""Losses for a sparse, noisily-labelled binary segmentation target.

The mix is chosen for a specific reason.  Because roughly a third of the images
are labelled by more than one annotator and those annotators agree only
moderately, the per-pixel target is genuinely stochastic: the right thing for
the network to predict is P(a randomly drawn annotator marks this pixel).
Binary cross-entropy is a proper scoring rule, so its minimiser *is* that
posterior - which is exactly what the operating-point analysis in
``postprocess.py`` assumes it is thresholding.

Soft Dice is not proper in that sense and pulls predictions towards the mode,
but it copes far better with the ~0.3% positive rate.  Both are used, with Dice
weighted low enough to act as a conditioner on the imbalance rather than as the
objective.  Any residual miscalibration is absorbed by tuning the decision
threshold against Panoptic Quality directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    pos_weight: float = 4.0,
) -> torch.Tensor:
    """BCE restricted to on-disk pixels, with positives up-weighted."""
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def soft_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Dice computed per image, then averaged.

    Per-image (rather than per-batch) so that a tile containing one small
    filament counts as much as a tile containing five large ones.
    """
    probs = torch.sigmoid(logits) * weight
    target = target * weight
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dims)
    total = probs.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + smooth) / (total + smooth)).mean()


class FilamentLoss(nn.Module):
    """``bce + dice_weight * dice``, with optional deep supervision."""

    def __init__(
        self,
        pos_weight: float = 4.0,
        dice_weight: float = 0.5,
        aux_weights: tuple[float, ...] = (0.4, 0.2),
        spine_weight: float = 0.0,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.aux_weights = aux_weights
        # Weight on the auxiliary spine channel.  Kept well below 1: the spine
        # is a means of shaping the representation - teaching the network that a
        # filament is one elongated object with an axis, unlike a sunspot - not
        # something the metric ever asks for.
        self.spine_weight = spine_weight

    def _single(
        self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Loss over channel 0, plus the spine channel when both provide one."""
        loss = masked_bce(logits[:, :1], target[:, :1], weight, self.pos_weight) + (
            self.dice_weight * soft_dice(logits[:, :1], target[:, :1], weight)
        )
        if self.spine_weight > 0 and logits.shape[1] > 1 and target.shape[1] > 1:
            spine = masked_bce(
                logits[:, 1:2], target[:, 1:2], weight, self.pos_weight
            ) + self.dice_weight * soft_dice(logits[:, 1:2], target[:, 1:2], weight)
            loss = loss + self.spine_weight * spine
        return loss

    def forward(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, tuple):
            logits, aux_logits = outputs
        else:
            logits, aux_logits = outputs, []

        loss = self._single(logits, target, weight)
        for aux, aux_weight in zip(aux_logits, self.aux_weights):
            # Auxiliary heads sit at stride 2 and 4; pool the target to match
            # rather than upsampling the logits, which would be much costlier.
            scale = target.shape[-1] // aux.shape[-1]
            small_target = F.avg_pool2d(target, scale)
            small_weight = F.avg_pool2d(weight, scale)
            loss = loss + aux_weight * self._single(aux, small_target, small_weight)
        return loss
