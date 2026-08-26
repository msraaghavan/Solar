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
    smoothing: float = 0.0,
) -> torch.Tensor:
    """BCE restricted to on-disk pixels, with positives up-weighted.

    ``smoothing`` shrinks the positive target from 1 towards ``1 - smoothing``
    and leaves negatives at 0.  The asymmetry is deliberate.  Annotator
    disagreement is overwhelmingly about *which faint structures count as a
    filament* rather than about the quiet Sun, and negatives outnumber positives
    roughly 280:1, so lifting the negative target even slightly would swamp the
    objective.  Capping the positive target instead removes the incentive to
    drive one annotator's marks to probability 1 - which is what the training
    curves show happening as PQ falls away.
    """
    if smoothing > 0.0:
        target = target * (1.0 - smoothing)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def boundary_band(target: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """1 inside a band of ``radius`` pixels either side of every mask edge.

    Morphological gradient, done with pooling so it stays on the GPU and inside
    autograd's world: dilation is max-pooling, erosion is max-pooling the
    negative.  The target is binary per reading, so the difference is exactly the
    transition band.

    Measured on 244 real instances, the max-inscribed width is median 15.4 px
    (p10 8.4, p90 29.7) - filaments are elongated but not hairline, and at
    ``radius=2`` only 0.4% of instances erode away entirely (9.8% at radius 3).
    So the band is a genuine rim of a few pixels either side of the edge rather
    than the whole object, and radius is the knob that decides how much of a
    ~15 px cross-section counts as "edge".

    (An earlier version of this docstring asserted that erosion "frequently
    removes a whole instance".  It does not, and the number above is why the
    claim is now a measurement.)
    """
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(target, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-target, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


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
        smoothing: float = 0.0,
        boundary_weight: float = 0.0,
        boundary_radius: int = 2,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.aux_weights = aux_weights
        self.smoothing = smoothing
        # Extra weight on pixels within boundary_radius of a mask edge; 0 is off
        # and reproduces every result measured before this existed, exactly.
        #
        # Why here and not a Lovasz term: the metric matches at IoU > 0.5
        # strictly and scores each matched pair by its IoU, and the measured
        # sensitivity is about 1.06 PQ per unit of mean matched IoU
        # (tools/iou_headroom.py).  A filament's cross-section is ~15 px, so a
        # rim two pixels either side of the edge is a small share of its area but
        # carries most of the disagreement: the predictions are only 8% too large
        # by area (median_area_ratio 1.082), which is far too little to explain
        # an IoU of 0.67 - so the loss is in *where* the edge sits, not in how
        # big the mask is overall.
        #
        # Crucially this does *not* break the argument the module docstring makes
        # for BCE.  BCE decomposes over pixels, so a per-pixel weight leaves each
        # pixel's minimiser at P(a random annotator marks it) and only changes how
        # much that pixel contributes to the gradient.  A Lovasz or IoU surrogate
        # would not be separable and would distort the calibration that
        # postprocess.py's threshold tuning depends on.
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius
        # Weight on the auxiliary spine channel.  Kept well below 1: the spine
        # is a means of shaping the representation - teaching the network that a
        # filament is one elongated object with an axis, unlike a sunspot - not
        # something the metric ever asks for.
        self.spine_weight = spine_weight

    def _single(
        self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Loss over channel 0, plus the spine channel when both provide one."""
        bce_weight = weight
        if self.boundary_weight > 0:
            band = boundary_band(target[:, :1], self.boundary_radius)
            bce_weight = weight * (1.0 + self.boundary_weight * band)
        # The boundary emphasis applies to BCE only.  soft_dice keeps the plain
        # on-disk weight: it is a ratio of masses, so re-weighting pixels inside
        # it would change what "overlap" means rather than where the loss looks.
        loss = masked_bce(
            logits[:, :1], target[:, :1], bce_weight, self.pos_weight, self.smoothing
        ) + self.dice_weight * soft_dice(logits[:, :1], target[:, :1], weight)
        if self.spine_weight > 0 and logits.shape[1] > 1 and target.shape[1] > 1:
            spine = masked_bce(
                logits[:, 1:2], target[:, 1:2], weight, self.pos_weight, self.smoothing
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
