import torch.nn.functional as F
import torch
from torch import nn, Tensor
import numpy as np
import torch.nn as nn
from skimage.transform import resize
from itertools import filterfalse as ifilterfalse
from torch.autograd import Variable

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def focal_loss(prediction, target, alpha_in=0.25, gamma_in=2):
    bce = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
    p = torch.sigmoid(prediction)
    alpha = alpha_in * target + (1 - alpha_in) * (1 - target)
    focal_weight = alpha * (1 - p).pow(gamma_in)
    focal_loss = focal_weight * bce
    return focal_loss.mean()

def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    # focalloss = focal_loss(prediction, target)
    prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


# --------------------------- HELPER FUNCTIONS ---------------------------

def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
