from .base import BaseWeightedLoss
from .cross_entropy_loss import BCELossWithLogits, CrossEntropyLoss, BCELoss, BCELossWithLogitsAndIgnore, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .focal_loss import SoftmaxFocalLossMultiClass

from .contrastive_loss import NormSoftmaxLoss, ExclusiveNCEwithRankingLoss

__all__ = [
    'BaseWeightedLoss',
    'NormSoftmaxLoss', 'ExclusiveNCEwithRankingLoss', 'SoftmaxFocalLossMultiClass',
    'BCELossWithLogits', 'CrossEntropyLoss', 'BCELoss', 'BCELossWithLogitsAndIgnore', 'LabelSmoothingCrossEntropy', 'SoftTargetCrossEntropy',
]
