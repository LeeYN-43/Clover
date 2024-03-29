import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss
from mmcv.runner import force_fp32

@LOSSES.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        self.fp16_enabled = False
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
    
    @force_fp32(apply_to=('cls_score'))
    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls


@LOSSES.register_module()
class LabelSmoothingCrossEntropy(BaseWeightedLoss):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, loss_weight=1.0, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__(loss_weight=loss_weight)

        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _forward(self, cls_score, label, **kwargs):
        logprobs = F.log_softmax(cls_score, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=label.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


@LOSSES.register_module()
class SoftTargetCrossEntropy(BaseWeightedLoss):

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, cls_score, label, **kwargs):
        loss = torch.sum(-label * F.log_softmax(cls_score, dim=-1), dim=-1)
        return loss.mean()


@LOSSES.register_module()
class BCELoss(BaseWeightedLoss):

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score, label, **kwargs):
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy(cls_score, label.type_as(cls_score), **kwargs)
        return loss_cls


@LOSSES.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls


@LOSSES.register_module()
class BCELossWithLogitsAndIgnore(BCELossWithLogits):
    def __init__(self, loss_weight=1.0, class_weight=None, ignore=-1):
        super().__init__(loss_weight=loss_weight, class_weight=class_weight)
        self.ignore = ignore

    def _forward(self, cls_score, label, **kwargs):
        kwargs['reduction'] = 'none'
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label, **kwargs)
        mask = label != self.ignore
        elt_sum = 1
        for dim in cls_score.shape:
            elt_sum *= dim
        rescale = elt_sum / float(torch.sum(mask) + 1e-6)
        loss_cls = torch.mean(loss_cls * mask) * rescale
        return loss_cls


@LOSSES.register_module()
class MultiTaskBCELossWithLogits(BCELossWithLogits):
    def __init__(self, loss_weight=1.0, class_weight=None, head_id=0, num_classes=1, ignore=-1):
        super().__init__(loss_weight=loss_weight, class_weight=class_weight)
        self.head_id = head_id
        self.num_classes = num_classes
        self.ignore = ignore

    def _forward(self, cls_score, label, **kwargs):
        kwargs['reduction'] = 'none'
        task_id = kwargs.pop('task_id')
        label = label[..., :self.num_classes]
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label, **kwargs)
        mask = task_id == self.head_id
        mask = mask.expand_as(label).type_as(label)
        mask_ignore = label != self.ignore
        mask *= mask_ignore
        elt_sum = 1
        for dim in cls_score.shape:
            elt_sum *= dim
        rescale = elt_sum / float(torch.sum(mask) + 1e-6)  # 1e-6防止mask全为false
        loss_cls = torch.mean(loss_cls * mask) * rescale
        return loss_cls