from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

from .builder import BLENDINGS

__all__ = ['BaseMiniBatchBlending', 'MixupBlending', 'CutmixBlending']


class BaseMiniBatchBlending(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    @abstractmethod
    def do_blending(self, imgs, label, **kwargs):
        pass

    def __call__(self, imgs, label, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probablity distribution over classes) are float tensors
        with the shape of (B, 1, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label (torch.Tensor): Hard labels, integer tensor with the shape
                of (B, 1) and all elements are in range [0, num_classes).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blending images, float tensor with the
                same shape of the input imgs.
            mixed_label (torch.Tensor): Blended soft labels, float tensor with
                the shape of (B, 1, num_classes) and all elements are in range
                [0, 1].
        """
        one_hot_label = F.one_hot(label, num_classes=self.num_classes)

        mixed_imgs, mixed_label = self.do_blending(imgs, one_hot_label,
                                                   **kwargs)

        return mixed_imgs, mixed_label


@BLENDINGS.register_module()
class MixupBlending(BaseMiniBatchBlending):
    """Implementing Mixup in a mini-batch.

    This module is proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.
    Code Reference https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/mixup.py # noqa

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label


@BLENDINGS.register_module()
class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.

    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        return imgs, label


@BLENDINGS.register_module()
class MixupCutmixBlending(BaseMiniBatchBlending):
    """
    joint mixup and cutmix
    """

    def __init__(self, num_classes, mixup_alpha=.2, cutmix_alpha=.2, multi_class=True, prob=1.0, switch_prob=0.5,
                 label_smoothing=-1):
        super().__init__(num_classes=num_classes)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.multi_class = multi_class
        self.label_smoothing = label_smoothing
        self.prob = prob
        self.switch_prob = switch_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1,))[0]
        cy = torch.randint(h, (1,))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'
        batch_size = imgs.size(0)

        use_cutmix = False
        if np.random.rand() < self.prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
            if self.mixup_alpha < 0. and self.cutmix_alpha < 0.:
                raise ValueError(f"Expect one of mixup_alpha > 0., cutmix_alpha > 0. "
                                 f"But get mixup_alpha={self.mixup_alpha}, cutmix_alpha={self.cutmix_alpha}")

        if use_cutmix:
            # cutmix
            cutmix_rand_index = torch.randperm(batch_size)
            cutmix_lam = self.cutmix_beta.sample()

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), cutmix_lam)
            imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[cutmix_rand_index, ..., bby1:bby2,
                                                 bbx1:bbx2]
            cutmix_lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                              (imgs.size()[-1] * imgs.size()[-2]))

            label = cutmix_lam * label + (1 - cutmix_lam) * label[cutmix_rand_index, :]
        else:
            # mixup
            mixup_lam = self.mixup_beta.sample()
            mixup_rand_index = torch.randperm(batch_size)

            imgs = mixup_lam * imgs + (1 - mixup_lam) * imgs[mixup_rand_index, :]
            label = mixup_lam * label + (1 - mixup_lam) * label[mixup_rand_index, :]

        return imgs, label

    def __call__(self, imgs, label, **kwargs):
        if not self.multi_class:  # 对于multi_class的情况，label已经是one_hot形式
            label = F.one_hot(label, num_classes=self.num_classes)

        if self.label_smoothing > 0.:
            off_value = self.label_smoothing / self.num_classes
            label = label * (1. - self.label_smoothing) + off_value

        mixed_imgs, mixed_label = self.do_blending(imgs, label, **kwargs)
        return mixed_imgs, mixed_label
