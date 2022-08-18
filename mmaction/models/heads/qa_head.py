import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS

@HEADS.register_module()
class QA_MC_head(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio=0.1):
        super().__init__()
        self.mc_vqa_classifier = nn.Sequential(
                nn.Dropout(dropout_ratio),
                nn.Linear(hidden_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for n, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        itm_score = self.mc_vqa_classifier(x)
        return itm_score


@HEADS.register_module()
class QA_OE_Head(nn.Module):
    """Video qa head for OpenEnded(OE).

    Args:
        hidden_dim
        dropout_ratio
        num_labels
    """

    def __init__(self,
                 hidden_dim=768,
                 dropout_ratio=0.5,
                 num_labels=None,
                 **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.vqa_classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, self.num_labels),
        )
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for n, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, cls_feature):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        itm_score = self.vqa_classifier(cls_feature)
        return itm_score

