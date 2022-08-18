import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
from einops import rearrange


@HEADS.register_module()
class NCEHeadForMM(nn.Module):
    """contrastive head for InfoNce in mm pretrain.

    Args:
        in_channels (int): Number of channels in input feature.
        img_embed_dim (int): Number of img embedding dim
        text_embed_dim (int): Number of text embedding dim
        loss_type (dict): Config for building loss.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 visual_in_channels,
                 text_in_channels,
                 img_hidden_dim,
                 vts_embed_dim,
                 spatial_type='avg',
                 text_agg_type='avg',
                 ln=False,
                 text_bn=False,
                 dropout_ratio=0.1,
                 init_std=0.01,
                 **kwargs):
        super().__init__()
        self.vis_in_channels = visual_in_channels
        self.text_in_channels = text_in_channels
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.img_hidden_dim = img_hidden_dim
        self.vts_embed_dim = vts_embed_dim
        self.fp16_enabled = False
        self.ln = ln
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.img_projector = nn.Sequential(
                    nn.Linear(self.vis_in_channels, self.img_hidden_dim),  # 768, 768 * 2
                    nn.BatchNorm1d(self.img_hidden_dim) if not self.ln else nn.LayerNorm(self.img_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.img_hidden_dim, self.vts_embed_dim),   # 768 * 2, 768
                    nn.BatchNorm1d(self.vts_embed_dim) if not self.ln else nn.LayerNorm(self.vts_embed_dim)
        )
        if text_bn:
            self.text_projector = nn.Sequential(
                        nn.Linear(self.text_in_channels, self.text_in_channels),# 768, 768
                        nn.BatchNorm1d(self.text_in_channels),
                        nn.GELU(),
                        nn.Linear(self.text_in_channels, self.vts_embed_dim)   # 768, 768
            )
        else:
            self.text_projector = nn.Sequential(
                        nn.Linear(self.text_in_channels, self.text_in_channels),# 768, 768
                        nn.GELU(),
                        nn.Linear(self.text_in_channels, self.vts_embed_dim)   # 768, 768
            )
        self.init_weights()
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        self.text_agg_type = text_agg_type

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

    def forward(self, img, text, text_mask=None, token_ids=None):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        img = self.forward_vision(img)
        text = self.forward_text(text, text_mask, token_ids)
        return img, text

    def forward_vision(self, img):
        # [N, in_channels, num_segments, 7, 7]
        if self.avg_pool is not None:
            img = self.avg_pool(img)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            img = self.dropout(img)
        # [N, in_channels, 1, 1, 1]
        img = rearrange(img, "b d t h w -> b (t h w) d")
        # [N, in_channels]
        img = img.squeeze() if img.shape[0] != 1 else img.squeeze().unsqueeze(0)
        img = self.img_projector(img)
        # [N, img_embed_dim]
        return img
    
    def forward_text(self, text, text_mask=None, token_ids=None):
        # [N, seq, text_dim]
        if self.text_agg_type == 'avg':
            # remove cls and sep, text_mask is used to remove pad
            text_mask = torch.where(token_ids != 102, text_mask, torch.tensor(0).long().to(token_ids.device))
            text = text[:, 1:]
            text_mask = text_mask[:, 1:]
            text = text * text_mask.unsqueeze(-1)
            text = text.sum(1)
            text = text / text_mask.sum(1, keepdim=True)
        elif self.text_agg_type == 'cls':
            text = text[:, 0]
        elif self.text_agg_type == 'max':
            text_mask = torch.where(token_ids != 102, text_mask, torch.tensor(0).long().to(token_ids.device))
            text = text[:, 1:]
            text_mask = text_mask[:, 1:]
            text = text * text_mask.unsqueeze(-1)
            text = torch.max(text, dim=1)[0]
        
        text = self.text_projector(text)

        return text


@HEADS.register_module()
class NCEHeadForVision(nn.Module):
    """contrastive head for InfoNce in ctv pretrain.

    Args:
        in_channels (int): Number of channels in input feature.
        img_embed_dim (int): Number of img embedding dim
        text_embed_dim (int): Number of text embedding dim
        loss_type (dict): Config for building loss.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 cross_in_channels=768,
                 visual_in_channels=1024,
                 hidden_dim=768,
                 vts_embed_dim=768,
                 dropout_ratio=0.1,
                 ln=False,
                 init_std=0.01,
                 **kwargs):
        super().__init__()
        self.cross_in_channels = cross_in_channels
        self.visual_in_channels = visual_in_channels
        self.vts_embed_dim = vts_embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.ln = ln
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # visual project
        self.img_fc1 = nn.Linear(self.visual_in_channels, self.hidden_dim * 2)  # 768, 768 * 2
        self.img_bn1 = nn.BatchNorm1d(self.hidden_dim * 2) if not self.ln else nn.LayerNorm(self.hidden_dim * 2)
        self.img_act = nn.GELU()
        self.img_fc2 = nn.Linear(self.hidden_dim * 2, self.vts_embed_dim)   # 768 * 2, 768
        self.img_bn2 = nn.BatchNorm1d(self.vts_embed_dim) if not self.ln else nn.LayerNorm(self.vts_embed_dim)

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

    def forward(self, img):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [b, 4*7*7, in_channels]
        img = img.mean(dim=1)
        if self.dropout is not None:
            img = self.dropout(img)
        # [b, in_channels]
        img = self.img_fc1(img)
        img = self.img_bn1(img)
        img = self.img_act(img)
        img = self.img_fc2(img)
        img = self.img_bn2(img)
        # [b, img_embed_dim]

        return img


@HEADS.register_module()
class NCEHeadForText(nn.Module):
    """contrastive head for InfoNce in ctv pretrain.

    Args:
        in_channels (int): Number of channels in input feature.
        img_embed_dim (int): Number of img embedding dim
        text_embed_dim (int): Number of text embedding dim
        loss_type (dict): Config for building loss.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 cross_in_channels=768,
                 vts_embed_dim=768,
                 dropout_ratio=0.1,
                 text_bn=False,
                 **kwargs):
        super().__init__()
        self.cross_in_channels = cross_in_channels
        self.vts_embed_dim = vts_embed_dim
        self.dropout_ratio = dropout_ratio
        self.fp16_enabled = False
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        # mask word projector
        self.text_bn = text_bn
        self.fc1 = nn.Linear(self.cross_in_channels, self.cross_in_channels)# 768, 768
        self.bn = nn.BatchNorm1d(self.cross_in_channels) if self.text_bn else None
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.cross_in_channels, self.vts_embed_dim)   # 768, 768

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

    def forward(self, mask_word_feat):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # mask word project  m * d
        mask_word_feat = self.fc1(mask_word_feat)

        if self.text_bn:
            mask_word_feat = self.bn(mask_word_feat)

        mask_word_feat = self.act(mask_word_feat)

        if self.dropout is not None:
            mask_word_feat = self.dropout(mask_word_feat)

        mask_word_feat = self.fc2(mask_word_feat)

        return mask_word_feat
