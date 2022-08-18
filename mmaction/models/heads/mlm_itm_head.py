import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.core.hooks.fp16_utils import force_fp32, auto_fp16

from ..builder import HEADS
from transformers import BertForMaskedLM
from einops import rearrange

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.fp16_enabled = False
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        # self.transform = BertPredictionHeadTransform(hidden_size)
        # # The output weights are the same as the input embeddings, but there is
        # # an output-only bias for each token.
        # self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        # # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.transform = bert.cls.predictions.transform
        self.decoder = bert.cls.predictions.decoder
        self.fp16_enabled = False
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

@HEADS.register_module()
class MLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size)
        self.fp16_enabled = False
    
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@HEADS.register_module()
class ITMHead(nn.Module):
    """contrastive head for MILNCE.

    Args:
        hidden_dim
        dropout_ratio
    """

    def __init__(self,
                 hidden_dim=768,
                 **kwargs):
        super().__init__()
        self.itm_projector = nn.Sequential(*[nn.Dropout(p=0.1),
                                    nn.Linear(hidden_dim, hidden_dim), 
                                    nn.Tanh(), 
                                    nn.Linear(hidden_dim, 2)])
        self.init_weights()
        self.fp16_enabled = False
    
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
                
    @auto_fp16()
    def forward(self, cls_feature):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        itm_score = self.itm_projector(cls_feature)
        # 多此一举，F.cross_entropy 集成了softmax
        # itm_score = torch.softmax(itm_score, dim=-1)
        return itm_score
