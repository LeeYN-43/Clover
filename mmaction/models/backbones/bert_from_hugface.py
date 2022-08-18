import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from ..builder import BACKBONES

@BACKBONES.register_module()
class BertFromPretrained(nn.Module):

    def __init__(self, pretrained_model='bert-base-uncased', layer_norm_eps=1e-12, num_hidden_layers=12, **kwargs):
        super().__init__()
        # 文本
        bert_config = BertConfig.from_pretrained(pretrained_model, layer_norm_eps=layer_norm_eps)
        bert_config.num_hidden_layers = num_hidden_layers
        self.bert = BertModel.from_pretrained(pretrained_model, config=bert_config)
        # self.init_weights()
        # self.freeze_params(self.config.TRAIN.freeze_prefix)

    def init_weights(self):
        if self.with_project:
            if isinstance(self.projector, nn.Linear):
                nn.init.xavier_uniform_(self.projector.weight)
                if isinstance(self.projector, nn.Linear) and self.projector.bias is not None:
                    self.projector.bias.data.zero_()

    def forward(self, token_ids=None, input_mask=None, **kwargs):
        """
        这里就是用一下，相加是之后再说
        """
        tout = self.bert(input_ids=token_ids, attention_mask=input_mask)

        return tout


