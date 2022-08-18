import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertForPreTraining, BertConfig
from packaging import version
from ..builder import BACKBONES
from mmaction.core.hooks.fp16_utils import auto_fp16
from timm.models.layers import trunc_normal_


@BACKBONES.register_module()
class CrossModalTransformerFromPretrained(nn.Module):

    def __init__(self, pretrained_model='bert-base-uncased', img_in_size=768, hidden_size=768, num_frames=4, spacial_tokens=7*7, 
                        token_types=2, num_hidden_layers=12, layer_norm_eps=1e-12, word_pos_start=False, use_prompt=False,
                        use_text_cls=False, return_mask=False, **kwargs):
        '''
            pretrained_model: the pretrained model(bert-base)
            hidden_size: hidden_dimension
            num_frames: the time dimension of video token
        '''

        super().__init__()
        bert_config = BertConfig.from_pretrained(pretrained_model, layer_norm_eps=layer_norm_eps)
        bert_config.num_hidden_layers = num_hidden_layers
        bert = BertForPreTraining.from_pretrained(pretrained_model, config=bert_config)
        self.bert_embedding = bert.bert.embeddings
        self.bert_extended_mask = bert.get_extended_attention_mask
        self.bert_encoder = bert.bert.encoder
        self.use_prompt = use_prompt
        if not use_text_cls:
            self.all_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            trunc_normal_(self.all_cls_token, mean=0., std=.02)
            if self.use_prompt:
                self.prompt_token = nn.Parameter(torch.zeros(1, 4, hidden_size))
                trunc_normal_(self.prompt_token, mean=0., std=.02)

        else:
            self.all_cls_token = None
        self.vis_space_pos = nn.Parameter(0.02 * torch.randn(1, 1, spacial_tokens, hidden_size))
        self.vis_tempor_pos = nn.Parameter(0.02 * torch.randn(1, num_frames, 1, hidden_size))
        self.token_type_embeddings = nn.Embedding(token_types, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.word_pos_start = word_pos_start
        self.num_frames = num_frames
        self.spacial_tokens = spacial_tokens
        self.img_in_size = img_in_size
        self.hidden_size = hidden_size
        self.return_mask = return_mask
        if img_in_size != hidden_size:
            self.fc_in = nn.Linear(img_in_size, hidden_size)

        self.fp16_enabled = False
        # self.init_weights()
        # self.freeze_params(self.config.TRAIN.freeze_prefix)

    def init_weights(self):
        if self.with_project:
            if isinstance(self.projector, nn.Linear):
                nn.init.xavier_uniform_(self.projector.weight)
                if isinstance(self.projector, nn.Linear) and self.projector.bias is not None:
                    self.projector.bias.data.zero_()
    
    @auto_fp16(apply_to=('visual_token',))
    def forward(self, visual_token=None, text_input_ids=None, text_input_mask=None, text_input_embeds=None, **kwargs):
        """multimodal transformer forward function
            [cls] visual token (+ pos + temper) [cls] word tokens [sep]
        """
        if self.img_in_size != self.hidden_size:
            visual_token = self.fc_in(visual_token)
        B, T, S, D = visual_token.shape
        # past_key_values_length is where the word start
        # here we set to zero, use for word abs position embedding
        p_k_v_l = T * S + 1 if self.word_pos_start else 0
        if text_input_embeds is None:
            text_embeddings = self.bert_embedding(input_ids=text_input_ids, past_key_values_length=p_k_v_l)  
        else:
            text_embeddings = text_input_embeds
        if text_embeddings.shape[0] != B:
            # milnce  (b*n, seq, dim) -> (b, n*seq, dim)
            text_embeddings = text_embeddings.view(B, -1, text_embeddings.shape[-1])
            text_input_mask = text_input_mask.view(B, -1)

        text_token_type = torch.ones(text_embeddings.size()[:-1], dtype=torch.long, device=text_embeddings.device)
        text_token_type_embedding = self.token_type_embeddings(text_token_type)
        text_embeddings = text_embeddings + text_token_type_embedding

        # I think add one to represent all make sense, cause the frame num in train and test is not all the same
        visual_token = visual_token + self.vis_space_pos + self.vis_tempor_pos[:, :T, :, :]   # 有多少帧，加多少个
        visual_token = visual_token.contiguous().view(B, T * S, D) 
        # 加了个类别embedding，让模型区分输入的是图像还是文本token，all_cls没加
        visual_token_type = torch.zeros(visual_token.size()[:-1], dtype=torch.long, device=visual_token.device)
        visual_token_type_embedding = self.token_type_embeddings(visual_token_type)
        visual_token = visual_token + visual_token_type_embedding

        # additional layer norm from VIOLET
        visual_token = self.norm(visual_token)
        
        if self.use_prompt:
            visual_token = torch.cat([visual_token, self.prompt_token.expand(B, -1, -1), self.all_cls_token.expand(B, -1, -1)], dim=1)
            visual_input_mask = torch.ones(B, T * S + 5).long().to(visual_token.device)
        elif self.all_cls_token is not None:
            visual_token = torch.cat([visual_token, self.all_cls_token.expand(B, -1, -1)], dim=1)
            visual_input_mask = torch.ones(B, T * S + 1).long().to(visual_token.device)
        else:
            visual_input_mask = torch.ones(B, T * S).long().to(visual_token.device)

        multimodal_feat, multimodal_mask = torch.cat([visual_token, text_embeddings], dim=1), torch.cat([visual_input_mask, text_input_mask], dim=1)
        mask = self.bert_extended_mask(multimodal_mask, multimodal_mask.shape, multimodal_mask.device)
        out = self.bert_encoder(multimodal_feat, mask, output_attentions=True)
        if self.all_cls_token is not None and not self.use_prompt:
            v_seq_len = 1 + T * S
        elif self.use_prompt:
            v_seq_len = 5 + T * S
        else:
            v_seq_len =  T * S
        out['t_last_hidden_state'] = out['last_hidden_state'][:, v_seq_len :]
        out['v_last_hidden_state'] = out['last_hidden_state'][:, : T * S]
        if self.all_cls_token is not None:
            out['cls_last_hidden_state'] = out['last_hidden_state'][:, v_seq_len - 1 : v_seq_len]

        if self.return_mask:
            return out, multimodal_mask
        return out

    def forward_text(self, text_input_ids=None, text_input_mask=None, **kwargs):
        visual_seq_len = self.num_frames * self.spacial_tokens + 1 if self.word_pos_start else 0
        text_embeddings = self.bert_embedding(input_ids=text_input_ids, past_key_values_length=visual_seq_len)
        token_type_language = torch.ones(text_embeddings.size()[:-1], dtype=torch.long, device=text_embeddings.device)
        token_type_embedding = self.token_type_embeddings(token_type_language)
        text_embeddings += token_type_embedding

        # B, _, D = text_embeddings.shape
        # visual_padding_token = torch.zeros(B, visual_seq_len, D).to(text_input_ids.device)
        # visual_padding_token_mask = torch.zeros(B, visual_seq_len).long().to(text_input_ids.device)

        # multimodal_feat, multimodal_mask = torch.cat([visual_padding_token, text_embeddings], dim=1), torch.cat([visual_padding_token_mask, text_input_mask], dim=1)
        text_mask = self.bert_extended_mask(text_input_mask, text_input_mask.shape, text_input_mask.device)
        out = self.bert_encoder(text_embeddings, text_mask, output_attentions=True)
        out['last_hidden_state'] = out['last_hidden_state'][:, visual_seq_len:, :]
        return out
