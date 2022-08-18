import torch
from torch import nn
import torch.nn.functional as F
from mmaction.core.hooks.fp16_utils import auto_fp16
from ..builder import RECOGNIZERS, build_backbone, build_head, build_loss
from .base import BaseRecognizer
import random
from einops import rearrange


@RECOGNIZERS.register_module()
class CloverFinetune(BaseRecognizer):
    def __init__(self,
                 mm_backbone,
                 text_backbone=None,
                 freeze_text_backbone=None,
                 loss_type=None,
                 task=None,
                 ssl_head=None,
                 itm_head=None,
                 answer_mask=False,
                 answer_cls=False,
                 qa_head=None,
                 vision_q_head=None,
                 from_scratch=False,
                 text_vocab_size=30522,
                 separate_test=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.multimodal_backbone = build_backbone(mm_backbone)
        self.text_backbone = build_backbone(text_backbone)
        self.text_vocab_size = text_vocab_size
        self.from_scratch = from_scratch
        self.separate_test = separate_test
        self.task = task
        if self.task == 'retrieval':
            self.ssl_head = build_head(ssl_head)
            self.mlm_ssl_T_head = build_head(vision_q_head) if vision_q_head is not None else None
            self.loss_func = build_loss(loss_type)
        elif self.task == 'video_qa' or self.task == "FIB":
            self.answer_mask = answer_mask
            self.answer_cls = answer_cls
            self.itm_head = build_head(itm_head) if itm_head is not None else None
            self.mlm_ssl_T_head = build_head(vision_q_head) if vision_q_head is not None else None
            self.qa_head = build_head(qa_head) if qa_head is not None else None
            self.loss_func = build_loss(loss_type)
            self.loss_type = loss_type['type']
        elif self.task == 'video_qa_ret':
            self.ssl_head = build_head(ssl_head)
            self.mlm_ssl_T_head = build_head(vision_q_head)
            self.loss_func = build_loss(loss_type)
        else:
            raise NotImplementedError(f"must have head to do downstream finetuning")


    @auto_fp16(apply_to='imgs')
    def extract_visual_feat(self, imgs, token_ids=None, segment_ids=None, input_mask=None):
        if (hasattr(self.backbone, 'features')
                and self.backbone_from == 'torchvision'):
            visual_emb = self.backbone.features(imgs)
        elif self.backbone_from == 'timm':
            visual_emb = self.backbone.forward_features(imgs)
        else:
            visual_emb = self.backbone(imgs)
        return visual_emb

    def forward_train(self, imgs, label, token_ids=None, segment_ids=None, input_mask=None, ans_ids=None, ans_mask=None, **kwargs):
        """Defines the computation performed at every call when training."""
        # (batch_size, num_clips*num_crops, channel, num_segments, h, w) -> (batch_size*num_clips*num_crops, channel, num_segments, h, w)
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) 
        if self.from_scratch:
            imgs = imgs / 255.0
        B_text = token_ids.shape[0]
        # text reshape:  (batch_size, num_candidates, seq_length) -> (batch_size * num_candidates, seq_length)
        token_ids = token_ids.reshape((-1, ) + token_ids.shape[2:])
        segment_ids = segment_ids.reshape((-1, ) + segment_ids.shape[2:])
        input_mask = input_mask.reshape((-1, ) + input_mask.shape[2:])
        losses = dict()
        visual_token = self.extract_visual_feat(imgs) # b, d, T, h, w
        B, D, T, H, W = visual_token.shape
        if B_text != B:
            visual_token = visual_token.view(B_text, -1, D, T, H, W)
            visual_token = visual_token.mean(dim=1)
 
        # text feature #
        text_out_with_mask = self.text_backbone(token_ids, input_mask)
        text_out_last_hidden_state = text_out_with_mask['last_hidden_state']

        #  contrastive type finetuning retrieval #
        if self.task == 'retrieval':
            # text_only_out = self.text_backbone(token_ids, input_mask)
            visual_emb, text_emb = self.ssl_head(visual_token, text_out_last_hidden_state, input_mask, token_ids)
            nce_loss = self.loss_func(visual_emb, text_emb)
            losses['retrieval_nce_loss'] = nce_loss  
        elif self.task == 'video_qa' or self.task == 'FIB':
            B, D, T, H, W = visual_token.shape
            visual_token = visual_token.view(B, D, T, -1).permute(0, 2, 3, 1)
            if hasattr(self.qa_head, 'num_labels'):
                num_choices = self.qa_head.num_labels
                visual_token_all = visual_token
            else:
                num_choices = int(token_ids.shape[0] / B)
                visual_token_all = visual_token.unsqueeze(1).expand(-1, num_choices, -1, -1, -1).flatten(0,1)

            output = self.multimodal_backbone(visual_token=visual_token_all, text_input_mask=input_mask, text_input_embeds=text_out_last_hidden_state)
            
            if self.answer_mask:
                mask_idx = torch.where(token_ids == 103)
                itm_output = output['t_last_hidden_state'][mask_idx]
            elif self.answer_cls:
                if 'cls_last_hidden_state' in output:
                    itm_output = output['cls_last_hidden_state'].squeeze()
                else:
                    itm_output = output['t_last_hidden_state'][:, 0]
                if self.itm_head is not None:
                    itm_output = self.itm_head(itm_output)
                if self.mlm_ssl_T_head is not None:
                    itm_output = self.mlm_ssl_T_head(itm_output)
            else:
                all_cls_emb = output['last_hidden_state'][:, 0]
                itm_output = self.itm_head(all_cls_emb)
            
            if self.qa_head is not None:
                final_output = self.qa_head(itm_output).view(-1, num_choices)
                final_label = label
            else:
                final_output = itm_output[:, 1]
                final_label = label


            qa_loss = self.loss_func(final_output, final_label.view(-1))
            losses['qa_loss'] = qa_loss

        elif self.task == 'video_qa_ret':
            B, D, T, H, W = visual_token.shape
            visual_token = visual_token.view(B, D, T, -1).permute(0, 2, 3, 1)
            ans_ids = ans_ids.reshape((-1, ) + ans_ids.shape[2:])
            ans_mask = ans_mask.reshape((-1, ) + ans_mask.shape[2:])
            output = self.multimodal_backbone(visual_token=visual_token, text_input_mask=input_mask, text_input_embeds=text_out_last_hidden_state)
            v_q_out = output['t_last_hidden_state'][:, 0] 
            ans_out = self.text_backbone(ans_ids, ans_mask)['last_hidden_state']
            v_q_emb = self.mlm_ssl_T_head(v_q_out)
            ans_emb = self.ssl_head.forward_text(ans_out)
            nce_loss = self.loss_func(v_q_emb, ans_emb)
            losses['video_qa_mc_ret_loss'] = nce_loss


        return losses


    def forward_test(self, imgs, token_ids=None, segment_ids=None, input_mask=None, ans_ids=None, ans_mask=None, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        if self.from_scratch:
            imgs = imgs / 255.0
        visual_token = self.extract_visual_feat(imgs) # b, d, T, h, w
        B, D, T, H, W = visual_token.shape
        B_text = token_ids.shape[0]
        if B_text != B:
            visual_token = visual_token.view(B_text, -1, D, T, H, W)
            visual_token = visual_token.mean(dim=1)
            B = B_text
        token_ids = token_ids.reshape((-1, ) + token_ids.shape[2:])
        segment_ids = segment_ids.reshape((-1, ) + segment_ids.shape[2:])
        input_mask = input_mask.reshape((-1, ) + input_mask.shape[2:])

        # text feature #
        text_out_with_mask = self.text_backbone(token_ids, input_mask)
        text_out_last_hidden_state = text_out_with_mask['last_hidden_state']

        # only use the uni-modal transformer for retrieval 
        if self.separate_test:
            visual_emb, text_emb = self.ssl_head(visual_token, text_out_last_hidden_state, input_mask, token_ids)
            return visual_emb, text_emb


        if self.task == 'video_qa' or self.task == 'FIB':
            B, D, T, H, W = visual_token.shape
            visual_token = visual_token.view(B, D, T, -1).permute(0, 2, 3, 1)
            if hasattr(self.qa_head, 'num_labels'):
                num_choices = self.qa_head.num_labels
                visual_token_all = visual_token
            else:
                num_choices = int(token_ids.shape[0] / B)
                visual_token_all = visual_token.unsqueeze(1).expand(-1, num_choices, -1, -1, -1).flatten(0,1)

            output = self.multimodal_backbone(visual_token=visual_token_all, text_input_mask=input_mask, text_input_embeds=text_out_last_hidden_state)
            
            if self.answer_mask:    
                mask_idx = torch.where(token_ids == 103)
                itm_output = output['t_last_hidden_state'][mask_idx]
            elif self.answer_cls:
                if 'cls_last_hidden_state' in output:
                    itm_output = output['cls_last_hidden_state'].squeeze()
                else:
                    itm_output = output['t_last_hidden_state'][:, 0]
                if self.itm_head is not None:
                    itm_output = self.itm_head(itm_output)
                if self.mlm_ssl_T_head is not None:
                    itm_output = self.mlm_ssl_T_head(itm_output)
            else:
                all_cls_emb = output['last_hidden_state'][:, 0]
                itm_output = self.itm_head(all_cls_emb)

            if self.qa_head is not None:
                qa_output = self.qa_head(itm_output).view(-1, num_choices)
            else:
                qa_output = torch.softmax(itm_output, dim=-1)[:, 1]
                qa_output = qa_output.view(-1, num_choices)
            
            itm_output_all = {}
            itm_output_all['result'] = qa_output.to(torch.float32)
        elif self.task == 'video_qa_ret':
            B, D, T, H, W = visual_token.shape
            visual_token = visual_token.view(B, D, T, -1).permute(0, 2, 3, 1)
            ans_ids = ans_ids.reshape((-1, ) + ans_ids.shape[2:])
            ans_mask = ans_mask.reshape((-1, ) + ans_mask.shape[2:])
            output = self.multimodal_backbone(visual_token=visual_token, text_input_mask=input_mask, text_input_embeds=text_out_last_hidden_state)
            v_q_out = output['t_last_hidden_state'][:, 0] 
            ans_out = self.text_backbone(ans_ids, ans_mask)['last_hidden_state']
            v_q_emb = self.mlm_ssl_T_head(v_q_out)
            ans_emb = self.ssl_head.forward_text(ans_out)
            return v_q_emb.float(), ans_emb.float()        
        
        else:
            raise NotImplementedError("not implement the finetune test method")
            
        return itm_output_all    

    def forward_gradcam(self, imgs, token_ids=None, segment_ids=None, input_mask=None):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self.forward_test(imgs, token_ids, segment_ids, input_mask)

