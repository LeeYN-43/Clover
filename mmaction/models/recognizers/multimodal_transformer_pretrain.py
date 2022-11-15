import torch
from torch import nn
import torch.nn.functional as F
from mmaction.core.hooks.fp16_utils import auto_fp16
from ..builder import RECOGNIZERS, build_backbone, build_head, build_loss
from .base import BaseRecognizer
import random
from einops import rearrange


@RECOGNIZERS.register_module()
class CloverPretrain(BaseRecognizer):
    def __init__(self,
                 mm_backbone,
                 text_backbone=None,
                 freeze_text_backbone=None,
                 freeze_dvae_backbone=None,
                 loss_type=None,
                 ssl_loss=None, 
                 ssl_head=None,
                 mlm_head=None,
                 mlm_loss=None,
                 mlm_ssl_head=None,
                 symmetry_rank=False,
                 separate_test=False,
                 from_scratch=False,
                 use_Cmask=True,
                 text_vocab_size=30522,
                 **kwargs):
        super().__init__(**kwargs)

        self.multimodal_backbone = build_backbone(mm_backbone)
        self.text_backbone = build_backbone(text_backbone)
        self.text_vocab_size = text_vocab_size
        self.loss_func = build_loss(loss_type)
        self.use_Cmask = use_Cmask

        self.mlm_head = build_head(mlm_head) if mlm_head is not None else None
        if mlm_ssl_head is not None:
            self.mlm_ssl_V_head = build_head(mlm_ssl_head['V']) if mlm_ssl_head.get("V") else None
            self.mlm_ssl_T_head = build_head(mlm_ssl_head['T']) if mlm_ssl_head.get("T") else None
        else:
            self.mlm_ssl_V_head = None
            self.mlm_ssl_T_head = None

        self.mlm_loss_func = build_loss(mlm_loss) if mlm_loss is not None else None
        self.symmetry_rank = symmetry_rank
 
        self.from_scratch = from_scratch
        self.separate_test = separate_test
        if ssl_head is not None:
            self.ssl_head_name = ssl_head['type']
            self.ssl_head = build_head(ssl_head)
            self.ssl_loss = build_loss(ssl_loss)

        self.fp16_enabled = False

        if from_scratch:
            print("from scratch")
        if freeze_dvae_backbone is not None:
            self._freeze(freeze_stage=freeze_dvae_backbone, freeze_except=[])
        if freeze_text_backbone is not None:
            self._freeze(freeze_stage=freeze_text_backbone, freeze_except=[])
            
    @auto_fp16(apply_to=('imgs'))
    def extract_visual_feat(self, imgs, mask=None):
        if (hasattr(self.backbone, 'features')
                and self.backbone_from == 'torchvision'):
            visual_emb = self.backbone.features(imgs)
        elif self.backbone_from == 'timm':
            visual_emb = self.backbone.forward_features(imgs)
        else:
            visual_emb = self.backbone(imgs, mask)
        return visual_emb
    
    @auto_fp16(apply_to=('imgs', 'dvae_imgs'))
    def forward_train(self, imgs, label, token_ids=None, segment_ids=None, input_mask=None, 
                      mlm_label=None, dvae_imgs=None, v_token_mask=None, hog_features=None, img_metas=None, **kwargs):
        """Defines the computation performed at every call when training."""            
        # (batch_size, num_clips*num_crops, channel, num_segments, h, w) -> (batch_size*num_clips*num_crops, channel, num_segments, h, w)
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) 
        if self.from_scratch:
            imgs = imgs / 255.0
        # text reshape:  (batch_size, num_candidates, seq_length) -> (batch_size * num_candidates, seq_length)
        token_ids = token_ids.reshape((-1, ) + token_ids.shape[2:])
        text_input_mask = input_mask.reshape((-1, ) + input_mask.shape[2:])
        if mlm_label is not None:
            mlm_label = mlm_label.reshape((-1, ) + mlm_label.shape[2:])


        visual_token = self.extract_visual_feat(imgs) # b, d, T, h, w

        B, D, T, H, W = visual_token.shape
        losses = dict()
        # --------------  nce loss ------------------- #
        if hasattr(self, 'ssl_head'):
            input_ssl_ids = torch.where(mlm_label == -100, token_ids.clone(), mlm_label.clone())
            input_ssl_mask = text_input_mask.clone()
            text_only_out = self.text_backbone(input_ssl_ids, input_ssl_mask)
            #  ------------   complete T -------------- #
            text_out_no_mask = text_only_out['last_hidden_state']
            visual_emb, text_emb = self.ssl_head(visual_token, text_out_no_mask, input_ssl_mask, input_ssl_ids)


        #  ------------ complete V ---------------- #
        visual_token = visual_token.view(B, D, T, -1).permute(0, 2, 3, 1)


        # ---------- foward mask text input ---------- # 
        text_out_with_mask = self.text_backbone(token_ids, text_input_mask)
        text_out_last_hidden_state = text_out_with_mask['last_hidden_state']

        # ---------- forward mask v input ------------ #
        visual_token_with_mask, v_mask = self.extract_visual_feat(imgs.clone(), v_token_mask) # b, d, T, h, w
        visual_token_mask = visual_token_with_mask.view(B, D, T, -1).permute(0, 2, 3, 1)
        
        v_fusion_output = self.multimodal_backbone(visual_token=visual_token_mask, text_input_mask=text_input_mask.clone(), text_input_embeds=text_out_no_mask.clone())
        
        t_fusion_output = self.multimodal_backbone(visual_token=visual_token, text_input_mask=text_input_mask, text_input_embeds=text_out_last_hidden_state)
        # for mlm #
        t_last_hidden_state = t_fusion_output['t_last_hidden_state']





        # ------------ MLM loss ------------ #

        if mlm_label is not None and self.mlm_head is not None:
            # we use mask text for MLM
            # because we doubt there will be miss interaction between wrong img-text pair 
            # and the model not learn good relationship between vision and language
            # --------  forward masked text ----------- #
            mlm_prediction_score = self.mlm_head(t_last_hidden_state)
            
            if self.mlm_loss_func is not None:
                mlm_label_idx = torch.where(mlm_label.view(-1) != -100)
                mlm_prediction_mask_score = mlm_prediction_score.view(-1, self.text_vocab_size)[mlm_label_idx[0], :]
                mlm_label_mask = mlm_label.view(-1)[mlm_label_idx]
                mlm_loss = self.mlm_loss_func(mlm_prediction_mask_score, mlm_label_mask)
            else:
                mlm_loss = self.loss_func(mlm_prediction_score.view(-1, self.text_vocab_size), mlm_label.view(-1))
            losses['mlm_loss'] = mlm_loss


        #  -------  Tri-modal alignment with mask sample and ranking  --------- #
        if self.mlm_ssl_V_head is not None:
            mlm_visual_feat = v_fusion_output['t_last_hidden_state'][:, 0]
            mask_visual_recon_emb = self.mlm_ssl_V_head(mlm_visual_feat)
            mask_word_emb = self.ssl_head.forward_text(text_out_last_hidden_state) if self.use_Cmask else None
            loss_cvt_rank = self.ssl_loss(visual_emb, text_emb, mask_word_emb, mask_visual_recon_emb)
            losses.update(loss_cvt_rank)


        if self.symmetry_rank:
            mlm_word_feat = t_last_hidden_state[:, 0]
            mask_word_recon_emb = self.mlm_ssl_T_head(mlm_word_feat)

            mask_visual_emb = self.ssl_head.forward_vision(visual_token_with_mask) if self.use_Cmask else None
      
            loss_ctv_rank = self.ssl_loss(text_emb, visual_emb, mask_visual_emb, mask_word_recon_emb)
            loss_ctv_rank['v_nce_loss'] = loss_ctv_rank.pop('nce_loss')
            
            if self.ssl_loss.use_rank:
                loss_ctv_rank['rank_v_vm_loss'] = loss_ctv_rank.pop('rank_t_tm_loss')

            

            losses.update(loss_ctv_rank)



        return losses

    def forward_dummy(self, imgs, label, token_ids=None, segment_ids=None, input_mask=None, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        visual_emb, text_emb = self.extract_feat(imgs)

        outs = self.loss_fun(visual_emb, text_emb)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_test(self, imgs, token_ids=None, segment_ids=None, input_mask=None, **kwargs):
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
        # only use the multimodal transformer for  
        if self.separate_test:
            # text_only_out = self.multimodal_backbone.forward_text(token_ids, input_mask)
            # visual_emb, text_emb = self.ssl_head(visual_token, text_only_out['last_hidden_state'], input_mask, token_ids)
            text_only_out = self.text_backbone(token_ids, input_mask)
            text_out_last_hidden_state = text_only_out['last_hidden_state']
            visual_emb, text_emb = self.ssl_head(visual_token, text_out_last_hidden_state, input_mask, token_ids)

            return visual_emb, text_emb
        
        visual_token = visual_token.view(B, D, T, -1).permute(0, 2, 3, 1)
        output = self.multimodal_backbone(visual_token, token_ids, segment_ids, input_mask)
        all_cls_emb = output['last_hidden_state'][:, 0]
        itm_output = self.itm_head(all_cls_emb)
        itm_output = torch.softmax(itm_output, dim=-1)
        return itm_output    

    def forward_gradcam(self, imgs, token_ids=None, input_mask=None):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self.forward_test(imgs, token_ids, input_mask)