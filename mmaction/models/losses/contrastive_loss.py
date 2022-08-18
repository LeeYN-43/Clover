import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from mmcv.runner import get_dist_info
from mmaction.core.hooks.fp16_utils import force_fp32
from mmaction.models.utils.gather_loss import GatherLoss, VariedShapeGatherLoss


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1)[:, None], b.norm(dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.matmul(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def cos_norm(a, eps=1e-8):
    if a is None:
        return a
    a_n = a.norm(dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


@LOSSES.register_module()
class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.07, cos_sim=False):
        super().__init__()
        self.t = temperature
        self.use_cos_similarity = cos_sim
        self.allgather = GatherLoss.apply
        self.rank, self.world_size = get_dist_info()
        if self.use_cos_similarity:
            print("use cosine similarity")
        self.fp16_enabled = False

    @force_fp32()
    def forward(self, video_embd=None, text_embd=None, sim_mat=None):
        if sim_mat is None:           
            video_embd = self.allgather(video_embd, self.rank, self.world_size)
            text_embd = self.allgather(text_embd, self.rank, self.world_size)

            # video_embd shape: B x D
            # text_embd  shape: B x D
            if self.use_cos_similarity:
                x = sim_matrix(video_embd, text_embd) / self.t
            else:
                video_embd = F.normalize(video_embd, dim=-1)
                text_embd = F.normalize(text_embd, dim=-1)
                x = torch.matmul(video_embd, text_embd.t()) / self.t
            "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        else:
            x = sim_mat
        
        i_logsm = F.log_softmax(x, dim=1)
        j_logsm = F.log_softmax(x.t(), dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j


@LOSSES.register_module()
class ExclusiveNCEwithRankingLoss(nn.Module):
    def __init__(self, temperature=0.05, use_rank=False, use_rank_ttm=True, use_rank_trtm=True, margin_ttm=5., margin_trtm=10.,):
        super().__init__()
        self.t = temperature
        self.allgather = VariedShapeGatherLoss.apply
        self.rank, self.world_size = get_dist_info()
        self.margin_ttm = margin_ttm
        self.margin_trtm = margin_trtm
        self.use_rank = use_rank
        self.use_rank_ttm = use_rank_ttm
        self.use_rank_trtm = use_rank_trtm
        if self.use_rank_ttm:
            self.margin_ranking_ttm = nn.MarginRankingLoss(self.margin_ttm)
        if self.use_rank_trtm:
            self.margin_ranking_trtm = nn.MarginRankingLoss(self.margin_trtm)
        self.fp16_enabled = False

    def compute_loss(self, sim_mat, reverse=True):
        i_logsm = F.log_softmax(sim_mat, dim=1)
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)
        if reverse:
            j_logsm = F.log_softmax(sim_mat.t(), dim=1)
            jdiag = torch.diag(j_logsm)
            loss_j = jdiag.sum() / len(jdiag)
        else:
            loss_j = 0
        return - loss_i - loss_j


    @force_fp32()
    def forward(self, video_embd=None, text_embd=None, text_mask_embd=None, text_recon_embd=None, **kwargs):
        #  all_gather from other gpu # 
        video_embd = self.allgather(video_embd, self.rank, self.world_size) if video_embd is not None else None
        text_embd = self.allgather(text_embd, self.rank, self.world_size) if text_embd is not None else None
        text_mask_embd = self.allgather(text_mask_embd, self.rank, self.world_size) if text_mask_embd is not None else None
        text_recon_embd = self.allgather(text_recon_embd, self.rank, self.world_size) if text_recon_embd is not None else None

        losses = {}

        # video_embd shape: B x D
        # text_all_embd  shape: 3B x D
        video_embd_norm = cos_norm(video_embd)
        text_embd_norm = cos_norm(text_embd)
        text_mask_embd_norm = cos_norm(text_mask_embd)
        text_recon_embd_norm = cos_norm(text_recon_embd)

        sim_vt = torch.matmul(video_embd_norm, text_embd_norm.t()) / self.t
        sim_vtm = torch.matmul(video_embd_norm, text_mask_embd_norm.t()) / self.t
        sim_vtr = torch.matmul(video_embd_norm, text_recon_embd_norm.t()) / self.t

        vt_diag = torch.diag(sim_vt)
        vtm_diag = torch.diag(sim_vtm)
        vtr_diag = torch.diag(sim_vtr)
        #  x:  B, B, 3
        ####  exclusive-Nce  V -> [T, T_mask, T_reconstruct]  ####

        #  v2t
        v2t_forvt = torch.cat([sim_vt, sim_vtm - (torch.diag_embed(vtm_diag + 10000.)), sim_vtr - (torch.diag_embed(vtr_diag + 10000.))], dim=1)  # B, 3B
        v2t_forvtm = torch.cat([sim_vt - (torch.diag_embed(vt_diag + 10000.)), sim_vtm, sim_vtr - (torch.diag_embed(vtr_diag + 10000.))], dim=1)  # B, 3B
        v2t_forvtr = torch.cat([sim_vt - (torch.diag_embed(vt_diag + 10000.)), sim_vtm - (torch.diag_embed(vtm_diag + 10000.)), sim_vtr], dim=1)  # B, 3B

        B = v2t_forvt.size()[0]

        vt_logsm = F.log_softmax(v2t_forvt, dim=1)[:, :B]
        vtm_logsm = F.log_softmax(v2t_forvtm, dim=1)[:, B:2*B]
        vtr_logsm = F.log_softmax(v2t_forvtr, dim=1)[:, 2*B:3*B]

        vtall_diag = torch.diag(vt_logsm) + torch.diag(vtm_logsm) + torch.diag(vtr_logsm)
        loss_v = - (vtall_diag.sum() / len(vtall_diag))
        

        # t2v
        t2v = torch.cat([sim_vt, sim_vtm, sim_vtr], dim=1).t()
        t2v_logsm = F.log_softmax(t2v, dim=1)  # 3B, B 
        t2v_logsm = t2v_logsm.view(-1, t2v.shape[1], t2v.shape[1])   # 3, B, B
        t2v_diag = t2v_logsm.diagonal(dim1=1, dim2=2)  # 3, B
        t2v_value = t2v_diag.mean(dim=1)               # 3
        loss_t = -torch.mean(t2v_value)
        losses['nce_loss'] = loss_v + loss_t


        #### rank loss #####
        if self.use_rank:
            y = torch.ones(vt_diag.size(), dtype=torch.long, device=vt_diag.device)
            if self.use_rank_ttm:
                loss_t_tm = self.margin_ranking_ttm(vt_diag, vtm_diag, y)
                losses['rank_t_tm_loss'] = loss_t_tm

        return losses



