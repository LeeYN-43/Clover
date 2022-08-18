_base_ = [
    '../_base_/datasets_local/pretrain_webvid_cc3m.py',
    '../_base_/models/swin3d/swin3d_base_stride.py',
    '../_base_/default_runtime.py'
]
# model settings
num_classes = 56
weight_decay = 0.005
videos_per_gpu = 32
num_gpus = 8
machines = 8
num_frames = 8
base_lr = 5e-5 / (videos_per_gpu * num_gpus * machines)
multi_class = False
aux_info = ['token_ids', 'segment_ids', 'input_mask', 'mlm_label', 'v_token_mask']
save_root = '/home/lyn/'
load_pretrained_ckpts = save_root + 'swin_pretrained/swin_base_patch244_window877_kinetics400_22k_nobackbone.pth'
pretrained_textbackbone='bert-base-uncased'
resume_from = None
SyncBN = True
find_unused_parameters = True  
fp16 = dict(loss_scale='dynamic')
model = dict(
    type='CloverPretrain',
    freeze_stage=None,
    separate_test=True,
    use_Cmask=True,
    # freeze_stage=('backbone.patch_embed.',),   # freeze_stage=('text_backbone.', 'backbone.'),
    # freeze_except=('backbone.layers.3.',),
    backbone=dict(
        type='SwinTransformer3D',
        stride=(2, 4, 4),
        mask_token=True,
        pretrained2d=False,
        pretrained=load_pretrained_ckpts),
    freeze_text_backbone=None,
    text_vocab_size=30522,
    mm_backbone=dict(
        type='CrossModalTransformerFromPretrained', 
        use_text_cls=True,
        use_prompt=False,
        pretrained_model=pretrained_textbackbone,
        num_hidden_layers=3,
        img_in_size=1024,
        hidden_size=768, 
        num_frames=round(num_frames/2), 
        spacial_tokens=7*7,
        token_types=2,
        layer_norm_eps=1e-12,
        word_pos_start=False,
        ),
    text_backbone=dict(
        type='BertFromPretrained',
        num_hidden_layers=12,
    ),
    cls_head=None,    
    ssl_head=dict(
        type='NCEHeadForMM',
        visual_in_channels=1024,
        text_in_channels=768,
        img_hidden_dim=768*2,
        vts_embed_dim=768,
        ln=True,
        spatial_type='avg',
        text_agg_type='cls',
        dropout_ratio=0,),
    mlm_head=dict(
        type='MLMHead',
        hidden_size=768,
        vocab_size=30522,
    ),
    mlm_ssl_head=dict(
        V=dict(
            type='NCEHeadForVision',
            visual_in_channels=768,
            cross_in_channels=768,
            hidden_dim=768,
            ln=True,
            vts_embed_dim=768,
            dropout_ratio=0,
        ),
        T=dict(
            type='NCEHeadForText',
            cross_in_channels=768,
            vts_embed_dim=768,
            dropout_ratio=0.1,
        ),
    ),
    mlm_loss=dict(
        type="SoftmaxFocalLossMultiClass",
        gamma=2.0,
    ),
    loss_type=dict(
        type="CrossEntropyLoss",
        ),
    ssl_loss=dict(
        type="ExclusiveNCEwithRankingLoss",
        temperature=0.05,
        use_rank=True,
        use_rank_ttm=True,
        use_rank_trtm=False,
        margin_ttm=5.,
        margin_trtm=10.,
    ),
    symmetry_rank=True,
    train_cfg=dict(aux_info=aux_info))

    
data = dict(
    train_dataloader = dict(
        train_dataloader1=dict(
            videos_per_gpu=videos_per_gpu,
            workers_per_gpu=4,
            pin_memory=True,),
        train_dataloader2=dict(
            videos_per_gpu=videos_per_gpu,
            workers_per_gpu=4,
            pin_memory=True,),
        ),
    val_dataloader=dict(
        videos_per_gpu=videos_per_gpu,
        workers_per_gpu=4,
        ),
)


evaluation = dict(interval=1, metrics=['recall_for_video_text_retrieval'], gpu_collect=True, test_fn='recall_for_video_text_retrieval', save_best='Recall@all')
# optimizer
optimizer = dict(
    type='AdamW', base_lr=base_lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=weight_decay,
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                     'relative_position_bias_table': dict(decay_mult=0.),
                     }))
optimizer_config = dict(grad_clip=dict(max_norm=15))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1e-3, by_epoch=False,
                 warmup='linear', warmup_iters=4, warmup_ratio=0.001, warmup_by_epoch=True)
total_epochs = 40
checkpoint_config = dict(type='MYCheckpointHook', interval=1, ## del_local_ckpt=True,
                         save_root=save_root+'/Clover/work_dirs/')
