_base_ = [
    '../_base_/datasets_local/didemo_retrieval.py',
    '../_base_/models/swin3d/swin3d_base.py',
    '../_base_/default_runtime.py'
]
# model settings
weight_decay = 0.01
videos_per_gpu = 32
num_gpus = 8
machines = 1
num_frames = 8
base_lr = 1.2e-5 / (videos_per_gpu * num_gpus * machines)
multi_class = False
aux_info = ['token_ids', 'segment_ids', 'input_mask']
save_root = '/home/lyn/'
load_pretrained_ckpts = None
pretrained_textbackbone='bert-base-uncased'
resume_from = None
SyncBN = False
find_unused_parameters = True   
# fp16 = dict(loss_scale='dynamic')
finetune_task =  "retrieval"
model = dict(
    type='CloverFinetune',
    freeze_stage=None,
    separate_test=True,
    # freeze_stage=('backbone.patch_embed.',),   # freeze_stage=('text_backbone.', 'backbone.'),
    # freeze_except=('backbone.layers.3.',),
    backbone=dict(
        type='SwinTransformer3D',
        stride=(2, 4, 4),
        pretrained2d=False,
        pretrained=load_pretrained_ckpts),
    freeze_text_backbone=None,
    text_vocab_size=30522,
    mm_backbone=dict(
        type='CrossModalTransformerFromPretrained',
        use_text_cls=True,
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
    task='retrieval',
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
    itm_head=None,
    loss_type=dict(
        type="NormSoftmaxLoss",
        cos_sim=True,
        temperature=0.05,
        ),
    train_cfg=dict(aux_info=aux_info),
    test_cfg=dict(feature_extraction=False,))
data = dict(
    train_dataloader=dict(
        videos_per_gpu=videos_per_gpu,
        workers_per_gpu=4,
        ),
    val_dataloader=dict(
        videos_per_gpu=videos_per_gpu,
        workers_per_gpu=4,
        ),
    test_dataloader=dict(
        videos_per_gpu=videos_per_gpu,
        workers_per_gpu=4,
        ),
    )
evaluation = dict(interval=1, metrics=['recall_for_video_text_retrieval'], gpu_collect=True, test_fn='recall_for_video_text_retrieval')
# optimizer
optimizer = dict(
    type='AdamW', base_lr=base_lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=weight_decay,
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                     'relative_position_bias_table': dict(decay_mult=0.),
                     }))
optimizer_config = dict(grad_clip=dict(max_norm=5))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0, by_epoch=True,
                 warmup='linear', warmup_iters=5, warmup_ratio=0.001, warmup_by_epoch=True)
total_epochs = 50
checkpoint_config = dict(type='MYCheckpointHook', interval=-1, # del_local_ckpt=True,
                         save_root=save_root+'Clover/work_dirs')
