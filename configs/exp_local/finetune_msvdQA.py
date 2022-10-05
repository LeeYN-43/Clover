_base_ = [
    '../_base_/datasets_local/msvd_QA.py',
    '../_base_/models/swin3d/swin3d_base.py',
    '../_base_/default_runtime.py'
]
# model settings
weight_decay = 0.01
videos_per_gpu = 16
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
SyncBN = True
find_unused_parameters = False   
fp16 = dict(loss_scale='dynamic')
finetune_task = "video_qa"
model = dict(
    type='CloverFinetune',
    freeze_stage=None,
    separate_test=False,
    # freeze_stage=('backbone.patch_embed.',),   # freeze_stage=('text_backbone.', 'backbone.'),
    # freeze_except=('backbone.layers.3.',),
    backbone=dict(
        type='SwinTransformer3D',
        pretrained2d=False,
        pretrained=load_pretrained_ckpts),
   freeze_text_backbone=False,
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
    task=finetune_task,
    ssl_head=None,
    itm_head=None,
    answer_cls=True,
    qa_head=dict(
        type='QA_OE_Head',
        hidden_dim=768,
        dropout_ratio=0.5,
        num_labels=1000,
    ),
    loss_type=dict(
        type="CrossEntropyLoss",
        ),
    train_cfg=dict(aux_info=aux_info))

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
    
evaluation = dict(interval=1, metrics=['video_qa_oe'], gpu_collect=True, test_fn='use_itm_head_fn')
# optimizer
optimizer = dict(
    type='AdamW', base_lr=base_lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=weight_decay,
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
                     'qa_head': dict(lr_mult=10)
                     }))
optimizer_config = dict(grad_clip=dict(max_norm=50))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0, by_epoch=False,
                 warmup='linear', warmup_iters=4, warmup_ratio=0.0001, warmup_by_epoch=True)
total_epochs = 40
checkpoint_config = dict(type='MYCheckpointHook', interval=-1, # del_local_ckpt=True,
                         save_root=save_root+'/Clover/work_dirs/')
