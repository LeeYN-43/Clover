# optimizer
optimizer = dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0.05, amsgrad=False)  # this lr is used for batch size as 1024
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0,
                 warmup='linear', warmup_iters=2, warmup_ratio=0.001, warmup_by_epoch=True)
total_epochs = 100
