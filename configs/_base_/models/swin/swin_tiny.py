# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
        ape=False, patch_norm=True,
        pretrained='pretrained_models/swin_tiny_patch4_window7_224.pth',  # 根据实际pretrain模型调整
        use_checkpoint=False),
    cls_head=dict(
        type='FakeHead',
        num_classes=1000,
        multi_class=False,
        label_smooth_eps=0.),
    test_cfg=None)
weight_decay = 0.05
