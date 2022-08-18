# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        pretrained=None,  # 'pretrained_models/swin_tiny_patch4_window7_224.pth',
        pretrained2d=True,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg=dict(average_clips='prob'))
weight_decay = 0.02
