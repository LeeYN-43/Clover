# model settings
_base_ = "swin_tiny.py"
model = dict(backbone=dict(pretrained=None,  # 'pretrained_models/swin_large_patch4_window7_224_22kto1k.pth',
                           depths=[2, 2, 18, 2],
                           embed_dim=192,
                           drop_path_rate=0.5,
                           num_heads=[6, 12, 24, 48]))
#cls_head = dict(in_channels=1536))
# weight_decay = 0.05
