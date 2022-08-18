# model settings
_base_ = "swin_tiny.py"
model = dict(backbone=dict(pretrained=None,  # 'pretrained_models/swin_base_patch4_window7_224_22kto1k.pth',
                           depths=[2, 2, 18, 2],
                           embed_dim=128,
                           drop_path_rate=0.5,
                           num_heads=[4, 8, 16, 32]))
# cls_head = dict(in_channels=1024)
# weight_decay = 0.05
