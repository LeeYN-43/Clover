# model settings
_base_ = "swin_tiny.py"
model = dict(backbone=dict(pretrained=None,  # 'pretrained_models/swin_small_patch4_window7_224.pth',
                           depths=[2, 2, 18, 2],
                           drop_path_rate=0.3))
weight_decay = 0.05
