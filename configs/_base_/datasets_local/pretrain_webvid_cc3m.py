# dataset settings
videos_per_gpu = 16
num_frames = 8
multi_class = False
multi_view_test = False
webvid_data_root = '/home/lyn/webvid/'
webvid_dataset_type = 'WebVidDataset'
ann_file_train_webvid = webvid_data_root + 'webvid_val.pkl'
train_webvid_key_prefixes = None
train_webvid_data_paths = webvid_data_root + 'videos/val_video/'
cc3m_dataset_type = 'CC3MDataset'
cc3m_data_root = '/home/BigDataset/CC3M/'
ann_file_train_cc3m = cc3m_data_root + 'cc3m_val.pkl'
train_cc3m_key_prefixes = None
train_cc3m_data_paths = cc3m_data_root + 'raw_img/'
val_dataset_type = 'MsrvttVideoDataset'
ann_file_val = '/home/lyn/MSRVTT_anno/test_Retri_1k_anno_new.pkl'
val_key_prefixes = None
val_data_paths = '/home/lyn/datassd/Video/MSRVTT/data/MSRVTT/videos/all/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)  
pretrained_texttokenizer='bert-base-uncased'
train_webvid_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_frames),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW_TSN'),
    dict(type='MaskingGenerator', window_size=7, num_masking_patches=10), 
    dict(
        type='BertTokenizer',
        vocab_file_path=None,
        pretrained_model=pretrained_texttokenizer,
        max_length=30,
        do_lower_case=True,
        do_mask=True,
        whole_word_mask=True,
        scene_graph_mask_obj_verb=True,
        mlm_probability=0.3,
        temporal_cat=False,
        skip_existing=False),
    dict(type='ToTensor', keys=['imgs']),
    dict(type='Collect', keys=['imgs', 'label', 'token_ids', 'segment_ids', 'input_mask', 'mlm_label', 'v_token_mask'], meta_keys=[]),
]
train_cc3m_pipeline = [
    dict(type='CLSLoadImageFromFile'),
    dict(type='CLSRandomResizedCrop', size=224, backend='cv2', interpolation='bicubic'),
    # dict(type='CLSRandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='CLSNormalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='MaskingGenerator', window_size=7, num_masking_patches=10),  
    dict(
        type='BertTokenizer',
        vocab_file_path=None,
        pretrained_model=pretrained_texttokenizer,
        max_length=30,
        do_lower_case=True,
        do_mask=True,
        whole_word_mask=True,
        scene_graph_mask_obj_verb=True,
        mlm_probability=0.3,
        temporal_cat=False,
        skip_existing=False),
    dict(type='ToTensor', keys=['imgs']),
    dict(type='Collect', keys=['imgs', 'label', 'token_ids', 'segment_ids', 'input_mask', 'mlm_label', 'v_token_mask'], meta_keys=[]),
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16 if multi_view_test else 1,
        frame_interval=1,
        num_clips=10 if multi_view_test else 32,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop' if multi_view_test else 'CenterCrop', crop_size=224),
    # dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW_TSN'),
    dict(
        type='BertTokenizer',
        vocab_file_path=None,
        pretrained_model=pretrained_texttokenizer,
        max_length=30, 
        do_lower_case=True,
        itm_test_for_retrieval=True,
        do_mask=False,
        temporal_cat=False,
        skip_existing=False),
    dict(type='ToTensor', keys=['imgs']),
    dict(type='Collect', keys=['imgs', 'token_ids', 'segment_ids', 'input_mask', 'index'],
         meta_keys=['filename', 'text']),
]
evaluation = dict(
    interval=1, metrics=['recall_for_video_text_retrieval'], gpu_collect=False)
data = dict(
    train=dict(
        train_set1=dict(
            type=webvid_dataset_type,
            ann_file=ann_file_train_webvid,
            pipeline=train_webvid_pipeline,
            start_index=0,
            data_prefix=train_webvid_data_paths,
        ),
        train_set2=dict(
            type=cc3m_dataset_type,
            ann_file=ann_file_train_cc3m,
            pipeline=train_cc3m_pipeline,
            start_index=0,
            data_prefix=train_cc3m_data_paths,
        ),
    ),
    val=dict(
        type=val_dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        start_index=0,
        data_prefix=val_data_paths,
    ),
)


