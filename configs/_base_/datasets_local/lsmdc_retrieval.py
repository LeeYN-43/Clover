# dataset settings
multi_view_test = False
data_root = '/home/lyn/lsmdc/'
dataset_type = 'MsrvttVideoDataset'
ann_file_train = data_root + 'lsmdc_retrieval_train.pkl'
ann_file_test = data_root + 'lsmdc_retrieval_test.pkl'
train_key_prefixes = None
train_data_paths = data_root + 'videos/'
test_key_prefixes = None
test_data_paths = data_root + 'videos/'
pretrained_texttokenizer='bert-base-uncased'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)   
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW_TSN'),
    dict(
        type='BertTokenizer',
        vocab_file_path=None,
        pretrained_model=pretrained_texttokenizer,
        max_length=40,
        do_lower_case=True,
        do_mask=False,
        temporal_cat=False,
        skip_existing=False),
    dict(type='ToTensor', keys=['imgs']),
    dict(type='Collect', keys=['imgs', 'label', 'token_ids', 'segment_ids', 'input_mask'], meta_keys=[]),
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=10 if multi_view_test else 1,
        frame_interval=1,
        num_clips=10 if multi_view_test else 4*8,
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
        max_length=40,
        do_lower_case=True,
        itm_test_for_retrieval=False,
        do_mask=False,
        temporal_cat=False,
        skip_existing=False),
    dict(type='ToTensor', keys=['imgs']),
    dict(type='Collect', keys=['imgs', 'token_ids', 'segment_ids', 'input_mask', 'index'],
         meta_keys=['filename', 'text']),
]
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        start_index=0,
        data_prefix=train_data_paths,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        start_index=0,
        data_prefix=test_data_paths,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        start_index=0,
        data_prefix=test_data_paths,
        )
    ) 
      