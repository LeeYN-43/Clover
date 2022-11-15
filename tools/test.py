import argparse
import os
import os.path as osp
import warnings
import copy
import numpy as np
import json
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import dump, file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from mmaction.core.hooks.fp16_utils import wrap_fp16_model
from mmaction.datasets import build_dataset, build_dataloader
from mmaction.models import build_model
from mmaction.utils import register_module_hooks
from mmaction.core.hooks import (multi_gpu_test_retrieval_varied, multi_gpu_test_action_recognition,
                                 multi_gpu_test_retrieval, multi_gpu_test_itm_finetune)
from mmaction.utils.my_io import hlist_files

from mmcv.runner.dist_utils import _init_dist_pytorch


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def inference_pytorch(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if not cfg.model.get('type') == 'VisualTextS3DMILNCERecognizer3D':
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    outputs = {}
    print(distributed, args.checkpoint)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    if len(args.eval) == 1 and args.eval[0] == 'recall_for_video_text_retrieval':
        outputs[args.checkpoint] = multi_gpu_test_retrieval(model, data_loader, args.tmpdir,
                                args.gpu_collect)   
    elif len(args.eval) == 1 and args.eval[0] == 'recall_for_video_text_retrieval_varied':
        outputs[args.checkpoint] = multi_gpu_test_retrieval_varied(model, data_loader, args.tmpdir,
                                args.gpu_collect)       
    elif len(args.eval) == 1 and args.eval[0].startswith('video_qa'):
        outputs[args.checkpoint] = multi_gpu_test_itm_finetune(model, data_loader, args.tmpdir,
                                args.gpu_collect)
    elif len(args.eval) == 1 and args.eval[0] == 'zeroshot_action_recognition':
        outputs[args.checkpoint] = multi_gpu_test_action_recognition(model, data_loader, args.tmpdir,
                                args.gpu_collect) 

    
    return outputs

def inference_pytorch_multi_checkpoints(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models with multi checkpoints."""
    outputs = {}
    checkpoint_paths = hlist_files([args.checkpoint])
    for i in checkpoint_paths:
        print(i.split('/')[-1])
    print("test on {} checkpoints".format(len(checkpoint_paths)))
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.endswith('.pth'):
            continue
        if args.average_clips is not None:
            # You can set average_clips during testing, it will override the
            # original setting
            if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
                cfg.model.setdefault('test_cfg',
                                    dict(average_clips=args.average_clips))
            else:
                if cfg.model.get('test_cfg') is not None:
                    cfg.model.test_cfg.average_clips = args.average_clips
                else:
                    cfg.test_cfg.average_clips = args.average_clips

        # remove redundant pretrain steps for testing
        turn_off_pretrained(cfg.model)

        # build the model and load checkpoint
        model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

        if len(cfg.module_hooks) > 0:
            register_module_hooks(model, cfg.module_hooks)

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        load_checkpoint(model, checkpoint_path, map_location='cpu')

        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if len(args.eval) == 1 and args.eval[0] == 'recall_for_video_text_retrieval':
            outputs[checkpoint_path] = multi_gpu_test_retrieval(model, data_loader, args.tmpdir,
                                args.gpu_collect)
        elif len(args.eval) == 1 and args.eval[0] == 'recall_for_video_text_retrieval_varied':
            outputs[checkpoint_path] = multi_gpu_test_retrieval_varied(model, data_loader, args.tmpdir,
                                args.gpu_collect)  
        elif len(args.eval) == 1 and args.eval[0].startswith('video_qa'):
            outputs[args.checkpoint] = multi_gpu_test_itm_finetune(model, data_loader, args.tmpdir,
                                args.gpu_collect)

    return outputs





def main():
    args = parse_args()
    print(args.eval)

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank, world_size = 0, 1
    else:
        distributed = True
        _init_dist_pytorch(**cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.test.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.test.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    if args.checkpoint.endswith('.pth'):
        outputs = inference_pytorch(args, cfg, distributed, data_loader)
    else:
        outputs = inference_pytorch_multi_checkpoints(args, cfg, distributed, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        if output_config.get('out', None):
            out = output_config['out']
            print(f'\nwriting results to {out}')
            # dataset.dump_results(outputs, **output_config)
        if eval_config:
            print('results: \n')
            count = 0
            dump_dict = {}
            for key, output in outputs.items():
                eval_res = dataset.evaluate(output, **eval_config)
                epoch_num = key.split('/')[-1]
                print(epoch_num, eval_res)
                for name, val in eval_res.items():
                    if isinstance(val, (list, tuple)): 
                        if isinstance(val[0], (list, tuple)):
                            val = [[round(vv, 4) for vv in v] for v in val]
                        else:
                            val = [round(v, 4) for v in val]
                        print(f'{name}: {val}')
                    elif isinstance(val, str):
                        print(f'{name}: {val}')
                    elif isinstance(val, dict):
                        r1 = val['R1']
                        r5 = val['R5']
                        r10 = val['R10']
                        medR = val['MR']
                        keyname = key.split('/')[-2]
                        epoch_num = key.split('/')[-1]
                        if count == 0:
                            print(keyname)
                            count += 1
                        print(f'{epoch_num}: {name}: R@1: {r1:.2f} - R@5: {r5:.2f} - R@10: {r10:.2f} - Median R: {medR:.2f}') 
                        dump_dict[int(epoch_num.split('_')[-1].split('.')[0])] = {name: val}
                    else:
                        print(f'{name}: {val:.04f}')
            if output_config.get('out', None):
                out = output_config['out']
                with open(out, 'w') as f:
                    json.dump(dump_dict, f)



if __name__ == '__main__':
    main()
