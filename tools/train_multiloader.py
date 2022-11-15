import argparse
import copy
import os
import os.path as osp
import time
import warnings
import torch

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, OptimizerHook, build_optimizer, get_dist_info, set_random_seed)
from mmaction.core.hooks.mmcv_Fp16OptimizerHook import Fp16OptimizerHook
from mmcv.utils import build_from_cfg, get_git_hash

from mmaction.core.hooks import ExpMomentumEMAHook, LinearMomentumEMAHook

from mmaction import __version__
from mmaction.models import build_model
from mmaction.utils import collect_env, register_module_hooks, get_root_logger
from mmaction.core.hooks import MyEvalHook, MyDistEvalHook
from mmaction.core.runner import MyEpochBasedMultiDatasetRunner as Runner
from mmaction.datasets import build_dataset, build_dataloader
from mmcv.runner.dist_utils import _init_dist_pytorch, _init_dist_slurm, _init_dist_mpi


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load from')
    parser.add_argument(
        '--validate',
        type=bool,
        default=False,
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        type=bool,
        default=True,
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
        # os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def _init_dist(launcher='pytorch', backend='nccl', **kwargs):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)
    rank, world_size = get_dist_info()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_settings = [dict(dataloader_setting,
                              **v) for k, v in cfg.data.train_dataloader.items()]

    data_loaders = [
        build_dataloader(ds, **dl_setting) for ds, dl_setting in zip(dataset, dataloader_settings)
    ]
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    if 'base_lr' in cfg.optimizer:
        base_lr = cfg.optimizer.pop('base_lr')  # If base_lr is set, lr is automatically adjusted according to the linear principle
        videos_per_gpu = cfg.get('videos_per_gpu', 1) # sum(cfg.data.train.get('batch_sizes', [1]))
        lr = base_lr * videos_per_gpu * world_size
        cfg.optimizer['lr'] = lr
        logger.info(f'According to the Linear Scaling Rule, '
                    f'set "lr=base_lr*videos_per_gpu*world_size={base_lr}*{videos_per_gpu}*{world_size}={lr}"')
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_cfg = copy.deepcopy(cfg.data.val)
        val_dataset = build_dataset(val_cfg, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)

        eval_hook = MyDistEvalHook(val_dataloader, **eval_cfg) if distributed \
            else MyEvalHook(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook)

    # ema hook from mmdetection, add by lyn
    if cfg.get('ema_hook', None):
        ema_hook_config = cfg.ema_hook
        assert isinstance(ema_hook_config, dict), \
            f'ema_hook expects dict type, but got {type(hook_cfg)}'
        ema_hook_type = ema_hook_config.pop('type')
        priority = ema_hook_config.pop('priority') if 'priority' in ema_hook_config else 49
        ema_hook = ExpMomentumEMAHook(**ema_hook_config) if ema_hook_type == 'ExpMomentumEMAHook' \
            else LinearMomentumEMAHook(**ema_hook_config)
        runner.register_hook(ema_hook, priority=priority)

    # user-defined hooks   
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # user-defined hooks   
    if cfg.get('mmcv_hooks', None):
        mmcv_hooks = cfg.mmcv_hooks
        assert isinstance(mmcv_hooks, list), \
            f'custom_hooks expect list type, but got {type(mmcv_hooks)}'
        for hook_cfg in cfg.mmcv_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, **runner_kwargs)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from    
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank, world_size = 0, 1
    else:
        distributed = True
        _init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if cfg.get('SyncBN'):
        logger.info('converting SyncBN...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    # build train datasets
    datasets = []
    train_cfg_dict = copy.deepcopy(cfg.data.train)
    for train_cfg in list(train_cfg_dict.values()):
        datasets.append(build_dataset(train_cfg, dict()))

    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        test=test_option,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
