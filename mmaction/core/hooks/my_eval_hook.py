import os.path as osp
import warnings
from math import inf
import time
import numpy as np
from typing import Sequence

import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

import mmcv
from mmcv.utils import is_seq_of
from mmcv.runner import Hook
from mmcv.runner import get_dist_info
from mmcv.engine import collect_results_gpu, collect_results_cpu
from mmaction.utils import hdelete, hexists


def single_gpu_test(model, data_loader):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    indices = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader.data_iter):  # modified  , replace "data_loader.dataset" with data_loader
        # index = data.pop('index').cpu().numpy().tolist()
        # indices.extend(index)
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

        if i >= (len(data_loader) - 1):
            break

    indices, unique_idx = np.unique(np.array(indices), return_index=True)
    arg_indices = np.argsort(indices, kind='mergesort')
    unique_idx = unique_idx[arg_indices]
    results = [results[idx] for idx in unique_idx]

    if len(results) != len(data_loader.video_infos):
        print('Warning: len(results) != len(data_loader.video_infos)')
        indices = indices[arg_indices].tolist()
        print(f'sorted indices: {indices}')
        for i, idx in enumerate(indices):
            if i != idx:
                print(f'i, idx: {i, idx}')
        data_loader.terminate()
        raise ValueError
    return results

def single_gpu_test_retrieval(model, data_loader):
    """Test model with a single gpu.
    --- Add by lyn ---
    This method tests model with a single gpu and displays test progress bar
    and calculate video-text retrieval scores. 

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        dict: The prediction results.
    """
    model.eval()
    results = {}
    all_video_embd = []
    all_text_embd = []
    indices = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader.data_iter):  # modified  , replace "data_loader.dataset" with data_loader
        index = data.pop('index').cpu().numpy().tolist()
        indices.extend(index)
        with torch.no_grad():
            video_embd, text_embd = model(return_loss=False, **data)
            text_embd = text_embd.cpu().numpy()
            if video_embd.shape[0] != text_embd.shape[0]:
                video_embd = video_embd.view(text_embd.shape[0], -1, text_embd.shape[1])
                video_embd = video_embd.mean(dim=1)
            video_embd = video_embd.cpu().numpy()
            all_video_embd.extend(video_embd)
            all_text_embd.extend(text_embd)

        batch_size = len(all_video_embd)
        # for _ in range(batch_size):
        prog_bar.update()

        if i >= (len(data_loader) - 1):
            break
    indices, unique_idx = np.unique(np.array(indices), return_index=True)
    arg_indices = np.argsort(indices, kind='mergesort')
    unique_idx = unique_idx[arg_indices]
    all_video_embd = [all_video_embd[idx] for idx in unique_idx]
    all_text_embd = [all_text_embd[idx] for idx in unique_idx]
    if len(all_video_embd) != len(data_loader.video_infos):
        print('Warning: len(results) != len(data_loader.video_infos)')
        indices = indices[arg_indices].tolist()
        print(f'sorted indices: {indices}')
        for i, idx in enumerate(indices):
            if i != idx:
                print(f'i, idx: {i, idx}')
        data_loader.terminate()
        raise ValueError
        
    results['video_embd'] = all_video_embd
    results['text_embd'] = all_text_embd
    return results    


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    indices = []
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))  # modified  , replace "data_loader.dataset" with data_loader
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader.data_iter):
        index = data.pop('index').cpu().numpy().tolist()
        indices.extend(index)
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)
        if rank == 0:
            # batch_size = len(result)
            # batch_size_all = batch_size * world_size
            # if batch_size_all + prog_bar.completed > len(data_loader):
            #     batch_size_all = len(data_loader) - prog_bar.completed
            # for _ in range(batch_size_all):
            prog_bar.update()
        if i >= (len(data_loader) - 1):
            break

    # collect results from all ranks
    if world_size // torch.cuda.device_count() > 1:
        gpu_collect = True

    if gpu_collect:
        results = collect_results_gpu(results, size=None)  # 对于用ArnoldDataset从HDFS读取，len(data_loader)是不对的,改为len(data_loader.video_infos)
    else:
        results = collect_results_cpu(results, size=None, tmpdir=tmpdir)

    indices = collect_results_gpu(indices, size=None)
    if rank == 0:
        indices, unique_idx = np.unique(np.array(indices), return_index=True)
        arg_indices = np.argsort(indices, kind='mergesort')
        unique_idx = unique_idx[arg_indices]
        results = [results[idx] for idx in unique_idx]

        if len(results) != len(data_loader.video_infos):
            print('Warning: len(results) != len(data_loader.video_infos)')
            indices = indices[arg_indices].tolist()
            print(f'sorted indices: {indices}')
            for i, idx in enumerate(indices):
                if i != idx:
                    print(f'i, idx: {i, idx}')
            data_loader.terminate()
            raise ValueError
    return results


def multi_gpu_test_retrieval(model, data_loader, tmpdir=None, gpu_collect=False, separate=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = {}
    all_video_embd = []
    all_text_embd = []
    indices = []
    metas = []
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))  # modified  , replace "data_loader.dataset" with data_loader
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        index = data.pop('index').cpu().numpy().tolist()
        indices.extend(index)
        if 'img_metas' in data:
            imgs_metas = data.pop('img_metas').data[0]         
            metas.extend(imgs_metas)

        with torch.no_grad():
            video_embd, text_embd = model(return_loss=False, **data)
            if video_embd.shape[0] > text_embd.shape[0]:
                video_embd = video_embd.view(text_embd.shape[0], -1, text_embd.shape[1])
                video_embd = video_embd.mean(dim=1)
            elif video_embd.shape[0] < text_embd.shape[0]:
                # for msrvtt mc
                text_embd = text_embd.view(video_embd.shape[0], -1, text_embd.shape[1])

            text_embd = text_embd.cpu().numpy()
            video_embd = video_embd.cpu().numpy()
        all_video_embd.extend(video_embd)
        all_text_embd.extend(text_embd)
        if rank == 0:
            prog_bar.update()
        if i >= (len(data_loader) - 1):
            break

    # collect results from all ranks
    if world_size // torch.cuda.device_count() > 1:
        gpu_collect = True
    s_t = time.time()
    if gpu_collect:
        all_video_embd = collect_results_gpu(all_video_embd, size=None)  # 对于用ArnoldDataset从HDFS读取，len(data_loader)是不对的,改为len(data_loader.video_infos)
        all_text_embd = collect_results_gpu(all_text_embd, size=None)  # 对于用ArnoldDataset从HDFS读取，len(data_loader)是不对的,改为len(data_loader.video_infos)
    else:
        all_video_embd = collect_results_cpu(all_video_embd, size=None, tmpdir=tmpdir)  
        all_text_embd = collect_results_cpu(all_text_embd, size=None, tmpdir=tmpdir)  

    indices = collect_results_gpu(indices, size=None)
    if len(metas) != 0:
        metas = collect_results_gpu(metas, size=None)

    if rank == 0:
        print('\n')
        print("collect_time: ", time.time() - s_t)
        indices, unique_idx = np.unique(np.array(indices), return_index=True)
        arg_indices = np.argsort(indices, kind='mergesort')
        unique_idx = unique_idx[arg_indices]
        all_video_embd = [all_video_embd[idx] for idx in unique_idx]
        all_text_embd = [all_text_embd[idx] for idx in unique_idx]
        if len(metas) != 0:
            metas = [metas[idx] for idx in unique_idx]
        if len(all_video_embd) != len(data_loader.dataset.video_infos):
            print('Warning: len(results) != len(data_loader.video_infos)')
            indices = indices[arg_indices].tolist()
            print(f'sorted indices: {indices}')
            for i, idx in enumerate(indices):
                if i != idx:
                    print(f'i, idx: {i, idx}')
            data_loader.terminate()
            raise ValueError

    results['video_embd'] = all_video_embd
    results['text_embd'] = all_text_embd
    results['metas'] = metas
    return results


def multi_gpu_test_retrieval_varied(model, data_loader, tmpdir=None, gpu_collect=False, separate=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = {}
    all_video_embd = []
    all_text_embd = []
    indices = []
    metas = []
    # texts = []
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))  # modified  , replace "data_loader.dataset" with data_loader
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        index = data.pop('index').cpu().numpy().tolist()
        indices.extend(index)
        if 'img_metas' in data:
            imgs_metas = data.pop('img_metas').data[0]
            text_len = len(imgs_metas[0]['text'])
            repeat_vname = [[imgs_metas[0]['filename'] for i in range(text_len)]]
            metas.extend(repeat_vname)
        # text = data['img_metas'].data
        # texts.extend(text[0])
        # data['separate'] = separate
        with torch.no_grad():
            video_embd, text_embd = model(return_loss=False, **data)
            if video_embd.ndim == 1:
                video_embd = video_embd.unsqueeze(0)
            if text_embd.ndim == 2:
                text_embd = text_embd.unsqueeze(0)
            if video_embd.ndim > 1 and video_embd.shape[0] > text_embd.shape[0]:
                video_embd = video_embd.view(text_embd.shape[0], -1, text_embd.shape[1])
                video_embd = video_embd.mean(dim=1)

            text_embd = text_embd.cpu().numpy()
            video_embd = video_embd.cpu().numpy()
        all_video_embd.extend(video_embd)
        all_text_embd.extend(text_embd)
        if rank == 0:
            prog_bar.update()
        if i >= (len(data_loader) - 1):
            break

    # collect results from all ranks
    if world_size // torch.cuda.device_count() > 1:
        gpu_collect = True
    s_t = time.time()
    if gpu_collect:
        all_video_embd = collect_results_gpu(all_video_embd, size=None)  # 对于用ArnoldDataset从HDFS读取，len(data_loader)是不对的,改为len(data_loader.video_infos)
        all_text_embd = collect_results_gpu(all_text_embd, size=None)  # 对于用ArnoldDataset从HDFS读取，len(data_loader)是不对的,改为len(data_loader.video_infos)
    else:
        all_video_embd = collect_results_cpu(all_video_embd, size=None, tmpdir=tmpdir)  
        all_text_embd = collect_results_cpu(all_text_embd, size=None, tmpdir=tmpdir)  

    indices = collect_results_gpu(indices, size=None)
    if len(metas) != 0:
        metas = collect_results_gpu(metas, size=None)
    # texts = collect_results_cpu(texts, size=None, tmpdir=tmpdir)
    if rank == 0:
        print('\n')
        print("collect_time: ", time.time() - s_t)
        indices, unique_idx = np.unique(np.array(indices), return_index=True)
        arg_indices = np.argsort(indices, kind='mergesort')
        unique_idx = unique_idx[arg_indices]
        all_video_embd = [all_video_embd[idx] for idx in unique_idx]
        all_text_embd = [all_text_embd[idx] for idx in unique_idx]
        if len(metas) != 0:
            metas = [metas[idx] for idx in unique_idx]
        # texts = [texts[idx] for idx in unique_idx]
        if len(all_video_embd) != len(data_loader.dataset.video_infos):
            print('Warning: len(results) != len(data_loader.video_infos)')
            indices = indices[arg_indices].tolist()
            print(f'sorted indices: {indices}')
            for i, idx in enumerate(indices):
                if i != idx:
                    print(f'i, idx: {i, idx}')
            data_loader.terminate()
            raise ValueError

    results['video_embd'] = all_video_embd
    results['text_embd'] = all_text_embd
    results['metas'] = metas
    # results['text'] = texts
    return results



def multi_gpu_test_itm_finetune(model, data_loader, tmpdir=None, gpu_collect=False, separate=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    anses = []
    indices = []
    metas = []
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))  # modified  , replace "data_loader.dataset" with data_loader
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        index = data.pop('index').cpu().numpy().tolist()
        indices.extend(index)
        if 'img_metas' in data:
            imgs_metas = data.pop('img_metas').data[0]         
            metas.extend(imgs_metas)
        with torch.no_grad():
            result_all = model(return_loss=False, **data)
            result = result_all['result']
            if 'ans' in result_all:
                ans = result_all['ans']
            else:
                ans = data.pop('label')
        results.extend(result.cpu())
        if ans is not None:
            anses.extend(ans.cpu())
        if rank == 0:
            prog_bar.update()
        if i >= (len(data_loader) - 1):
            break

    # collect results from all ranks
    if world_size // torch.cuda.device_count() > 1:
        gpu_collect = True
    if gpu_collect:
        results = collect_results_gpu(results, size=None)  # 对于用ArnoldDataset从HDFS读取，len(data_loader)是不对的,改为len(data_loader.video_infos)
        anses = collect_results_gpu(anses, size=None)
    else:
        results = collect_results_cpu(results, size=None, tmpdir=tmpdir)
        anses = collect_results_cpu(anses, size=None, tmpdir=tmpdir)

    indices = collect_results_gpu(indices, size=None)
    if len(metas) != 0:
        metas = collect_results_gpu(metas, size=None)
    # texts = collect_results_cpu(texts, size=None, tmpdir=tmpdir)
    if rank == 0:
        indices, unique_idx = np.unique(np.array(indices), return_index=True)
        arg_indices = np.argsort(indices, kind='mergesort')
        unique_idx = unique_idx[arg_indices]
        results = [results[idx] for idx in unique_idx]
        anses = [anses[idx] for idx in unique_idx]
        if len(metas) != 0:
            metas = [metas[idx] for idx in unique_idx]
        if len(results) != len(data_loader.dataset.video_infos):
            print('Warning: len(results) != len(data_loader.video_infos)')
            indices = indices[arg_indices].tolist()
            print(f'sorted indices: {indices}')
            for i, idx in enumerate(indices):
                if i != idx:
                    print(f'i, idx: {i, idx}')
            data_loader.terminate()
            raise ValueError
    if metas is None or len(metas) == 0:
        return (results, anses)
    else:
        return (results, anses, metas)


class MyEvalHook(Hook):
    """Non-Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be saved in ``runner.meta['hook_msgs']`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resume checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
             ``OrderedDict`` result will be used. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        test_fn (callable, optional): test a model with samples from a
            dataloader, and return the test results. If ``None``, the default
            test function ``mmcv.engine.single_gpu_test`` will be used.
            (default: ``None``)
        greater_keys (List[str] | None, optional): Metric keys that will be
            inferred by 'greater' comparison rule rule. If ``None``,
            _default_greater_keys will be used. (default: ``None``)
        less_keys (List[str] | None, optional): Metric keys that will be
            inferred by 'less' comparison rule. If ``None``, _default_less_keys
            will be used. (default: ``None``)
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.

    Notes:
        If new arguments are added for EvalHook, tools/test.py,
        tools/eval_metric.py may be affected.
    """

    # Since the key for determine greater or less is related to the downstream
    # tasks, downstream repos may need to overwrite the following inner
    # variable accordingly.

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best='auto',
                 rule=None,
                 test_fn=None,
                 greater_keys=['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
                               'mAcc', 'aAcc', 'Recall@', 'accuracy'],
                 less_keys=['loss'],
                 **eval_kwargs):
        # if not isinstance(dataloader, DataLoader):  # Comment out  
        #     raise TypeError(f'dataloader must be a pytorch DataLoader, '
        #                     f'but got {type(dataloader)}')

        self.best_score_info = ''
        self.existing_score_info = 'existing score info: '

        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')

        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch

        assert isinstance(save_best, str) or save_best is None, \
            '""save_best"" should be a str or None ' \
            f'rather than {type(save_best)}'
        self.save_best = save_best
        self.eval_kwargs = eval_kwargs
        self.initial_flag = True

        if test_fn is None:
            self.test_fn = single_gpu_test
        else:
            self.test_fn = test_fn

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys, )
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys, )
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys

        if self.save_best is not None:
            self.best_ckpt_path = None
            self._init_rule(rule, self.save_best)
            if self.key_indicator != 'auto':
                cur_type = 'epoch' if self.by_epoch else 'iter'
                self.existing_score_info += f'[{self.key_indicator}, {cur_type}]: ['

    def _init_rule(self, rule, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Here is the rule to determine which rule is used for key indicator
        when the rule is not specific (note that the key indicator matching
        is case-insensitive):
        1. If the key indicator is in ``self.greater_keys``, the rule will be
           specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule will be
           specified as 'less'.
        3. Or if the key indicator is equal to the substring in any one item
           in ``self.greater_keys``, the rule will be specified as 'greater'.
        4. Or if the key indicator is equal to the substring in any one item
           in ``self.less_keys``, the rule will be specified as 'less'.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None:
            if key_indicator != 'auto':
                # `_lc` here means we use the lower case of keys for
                # case-insensitive matching
                key_indicator_lc = key_indicator.lower()
                greater_keys = [key.lower() for key in self.greater_keys]
                less_keys = [key.lower() for key in self.less_keys]

                if key_indicator_lc in greater_keys:
                    rule = 'greater'
                elif key_indicator_lc in less_keys:
                    rule = 'less'
                elif any(key in key_indicator_lc for key in greater_keys):
                    rule = 'greater'
                elif any(key in key_indicator_lc for key in less_keys):
                    rule = 'less'
                else:
                    raise ValueError(f'Cannot infer the rule for key '
                                     f'{key_indicator}, thus a specific rule '
                                     f'must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def before_run(self, runner):
        if self.save_best is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating an empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())
            self.best_ckpt_path = runner.meta['hook_msgs'].get(
                'best_ckpt', None)

    def before_train_iter(self, runner):
        """Evaluate the model only at the start of training by iteration."""
        if self.by_epoch or not self.initial_flag:
            return
        if self.start is not None and runner.iter >= self.start:
            self.after_train_iter(runner)
        self.initial_flag = False

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        if not (self.by_epoch and self.initial_flag):
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_flag = False

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch:
            self._do_evaluate(runner)

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch:
            self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        results = self.test_fn(runner.model, self.dataloader)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        eval_res = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, eval_res)
            runner.logger.info(self.best_score_info)
            runner.logger.info(self.existing_score_info + ']')

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True

    def _save_ckpt(self, runner, eval_res):
        """Save the best checkpoint.

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if hasattr(runner, 'hdfs_work_dir'):
            work_dir = runner.hdfs_work_dir
        else:
            work_dir = runner.work_dir
        basename = osp.basename(runner.work_dir)

        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        if self.key_indicator == 'auto':
            # infer from eval_results
            self._init_rule(self.rule, list(eval_res.keys())[0])
            self.existing_score_info += f'[{self.key_indicator}, {cur_type}]: ['
        key_score = eval_res[self.key_indicator]
        self.existing_score_info += f'[{key_score:.4f}, {cur_time}], '

        best_score = runner.meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score

            if self.best_ckpt_path and hexists(self.best_ckpt_path):
                hdelete(self.best_ckpt_path)  # 兼容hdfs

            best_ckpt_name = f'{basename}_best_{self.key_indicator}_{current}.pth'
            self.best_ckpt_path = osp.join(work_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path

            runner.save_checkpoint(
                work_dir, best_ckpt_name, create_symlink=False)
            best_score_info = f'Now best checkpoint is saved as {best_ckpt_name}\n' \
                              f'Best {self.key_indicator} is {best_score:0.4f} at {cur_time} {cur_type}\n' \
                              f'Evaluate results at {cur_time} {cur_type}:\n'
            for key, value in eval_res.items():
                if not isinstance(value, Sequence):
                    if isinstance(value, (int, float)):
                        best_score_info += f'{key}: {value:.4f}\n'
                    else:
                        best_score_info += f'{key}: {value}\n'
                elif isinstance(value, Sequence) and isinstance(value[0], Sequence):
                    try:
                        res_value = '['
                        for v in value:
                            res_value += '[' + ', '.join([f'{vv:.4f}' for vv in v]) + '], '
                        res_value += ']'
                    except:
                        res_value = value
                    best_score_info += f'{key}: {res_value}\n'
                elif isinstance(value, Sequence):
                    try:
                        res_value = '[' + ', '.join([f'{v:.4f}' for v in value]) + ']'
                    except:
                        res_value = value
                    best_score_info += f'{key}: {res_value}\n'
                else:
                    best_score_info += f'{key}: {value}\n'

            self.best_score_info = best_score_info

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(  
            results, logger=runner.logger, **self.eval_kwargs)   

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            return eval_res

        return None


class MyDistEvalHook(MyEvalHook):
    """Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be saved in ``runner.meta['hook_msgs']`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resume checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
             ``OrderedDict`` result will be used. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        test_fn (callable, optional): test a model with samples from a
            dataloader in a multi-gpu manner, and return the test results. If
            ``None``, the default test function ``mmcv.engine.multi_gpu_test``
            will be used. (default: ``None``)
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        broadcast_bn_buffer (bool): Whether to broadcast the
            buffer(running_mean and running_var) of rank 0 to other rank
            before evaluation. Default: True.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best='auto',
                 rule=None,
                 test_fn=None,
                 greater_keys=['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
                               'mAcc', 'aAcc', 'Recall@', 'accuracy'],
                 less_keys=['loss'],
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,
                 **eval_kwargs):
        if test_fn is None:
            test_fn = multi_gpu_test
        elif test_fn == 'recall_for_video_text_retrieval':
            test_fn = multi_gpu_test_retrieval
        elif test_fn == 'use_itm_head_fn':
            test_fn = multi_gpu_test_itm_finetune

        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            test_fn=test_fn,
            greater_keys=greater_keys,
            less_keys=less_keys,
            **eval_kwargs)
        
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)
        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            data_loader=self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            runner.logger.info(str(key_score))
            if self.save_best:
                self._save_ckpt(runner, key_score)
                runner.logger.info(self.best_score_info)
                runner.logger.info(self.existing_score_info + ']')


