import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from torch.utils.data import Dataset
from mmaction.utils import hsave_pkl

from ..core import (mean_average_precision, mean_class_accuracy, classwise_accuracy,
                    mmit_mean_average_precision, top_k_accuracy, classwise_average_precision,
                    specify_precision_recall, specify_threshold, recall_at_precision)
from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False):
        super().__init__()

        self.ann_file = ann_file
        if not isinstance(self.ann_file, (list, tuple)):
            self.data_prefix = osp.realpath(
                data_prefix) if data_prefix is not None and osp.isdir(
                    data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length

        # assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        if not isinstance(self.ann_file, (list, tuple)):
            self.video_infos = self.load_annotations()  # this will lead to the specific load_annotation function, cause it's abstart function
        else:
            self.video_infos = []
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for k, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        if not self.multi_class:
            for item in self.video_infos:
                label = item['label']
                video_infos_by_class[label].append(item)
        else:
            for item in self.video_infos:
                label = item['label']
                if not isinstance(label, list):
                    label = np.where(label == 1.)[0].tolist()
                for l in label:
                    video_infos_by_class[l].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results. [(batch_size, num_classes), ...]
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self.video_infos), (  # modified  , len(self) -> len(self.video_infos)
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self.video_infos)}')  # comment out 

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        # allowed_metrics = [
        #     'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
        #     'mmit_mean_average_precision', 'classwise_average_precision', 'classwise_accuracy',
        #     'specify_precision_recall', 'specify_threshold'
        # ]
        #
        # for metric in metrics:
        #     if metric not in allowed_metrics:
        #         raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                # log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                #     log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                # log_msg = ''.join(log_msg)
                # print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                # log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                # print_log(log_msg, logger=logger)
                continue
            if metric == 'classwise_accuracy':
                clswise_acc = classwise_accuracy(results, gt_labels)
                eval_results['classwise_accuracy'] = clswise_acc

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                if isinstance(gt_labels[0], (list, int)):  # add    convert to onehot format
                    gt_labels = [
                        self.label2array(self.num_classes, label)
                        for label in gt_labels
                    ]
                mAP = eval(metric)(results, gt_labels)
                eval_results[metric] = mAP
                continue
            if metric in ['classwise_average_precision', ]:
                if isinstance(gt_labels[0], (list, int)):
                    gt_labels = [
                        self.label2array(self.num_classes, label)
                        for label in gt_labels
                    ]
                results = eval(metric)(results, gt_labels)  # add  
                results_filter = [x for x in results if not np.isnan(x)]
                if results_filter == []:
                    mAP = np.nan
                else:
                    mAP = np.mean(results_filter)
                eval_results['mAP'] = mAP  # modified  
                eval_results['classwise_average_precision'] = results  # modified  
                continue
            if metric in ['specify_precision_recall', ]:  # add  
                if isinstance(gt_labels[0], (list, int)):
                    gt_labels = [
                        self.label2array(self.num_classes, label)
                        for label in gt_labels
                    ]
                prec_spec = metric_options.get('prec_spec', 0.9)
                rec_spec = metric_options.get('rec_spec', None)
                mode, mode_value, ap, ar, results = eval(metric)(results, gt_labels, prec_spec, rec_spec)
                eval_results['mode'] = mode
                eval_results['mode_value'] = mode_value
                eval_results['average_precision'] = ap
                eval_results['average_recall'] = ar
                eval_results['precision_recall_threshold'] = results
                continue
            if metric in ['specify_threshold', ]:  # add  
                if isinstance(gt_labels[0], (list, int)):
                    gt_labels = [
                        self.label2array(self.num_classes, label)
                        for label in gt_labels
                    ]
                thr = metric_options.get('thr', 0.8)
                average = metric_options.get('average', None)
                ap, ar = eval(metric)(results, gt_labels, thr=thr, average=average)
                eval_results['threshold'] = thr
                if average is not None:
                    eval_results['average_precision'] = ap
                    eval_results['average_recall'] = ar
                else:
                    eval_results['precision'] = ap
                    eval_results['recall'] = ar
                continue
            if metric in ['Recall@90', 'Recall@95', 'Recall@99', 'Recall@100']:
                prec_map = {
                    'Recall@90': 0.9, 'Recall@95': 0.95, 'Recall@99': 0.99, 'Recall@100': 1.
                }
                prec = prec_map[metric]
                if isinstance(gt_labels[0], (list, int)):
                    gt_labels = [
                        self.label2array(self.num_classes, label)
                        for label in gt_labels
                    ]
                ar, ap, precision, recall, threshold = recall_at_precision(results, gt_labels, prec)
                eval_results[metric] = ar
                eval_results['[average_recall, average_precision]'] = [ar, ap]
                eval_results['[precision, recall, threshold]'] = [
                    [p, r, thr] for p, r, thr in zip(precision, recall, threshold)]
                # log_msg = [f'average_recall: {ar:.4f}, average_precision: {ap:.4f}']
                # for p, r, thr in zip(precision, recall, threshold):
                #     log_msg.append(f'precision: {p:.4f}\trecall: {r:.4f}\tthreshold: {thr:.4f}')
                # log_msg = '\n'.join(log_msg)
                # print_log(log_msg, logger=logger)
                continue
        return eval_results

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        assert out.endswith('.pkl')
        return hsave_pkl(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        # if self.sample_by_class:
        #     # Then, the idx is the class index
        #     samples = self.video_infos_by_class[idx]
        #     results = copy.deepcopy(np.random.choice(samples))
        # else:
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        # if self.sample_by_class:
        #     # Then, the idx is the class index
        #     samples = self.video_infos_by_class[idx]
        #     results = copy.deepcopy(np.random.choice(samples))
        # else:
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)
