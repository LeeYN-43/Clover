import os.path as osp
import random
import numpy as np
import warnings
import torch
import random as rnd
from mmaction.utils import hload_pkl
import copy
from .base import BaseDataset
from .builder import DATASETS
from mmaction.core.evaluation.accuracy import recall_for_video_text_retrieval_varied, recall_for_video_text_retrieval, acc_for_msrvtt_mc
 

@DATASETS.register_module()
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, min_video_num=-1, **kwargs):
        self.min_video_num = min_video_num
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label))
        while len(video_infos) < self.min_video_num:
            left_num = min(self.min_video_num - len(video_infos), len(video_infos))
            video_infos.extend(random.sample(video_infos, left_num))
        return video_infos


@DATASETS.register_module()
class PKLVideoDataset(VideoDataset):
    """
    annotation is in [{},{},...{}] format
    Each dict saves the annotation information of a video:
    'filename': video_id
    'text': corresponding text
    """

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        data = hload_pkl(self.ann_file)

        video_infos = []
        for video_info in data:
            filename = video_info['filename']
            if self.data_prefix is not None:
                filename = osp.join(self.data_prefix, filename)
            video_info['filename'] = filename
            label = video_info['label']
            if self.multi_class and isinstance(label, np.ndarray):
                video_info['label'] = label.astype(np.float32)

            video_infos.append(video_info)

        while len(video_infos) < self.min_video_num:
            left_num = min(self.min_video_num - len(video_infos), len(video_infos))
            video_infos.extend(random.sample(video_infos, left_num))
        return video_infos



@DATASETS.register_module()
class MsrvttVideoDataset(PKLVideoDataset):

    def __init__(self, is_mc=False, is_qa=False, is_ret=False, **kwargs):
        self.is_mc = is_mc
        self.is_qa = is_qa
        self.is_ret = is_ret
        super().__init__(**kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        ann_info = hload_pkl(self.ann_file)

        video_infos = []
        for i, video_info in enumerate(ann_info):
            frame_dir = video_info['filename']
            filename = osp.join(self.data_prefix, video_info['filename']+'.mp4') 
            video_info['filename'] = filename
            video_info['frame_dir'] = frame_dir
            video_info['index'] = i
            video_info['label'] = -1 if 'answer_idx' not in video_info else video_info['answer_idx']
      
            if isinstance(video_info['text'], str):
                video_info['text'] = [video_info['text']] 
            else:
                if not self.is_mc and not self.is_qa:
                    video_info['text'] = [rnd.choice(video_info['text'])]
                elif self.is_mc:
                    video_info['clip_text_candidate'] = [0, 1, 2, 3, 4]
                elif self.is_ret:
                    video_info['clip_text_candidate'] = list(range(len(video_info['text'])))
            video_infos.append(video_info) 
        del ann_info

        return video_infos
            
    def evaluate(self, results, metrics='recall_for_video_text_retrieval', metric_options=None, logger=None, normalize=False, **deprecated_kwargs):
        '''Evaluate the dataset with
            Retrieval / video QA / classification / video caption ?
            TODO: 后面三个
        Args:
            results (dict): Testing results of the dataset.
            metrics (str | list[str]): Metrics to be evaluated.
                Default value is `recall_for_video_text_retrieval`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        '''

        if deprecated_kwargs != {}:
            warnings.warn(f'Unrecognized parameters: {deprecated_kwargs}')

        if isinstance(metrics, str):
            metrics = [metrics]
        else:
            metrics = metrics


        eval_results = {}
        for metric in metrics:
            if metric == 'video_qa_mc':
                all_video_embd = results['video_embd']
                all_text_embd = results['text_embd']
                ans = [item['label'] for item in results['metas']]
                ans = torch.tensor(ans, dtype=torch.long)
                texts = results.get('text', None)
                video_embd = np.stack(all_video_embd, axis=0)   # 2990, 768
                text_embd = np.stack(all_text_embd, axis=0)     # 2990, 5, 768
                text_embd = text_embd.reshape(-1, video_embd.shape[-1])
                eval_results = acc_for_msrvtt_mc(video_embd, text_embd, ans, use_sim=True, texts=texts)
            elif metric == 'video_qa_oe':
                scores, ans = results[0], results[1]
                scores, ans = torch.stack(scores), torch.stack(ans)
                scores = torch.argmax(scores, dim=-1)
                acc =  (scores == ans).float().mean().item()
                eval_results['overall_acc'] = acc
            elif metric == 'recall_for_video_text_retrieval':
                all_video_embd = results['video_embd']
                all_text_embd = results['text_embd']
                metas = results.get('metas', None)
                video_embd = np.stack(all_video_embd, axis=0)
                text_embd = np.stack(all_text_embd, axis=0)
                eval_results = recall_for_video_text_retrieval(video_embd, text_embd, use_sim=True, texts=metas)
            elif metric == 'recall_for_video_text_retrieval_varied':
                all_video_embd = results['video_embd']
                all_text_embd = [np.squeeze(embd) for embd in results['text_embd']]

                texts = results.get('text', None)
                video_embd = np.stack(all_video_embd, axis=0)
                text_embd = np.concatenate(all_text_embd, axis=0)
                eval_results = recall_for_video_text_retrieval_varied(video_embd, text_embd, results['metas'])

            elif metric == 'recall_for_itm_t2v_retrieval':
                rank = {}
                from tqdm import tqdm
                for item in tqdm(results):
                    vid, tid, score = int(item[0]), int(item[1]), item[2]
                    if tid not in rank:
                        rank[tid] = []
                    rank[tid].append([vid, score])
                res = {'Recall@1': 0, 'Recall@5': 0, 'Recall@10': 0, 'MR': []}
                for tid in rank:
                    tmp = sorted(rank[tid], key=lambda d: -d[1])
                    p = [d[0] for d in tmp].index(tid) + 1
                    
                    if p <= 1:
                        res['Recall@1'] += 1.0/len(rank)
                    if p <= 5:
                        res['Recall@5'] += 1.0/len(rank)
                    if p <= 10:
                        res['Recall@10'] += 1.0/len(rank)
                    res['MR'].append(p)
                res['Recall@1'] = res['Recall@1'] * 100
                res['Recall@5'] = res['Recall@5'] * 100
                res['Recall@10'] = res['Recall@10'] * 100
                res['MR'] = np.median(res['MR'])
                res['Recall@all'] = res['Recall@1'] + res['Recall@5'] + res['Recall@10'] - res['MR']
                eval_results = res 
            elif metric == 'acc_for_val':
                scores, ans = results[0], results[1]
                scores, ans = torch.stack(scores), torch.stack(ans)
                scores = torch.argmax(scores, dim=-1)
                acc =  (scores == ans).float().mean().item()
                eval_results['acc'] = acc

        return eval_results


@DATASETS.register_module()
class MsvdDataset(MsrvttVideoDataset):
    def __init__(self, is_qa=False, test_ret=False, **kwargs):
        self.is_qa = is_qa
        self.test_ret = test_ret
        super().__init__(**kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        ann_info = hload_pkl(self.ann_file)

        video_infos = []
        for i, video_info in enumerate(ann_info):
            frame_dir = video_info['filename']
            video_info['frame_dir'] = frame_dir 
            video_info['index'] = i
            video_info['label'] = -1 if 'answer_idx' not in video_info else video_info['answer_idx']

            if isinstance(video_info['text'], str):
                video_info['text'] = [video_info['text']] 
            else:
                if not self.test_ret:
                    video_info['text'] = [rnd.choice(video_info['text'])]
                else:
                    video_info['clip_text_candidate'] = list(range(len(video_info['text'])))

            video_infos.append(video_info) 
        del ann_info

        return video_infos


@DATASETS.register_module()
class VideoQADataset(PKLVideoDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        ann_info = hload_pkl(self.ann_file)

        video_infos = []
        for i, video_info in enumerate(ann_info):
            info_dict = {}   
            info_dict['filename'] = video_info['vid_name']
            frame_dir = info_dict['filename']
            info_dict['frame_dir'] = frame_dir
            info_dict['index'] = i
            info_dict['label'] = video_info['answer_idx']
            info_dict['answers'] = video_info['answers']
            info_dict['question'] = video_info['q']
            info_dict['subtitle'] = video_info['located_sub_text']
            info_dict['frame_ind'] = video_info['located_frame']
            info_dict['total_frames'] = video_info.get('total_frames', -1)
            video_infos.append(info_dict)  
        del ann_info

        return video_infos 

    def evaluate(self, results, metrics='video_qa_mc', metric_options=None, logger=None, normalize=False, **deprecated_kwargs):
        '''Evaluate the dataset with
            video QA MC / video QA OE / video caption ?
            TODO: caption
        Args:
            results (dict): Testing results of the dataset.
            metrics (str | list[str]): Metrics to be evaluated.
                Default value is `recall_for_video_text_retrieval`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        '''

        if deprecated_kwargs != {}:
            warnings.warn(f'Unrecognized parameters: {deprecated_kwargs}')

        if isinstance(metrics, str):
            metrics = [metrics]
        else:
            metrics = metrics


        eval_results = {}
        for metric in metrics:
            if metric == 'video_qa_mc':     
                scores, ans = results[0], results[1]
                scores, ans = torch.stack(scores), torch.stack(ans)
                scores = torch.argmax(scores, dim=-1)
                acc =  (scores == ans).float().mean().item()
                eval_results['acc'] = acc
            elif metric == 'video_qa_oe':
                scores, ans = results[0], results[1]
                scores, ans = torch.stack(scores), torch.stack(ans)
                scores = torch.argmax(scores, dim=-1)
                acc =  (scores == ans).float().mean().item()
                eval_results['overall_acc'] = acc
            elif metric == 'video_qa_mc_ret':
                all_video_embd = results['video_embd']
                all_text_embd = results['text_embd']
                ans = [item['label'] for item in results['metas']]
                ans = torch.tensor(ans, dtype=torch.long)
                texts = results.get('text', None)
                video_embd = np.stack(all_video_embd, axis=0)   # N, 768
                text_embd = np.stack(all_text_embd, axis=0)     # N, ans_num, 768
                text_embd = text_embd.reshape(-1, video_embd.shape[-1])
                eval_results = acc_for_msrvtt_mc(video_embd, text_embd, ans, use_sim=True, texts=texts)


        return eval_results

@DATASETS.register_module()
class TGIFVideoQADataset(VideoQADataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        ann_info = hload_pkl(self.ann_file)

        video_infos = []
        for i, video_info in enumerate(ann_info):
            info_dict = {}   
            info_dict['filename'] = video_info['vid_name'] if 'filename' not in video_info else video_info['filename']
            frame_dir = info_dict['filename']
            info_dict['frame_dir'] = frame_dir
            info_dict['index'] = i
            info_dict['label'] = video_info['answer_idx']
            info_dict['answers'] = video_info['answers'] if 'answers' in video_info else video_info['text']
            info_dict['question'] = video_info['question'] if 'question' in video_info else ""
            video_infos.append(info_dict) 
        del ann_info

        return video_infos



@DATASETS.register_module()
class WebVidDataset(VideoDataset):
    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        ann_info = hload_pkl(self.ann_file)

        video_infos = []
        for i, video_info in enumerate(ann_info):
            filename = osp.join(self.data_prefix, video_info['filename']) 
            video_info['filename'] = filename
            frame_dir = video_info['filename']
            video_info['frame_dir'] = frame_dir 
            video_info['index'] = i
            video_info['label'] = -1 
            video_info['text'] = [video_info['text']] 
            video_infos.append(video_info) 
        del ann_info
        return video_infos


@DATASETS.register_module()
class CC3MDataset(WebVidDataset):
    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        ann_info = hload_pkl(self.ann_file)

        video_infos = []
        for i, video_info in enumerate(ann_info):
            filename = osp.join(self.data_prefix, video_info['filename']) 
            video_info['filename'] = filename
            frame_dir = video_info['filename']
            video_info['frame_dir'] = frame_dir 
            video_info['index'] = i
            video_info['label'] = -1 
            video_info['text'] = [video_info['text']] 
            video_infos.append(video_info) 
        del ann_info
        return video_infos


    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        if self.multi_class and isinstance(results['label'], list):
            onehot = self.label2array(self.num_classes, results['label'])
            results['label'] = onehot

        filename = results.pop('filename')
        results['img_prefix'] = None
        results['img_info'] = {'filename': filename}

        return self.pipeline(results)
