import copy as cp
import io
from typing import Any, Dict, List,  Union
import os
import os.path as osp
import shutil
import warnings
import random
import mmcv
import ffmpeg
import numpy as np
import torch
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair
from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..builder import PIPELINES
import sng_parser
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
from spacy.util import filter_spans

# for TextTokenizer
from .tokenization import BertTokenizer_FromPretrained
from ...utils import ENGLISH_STOP_WORDS, ENGLISH_STOP_WORDS_BERT_TOKENS, _is_punctuation, bruteforce


@PIPELINES.register_module()
class LoadHVULabel:
    """Convert the HVU label from dictionaries to torch tensors.

    Required keys are "label", "categories", "category_nums", added or modified
    keys are "label", "mask" and "category_mask".
    """

    def __init__(self, **kwargs):
        self.hvu_initialized = False
        self.kwargs = kwargs

    def init_hvu_info(self, categories, category_nums):
        assert len(categories) == len(category_nums)
        self.categories = categories
        self.category_nums = category_nums
        self.num_categories = len(self.categories)
        self.num_tags = sum(self.category_nums)
        self.category2num = dict(zip(categories, category_nums))
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] + self.category_nums[i])
        self.category2startidx = dict(zip(categories, self.start_idx))
        self.hvu_initialized = True

    def __call__(self, results):
        """Convert the label dictionary to 3 tensors: "label", "mask" and
        "category_mask".

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        if not self.hvu_initialized:
            self.init_hvu_info(results['categories'], results['category_nums'])

        onehot = torch.zeros(self.num_tags)
        onehot_mask = torch.zeros(self.num_tags)
        category_mask = torch.zeros(self.num_categories)

        for category, tags in results['label'].items():
            category_mask[self.categories.index(category)] = 1.
            start_idx = self.category2startidx[category]
            category_num = self.category2num[category]
            tags = [idx + start_idx for idx in tags]
            onehot[tags] = 1.
            onehot_mask[start_idx:category_num + start_idx] = 1.

        results['label'] = onehot
        results['mask'] = onehot_mask
        results['category_mask'] = category_mask
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'hvu_initialized={self.hvu_initialized})')
        return repr_str


@PIPELINES.register_module()
class BertTokenizer:
    def __init__(self,
                 vocab_file_path=None,
                 pretrained_model='bert-base-uncased',
                 max_length=25,
                 do_lower_case=True,
                 do_mask=False,
                 mlm_probability=0.15,
                 is_ans=False,
                 temporal_cat=False,
                 whole_word_mask=False,
                 pos_tag_mask=False,
                 scene_graph_mask_obj_verb=False,
                 scene_graph_mask_obj_rel=False,
                 itm_test_for_retrieval=False,
                 scene_graph_mask=False,
                 skip_existing=False):
        """
        skip_existing: Whether to regenerate the token_ids and other information 
                       that already exists in the annotation file (if it exists)
        """
        self.vocab_file_path = vocab_file_path
        self.pretrained_model = pretrained_model
        self.do_lower_case = do_lower_case
        self.skip_existing = skip_existing
        self.max_length = max_length
        self.do_mask = do_mask
        self.whole_word_mask = whole_word_mask
        self.mlm_probability = mlm_probability
        self.itm_test_for_retrieval = itm_test_for_retrieval
        self.scene_graph_mask = scene_graph_mask
        self.scene_graph_mask_obj_verb = scene_graph_mask_obj_verb
        self.scene_graph_mask_obj_rel = scene_graph_mask_obj_rel
        self.pos_tag_mask = pos_tag_mask
        self.temporal_cat = temporal_cat
        self.is_ans = is_ans
        self.tokenizer = BertTokenizer_FromPretrained(from_pretrained=self.pretrained_model, vocab_file=self.vocab_file_path, do_lower_case=self.do_lower_case, 
                                                        remove_space=True, keep_accents=False)
        self.stop_words = ENGLISH_STOP_WORDS_BERT_TOKENS

    def mask_fromhugface(self, results, already_mask=None):
        inputs = results['token_ids']
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        if already_mask is None:
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)  # only mask unspecial tokens

            stop_words_tokens_mask = [
                False if val not in self.stop_words else True for val in labels.squeeze().tolist() 
            ]
            stop_words_tokens_mask = torch.tensor(stop_words_tokens_mask, dtype=torch.bool)
            probability_matrix.masked_fill_(stop_words_tokens_mask, value=0.0)  # only mask not stop words tokens
            masked_indices = torch.bernoulli(probability_matrix).bool()
        else:
            probability_matrix = already_mask
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)  # only mask unspecial tokens
            masked_indices = probability_matrix.bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        results['token_ids'] = inputs
        results['mlm_label'] = labels
        results['speical_tokens_mask'] = special_tokens_mask
        return results

    def whole_word_mask_call(self, results):

        input_ids = results['token_ids']
        mask_labels = []
        ref_tokens = []

        for id in input_ids.squeeze().tolist():
            token = self.tokenizer.tokenizer._convert_id_to_token(id)
            ref_tokens.append(token)

        if self.scene_graph_mask_obj_verb:
            mask_labels.append(self._whole_word_mask_with_scene_graph(ref_tokens, input_ids))
        else:
            mask_labels.append(self._whole_word_mask(ref_tokens))

        mask_labels = torch.tensor(mask_labels)
        
        return self.mask_fromhugface(results, mask_labels)
        

    def scene_graph_parser(self, cand_idxs, input_tokens, input_text):
        origin_text = ' '.join(input_text)
        scene_graph = sng_parser.parse(origin_text)
        object, object_idxs = [], []
        attr, attr_idxs = [], []
        relation, relation_idxs = [], []
        for entity in scene_graph['entities']:
            object.extend(entity['head'].split(' '))
            for modifier in entity['modifiers']:
                if modifier['dep'] != 'det':
                    attr.extend(modifier['span'].split(' '))
        for rel in scene_graph['relations']:
            relation.extend(rel['relation'].split(' '))

        object, attr, relation = set(object), set(attr), set(relation)

        sng_indexes = []
        for token_id in cand_idxs:
            text = ""
            if len(token_id) > 1:
                for i in token_id:
                    text += input_tokens[i].replace('#', '')
            else:
                text = input_tokens[token_id[0]]
            
            if text in object:
                object_idxs.append(token_id)
            elif text in attr:
                attr_idxs.append(token_id)
            elif text in relation:
                relation_idxs.append(token_id)
        
        sng_indexes = object_idxs + attr_idxs + relation_idxs

        return sng_indexes


    def scene_graph_parser_obj_verb(self, cand_idxs, input_tokens, input_text):
        origin_text = ' '.join(input_text)
        doc = nlp(origin_text)
        token_pos = [[token.text, token.pos_] for token in doc]

        token_spacy_to_real_text = {}
        if len(token_pos) != len(input_text):
            token_pos_new = []
            part_text = ""
            real_idx = 0
            for idx in range(len(token_pos)):
                token, pos = token_pos[idx]
                if token != input_text[real_idx]:
                    if part_text + token == input_text[real_idx]:
                        token_pos_new.append([part_text + token, pos])
                        part_text = ""
                        token_spacy_to_real_text[idx] = real_idx
                        real_idx += 1
                    else:
                        part_text += token
                        token_spacy_to_real_text[idx] = real_idx

                else:
                    token_pos_new.append([token, pos])
                    token_spacy_to_real_text[idx] = real_idx
                    real_idx += 1

            token_pos = token_pos_new

        object, object_idxs = [], []
        attr, attr_idxs = [], []
        verb, verb_idxs = [], []
    
        pattern = [{'POS': 'VERB', 'OP': '?'},
                {'POS': 'ADV', 'OP': '*'},
                {'POS': 'AUX', 'OP': '*'},
                {'POS': 'VERB', 'OP': '+'}]

        # instantiate a Matcher instance
        matcher = Matcher(nlp.vocab)
        matcher.add('Verb Phrase', [pattern], on_match=None)

        matches = matcher(doc)
        #  verb phrase
        for _, start, end in matches:
            for v_id in range(start, end):
                if len(token_spacy_to_real_text) != 0:
                    v_id = token_spacy_to_real_text[v_id]
                try:
                    verb.append(input_text[v_id])
                    verb_idxs.append([c_id for c_id in cand_idxs[v_id]])
                except IndexError:
                    print("got index error")

        for p_id, (word, pos) in enumerate(token_pos):
            if pos in ['NOUN', 'PROPN']:
                object.append(input_text[p_id])
                object_idxs.append([v_id for v_id in cand_idxs[p_id]])

        # noun
        sng_indexes = object_idxs + verb_idxs + attr_idxs

        return sng_indexes    


    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]" or token == '[PAD]':
                continue
            # if punctuation 
            if len(token) == 1:
                if _is_punctuation(token):
                    continue
            # if stop words
            if token in ENGLISH_STOP_WORDS:
                continue
            
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(cand_indexes) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels


    def _whole_word_mask_with_scene_graph(self, input_tokens, input_ids=None, max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        cand_indexes = []
        real_text = []

        #  只去除 special tokens
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]" or token == '[PAD]':
                continue            
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
      
        for idx in cand_indexes:
            cur_token = ""
            for t_id in idx:
                cur_token += input_tokens[t_id].replace("#", "")
            real_text.append(cur_token) 

        if self.scene_graph_mask_obj_verb:
            sng_indexes = self.scene_graph_parser_obj_verb(cand_indexes, input_tokens, real_text)
        else:
            sng_indexes = self.scene_graph_parser(cand_indexes, input_tokens, real_text)
    
        random.shuffle(sng_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(sng_indexes) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        count = 0
        for index_set in sng_indexes:
            if count >= num_to_predict:
                break

            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)
            count += 1

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels


    def __call__(self, results):
        if ('token_ids' not in results) or self.skip_existing:  
            # If there are no token_ids in the annotation file or token_ids need to be regenerated
            # clip_ids, set to the first element of list.
            # make sure result['text'] is a list
            clip_ids = results['clip_text_candidate'] if 'clip_text_candidate' in results else [0]  # 这个也要是个list
            text_list = [results['text'][idx] for idx in clip_ids]
            if self.itm_test_for_retrieval:
                text_list = results['text']
            num_texts = len(text_list)
            if self.temporal_cat:
                text_list = [' '.join(text_list)]
                tokenize_result = self.tokenizer.tokenize(text_list, add_special_tokens=True, max_length=self.max_length*num_texts,
                                                  padding='max_length', truncation=True, return_tensors='pt')
            else:
                tokenize_result = self.tokenizer.tokenize(text_list, add_special_tokens=True, max_length=self.max_length,
                                                  padding='max_length', truncation=True, return_tensors='pt')
            token_ids = tokenize_result['input_ids']
            segment_ids = tokenize_result['token_type_ids']
            input_mask = tokenize_result['attention_mask']
            del text_list
        else:
            token_ids = torch.tensor(results['token_ids'], dtype=torch.long)
            segment_ids = torch.tensor(results['segment_ids'], dtype=torch.long)
            input_mask = torch.tensor(results['input_mask'], dtype=torch.long)

        if not self.is_ans:
            results['token_ids'] = token_ids
            results['segment_ids'] = segment_ids
            results['input_mask'] = input_mask
        else:
            results['ans_ids'] = token_ids
            results['ans_mask'] = input_mask
        if self.do_mask:
            if self.whole_word_mask:
                return self.whole_word_mask_call(results)
            else:
                return self.mask_fromhugface(results)
    
        return results

    def remove_stop_words(self):
        stop_words_ids = self.tokenizer.tokenizer(list(ENGLISH_STOP_WORDS), add_special_tokens=False)['input_ids']
        stop_words_sets = set([i for item in stop_words_ids for i in item])
        print(stop_words_sets)
        return stop_words_sets

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'vocab_file_path={self.vocab_file_path}, '
                    f'pretrained_model={self.pretrained_model}, '
                    f'do_lower_case={self.do_lower_case}, '
                    f'skip_existing={self.skip_existing}')
        return repr_str

@PIPELINES.register_module()
class QATextPrepare:
    '''Prepare question answer subtitle text
    '''
    def __init__(self, split_token='[SEP]', use_subtitle=False, use_mask=False, use_all_ans=False, vlep=False, **kwargs):
        self.split_token = split_token
        self.use_subtitle = use_subtitle
        self.use_mask = use_mask
        self.use_all_ans = use_all_ans
        self.vlep = vlep

    def __call__(self, results):
        question = results.get('question', "") if not self.vlep else "What is more likely to happen next ? " 
        subtitle = results.get('subtitle', None) if self.use_subtitle else None
        if self.use_all_ans:
            options = ' '.join(results['answers'][i] for i in range(len(results['answers'])))
            text = []
            for i in range(len(results['answers'])):
                if self.vlep:
                    text.append(question + " Answer: " + results['answers'][i] + " Subtitle: " + subtitle) 
                elif self.use_subtitle:
                    text.append(question + " Options: " + options + " Answer: " + results['answers'][i] + " Subtitle: " + subtitle)          
                else:
                    text.append(question + " Options: " + options + " Answer: " + results['answers'][i])
        else:
            if subtitle is not None:  # tvqa  vlep
                if question is not "":
                    text = [' '.join([question, self.split_token, a, self.split_token, subtitle]) for a in results['answers']] 
                else:
                    text = [' '.join([a, self.split_token, subtitle]) for a in results['answers']]

            elif 'answers' in results and len(results['answers']) > 0: # for multiple choice qa
                if self.use_mask:
                    text = [' '.join([question, "The answer is", a, "It is a [MASK] answer"]) for a in results['answers']]
                else:
                    text = [' '.join([question, self.split_token, a]) for a in results['answers']]
            else: # for open-ended qa , msrvtt/msvd qa question is None
                text = [question] if question is not "" else results['text']
                if self.use_mask:
                    text.append("The answer is [MASK]")
                    text = [' '.join(text)]
        results['text'] = text
        results['subtitle'] = None
        results['clip_text_candidate'] = [i for i in range(len(text))]
        return results
        
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'split_token={self.split_token})')
        return repr_str


@PIPELINES.register_module()
class MCRetTextPrepare:
    '''Prepare question answer subtitle text

    '''
    def __init__(self, is_question=False, is_answer=False, test_mode=False, **kwargs):
        self.is_question = is_question
        self.is_answer = is_answer
        self.test_mode = test_mode

    def question_prepare(self, results):
        question = results.get('question', None)
        subtitle = results.get('subtitle', None)
        if subtitle is not None:  # tvqa  vlep
            text = [' '.join([question, '[SEP]', subtitle])]
        else:
            text = [question]
        return text

    def answer_prepare(self, results):
        ans = results['answers']
        ans_idx = results['label']
        text = [a for a in ans]
        if not self.test_mode:
            # Need to put the correct one at first
            match_ans = text[ans_idx]
            del text[ans_idx]
            text = [match_ans] + text
        return text

    def __call__(self, results):
        if self.is_question:
            text = self.question_prepare(results)
        elif self.is_answer:
            text = self.answer_prepare(results)
        else:
            raise NotImplementedError
        results['text'] = text
        results['subtitle'] = None
        results['clip_text_candidate'] = [i for i in range(len(text))]
        return results
        
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'split_token={self.split_token})')
        return repr_str



@PIPELINES.register_module()
class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 keep_tail_frames=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(int)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=int)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                # In this step, according to the number of clips, 
                # all frames are divided into num_clip segments, 
                # and the starting point of each segment is taken.
                base_offsets = np.arange(self.num_clips) * avg_interval  
                # Then get a random offset to achieve the purpose of getting the starting point of a random fragment
                clip_offsets = base_offsets + np.random.randint(         
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)

        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)       

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class UntrimmedSampleFrames:
    """Sample frames from the untrimmed video.

    Required keys are "filename", "total_frames", added or modified keys are
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): The length of sampled clips. Default: 1.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 16.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self, clip_len=1, frame_interval=16, start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        start_index = results['start_index']

        clip_centers = np.arange(self.frame_interval // 2, total_frames,
                                 self.frame_interval)
        num_clips = clip_centers.shape[0]
        frame_inds = clip_centers[:, None] + np.arange(
            -(self.clip_len // 2), self.clip_len -
            (self.clip_len // 2))[None, :]
        # clip frame_inds to legal range
        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval})')
        return repr_str


@PIPELINES.register_module()
class DenseSampleFrames(SampleFrames):
    """Select frames from the video by dense sample strategy.

    Required keys are "filename", added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        sample_range (int): Total sample range for dense sample.
            Default: 64.
        num_sample_positions (int): Number of sample start positions, Which is
            only used in test mode. Default: 10. That is to say, by default,
            there are at least 10 clips for one input sample in test mode.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 *args,
                 sample_range=64,
                 num_sample_positions=10,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_range = sample_range
        self.num_sample_positions = num_sample_positions

    def _get_train_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in train mode.

        It will calculate a sample position and sample interval and set
        start index 0 when sample_pos == 1 or randomly choose from
        [0, sample_pos - 1]. Then it will shift the start index by each
        base offset.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_idx = 0 if sample_position == 1 else np.random.randint(
            0, sample_position - 1)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = (base_offsets + start_idx) % num_frames
        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in test mode.

        It will calculate a sample position and sample interval and evenly
        sample several start indexes as start positions between
        [0, sample_position-1]. Then it will shift each start index by the
        base offsets.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_list = np.linspace(
            0, sample_position - 1, num=self.num_sample_positions, dtype=int)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = list()
        for start_idx in start_list:
            clip_offsets.extend((base_offsets + start_idx) % num_frames)
        clip_offsets = np.array(clip_offsets)
        return clip_offsets

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'sample_range={self.sample_range}, '
                    f'num_sample_positions={self.num_sample_positions}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SampleAVAFrames(SampleFrames):

    def __init__(self, clip_len, frame_interval=2, test_mode=False):

        super().__init__(clip_len, frame_interval, test_mode=test_mode)

    def _get_clips(self, center_index, skip_offsets, shot_info):
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        if not self.test_mode:
            frame_inds = frame_inds + skip_offsets
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)
        return frame_inds

    def __call__(self, results):
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        shot_info = results['shot_info']

        center_index = fps * (timestamp - timestamp_start) + 1

        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)

        results['frame_inds'] = np.array(frame_inds, dtype=int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SampleProposalFrames(SampleFrames):
    """Sample frames from proposals in the video.

    Required keys are "total_frames" and "out_proposals", added or
    modified keys are "frame_inds", "frame_interval", "num_clips",
    'clip_len' and 'num_proposals'.

    Args:
        clip_len (int): Frames of each sampled output clip.
        body_segments (int): Number of segments in course period.
        aug_segments (list[int]): Number of segments in starting and
            ending period.
        aug_ratio (int | float | tuple[int | float]): The ratio
            of the length of augmentation to that of the proposal.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        test_interval (int): Temporal interval of adjacent sampled frames
            in test mode. Default: 6.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        mode (str): Choose 'train', 'val' or 'test' mode.
            Default: 'train'.
    """

    def __init__(self,
                 clip_len,
                 body_segments,
                 aug_segments,
                 aug_ratio,
                 frame_interval=1,
                 test_interval=6,
                 temporal_jitter=False,
                 mode='train'):
        super().__init__(
            clip_len,
            frame_interval=frame_interval,
            temporal_jitter=temporal_jitter)
        self.body_segments = body_segments
        self.aug_segments = aug_segments
        self.aug_ratio = _pair(aug_ratio)
        if not mmcv.is_tuple_of(self.aug_ratio, (int, float)):
            raise TypeError(f'aug_ratio should be int, float'
                            f'or tuple of int and float, '
                            f'but got {type(aug_ratio)}')
        assert len(self.aug_ratio) == 2
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.test_interval = test_interval

    @staticmethod
    def _get_train_indices(valid_length, num_segments):
        """Get indices of different stages of proposals in train mode.

        It will calculate the average interval for each segment,
        and randomly shift them within offsets between [0, average_duration].
        If the total number of frames is smaller than num segments, it will
        return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        avg_interval = (valid_length + 1) // num_segments
        if avg_interval > 0:
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = base_offsets + np.random.randint(
                avg_interval, size=num_segments)
        else:
            offsets = np.zeros((num_segments, ), dtype=int)

        return offsets

    @staticmethod
    def _get_val_indices(valid_length, num_segments):
        """Get indices of different stages of proposals in validation mode.

        It will calculate the average interval for each segment.
        If the total number of valid length is smaller than num segments,
        it will return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in validation mode.
        """
        if valid_length >= num_segments:
            avg_interval = valid_length / float(num_segments)
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = (base_offsets + avg_interval / 2.0).astype(int)
        else:
            offsets = np.zeros((num_segments, ), dtype=int)

        return offsets

    def _get_proposal_clips(self, proposal, num_frames):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices in the proposal's three
        stages: starting, course and ending stage.

        Args:
            proposal (obj): The proposal object.
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        # proposal interval: [start_frame, end_frame)
        start_frame = proposal.start_frame
        end_frame = proposal.end_frame
        ori_clip_len = self.clip_len * self.frame_interval

        duration = end_frame - start_frame
        assert duration != 0
        valid_length = duration - ori_clip_len

        valid_starting = max(0,
                             start_frame - int(duration * self.aug_ratio[0]))
        valid_ending = min(num_frames - ori_clip_len + 1,
                           end_frame - 1 + int(duration * self.aug_ratio[1]))

        valid_starting_length = start_frame - valid_starting - ori_clip_len
        valid_ending_length = (valid_ending - end_frame + 1) - ori_clip_len

        if self.mode == 'train':
            starting_offsets = self._get_train_indices(valid_starting_length,
                                                       self.aug_segments[0])
            course_offsets = self._get_train_indices(valid_length,
                                                     self.body_segments)
            ending_offsets = self._get_train_indices(valid_ending_length,
                                                     self.aug_segments[1])
        elif self.mode == 'val':
            starting_offsets = self._get_val_indices(valid_starting_length,
                                                     self.aug_segments[0])
            course_offsets = self._get_val_indices(valid_length,
                                                   self.body_segments)
            ending_offsets = self._get_val_indices(valid_ending_length,
                                                   self.aug_segments[1])
        starting_offsets += valid_starting
        course_offsets += start_frame
        ending_offsets += end_frame

        offsets = np.concatenate(
            (starting_offsets, course_offsets, ending_offsets))
        return offsets

    def _get_train_clips(self, num_frames, proposals):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices of each proposal, and then
        assemble them.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list): Proposals fetched.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        clip_offsets = []
        for proposal in proposals:
            proposal_clip_offsets = self._get_proposal_clips(
                proposal[0][1], num_frames)
            clip_offsets = np.concatenate(
                [clip_offsets, proposal_clip_offsets])

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        It will calculate sampled frame indices based on test interval.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        return np.arange(
            0, num_frames - ori_clip_len, self.test_interval, dtype=int)

    def _sample_clips(self, num_frames, proposals):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list | None): Proposals fetched.
                It is set to None in test mode.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.mode == 'test':
            clip_offsets = self._get_test_clips(num_frames)
        else:
            assert proposals is not None
            clip_offsets = self._get_train_clips(num_frames, proposals)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        out_proposals = results.get('out_proposals', None)
        clip_offsets = self._sample_clips(total_frames, out_proposals)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        start_index = results['start_index']
        frame_inds = np.mod(frame_inds, total_frames) + start_index

        results['frame_inds'] = np.array(frame_inds).astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = (
            self.body_segments + self.aug_segments[0] + self.aug_segments[1])
        if self.mode in ['train', 'val']:
            results['num_proposals'] = len(results['out_proposals'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'body_segments={self.body_segments}, '
                    f'aug_segments={self.aug_segments}, '
                    f'aug_ratio={self.aug_ratio}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_interval={self.test_interval}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'mode={self.mode})')
        return repr_str


@PIPELINES.register_module()
class PyAVInit:
    """Using pyav to initialize the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the PyAV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip3 install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = av.open(file_obj, metadata_errors="ignore")

        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend={self.io_backend})'
        return repr_str


@PIPELINES.register_module()
class PyAVDecode:
    """Using PyAV to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            the nearest key frames, which may be duplicated and inaccurate,
            and more suitable for large scene-based video datasets.
            Default: 'accurate'.
    """

    def __init__(self, multi_thread=False, mode='accurate'):
        self.multi_thread = multi_thread
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    @staticmethod
    def frame_generator(container, stream):
        """Frame generator for PyAV."""
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_rgb().to_ndarray()

    def __call__(self, results):
        """Perform the PyAV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        if self.mode == 'accurate':
            # set max indice to make early stop
            max_inds = max(results['frame_inds'])
            i = 0
            for frame in container.decode(video=0):
                if i > max_inds + 1:
                    break
                imgs.append(frame.to_rgb().to_ndarray()[..., ::-1]) # add by lyn, to bgr format
                i += 1

            # the available frame in pyav may be less than its length,
            # which may raise error
            results['imgs'] = [
                imgs[i % len(imgs)] for i in results['frame_inds']
            ]
        elif self.mode == 'efficient':
            for frame in container.decode(video=0):
                backup_frame = frame
                break
            stream = container.streams.video[0]
            for idx in results['frame_inds']:
                pts_scale = stream.average_rate * stream.time_base
                frame_pts = int(idx / pts_scale)
                container.seek(
                    frame_pts, any_frame=False, backward=True, stream=stream)
                frame = self.frame_generator(container, stream)
                if frame is not None:
                    imgs.append(frame)
                    backup_frame = frame
                else:
                    imgs.append(backup_frame)
            results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['video_reader'] = None
        del container

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread}, mode={self.mode})'
        return repr_str


@PIPELINES.register_module()
class PIMSInit:
    """Use PIMS to initialize the video.

    PIMS: https://github.com/soft-matter/pims

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will always use ``pims.PyAVReaderIndexed``
            to decode videos into accurate frames. If set to 'efficient', it
            will adopt fast seeking by using ``pims.PyAVReaderTimed``.
            Both will return the accurate frames in most cases.
            Default: 'accurate'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', mode='accurate', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def __call__(self, results):
        try:
            import pims
        except ImportError:
            raise ImportError('Please run "conda install pims -c conda-forge" '
                              'or "pip3 install pims" to install pims first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        if self.mode == 'accurate':
            container = pims.PyAVReaderIndexed(file_obj)
        else:
            container = pims.PyAVReaderTimed(file_obj)

        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(io_backend={self.io_backend}, '
                    f'mode={self.mode})')
        return repr_str


@PIPELINES.register_module()
class PIMSDecode:
    """Using PIMS to decode the videos.

    PIMS: https://github.com/soft-matter/pims

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = [container[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class PyAVDecodeMotionVector(PyAVDecode):
    """Using pyav to decode the motion vectors from video.

    Reference: https://github.com/PyAV-Org/PyAV/
        blob/main/tests/test_decode.py

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "motion_vectors", "frame_inds".
    """

    @staticmethod
    def _parse_vectors(mv, vectors, height, width):
        """Parse the returned vectors."""
        (w, h, src_x, src_y, dst_x,
         dst_y) = (vectors['w'], vectors['h'], vectors['src_x'],
                   vectors['src_y'], vectors['dst_x'], vectors['dst_y'])
        val_x = dst_x - src_x
        val_y = dst_y - src_y
        start_x = dst_x - w // 2
        start_y = dst_y - h // 2
        end_x = start_x + w
        end_y = start_y + h
        for sx, ex, sy, ey, vx, vy in zip(start_x, end_x, start_y, end_y,
                                          val_x, val_y):
            if (sx >= 0 and ex < width and sy >= 0 and ey < height):
                mv[sy:ey, sx:ex] = (vx, vy)

        return mv

    def __call__(self, results):
        """Perform the PyAV motion vector decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max index to make early stop
        max_idx = max(results['frame_inds'])
        i = 0
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        codec_context.options = {'flags2': '+export_mvs'}
        for packet in container.demux(stream):
            for frame in packet.decode():
                if i > max_idx + 1:
                    break
                i += 1
                height = frame.height
                width = frame.width
                mv = np.zeros((height, width, 2), dtype=np.int8)
                vectors = frame.side_data.get('MOTION_VECTORS')
                if frame.key_frame:
                    # Key frame don't have motion vectors
                    assert vectors is None
                if vectors is not None and len(vectors) > 0:
                    mv = self._parse_vectors(mv, vectors.to_ndarray(), height,
                                             width)
                imgs.append(mv)

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['motion_vectors'] = np.array(
            [imgs[i % len(imgs)] for i in results['frame_inds']])
        return results


@PIPELINES.register_module()
class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        num_threads (int): Number of thread to decode the video. Default: 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip3 install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        results['video_reader'] = container
        results['total_frames'] = len(container)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@PIPELINES.register_module()
class DecordDecode:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets. Default: 'accurate'.
    """

    def __init__(self, mode='accurate'):
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']

        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            # imgs = container.get_batch(frame_inds).asnumpy()[..., ::-1]  # modified  , convert to bgr format
            imgs = list(imgs)
            imgs = [img[..., ::-1] for img in imgs]  # modified  , convert to bgr format
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy()[..., ::-1])  # modified  , convert to bgr format
        
        # if self.need_time:
        #     results['start_time'] = container.get_frame_timestamp(results['clip_start'])[:,0] # first frame start time
        #     results['end_time'] = container.get_frame_timestamp(results['clip_end'])[:,1]     # last frame end time


        results['video_reader'] = None
        del container
        # gc.collect()
        results['imgs'] = imgs
    
        # import copy
        # raw_imgs = copy.deepcopy(results['imgs'])
        # results['raw_imgs'] = raw_imgs
        
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@PIPELINES.register_module()
class FFmpegDecode:
    """Using ffmpeg-python to initialize the video_reader.

    ffmpeg-python: https://github.com/kkroening/ffmpeg-python

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        num_threads (int): Number of thread to decode the video. Default: 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self, size=224, num_frames=32, fps=10, crop='center', crop_only=False, num_clips=1, **kwargs):
        self.size = size
        self.num_frames = num_frames
        self.crop = crop
        self.crop_only = crop_only
        self.num_clips = num_clips
        self.num_sec = num_frames / float(fps)
        self.fps = fps
        self.kwargs = kwargs

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_path = results['filename']
        if 'start' in results:
            start = results['start']
            end = results['end']
        else:
            start = 0
            end = float(self._get_duration(video_path))
        results['imgs'] = self._get_video(video_path, start, end, self.num_clips)
        results['num_clips'] = self.num_clips
        results['clip_len'] = self.num_frames
        return results

    def _get_video(self, video_path, start, end, num_clip):
        # video = torch.zeros(num_clip, self.num_frames, self.size, self.size, 3)
        video = torch.zeros(num_clip, 3, self.num_frames, self.size, self.size)
        start_ind = np.linspace(start, max(start, end-self.num_sec - 0.4), num_clip) 
        for i, s in enumerate(start_ind):
            video[i] = self._get_video_start(video_path, s) 
        # video = video.view(-1, self.size, self.size, 3)
        return video

    def _get_duration(self, video_path):
        probe = ffmpeg.probe(video_path)
        return probe['format']['duration']
    
    def _get_video_start(self, video_path, start):
        start_seek = start
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )
        if self.crop == 'center':
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = video[:, :, :, ::-1]  # to bgr add by lyn
        video = torch.from_numpy(video.copy())
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = torch.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=torch.uint8)
            video = torch.cat((video, zeros), axis=1)
        # if video.shape[0] < self.num_frames:
        #     zeros = torch.zeros((self.num_frames - video.shape[1], self.size, self.size, 3), dtype=torch.uint8)
        #     video = torch.cat((video, zeros), axis=0)
        
        return video[:, :self.num_frames, :]

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'size={self.size}, '
                    f'num_sec={self.num_sec}, '
                    f'fps={self.fps}, '
                    f'num_frames={self.num_frames}, '
                    f'crop={self.crop}, '
                    f'crop_only={self.crop_only}, '
                    f'num_clips={self.num_clips})')
        return repr_str


@PIPELINES.register_module()
class OpenCVInit:
    """Using OpenCV to initialize the video_reader.

    Required keys are "filename", added or modified keys are "new_path",
    "video_reader" and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.tmp_folder = None
        if self.io_backend != 'disk':
            random_string = get_random_string()
            thread_id = get_thread_id()
            self.tmp_folder = osp.join(get_shm_dir(),
                                       f'{random_string}_{thread_id}')
            os.mkdir(self.tmp_folder)

    def __call__(self, results):
        """Perform the OpenCV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.io_backend == 'disk':
            new_path = results['filename']
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()
            # save the file of same thread at the same place
            new_path = osp.join(self.tmp_folder, f'tmp_{thread_id}.mp4')
            with open(new_path, 'wb') as f:
                f.write(self.file_client.get(results['filename']))

        container = mmcv.VideoReader(new_path)
        results['new_path'] = new_path
        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __del__(self):
        if self.tmp_folder and osp.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend})')
        return repr_str


@PIPELINES.register_module()
class OpenCVDecode:
    """Using OpenCV to decode the video.

    Required keys are "video_reader", "filename" and "frame_inds", added or
    modified keys are "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        """Perform the OpenCV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_ind in results['frame_inds']:
            cur_frame = container[frame_ind]
            # last frame may be None in OpenCV
            while isinstance(cur_frame, type(None)):
                frame_ind -= 1
                cur_frame = container[frame_ind]
            imgs.append(cur_frame)

        results['video_reader'] = None
        del container

        imgs = np.array(imgs)
        # # The default channel order of OpenCV is BGR, thus we change it to RGB
        # imgs = imgs[:, :, :, ::-1]  # modified  , keep it in bgr format
        results['imgs'] = list(imgs)
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class RawFrameDecode:
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                if modality == 'RGB':
                    imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                else:
                    imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx]]))
                    imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx] + 1]))
                continue
            else:
                cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='bgr')  # modified  , convert to bgr format
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str


@PIPELINES.register_module()
class ImageDecode:
    """Load and decode images.

    Required key is "filename", added or modified keys are "imgs", "img_shape"
    and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``ImageDecode`` to load image given the file path.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        filename = results['filename']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        img_bytes = self.file_client.get(filename)

        img = mmcv.imfrombytes(img_bytes, channel_order='bgr')  # modified  , keep it in bgr format
        imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results


