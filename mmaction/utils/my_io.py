from typing import IO, Any, List

import shutil
import subprocess
from contextlib import contextmanager
import os
import glob
import threading
import torch
import io
import collections
from PIL import Image
import cv2
import numpy as np
import mmcv
import pickle
from mmcv.runner import CheckpointLoader



__all__ = [
    'hlist_files', 'hexists', 'hmkdir', 'hglob', 'hisdir', 'hcountline',
    'hload_pkl', 'hload_mmcv', 'hload_vocab', 'hload_pil', 'hload_cv2', 'hsave_torch', 'hload_torch',
    'hsave_pkl', 'hdelete', 'hmove'
]


def hload_pkl(filepath):
    """ load pickle """
    with open(filepath, 'rb') as fr:
        file = pickle.load(fr)
        return file


def hsave_pkl(pkl_file, filepath):
    """ save pickle """
    mmcv.dump(pkl_file, filepath)


def hload_mmcv(filepath, flag='color', channel_order='rgb', backend=None):
    """
    load image using mmcv
    """
    img = mmcv.imread(filepath, flag=flag, channel_order=channel_order, backend=backend)
    return img


def hload_pil(filepath):
    """
    load image using PIL
    """
    img = Image.open(filepath)
    return img


def hload_cv2(filepath):
    """
    load image using cv2
    """
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img


def hload_vocab(vocab_path):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_path, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
        return vocab


def hload_torch(filepath: str, map_location='cpu', **kwargs):
    """ load model"""
    return torch.load(filepath, map_location=map_location, **kwargs)


def hsave_torch(obj, filepath: str, **kwargs):
    """ save model """
    torch.save(obj, filepath, **kwargs)




def hlist_files(folders: List[str]) -> List[str]:
    """
        Args:
            folders (List): file path list
        Returns:
            a list of file
    """
    files = []
    for folder in folders:
        if os.path.isdir(folder):
            files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
        elif os.path.isfile(folder):
            files.append(folder)
        else:
            print('Path {} is invalid'.format(folder))

    return files


def hexists(file_path: str) -> bool:
    """ check whether a file_path is exists """
    return os.path.exists(file_path)


def hdelete(file_path: str):
    """ delete a file_path """
    os.remove(file_path)


def hisdir(file_path: str) -> bool:
    """ check whether a file_path is a dir """
    return os.path.isdir(file_path)


def hmkdir(file_path: str) -> bool:
    """ mkdir """
    os.mkdir(file_path)
    return True



def hglob(search_path, sort_by_time=False):
    """ glob """
    files = glob.glob(search_path)
    if sort_by_time:
        files = sorted(files, key=lambda x: os.path.getmtime(x))
    return files





def hmove(src_path, res_path):
    """
    move src_path to res_path
    """
    os.rename(src_path, res_path)



def hcountline(path):
    '''
    count line in file
    '''
    count = 0
    with open(path, 'r') as f:
        for line in f:
            count += 1
    return count
