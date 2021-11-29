import collections
import logging
import os

import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, vstack as sp_vstack
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from tqdm import tqdm
from .MNLoss import NonzeroDataset, CrossDataset, CrossRandomBatchSampler

UNK = Vocab.UNK
PAD = '**PAD**'


def newtokenize(text, word_dict, max_seq_length):
    text = ' '.join(text.split(','))
    tokenizer = RegexpTokenizer(r'\w+')
    return [word_dict[t.lower()] for t in tokenizer.tokenize(text) if not t.isnumeric()][:max_seq_length]

def generate_batch_sogram(data_batch):
    raise NotImplementedError
    data = data_batch[0]
    us = [torch.LongTensor(u) if isinstance(u, list) else torch.LongTensor(u.tolist()) for u in data['u']]
    vs = [torch.LongTensor(v) for v in data['v']]
    return {
        'us': pad_sequence(us, batch_first=True),
        'vs': pad_sequence(vs, batch_first=True),
        '_as':  torch.FloatTensor(data['_a']),
        '_bs':  torch.FloatTensor(data['_b']),
        '_abs': torch.FloatTensor(data['_ab']),
        '_bbs': torch.FloatTensor(data['_bb']),
        'ys': torch.FloatTensor(data['y'].A1.ravel()),
    }

def generate_batch_cross(data_batch):
    data = data_batch[0]
    def _helper(feat):
        if feat is not None:
            feat = feat.tolist()
            fs, fvals = zip(*feat)
            fs = [torch.LongTensor(f) for f in fs]
            fvals = [torch.FloatTensor(fval) for fval in fvals]
        else:
            fs, fvals = None, None
        return fs, fvals
    us, uvals = _helper(data['us'])
    vs, vvals = _helper(data['vs'])
    return {
        'us': pad_sequence(us, batch_first=True) if us is not None else None,
        'vs': pad_sequence(vs, batch_first=True) if vs is not None else None,
        'uvals': pad_sequence(uvals, batch_first=True) if us is not None else None,
        'vvals': pad_sequence(vvals, batch_first=True) if vs is not None else None,
        '_as':  torch.FloatTensor(data['_as']) if us is not None else None,
        '_bs':  torch.FloatTensor(data['_bs']) if vs is not None else None,
        '_abs': torch.FloatTensor(data['_abs']) if us is not None else None,
        '_bbs': torch.FloatTensor(data['_bbs']) if vs is not None else None,
        'ys': _spmtx2tensor(data['ys']) if (us is not None and vs is not None) else None,
    }

def _spmtx2tensor(spmtx):
    coo = coo_matrix(spmtx)
    idx = np.vstack((coo.row, coo.col))
    val = coo.data
    idx = torch.LongTensor(idx)
    val = torch.FloatTensor(val)
    shape = coo.shape
    return torch.sparse_coo_tensor(idx, val, torch.Size(shape))

