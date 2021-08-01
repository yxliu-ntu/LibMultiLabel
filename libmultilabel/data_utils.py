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

def generate_batch_nonzero(data_batch):
    us = [torch.LongTensor(data['u']) if isinstance(data['u'], list) else torch.LongTensor(data['u'].tolist()) for data in data_batch]
    vs = [torch.LongTensor(data['v']) for data in data_batch]
    return {
        'us': pad_sequence(us, batch_first=True),
        'vs': pad_sequence(vs, batch_first=True),
        '_as':  torch.FloatTensor([data['_a'] for data in data_batch]),
        '_bs':  torch.FloatTensor([data['_b'] for data in data_batch]),
        '_abs': torch.FloatTensor([data['_ab'] for data in data_batch]),
        '_bbs': torch.FloatTensor([data['_bb'] for data in data_batch]),
        'ys': torch.FloatTensor([data['y'] for data in data_batch]),
    }

def generate_batch_cross(data_batch):
    data = data_batch[0]
    if data['us'] is not None:
        us = [torch.LongTensor(u) if isinstance(u, list) else torch.LongTensor(u.tolist()) for u in data['us']]
    else:
        us = None
    if data['vs'] is not None:
        vs = [torch.LongTensor(v) for v in data['vs']]
    else:
        vs = None
    return {
        'U': pad_sequence(us, batch_first=True) if us is not None else None,
        'V': pad_sequence(vs, batch_first=True) if vs is not None else None,
        'A':  torch.FloatTensor(data['_as']) if us is not None else None,
        'B':  torch.FloatTensor(data['_bs']) if vs is not None else None,
        'Ab': torch.FloatTensor(data['_abs']) if us is not None else None,
        'Bb': torch.FloatTensor(data['_bbs']) if vs is not None else None,
        'Y': _spmtx2tensor(data['ys']) if (us is not None and vs is not None) else None,
    }

def _spmtx2tensor(spmtx):
    coo = coo_matrix(spmtx)
    idx = np.vstack((coo.row, coo.col))
    val = coo.data
    idx = torch.LongTensor(idx)
    val = torch.FloatTensor(val)
    shape = coo.shape
    return torch.sparse_coo_tensor(idx, val, torch.Size(shape))

