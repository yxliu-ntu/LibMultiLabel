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
    us = [torch.LongTensor(u) if isinstance(u, list) else torch.LongTensor(u.tolist()) for u in data['us']]
    vs = [torch.LongTensor(v) for v in data['vs']]
    return {
        'U': pad_sequence(us, batch_first=True),
        'V': pad_sequence(vs, batch_first=True),
        'A':  torch.FloatTensor(data['_as']),
        'B':  torch.FloatTensor(data['_bs']),
        'Ab': torch.FloatTensor(data['_abs']),
        'Bb': torch.FloatTensor(data['_bbs']),
        'Y': _spmtx2tensor(data['ys']),
    }

def _spmtx2tensor(spmtx):
    coo = coo_matrix(spmtx)
    idx = np.vstack((coo.row, coo.col))
    val = coo.data
    idx = torch.LongTensor(idx)
    val = torch.FloatTensor(val)
    shape = coo.shape
    return torch.sparse_coo_tensor(idx, val, torch.Size(shape))

