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
from tqdm import tqdm
from .MNLoss import NonzeroDataset, CrossDataset, CrossRandomBatchSampler

UNK = '<unk>'
PAD = '**PAD**'
MASK_MIN = -np.inf


def gen_skip_mask(Y, skip_mask):
    if skip_mask is None:
        return None
    skip_mask_file = open(skip_mask, 'r')
    lines = skip_mask_file.readlines()
    mask = []
    for l in lines:
        mask.append(int(l))
    mask = np.array(mask)
    return Y[mask]

def svm_data_proc(x):
    x = [(int(i.split(':')[0]), float(i.split(':')[1])) for i in x.split()]
    idx, val = zip(*x)
    return idx, val

def obj_arr_to_csr(U, max_j=None):
    ids, vals = zip(*U)
    indices, indptr = [], [0]
    for i in ids:
        indices.extend(i)
        indptr.append(indptr[-1] + len(i))
    data = [v for vs in vals for v in vs]
    if max_j is None:
        return csr_matrix((data, indices, indptr))
    else:
        return csr_matrix((data, indices, indptr), shape=(len(ids), max_j))

def newtokenize(text, word_dict, max_seq_length):
    text = ' '.join(text.split(','))
    tokenizer = RegexpTokenizer(r'\w+')
    return [word_dict[t.lower()] for t in tokenizer.tokenize(text) if not t.isnumeric()][:max_seq_length]

def generate_batch_sogram(data_batch):
    data = data_batch[0]
    us = data['u'] #[torch.LongTensor(u) if isinstance(u, list) else torch.LongTensor(u.tolist()) for u in data['u']]
    vs = data['v'] #[torch.LongTensor(v) for v in data['v']]
    return {
        'us': spmtx2tensor(us),
        'vs': spmtx2tensor(vs),
        '_as':  torch.Tensor(data['_a']),
        '_bs':  torch.Tensor(data['_b']),
        '_abs': torch.Tensor(data['_ab']),
        '_bbs': torch.Tensor(data['_bb']),
        'ys': torch.Tensor(data['y'].A1.ravel()),
    }

def generate_batch_cross(data_batch):
    data = data_batch[0]
    us, vs = data['us'], data['vs']
    return {
        'us': spmtx2tensor(us) if us is not None else None,
        'vs': spmtx2tensor(vs) if vs is not None else None,
        '_as':  torch.Tensor(data['_as']) if us is not None else None,
        '_bs':  torch.Tensor(data['_bs']) if vs is not None else None,
        '_abs': torch.Tensor(data['_abs']) if us is not None else None,
        '_bbs': torch.Tensor(data['_bbs']) if vs is not None else None,
        'ys': spmtx2tensor(data['ys']) if (us is not None and vs is not None) else None,
    }

def spmtx2tensor(spmtx):
    coo = coo_matrix(spmtx)
    idx = np.vstack((coo.row, coo.col))
    val = coo.data
    idx = torch.LongTensor(idx)
    #val = torch.FloatTensor(val)
    val = torch.Tensor(val)
    shape = coo.shape
    return torch.sparse_coo_tensor(idx, val, torch.Size(shape))

