import collections
import logging
import os

import torch
import math
import numpy as np
from scipy.sparse import spmatrix, coo_matrix, csr_matrix, vstack as sp_vstack
from typing import Tuple, Sized, Optional, Iterator
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from tqdm import tqdm

UNK = Vocab.UNK
PAD = '**PAD**'


class TextDataset(Dataset):
    """Class for text dataset"""

    def __init__(self, data, word_dict, classes, max_seq_length):
        self.data = data
        self.word_dict = word_dict
        self.classes = classes
        self.max_seq_length = max_seq_length
        self.num_classes = len(self.classes)
        self.label_binarizer = MultiLabelBinarizer().fit([classes])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return {
            'text': torch.LongTensor([self.word_dict[word] for word in data['text']][:self.max_seq_length]),
            'label': torch.FloatTensor(self.label_binarizer.transform([data['label']])[0]),
            'index': data.get('index', 0)
        }

class TwoTowerDataset(Dataset):
    """Class for text dataset for 2-tower structure"""

    def __init__(self, U, V, Yu, Yv, A=None, B=None):
        '''
        U, V: the two feature matrices for the two corresponding tower
        Yu: the relationship sparse matrix (from U to V)
        Yv: the relationship sparse matrix (from V to U), can be None if Yu exists
        A, B: the weight matrices for U and V
        '''
        self.U = U
        self.V = V
        self.Yu = csr_matrix(Yu)
        self.Yv = csr_matrix(Yv)
        self.M, self.N = self.Yu.shape
        self.A = np.ones(self.M) if A is None else A
        self.B = np.ones(self.N) if B is None else B
        self.coos = self.Yu.nonzero() ## LTD, check if coos is a pointer
        self.nnz = len(self.coos[0])
        ## LTD, it's very likely that col_nnz have zero values.
        col_nnz = self.Yu.sum(axis=0)
        row_nnz = self.Yu.sum(axis=1)
        assert (row_nnz > 0).all() and (col_nnz > 0).all(), "row_flag:{} col_flag:{}".format((row_nnz > 0).all(), (col_nnz > 0).all())
        self.Ab = self.A / row_nnz.A1
        self.Bb = self.B / col_nnz.A1
        #self.Ab = self.A / row_nnz.sum()
        #self.Bb = self.B / col_nnz.sum()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def generate_batch(data_batch):
        raise NotImplementedError


class NonzeroTextDataset(TwoTowerDataset):
    """Class for text dataset for 2-tower structure"""

    def __init__(self, U, V, Yu, Yv, A=None, B=None):
        super(NonzeroTextDataset, self).__init__(U, V, Yu, Yv, A, B)

    def __len__(self):
        return self.nnz

    def __getitem__(self, index: int):
        coos_x = self.coos[0][index]
        coos_y = self.coos[1][index]
        return {
                'u': self.U[coos_x],  # need to customize
                'v': self.V[coos_y],
                '_a': self.A[coos_x],
                '_b': self.B[coos_y],
                '_ab': self.Ab[coos_x],
                '_bb': self.Bb[coos_y],
                'y': self.Yu[coos_x, coos_y],
                }

    @staticmethod
    def generate_batch(data_batch):
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


class CrossTextDataset(TwoTowerDataset):
    """Class for text dataset for 2-tower structure"""

    def __init__(self, U, V, Yu, Yv, A=None, B=None):
        super(CrossTextDataset, self).__init__(U, V, Yu, Yv, A, B)

    #def __len__(self) -> tuple:
    #    return (self.M, self.N)

    def __getitem__(self, index: tuple):
        coos_x, coos_y = index
        return {
                'us': self.U[coos_x],
                'vs': self.V[coos_y],
                '_as': self.A[coos_x],
                '_bs': self.B[coos_y],
                '_abs': self.Ab[coos_x],
                '_bbs': self.Bb[coos_y],
                'ys': self.Yu[coos_x, :][:, coos_y],
                }

    @staticmethod
    def generate_batch(data_batch):
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
            'Y': spmtx2tensor(data['ys']),
        }


class TextDatasetNew(TwoTowerDataset):
    """Class for text dataset for 2-tower structure"""

    def __init__(self, U, V, Yu, Yv, A=None, B=None):
        super(TextDatasetNew, self).__init__(U, V, Yu, Yv, A, B)

    def __len__(self):
        return self.M

    def __getitem__(self, index: int):
        return {
                'text': torch.LongTensor(self.U[index]),
                'label': torch.FloatTensor(self.Yu[index].todense().A1),
                'index': 0
                }

class CrossRandomBatchSampler(Sampler[Tuple]):
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, bsize_i: int, bsize_j: int, \
            shuffle: bool=True, generator=None) -> None:
        self.data_source = data_source
        self.bsize_i = bsize_i
        self.bsize_j = bsize_j
        self.m, self.n = self.data_source.M, self.data_source.N
        self.nb_i = math.ceil(self.m / self.bsize_i)
        self.nb_j = math.ceil(self.n / self.bsize_j)
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[Tuple]:
        if self.shuffle:
            if self.generator is None:
                ids_i = np.random.permutation(self.m)
                ids_j = np.random.permutation(self.n)
            else:
                ids_i = self.generator.permutation(self.m)
                ids_j = self.generator.permutation(self.n)
        else:
            ids_i = np.arange(self.m)
            ids_j = np.arange(self.n)
        for i in range(self.nb_i):
            for j in range(self.nb_j):
                yield [(ids_i[i*self.bsize_i:(i+1)*self.bsize_i], ids_j[j*self.bsize_j:(j+1)*self.bsize_j])]

    def __len__(self) -> int:
        return self.nb_i * self.nb_j

def _build_mn_matrix(data, classes):
    label_binarizer = MultiLabelBinarizer().fit([classes])
    binary_label_matrix = [coo_matrix(label_binarizer.transform([d['label']])[0]) for d in data]
    return sp_vstack(binary_label_matrix)

def spmtx2tensor(spmtx):
    coo = coo_matrix(spmtx)
    idx = np.vstack((coo.row, coo.col))
    val = coo.data
    idx = torch.LongTensor(idx)
    val = torch.FloatTensor(val)
    shape = coo.shape
    return torch.sparse_coo_tensor(idx, val, torch.Size(shape))

def _data_transform(max_seq_length, data, word_dict, classes):
    U = np.array([[word_dict[word] for word in d['text'][:max_seq_length]] \
            for d in data], dtype=object)
    V = np.arange(len(classes)).reshape(-1, 1)
    Yu = _build_mn_matrix(data, classes)
    Yv = Yu.transpose()
    return U,V, Yu, Yv

def generate_batch(data_batch):
    text_list = [data['text'] for data in data_batch]
    label_list = [data['label'] for data in data_batch]
    return {
        'index': [data['index'] for data in data_batch],
        'text': pad_sequence(text_list, batch_first=True),
        'label': torch.stack(label_list)
    }


def get_dataset_loader(config, data, word_dict, classes, shuffle=False, train=True):
    if train:
        if 'Ori' in config.loss:
            dataset = TextDataset(data, word_dict, classes, config.max_seq_length)
        elif 'Sogram' in config.loss:
            U, V, Yu, Yv = _data_transform(config.max_seq_length, data, word_dict, classes)
            dataset = NonzeroTextDataset(U, V, Yu, Yv)
        else:
            U, V, Yu, Yv = _data_transform(config.max_seq_length, data, word_dict, classes)
            dataset = CrossTextDataset(U, V, Yu, Yv)
    else:
        dataset = TextDataset(data, word_dict, classes, config.max_seq_length)

    if isinstance(dataset, CrossTextDataset):
        dataset_loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=CrossRandomBatchSampler(
                    dataset, shuffle=shuffle,
                    bsize_i=config.batch_size,
                    bsize_j=config.batch_size_j if config.batch_size_j is not None else dataset.N),
                num_workers=0,
                collate_fn=dataset.generate_batch,  # LTD, generate_batch() move to in dataset
                )
    else:
        dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size if train else config.eval_batch_size,
            shuffle=shuffle,
            num_workers=config.data_workers,
            collate_fn=dataset.generate_batch if (train and 'Sogram' in config.loss) else generate_batch,
            pin_memory='cuda' in config.device.type,
        )
    return dataset_loader


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]


def _load_raw_data(path, is_test=False):
    logging.info(f'Load data from {path}.')
    data = pd.read_csv(path, sep='\t', names=['label', 'text'],
                       converters={'label': lambda s: s.split(),
                                   'text': tokenize})
    data = data.reset_index().to_dict('records')
    if not is_test:
        data = [d for d in data if len(d['label']) > 0]
    return data


def load_datasets(config):
    datasets = {}
    test_path = config.test_path or os.path.join(config.data_dir, 'test.txt')
    if config.eval:
        datasets['test'] = _load_raw_data(test_path, is_test=True)
    else:
        if os.path.exists(test_path):
            datasets['test'] = _load_raw_data(test_path, is_test=True)
        train_path = config.train_path or os.path.join(config.data_dir, 'train.txt')
        datasets['train'] = _load_raw_data(train_path)
        val_path = config.val_path or os.path.join(config.data_dir, 'valid.txt')
        if os.path.exists(val_path):
            datasets['val'] = _load_raw_data(val_path)
        else:
            datasets['train'], datasets['val'] = train_test_split(
                datasets['train'], test_size=config.val_size, random_state=42)

    msg = ' / '.join(f'{k}: {len(v)}' for k, v in datasets.items())
    logging.info(f'Finish loading dataset ({msg})')
    return datasets


def load_or_build_text_dict(config, dataset):
    if config.vocab_file:
        logging.info(f'Load vocab from {config.vocab_file}')
        with open(config.vocab_file, 'r') as fp:
            vocab_list = [PAD] + [vocab.strip() for vocab in fp.readlines()]
        vocabs = Vocab(collections.Counter(vocab_list), specials=[UNK],
                       min_freq=1, specials_first=False) # specials_first=False to keep PAD index 0
    else:
        counter = collections.Counter()
        for data in dataset:
            unique_tokens = set(data['text'])
            counter.update(unique_tokens)
        vocabs = Vocab(counter, specials=[PAD, UNK],
                       min_freq=config.min_vocab_freq)
    logging.info(f'Read {len(vocabs)} vocabularies.')

    if os.path.exists(config.embed_file):
        logging.info(f'Load pretrained embedding from file: {config.embed_file}.')
        embedding_weights = get_embedding_weights_from_file(vocabs, config.embed_file, config.silent)
        vocabs.set_vectors(vocabs.stoi, embedding_weights,
                           dim=embedding_weights.shape[1], unk_init=False)
    elif not config.embed_file.isdigit():
        logging.info(f'Load pretrained embedding from torchtext.')
        vocabs.load_vectors(config.embed_file, cache=config.embed_cache_dir)
    else:
        raise NotImplementedError

    return vocabs


def load_or_build_label(config, datasets):
    if config.label_file:
        logging.info('Load labels from {config.label_file}')
        with open(config.label_file, 'r') as fp:
            classes = sorted([s.strip() for s in fp.readlines()])
    else:
        classes = set()
        for dataset in datasets.values():
            for d in tqdm(dataset, disable=config.silent):
                classes.update(d['label'])
        classes = sorted(classes)
    return classes


def get_embedding_weights_from_file(word_dict, embed_file, silent=False):
    """If there is an embedding file, load pretrained word embedding.
    Otherwise, assign a zero vector to that word.
    """

    with open(embed_file) as f:
        word_vectors = f.readlines()

    embed_size = len(word_vectors[0].split())-1
    embedding_weights = [np.zeros(embed_size) for i in range(len(word_dict))]

    vec_counts = 0
    for word_vector in tqdm(word_vectors, disable=silent):
        word, vector = word_vector.rstrip().split(' ', 1)
        vector = np.array(vector.split()).astype(np.float)
        vector = vector / float(np.linalg.norm(vector) + 1e-6)
        embedding_weights[word_dict[word]] = vector
        vec_counts += 1

    logging.info(f'loaded {vec_counts}/{len(word_dict)} word embeddings')

    """ Add UNK embedding.
    Attention xml: np.random.uniform(-1.0, 1.0, emb_size)
    CAML: np.random.randn(embed_size)
    TODO. callback
    """
    unk_vector = np.random.randn(embed_size)
    unk_vector = unk_vector / float(np.linalg.norm(unk_vector) + 1e-6)
    embedding_weights[word_dict[word_dict.UNK]] = unk_vector

    return torch.Tensor(embedding_weights)
