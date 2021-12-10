import os
import logging
import numpy as np
import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from abc import abstractmethod
from argparse import Namespace
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities.parsing import AttributeDict
from . import networks
from . import MNLoss
from .metrics import MultiLabelMetrics
from .utils import dump_log, argsort_top_k, dense_to_sparse
from .data_utils import spmtx2tensor
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize


class TwoTowerModel(pl.LightningModule):
    """Concrete class handling Pytorch Lightning training flow"""
    def __init__(self, config, Y_eval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if isinstance(config, Namespace):
            config = vars(config)
        if isinstance(config, dict):
            config = AttributeDict(config)
        self.config = self.hparams.config
        self.Y_eval = self.hparams.Y_eval
        self.eval_metric = MultiLabelMetrics(self.config)
        self.network = getattr(networks, self.config.model_name)(self.config)
        #self.tbwriter = SummaryWriter(os.path.join(config.tfboard_log_dir, config.run_name))

        # init loss
        if self.config.loss == 'Naive-LogSoftmax':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = torch.nn.CrossEntropyLoss(reduction='sum')
            self.step = self._logsoftmax_step
        elif self.config.loss == 'Linear-LR':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = self._weighted_lrloss
            self.step = self._linearlr_step
        elif self.config.loss == 'Naive-LRLR':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    loss_func_minus=torch.nn.functional.binary_cross_entropy_with_logits,
                    )
            self.step = self._lrlrsq_step
        elif self.config.loss == 'Naive-LRSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    )
            self.step = self._lrlrsq_step
        elif self.config.loss == 'Minibatch-LRSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.MinibatchMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    )
            self.step = self._minibatch_step
        elif self.config.loss == 'Sogram-LRSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.SogramMNLoss(
                    self.config.k,
                    self.config.k1,
                    alpha=self.config.alpha,
                    omega=self.config.omega,
                    nnz=self.config.nnz
                    )
            self.step = self._sogram_step
        else:
            raise

    def configure_optimizers(self):
        """
        Initialize an optimizer for the free parameters of the network.
        """

        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer_name = self.config.optimizer
        scheduler = None
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(parameters, self.config.learning_rate,
                                  momentum=self.config.momentum,
                                  weight_decay=self.config.weight_decay)
            #torch.nn.utils.clip_grad_value_(parameters, 0.5)
        elif optimizer_name == 'adagrad':
            optimizer = optim.Adagrad(parameters,
                                   weight_decay=self.config.weight_decay,
                                   lr=self.config.learning_rate,
                                   initial_accumulator_value=0.1,
                                   eps=1e-7)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(parameters,
                                   weight_decay=self.config.weight_decay,
                                   lr=self.config.learning_rate)
            #torch.nn.utils.clip_grad_value_(parameters, 0.5)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(parameters,
                                    weight_decay=self.config.weight_decay,
                                    lr=self.config.learning_rate)
            #torch.nn.utils.clip_grad_value_(parameters, 0.5)
        else:
            raise RuntimeError(
                'Unsupported optimizer: {self.config.optimizer}')

        return optimizer

    def _gnorm(self):
        gnorm = torch.tensor(0.)
        for param in self.parameters():
            if param.requires_grad:
                gnorm += torch.norm(param.grad)**2
        return gnorm**0.5

    def _wnorm_sq(self):
        wnorm_sq = torch.tensor(0.)
        for param in self.parameters():
            if param.requires_grad:
                wnorm_sq += torch.norm(param)**2
        return wnorm_sq

    def _logsoftmax_step(self, batch, batch_idx):
        raise NotImplementedError
        #ps, qs = self.network(batch['us'], batch['vs'])
        #amp = self.config.M * self.config.N / ps.shape[0] / qs.shape[0]
        #ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        #logits = ps @ qs.T
        #loss = amp*self.mnloss(logits, ys)
        #logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        #return loss

    def _lrlrsq_step(self, batch, batch_idx):
        raise NotImplementedError
        #ps, qs = self.network(batch['us'], batch['vs'], batch['uvals'], batch['vvals'])
        #amp = self.config.M * self.config.N / ps.shape[0] / qs.shape[0]
        #pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        #qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        #ys = batch['ys'] #dense_to_sparse(torch.diag(batch['ys']).detach())
        #_as = batch['_as']
        #_bs = batch['_bs']
        #loss = amp*self.mnloss(ys, _as, _bs, ps, qs, pts, qts)
        #logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        #return loss

    def _sogram_step(self, batch, batch_idx):
        raise NotImplementedError
        #ps, qs = self.network(batch['us'], batch['vs'], batch['uvals'], batch['vvals'])
        #pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        #qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        #ys = batch['ys']
        #_as = batch['_as']
        #_bs = batch['_bs']
        #_abs = batch['_abs']
        #_bbs = batch['_bbs']
        #loss = self.mnloss(ys, _as, _bs, _abs, _bbs, ps, qs, pts, qts)
        #logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        #return loss

    def _weighted_lrloss(self, logits, Y):
        _lrloss = F.binary_cross_entropy_with_logits
        coos = Y._indices()
        logits_pos = logits[coos[0], coos[1]]
        L_plus_part1 = _lrloss(logits_pos, (Y._values() > 0).to(logits_pos.dtype), reduction='none')
        L_plus_part2 = _lrloss(logits_pos, logits_pos.new_zeros(logits_pos.size()), reduction='none')
        L_plus = L_plus_part1 - self.config.omega * L_plus_part2
        L_minus = _lrloss(logits, logits.new_zeros(logits.size()), reduction='none')
        return L_plus.sum() + self.config.omega * L_minus.sum()

    def _linearlr_step(self, batch, batch_idx):
        us, vs = batch['us'], batch['vs']
        Y = batch['ys']

        ps, us_norm_sq, qs, vs_norm_sq = self.network(us, vs)
        logits = ps.sum(dim=-1, keepdim=True) + qs.sum(dim=-1) # outer sum
        if self.config.isl2norm:
            uv_norm = (us_norm_sq.unsqueeze(dim=-1) + vs_norm_sq) ** 0.5 # outer sum
            logits = torch.div(logits, uv_norm)

        amp = self.config.M * self.config.N / ps.shape[0] / qs.shape[0]
        loss = (amp*self.mnloss(logits, Y) + 0.5 * self.config.l2_lambda * self._wnorm_sq()) / self.config.l2_lambda
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        #print(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        return loss

    def _minibatch_step(self, batch, batch_idx):
        us, vs = batch['us'], batch['vs']
        Y = batch['ys']
        A = batch['_as']
        B = batch['_bs']

        P, Q = self.network(us, vs)
        Pt = P.new_ones(P.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        Qt = Q.new_ones(Q.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        loss = self.mnloss(Y, A, B, P, Q, Pt, Qt) + 0.5 * self.config.l2_lambda * self._wnorm_sq()
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        #print(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        return loss

    def _calc_func_val(self, bsize_i=4096, bsize_j=65536):
        opt = self.optimizers()
        Utr = spmtx2tensor(self.trainer.train_dataloader.dataset.datasets.U)
        Vtr = spmtx2tensor(self.trainer.train_dataloader.dataset.datasets.V)
        Atr = torch.FloatTensor(self.trainer.train_dataloader.dataset.datasets.A)
        Btr = torch.FloatTensor(self.trainer.train_dataloader.dataset.datasets.B)
        Ytr = self.trainer.train_dataloader.dataset.datasets.Yu
        m, n = Utr.shape[0], Vtr.shape[0]
        segment_m = math.ceil(m/bsize_i)
        segment_n = math.ceil(n/bsize_j)

        P, Unorm_sq, Q, Vnorm_sq = None, None, None, None
        if self.config.loss.startswith('Linear-LR'):
            P, Unorm_sq, Q, Vnorm_sq = self.network(Utr, Vtr)
        else:
            P, Q = self.network(Utr, Vtr)
            Pt = P.new_ones(P.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
            Qt = Q.new_ones(Q.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)

        func_val = torch.tensor(0.)
        for i in range(segment_m):
            i_start, i_end = i*bsize_i, min((i+1)*bsize_i, m)
            logits = torch.zeros(i_end-i_start, n)
            target = spmtx2tensor(Ytr[i_start:i_end])
            for j in range(segment_n):
                j_start, j_end = j*bsize_j, min((j+1)*bsize_j, n)
                if self.config.loss.startswith('Linear-LR'):
                    logits[:, j_start:j_end] = P[i_start:i_end].sum(dim=-1, keepdim=True) + Q[j_start:j_end].sum(dim=-1)
                    if self.config.isl2norm:
                        uv_norm = (Unorm_sq[i_start:i_end].unsqueeze(dim=-1) + Vnorm_sq[j_start:j_end]) ** 0.5
                        logits[:, j_start:j_end] = torch.div(logits[:, j_start:j_end], uv_norm)
                    func_val = func_val + self.mnloss(logits, target)
                else:
                    func_val = func_val + self.mnloss(
                            target,
                            Atr[i_start:i_end],
                            Btr[j_start:j_end],
                            P[i_start:i_end],
                            Q[j_start:j_end],
                            Pt[i_start:i_end],
                            Qt[j_start:j_end]
                            )
        func_val = (func_val + 0.5 * self.config.l2_lambda * self._wnorm_sq()) / self.config.l2_lambda
        gnorm = self._gnorm()
        full_batch_msg = (f'epoch: {self.current_epoch}, gnorm: {gnorm.item():.4e}, func_val: {func_val.item():.4e}')
        logging.debug(full_batch_msg)
        opt.zero_grad()
        return full_batch_msg

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return loss

    def training_epoch_end(self, training_step_outputs):
        if self.config.check_func_val:
            self.full_batch_msg = self._calc_func_val()
        return

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def validation_step_end(self, batch_parts):
        return self._shared_eval_step_end(batch_parts)

    def validation_epoch_end(self, step_outputs):
        return self._shared_eval_epoch_end(step_outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step_end(self, batch_parts):
        return self._shared_eval_step_end(batch_parts)

    def test_epoch_end(self, step_outputs):
        return self._shared_eval_epoch_end(step_outputs, 'test')

    def _shared_eval_step(self, batch, batch_idx):
        P, Unorm_sq, Q, Vnorm_sq = None, None, None, None
        if self.config.loss.startswith('Linear-LR'):
            P, Unorm_sq, Q, Vnorm_sq = self.network(batch['us'], batch['vs'])
        else:
            P, Q = self.network(batch['us'], batch['vs'])
        return {
                'P': P.detach().cpu().numpy() if P is not None else None, 
                'Q': Q.detach().cpu().numpy() if Q is not None else None,
                'Unorm_sq': Unorm_sq.detach().cpu().numpy() if Unorm_sq is not None else None, 
                'Vnorm_sq': Vnorm_sq.detach().cpu().numpy() if Vnorm_sq is not None else None,
                }

    def _shared_eval_step_end(self, batch_parts):
        P = batch_parts['P']
        Q = batch_parts['Q']
        Unorm_sq = batch_parts['Unorm_sq']
        Vnorm_sq = batch_parts['Vnorm_sq']
        return P, Q, Unorm_sq, Vnorm_sq

    def _shared_eval_epoch_end(self, step_outputs, split):
        P, Q, Unorm_sq, Vnorm_sq = zip(*step_outputs)
        P = np.vstack([_data for _data in P if _data is not None])
        Q = np.vstack([_data for _data in Q if _data is not None])
        if self.config.loss.startswith('Linear-LR') and self.config.isl2norm:
            Unorm_sq = np.vstack([_data for _data in Unorm_sq if _data is not None])
            Vnorm_sq = np.vstack([_data for _data in Vnorm_sq if _data is not None])

        m, n = P.shape[0], Q.shape[0]
        bsize_i = self.config.eval_bsize_i
        bsize_j = self.config.eval_bsize_j if self.config.eval_bsize_j is not None else n
        segment_m = math.ceil(m/bsize_i)
        segment_n = math.ceil(n/bsize_j)
        for i in range(segment_m):
            i_start, i_end = i*bsize_i, min((i+1)*bsize_i, m)
            score_mat = np.zeros((i_end-i_start, n))
            target = self.Y_eval[i_start:i_end].todense()
            for j in range(segment_n):
                j_start, j_end = j*bsize_j, min((j+1)*bsize_j, n)
                if self.config.loss.startswith('Linear-LR'):
                    score_mat[:, j_start:j_end] = np.add.outer(P[i_start:i_end].sum(axis=-1), Q[j_start:j_end].sum(axis=-1))
                    if self.config.isl2norm:
                        uv_norm = np.add.outer(Unorm_sq[i_start:i_end].sum(axis=-1), Vnorm_sq[j_start:j_end].sum(axis=-1)) ** 0.5
                        score_mat[:, j_start:j_end] = score_mat[:, j_start:j_end] / uv_norm
                else:
                    score_mat[:, j_start:j_end] = P[i_start:i_end].dot(Q[j_start:j_end].T)
            self.eval_metric.update(target, score_mat)
        metric_dict = self.eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        dump_log(config=self.config, metrics=metric_dict, split=split)

        if self.config.check_func_val:
            print(self.full_batch_msg)

        if not self.config.silent and (not self.trainer or self.trainer.is_global_zero):
            print(f'====== {split} dataset evaluation result =======')
            print(self.eval_metric)
            print('')
            logging.debug(f'{split} dataset evaluation result:\n{self.eval_metric}')
        self.eval_metric.reset()

        return metric_dict
