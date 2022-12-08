import os
import time
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
from .data_utils import spmtx2tensor, gen_skip_mask, MASK_MIN
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

class LogExpMNLoss(torch.nn.Module):
    def __init__(self,
            M: int,
            N: int,
            loss_type: str,
            omega: int=1.0,
            ):
        super(LogExpMNLoss, self).__init__()
        self.omega = omega
        self.M = M
        self.N = N
        assert loss_type in ['OVR', 'PAL', 'nOVR', 'nPAL']
        self.loss_type = loss_type

    def _weighted_logsumexp(self, value, weight, dim):
        assert dim in [0, 1, -1]
        assert len(value.shape) == 2 and len(weight.shape) == 1
        weight = weight.unsqueeze(1) if dim == 0 else weight.unsqueeze(0) # dim=1,-1 (1, n) or dim=0 (m, 1)
        eps = 1e-20
        _c, _ = torch.max(value, dim=dim, keepdim=True) # dim=1,-1 (m, 1) or dim=0 (1, n)
        return _c.squeeze(dim) +\
                torch.log(torch.sum(torch.exp(value - _c)*weight, dim=dim) + eps)

    def forward(self, Y, A, B, Ab, Bb, P, Q):
        coos = Y._indices()
        Y_hat = P @ Q.T
        Y_hat_pos = Y_hat[coos[0], coos[1]]

        assert self.omega == 1.0, "Tuning omega is not supported yet."
        if self.loss_type == 'OVR':
            loss = - (Y_hat_pos.sum() + self.omega*torch.einsum('i,j,ij', A, B, F.logsigmoid(-Y_hat)))
        elif self.loss_type == 'nOVR':
            loss = - (Y_hat_pos.sum() + self.omega*torch.einsum('i,j,ij', A/Ab, B/Bb, F.logsigmoid(-Y_hat)))
        elif self.loss_type == 'PAL':
            loss = - (Y_hat_pos.sum() - self.omega*(A/Ab*self._weighted_logsumexp(Y_hat, B, dim=-1)).sum())
        else:
            loss = - (Y_hat_pos.sum() - self.omega*(A/Ab*self._weighted_logsumexp(Y_hat, B/Bb, dim=-1)).sum())
        return loss

class TwoTowerModel(pl.LightningModule):
    """Concrete class handling Pytorch Lightning training flow"""
    def __init__(self, config, Y_eval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False

        if isinstance(config, Namespace):
            config = vars(config)
        if isinstance(config, dict):
            config = AttributeDict(config)
        self.config = self.hparams.config
        self.Y_eval = self.hparams.Y_eval
        self.eval_metric = MultiLabelMetrics(self.config)
        self.network = getattr(networks, self.config.model_name)(self.config)
        self.tr_time = 0.0
        #self.tbwriter = SummaryWriter(os.path.join(config.tfboard_log_dir, config.run_name))

        # init loss
        if self.config.loss == 'Cross-PAL':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = LogExpMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    loss_type='PAL',
                    )
            self.step = self._trans_step
        elif self.config.loss == 'NonZero-PAL':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = LogExpMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    loss_type='nPAL',
                    )
            self.step = self._trans_step
        elif self.config.loss == 'Cross-OVR':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = LogExpMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    loss_type='OVR',
                    )
            self.step = self._trans_step
        elif self.config.loss == 'NonZero-OVR':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = LogExpMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    loss_type='nOVR',
                    )
            self.step = self._trans_step
        #elif self.config.loss == 'Naive-LRLR':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.NaiveMNLoss(
        #            omega=self.config.omega,
        #            M=self.config.M,
        #            N=self.config.N,
        #            loss_func_minus=torch.nn.functional.binary_cross_entropy_with_logits,
        #            )
        #    self.step = self._naive_step
        #elif self.config.loss == 'Naive-LRSQ':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.NaiveMNLoss(
        #            omega=self.config.omega,
        #            M=self.config.M,
        #            N=self.config.N,
        #            )
        #    self.step = self._naive_step
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
            self.fmnloss = MNLoss.SogramMNLoss(
                    self.config.k,
                    self.config.k1,
                    alpha=self.config.alpha,
                    omega=self.config.omega,
                    nnz=self.config.nnz
                    )
            self.mnloss = MNLoss.MinibatchMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
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
                                   eps=self.config.eps)
            #print('opt:%s'%optimizer_name, 'eps:%f'%self.config.eps)
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
        ps, qs = self.network(batch['us'], batch['vs'])
        amp = self.config.M * self.config.N / ps.shape[0] / qs.shape[0]
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = ps @ qs.T
        loss = amp*self.mnloss(logits, ys)
        return loss

    def _trans_step(self, batch, batch_idx):
        us, vs = batch['us'], batch['vs']
        ys = batch['ys']
        _as = batch['_as']
        _bs = batch['_bs']
        _abs = batch['_abs']
        _bbs = batch['_bbs']

        ps, qs = self.network(us, vs)
        loss = self.mnloss(ys, _as, _bs, _abs, _bbs, ps, qs)
        return loss

    def _naive_step(self, batch, batch_idx):
        us, vs = batch['us'], batch['vs']
        ys = batch['ys']
        _as = batch['_as']
        _bs = batch['_bs']

        ps, qs = self.network(batch['us'], batch['vs'])
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        loss = self.mnloss(ys, _as, _bs, ps, qs, pts, qts, isscaling=batch_idx>0)
        return loss

    def _sogram_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])#, batch['uvals'], batch['vvals'])
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        ys = batch['ys']
        _as = batch['_as']
        _bs = batch['_bs']
        _abs = batch['_abs']
        _bbs = batch['_bbs']
        loss = self.fmnloss(ys, _as, _bs, _abs, _bbs, ps, qs, pts, qts)
        return loss

    def _minibatch_step(self, batch, batch_idx):
        us, vs = batch['us'], batch['vs']
        Y = batch['ys']
        A = batch['_as']
        B = batch['_bs']

        P, Q = self.network(us, vs)
        Pt = P.new_ones(P.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        Qt = Q.new_ones(Q.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        loss = self.mnloss(Y, A, B, P, Q, Pt, Qt)
        return loss

    def _calc_func_val(self, bsize_i=4096, bsize_j=65536):
        with torch.enable_grad():
            trainset = self.trainer.train_dataloader.dataset.datasets
            Atr = torch.Tensor(trainset.A)
            Btr = torch.Tensor(trainset.B)
            Abtr = torch.Tensor(trainset.Ab)
            Bbtr = torch.Tensor(trainset.Bb)
            Ytr = trainset.Yu
            m, n = Ytr.shape
            segment_m = math.ceil(m/bsize_i)
            segment_n = math.ceil(n/bsize_j)

            func_val = torch.tensor(0.)
            for i in range(segment_m):
                i_start, i_end = i*bsize_i, min((i+1)*bsize_i, m)
                Utr = spmtx2tensor(trainset.U[i_start:i_end])
                for j in range(segment_n):
                    j_start, j_end = j*bsize_j, min((j+1)*bsize_j, n)
                    Vtr = spmtx2tensor(trainset.V[j_start:j_end])
                    target = spmtx2tensor(Ytr[i_start:i_end, j_start:j_end])
                    _batch = {
                            'ys': target,
                            'us': Utr,
                            'vs': Vtr,
                            '_as':Atr[i_start:i_end],
                            '_bs':Btr[j_start:j_end],
                            '_abs':Abtr[i_start:i_end],
                            '_bbs':Bbtr[j_start:j_end],
                            }
                    func_val = func_val + self.step(_batch, -1)
            if self.config.l2_lambda > 0.:
                func_val = func_val + 0.5 * self.config.l2_lambda * self._wnorm_sq()

            opt = self.optimizers()
            self.manual_backward(func_val)
            gnorm = self._gnorm()
            opt.zero_grad()

        msg = ('global_step: {}, epoch: {}, training_time: {:.3f}, gnorm: {:.3f}, func_val: {:.3f}'.format(
            self.global_step,
            self.current_epoch,
            self.tr_time,
            gnorm.item(),
            func_val.item()
            ))
        logging.debug(msg)
        print(msg)
        return

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        start_time = time.time()
        loss = self.step(batch, batch_idx)
        if self.config.l2_lambda > 0.:
            loss += 0.5 * self.config.l2_lambda * self._wnorm_sq()
        self.manual_backward(loss)
        opt.step()
        self.tr_time += time.time() - start_time
        logging.warning(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        return loss

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
        skip_mask = self.config['%s_skip_mask'%split]

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
                score_mat[:, j_start:j_end] = P[i_start:i_end].dot(Q[j_start:j_end].T)
            if skip_mask is not None:
                score_mat[skip_mask[i_start:i_end].nonzero()] = MASK_MIN
            self.eval_metric.update(target, score_mat)
        metric_dict = self.eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        #dump_log(config=self.config, metrics=metric_dict, split=split)

        if self.config.check_func_val and split == 'val' and self.config.bratio < 1.:
            self._calc_func_val()

        if not self.config.silent and (not self.trainer or self.trainer.is_global_zero):
            logging.warning(split + ' ' +','.join('%s:%.3f'%(_k, metric_dict[_k]*100) for _k in metric_dict))

        self.eval_metric.reset()

        return metric_dict
