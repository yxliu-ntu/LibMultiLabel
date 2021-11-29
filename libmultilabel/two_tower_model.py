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
from scipy.spatial.distance import cdist


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
            self.mnloss = torch.nn.BCEWithLogitsLoss(reduction='sum')
            self.step = self._linearlr_step
        elif self.config.loss == 'Naive-LRLR':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    loss_func_minus=torch.nn.functional.binary_cross_entropy_with_logits,
                    )
            self.step = self._lrlrsq_step
        elif self.config.loss == 'Naive-LRSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
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

    def _l2norm(self, x):
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
        return torch.div(x, x_norm)

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
            torch.nn.utils.clip_grad_value_(parameters, 0.5)
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
            torch.nn.utils.clip_grad_value_(parameters, 0.5)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(parameters,
                                    weight_decay=self.config.weight_decay,
                                    lr=self.config.learning_rate)
            torch.nn.utils.clip_grad_value_(parameters, 0.5)
        else:
            raise RuntimeError(
                'Unsupported optimizer: {self.config.optimizer}')

        return optimizer

    def _l2_reg(self, l2_lambda):
        l2_reg = torch.tensor(0.)
        if l2_lambda > 0:
            for param in self.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param)**2
            return 0.5 * l2_lambda * l2_reg
        else:
            return 0.0

    def _logsoftmax_step(self, batch, batch_idx):
        raise NotImplementedError
        #ps, qs = self.network(batch['us'], batch['vs'])
        #ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        #logits = ps @ qs.T
        #loss = self.mnloss(logits, ys)
        #logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        #return loss

    def _lrlrsq_step(self, batch, batch_idx):
        raise NotImplementedError
        #ps, qs = self.network(batch['us'], batch['vs'], batch['uvals'], batch['vvals'])
        #pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        #qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        #ys = batch['ys'] #dense_to_sparse(torch.diag(batch['ys']).detach())
        #_as = batch['_as']
        #_bs = batch['_bs']
        #loss = self.mnloss(ys, _as, _bs, ps, qs, pts, qts)
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

    def _linearlr_step(self, batch, batch_idx):
        us, vs, uvals, vvals = batch['us'], batch['vs'], batch['uvals'], batch['vvals']
        P, Q = self.network(us, vs, uvals, vvals)
        pqs = P.sum(dim=-1, keepdim=True) + Q.sum(dim=-1) # outer sum
        if self.config.isl2norm:
            pq_norms = uvals.sum(dim=-1, keepdim=True) + vvals.sum(dim=-1) # outer sum
            pq_norms = pq_norms ** 0.5
            logits = torch.div(pqs, pq_norms.detach())
        Y = batch['ys'].to_dense()
        loss = self.mnloss(logits, Y) + self._l2_reg(self.config.l2_lambda)
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        print(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        return loss

    def _minibatch_step(self, batch, batch_idx):
        #print(batch['us'][:10, :], batch['vs'][:10, :], batch['uvals'][:10, :], batch['vvals'][:10, :])
        #print(batch['us'].shape, batch['vs'].shape, batch['uvals'].shape, batch['vvals'].shape)
        #print(batch['us'].sum(), batch['vs'].sum(), batch['uvals'].sum(), batch['vvals'].sum())
        us, vs, uvals, vvals = batch['us'], batch['vs'], batch['uvals'], batch['vvals']
        if self.config.isl2norm:
            uvals = self._l2norm(uvals)
            vvals = self._l2norm(vvals)
        P, Q = self.network(us, vs, uvals, vvals)
        print(P.shape, Q.shape)
        print(P.sum(), Q.sum())
        Pt = P.new_ones(P.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        Qt = Q.new_ones(Q.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        Y = batch['ys']
        A = batch['_as']
        B = batch['_bs']
        loss = self.mnloss(Y, A, B, P, Q, Pt, Qt) + self._l2_reg(self.config.l2_lambda)
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        #print(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
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
        P, Q = self.network(batch['us'], batch['vs'], batch['uvals'], batch['vvals'])
        return {
                'P': P.detach().cpu().numpy() if P is not None else None, 
                'Q': Q.detach().cpu().numpy() if Q is not None else None,
                }

    def _shared_eval_step_end(self, batch_parts):
        P = batch_parts['P'] if batch_parts['P'] is not None else None
        Q = batch_parts['Q'] if batch_parts['Q'] is not None else None
        return P, Q

    def _shared_eval_epoch_end(self, step_outputs, split):
        P, Q = zip(*step_outputs)
        P = np.vstack([_data for _data in P if _data is not None])
        Q = np.vstack([_data for _data in Q if _data is not None])
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
            self.eval_metric.update(target, score_mat)
        metric_dict = self.eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        dump_log(config=self.config, metrics=metric_dict, split=split)

        if not self.config.silent and (not self.trainer or self.trainer.is_global_zero):
            print(f'====== {split} dataset evaluation result =======')
            print(self.eval_metric)
            print('')
            logging.debug(f'{split} dataset evaluation result:\n{self.eval_metric}')
        self.eval_metric.reset()

        return metric_dict
