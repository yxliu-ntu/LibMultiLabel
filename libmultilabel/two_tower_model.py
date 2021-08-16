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
        self.tbwriter = SummaryWriter(os.path.join(config.tfboard_log_dir, config.run_name))

        # init loss
        if self.config.loss == 'DPR':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = torch.nn.CrossEntropyLoss(reduction='mean')
            self.step = self._dpr_step
        elif self.config.loss == 'DPR-LRLR':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    loss_func_minus=torch.nn.functional.binary_cross_entropy_with_logits,
                    )
            self.step = self._dpr_lrlrsq_step
        elif self.config.loss == 'DPR-LRSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    loss_func_minus=torch.nn.functional.mse_loss,
                    )
            self.step = self._dpr_lrlrsq_step
        elif self.config.loss == 'Minibatch':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.MinibatchMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    )
            self.step = self._minibatch_step
        elif self.config.loss == 'Sogram':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.SogramMNLoss(
                    self.config.k,
                    self.config.k1,
                    alpha=self.config.alpha,
                    omega=self.config.omega,
                    nnz=self.config.nnz
                    )
            self.step = self._sogram_step
        elif self.config.loss == 'Sogram-Scale':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.SogramMNLoss(
                    self.config.k,
                    self.config.k1,
                    alpha=self.config.alpha,
                    omega=self.config.omega,
                    nnz=self.config.nnz
                    )
            self.step = self._sogram_scale_step
        elif self.config.loss == 'Sogram-Cosine':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.SogramMNLoss(
                    self.config.k,
                    self.config.k1,
                    alpha=self.config.alpha,
                    omega=self.config.omega,
                    nnz=self.config.nnz,
                    loss_func=torch.nn.functional.mse_loss
                    )
            self.step = self._sogram_cosine_step
        else:
            raise

    def _tb_log(self, ps, qs):
        self.tbwriter.add_scalar("ps_mean", ps.norm(dim=1).mean(), self.global_step)
        self.tbwriter.add_scalar("qs_mean", qs.norm(dim=1).mean(), self.global_step)
        #self.logger.experiment.add_scalar("ps_mean", ps.norm(dim=1).mean(), self.global_step)
        #self.logger.experiment.add_scalar("ps_max",  ps.norm(dim=1).max(),  self.global_step)
        #self.logger.experiment.add_scalar("ps_min",  ps.norm(dim=1).min(),  self.global_step)
        #self.logger.experiment.add_scalar("qs_mean", qs.norm(dim=1).mean(), self.global_step)
        #self.logger.experiment.add_scalar("qs_max",  qs.norm(dim=1).max(),  self.global_step)
        #self.logger.experiment.add_scalar("qs_min",  qs.norm(dim=1).min(),  self.global_step)

        return

    def configure_optimizers(self):
        """
        Initialize an optimizer for the free parameters of the network.
        """
        def _get_dpr_optimizer(
                model: torch.nn.Module,
                learning_rate: float = 1e-5,
                adam_eps: float = 1e-8,
                weight_decay: float = 0.0,
                ) -> torch.optim.Optimizer:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                        ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                        ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = optim.AdamW(parameters, lr=learning_rate, eps=adam_eps)
            return optimizer

        def _get_schedule_linear(
                optimizer,
                warmup_steps,
                total_training_steps,
                steps_shift=0,
                last_epoch=-1,
                ):
            """
            Create a schedule with a learning rate that decreases linearly after
            linearly increasing during a warmup period.
            """
            def _lr_lambda(current_step):
                logging.debug(f'current_step:{current_step}')
                current_step += steps_shift
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                        1e-7,
                        float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
                        )
            return LambdaLR(optimizer, _lr_lambda, last_epoch)

        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer_name = self.config.optimizer
        scheduler = None
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(parameters, self.config.learning_rate,
                                  momentum=self.config.momentum,
                                  weight_decay=self.config.weight_decay)
            torch.nn.utils.clip_grad_value_(parameters, 0.5)
        elif optimizer_name == 'adagrad':
            optimizer = optim.Adagrad(self.network.parameters(),
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
        elif optimizer_name == 'adamw-dpr':
            interval = 'step'
            optimizer = _get_dpr_optimizer(
                    self.network,
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    )
            scheduler = _get_schedule_linear(
                    optimizer,
                    self.config.warmup_steps,
                    self.config.total_steps
                    )
        else:
            raise RuntimeError(
                'Unsupported optimizer: {self.config.optimizer}')

        if scheduler is None:
            return optimizer
        else:
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": interval,
                        }
                    }

    def _dpr_step(self, batch, batch_idx):
        #us = (batch['us']).cpu().numpy().flatten()
        #vs = (batch['vs']).cpu().numpy().flatten()
        ps, qs = self.network(batch['us'], batch['vs'])
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = ps @ qs.T
        loss = self.mnloss(logits, ys)
        #logging.debug(f'us: {us}, vs: {vs}')
        #logging.debug(f'ps:{ps.sum().item()}, qs:{qs.sum().item()}')
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _dpr_lrlrsq_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        ys = dense_to_sparse(torch.diag(batch['ys']))
        _as = batch['_as']
        _bs = batch['_bs']
        loss = self.mnloss(ys, _as, _bs, ps, qs, pts, qts)
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _sogram_step(self, batch, batch_idx):
        #us = (batch['us'] - 1).cpu().numpy().flatten() # -1 for ml1m only
        #vs = (batch['vs'] - 1).cpu().numpy().flatten()
        ps, qs = self.network(batch['us'], batch['vs'])
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        ys = batch['ys']
        _as = batch['_as']
        _bs = batch['_bs']
        _abs = batch['_abs']
        _bbs = batch['_bbs']
        loss = self.mnloss(ys, _as, _bs, _abs, _bbs, ps, qs, pts, qts)
        #logging.debug(f'us: {us}, vs: {vs}')
        #sogram_bsize = ps.size()[0]//2
        #logging.debug(f'ps: {ps[sogram_bsize:].sum().item()}, qs: {qs[sogram_bsize:].sum().item()}')
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _sogram_scale_step(self, batch, batch_idx):
        def _embedding_scale(x, scaler=8.0):
            x = x/scaler
            return x
        ps, qs = self.network(batch['us'], batch['vs'])
        self._tb_log(ps.detach(), qs.detach())
        ps = _embedding_scale(ps)
        qs = _embedding_scale(qs)
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        ys = batch['ys']
        _as = batch['_as']
        _bs = batch['_bs']
        _abs = batch['_abs']
        _bbs = batch['_bbs']
        loss = self.mnloss(ys, _as, _bs, _abs, _bbs, ps, qs, pts, qts)
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        return loss

    def _sogram_cosine_step(self, batch, batch_idx):
        def _embedding_norm(x):
            x_norm = torch.norm(x, dim=1, keepdim=True).detach()
            x = x/x_norm
            return x
        ps, qs = self.network(batch['us'], batch['vs'])
        self._tb_log(ps.detach(), qs.detach())
        ps = _embedding_norm(ps)
        qs = _embedding_norm(qs)
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        ys = batch['ys']
        _as = batch['_as']
        _bs = batch['_bs']
        _abs = batch['_abs']
        _bbs = batch['_bbs']
        loss = self.mnloss(ys, _as, _bs, _abs, _bbs, ps, qs, pts, qts)
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        return loss

    def _minibatch_step(self, batch, batch_idx):
        #us = (batch['U'] - 1).cpu().numpy().flatten()
        #vs = (batch['V'] - 1).cpu().numpy().flatten()
        P, Q = self.network(batch['U'], batch['V'])
        Pt = P.new_ones(P.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        Qt = Q.new_ones(Q.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        Y = batch['Y']
        A = batch['A']
        B = batch['B']
        loss = self.mnloss(Y, A, B, P, Q, Pt, Qt)
        #logging.debug(f'us: {us}, vs: {vs}')
        #logging.debug(f'ps: {P.sum().item()}, qs: {Q.sum().item()}')
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        self._tb_log(P.detach(), Q.detach())
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
        P, Q = self.network(batch['U'], batch['V'])
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
        #exit()
        return metric_dict
