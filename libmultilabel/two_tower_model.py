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
        self.tbwriter = SummaryWriter(os.path.join(config.tfboard_log_dir, config.run_name))

        # init loss
        if self.config.loss == 'DPR':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = torch.nn.CrossEntropyLoss(reduction='mean')
            self.step = self._dpr_step
        elif self.config.loss == 'DPR-L2Dist':
            logging.info(f'loss_type: {self.config.loss}')
            #self.mnloss = torch.nn.CrossEntropyLoss(reduction='mean')
            self.step = self._dpr_l2dist_step
        elif self.config.loss == 'DPR-L2Dist-Var1':
            logging.info(f'loss_type: {self.config.loss}')
            #self.mnloss = torch.nn.CrossEntropyLoss(reduction='sum')
            self.step = self._dpr_l2dist_var1_step
        elif self.config.loss == 'DPR-L2Dist-Var2':
            logging.info(f'loss_type: {self.config.loss}')
            #self.mnloss = torch.nn.CrossEntropyLoss(reduction='sum')
            self.step = self._dpr_l2dist_var2_step
        #elif self.config.loss == 'DPR-L2Dist-Var3':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    #self.mnloss = torch.nn.CrossEntropyLoss(reduction='sum')
        #    self.step = self._dpr_l2dist_var3_step
        elif self.config.loss == 'DPR-L2Dist-Exp1':
            logging.info(f'loss_type: {self.config.loss}')
            #self.mnloss = torch.nn.CrossEntropyLoss(reduction='sum')
            self.step = self._dpr_l2dist_exp_step
        elif self.config.loss == 'DPR-L2Dist-Exp2':
            logging.info(f'loss_type: {self.config.loss}')
            #self.mnloss = torch.nn.CrossEntropyLoss(reduction='sum')
            self.step = self._dpr_l2dist_exp2_step
        elif self.config.loss == 'DPR-RankingMSE':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = torch.nn.MSELoss(reduction='sum')
            self.step = self._rankingmse_step
        elif self.config.loss == 'DPR-L2Dist-RankingExp':
            logging.info(f'loss_type: {self.config.loss}')
            self.step = self._l2dist_rankingexp_step
        elif self.config.loss == 'DPR-Triplet':
            logging.info(f'loss_type: {self.config.loss}')
            self.step = self._triplet_step
        #elif self.config.loss == 'DPR-MAEMAE':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.NaiveMNLoss(
        #            omega=self.config.omega,
        #            loss_func_plus=MNLoss.dual_mae_loss,
        #            loss_func_minus=MNLoss.dual_mae_loss,
        #            )
        #    self.step = self._dpr_lrlrsq_step
        #elif self.config.loss == 'DPR-MSEMSE':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.NaiveMNLoss(
        #            omega=self.config.omega,
        #            loss_func_plus=MNLoss.dual_mse_loss,
        #            loss_func_minus=MNLoss.dual_mse_loss,
        #            )
        #    self.step = self._dpr_lrlrsq_step
        elif self.config.loss == 'DPR-Cosine':
            logging.info(f'loss_type: {self.config.loss}')
            assert self.config.imp_r < 1.0 and self.config.imp_r >= -1.0
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    loss_func_plus=MNLoss.dual_mse_loss,
                    loss_func_minus=MNLoss.dual_mse_loss,
                    )
            self.step = self._dpr_cosine_step
        elif self.config.loss == 'DPR-L2Dist-L1H':
            logging.info(f'loss_type: {self.config.loss}')
            self.step = self._dpr_l2dist_l1h_step
        elif self.config.loss == 'DPR-L2Dist-L2H':
            logging.info(f'loss_type: {self.config.loss}')
            self.step = self._dpr_l2dist_l2h_step
        elif self.config.loss == 'DPR-L1HL1H':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    loss_func_plus=MNLoss.l1_hinge_loss,
                    loss_func_minus=MNLoss.l1_hinge_loss,
                    )
            self.step = self._dpr_lrlrsq_step
        elif self.config.loss == 'DPR-L2HL2H':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    loss_func_plus=MNLoss.l2_hinge_loss,
                    loss_func_minus=MNLoss.l2_hinge_loss,
                    )
            self.step = self._dpr_lrlrsq_step
        #elif self.config.loss == 'DPR-MAEL1H':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.NaiveMNLoss(
        #            omega=self.config.omega,
        #            loss_func_plus=MNLoss.dual_mae_loss,
        #            loss_func_minus=MNLoss.l1_hinge_loss,
        #            )
        #    self.step = self._dpr_lrlrsq_step
        #elif self.config.loss == 'DPR-L1HMAE':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.NaiveMNLoss(
        #            omega=self.config.omega,
        #            loss_func_plus=MNLoss.l1_hinge_loss,
        #            loss_func_minus=torch.nn.functional.l1_loss,
        #            )
        #    self.step = self._dpr_lrlrsq_step
        elif self.config.loss == 'DPR-L1HSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    loss_func_plus=MNLoss.l1_hinge_loss,
                    loss_func_minus=torch.nn.functional.mse_loss,
                    )
            self.step = self._dpr_lrlrsq_step
        #elif self.config.loss == 'DPR-SQL2H':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.NaiveMNLoss(
        #            omega=self.config.omega,
        #            loss_func_plus=MNLoss.dual_mse_loss,
        #            loss_func_minus=MNLoss.l2_hinge_loss,
        #            )
        #    self.step = self._dpr_lrlrsq_step
        elif self.config.loss == 'DPR-L2HSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    loss_func_plus=MNLoss.l2_hinge_loss,
                    loss_func_minus=torch.nn.functional.mse_loss,
                    )
            self.step = self._dpr_lrlrsq_step
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
                    #has_bias=True,
                    )
            self.step = self._dpr_lrlrsq_step
        #elif self.config.loss == 'Minibatch':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.MinibatchMNLoss(
        #            omega=self.config.omega,
        #            M=self.config.M,
        #            N=self.config.N,
        #            )
        #    self.step = self._minibatch_step
        #elif self.config.loss == 'Sogram':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.SogramMNLoss(
        #            self.config.k,
        #            self.config.k1,
        #            alpha=self.config.alpha,
        #            omega=self.config.omega,
        #            nnz=self.config.nnz
        #            )
        #    self.step = self._sogram_step
        #elif self.config.loss == 'Sogram-Scale':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.SogramMNLoss(
        #            self.config.k,
        #            self.config.k1,
        #            alpha=self.config.alpha,
        #            omega=self.config.omega,
        #            nnz=self.config.nnz
        #            )
        #    self.step = self._sogram_scale_step
        #elif self.config.loss == 'Sogram-Cosine':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    assert self.config.imp_r < 1.0 and self.config.imp_r >= -1.0
        #    self.mnloss = MNLoss.SogramMNLoss(
        #            self.config.k,
        #            self.config.k1,
        #            alpha=self.config.alpha,
        #            omega=self.config.omega,
        #            nnz=self.config.nnz,
        #            loss_func=torch.nn.functional.mse_loss
        #            )
        #    self.step = self._sogram_cosine_step
        else:
            raise

    def _tb_log(self, ps, qs):
        logits = ps @ qs.T
        neg_mask = 1 - torch.eye(logits.size()[0])
        pos = torch.diagonal(logits)
        neg = torch.masked_select(logits, neg_mask.bool())
        self.tbwriter.add_scalar("pos_mean", pos.mean(), self.global_step)
        self.tbwriter.add_scalar("neg_mean", neg.mean(), self.global_step)

        pos_neg_diff = pos.reshape(-1, 1) - logits
        pos_neg_diff = pos_neg_diff.sum(dim=-1) / (pos_neg_diff.size()[-1] - 1.)
        pos_neg_diff = (pos_neg_diff / pos).mean()
        self.tbwriter.add_scalar("pos_neg_diff", pos_neg_diff, self.global_step)

        #self.tbwriter.add_scalar("ps_mean", ps.norm(dim=1).mean(), self.global_step)
        #self.tbwriter.add_scalar("qs_mean", qs.norm(dim=1).mean(), self.global_step)
        #self.logger.experiment.add_scalar("ps_mean", ps.norm(dim=1).mean(), self.global_step)
        #self.logger.experiment.add_scalar("ps_max",  ps.norm(dim=1).max(),  self.global_step)
        #self.logger.experiment.add_scalar("ps_min",  ps.norm(dim=1).min(),  self.global_step)
        #self.logger.experiment.add_scalar("qs_mean", qs.norm(dim=1).mean(), self.global_step)
        #self.logger.experiment.add_scalar("qs_max",  qs.norm(dim=1).max(),  self.global_step)
        #self.logger.experiment.add_scalar("qs_min",  qs.norm(dim=1).min(),  self.global_step)

        return

    def _embedding_norm(self, x):
        x_norm = torch.norm(x, dim=1, keepdim=True).detach()
        x = x/x_norm
        return x

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
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
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

    def _dpr_l2dist_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        #loss = self.mnloss(logits, ys)
        ploss = -torch.diagonal(logits).sum()
        nloss = torch.logsumexp(logits, -1).sum()
        loss = ploss + nloss
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}, ploss: {ploss.item()}, nloss: {nloss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _dpr_l2dist_l1h_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        mask = (1.0 - torch.diag(batch['ys'])).bool() # mask for negative pairs
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        ploss = -torch.diagonal(logits).sum()
        nlogits = torch.masked_select(logits, mask)
        nloss = torch.maximum(self.config.margin + nlogits, torch.zeros_like(nlogits)).sum()
        loss = ploss + nloss
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}, ploss: {ploss.item()}, nloss: {nloss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _dpr_l2dist_l2h_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        mask = (1.0 - torch.diag(batch['ys'])).bool() # mask for negative pairs
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        ploss = -torch.diagonal(logits).sum()
        nlogits = torch.masked_select(logits, mask)
        nloss = torch.maximum(self.config.margin + nlogits, torch.zeros_like(nlogits)).pow(2).sum()
        loss = ploss + nloss
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}, ploss: {ploss.item()}, nloss: {nloss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _dpr_l2dist_var1_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        plogits = torch.diagonal(logits)  # -||p-q^+||^2
        ploss = -plogits.sum()
        nloss = self.config.omega * ps.shape[0] * torch.logsumexp(logits.reshape(1, -1), -1)
        loss = ploss + nloss
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}, ploss: {ploss.item()}, nloss: {nloss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _dpr_l2dist_var2_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        plogits = torch.diagonal(logits)  # -||p-q^+||^2
        ploss = -plogits.sum() #- self.config.omega * self.config.M * torch.logsumexp(plogits.reshape(1,-1), -1)
        nloss = self.config.omega * ps.shape[0] * torch.logsumexp((logits - torch.diag(plogits)).reshape(1, -1), -1)
        loss = ploss + nloss
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}, ploss: {ploss.item()}, nloss: {nloss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _dpr_l2dist_var3_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        plogits = torch.diagonal(logits)  # -||p-q^+||^2
        ploss = -plogits.sum() - self.config.omega * ps.shape[0] * torch.logsumexp(plogits.reshape(1,-1), -1)
        nloss = self.config.omega * ps.shape[0] * torch.logsumexp(logits.reshape(1, -1), -1)
        loss = ploss + nloss
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}, ploss: {ploss.item()}, nloss: {nloss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _dpr_l2dist_exp_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        plogits = torch.diagonal(logits)  # -||p-q^+||^2
        ploss = -self.config.nu * plogits.sum() - self.config.N / qs.shape[0] * torch.exp(self.config.nu * plogits).sum()
        nloss = self.config.N / qs.shape[0] * torch.exp(self.config.nu * logits).sum()
        loss = ploss + nloss
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}, ploss: {ploss.item()}, nloss: {nloss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _dpr_l2dist_exp2_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        plogits = torch.diagonal(logits)  # -||p-q^+||^2
        ploss = -plogits.sum() - self.config.N / qs.shape[0] * torch.exp(self.config.nu * plogits).sum()
        nloss = self.config.N / qs.shape[0] * torch.exp(self.config.nu * logits).sum()
        loss = ploss + nloss
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}, ploss: {ploss.item()}, nloss: {nloss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _rankingmse_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        ys = 1.0 - torch.diag(batch['ys']) # negative pairs regress to 1 while pos pairs regress to 0
        logits = ps @ qs.T
        diffs = torch.diagonal(logits).reshape(-1, 1) - logits # pos - neg
        loss = self.mnloss(diffs, ys*self.config.margin)
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _l2dist_rankingexp_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        mask = (1.0 - torch.diag(batch['ys'])).bool() # mask for negative pairs
        logits = -(torch.cdist(ps.contiguous(), qs.contiguous(), p=2)**2)  # -||p-q||^2
        diffs = torch.diagonal(logits).reshape(-1, 1) - logits # pos - neg 
        diffs = torch.masked_select(diffs, mask)
        loss = torch.logsumexp(-self.config.nu * diffs.reshape(1, -1), -1)
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        self._tb_log(ps.detach(), qs.detach())
        return loss

    def _triplet_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        mask = (1.0 - torch.diag(batch['ys'])).bool() # mask for negative pairs
        logits = ps @ qs.T
        diffs = logits - torch.diagonal(logits).reshape(-1, 1) # neg - pos
        diffs = torch.masked_select(diffs, mask)
        loss = torch.maximum(diffs + self.config.margin, torch.zeros_like(diffs)).sum()
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

    def _dpr_cosine_step(self, batch, batch_idx):
        ps, qs = self.network(batch['us'], batch['vs'])
        self._tb_log(ps.detach(), qs.detach())
        ps = self._embedding_norm(ps)
        qs = self._embedding_norm(qs)
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        ys = dense_to_sparse(torch.diag(batch['ys']))
        _as = batch['_as']
        _bs = batch['_bs']
        loss = self.mnloss(ys, _as, _bs, ps, qs, pts, qts)
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
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
        ps, qs = self.network(batch['us'], batch['vs'])
        self._tb_log(ps.detach(), qs.detach())
        ps = self._embedding_norm(ps)
        qs = self._embedding_norm(qs)
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
        if 'cosine' in self.config.loss.lower():
            P = self._embedding_norm(P) if P is not None else None
            Q = self._embedding_norm(Q) if Q is not None else None
        if 'scale' in self.config.loss.lower():
            raise ValueError
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
                if 'l2dist' in self.config.loss.lower():
                    score_mat[:, j_start:j_end] = -cdist(P[i_start:i_end], Q[j_start:j_end])**2
                else:
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
