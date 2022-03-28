import os
import time
import logging
import numpy as np
import scipy as sp
import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from abc import abstractmethod
from argparse import Namespace
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.autograd.functional import jacobian
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
        self.automatic_optimization = False

        if isinstance(config, Namespace):
            config = vars(config)
        if isinstance(config, dict):
            config = AttributeDict(config)
        self.config = self.hparams.config
        self.Y_eval = self.hparams.Y_eval
        self.eval_metric = MultiLabelMetrics(self.config)
        self.tr_time = 0.0
        self.mycount = 0
        self.tbwriter = SummaryWriter(os.path.join(config.tfboard_log_dir, config.run_name))

        # init network
        self.network = getattr(networks, self.config.model_name)(self.config)

        # init loss
        self._init_loss_and_step()

        return

    def _init_loss_and_step(self):
        #if self.config.loss == 'Naive-LogSoftmax':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = torch.nn.CrossEntropyLoss(reduction='sum')
        #    self.step = self._logsoftmax_step
        if self.config.loss == 'Linear-LR':
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
            self.step = self._naive_step
        elif self.config.loss == 'Naive-LRSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.NaiveMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    )
            self.step = self._naive_step
        elif self.config.loss == 'Minibatch-LRSQ':
            logging.info(f'loss_type: {self.config.loss}')
            self.mnloss = MNLoss.MinibatchMNLoss(
                    omega=self.config.omega,
                    M=self.config.M,
                    N=self.config.N,
                    )
            self.step = self._minibatch_step
        #elif self.config.loss == 'Sogram-LRSQ':
        #    logging.info(f'loss_type: {self.config.loss}')
        #    self.mnloss = MNLoss.SogramMNLoss(
        #            self.config.k,
        #            self.config.k1,
        #            alpha=self.config.alpha,
        #            omega=self.config.omega,
        #            nnz=self.config.nnz
        #            )
        #    self.step = self._sogram_step
        else:
            raise
        return

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
        elif optimizer_name == 'adagrad':
            optimizer = optim.Adagrad(parameters,
                                   weight_decay=self.config.weight_decay,
                                   lr=self.config.learning_rate,
                                   initial_accumulator_value=0.1,
                                   eps=self.config.eps)
            print('opt:%s'%optimizer_name, 'eps:%f'%self.config.eps)
        #elif optimizer_name == 'adam':
        #    optimizer = optim.Adam(parameters,
        #                           weight_decay=self.config.weight_decay,
        #                           lr=self.config.learning_rate)
        #elif optimizer_name == 'adamw':
        #    optimizer = optim.AdamW(parameters,
        #                            weight_decay=self.config.weight_decay,
        #                            lr=self.config.learning_rate)
        else:
            raise RuntimeError(
                'Unsupported optimizer: {self.config.optimizer}')

        return optimizer

    def _ginfo1(self):
        gExp = []
        for param in self.parameters():
            if param.requires_grad:
                gExp.append(param.grad.detach().clone())
        return gExp # [P.grad, Q.grad]

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

    def _naive_step(self, batch, batch_idx):
        us, vs = batch['us'], batch['vs']
        ys, os = batch['ys'], batch['os']
        _as = batch['_as']
        _bs = batch['_bs']

        ps, qs = self.network(batch['us'], batch['vs'])
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        loss = self.mnloss(ys, _as, _bs, ps, qs, pts, qts, isscaling=True, mask=os) + 0.5 * self.config.l2_lambda * self._wnorm_sq()
        if self.config.reduce_mode == 'mean':
            scaler = os._nnz() if os is not None else ys.size()[0]*ys.size()[1]
            loss = loss/scaler
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, omega: {self.config.omega}, loss: {loss.item()}')
        return loss

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

    def _weighted_lrloss(self, logits, Y, mask=None):
        _lrloss = F.binary_cross_entropy_with_logits
        coos = Y._indices()
        logits_pos = logits[coos[0], coos[1]]
        L_plus_part1 = _lrloss(logits_pos, (Y._values() > 0).to(logits_pos.dtype), reduction='none')
        L_plus_part2 = _lrloss(logits_pos, logits_pos.new_zeros(logits_pos.size()), reduction='none')
        L_plus = L_plus_part1 - self.config.omega * L_plus_part2
        L_minus = _lrloss(logits, logits.new_zeros(logits.size()), reduction='none')
        if mask is not None:
            mask_coos = mask._indices()
            L_minus = L_minus[mask_coos[0], mask_coos[1]]
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
        #logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
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
        #logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        #print(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
        return loss

    def _calc_func_val(self): #bsize_i=8192, bsize_j=32768):
        def _inner_forward(Utr, Vtr):
            if self.config.loss.startswith('Linear-LR'):
                P, Unorm_sq, Q, Vnorm_sq = self.network(Utr, Vtr)
                logits = P.sum(dim=-1, keepdim=True) + Q.sum(dim=-1)
                if self.config.isl2norm:
                    uv_norm = (Unorm_sq.unsqueeze(dim=-1) + Vnorm_sq) ** 0.5
                    logits = torch.div(logits, uv_norm)
            else:
                P, Q = self.network(Utr, Vtr)
                logits = P @ Q.T
            return logits, P ,Q

        def _ploss(logits):
            if self.config.loss.startswith('Linear-LR'):
                return self._weighted_lrloss(logits, target) / self.config.l2_lambda
            else:
                return self._weighted_lrloss(logits, target, target)

        def _nloss(logits):
            if self.config.loss.startswith('Linear-LR'):
                return self._weighted_lrloss(logits, target) / self.config.l2_lambda
            else:
                return self._weighted_lrloss(logits, target, neg_mask_tr)

        def _loss(logits):
            if self.config.loss.startswith('Linear-LR'):
                return self._weighted_lrloss(logits, target) / self.config.l2_lambda
            else:
                return self._weighted_lrloss(logits, target, pn_mask_tr)

        save_dir = os.path.join(self.config.tfboard_log_dir, self.config.run_name)
        save_Ytr = os.path.join(save_dir, 'Ytr.npz')
        save_pn_mask_tr = os.path.join(save_dir, 'pn_mask_tr.npz')
        save_psg = os.path.join(save_dir, 'persample_grad_sq_%d.npy'%self.global_step)

        fullbatch_grad_sq = 0.0
        with torch.enable_grad():
            opt = self.optimizers()
            opt.zero_grad()

            Ytr = self.trainer.train_dataloader.dataset.datasets.Yu
            m, n = self.config.M, self.config.N
            if not os.path.isfile(save_Ytr):
                sp.sparse.save_npz(save_Ytr, Ytr)

            pn_mask_tr = self.trainer.train_dataloader.dataset.datasets.pn_mask
            if pn_mask_tr is not None and (not os.path.isfile(save_pn_mask_tr)):
                sp.sparse.save_npz(save_pn_mask_tr, pn_mask_tr)
            if pn_mask_tr is None:
                scaler = m*n
                neg_mask_tr = sp.sparse.csr_matrix(1.0 - Ytr.A)
            else:
                scaler = pn_mask_tr.sum()
                neg_mask_tr = pn_mask_tr - Ytr
            pos_num = Ytr.sum()
            neg_num = neg_mask_tr.sum()

            #start_time = time.time()
            Utr = spmtx2tensor(self.trainer.train_dataloader.dataset.datasets.U)
            Vtr = spmtx2tensor(self.trainer.train_dataloader.dataset.datasets.V)
            #Atr = torch.FloatTensor(self.trainer.train_dataloader.dataset.datasets.A)
            #Btr = torch.FloatTensor(self.trainer.train_dataloader.dataset.datasets.B)
            target = spmtx2tensor(Ytr)
            if pn_mask_tr is not None:
                pn_mask_tr = spmtx2tensor(pn_mask_tr)
            neg_mask_tr = spmtx2tensor(neg_mask_tr)

            ## Pos
            opt.zero_grad()
            logits, P, Q = _inner_forward(Utr, Vtr)
            w_sq = self._wnorm_sq()

            if self.config.loss.startswith('Naive-LRLR'):
                loss1 = _ploss(logits)*m*n/pos_num + 0.5 * self.config.l2_lambda * w_sq
            else:
                raise
            if self.config.reduce_mode == 'mean':
                loss1 = loss1/scaler

            self.manual_backward(loss1)
            pos_grad = self._ginfo1() # params size
            pos_grad_sq = sum([ge.pow(2).sum().item() for ge in pos_grad]) # each param grad ** 2, sum

            ## Neg
            opt.zero_grad()
            logits, P, Q = _inner_forward(Utr, Vtr)
            w_sq = self._wnorm_sq()

            if self.config.loss.startswith('Naive-LRLR'):
                loss1 = _nloss(logits)*m*n/neg_num + 0.5 * self.config.l2_lambda * w_sq
            else:
                raise
            if self.config.reduce_mode == 'mean':
                loss1 = loss1/scaler

            self.manual_backward(loss1)
            neg_grad = self._ginfo1() # params size
            neg_grad_sq = sum([ge.pow(2).sum().item() for ge in neg_grad]) # each param grad ** 2, sum

            # Pos - Neg
            pn_grad_sq = sum([(pg - ng).pow(2).sum().item() for ng, pg in zip(neg_grad, pos_grad)]) # each param grad ** 2, sum

            ## Full
            opt.zero_grad()
            logits, P, Q = _inner_forward(Utr, Vtr)
            w_sq = self._wnorm_sq()

            if self.config.loss.startswith('Linear-LR'):
                loss1 = _loss(logits) + 0.5 * w_sq
            elif self.config.loss.startswith('Naive-LRLR'):
                loss1 = _loss(logits) + 0.5 * self.config.l2_lambda * w_sq
            else:
                raise
            if self.config.reduce_mode == 'mean':
                loss1 = loss1/scaler

            self.manual_backward(loss1)
            fullbatch_grad = self._ginfo1() # params size
            fullbatch_grad_sq = sum([ge.pow(2).sum().item() for ge in fullbatch_grad]) # each param grad ** 2, sum

            jcb = jacobian(_loss, logits)
            if self.config.reduce_mode == 'mean':
                jcb = jcb/scaler
                w_sq = w_sq/scaler/scaler
            if self.config.loss.startswith('Linear-LR'):
                persample_grad_sq = w_sq + (scaler*jcb)**2 + 2*scaler*jcb*logits
            else:
                def _helper(_UV, _PQ):
                    if _PQ.shape[0] == m:
                        _PQ_norm_sq = torch.norm(_PQ, dim=1).unsqueeze(1)**2 # (M, 1)
                    else:
                        _PQ_norm_sq = torch.norm(_PQ, dim=1).unsqueeze(0)**2 # (1, N)
                    return scaler*scaler*_PQ_norm_sq*(jcb**2) + 2*self.config.l2_lambda*scaler*jcb*logits
                persample_grad_sq =  _helper(Utr, Q)
                persample_grad_sq += _helper(Vtr, P)
                persample_grad_sq += self.config.l2_lambda*self.config.l2_lambda* w_sq

            np.save(save_psg, persample_grad_sq.detach().cpu().numpy())
            if pn_mask_tr is not None:
                mask_coos = pn_mask_tr._indices()
                persample_grad_sq = persample_grad_sq[mask_coos[0], mask_coos[1]]
            grad_var = persample_grad_sq.mean() - fullbatch_grad_sq
            opt.zero_grad()

        msg = ('global_step: {}, epoch: {}, training_time: {:.3f}, gradExpSq: {:.6e}, gVar: {:.6e}, func_val: {:.6e}, gpos: {:.6e}, gneg: {:.6e}, gpn: {:.6e}'.format(
            self.global_step,
            self.current_epoch,
            self.tr_time,
            fullbatch_grad_sq, #gExpSq,
            grad_var.item(), #gVar.item(), #if self.config.check_grad_var else np.nan,
            loss1.item(), #fval[0].item(),
            pos_grad_sq,
            neg_grad_sq,
            pn_grad_sq,
            ))
        logging.debug(msg)
        print(msg)
        return

    def training_step(self, batch, batch_idx):
        if self.global_step % 1000 == 0:
            self._calc_func_val()
        opt = self.optimizers()
        opt.zero_grad()
        start_time = time.time()
        loss = self.step(batch, batch_idx)
        self.manual_backward(loss)
        opt.step()
        self.tr_time += time.time() - start_time
        logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
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
            Unorm_sq = np.vstack([_data.reshape(-1, 1) for _data in Unorm_sq if _data is not None])
            Vnorm_sq = np.vstack([_data.reshape(-1, 1) for _data in Vnorm_sq if _data is not None])

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

        #if self.global_step // 400000 >= self.mycount: #self.config.check_func_val and split == 'val':
        #if self.global_step // 10000 >= self.mycount and split == 'val':
        #    self._calc_func_val()
        #    self.mycount += 1
        #if split == 'val':
        #    self._calc_func_val()

        if not self.config.silent and (not self.trainer or self.trainer.is_global_zero):
            print(f'====== {split} dataset evaluation result =======')
            print(self.eval_metric)
            print('')
            logging.debug(f'{split} dataset evaluation result:\n{self.eval_metric}')
        self.eval_metric.reset()

        return metric_dict
