import os
import time
import logging
import numpy as np
import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from .autograd_hacks import autograd_hacks as aghacks
from tqdm import trange

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
        self.automatic_optimization = False

        if isinstance(config, Namespace):
            config = vars(config)
        if isinstance(config, dict):
            config = AttributeDict(config)
        self.config = self.hparams.config
        self.Y_eval = self.hparams.Y_eval
        self.eval_metric = MultiLabelMetrics(self.config)
        self.tr_time = 0.0
        #self.tbwriter = SummaryWriter(os.path.join(config.tfboard_log_dir, config.run_name))

        # init network
        self.network = getattr(networks, self.config.model_name)(self.config)
        if self.config.check_grad_var:
            aghacks.add_hooks(self.network)

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
            #torch.nn.utils.clip_grad_value_(parameters, 0.5)
        elif optimizer_name == 'adagrad':
            optimizer = optim.Adagrad(parameters,
                                   weight_decay=self.config.weight_decay,
                                   lr=self.config.learning_rate,
                                   initial_accumulator_value=0.1,
                                   #eps=1e-7)
                                   eps=self.config.eps)
            print('opt:%s'%optimizer_name, 'eps:%f'%self.config.eps)
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

    def _ginfo1(self):
        grad = torch.tensor(0.)
        for param in self.parameters():
            if param.requires_grad:
                grad += (torch.norm(param.grad.detach())**2)
        return grad

    def _ginfo2(self, _stage):
        start_time = time.time()
        scaler = self.config.M * self.config.N
        bs = self.config.M if _stage==0 else self.config.N
        grad = []
        gs = torch.tensor(0.)
        #print('----ginfo_init:', 0)

        for param in self.parameters():
            if param.requires_grad:
                # (1, K)
                grad.append(param.grad.detach().reshape(1, -1))
                #print('----grad:', time.time() - start_time)
                if param.grad1.shape[0] == bs:
                    # (bs, K) + (1, K) -> (bs, K) -> (1, )
                    #gs += ((param.grad1 * scaler + self.config.l2_lambda * param.data.unsqueeze(0))**2).sum()
                    gs += torch.norm(param.grad1 * scaler + self.config.l2_lambda * param.data.unsqueeze(0))**2
                elif param.grad1.shape[0] == 1:
                    pass
                else: # there are only two towers
                    raise
                #print('----gs:', time.time() - start_time)
        return torch.cat(grad, dim=-1), gs

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
        ys = batch['ys']
        _as = batch['_as']
        _bs = batch['_bs']

        ps, qs = self.network(batch['us'], batch['vs'])
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        loss = self.mnloss(ys, _as, _bs, ps, qs, pts, qts, isscaling=True) + 0.5 * self.config.l2_lambda * self._wnorm_sq()
        #logging.debug(f'epoch: {self.current_epoch}, batch: {batch_idx}, loss: {loss.item()}')
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
        def _inner_forward(Utr, Vtr, target):
            if self.config.loss.startswith('Linear-LR'):
                P, Unorm_sq, Q, Vnorm_sq = self.network(Utr, Vtr)
                logits = P.sum(dim=-1, keepdim=True) + Q.sum(dim=-1)
                if self.config.isl2norm:
                    uv_norm = (Unorm_sq.unsqueeze(dim=-1) + Vnorm_sq) ** 0.5
                    logits = torch.div(logits, uv_norm)
                loss = self.mnloss(logits, target) / self.config.l2_lambda
            else:
                P, Q = self.network(Utr, Vtr)
                Pt = P.new_ones(P.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
                Qt = Q.new_ones(Q.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
                if self.config.loss.startswith('Naive'):
                    loss = self.mnloss(target, Atr, Btr, P, Q, Pt, Qt, isscaling=False)
                else:
                    raise # the gradient of minibatch loss may have some bugs.
            return loss

        with torch.enable_grad():
            opt = self.optimizers()
            Ytr = self.trainer.train_dataloader.dataset.datasets.Yu
            m, n = self.config.M, self.config.N
            loss_reduce_type = 'sum'

            # get gradExpSq
            Utr = spmtx2tensor(self.trainer.train_dataloader.dataset.datasets.U)
            Vtr = spmtx2tensor(self.trainer.train_dataloader.dataset.datasets.V)
            Atr = torch.FloatTensor(self.trainer.train_dataloader.dataset.datasets.A)
            Btr = torch.FloatTensor(self.trainer.train_dataloader.dataset.datasets.B)
            target = spmtx2tensor(Ytr)
            loss1 = _inner_forward(Utr, Vtr, target)
            if self.config.loss.startswith('Linear-LR'):
                loss1 = _inner_forward(Utr, Vtr, target) + 0.5 * self._wnorm_sq()
            else:
                loss1 = _inner_forward(Utr, Vtr, target) + 0.5 * self.config.l2_lambda * self._wnorm_sq()
            self.manual_backward(loss1)
            gExpSq1 = self._ginfo1()
            opt.zero_grad()

            # get gradSqExp
            aghacks.enable_hooks()
            grads = []
            fval = torch.zeros(2)
            gsP = torch.tensor(0.)
            gsQ = torch.tensor(0.)
            for _stage in range(2):
                bsize_i = m if _stage == 0 else 1
                bsize_j = 1 if _stage == 0 else n
                segment_m = math.ceil(m/bsize_i)
                segment_n = math.ceil(n/bsize_j)
                print(bsize_i, bsize_j, segment_m, segment_n)

                P, Unorm_sq, Q, Vnorm_sq = None, None, None, None
                func_val = torch.tensor(0.)
                for i in range(segment_m):
                    i_start, i_end = i*bsize_i, min((i+1)*bsize_i, m)
                    for j in range(segment_n):
                        print(i, j)
                        start_time = time.time()
                        j_start, j_end = j*bsize_j, min((j+1)*bsize_j, n)
                        Utr = spmtx2tensor(self.trainer.train_dataloader.dataset.datasets.U[i_start:i_end])
                        Vtr = spmtx2tensor(self.trainer.train_dataloader.dataset.datasets.V[j_start:j_end])
                        Atr = torch.FloatTensor(self.trainer.train_dataloader.dataset.datasets.A[i_start:i_end])
                        Btr = torch.FloatTensor(self.trainer.train_dataloader.dataset.datasets.B[j_start:j_end])
                        target = spmtx2tensor(Ytr[i_start:i_end, j_start:j_end])
                        #logits = torch.zeros(i_end-i_start, n)
                        print('dataload:', time.time() - start_time)

                        # Forward
                        if self.config.loss.startswith('Linear-LR'):
                            loss = _inner_forward(Utr, Vtr, target) + 0.5 * self._wnorm_sq() / (segment_m * segment_n)
                        else:
                            loss = _inner_forward(Utr, Vtr, target) + 0.5 * self.config.l2_lambda * self._wnorm_sq() / (segment_m * segment_n)

                        fval[_stage] += loss.detach()
                        print('forward:', time.time() - start_time)

                        self.manual_backward(loss, retain_graph=True)
                        aghacks.compute_grad1(self.network, loss_reduce_type)
                        aghacks.clear_backprops(self.network)
                        print('backward:', time.time() - start_time)

                        ## unitest
                        #for layer in self.network.modules():
                        #    if not aghacks.is_supported(layer):
                        #        continue
                        #    for param in layer.parameters():
                        #        assert torch.allclose(param.grad, param.grad1.sum(dim=0) + self.config.l2_lambda * param.data.detach(), rtol=1e-05, atol=1e-05, equal_nan=True)
                        if _stage == 0:
                            _grad, _gsP = self._ginfo2(_stage)
                            print('ginfo:', time.time() - start_time)
                            gsP += _gsP
                        else:
                            _grad, _gsQ = self._ginfo2(_stage)
                            print('ginfo:', time.time() - start_time)
                            gsQ += _gsQ
                        if len(grads) == _stage:
                            grads.append(_grad)
                        else:
                            grads[_stage] += _grad
                        opt.zero_grad()
                        print('finish:', time.time() - start_time)
            print(grads[0].pow(2).sum(), grads[1].pow(2).sum())
            print(fval[0], fval[1])
            print(loss1.item(), gExpSq1.item())
            gExpSq = torch.norm(grads[0])**2
            gSqExp = (gsP + gsQ) / (self.config.M * self.config.N)
            gVar = (gSqExp - gExpSq) / (self.config.M * self.config.N * self.config.bratio) if not self.config.bratio == 1 else 0

        msg = ('global_step: {}, epoch: {}, training_time: {:.3f}, gExpSq: {:.6e}, gVar: {:.6e}, func_val: {:.6e}'.format(
            self.global_step,
            self.current_epoch,
            self.tr_time,
            gExpSq.item(),
            gVar.item(), #if self.config.check_grad_var else np.nan,
            fval[0].item()
            ))
        logging.debug(msg)
        print(msg)
        return

    def training_step(self, batch, batch_idx):
        if self.config.check_grad_var:
            aghacks.disable_hooks()
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

        if self.config.check_func_val and split == 'val':
            self._calc_func_val()

        if not self.config.silent and (not self.trainer or self.trainer.is_global_zero):
            print(f'====== {split} dataset evaluation result =======')
            print(self.eval_metric)
            print('')
            logging.debug(f'{split} dataset evaluation result:\n{self.eval_metric}')
        self.eval_metric.reset()
        exit()

        return metric_dict
