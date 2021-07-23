import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from abc import abstractmethod
from argparse import Namespace
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities.parsing import AttributeDict

from . import networks
from . import MNLoss
from .metrics import MultiLabelMetrics
from .utils import dump_log, argsort_top_k


class TwoTowerModel(pl.LightningModule):
    """Concrete class handling Pytorch Lightning training flow"""
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if isinstance(config, Namespace):
            config = vars(config)
        if isinstance(config, dict):
            config = AttributeDict(config)
        self.config = config
        self.eval_metric = MultiLabelMetrics(self.config)
        self.network = getattr(networks, self.config.model_name)(self.config)

        # init loss
        if self.config.loss == 'DPR':
            self.mnloss = torch.nn.CrossEntropyLoss(reduction='mean')
            self.step = self._dpr_step
        #elif self.config.loss == 'Naive-LRLR':
        #    self.step = self._minibatch_step
        #    self.mnloss = MNLoss.NaiveMNLoss(omega=self.config.omega,
        #            loss_func_minus=torch.nn.functional.binary_cross_entropy_with_logits)
        #elif self.config.loss == 'Naive-LRSQ':
        #    self.step = self._minibatch_step
        #    self.mnloss = MNLoss.NaiveMNLoss(omega=self.config.omega)
        elif self.config.loss == 'Minibatch':
            self.mnloss = MNLoss.MinibatchMNLoss(omega=self.config.omega)
            self.step = self._minibatch_step
        elif self.config.loss == 'Sogram':
            self.mnloss = MNLoss.SogramMNLoss(self.config.k,
                    self.config.k1, alpha=self.config.alpha, omega=self.config.omega)
            self.step = self._sogram_step
        else:
            raise

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
                #print(current_step, warmup_steps, total_training_steps)
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
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(parameters,
                                   weight_decay=self.config.weight_decay,
                                   lr=self.config.learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(parameters,
                                    weight_decay=self.config.weight_decay,
                                    lr=self.config.learning_rate)
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
        #torch.nn.utils.clip_grad_value_(parameters, 0.5)

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

    def _dpr_step(self, batch):
        ps, qs = self.network(batch['us'], batch['vs'])
        #ys = torch.diag(batch['ys'])
        ys = torch.arange(batch['ys'].shape[0], dtype=torch.long, device=ps.device)
        logits = ps @ qs.T # (m, m)
        loss = self.mnloss(logits, ys)
        f = open('./train.log', 'a')
        print('-'*10, file=f)
        #for _i in range(batch['us'].shape[0]):
        #    #print(batch['us'][_i, :], file=f)
        #    print(ps[_i, :], file=f)
        #for _i in range(batch['vs'].shape[0]):
        #    #print(batch['vs'][_i, :], file=f)
        #    print(qs[_i, :], file=f)
        print(batch['us'].detach().sum().item(), batch['vs'].detach().sum().item(), file=f)
        print(ps.detach().sum().item(), qs.detach().sum().item(), file=f)
        print('loss:', loss.item(), file=f)
        f.close()
        return loss#, P, Q

    def _sogram_step(self, batch):
        ps, qs = self.network(batch['us'], batch['vs'])
        pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        ys = batch['ys']
        _as = batch['_as']
        _bs = batch['_bs']
        _abs = batch['_abs']
        _bbs = batch['_bbs']
        loss = self.mnloss(ys, _as, _bs, _abs, _bbs, ps, qs, pts, qts)
        return loss#, ps, qs

    def _minibatch_step(self, batch):
        P, Q = self.network(batch['U'], batch['V'])
        Pt = P.new_ones(P.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
        Qt = Q.new_ones(Q.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
        Y = batch['Y']
        A = batch['A']
        B = batch['B']
        loss = self.mnloss(Y, A, B, P, Q, Pt, Qt)
        return loss#, P, Q

    def training_step(self, batch, batch_idx):
        #opt = self.optimizers()
        #print(opt.param_groups[0]["lr"])
        loss = self.step(batch)
        #if batch_idx < 100:
        #    print(batch_idx, loss.item()) 
        #    #print(batch)
        #else:
        #    exit(0)
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
        pred_logits = P.detach() @ Q.detach().T
        #f = open('./test.log', 'a')
        #print('-.'*10, file=f)
        #for _i in range(batch['U'].shape[0]):
        #    print('>>>U<<<', batch['U'][_i, :], file=f)
        #    print(','.join(['%.6f'%s for s in pred_logits.cpu().numpy()[_i,:]]), file=f)
        #for _i in range(batch['V'].shape[0]):
        #    print('>>>V<<<', batch['V'][_i, :], file=f)
        #print(batch['U'].detach().sum().item(), batch['V'].detach().sum().item(), file=f)
        #print(P.detach().sum().item(), Q.detach().sum().item(), file=f)
        #f.close()
        #return {'pred_scores': torch.sigmoid(pred_logits).cpu().numpy(),
        return {'pred_scores': pred_logits.cpu().numpy(),
                'target': batch['Y'].cpu().to_dense().numpy() if 'Y' in batch else batch['label'].detach().cpu().numpy()}

    def _shared_eval_step_end(self, batch_parts):
        pred_scores = np.vstack(batch_parts['pred_scores'])
        target = np.vstack(batch_parts['target'])
        return self.eval_metric.update(target, pred_scores)

    def _shared_eval_epoch_end(self, step_outputs, split):
        metric_dict = self.eval_metric.get_metric_dict()
        f = open('./test.log', 'a')
        print("%.4f"%metric_dict['Aver-Rank'], file=f)
        f.close()
        self.log_dict(metric_dict)
        dump_log(config=self.config, metrics=metric_dict, split=split)

        if not self.config.silent and (not self.trainer or self.trainer.is_global_zero):
            print(f'====== {split} dataset evaluation result =======')
            print(self.eval_metric)
            print()
        self.eval_metric.reset()
        #exit()
        return metric_dict

    #def predict_step(self, batch, batch_idx, dataloader_idx):
    #    outputs = self.network(batch['text'])
    #    pred_scores= torch.sigmoid(outputs['logits']).detach().cpu().numpy()
    #    k = self.config.save_k_predictions
    #    top_k_idx = argsort_top_k(pred_scores, k, axis=1)
    #    top_k_scores = np.take_along_axis(pred_scores, top_k_idx, axis=1)

    #    return {'top_k_pred': top_k_idx,
    #            'top_k_pred_scores': top_k_scores}
