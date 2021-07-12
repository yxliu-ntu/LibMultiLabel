from abc import abstractmethod
from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.parsing import AttributeDict

from . import networks
from . import MNLoss
from .metrics import MultiLabelMetrics
from .utils import dump_log, argsort_top_k


class TwoTowerBaseModel(pl.LightningModule):
    """Abstract class handling Pytorch Lightning training flow"""

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, Namespace):
            config = vars(config)
        if isinstance(config, dict):
            config = AttributeDict(config)
        self.config = config
        self.eval_metric = MultiLabelMetrics(self.config)

    def configure_optimizers(self):
        """Initialize an optimizer for the free parameters of the network.
        """
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer_name = self.config.optimizer
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
        else:
            raise RuntimeError(
                'Unsupported optimizer: {self.config.optimizer}')

        torch.nn.utils.clip_grad_value_(parameters, 0.5)

        return optimizer

    @abstractmethod
    def shared_step(self, batch, is_train):
        """Return loss and network outputs"""
        return NotImplemented

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, True)
        if batch_idx < 100:
            print(batch_idx, loss.item()) 
            #print(batch)
        else:
            exit(0)
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
        loss, P, Q = self.shared_step(batch, False)
        pred_logits = P.detach() @ Q.detach().T
        return {'loss': loss.item() if loss is not None else None,
                'pred_scores': torch.sigmoid(pred_logits).cpu().numpy(),
                'target': batch['Y'].to_dense().cpu().numpy() if 'Y' in batch else batch['label'].detach().cpu().numpy()}

    def _shared_eval_step_end(self, batch_parts):
        pred_scores = np.vstack(batch_parts['pred_scores'])
        target = np.vstack(batch_parts['target'])
        return self.eval_metric.update(target, pred_scores)

    def _shared_eval_epoch_end(self, step_outputs, split):
        metric_dict = self.eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        dump_log(config=self.config, metrics=metric_dict, split=split)

        if not self.config.silent and (not self.trainer or self.trainer.is_global_zero):
            print(f'====== {split} dataset evaluation result =======')
            print(self.eval_metric)
            print()
        self.eval_metric.reset()
        return metric_dict

    def predict_step(self, batch, batch_idx, dataloader_idx):
        outputs = self.network(batch['text'])
        pred_scores= torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        k = self.config.save_k_predictions
        top_k_idx = argsort_top_k(pred_scores, k, axis=1)
        top_k_scores = np.take_along_axis(pred_scores, top_k_idx, axis=1)

        return {'top_k_pred': top_k_idx,
                'top_k_pred_scores': top_k_scores}


class TwoTowerModel(TwoTowerBaseModel):
    def __init__(self, config, word_dict=None, classes=None):
        super().__init__(config)
        self.save_hyperparameters()

        self.word_dict = word_dict
        self.classes = classes
        self.config.num_classes = len(self.classes)

        embed_vecs = self.word_dict.vectors
        self.network = getattr(networks, self.config.model_name)(
            self.config, embed_vecs)

        if config.init_weight is not None:
            init_weight = networks.get_init_weight_func(self.config)
            self.apply(init_weight)

        # init loss
        if self.config.loss == 'Ori-LRLR':
            self.mnloss = None
        elif self.config.loss == 'Naive-LRLR':
            self.mnloss = MNLoss.NaiveMNLoss(omega=self.config.omega,
                    loss_func_minus=torch.nn.functional.binary_cross_entropy_with_logits)
        elif self.config.loss == 'Naive-LRSQ':
            self.mnloss = MNLoss.NaiveMNLoss(omega=self.config.omega)
        elif self.config.loss == 'Minibatch-LRSQ':
            self.mnloss = MNLoss.MinibatchMNLoss(omega=self.config.omega)
        elif self.config.loss == 'Sogram-LRSQ':
            self.mnloss = MNLoss.SogramMNLoss(self.config.num_filter_per_size*len(self.config.filter_sizes),
                    self.config.k1, alpha=self.config.alpha, omega=self.config.omega)
        else:
            raise

    def shared_step(self, batch, is_train):
        if self.mnloss is None:
            target_labels = batch['label']
            P, Q = self.network(batch['text'])
            logits = P @ Q.T
            loss = F.binary_cross_entropy_with_logits(logits, target_labels, reduction='sum')
            return loss, P, Q
        elif isinstance(self.mnloss, MNLoss.SogramMNLoss) and is_train:
            ps, qs = self.network(batch['us'], batch['vs'])
            pts = ps.new_ones(ps.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
            qts = qs.new_ones(qs.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
            ys = batch['ys']
            _as = batch['_as']
            _bs = batch['_bs']
            _abs = batch['_abs']
            _bbs = batch['_bbs']
            loss = self.mnloss(ys, _as, _bs, _abs, _bbs, ps, qs, pts, qts)
            return loss, ps, qs
        else:
            P, Q = self.network(batch['U'], batch['V'])
            if is_train:
                Pt = P.new_ones(P.size()[0], self.config.k1) * np.sqrt(1./self.config.k1) * self.config.imp_r
                Qt = Q.new_ones(Q.size()[0], self.config.k1) * np.sqrt(1./self.config.k1)
                Y = batch['Y']
                A = batch['A']
                B = batch['B']
                loss = self.mnloss(Y, A, B, P, Q, Pt, Qt)
                return loss, P, Q
            else:
                return None, P, Q
