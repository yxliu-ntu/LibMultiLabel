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
from .utils import dump_log


class TwoTowerBaseModel(pl.LightningModule):
    """Abstract class handling Pytorch Lightning training flow"""

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, Namespace):
            config = vars(config)
        if isinstance(config, dict):
            config = AttributeDict(config)
        self.config = config

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
        loss, P, Q = self.shared_step(batch, False)
        return {'loss': loss.item() if loss is not None else None,
                'P': P.detach(),
                'Q': Q.detach(),
                'Y': batch['Y'].to_dense().cpu().numpy() if 'Y' in batch else batch['label'].detach().cpu().numpy()}

    def validation_epoch_end(self, step_outputs):
        eval_metric = self.evaluate(step_outputs, 'val')
        return eval_metric

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, step_outputs):
        eval_metric = self.evaluate(step_outputs, 'test')
        self.test_results = eval_metric
        return eval_metric

    def evaluate(self, step_outputs, split):
        eval_metric = MultiLabelMetrics(self.config)
        for step_output in step_outputs:
            pred_scores = torch.sigmoid(step_output['P'] @ step_output['Q'].T).cpu().numpy()
            eval_metric.add_values(y_pred=pred_scores,
                                   y_true=step_output['Y'])
        metric_dict = eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        dump_log(config=self.config, metrics=metric_dict, split=split)

        self.print(f'\n====== {split.upper()} dataset evaluation result =======')
        self.print(eval_metric)
        return eval_metric

    def print(self, string):
        if not self.config.get('silent', False):
            if not self.trainer or self.trainer.is_global_zero:
                print(string)


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
