import re
import logging

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from scipy.special import expit

from .utils import argsort_top_k


def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def precision_recall_at_ks(y_true, y_pred, top_ks):
    max_k = max(top_ks)
    top_idx = argsort_top_k(y_pred, max_k, axis=1)
    n_pos = y_true.sum(axis=1)
    scores = {}
    for k in top_ks:
        n_pos_in_top_k = np.take_along_axis(y_true, top_idx[:,:k], axis=1).sum(axis=1)
        scores[f'P@{k}'] = np.mean(n_pos_in_top_k / k).item()  # precision at k
        scores[f'R@{k}'] = np.mean(n_pos_in_top_k / (n_pos + 1e-10)).item()  # recall at k
    return scores

def get_ranks(y_pred, y_true):
    positive_idx_per_question = y_true.nonzero()[1]
    indices = np.argsort(y_pred*-1, axis=1, kind='stable')
    ranks = []
    for i, idx in enumerate(positive_idx_per_question):
        # aggregate the rank of the known gold passage in the sorted results for each question
        gold_idx = (indices[i, :] == idx).ravel().nonzero()[-1][0]
        #logging.debug(f'gt:{idx}, rank:{gold_idx}, len:{indices[i, :].shape[-1]}')
        ranks.append(gold_idx + 1)
    return ranks

class MultiLabelMetrics():
    def __init__(self, config):
        self.monitor_metrics = config.get('monitor_metrics', [])
        self.metric_threshold = config.get('metric_threshold', 0.5)

        self.n_eval = 0
        self.multilabel_confusion_matrix = 0.
        self.metric_stats = {}
        self.ranks = []

        self.top_ks = set()
        self.prec_recall_metrics = []
        for metric in self.monitor_metrics:
            if re.match('[P|R]@\d+', metric):
                top_k = int(metric[2:])
                self.top_ks.add(top_k)
                self.metric_stats[metric] = 0.
                self.prec_recall_metrics.append(metric)
            elif metric not in ['Micro-Precision', 'Micro-Recall', 'Micro-F1', 'Macro-F1', 'Another-Macro-F1', 'Aver-Rank', 'MRR@10']:
                raise ValueError(f'Invalid metric: {metric}')

    def reset(self):
        self.n_eval = 0
        self.ranks = []
        self.multilabel_confusion_matrix = 0.
        for metric in self.metric_stats:
            self.metric_stats[metric] = 0.

    def update(self, y_true, y_pred):
        """Add evaluation results of a batch of y_true and y_pred.

        Args:
            y_true (ndarray): an array with ground truth labels (shape: batch_size * number of classes)
            y_pred (ndarray): an array with predicted label values (shape: batch_size * number of classes)
        """
        y_pred_pos = expit(y_pred) > self.metric_threshold

        n_eval = len(y_true)
        self.n_eval += n_eval
        self.multilabel_confusion_matrix += multilabel_confusion_matrix(y_true, y_pred_pos)

        # Add metrics like P@k, R@k to the result dict. Multiply n_eval for
        # cumulation.
        scores = precision_recall_at_ks(y_true, y_pred, top_ks=self.top_ks)
        for metric in self.prec_recall_metrics:
            self.metric_stats[metric] += (scores[metric] * n_eval)

        # Add averaged rank
        if 'Aver-Rank' in self.monitor_metrics or 'MRR@10' in self.monitor_metrics:
            _ranks = get_ranks(y_pred, y_true)
            self.ranks.extend(_ranks)

    def get_metric_dict(self):
        """Get evaluation results."""

        self.ranks = np.array(self.ranks)
        cm = self.multilabel_confusion_matrix
        cm_sum = cm.sum(axis=0)
        tp_sum, fp_sum, fn_sum = cm_sum[1,1], cm_sum[0,1], cm_sum[1,0]
        micro_precision = tp_sum / (tp_sum + fp_sum + 1e-10)
        micro_recall = tp_sum / (tp_sum + fn_sum + 1e-10)

        # Use lablewise tp, fp, fn to calculate Macro results
        tp, fp, fn = cm[:,1,1], cm[:,0,1], cm[:,1,0]
        labelwise_precision = tp / (tp + fp + 1e-10)
        labelwise_recall = tp / (tp + fn + 1e-10)
        macro_precision = labelwise_precision.mean()
        macro_recall = labelwise_recall.mean()

        result = {
                'Micro-Precision': micro_precision,
                'Micro-Recall': micro_recall,
                #'Micro-F1': f1(micro_precision, micro_recall),
                #'Macro-F1': f1(labelwise_precision, labelwise_recall).mean(),
                ## The f1 value of macro_precision and macro_recall. This variant of
                ## macro_f1 is less preferred but is used in some works. Please
                ## refer to Opitz et al. 2019 [https://arxiv.org/pdf/1911.03347.pdf]
                #'Another-Macro-F1': f1(macro_precision, macro_recall),
                #'Aver-Rank': self.ranks.mean()/100.,
        }
        for metric, val in self.metric_stats.items():
            result[metric] = val / self.n_eval
        if 'Aver-Rank' in self.monitor_metrics:
            result['Aver-Rank'] = self.ranks.mean()/100.
        if 'MRR@10' in self.monitor_metrics:
            inv_ranks = np.array([1/r if r <= 10 else 0.0 for r in self.ranks])
            result['MRR@10'] = inv_ranks.mean()/100.
        return result

    def __repr__(self):
        """Return evaluation results in markdown."""
        result = self.get_metric_dict()
        header = '|'.join([f'{k:^18}' for k in result.keys()])
        values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in result.values()])
        return f"|{header}|\n|{'-----------------:|' * len(result)}\n|{values}|"
