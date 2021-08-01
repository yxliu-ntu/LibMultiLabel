import argparse
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from functools import partial

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.parsing import AttributeDict

from libmultilabel import data_utils, MNLoss
from libmultilabel.model import Model
from libmultilabel.two_tower_model import TwoTowerModel
from libmultilabel.utils import Timer, dump_log, init_device, set_seed


def get_config():
    parser = argparse.ArgumentParser(
        add_help=False,
        description='multi-label learning for text classification')

    # load params from config file
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args, _ = parser.parse_known_args()
    config = {}
    if args.config:
        with open(args.config) as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

    # path / directory
    parser.add_argument('--data_dir', default='./data/dpr',
                        help='The directory to load data (default: %(default)s)')
    parser.add_argument('--result_dir', default='./runs',
                        help='The directory to save checkpoints and logs (default: %(default)s)')

    # data
    parser.add_argument('--data_name', default='rcv1',
                        help='Dataset name (default: %(default)s)')
    parser.add_argument('--trainL_path',
                        help='Path to training data (default: [data_dir]/trainL.csv)')
    parser.add_argument('--trainR_path',
                        help='Path to training data (default: [data_dir]/trainR.csv)')
    parser.add_argument('--validL_path',
                        help='Path to validation data (default: [data_dir]/validL.csv)')
    parser.add_argument('--validR_path',
                        help='Path to validation data (default: [data_dir]/validR.csv)')
    parser.add_argument('--testL_path',
                        help='Path to test data (default: [data_dir]/testL.csv)')
    parser.add_argument('--testR_path',
                        help='Path to test data (default: [data_dir]/testR.csv)')
    parser.add_argument('--bert_path',
                        help='Path to test data')
    #parser.add_argument('--val_size', type=float, default=0.2,
    #                    help='Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set (default: %(default)s).')
    #parser.add_argument('--min_vocab_freq', type=int, default=1,
    #                    help='The minimum frequency needed to include a token in the vocabulary (default: %(default)s)')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='The maximum number of tokens of a sample (default: %(default)s)')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether to shuffle training data before each epoch (default: %(default)s)')
    parser.add_argument('--drop_last', type=bool, default=False,
                        help='Whether to drop the last batch each epoch (default: %(default)s)')

    # train
    parser.add_argument('--seed', type=int,
                        help='Random seed (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=0.0,
                        help='Number of warm-up steps for training (default: %(default)s)')
    parser.add_argument('--bratio', type=float, default=None,
                        help='batch ratio of training samples for Sogram (default: %(default)s)')
    parser.add_argument('--bsize_i', type=int, default=16,
                        help='Size of training batches along rows of label matrix (default: %(default)s)')
    parser.add_argument('--bsize_j', type=int, default=None,
                        help='Size of training batches along cols of label matrix (default: %(default)s)')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'adamw', 'adamw-dpr', 'adagrad'],
                        help='Optimizer: SGD or Adam (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for optimizer (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay factor (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum factor for SGD only (default: %(default)s)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to wait for improvement before early stopping (default: %(default)s)')
    parser.add_argument('--gradient_clip_val', type=float, default=0.0,
                        help='value for clipping gradient, 0 means don’t clip')
    parser.add_argument('--gradient_clip_algorithm', type=str, choices=['norm', 'value'], default='norm',
                        help='value means clip_by_value, norm means clip_by_norm. Default: norm')
    parser.add_argument('--loss', type=str, choices=['Minibatch', 'Sogram', 'DPR'], default='DPR',
                        help='Type of loss function. Except for Ori-LRLR, the others only support two-tower models.')
    parser.add_argument('--omega', type=float, default=1.0,
                        help='Cost weight for the negative part of the loss function')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for updating Gramian')
    parser.add_argument('--imp_r', type=float, default=0.0,
                        help='Imputed value for the negative part of the loss function')

    # model
    parser.add_argument('--model_name', default='KimCNN',
                        help='Model to be used (default: %(default)s)')
    parser.add_argument('--init_weight', default='kaiming_uniform',
                        help='Weight initialization to be used (default: %(default)s)')
    parser.add_argument('--activation', default='relu',
                        help='Activation function to be used (default: %(default)s)')
    parser.add_argument('--k', type=int, default=128,
                        help='embedding dimension for each tower')
    parser.add_argument('--k1', type=int, default=4,
                        help='embedding dimension for imputed vectors')
    parser.add_argument('--pad_id', type=int, default=0,
                        help='pad id for bert model')
    #parser.add_argument('--num_filter_per_size', type=int, default=128,
    #                    help='Number of filters in convolutional layers in each size (default: %(default)s)')
    #parser.add_argument('--filter_sizes', type=int, nargs='+',
    #                    default=[4], help='Size of convolutional filters (default: %(default)s)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Optional specification of dropout (default: %(default)s)')
    #parser.add_argument('--dropout2', type=float, default=0.2,
    #                    help='Optional specification of the second dropout (default: %(default)s)')
    #parser.add_argument('--num_pool', type=int, default=1,
    #                    help='Number of pool for dynamic max-pooling (default: %(default)s)')

    # eval
    parser.add_argument('--eval_bsize_i', type=int, default=512,
                        help='Size of evaluating batches (default: %(default)s)')
    parser.add_argument('--eval_bsize_j', type=int, default=None,
                        help='Size of evaluating batches (default: %(default)s)')
    parser.add_argument('--metrics_threshold', type=float, default=0.5,
                        help='Thresholds to monitor for metrics (default: %(default)s)')
    parser.add_argument('--monitor_metrics', nargs='+', default=['P@1', 'P@3', 'P@5'],
                        help='Metrics to monitor while validating (default: %(default)s)')
    parser.add_argument('--val_metric', default='P@1',
                        help='The metric to monitor for early stopping (default: %(default)s)')

    ## pretrained vocab / embeddings
    #parser.add_argument('--vocab_file', type=str,
    #                    help='Path to a file holding vocabuaries (default: %(default)s)')
    #parser.add_argument('--embed_file', type=str,
    #                    help='Path to a file holding pre-trained embeddings (default: %(default)s)')
    #parser.add_argument('--label_file', type=str,
    #                    help='Path to a file holding all labels (default: %(default)s)')

    ## log
    #parser.add_argument('--save_k_predictions', type=int, nargs='?', const=100, default=0,
    #                    help='Save top k predictions on test set. k=%(const)s if not specified. (default: %(default)s)')
    #parser.add_argument('--predict_out_path',
    #                    help='Path to the an output file holding top 100 label results (default: %(default)s)')

    # others
    parser.add_argument('--cpu', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--silent', action='store_true',
                        help='Enable silent mode')
    parser.add_argument('--eval_sqrt_mode', action='store_true',
                        help='evaluate model every sqrt(len(dataloader)) steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Use multi-cpu core for data pre-processing (default: %(default)s)')
    #parser.add_argument('--embed_cache_dir', type=str,
    #                    help='For parameter search only: path to a directory for storing embeddings for multiple runs. (default: %(default)s)')
    parser.add_argument('--eval', action='store_true',
                        help='Only run evaluation on the test set (default: %(default)s)')
    parser.add_argument('--checkpoint_path',
                        help='The checkpoint to warm-up with (default: %(default)s)')
    parser.add_argument('-h', '--help', action='help')

    parser.set_defaults(**config)
    args = parser.parse_args()
    config = AttributeDict(vars(args))

    for i in ['trainL', 'trainR', 'validL', 'validR', 'testL', 'testR']:
        if config['%s_path'%i] is None:
            config['%s_path'%i] = os.path.join(config.data_dir, '%s.csv'%i)
    config['dataset_type'] = 'cross' if 'Minibatch' in config.loss else 'nonzero'
    return config


#def save_predictions(trainer, model, dataloader, predict_out_path):
#    batch_predictions = trainer.predict(model, dataloaders=dataloader)
#    pred_labels = np.vstack([batch['top_k_pred'] for batch in batch_predictions])
#    pred_scores = np.vstack([batch['top_k_pred_scores'] for batch in batch_predictions])
#    with open(predict_out_path, 'w') as fp:
#        for pred_label, pred_score in zip(pred_labels, pred_scores):
#            out_str = ' '.join([f'{model.classes[label]}:{score:.4}' for label, score in zip(pred_label, pred_score)])
#            fp.write(out_str+'\n')
#    logging.info(f'Saved predictions to: {predict_out_path}')

def data_proc(x, max_seq_len):
    x = [int(i.split(':')[0]) for i in x.split(',')]
    x = [101] + x + [102] + [0]*(max_seq_len-len(x)-3) + [102]
    return x


def setup_loggers(log_path:str, is_silent: bool):
    logging.basicConfig(
            filename=log_path,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt='%m-%d %H:%M',
            level=logging.DEBUG,
            force=True
            )

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    log_level = logging.WARNING if is_silent else logging.INFO
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    console.setLevel(log_level)
    logging.getLogger('').addHandler(console)

    return

def main():
    config = get_config()
    set_seed(seed=config.seed)

    config.device = init_device(use_cpu=config.cpu)
    config.pin_memory = 'cuda' in config.device.type
    config.run_name = '{}_{}_{}'.format(
        config.data_name,
        Path(config.config).stem if config.config else config.model_name,
        datetime.now().strftime('%Y%m%d%H%M%S'),
    )
    config['is_sogram'] = config.loss == 'Sogram'

    _Model = TwoTowerModel
    checkpoint_dir = os.path.join(config.result_dir, config.run_name)
    checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best_model',
            save_last=True, save_top_k=1,
            monitor=config.val_metric, mode='max'
            )
    earlystopping_callback = EarlyStopping(
            patience=config.patience,
            monitor=config.val_metric, mode='max'
            )
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataloader_factory = MNLoss.DataloaderFactory(
            config,
            partial(data_proc, max_seq_len=config.max_seq_len),
            partial(data_proc, max_seq_len=config.max_seq_len),
            #lambda x: [int(i.split(':')[0])+1 for i in x.split(',')], # idx 0 reserved for padding,
            #lambda x: [int(i.split(':')[0])+1 for i in x.split(',')],
            data_utils.generate_batch_cross,
            data_utils.generate_batch_nonzero if not config['is_sogram'] else data_utils.generate_batch_sogram,
            )
    dataloaders = dataloader_factory.get_loaders()
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']
    assert valid_loader is not None
    config['total_steps'] = config.epochs * len(train_loader)
    config['nnz'] = train_loader.dataset.nnz
    config['M'] = train_loader.dataset.M
    config['N'] = train_loader.dataset.N

    if config.loss == 'Sogram' and config.eval_sqrt_mode:
        val_check_interval = int(np.sqrt(len(train_loader)))
    else:
        val_check_interval = 1.0
    trainer = pl.Trainer(
            logger=False,
            num_sanity_val_steps=0,
            gpus=0 if config.cpu else 1,
            progress_bar_refresh_rate=0 if config.silent else 1,
            max_epochs=config.epochs,
            gradient_clip_val=config.gradient_clip_val,
            gradient_clip_algorithm=config.gradient_clip_algorithm,
            callbacks=[checkpoint_callback, earlystopping_callback],
            val_check_interval=val_check_interval,
            )

    setup_loggers(os.path.join(checkpoint_dir, 'log'), config.silent)
    logging.info(f'Run name: {config.run_name}')
    logging.debug(f'Config as:\n{config}')
    if config.eval:
        model = _Model.load_from_checkpoint(
                config.checkpoint_path,
                config=config,
                Y_eval=test_loader.dataset.Yu,
                )
    else:
        if config.checkpoint_path:
            model = _Model.load_from_checkpoint(
                    config.checkpoint_path,
                    config=config,
                    Y_eval=valid_loader.dataset.Yu,
                    )
        else:
            model = _Model(
                    config=config,
                    Y_eval=valid_loader.dataset.Yu,
                    )

        trainer.fit(model, train_loader, valid_loader)


    if test_loader is not None:
        logging.info(f'Loading best model from `{checkpoint_callback.best_model_path}`...')
        model = _Model.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                config=config,
                Y_eval=test_loader.dataset.Yu,
                )
        trainer.test(model, test_dataloaders=test_loader)


if __name__ == '__main__':
    wall_time = Timer()
    main()
    logging.info(f'Wall time: {wall_time.time():.2f} (s)')
