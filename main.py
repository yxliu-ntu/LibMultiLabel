import argparse
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from functools import partial

import math
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning import loggers as pl_loggers

from libmultilabel import data_utils, MNLoss
from libmultilabel.two_tower_model import TwoTowerModel
from libmultilabel.utils import Timer, dump_log, init_device, set_seed


def get_config():
    parser = argparse.ArgumentParser(
        add_help=False,
        description='Extreme similarity learning')

    # load params from config file
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args, _ = parser.parse_known_args()
    config = {}
    if args.config:
        with open(args.config) as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

    # path / directory
    parser.add_argument('--data_dir', default='./data/ml-1m/',
                        help='The directory to load data (default: %(default)s)')
    parser.add_argument('--result_dir', default='./runs',
                        help='The directory to save checkpoints and logs (default: %(default)s)')
    #parser.add_argument('--tfboard_log_dir', default='./tfboard_logs',
    #                    help='The directory to save tensorboard logs (default: %(default)s)')

    # data
    parser.add_argument('--data_name', default='ml-1m',
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
    #parser.add_argument('--val_size', type=float, default=0.2,
    #                    help='Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set (default: %(default)s).')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether to shuffle training data before each epoch (default: %(default)s)')
    #parser.add_argument('--drop_last', type=bool, default=False,
    #                    help='Whether to drop the last batch each epoch (default: %(default)s)')

    # train
    parser.add_argument('--seed', type=int,
                        help='Random seed (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train (default: %(default)s)')
    #parser.add_argument('--warmup_steps', type=int, default=0.0,
    #                    help='Number of warm-up steps for training (default: %(default)s)')
    parser.add_argument('--bratio', type=float, default=None,
                        help='batch ratio of training samples for Sogram (default: %(default)s)')
    parser.add_argument('--bsize_i', type=int, default=16,
                        help='Size of training batches along rows of label matrix (default: %(default)s)')
    parser.add_argument('--bsize_j', type=int, default=None,
                        help='Size of training batches along cols of label matrix (default: %(default)s)')
    parser.add_argument('--optimizer', default='adagrad', choices=['adam', 'sgd', 'adamw', 'adagrad'],
                        help='Optimizer: SGD or Adam (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for optimizer (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay factor (default: %(default)s)')
    parser.add_argument('--l2_lambda', type=float, default=0,
                        help='L2 regularization factor (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum factor for SGD only (default: %(default)s)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to wait for improvement before early stopping (default: %(default)s)')
    parser.add_argument('--gradient_clip_val', type=float, default=0.0,
                        help='value for clipping gradient, 0 means don’t clip')
    parser.add_argument('--gradient_clip_algorithm', type=str, choices=['norm', 'value'], default='norm',
                        help='value means clip_by_value, norm means clip_by_norm. Default: norm')
    parser.add_argument('--loss', type=str,
                        choices=[
                            'Naive-LogSoftmax',
                            'Naive-LRLR',
                            'Naive-LRSQ',
                            'Linear-LR',
                            'Minibatch-LRSQ',
                            'Sogram-LRSQ',
                            ], 
                        default=None,
                        help='Type of loss function. All only support two-tower models.')
    parser.add_argument('--omega', type=float, default=1.0,
                        help='Cost weight for the negative part of the loss function')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for updating Gramian')
    parser.add_argument('--imp_r', type=float, default=0.0,
                        help='Imputed value for the negative part of the loss function')

    # model
    parser.add_argument('--model_name', default='ffm',
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
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Optional specification of dropout (default: %(default)s)')
    parser.add_argument('--isl2norm', action='store_true',
                        help='l2-normalize features by sample')

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

    ## log
    #parser.add_argument('--save_k_predictions', type=int, nargs='?', const=100, default=0,
    #                    help='Save top k predictions on test set. k=%(const)s if not specified. (default: %(default)s)')
    #parser.add_argument('--predict_out_path',
    #                    help='Path to the an output file holding top 100 label results (default: %(default)s)')

    # others
    parser.add_argument('--cpu', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--check_func_val', action='store_true',
                        help='Check function value when training')
    parser.add_argument('--silent', action='store_true',
                        help='Enable silent mode')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Use multi-cpu core for data pre-processing (default: %(default)s)')
    parser.add_argument('--eval', action='store_true',
                        help='Only run evaluation on the test set (default: %(default)s)')
    parser.add_argument('--checkpoint_path',
                        help='The checkpoint to warm-up with (default: %(default)s)')
    parser.add_argument('-h', '--help', action='help')

    parser.set_defaults(**config)
    args = parser.parse_args()
    config = AttributeDict(vars(args))

    #for i in ['trainL', 'trainR', 'validL', 'validR', 'testL', 'testR']:
    #    if config['%s_path'%i] is None:
    #        config['%s_path'%i] = os.path.join(config.data_dir, '%s.csv'%i)
    config['dataset_type'] = 'nonzero' if 'Sogram' in config.loss else 'cross'
    return config

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
    ## Load config
    config = get_config()
    set_seed(seed=config.seed)

    config.device = init_device(use_cpu=config.cpu)
    config.pin_memory = 'cuda' in config.device.type
    config.run_name = '{}_{}_{}_{}'.format(
        config.data_name,
        Path(config.config).stem if config.config else config.model_name,
        config.loss,
        datetime.now().strftime('%Y%m%d%H%M%S'),
    )
    config['is_sogram'] = 'Sogram' in config.loss

    ## Build model, set logger and checkpoint
    _Model = TwoTowerModel
    checkpoint_dir = os.path.join(config.result_dir, config.run_name)
    checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best_model',
            save_last=True,
            save_top_k=1,
            monitor=config.val_metric,
            mode='max' if config.val_metric != 'Aver-Rank' else 'min',
            )
    earlystopping_callback = EarlyStopping(
            patience=config.patience,
            monitor=config.val_metric,
            mode='max' if config.val_metric != 'Aver-Rank' else 'min',
            )
    #tb_logger = pl_loggers.TensorBoardLogger(config.tfboard_log_dir, name=config.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ## Set dataloader
    rng = np.random.default_rng(seed=1024) # CT, for code testing
    dataloader_factory = MNLoss.DataloaderFactory(
            config,
            data_utils.svm_data_proc,
            data_utils.svm_data_proc,
            data_utils.generate_batch_cross,
            data_utils.generate_batch_sogram,
            rng=rng,
            )
    dataloaders = dataloader_factory.get_loaders()
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']
    assert valid_loader is not None
    for loader in dataloaders:
        loader = dataloaders[loader]
        loader.dataset.U = data_utils.obj_arr_to_csr(loader.dataset.U)
        loader.dataset.V = data_utils.obj_arr_to_csr(loader.dataset.V)
    config['nnz'] = train_loader.dataset.nnz
    config['M'] = train_loader.dataset.U.shape[0]
    config['N'] = train_loader.dataset.V.shape[0]
    config['Du'] = train_loader.dataset.U.shape[1]
    config['Dv'] = train_loader.dataset.V.shape[1]
    print('M: %d, N: %d, Du: %d, Dv: %d'%(config.M, config.N, config.Du, config.Dv))

    val_check_interval = math.ceil(len(train_loader)/10.)
    config['total_steps'] = config.epochs * len(train_loader)
    trainer = pl.Trainer(
            logger=False,
            #logger=tb_logger,
            num_sanity_val_steps=0,
            gpus=0 if config.cpu else 1,
            progress_bar_refresh_rate=0 if config.silent else 1,
            max_steps=config.total_steps,
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
