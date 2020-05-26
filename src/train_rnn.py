import sys
sys.path.append('./')

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
#from data_loader.Dataset import Datasets
from model import model_rnn
from logger import set_logger
import logging
from config import option
import argparse
import torch
from pathlib import Path
from trainer import trainer_Dual_RNN
from data_loader.dataset_mixer import DatasetMixer, AudioVideoCollate

def make_dataloader(opt):
    # make train's dataloader
    path_to_video = Path('../trainval')
    path_to_features = Path('../video_features')

    train_dataset = DatasetMixer(path_to_video=path_to_video, n_samples=100000, path_to_features=path_to_features, cache_directory=Path('train_caches'))
    train_dataloader = DataLoader(train_dataset, batch_size=36, collate_fn=AudioVideoCollate, shuffle=True)


    # train_dataset = Datasets(
    #     Path(opt['datasets']['train']['dataroot']),
    #     opt['datasets']['train']['dataroot_csv'],
    #     **opt['datasets']['audio_setting'])
    # train_dataloader = Loader(train_dataset,
    #                           batch_size=opt['datasets']['dataloader_setting']['batch_size'],
    #                           num_workers=opt['datasets']['dataloader_setting']['num_workers'],
    #                           shuffle=opt['datasets']['dataloader_setting']['shuffle'])

    
    # make validation dataloader


    path_to_val_video = Path('../test')
    path_to_val_features = Path('../video_features_test')

    val_dataset = DatasetMixer(path_to_video=path_to_val_video, n_samples=6000, path_to_features=path_to_val_features, cache_directory=Path('test_caches'))
    val_dataloader = DataLoader(val_dataset, batch_size=120, collate_fn=AudioVideoCollate, shuffle=True)
    
    # val_dataset = Datasets(
    #     Path(opt['datasets']['val']['dataroot']),
    #     opt['datasets']['val']['dataroot_csv'],
    #     **opt['datasets']['audio_setting'])
    # val_dataloader = Loader(val_dataset,
    #                         batch_size=opt['datasets']['dataloader_setting']['batch_size'] * 8,
    #                         num_workers=opt['datasets']['dataloader_setting']['num_workers'],
    #                         shuffle=False)
    
    return train_dataloader, val_dataloader


def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training Dual-Path-RNN')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.', default='config/Dual_RNN/train.yaml')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    # build dataloader
    print('Building the dataloader of Dual-Path-RNN')
    train_dataloader, val_dataloader = make_dataloader(opt)


    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'], tensorboard=opt['logger']['tofile'],
                            num_iter_train=len(train_dataloader), num_iter_val=len(val_dataloader), sr=opt['datasets']['audio_setting']['sample_rate'])
    logger = logging.getLogger(opt['logger']['name'])
    # build model
    logger.info("Building the model of Dual-Path-RNN")

    logger.info('Train Datasets Length: {}, Val Datasets Length: {}'.format(
        len(train_dataloader), len(val_dataloader)))
    
    # build trainer
    logger.info('Building the Trainer of Dual-Path-RNN')
    trainer = trainer_Dual_RNN.Trainer(train_dataloader, val_dataloader, opt)
    trainer.run()


if __name__ == "__main__":
    train()
