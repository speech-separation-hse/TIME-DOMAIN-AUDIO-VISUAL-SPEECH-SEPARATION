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
from data_loader.dataset_mixer import DatasetMixer, audio_video_collate_fn

def make_dataloader(opt):
    # make train's dataloader
   
    train_dataset = DatasetMixer(
        path_to_video=Path(opt['datasets']['train']['video_path']),
        n_samples=opt['datasets']['train']['n_samples'],
        duration=opt['datasets']['duration'],
        fps=opt['datasets']['video_setting']['fps'],
        path_to_features=Path(opt['datasets']['train']['video_features_path']),
        path_to_noise=Path(opt['datasets']['train']['noise_path']),
        deterministic=opt['datasets']['train']['deterministic'],
        cache_directory=Path(opt['datasets']['train']['cache_path'])
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opt['datasets']['train']['batch_size'],
        collate_fn=audio_video_collate_fn,
        shuffle=opt['datasets']['dataloader_setting']['shuffle']
    )


    val_dataset = DatasetMixer(
        path_to_video=Path(opt['datasets']['val']['video_path']),
        n_samples=opt['datasets']['val']['n_samples'],
        duration=opt['datasets']['duration'],
        fps=opt['datasets']['video_setting']['fps'],
        path_to_features=Path(opt['datasets']['val']['video_features_path']),
        path_to_noise=Path(opt['datasets']['val']['noise_path']),
        deterministic=opt['datasets']['val']['deterministic'],
        cache_directory=Path(opt['datasets']['val']['cache_path'])
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=opt['datasets']['val']['batch_size'],
        collate_fn=audio_video_collate_fn,
        shuffle=opt['datasets']['dataloader_setting']['shuffle']
    )
    
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
