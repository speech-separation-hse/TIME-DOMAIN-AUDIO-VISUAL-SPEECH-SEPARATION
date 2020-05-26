import sys
sys.path.append('../')

from utils.util import check_parameters
import time
import logging
from logger.set_logger import setup_logger
from model.loss import Loss
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import data_parallel
from model import model_rnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time import perf_counter
from apex import amp
from tqdm import tqdm
import random

def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer


class Trainer(object):
    def __init__(self, train_dataloader, val_dataloader, opt):
        super(Trainer).__init__()
        self.opt = opt
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_spks = opt['num_spks']
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.early_stop = opt['train']['early_stop']

        self.print_freq = opt['logger']['print_freq']
        # setup_logger(opt['logger']['name'], opt['logger']['path'],
        #             screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])
        self.checkpoint = opt['train']['path']
        self.name = opt['name']

        if opt['train']['gpuid']:
            self.logger.info('Load Nvida GPU .....')
            self.device = torch.device('cuda')#:{}'.format(opt['train']['gpuid'][0]))
            self.gpuid = opt['train']['gpuid']
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')

        self.dualrnn = model_rnn.Dual_RNN_model(**opt['Dual_Path_RNN']).to(self.device)

        self.logger.info(
            'Dual-Path-RNN parameters: {:.3f} Mb'.format(check_parameters(self.dualrnn)))

        if opt['resume']['state']:
            ckp = torch.load(opt['resume']['path'], map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            self.dualrnn.load_state_dict(ckp['model_state_dict'])
        optimizer = make_optimizer(self.dualrnn.parameters(), opt)
        if opt['resume']['state']:
            optimizer.load_state_dict(ckp['optim_state_dict'])

        self.dualrnn, optimizer = amp.initialize(self.dualrnn, optimizer)


        if opt['train']['distributed']:
            self.dualrnn = torch.nn.DataParallel(self.dualrnn)
        self.dualrnn = self.dualrnn.to(self.device)
        # build optimizer
        self.logger.info("Building the optimizer of Dual-Path-RNN")
        
        self.scheduler = ReduceLROnPlateau(
            optimizer, mode='min',
            factor=opt['scheduler']['factor'],
            patience=opt['scheduler']['patience'],
            verbose=True, min_lr=opt['scheduler']['min_lr'])
        self.optimizer = optimizer
        if opt['optim']['clip_norm']:
            self.clip_norm = opt['optim']['clip_norm']
            self.logger.info(
                "Gradient clipping by {}, default L2".format(self.clip_norm))
        else:
            self.clip_norm = 0


    def train(self, epoch):
        self.logger.info(
            'Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.dualrnn.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        num_index = 1
        start_time = time.time()
        pbar = tqdm(self.train_dataloader, leave=False)
        for batch in pbar:
            batch["first_videos_features"] = batch["first_videos_features"][:, :50, :].detach()
            batch["second_videos_features"] = batch["second_videos_features"][:, :50, :].detach()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            #mix = mix.to(self.device)
            ref = [batch[f"{i}_audios"] for i in ["first", "second"]]
            #ref = [ref[i].to(self.device) for i in range(self.num_spks)]
            self.optimizer.zero_grad()

            # if self.gpuid:
            #     out = torch.nn.parallel.data_parallel(self.dualrnn,mix,device_ids=self.gpuid)
            #     out = self.dualrnn(mix)
            # else:
            st_time = perf_counter()
            out = self.dualrnn(batch)#, batch["audios_lens"])
            st_time = perf_counter()
            l = Loss(out, ref)#, batch["audios_lens"])

            epoch_loss = l
            
            pbar.set_description(f"Loss: {round(l.item(), 4)}")
            #pbar.update(1)
            total_loss += epoch_loss.item()
            #epoch_loss.backward()
            with amp.scale_loss(epoch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.dualrnn.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = 'Train: <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                self.logger.info(message)
            num_index += 1
        end_time = time.time()
        total_loss = total_loss/num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        self.logger.info(message)
        return total_loss

    def validation(self, epoch):
        self.logger.info(
            'Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.dualrnn.eval()
        num_batchs = len(self.val_dataloader)
        num_index = 1
        total_loss = 0.0
        start_time = time.time()
        with torch.no_grad():
            pbar = tqdm(self.val_dataloader, desc='Val loop', leave=False)
            for batch in pbar:
                # mix = mix.to(self.device)
                # ref = [ref[i].to(self.device) for i in range(self.num_spks)]
                batch["first_videos_features"] = batch["first_videos_features"][:, :50, :].detach()
                batch["second_videos_features"] = batch["second_videos_features"][:, :50, :].detach()
                batch = {k: v.to(self.device) for k, v in batch.items()}

                ref = [batch[f"{i}_audios"] for i in ["first", "second"]]
                # self.optimizer.zero_grad()

                # if self.gpuid:
                #     #model = torch.nn.DataParallel(self.dualrnn)
                #     #out = model(mix)
                #     out = torch.nn.parallel.data_parallel(self.dualrnn,mix,device_ids=self.gpuid)
                # else:
                out = self.dualrnn(batch)#, batch["audios_lens"])
                
                l = Loss(out, ref)#, batch["audios_lens"])
                pbar.set_description(f"Loss: {round(l.item(), 4)}")
                epoch_loss = l
                total_loss += epoch_loss.item()
                if num_index:
                    message = 'Val: <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                    self.logger.info(message)
                num_index += 1
        end_time = time.time()
        total_loss = total_loss/num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        self.logger.info(message)

        audio_writer = [handler for handler in self.logger.handlers if hasattr(handler, 'write_audio')][0]
        
        audio_idx = random.randrange(0, out[0].size(0))

        audio_writer.write_audio((ref[0][audio_idx], ref[1][audio_idx]), (out[0][audio_idx], out[1][audio_idx]), batch['mix_audios'][audio_idx], batch["audios_lens"][audio_idx], epoch)


        return total_loss

    def run(self):
        train_loss = []
        val_loss = []
        #with torch.cuda.device(self.gpuid[0]):
        self.save_checkpoint(self.cur_epoch, best=False)
        v_loss = self.validation(self.cur_epoch)
        best_loss = v_loss
        
        self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
            self.cur_epoch, best_loss))
        no_improve = 0
        # starting training part
        while self.cur_epoch < self.total_epoch:
            t_loss = self.train(self.cur_epoch)
            self.cur_epoch += 1
            v_loss = self.validation(self.cur_epoch)

            train_loss.append(t_loss)
            val_loss.append(v_loss)

            # schedule here
            self.scheduler.step(v_loss)

            if v_loss >= best_loss:
                no_improve += 1
                self.logger.info(
                    'No improvement, Best Loss: {:.4f}'.format(best_loss))
            else:
                best_loss = v_loss
                no_improve = 0
                self.save_checkpoint(self.cur_epoch, best=True)
                self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(
                    self.cur_epoch, best_loss))

            if no_improve == self.early_stop:
                self.logger.info(
                    "Stop training cause no impr for {:d} epochs".format(
                        no_improve))
                break
        self.save_checkpoint(self.cur_epoch, best=False)
        self.logger.info("Training for {:d}/{:d} epoches done!".format(
            self.cur_epoch, self.total_epoch))

        # draw loss image
        plt.title("Loss of train and test")
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_loss, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(x, val_loss, 'c-', label=u'val_loss', linewidth=0.8)
        plt.legend()
        #plt.xticks(l, lx)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('loss.png')

    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.dualrnn.state_dict() if not self.opt['train']['distributed'] else self.dualrnn.module.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'amp_state_dict': amp.state_dict()
        }, os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best' if best else 'last')))
