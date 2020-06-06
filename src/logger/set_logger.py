import logging
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import re


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

class TensorboardHandler(logging.StreamHandler):
    def __init__(self, num_iter_train, num_iter_val, sr):
        super().__init__()
        self.summary_writer = SummaryWriter()
        self.num_iter_train = num_iter_train
        self.num_iter_val = num_iter_val
        self.sr = sr

    def write_audio(self, audio_real, audio_predicted, audio_mix, audio_len, noise, global_step):
        self.summary_writer.add_audio(f"audio/real_1", audio_real[0][:audio_len].unsqueeze(0), global_step, sample_rate = self.sr)
        self.summary_writer.add_audio(f"audio/real_2", audio_real[1][:audio_len].unsqueeze(0), global_step, sample_rate = self.sr)
        self.summary_writer.add_audio(f"audio/pred_1", (audio_predicted[0][:audio_len] / audio_predicted[0].max()).unsqueeze(0), global_step, sample_rate = self.sr)
        self.summary_writer.add_audio(f"audio/pred_2", (audio_predicted[1][:audio_len] / audio_predicted[1].max()).unsqueeze(0), global_step, sample_rate = self.sr)
        self.summary_writer.add_audio(f"audio/mix", audio_mix[:audio_len].unsqueeze(0), global_step, sample_rate = self.sr)
        self.summary_writer.add_audio(f"audio/noise", noise[:audio_len].unsqueeze(0), global_step, sample_rate = self.sr)

    def emit(self, record):
        try:
            record = self.format(record)
            if record.startswith("Train:"):
                n_iter = self.num_iter_train 
                part = "train"
            elif record.startswith("Val:"):
                n_iter = self.num_iter_val 
                part = "val"
            else:
                return

            n_iter = n_iter * int(record.partition('epoch:')[2].partition(',')[0]) + \
                    int(record.partition('iter:')[2].partition(',')[0])

            self.summary_writer.add_scalar(f"loss/{part}", float(record.partition('loss:')[2].partition('>')[0]), n_iter)
            self.summary_writer.add_scalar(f"lr/{part}", float(record.partition('lr:')[2].partition(',')[0]), n_iter)


        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False, tensorboard = False,
                 num_iter_train=0, num_iter_val=0, sr=8000):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    os.makedirs(root,exist_ok=True)
    if tofile:
        log_file = os.path.join(root, '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

    if tensorboard:
        sh = TensorboardHandler(num_iter_train, num_iter_val, sr)
        lg.addHandler(sh)


if __name__ == "__main__":
    setup_logger('base','root',level=logging.INFO,screen=True, tofile=False)
    logger = logging.getLogger('base')
    logger.info('hello')
