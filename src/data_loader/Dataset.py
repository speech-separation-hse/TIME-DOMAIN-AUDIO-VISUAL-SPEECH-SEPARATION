from pathlib import Path
import pandas as pd
from data_loader.AudioData import read_wav
import torch
from torch.utils.data import Dataset

import numpy as np


class Datasets(Dataset):
    '''
       Load audio data
       data_path (Path): folder path of all audio
       audio_csv_path: path to csv with all audio info (type: str)
       sample_rate (int, optional): sample rate (default: 8000)
       chunk_size (int, optional): split audio size (default: 4(s))
    '''

    def __init__(self, data_path, audio_csv_path, sample_rate=8000, chunk_size=4):
        super(Datasets, self).__init__()
        assert isinstance(data_path, Path)
        self.data_path = data_path
        self.audio_info = pd.read_csv(audio_csv_path)
        self.sample_rate = sample_rate
        self.chunk_size = sample_rate * chunk_size

    def __len__(self):
        return len(self.audio_info.index)

    def __getitem__(self, index):
        info = self.audio_info.iloc[index]
        duration = info['duration']
        offset = np.random.randint(0, int(duration * self.sample_rate - self.chunk_size))
        mix_audio = read_wav(self.data_path / info['mix_path'], offset=offset, num_frames=self.chunk_size)
        first_audio = read_wav(self.data_path / info['first_speaker_path'], offset=offset, num_frames=self.chunk_size)
        second_audio = read_wav(self.data_path / info['second_speaker_path'], offset=offset, num_frames=self.chunk_size)

        return mix_audio, [first_audio, second_audio]


if __name__ == "__main__":
    data_path = Path('..') / 'data'
    dataset = Datasets(data_path=data_path, audio_csv_path=data_path / 'test.csv')
    print(dataset[1])
