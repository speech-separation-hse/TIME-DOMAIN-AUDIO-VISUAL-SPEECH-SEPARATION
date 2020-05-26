import random
from pathlib import Path
from typing import Union
from tqdm import tqdm

try:
    from data_loader import audio_reader
except:
    import audio_reader

import torch

from torch.utils.data import DataLoader, Dataset


class DatasetMixer:
    def __init__(self,
                 path_to_video: Path,
                 n_samples: int,
                 path_to_features: Union[Path, None] = None,
                 reader: Union[audio_reader.AudioReader, None] = None,
                 deterministic: bool = True,
                 cache_directory: Union[Path, None] = None):
        self.path_to_video = path_to_video
        self.speakers: list[Path] = list(path_to_video.iterdir())
        self.n_samples = n_samples
        self.path_to_features = path_to_features
        self.reader = reader if reader else audio_reader.get_default_audio_reader()
        self.deterministic = deterministic
        self.cache_directory = cache_directory
        if self.cache_directory:
            self.cache_directory.mkdir(exist_ok=True)
        assert not (self.cache_directory and not self.deterministic), 'Could not cache while not deterministic'

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index) -> dict:
        if self.cache_directory:
            cache_filename = self.cache_directory / f'{index}.pt'
            if cache_filename.is_file():
                output = torch.load(cache_filename)
                return output

        if self.deterministic:
            random.seed(index)

        while True:

            first_speaker_path, second_speaker_path = random.sample(self.speakers, 2)

            first_speaker_video_path = random.choice(list(first_speaker_path.rglob('*.mp4')))
            second_speaker_video_path = random.choice(list(second_speaker_path.rglob('*.mp4')))

            #output = {'first_video_path': first_speaker_video_path, 'second_video_path': second_speaker_video_path}

            first_audio = self.reader.load(first_speaker_video_path)
            second_audio = self.reader.load(second_speaker_video_path)

            first_audio_duration = self.reader.length(first_audio)
            second_audio_duration = self.reader.length(second_audio)


            if first_audio_duration >= 2000 and second_audio_duration >= 2000:
                break
            #return self.__getitem__(random.randint(0, self.__len__()))
        min_duration = 2000#min(first_audio_duration, second_audio_duration, 3000)
        first_audio_offset = round(random.uniform(0, first_audio_duration - min_duration) - 500, -3)
        second_audio_offset = round(random.uniform(0, second_audio_duration - min_duration) -500, -3)
        output = {
            'duration': min_duration,
            'first_audio_offset': first_audio_offset,
            'second_audio_offset': second_audio_offset
        }

        first_audio = self.reader.slice(audio=first_audio, offset=first_audio_offset, duration=min_duration)
        second_audio = self.reader.slice(audio=second_audio, offset=second_audio_offset, duration=min_duration)
        mix_audio = self.reader.overlay(first_audio, second_audio)
        output.update({
            'first_audio': self.reader.to_tensor(first_audio),
            'second_audio': self.reader.to_tensor(second_audio),
            'mix_audio': self.reader.to_tensor(mix_audio)
        })

        if self.path_to_features:
            first_video_features_path = self.path_to_features / f'{first_speaker_path.stem}_{first_speaker_video_path.stem}.npy'
            first_video_features = torch.load(first_video_features_path, map_location='cpu').squeeze(1)
            second_video_features_path = self.path_to_features / f'{second_speaker_path.stem}_{second_speaker_video_path.stem}.npy'
            second_video_features = torch.load(second_video_features_path, map_location='cpu').squeeze(1)

            min_frames = min(first_video_features.shape[0], second_video_features.shape[0], 50)
            first_video_offset = int(first_audio_offset // 1000)*25
            first_video_features = first_video_features[first_video_offset:first_video_offset + min_frames]
            second_video_offset = int(second_audio_offset // 1000)*25
            second_video_features = second_video_features[second_video_offset:second_video_offset + min_frames]            
            output.update({
                'first_video_features': first_video_features,
                'second_video_features': second_video_features
            })



        if self.cache_directory:
            torch.save(output, cache_filename)

        return output

def AudioVideoCollate(batch):
    """Collate's training batch
    PARAMS
    ------
    batch:
    """
    # Right zero-pad audios and videos

    max_audio_size = max(x['first_audio'].size(1) for x in batch)
    max_video_size = max(x['first_video_features'].size(0) for x in batch)
    return {
        'first_audios': torch.nn.utils.rnn.pad_sequence([x['first_audio'].T for x in batch], batch_first=True).transpose(2, 1).squeeze(1), #torch.stack([torch.cat((x['first_audio'], torch.zeros(x['first_audio'].size(0), max_audio_size - x['first_audio'].size(1))), 1) for x in batch], 0),
        'second_audios': torch.nn.utils.rnn.pad_sequence([x['second_audio'].T for x in batch], batch_first=True).transpose(2, 1).squeeze(1), #torch.stack([torch.cat((x['second_audio'], torch.zeros(x['second_audio'].size(0), max_audio_size - x['second_audio'].size(1))), 1) for x in batch], 0),
        'mix_audios': torch.nn.utils.rnn.pad_sequence([x['mix_audio'].T for x in batch], batch_first=True).transpose(2, 1).squeeze(1), #torch.stack([torch.cat((x['mixed_audio'], torch.zeros(x['mixed_audio'].size(0), max_audio_size - x['mixed_audio'].size(1))), 1) for x in batch], 0),
        'audios_lens': torch.tensor([x['first_audio'].size(1) for x in batch]),
        'first_videos_features': torch.nn.utils.rnn.pad_sequence([x['first_video_features'] for x in batch], batch_first=True),#torch.stack([torch.cat((x['first_video_features'], torch.zeros(max_video_size - x['first_video_features'].size(0), x['first_video_features'].size(1))), 0) for x in batch], 0),
        'second_videos_features': torch.nn.utils.rnn.pad_sequence([x['second_video_features'] for x in batch], batch_first=True),#torch.stack([torch.cat((x['second_video_features'], torch.zeros(max_video_size - x['second_video_features'].size(0), x['second_video_features'].size(1))), 0) for x in batch], 0),
        'videos_lens': torch.tensor([x['first_video_features'].size(0) for x in batch]),
    }


if __name__ == '__main__':
    path_to_video = Path('../trainval')
    path_to_features = Path('../video_features')

    dataset = DatasetMixer(path_to_video=path_to_video, n_samples=100000, path_to_features=path_to_features, cache_directory=Path('train_caches'))
    dataloader = DataLoader(dataset, batch_size=48, collate_fn=AudioVideoCollate)
    # #cnt = 0
    # def f(x):
    #     return None

    # from joblib import Parallel, delayed
    # Parallel(n_jobs=48*5)(delayed(f)(x) for x in tqdm(dataset))

    for _ in range(2):
        for x in tqdm(dataloader):
            pass
    # print(cnt/180000)


    path_to_video = Path('../test')
    path_to_features = Path('../video_features_test')

    dataset = DatasetMixer(path_to_video=path_to_video, n_samples=6000, path_to_features=path_to_features, cache_directory=Path('test_caches'))
    dataloader = DataLoader(dataset, batch_size=48, collate_fn=AudioVideoCollate)
    
    for _ in range(2):
        for x in tqdm(dataloader):
            pass


    # print(cnt)

    # result2 = dataset[10]

    # result2 = dataset[11]

    # assert result.keys() == result2.keys()

    # import timeit

    # times = timeit.repeat(lambda: dataset[10], repeat=10, number=1)

    # print(min(times))