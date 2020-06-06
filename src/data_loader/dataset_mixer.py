import random
from pathlib import Path
from typing import Union
from tqdm import tqdm
import sys
sys.path.append('./')


from data_loader import audio_reader
from config import option

import torch

from torch.utils.data import DataLoader, Dataset


class DatasetMixer:
    def __init__(self,
                 path_to_video: Path,
                 n_samples: int,
                 duration: int,
                 fps: Union[int, None] = None,
                 path_to_features: Union[Path, None] = None,
                 path_to_noise: Union[Path, None] = None,
                 reader: Union[audio_reader.AudioReader, None] = None,
                 deterministic: bool = True,
                 cache_directory: Union[Path, None] = None):
        self.path_to_video = path_to_video
        self.speakers: list[Path] = list(path_to_video.iterdir()) if self.path_to_video else None
        self.n_samples = n_samples
        self.audio_duration = duration * 1000
        self.video_duration = duration * fps if fps else None
        self.fps = fps
        self.path_to_features: Union[Path, None] = path_to_features
        self.noises: Union[list[Path], None] = list(path_to_noise.rglob('*.wav')) if path_to_noise else None
        self.reader: audio_reader.AudioReader = reader if reader else audio_reader.get_default_audio_reader()
        self.deterministic: bool = deterministic
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

            if first_audio_duration >= self.audio_duration and second_audio_duration >= self.audio_duration:
                break

        first_audio_offset = round(random.uniform(0, first_audio_duration - self.audio_duration) - 500, -3)
        second_audio_offset = round(random.uniform(0, second_audio_duration - self.audio_duration) - 500, -3)
        output = {
            'duration': self.audio_duration,
            'first_audio_offset': first_audio_offset,
            'second_audio_offset': second_audio_offset
        }

        first_audio = self.reader.slice(audio=first_audio, offset=first_audio_offset, duration=self.audio_duration)
        second_audio = self.reader.slice(audio=second_audio, offset=second_audio_offset, duration=self.audio_duration)
        mix_audio = self.reader.overlay(first_audio, second_audio)
        output.update({
            'first_audio': self.reader.to_tensor(first_audio),
            'second_audio': self.reader.to_tensor(second_audio),
            'mix_audio': self.reader.to_tensor(mix_audio)
        })

        if self.noises:
            noise_path = random.choice(self.noises)
            noise_audio = self.reader.load(noise_path, format='wav')
            noise_audio = self.reader.change_volume(noise_audio, by_db=9)
            noise_audio_duration = self.reader.length(noise_audio)
            noise_audio_offset = random.uniform(0, noise_audio_duration - self.audio_duration)
            noise_audio = self.reader.slice(audio=noise_audio, offset=noise_audio_offset, duration=self.audio_duration)
            mix_noised_audio = self.reader.overlay(mix_audio, noise_audio)
            mix_noised_audio.export(f"mix/mix_{index}_noised_audio_export.wav", format="wav")
            # import torchaudio
            # y, sr = torchaudio.load("mix_noised_audio_export.wav")
            # normalization constant from here: https://github.com/pytorch/audio/blob/313f4f5ca65b82ff81e62165ee2328a4c10de013/torchaudio/__init__.py#L415
            output.update({
                'mix_noised_audio': self.reader.to_tensor(mix_noised_audio, normalization=1 << 31),
                'noise_audio': self.reader.to_tensor(noise_audio, normalization=1 << 31)
            })

        if self.path_to_features:
            first_video_features_path = self.path_to_features / f'{first_speaker_path.stem}_{first_speaker_video_path.stem}.npy'
            first_video_features = torch.load(first_video_features_path, map_location='cpu').squeeze(1)
            second_video_features_path = self.path_to_features / f'{second_speaker_path.stem}_{second_speaker_video_path.stem}.npy'
            second_video_features = torch.load(second_video_features_path, map_location='cpu').squeeze(1)

            first_video_offset = int(first_audio_offset // 1000) * self.fps
            first_video_features = first_video_features[first_video_offset:first_video_offset + self.video_duration]
            second_video_offset = int(second_audio_offset // 1000) * self.fps
            second_video_features = second_video_features[second_video_offset:second_video_offset + self.video_duration]            
            output.update({
                'first_video_features': first_video_features,
                'second_video_features': second_video_features
            })

        if self.cache_directory:
            torch.save(output, cache_filename)

        return output

def audio_video_collate_fn(batch):
    """
    Collate's training batch
    """
    # Right zero-pad audios and videos
    return {
        'first_audios': torch.nn.utils.rnn.pad_sequence([x['first_audio'].T for x in batch], batch_first=True).transpose(2, 1).squeeze(1), 
        'second_audios': torch.nn.utils.rnn.pad_sequence([x['second_audio'].T for x in batch], batch_first=True).transpose(2, 1).squeeze(1),
        'mix_audios': torch.nn.utils.rnn.pad_sequence([x['mix_audio'].T for x in batch], batch_first=True).transpose(2, 1).squeeze(1), 
        'mix_noised_audios': torch.nn.utils.rnn.pad_sequence([x['mix_noised_audio'].T for x in batch], batch_first=True).transpose(2, 1).squeeze(1), 
        'audios_lens': torch.tensor([x['first_audio'].size(1) for x in batch]),
        'first_videos_features': torch.nn.utils.rnn.pad_sequence([x['first_video_features'] for x in batch], batch_first=True), 
        'second_videos_features': torch.nn.utils.rnn.pad_sequence([x['second_video_features'] for x in batch], batch_first=True),
        'videos_lens': torch.tensor([x['first_video_features'].size(0) for x in batch]),
        'noise_audios': torch.nn.utils.rnn.pad_sequence([x['noise_audio'].T for x in batch], batch_first=True).transpose(2, 1).squeeze(1), 
    }


if __name__ == '__main__':
    config_path = 'config/Dual_RNN/train.yaml'

    opt = option.parse(config_path)

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

    for _ in range(2):
        for x in tqdm(train_dataloader):
            pass

    for _ in range(2):
        for x in tqdm(val_dataloader):
            pass