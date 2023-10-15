import librosa
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, valid_idxs=None, name='AudioCaps'):
        """
        Load audio clip's waveform.
        Args:
            name: 'AudioCaps', 'Clotho
        """
        super(AudioDataset, self).__init__()
        self.name = name
        audio_dir_prefix = f'{name.lower()}/waveforms'
        audio_dirs = [f'{audio_dir_prefix}/train/', f'{audio_dir_prefix}/test/', f'{audio_dir_prefix}/val/']
        self.audio_paths = [os.path.join(audio_dir, f) for audio_dir in audio_dirs for f in os.listdir(audio_dir)]
        self.audio_names = [Path(audio_path).stem for audio_path in self.audio_paths]
        if valid_idxs is not None:
            self.audio_paths = [self.audio_paths[idx] for idx in valid_idxs]
            self.audio_names = [self.audio_names[idx] for idx in valid_idxs]

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        audio_name = self.audio_names[idx]
        audio_path = self.audio_paths[idx]
        audio, _ = librosa.load(self.audio_paths[idx], sr=32000, mono=True)
        audio = AudioDataset.pad_or_truncate(audio, 32000 * 10)
        return audio, audio_name, audio_path, idx, len(audio)

    @staticmethod
    def pad_or_truncate(audio, audio_length):
        """Pad all audio to specific length."""
        length = len(audio)
        if length <= audio_length:
            return np.concatenate((audio, np.zeros(audio_length - length)), axis=0)
        else:
            return audio[:audio_length]