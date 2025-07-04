import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from data.dataset_io import pickle_load
from paths import DATA_PATH_SOUNDTRACKS, ANNOTATIONS_PATH_MIDLEVEL


class EmotionDataset(Dataset):
    """ Dataset containing Soundtracks data, with both emotion and mid-level feature annotations. """
    def __init__(self, dataset_path=DATA_PATH_SOUNDTRACKS, midlevel_path=ANNOTATIONS_PATH_MIDLEVEL):
        """
        Parameters
        ----------
        dataset_path : str
            Path pointing to Soundtracks dataset containing emotion data.
        midlevel_path : str
            Path pointing to mid-level metadata-annotation information.
        """
        self.dataset_path = Path(dataset_path)
        self.midlevel_meta_path = Path(midlevel_path)
        # this is path where pre-processed spectrograms should be stored, change if needed
        self.soundtrack_spec_path = self.dataset_path / 'specs'

        self.filepaths, self.annotations = self.load_soundtracks()
        self.ml_targets, self.ids_at_source = self.load_midlevel_subsets()

    def __len__(self):
        """ Returns length of dataset (= number of audio-files). """
        return len(self.filepaths)

    def __getitem__(self, idx):
        """ Given index, returns according path to audio sample, audio data, emotion and mid-level labels. """
        audio_path = self.filepaths[idx]
        x = pickle_load(audio_path)[:313, :]
        labels = self.annotations[idx]

        ml_targets = self.ml_targets[idx]
        sid = self.ids_at_source[idx]
        if not int(sid) == int(Path(audio_path).name[:3]):
            raise ValueError('Something went wrong, {} and {} should be the same'.format(sid, Path(audio_path).name[:3]))

        return audio_path, torch.tensor(x), np.vstack(labels), np.vstack(ml_targets)

    def load_soundtracks(self):
        """ Loads all Soundtrack filepaths and according emotion annotations. """
        annotations_df = pd.read_csv(self.dataset_path / 'mean_ratings_set1.csv')
        filepaths = []
        annotations = []

        for song_number in annotations_df['number']:
            file = self.soundtrack_spec_path / (f'00{song_number}'[-3:] + '.mp3.spec')
            filepaths.append(str(file))
            annotations.append(np.asarray(annotations_df[annotations_df['number'] == song_number])[0][1:-1] / 10)

        filepaths = np.hstack(filepaths)
        annotations = np.vstack(annotations)
        return [filepaths, annotations]

    def load_midlevel_subsets(self):
        """ Loads mid-level feature annotations for all songs in Soundtrack dataset (and mapping to the original song IDs). """
        subset = 'soundtracks'
        meta = pd.read_csv(self.midlevel_meta_path / 'metadata.csv', sep=';')
        annotations = pd.read_csv(self.midlevel_meta_path / 'annotations.csv')

        if not np.all(meta['song id'] == annotations['song_id']):
            raise ValueError('Something is wrong with the meta/annotation files - Song IDs do not match.')

        tracks_in_source = meta[meta['Source'] == subset]
        selected_train_set = annotations[annotations['song_id'].isin(tracks_in_source['song id'])]

        targets, ids_at_source = [], []
        for song_id in selected_train_set['song_id']:
            song_targets = np.asarray(
                selected_train_set[selected_train_set['song_id'] == song_id][selected_train_set.columns[1:]].values[0])
            sid = meta[meta['song id'] == song_id]['ID at file source'].values[0]
            targets.append(song_targets)
            ids_at_source.append(sid)

        targets = np.asarray(targets, dtype=np.float32) / 10

        return targets, ids_at_source
