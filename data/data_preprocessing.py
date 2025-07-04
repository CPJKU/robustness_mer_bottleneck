import torch
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from paths import DATA_PATH_SOUNDTRACKS
from data.dataset_io import pickle_save, pickle_load
from madmom.audio import SignalProcessor, FramedSignalProcessor, LogarithmicFilteredSpectrogramProcessor

SR = 22050
FPS = 31.3
NUM_BANDS = 24

def opts_parser():
    """ Command line argument setup. """
    parser = argparse.ArgumentParser(description='Script that pre-computes spectrograms used for model training.')
    parser.add_argument('--cache_dir', type=str, required=False, default=DATA_PATH_SOUNDTRACKS + '/specs',
                        help='Directory to which pre-processed spectrograms should be saved to. Default: data_path / specs.')
    return parser


def set_random_seed(seed):
    # make everything deterministic (for choosing random spectrogram excerpts)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def slice_spec(spec, length, processor=None, offset_seconds=0):
    """ Returns random slice of spectrogram with given length (in seconds). """
    offset_frames = int(processor.times_to_frames(offset_seconds))
    length = int(length)

    # pad if necessary
    while spec.shape[-1] < offset_frames + length:
        spec = np.append(spec, spec[:, :length - spec.shape[-1]], axis=1)
    xlen = spec.shape[-1]

    # choose random segment
    k = torch.randint(offset_frames, xlen - length + 1, (1,))[0].item()
    output = spec[:, k: k + length]

    return output


class AudioProcessor:
    """ Processor that transforms audio file to spectrogram. """
    def __init__(self, frame_size=2048):
        self.hop_size = SR // FPS

        self.sig_proc = SignalProcessor(num_channels=1, sample_rate=SR, norm=True)
        self.fsig_proc = FramedSignalProcessor(frame_size=frame_size, hop_size=self.hop_size, fps=FPS, origin='future')
        self.spec_proc = LogarithmicFilteredSpectrogramProcessor(num_bands=NUM_BANDS, fmin=20, fmax=16000)

    def times_to_frames(self, times):
        """ Converts time (sec) to frames. """
        return np.floor(np.array(times) * SR / self.hop_size).astype(int)

    def frames_to_times(self, frames):
        """ Converts frames to time (sec). """
        return frames * self.hop_size / SR

    def compute_spectrogram(self, file_path):
        """ Loads audio data, windows it, computes spectrogram which is then filtered and logarithmically scaled before it is returned. """
        sig = np.trim_zeros(self.sig_proc.process(file_path))
        fsig = self.fsig_proc.process(sig)
        spec = self.spec_proc.process(fsig)
        return spec.transpose()


def run_preprocessing(opts):
    """ Main processing loop of data preprocessing. Computes spectrograms and stores them in a pickle file. """
    # prepare save path and audio processor, set random seed
    cache_dir = Path(opts.cache_dir)
    audio_processor = AudioProcessor()
    data_dir = Path(DATA_PATH_SOUNDTRACKS) / 'Set1'
    set_random_seed(42)

    # make sure data path exists, also create cache directory if necessary
    if not data_dir.exists():
        raise NotADirectoryError('{} is not a valid directory, make sure to set the path correctly'.format(data_dir))
    if not cache_dir.exists():
        cache_dir.mkdir(exist_ok=True)

    # then iterate through all data, store them
    file_names = sorted(list(data_dir.glob('*.mp3')))
    mean, std = 0.0, 0.0
    for f in tqdm(file_names, desc='Precompute spectrograms...'):
        # compute spectrogram
        spec = audio_processor.compute_spectrogram(str(f))
        # save to file
        pickle_save(cache_dir / (f.name + '.spec'), spec)
        # update mean
        mean += np.mean(spec).item()
    mean = mean / len(file_names)

    # compute average standard deviation for normalisation
    for f in tqdm(file_names, desc='Compute std for normalisation...'):
        spec = np.array(pickle_load(cache_dir / (f.name + '.spec')))
        std += np.mean(np.square(spec - mean)).item()
    std = np.sqrt(std / len(file_names))

    # after we pre-compute spectrograms, we can normalise
    for f in tqdm(file_names, desc='Normalise and slice spectrograms...'):
        # load
        spec = np.array(pickle_load(cache_dir / (f.name + '.spec')))
        # normalise
        spec = (spec - mean) / std
        # slice
        slice_length = audio_processor.times_to_frames(10)
        spec_slice = slice_spec(spec, slice_length, audio_processor)
        # overwrite file with normalised + sliced spectrogram
        pickle_save(cache_dir / (f.name + '.spec'), spec_slice.transpose())


def main():
    # parse arguments
    parser = opts_parser()
    opts = parser.parse_args()

    run_preprocessing(opts)


if __name__ == '__main__':
    main()
