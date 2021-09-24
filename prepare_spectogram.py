import os
import numpy as np
import argparse
import audio
from hparams import create_hparams

def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def save_txt(path, txt):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelists_directory", type=str, default="/content/tacotron2/filelists", help="")
    parser.add_argument("--data_directory", type=str, default="/content/tacotron2/data", help="")
    parser.add_argument("--mel_directory", type=str, default="meian_spectrogram", help="")

    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate.")
    parser.add_argument("--num_fft", type=int, default=1102, help="Number of FFT frequencies.")
    parser.add_argument("--num_mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument("--stft_window_ms", type=float, default=50, help="STFT window size.")
    parser.add_argument("--stft_shift_ms", type=float, default=12.5, help="STFT window shift.")
    parser.add_argument("--no_preemphasis", action='store_false', help="Do not use preemphasis.")
    parser.add_argument("--preemphasis", type=float, default=0.97, help="Strength of preemphasis.")

    args = parser.parse_args()

    hparams = create_hparams()

    data_directory = args.data_directory
    mel_directory = os.path.join(data_directory, args.mel_directory)
    train_path = os.path.join(args.filelists_directory, hparams.training_files)
    train_old = load_txt(train_path)
    train_new = []

    os.makedirs(mel_directory, exist_ok=True)

    for line in train_old:
        wav_path, text = line.split('|')
        mel_name = wav_path.split('/')[-1][:-len('.wav')] + '.npy'

        mel_path = os.path.join(mel_directory, mel_name)

        audio_path = os.path.join(data_directory, wav_path)
        audio_data = audio.load(audio_path)

        train_new.append(f"{os.path.join(args.mel_directory, mel_name)}|{text}")

        np.save(mel_path, audio.spectrogram(audio_data, True))

    save_txt(train_path, '\n'.join(train_new))
