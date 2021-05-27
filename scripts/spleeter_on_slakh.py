import os
import shutil
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from tqdm import tqdm


def save_separation(prediction, save_folder, sample_rate=16000):
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    for key in prediction:
        audio = np.mean(prediction[key], 1, keepdims=True)
        sf.write(
            os.path.join(save_folder, f"{key}.flac"),
            audio,
            samplerate=sample_rate,
        )


def copy_track_files(org_path, new_path):
    Path(new_path).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.path.join(org_path, "all_src.mid"), os.path.join(new_path, "all_src.mid"))
    shutil.copyfile(os.path.join(org_path, "metadata.yaml"), os.path.join(new_path, "metadata.yaml"))
    shutil.copyfile(os.path.join(org_path, "mix.flac"), os.path.join(new_path, "mix.flac"))
    shutil.copytree(os.path.join(org_path, "MIDI"), os.path.join(new_path, "MIDI"), dirs_exist_ok=True)


def main():
    separator = Separator("spleeter:4stems")
    audio_loader = AudioAdapter.default()
    sample_rate = 16000

    base_org_folder = "data/slakh2100_flac_16k"
    len_base_org_folder = len(base_org_folder.split(os.sep))
    new_folder = "data/slakh2100_flac_16k_spleeter"

    track_folders = glob(os.path.join(base_org_folder, "*", "Track*"))
    assert len(track_folders) == 2100

    for track_folder in tqdm(track_folders):
        new_track_folder = os.path.join(new_folder, *track_folder.split(os.sep)[len_base_org_folder:])
        copy_track_files(org_path=track_folder, new_path=new_track_folder)

        audio_path = os.path.join(track_folder, "mix.flac")
        waveform, _ = audio_loader.load(audio_path, sample_rate=sample_rate)
        prediction = separator.separate(waveform)
        save_separation(
            prediction=prediction,
            save_folder=os.path.join(new_track_folder, "stems"),
            sample_rate=sample_rate,
        )


main()
