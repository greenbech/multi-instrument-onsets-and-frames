import os
import shutil
import sys
from glob import glob
from pathlib import Path
from random import shuffle
from time import sleep

import torch
import torchaudio
from openunmix import predict
from tqdm import tqdm


def save_separation(prediction, save_folder, in_sample_rate=44100, save_sample_rate=16000):
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    for key in prediction:
        # breakpoint()
        audio = prediction[key].squeeze(0)
        if in_sample_rate != save_sample_rate:
            audio = torchaudio.transforms.Resample(orig_freq=in_sample_rate, new_freq=save_sample_rate)(audio)
        audio = torch.mean(audio, 0, keepdims=True)
        target_path = os.path.join(save_folder, f"{key}.flac")
        torchaudio.save(
            target_path,
            audio.to("cpu"),
            sample_rate=save_sample_rate,
        )


def copy_track_files(org_path, new_path):
    Path(new_path).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.path.join(org_path, "all_src.mid"), os.path.join(new_path, "all_src.mid"))
    shutil.copyfile(os.path.join(org_path, "metadata.yaml"), os.path.join(new_path, "metadata.yaml"))
    shutil.copyfile(os.path.join(org_path, "mix.flac"), os.path.join(new_path, "mix.flac"))
    shutil.copytree(os.path.join(org_path, "MIDI"), os.path.join(new_path, "MIDI"), dirs_exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


def print_memory():
    sleep(0.2)
    # Debugging due to memory errors
    t = torch.cuda.get_device_properties(0).total_memory / 10 ** 9
    r = torch.cuda.memory_reserved(0) / 10 ** 9
    a = torch.cuda.memory_allocated(0) / 10 ** 9
    print(f"{r:.2f} / {a:.2f} / {t:.2f}")


print("Test print", file=sys.stderr)


def main():
    print(f"Running use_openunmix_on_slakh.py with device {device}", file=sys.stderr)
    save_sample_rate = 16000
    openunmix_sample_rate = 44100

    base_org_folder = "data/slakh2100_flac_16k"
    len_base_org_folder = len(base_org_folder.split(os.sep))
    new_folder = "data/slakh2100_flac_16k_umx"

    track_folders = glob(os.path.join(base_org_folder, "*", "Track*"))
    assert len(track_folders) == 2100
    # assert len(track_folders) == 1500

    created_track_folders = glob(os.path.join(new_folder, "*", "Track*"))
    created_track_folders_set = set((f.split(os.sep)[-1] for f in created_track_folders))
    track_folders = [f for f in track_folders if f.split(os.sep)[-1] not in created_track_folders_set]
    shuffle(track_folders)
    print(f"Processing {len(track_folders)} files", file=sys.stderr)

    separator = torch.hub.load("sigsep/open-unmix-pytorch", "umx", device=device)
    separator.eval()
    separator.freeze()
    with torch.no_grad():
        tqdm_data = tqdm(track_folders)
        for track_folder in tqdm_data:
            try:
                tqdm_data.set_description(desc=f"{track_folder}")
                print_memory()
                new_track_folder = os.path.join(new_folder, *track_folder.split(os.sep)[len_base_org_folder:])

                audio_path = os.path.join(track_folder, "mix.flac")
                audio_data, sample_rate = torchaudio.load(audio_path)
                audio_data.to(device)

                print(f"{sys.getsizeof(audio_data.storage()) / 10 ** 9}")

                print_memory()

                estimates = predict.separate(audio_data, separator=separator, rate=sample_rate, device=device)

                print_memory()

                for key in estimates:
                    estimates[key].to("cpu")
                audio_data.to("cpu")

                print_memory()

                save_separation(
                    prediction=estimates,
                    save_folder=os.path.join(new_track_folder, "stems"),
                    in_sample_rate=openunmix_sample_rate,
                    save_sample_rate=save_sample_rate,
                )
                copy_track_files(org_path=track_folder, new_path=new_track_folder)

                print_memory()
                sleep(0.5)
            except Exception:
                print(f"Unable to process {track_folder}")


main()
