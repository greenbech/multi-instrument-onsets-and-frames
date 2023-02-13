import hashlib
import os
from abc import abstractmethod
from collections import defaultdict
from glob import glob
from typing import NamedTuple

import numpy as np
import pretty_midi
import soundfile
import torch
import torchaudio
import yaml
from torch.utils.data import Dataset

from .constants import (
    DEFAULT_DEVICE,
    HOP_LENGTH,
    HOPS_IN_OFFSET,
    HOPS_IN_ONSET,
    MAX_MIDI,
    MIN_MIDI,
    SAMPLE_RATE,
)
from .data_classes import AudioAndLabels, MusicAnnotation
from .midi import parse_midi

# torchaudio.set_audio_backend("sox_io")
torchaudio.set_audio_backend("soundfile")


class Labels(NamedTuple):
    # path to audio file
    path: str
    # a matrix that contains the onset/offset/frame labels encoded as:
    # 3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
    label: torch.ByteTensor  # [num_steps, midi_bins]
    # a matrix that contains MIDI velocity values at the frame locations
    velocity: torch.ByteTensor  # [num_steps, midi_bins]


def get_instruments_from_names(names):
    if names == "bass":
        return list(range(32, 37))
    if names == "other":
        return list(range(0, 112))
    raise RuntimeError()


class PianoRollAudioDataset(Dataset):
    def __init__(
        self,
        path,
        instruments: str,
        groups=None,
        sequence_length=None,
        seed=42,
        device=DEFAULT_DEVICE,
        num_files=None,
        max_files_in_memory=-1,
        reproducable_load_sequences=False,
    ):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.instruments = instruments

        self.file_list = []
        for group in groups:
            for file in self.files(group):
                if num_files is not None and len(self.file_list) > num_files:
                    break
                self.file_list.append(file)
        self.labels = [None] * len(self.file_list)

        self.max_files_in_memory = max_files_in_memory
        if self.max_files_in_memory > 0:
            self.audios = [None] * self.max_files_in_memory
        self.reproducable_load_sequences = reproducable_load_sequences

    def __getitem__(self, index) -> AudioAndLabels:
        audio_path, tsv_path = self.file_list[index]
        audio = None
        if index < self.max_files_in_memory:
            audio = self.audios[index]

            # The first time the audio needs to be loaded in memory
            if audio is None:
                audio = torch.tensor(soundfile.read(audio_path, dtype="float32")[0])
                self.audios[index] = audio

        labels: Labels = self.labels[index]
        # The first the labels needs to be loaded in memory
        if labels is None:
            labels = self.load_labels(audio_path, tsv_path)
            self.labels[index] = labels

        if self.sequence_length is not None:
            audio_length = torchaudio.info(audio_path).num_frames
            possible_start_interval = audio_length - self.sequence_length
            if self.reproducable_load_sequences:
                step_begin = int(hashlib.sha256(audio_path.encode("utf-8")).hexdigest(), 16) % possible_start_interval
            else:
                step_begin = self.random.randint(possible_start_interval)
            step_begin //= HOP_LENGTH

            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length
            num_frames = end - begin

            if audio is None:
                # audio = torchaudio.load(audio_path, frame_offset=begin, num_frames=num_frames)[0].to(self.device)
                audio = torch.tensor(soundfile.read(audio_path, start=begin, dtype="float32", frames=num_frames)[0]).to(
                    self.device
                )
            else:
                audio = audio[begin:end].to(self.device)
            label = labels.label[step_begin:step_end, :].to(self.device)
            velocity = labels.velocity[step_begin:step_end, :].to(self.device)
        else:
            if audio is None:
                # audio = torchaudio.load(audio_path, frame_offset=begin, num_frames=num_frames)[0].to(self.device)
                audio = torch.tensor(soundfile.read(audio_path, start=0, dtype="float32", frames=-1)[0]).to(self.device)
            else:
                audio = audio.to(self.device)
            label = labels.label.to(self.device)
            velocity = labels.velocity.to(self.device).float()

        audio = audio.float().div_(32768.0)
        onset = (label == 3).float()
        offset = (label == 1).float()
        frame = (label > 1).float()
        velocity = velocity.float().div_(128.0)

        return AudioAndLabels(
            path=labels.path,
            audio=audio,
            annotation=MusicAnnotation(onset=onset, offset=offset, frame=frame, velocity=velocity),
        )

    def __len__(self):
        return len(self.file_list)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load_labels(self, audio_path: str, tsv_path: str) -> Labels:

        saved_data_path = tsv_path.replace(".tsv", ".pt")
        if os.path.exists(saved_data_path):
            label_dict = torch.load(saved_data_path)
        else:
            audio_length = torchaudio.info(audio_path).num_frames

            n_keys = MAX_MIDI - MIN_MIDI + 1
            n_steps = (audio_length - 1) // HOP_LENGTH + 1

            label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
            velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

            midi = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)

            if midi.size != 0:
                if midi.shape[1] == 4:
                    for onset, offset, note, vel in midi:
                        left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                        onset_right = min(n_steps, left + HOPS_IN_ONSET)
                        frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                        frame_right = min(n_steps, frame_right)
                        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                        f = int(note) - MIN_MIDI
                        label[left:onset_right, f] = 3
                        label[onset_right:frame_right, f] = 2
                        label[frame_right:offset_right, f] = 1
                        velocity[left:frame_right, f] = vel
                elif midi.shape[1] == 5:
                    for instrument, onset, offset, note, vel in midi:
                        left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                        onset_right = min(n_steps, left + HOPS_IN_ONSET)
                        frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                        frame_right = min(n_steps, frame_right)
                        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                        f = int(note) - MIN_MIDI
                        label[left:onset_right, f] = 3
                        label[onset_right:frame_right, f] = 2
                        label[frame_right:offset_right, f] = 1
                        velocity[left:frame_right, f] = vel
                else:
                    raise RuntimeError(f"Unsupported tsv shape {midi.shape}")
            label_dict = dict(path=audio_path, label=label, velocity=velocity)
            torch.save(label_dict, saved_data_path)
        return Labels(path=audio_path, label=label_dict["label"], velocity=label_dict["velocity"])


class Slakh(PianoRollAudioDataset):
    def __init__(
        self,
        path="data/slakh2100_wav_16k",
        instruments="bass",
        groups=None,
        sequence_length=None,
        seed=42,
        device=DEFAULT_DEVICE,
        num_files=None,
        max_files_in_memory=-1,
        reproducable_load_sequences=False,
    ):
        super().__init__(
            path,
            instruments,
            groups if groups is not None else ["train"],
            sequence_length,
            seed,
            device,
            num_files,
            max_files_in_memory,
            reproducable_load_sequences,
        )

    @classmethod
    def available_groups(cls):
        return ["train", "validation", "test"]

    def files(self, group):
        audio_files = sorted(glob(os.path.join(self.path, group, "*", "mix.wav")))
        midis = sorted(glob(os.path.join(self.path, group, "*", "all_src.mid")))
        yamls = sorted(glob(os.path.join(self.path, group, "*", "metadata.yaml")))
        files = list(zip(audio_files, midis, yamls))
        if len(files) == 0:
            raise RuntimeError(f"Group {group} is empty")

        result = []
        for audio_path, midi_path, yaml_path in files:
            tail, head = os.path.split(midi_path)
            tsv_filename = os.path.join(tail, head.replace(".mid", "") + f"_{self.instruments}.tsv")
            if not os.path.exists(tsv_filename):
                midi_program_to_remove = self._not_rendered_midi_program(yaml_path)

                extract_instruments = get_instruments_from_names(self.instruments)
                midi = parse_midi(midi_path, extract_instruments, remove_midi_programs=midi_program_to_remove)
                np.savetxt(
                    tsv_filename, midi, fmt="%.6f", delimiter="\t", header="instrument\tonset\toffset\tnote\tvelocity"
                )
            result.append((audio_path, tsv_filename))
        return result

    def _not_rendered_midi_program(self, yaml_path):
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        num_midi_programs = defaultdict(int)
        not_rendered_midi_programs = defaultdict(int)
        for key in yaml_data["stems"]:
            data = yaml_data["stems"][key]
            if not data["audio_rendered"]:
                not_rendered_midi_programs[data["program_num"]] += 1

            num_midi_programs[data["program_num"]] += 1

        for m in not_rendered_midi_programs:
            if num_midi_programs[m] != not_rendered_midi_programs[m]:
                print([(i, pretty_midi.utilities.program_to_instrument_class(i)) for i in not_rendered_midi_programs])
                print(f"Not rendered midi programs: {not_rendered_midi_programs}")
                print(f"Num midi programs: {num_midi_programs}")
                print(yaml_path)
                breakpoint()
                # raise RuntimeError("Ambigious not rendered midi program")

        return list(set(not_rendered_midi_programs.keys()))
