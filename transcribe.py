import argparse
import os

import numpy as np
import torch
import torchaudio
from mir_eval.util import midi_to_hz
from slakh_dataset.data_classes import MusicAnnotation
from tqdm import tqdm

from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, SAMPLE_RATE
from onsets_and_frames.decoding import extract_notes
from onsets_and_frames.midi import save_midi
from onsets_and_frames.utils import save_pianoroll, summary


def load_and_process_audio(audio_path, sequence_length, device):

    random = np.random.RandomState(seed=42)

    audio_info = torchaudio.info(audio_path)
    sr = audio_info.sample_rate

    if sequence_length is not None:
        audio_length = audio_info.num_frames
        step_begin = random.randint(audio_length - sequence_length) // HOP_LENGTH

        begin = step_begin * HOP_LENGTH
        end = begin + sequence_length
        num_frames = end - begin

        audio = torchaudio.load(audio_path, frame_offset=begin, num_frames=num_frames)[0].to(device)
    else:
        audio = torchaudio.load(audio_path)[0].to(device)

    if audio_info.num_channels == 2:
        audio = torch.mean(audio, 0)
        audio = torch.unsqueeze(audio, 0)

    if sr != SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(audio)

    assert len(audio.shape) == 2 and audio.shape[0] == 1
    return audio


def transcribe(model, audio) -> MusicAnnotation:

    mel = model.melspectrogram(audio).transpose(-1, -2)
    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

    return MusicAnnotation(
        onset=onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
        offset=offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
        frame=frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
        velocity=velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
        if velocity_pred is not None
        else None,
    )


def transcribe_file(
    model_file,
    audio_paths,
    save_folder,
    midi_program,
    is_drum,
    sequence_length,
    onset_threshold,
    frame_threshold,
    device,
):

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    tqdm_range = tqdm(audio_paths)
    for audio_path in tqdm_range:
        tqdm_range.set_description(f"Processing {audio_path}")

        audio = load_and_process_audio(audio_path, sequence_length, device)
        predictions = transcribe(model, audio)

        p_est, i_est, v_est = extract_notes(
            predictions.onset, predictions.frame, predictions.velocity, onset_threshold, frame_threshold
        )

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        save_name = audio_path.replace(os.sep, "-")
        os.makedirs(save_folder, exist_ok=True)
        pred_path = os.path.join(save_folder, save_name + ".pred.png")
        save_pianoroll(pred_path, predictions)
        midi_path = os.path.join(save_folder, save_name + ".pred.mid")
        save_midi(
            midi_path,
            p_est,
            i_est,
            v_est,
            midi_program=midi_program,
            is_drum=is_drum,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("audio_paths", type=str, nargs="+")
    parser.add_argument("--save-folder", type=str, default="tmp")
    parser.add_argument("--midi-program", default=0, type=int)
    parser.add_argument("--is_drum", action="store_true")
    parser.add_argument("--sequence-length", default=None, type=int)
    parser.add_argument("--onset-threshold", default=0.5, type=float)
    parser.add_argument("--frame-threshold", default=0.5, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))
