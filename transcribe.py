import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from mir_eval.util import midi_to_hz
from slakh_dataset.data_classes import MusicAnnotation
from slakh_dataset.midi import instrument_to_canonical_midi_program
from tqdm import tqdm

from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, SAMPLE_RATE
from onsets_and_frames.decoding import extract_notes, notes_music_annotation
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


def transcribe_file(
    model_file,
    audio_paths,
    save_folder,
    midi_programs,
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
        mel = model.mel(audio)
        onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)
        pitches_dict = {}
        min_midi = 28
        max_midi = 96
        midi_range = max_midi - min_midi + 1
        pred_multi = MusicAnnotation(
            onset=onset_pred.reshape((onset_pred.shape[0], onset_pred.shape[1], midi_range, -1)),
            offset=offset_pred.reshape((offset_pred.shape[0], offset_pred.shape[1], midi_range, -1)),
            frame=frame_pred.reshape((frame_pred.shape[0], frame_pred.shape[1], midi_range, -1)),
            velocity=velocity_pred.reshape((velocity_pred.shape[0], velocity_pred.shape[1], midi_range, -1))
            if velocity_pred is not None
            else None,
        )
        for value in pred_multi:
            if value is None:
                continue
            value.squeeze_(0).relu_()

        for i in range(pred_multi.onset.shape[-1]):
            pred = MusicAnnotation(
                onset=pred_multi.onset[:, :, i],
                frame=pred_multi.frame[:, :, i],
                offset=pred_multi.offset[:, :, i],
                velocity=pred_multi.velocity[:, :, i],
            )
            p_est, i_est, v_est = extract_notes(pred.onset, pred.frame, pred.velocity, onset_threshold, frame_threshold)

            pred_notes = notes_music_annotation(p_est, i_est, pred.frame.shape)
            scaling = HOP_LENGTH / SAMPLE_RATE

            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

            midi_program = midi_programs[i]
            pitches_dict[midi_program] = {
                "pitches": p_est,
                "intervals": i_est,
                "velocities": v_est,
            }

            save_name = f"{Path(audio_path).stem}-{midi_program}-f_thld_{frame_threshold}-o_thld_{onset_threshold}"
            os.makedirs(save_folder, exist_ok=True)
            pred_path = os.path.join(save_folder, save_name + ".pred.png")
            save_pianoroll(pred_path, mel, model, pred, pred_notes)

        save_name = f"{Path(audio_path).stem}-{'-'.join([str(i) for i in midi_programs])}-f_thld:{frame_threshold}-o_thld:{onset_threshold}"
        midi_path = os.path.join(save_folder, save_name + ".pred.mid")
        save_midi(midi_path, pitches_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("audio_paths", type=str, nargs="+")
    parser.add_argument("--save-folder", type=str, default="tmp")
    parser.add_argument("--instruments", type=str, nargs="+")
    parser.add_argument("--sequence-length", default=None, type=int)
    parser.add_argument("--onset-threshold", default=0.35, type=float)
    parser.add_argument("--frame-threshold", default=0.3, type=float)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    save_folder = os.path.join(args.save_folder, args.model_file.replace(os.sep, "-"))
    midi_programs = [instrument_to_canonical_midi_program(inst) for inst in args.instruments]

    with torch.no_grad():
        transcribe_file(
            model_file=args.model_file,
            audio_paths=args.audio_paths,
            save_folder=save_folder,
            midi_programs=midi_programs,
            sequence_length=args.sequence_length,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            device=args.device,
        )
