import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from time import gmtime, strftime

import numpy as np
import slakh_dataset.dataset as dataset_module
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import (
    precision_recall_f1_overlap as evaluate_notes_with_velocity,
)
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm

from onsets_and_frames import midi
from onsets_and_frames.constants import HOP_LENGTH, SAMPLE_RATE
from onsets_and_frames.decoding import (
    extract_notes,
    notes_music_annotation,
    notes_to_frames,
)
from onsets_and_frames.transcriber import OnsetsAndFrames
from onsets_and_frames.utils import save_pred_and_label_piano_roll, summary

eps = sys.float_info.epsilon


def evaluate(
    data: dataset_module.PianoRollAudioDataset,
    model: OnsetsAndFrames,
    onset_threshold=0.5,
    frame_threshold=0.5,
    save_folder=None,
    save_midi=False,
):
    metrics = defaultdict(list)

    csvfile = None
    for label in tqdm(data):
        pred, losses = model.run_on_batch(label)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for value in pred:
            if value is None:
                continue
            value.squeeze_(0).relu_()

        p_ref, i_ref, v_ref = extract_notes(
            label.annotation.onset, label.annotation.frame, label.annotation.velocity, add_onset_modification=True
        )
        p_est, i_est, v_est = extract_notes(pred.onset, pred.frame, pred.velocity, onset_threshold, frame_threshold)

        pred_notes = notes_music_annotation(p_est, i_est, pred.frame.shape)

        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label.annotation.frame.shape)
        t_est, f_est = notes_to_frames(p_est, i_est, pred.frame.shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(model.min_midi + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(model.min_midi + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(model.min_midi + midi) for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(model.min_midi + midi) for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics["metric/note/precision"].append(p)
        metrics["metric/note/recall"].append(r)
        metrics["metric/note/f1"].append(f)
        metrics["metric/note/overlap"].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics["metric/note-with-offsets/precision"].append(p)
        metrics["metric/note-with-offsets/recall"].append(r)
        metrics["metric/note-with-offsets/f1"].append(f)
        metrics["metric/note-with-offsets/overlap"].append(o)

        p, r, f, o = evaluate_notes_with_velocity(
            i_ref, p_ref, v_ref, i_est, p_est, v_est, offset_ratio=None, velocity_tolerance=0.1
        )
        metrics["metric/note-with-velocity/precision"].append(p)
        metrics["metric/note-with-velocity/recall"].append(r)
        metrics["metric/note-with-velocity/f1"].append(f)
        metrics["metric/note-with-velocity/overlap"].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
        metrics["metric/note-with-offsets-and-velocity/precision"].append(p)
        metrics["metric/note-with-offsets-and-velocity/recall"].append(r)
        metrics["metric/note-with-offsets-and-velocity/f1"].append(f)
        metrics["metric/note-with-offsets-and-velocity/overlap"].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics["metric/frame/f1"].append(
            hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps]) - eps
        )

        for key, loss in frame_metrics.items():
            metrics["metric/frame/" + key.lower().replace(" ", "_")].append(loss)

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            track = label.track

            if csvfile is None:
                csvfile = open(os.path.join(save_folder, "metrics.csv"), "w")
                fieldnames = ["track"] + list(metrics.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

            csv_row_dict = {"track": track}
            for key in metrics:
                csv_row_dict[key] = metrics[key][-1]
            writer.writerow(csv_row_dict)
            csvfile.flush()

            start_time = strftime("%M:%S", gmtime(label.start_time))
            end_time = strftime("%M:%S", gmtime(label.end_time))
            frame_f1 = metrics["metric/frame/f1"][-1]
            note_f1 = metrics["metric/note/f1"][-1]
            note_w_offset_f1 = metrics["metric/note-with-offsets/f1"][-1]
            file_name = f"{track}-{start_time}-{end_time}-F1s:{frame_f1:.3f}|{note_f1:.3f}|{note_w_offset_f1:.3f}"
            label_path = os.path.join(save_folder, file_name + ".label.png")
            save_pred_and_label_piano_roll(label_path, model, label, pred, pred_notes)
            if save_midi:
                midi_path = os.path.join(save_folder, file_name + ".pred.mid")
                midi.save_midi(midi_path, p_est, i_est, v_est)

    if save_folder is not None:
        with open(os.path.join(save_folder, "evaluation.txt"), "w") as f:
            out_string = metrics_to_string(metrics)
            print(out_string, file=f)
    if csvfile is not None:
        csvfile.close()
    return metrics


def evaluate_file_on_slakh_amt_dataset(
    model_file,
    group,
    split,
    audio,
    instrument,
    midi_programs,
    skip_pitch_bend_tracks,
    max_harmony,
    save_folder,
    onset_threshold,
    frame_threshold,
    device,
    path,
):

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    dataset = dataset_module.SlakhAmtDataset(
        path=path,
        split=split,
        audio=audio,
        instrument=instrument,
        midi_programs=midi_programs,
        groups=[group],
        skip_pitch_bend_tracks=skip_pitch_bend_tracks,
        device=device,
        min_midi=model.min_midi,
        max_midi=model.max_midi,
        max_harmony=max_harmony,
    )
    metrics = evaluate(tqdm(dataset), model, onset_threshold, frame_threshold, save_folder)
    print_metrics(metrics)

    frame_p = 100 * np.mean(metrics["metric/frame/precision"])
    frame_r = 100 * np.mean(metrics["metric/frame/recall"])
    frame_f1 = 100 * np.mean(metrics["metric/frame/f1"])
    note_p = 100 * np.mean(metrics["metric/note/precision"])
    note_r = 100 * np.mean(metrics["metric/note/recall"])
    note_f1 = 100 * np.mean(metrics["metric/note/f1"])
    note_wo_p = 100 * np.mean(metrics["metric/note-with-offsets/precision"])
    note_wo_r = 100 * np.mean(metrics["metric/note-with-offsets/recall"])
    note_wo_f1 = 100 * np.mean(metrics["metric/note-with-offsets/f1"])

    print(
        f"{frame_p:.1f} & {frame_r:.1f} & {frame_f1:.1f}   & {note_p:.1f} & {note_r:.1f} & {note_f1:.1f}   & {note_wo_p:.1f} & {note_wo_r:.1f} & {note_wo_f1:.1f}"
    )


def print_metrics(metrics, add_loss=False, file=sys.stdout):
    out_string = metrics_to_string(metrics=metrics, add_loss=add_loss)
    print(out_string, file=file)


def metrics_to_string(metrics, add_loss=False):
    out_strings = []
    for key, values in metrics.items():
        if add_loss and key.startswith("loss/"):
            category, name = key.split("/")
            out_strings.append(f"{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}")
        if key.startswith("metric/"):
            _, category, name = key.split("/")
            out_strings.append(f"{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    out_string = "\n".join(out_strings)
    return out_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("audio", type=str)
    parser.add_argument("--group", default="test", type=str)
    parser.add_argument("--instrument", type=str, default=None)
    parser.add_argument("--midi-programs", nargs="+", type=int)
    parser.add_argument("--skipbend", action="store_true")
    parser.add_argument("--save-folder", default=None)
    parser.add_argument("--sequence-length", default=None, type=int)
    parser.add_argument("--onset-threshold", default=0.5, type=float)
    parser.add_argument("--frame-threshold", default=0.5, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--path", default="data/slakh2100_flac_16k")
    parser.add_argument("--max-harmony", default=None, type=int)

    args = parser.parse_args()

    if args.save_folder is None:
        instrument_name = args.instrument if args.instrument else "_".join([str(i) for i in args.midi_programs])
        folder_name = f"{os.path.basename(args.model_file)}-{datetime.now().strftime('%y%m%d-%H%M%S')}-{args.group}-{args.split}-{args.audio.replace(os.sep, '_')}-{instrument_name}-{args.skipbend}-maxharm_{args.max_harmony}-o_thld_{args.onset_threshold}_f_thld_{args.frame_threshold}"
        args.save_folder = os.path.join(os.path.dirname(args.model_file), folder_name)
    print(args.save_folder)

    with torch.no_grad():
        evaluate_file_on_slakh_amt_dataset(
            model_file=args.model_file,
            group=args.group,
            split=args.split,
            audio=args.audio,
            instrument=args.instrument,
            midi_programs=args.midi_programs,
            skip_pitch_bend_tracks=args.skipbend,
            max_harmony=args.max_harmony,
            save_folder=args.save_folder,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            device=args.device,
            path=args.path,
        )
