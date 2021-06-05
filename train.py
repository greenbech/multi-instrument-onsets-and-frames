import os
import sys
from datetime import datetime

import numpy as np
import torch
import tqdm
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from slakh_dataset import SlakhAmtDataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate, print_metrics
from onsets_and_frames.constants import MAX_MIDI, MIN_MIDI, N_MELS
from onsets_and_frames.transcriber import OnsetsAndFrames
from onsets_and_frames.utils import cycle, get_grad_norm, summary

ex = Experiment("train_transcriber")

# flake8: noqa: F841
@ex.config
def config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations = 20000
    resume_iteration = None
    checkpoint_interval = 1000
    dataset = "Slakh"
    path = "data/slakh2100_flac_16k"
    split = "redux"
    audio = "individual"
    instrument = "electric-bass"
    midi_programs = None
    max_harmony = None
    skip_pitch_bend_tracks = True
    experiment = None

    experiment_name = experiment + "-" if experiment else ""
    logdir = f"runs/{experiment_name}{instrument}-{audio.replace(os.sep, '-')}-transcriber-" + datetime.now().strftime(
        "%y%m%d-%H%M%S"
    )

    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f"Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory")

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    auto_clip_gradient = True

    validation_length = 4 * sequence_length
    validation_interval = 1000
    num_validation_files = 40
    create_validation_images = True

    predict_velocity = False
    feed_velocity_to_onset = False
    add_unet_model = False
    min_midi = MIN_MIDI
    max_midi = MAX_MIDI
    n_mels = N_MELS

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(
    logdir,
    device,
    iterations,
    resume_iteration,
    checkpoint_interval,
    dataset,
    path,
    split,
    audio,
    instrument,
    midi_programs,
    max_harmony,
    skip_pitch_bend_tracks,
    batch_size,
    sequence_length,
    model_complexity,
    learning_rate,
    learning_rate_decay_steps,
    learning_rate_decay_rate,
    leave_one_out,
    auto_clip_gradient,
    validation_length,
    validation_interval,
    num_validation_files,
    create_validation_images,
    predict_velocity,
    feed_velocity_to_onset,
    add_unet_model,
    min_midi,
    max_midi,
    n_mels,
):

    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ["train"], ["validation"]

    if midi_programs is not None:
        instrument = None
    if dataset == "Slakh":
        dataset = SlakhAmtDataset(
            path=path,
            split=split,
            audio=audio,
            label_instruments=instrument,
            label_midi_programs=midi_programs,
            groups=train_groups,
            sequence_length=sequence_length,
            # max_files_in_memory=None,
            skip_pitch_bend_tracks=skip_pitch_bend_tracks,
            min_midi=min_midi,
            max_midi=max_midi,
            skip_missing_tracks=True,
            max_harmony=max_harmony,
        )
        validation_dataset = SlakhAmtDataset(
            path=path,
            split=split,
            audio=audio,
            label_instruments=instrument,
            label_midi_programs=midi_programs,
            groups=validation_groups,
            sequence_length=validation_length,
            num_files=num_validation_files,
            reproducable_load_sequences=True,
            skip_pitch_bend_tracks=skip_pitch_bend_tracks,
            min_midi=min_midi,
            max_midi=max_midi,
            skip_missing_tracks=True,
            max_harmony=max_harmony,
        )

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(
            n_mels,
            max_midi - min_midi + 1,
            model_complexity=model_complexity,
            predict_velocity=predict_velocity,
            feed_velocity_to_onset=feed_velocity_to_onset,
            add_unet_model=add_unet_model,
            min_midi=min_midi,
            max_midi=max_midi,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f"model-{resume_iteration}.pt")
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, "last-optimizer-state.pt")))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    grad_history = []

    loop = tqdm.trange(resume_iteration + 1, iterations + 1)
    for i, batch in zip(loop, cycle(loader)):
        _, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()

        obs_grad_norm = get_grad_norm(model=model)
        if auto_clip_gradient:
            grad_history.append(obs_grad_norm)
            clip_value = np.percentile(grad_history, 10)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        scheduler.step()

        for key, value in {"loss": loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)
        writer.add_scalar("loss/obs_grad_norm", obs_grad_norm, global_step=i)
        writer.add_scalar("loss/clipped_grad_norm", clip_value, global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                if create_validation_images:
                    validation_folder = os.path.join(logdir, f"model-{i}")
                else:
                    validation_folder = None
                metrics = evaluate(validation_dataset, model, save_folder=validation_folder, save_midi=False)
                print_metrics(metrics, add_loss=True, file=sys.stderr)
                print()
                for key, value in metrics.items():
                    writer.add_scalar("validation/" + key.replace(" ", "_"), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f"model-{i}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(logdir, "last-optimizer-state.pt"))
