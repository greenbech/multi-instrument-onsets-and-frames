import os
from datetime import datetime

import numpy as np
import torch
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from slakh_dataset import SlakhAmtDataset
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate, print_metrics
from onsets_and_frames.constants import MAX_MIDI, MIN_MIDI, N_MELS
from onsets_and_frames.dataset import Slakh
from onsets_and_frames.transcriber import OnsetsAndFrames
from onsets_and_frames.utils import cycle, summary

ex = Experiment("train_transcriber")

# flake8: noqa: F841
@ex.config
def config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000
    dataset = "Slakh"
    path = "data/slakh2100_flac_16k"
    split = "redux"
    audio = "individual"
    instrument = "electric-bass"
    skip_pitch_bend_tracks = True
    logdir = f"runs/{instrument}-{audio}-transcriber-" + datetime.now().strftime("%y%m%d-%H%M%S")

    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        # batch_size = 8
        sequence_length //= 2
        model_complexity //= 2
        print(f"Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory")

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = 4 * sequence_length
    validation_interval = 1000
    num_validation_files = 20
    create_validation_images = True

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
    skip_pitch_bend_tracks,
    batch_size,
    sequence_length,
    model_complexity,
    learning_rate,
    learning_rate_decay_steps,
    learning_rate_decay_rate,
    leave_one_out,
    clip_gradient_norm,
    validation_length,
    validation_interval,
    num_validation_files,
    create_validation_images,
):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ["train"], ["validation"]

    if dataset == "Slakh":
        dataset = SlakhAmtDataset(
            path=path,
            split=split,
            audio=audio,
            instrument=instrument,
            groups=train_groups,
            sequence_length=sequence_length,
            max_files_in_memory=200,
            skip_pitch_bend_tracks=skip_pitch_bend_tracks,
            min_midi=MIN_MIDI,
            max_midi=MAX_MIDI,
        )
        validation_dataset = SlakhAmtDataset(
            path=path,
            split=split,
            audio=audio,
            instrument=instrument,
            groups=validation_groups,
            sequence_length=validation_length,
            num_files=num_validation_files,
            reproducable_load_sequences=True,
            skip_pitch_bend_tracks=skip_pitch_bend_tracks,
            min_midi=MIN_MIDI,
            max_midi=MAX_MIDI,
        )

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f"model-{resume_iteration}.pt")
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, "last-optimizer-state.pt")))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1), initial=resume_iteration + 1)
    for i, batch in zip(loop, cycle(loader)):
        _, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {"loss": loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                if create_validation_images:
                    validation_path = os.path.join(logdir, f"model-{i}")
                else:
                    validation_path = None
                metrics = evaluate(validation_dataset, model, save_path=validation_path, is_validation=True)
                print_metrics(metrics, add_loss=True)
                print()
                for key, value in metrics.items():
                    writer.add_scalar("validation/" + key.replace(" ", "_"), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f"model-{i}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(logdir, "last-optimizer-state.pt"))
