import sys
from functools import reduce

import numpy as np
import torch
from PIL import Image
from torch.nn.modules.module import _addindent

from onsets_and_frames.data_classes import MusicAnnotation


# https://github.com/pseeth/autoclip/blob/master/autoclip.py
def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, "shape"):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        if file is sys.stdout:
            main_str += ", \033[92m{:,}\033[0m params".format(total_params)
        else:
            main_str += ", {:,} params".format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, "w")
        print(string, file=file)
        file.flush()

    return count


def save_pianoroll(path, mel, model, prediction: MusicAnnotation, prediction_notes: MusicAnnotation):
    """
    Saves a piano roll diagram
    """
    mel_image = create_mel_np_image(mel, model)
    prediction_weights_np = music_annotation_to_numpy_image(prediction)
    line = 100 * np.ones((1, prediction_weights_np.shape[1], prediction_weights_np.shape[2]), dtype=np.uint8)

    pred_frames = (255 // 2 * prediction_notes.frame.t()).to(torch.uint8).cpu()
    pred_onset = (255 // 2 * prediction_notes.onset.t()).to(torch.uint8).cpu()

    pred_image = (
        torch.stack([pred_frames + pred_onset, pred_frames + pred_onset, pred_frames + pred_onset], dim=2)
        .flip(0)
        .numpy()
    )

    image_data = np.concatenate((mel_image, line, prediction_weights_np, line, pred_image), axis=0)
    image = Image.fromarray(image_data, "RGB")
    image.save(path)


def music_annotation_to_numpy_image(music_annotation: MusicAnnotation):
    onsets = (255 * music_annotation.onset.t()).to(torch.uint8).cpu()
    frames = (255 * music_annotation.frame.t()).to(torch.uint8).cpu()
    offset = (255 * music_annotation.offset.t()).to(torch.uint8).cpu()
    image = torch.stack([onsets, frames, offset], dim=2).flip(0).numpy()
    return image


def create_mel_np_image(mel, model):
    mel = mel.squeeze()
    mel = mel - model.min_mel_value
    mel = 255 / (model.max_mel_value - model.min_mel_value) * mel
    mel_data = (mel.t()).to(torch.uint8).cpu()
    mel_image = torch.stack([mel_data, mel_data, mel_data], dim=2).flip(0).numpy()
    return mel_image


def save_pred_and_label_piano_roll(
    path,
    mel,
    model,
    reference: MusicAnnotation,
    prediction: MusicAnnotation,
    prediction_notes: MusicAnnotation,
    onset_threshold=0.5,
    offsets_threshold=0.5,
    frame_threshold=0.5,
):
    assert reference.onset.shape == prediction.onset.shape
    mel_image = create_mel_np_image(mel, model)

    prediction_weights_np = music_annotation_to_numpy_image(prediction)
    reference_np = music_annotation_to_numpy_image(reference)
    line = 100 * np.ones((1, prediction_weights_np.shape[1], prediction_weights_np.shape[2]), dtype=np.uint8)

    pred_frames = (255 // 2 * prediction_notes.frame.t()).to(torch.uint8).cpu()
    pred_onset = (255 // 2 * prediction_notes.onset.t()).to(torch.uint8).cpu()
    ref_frames = (255 // 2 * reference.frame.t()).to(torch.uint8).cpu()
    ref_onset = (255 // 2 * reference.onset.t()).to(torch.uint8).cpu()
    pred_image = (
        torch.stack([pred_frames + pred_onset, ref_frames + ref_onset, ref_frames + ref_onset], dim=2).flip(0).numpy()
    )

    image_data = np.concatenate((mel_image, line, prediction_weights_np, line, reference_np, line, pred_image), axis=0)
    image = Image.fromarray(image_data, "RGB")
    image.save(path)
