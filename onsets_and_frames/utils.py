import sys
from functools import reduce

import torch
from PIL import Image
from torch.nn.modules.module import _addindent

from onsets_and_frames.data_classes import MusicAnnotation


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


def save_pianoroll(path, music_annotation: MusicAnnotation):
    """
    Saves a piano roll diagram
    """
    onsets = (255 * music_annotation.onset.t()).to(torch.uint8).cpu()
    frames = (255 * music_annotation.frame.t()).to(torch.uint8).cpu()
    offset = (255 * music_annotation.offset.t()).to(torch.uint8).cpu()
    image = torch.stack([onsets, frames, offset], dim=2).flip(0).numpy()
    image = Image.fromarray(image, "RGB")
    image.save(path)


def save_pred_and_label_piano_roll(
    path,
    reference: MusicAnnotation,
    prediction: MusicAnnotation,
    onset_threshold=0.5,
    offsets_threshold=0.5,
    frame_threshold=0.5,
    zoom=4,
):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    assert reference.onset.shape == prediction.onset.shape
    ref_frame = (reference.frame).t().to(torch.uint8).cpu()
    reference_stack = torch.stack([ref_frame, ref_frame, ref_frame], dim=2).cpu()
    reference_image_data = reference_stack.flip(0).mul(int(255 // 4)).numpy()

    pred_onset = (prediction.onset > onset_threshold).t().to(torch.uint8).cpu()
    pred_frame = (prediction.frame > frame_threshold).t().to(torch.uint8).cpu()
    pred_offset = (prediction.offset > onset_threshold).t().to(torch.uint8).cpu()
    # ref_offset = (1 - reference.velocity).t().to(torch.uint8).cpu()
    pred_stack = torch.stack([pred_onset, pred_frame, pred_offset], dim=2).cpu()
    pred_image_data = pred_stack.flip(0).mul(255 // 2).numpy()

    image_data = reference_image_data + pred_image_data

    # image brightness enhancer
    image = Image.fromarray(image_data, "RGB")
    image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)
