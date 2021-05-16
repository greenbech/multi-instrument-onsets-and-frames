import numpy as np
import torch
from slakh_dataset.data_classes import MusicAnnotation


def extract_notes(onsets, frames, velocity=None, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: Optional[torch.FloatTensor, shape = [frames, bins]]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    # Increase onset value to position with higer frames values after
    frames_diff_avg = torch.cat([(frames[1:, :] + frames[:-1, :]) / 2, frames[-1:, :]], dim=0)
    onsets_copy = onsets.clone().detach()
    onsets_copy_modified = onsets_copy + (1 / frame_threshold * frames_diff_avg) * onsets_copy

    onsets_copy_modified[1:, :] -= torch.clip(frames[:-1, :] - frame_threshold, 0)

    onsets = (onsets_copy_modified > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []
    default_velocity = 0.9

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if velocity is not None:
                if onsets[offset, pitch].item():
                    velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset + 1:
            pitches.append(pitch)
            intervals.append([onset, offset])
            if velocity is not None:
                velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)
            else:
                velocities.append(default_velocity)

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs


def notes_music_annotation(pitches, intervals, shape) -> MusicAnnotation:
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]
    """
    onset_torch = torch.zeros(tuple(shape))
    frame_torch = torch.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        frame_torch[onset:offset, pitch] = 1
        onset_torch[onset : onset + 1, pitch] = 1

    return MusicAnnotation(
        onset=onset_torch,
        offset=None,
        frame=frame_torch,
        velocity=None,
    )
