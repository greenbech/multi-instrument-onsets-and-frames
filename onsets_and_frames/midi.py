import collections
from typing import List

import pretty_midi
from mir_eval.util import hz_to_midi
from pretty_midi import PrettyMIDI

from .constants import MAX_MIDI, MIN_MIDI


def instrument_class_to_number(instrument_class: str) -> int:
    instrument_classes = [i.lower().replace(" ", "-") for i in pretty_midi.constants.INSTRUMENT_CLASSES]
    instrument_classes.append("drum")
    instrument_classes.append("other")
    return instrument_classes.index(instrument_class)


def parse_midi(path, extract_instruments: List[int], extract_drums=False, remove_midi_programs: List[int] = None):
    """open midi file and return np.array of (instrument, onset, offset, note, velocity) rows"""
    mid = PrettyMIDI(path)

    data = []
    notes_out_of_range = set()
    for instrument in mid.instruments:
        if instrument.is_drum:
            if not extract_drums:
                continue
            instrument.program = -1
        else:
            if instrument.program not in extract_instruments:
                continue

        if instrument.program in remove_midi_programs:
            print(f"Removing midi program {instrument.program}")
            continue

        for note in instrument.notes:
            if int(note.pitch) in range(MIN_MIDI, MAX_MIDI + 1):
                data.append(
                    (
                        instrument.program,
                        note.start,
                        note.end,
                        int(note.pitch),
                        int(note.velocity),
                    )
                )
            else:
                notes_out_of_range.add(int(note.pitch))
    if len(notes_out_of_range) > 0:
        print(
            f"{len(notes_out_of_range)} notes out of MIDI range ({MIN_MIDI},{MAX_MIDI}) for file {path}. Excluded pitches: {notes_out_of_range}"
        )
    data.sort(key=lambda x: x[1])
    return data


def save_midi(path, pitches_dict):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = pretty_midi.PrettyMIDI()
    for midi_program in pitches_dict:
        if midi_program == -1:
            instrument = pretty_midi.Instrument(program=0, is_drum=True)
        else:
            instrument = pretty_midi.Instrument(program=midi_program, is_drum=False)
        pitches = pitches_dict[midi_program]["pitches"]
        intervals = pitches_dict[midi_program]["intervals"]
        velocities = pitches_dict[midi_program]["velocities"]
        add_notes_to_pretty_midi_instrument(
            instrument=instrument, pitches=pitches, intervals=intervals, velocities=velocities
        )
        file.instruments.append(instrument)
    file.write(path)


def add_notes_to_pretty_midi_instrument(instrument, pitches, intervals, velocities):
    # Remove overlapping intervals (end time should be smaller of equal start time of next note on the same pitch)
    intervals_dict = collections.defaultdict(list)
    for i in range(len(pitches)):
        pitch = int(round(hz_to_midi(pitches[i])))
        intervals_dict[pitch].append((intervals[i], i))
    for pitch in intervals_dict:
        interval_list = intervals_dict[pitch]
        interval_list.sort(key=lambda x: x[0][0])
        for i in range(len(interval_list) - 1):
            # assert interval_list[i][1] <= interval_list[i+1][0], f'End time should be smaller of equal start time of next note on the same pitch. It was {interval_list[i][1]}, {interval_list[i+1][0]} for pitch {key}'
            interval_list[i][0][1] = min(interval_list[i][0][1], interval_list[i + 1][0][0])

    for pitch in intervals_dict:
        interval_list = intervals_dict[pitch]
        for interval, i in interval_list:
            pitch = int(round(hz_to_midi(pitches[i])))
            velocity = int(90 + (127 - 90) * min(velocities[i], 1))
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=interval[0], end=interval[1])
            instrument.notes.append(note)
