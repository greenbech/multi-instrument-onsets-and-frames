import multiprocessing
import sys
from typing import List

import numpy as np
import pretty_midi
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
from mir_eval.util import hz_to_midi
from pretty_midi import PrettyMIDI
from tqdm import tqdm

from .constants import MAX_MIDI, MIN_MIDI


def instrument_class_to_number(instrument_class: str) -> int:
    instrument_classes = [i.lower().replace(" ", "-") for i in pretty_midi.constants.INSTRUMENT_CLASSES]
    instrument_classes.append("drum")
    instrument_classes.append("other")
    return instrument_classes.index(instrument_class)


def parse_midi(path, instruments: List[str], remove_midi_programs: List[int]):
    """open midi file and return np.array of (instrument, onset, offset, note, velocity) rows"""
    mid = PrettyMIDI(path)

    instruments = set(instruments)
    add_drum = "drum" in instruments
    if add_drum:
        instruments.remove("drum")
    add_other = "other" in instruments
    if add_other:
        instruments.remove("other")

    instrument_classes = [i.lower().replace(" ", "-") for i in pretty_midi.constants.INSTRUMENT_CLASSES]
    for instrument in instruments:
        if instrument not in instrument_classes:
            raise RuntimeError(f"Unsupported instrument class {instrument}. Avaliable classes are {instrument_classes}")

    data = []
    notes_out_of_range = set()
    for instrument in mid.instruments:
        if instrument.program in remove_midi_programs:
            print(f"Removing midi program {instrument.program}")
            continue
        if instrument.is_drum:
            if not add_drum:
                continue
            instrument_class = "drum"
        else:
            instrument_class = (
                pretty_midi.utilities.program_to_instrument_class(instrument.program).lower().replace(" ", "-")
            )
            if instrument_class not in instruments:
                sound_effect_instument_class = pretty_midi.constants.INSTRUMENT_CLASSES[-1]
                if (not add_other) or instrument_class == sound_effect_instument_class:
                    continue
                instrument_class = "other"

        for note in instrument.notes:
            if int(note.pitch) in range(MIN_MIDI, MAX_MIDI + 1):
                data.append(
                    (
                        instrument_class_to_number(instrument_class),
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


def save_midi(path, pitches, intervals, velocities):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type="on", pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type="off", pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row["time"])

    last_tick = 0
    for event in events:
        current_tick = int(event["time"] * ticks_per_second)
        velocity = int(event["velocity"] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event["pitch"])))
        track.append(Message("note_" + event["type"], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)


if __name__ == "__main__":

    def process(input_file, output_file):
        midi_data = parse_midi(input_file)
        np.savetxt(output_file, midi_data, "%.6f", "\t", header="onset\toffset\tnote\tvelocity")

    def files():
        for input_file in tqdm(sys.argv[1:]):
            if input_file.endswith(".mid"):
                output_file = input_file[:-4] + ".tsv"
            elif input_file.endswith(".midi"):
                output_file = input_file[:-5] + ".tsv"
            else:
                print("ignoring non-MIDI file %s" % input_file, file=sys.stderr)
                continue

            yield (input_file, output_file)

    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process)(in_file, out_file) for in_file, out_file in files())
