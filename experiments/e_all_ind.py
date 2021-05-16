from train import ex


def main():
    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    piano_midis = list(range(8))
    guitar_midis = list(range(24, 32))
    bass_midis = list(range(32, 40))
    brass_midis = list(range(57, 64))
    reed_midis = list(range(64, 72))

    ex.run(
        config_updates={
            "split": "redux",
            "audio": "individual",
            "instrument": "all-ind",
            "midi_programs": piano_midis + guitar_midis + bass_midis + brass_midis + reed_midis,
            "max_harmony": None,
            "skip_pitch_bend_tracks": True,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "model_complexity": model_complexity,
            "validation_length": 4 * sequence_length,
            "validation_interval": 500,
            "num_validation_files": 50,
            "create_validation_images": True,
            "predict_velocity": False,
            "min_midi": 28,  # E1
            "max_midi": 96,  # C7
        }
    )


main()
