from train import ex


def main():
    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    ex.run(
        config_updates={
            "split": "redux",
            "audio": "individual",
            "instrument": "electric-bass",
            "skip_pitch_bend_tracks": True,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "model_complexity": model_complexity,
            "validation_length": 4 * sequence_length,
            "validation_interval": 1000,
            "num_validation_files": 50,
            "create_validation_images": True,
            "predict_velocity": False,
            "min_midi": 47,  # low B,
            "max_midi": 91,  # high G (24th fret G string)
        }
    )


main()
