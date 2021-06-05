from train import ex


def main():
    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    ex.run(
        config_updates={
            "experiment": "e1c",
            "iterations": 50000,
            "split": "redux",
            "audio": "mix.flac",
            "instrument": "guitar",
            "skip_pitch_bend_tracks": True,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "model_complexity": model_complexity,
            "validation_length": 4 * sequence_length,
            "validation_interval": 500,
            "num_validation_files": 50,
            "create_validation_images": True,
            "predict_velocity": False,
            "min_midi": 36,
            "max_midi": 100,
        }
    )


main()
