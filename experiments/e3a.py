from train import ex


def main():
    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    ex.run(
        config_updates={
            "experiment": "e3a",
            "iterations": 50000,
            "split": "redux",
            "audio": "mix.flac",
            "instrument": "electric-bass",
            "max_harmony": 2,
            "skip_pitch_bend_tracks": True,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "model_complexity": model_complexity,
            "validation_length": 4 * sequence_length,
            "validation_interval": 500,
            "num_validation_files": 50,
            "create_validation_images": True,
            "predict_velocity": False,
            "add_unet_model": True,
            "n_mels": 256,
            "min_midi": 35,  # B1, https://github.com/ethman/slakh-generation/issues/2
            "max_midi": 67,  # G4 (12th fret G string),
        }
    )


main()
