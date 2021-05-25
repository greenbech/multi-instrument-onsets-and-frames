from train import ex


def main():
    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    ex.run(
        config_updates={
            "split": "redux",
            "audio": "mix.flac",
            # "instrument": ["piano"],
            "instrument": ["piano", "guitar", "bass", "all-pitched"],
            # "midi_programs": range(96),
            "max_harmony": None,
            "skip_pitch_bend_tracks": True,
            "batch_size": batch_size // 2,
            "sequence_length": sequence_length // 2,
            "model_complexity": model_complexity // 2,
            "validation_length": 4 * sequence_length,
            "validation_interval": 500,
            "num_validation_files": 50,
            "create_validation_images": True,
            "predict_velocity": True,
            "feed_velocity_to_onset": True,
            "add_unet_model": False,
            "n_mels": 256,
            "min_midi": 28,  # E1
            "max_midi": 96,  # C7
            "iterations": 50000,
        }
    )


main()
