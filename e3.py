from train import ex


def main():
    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    ex.run(
        config_updates={
            "path": "data/slakh2100_flac_16k_umx",
            "split": "redux",
            "audio": "stems/bass.flac",
            "instrument": "electric-bass",
            "skip_pitch_bend_tracks": True,
            "batch_size": batch_size,
            "sequence_length": sequence_length // 2,
            "model_complexity": model_complexity // 2,
            "validation_length": 2 * sequence_length,
            "validation_interval": 1000,
            "num_validation_files": 30,
            "create_validation_images": True,
            "predict_velocity": False,
            "min_midi": 47,  # low B,
            "max_midi": 91,  # high G (24th fret G string)
        }
    )


main()
