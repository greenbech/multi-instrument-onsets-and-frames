from train import ex


def main():
    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    ex.run(
        config_updates={
            # "resume_iteration": 13000,
            # "logdir": "runs/e0b-piano-individual-transcriber-210529-144732",
            "experiment": "e0b",
            "iterations": 50000,
            "split": "redux",
            "audio": "individual",
            "instrument": "piano",
            "skip_pitch_bend_tracks": True,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "model_complexity": model_complexity,
            "validation_length": 4 * sequence_length,
            "validation_interval": 500,
            "num_validation_files": 50,
            "create_validation_images": True,
            "predict_velocity": False,
            "min_midi": 21,
            "max_midi": 108,
        }
    )


main()
