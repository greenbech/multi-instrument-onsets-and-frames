from time import monotonic

from tqdm import tqdm

from onsets_and_frames.dataset import Slakh


def run_benchmark(dataset, dataset_name: str, sequence_length, num_runs=30):
    print(f"Running benchmark with dataset {dataset_name} {num_runs} runs")
    start = monotonic()
    for _ in range(num_runs - 1):
        for batch in tqdm(dataset):
            assert batch.audio.shape[0] == sequence_length

    start_last = monotonic()
    for batch in tqdm(dataset):
        assert batch.audio.shape[0] == sequence_length
    end_last = monotonic()
    last_run_time = end_last - start_last
    end = monotonic()

    run_time = end - start
    print(
        f"Loaded {len(dataset)} files with sequence length {sequence_length} in {run_time} seconds ({run_time / num_runs} avg. per run, {last_run_time} last run)"
    )
    print()


def test_slakh_in_memory():
    groups = ["test"]
    sequence_length = 327680
    dataset = Slakh(groups=groups, sequence_length=sequence_length, max_files_in_memory=1000)

    # Run in memory
    run_benchmark(dataset, "Slakh validation in memory", sequence_length)


def test_slakh_streaming():
    groups = ["test"]
    sequence_length = 327680
    # Run fully streaming
    dataset = Slakh(groups=groups, sequence_length=sequence_length, max_files_in_memory=-1)
    run_benchmark(dataset, "Slakh validation streaming", sequence_length)


def test_slakh_partly_streaming():
    groups = ["validation"]
    sequence_length = 327680
    # Run fully streaming
    dataset = Slakh(groups=groups, sequence_length=sequence_length, max_files_in_memory=200)
    run_benchmark(dataset, "Slakh partly streaming", sequence_length)


test_slakh_streaming()

# test_slakh_in_memory()

# test_slakh_partly_streaming()
