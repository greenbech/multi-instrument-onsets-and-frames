# Multi-instrument Onsets and Frames

This repository enables mult-instrument automatic music transcription by using the [Slakh dataset](http://www.slakh.com/). 

## Installation

### Using poetry

This repository uses [poetry](https://python-poetry.org/docs/) for dependencies and virtual environment management.
Look at the [official documentation](https://python-poetry.org/docs/#installation) for installation instructions.

### Install dependencies

```
poetry install
```

### Install pre-commit hooks
```
poetry run pre-commit install
```

Look at my other project, [Slakh PyTorch Dataset](https://github.com/greenbech/slakh-pytorch-dataset), for how to download and use the dataset.

## Usage

Look at the [`/experiments`](./experiments) folder for how to run different experiments using this codebase.
To run experiment 6, currently the [stream-multi-instrument branch](https://github.com/greenbech/multi-instrument-onsets-and-frames/tree/stream-multi-instrument) must be used.


## Acknowledgement

This repository is based on [Jong Wook Kim's Onsets and Frames implementation](https://github.com/jongwook/onsets-and-frames).
