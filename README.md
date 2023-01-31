# Multi-instrument Onsets and Frames using Self-Supervised Pre-Training

This repository explores training of a multi-instrument automatic music transcription model using the Self-Supervised Pre-Training and fully Supervised Training using the [Slakh2100 dataset](http://www.slakh.com/). 

## Installation

### Using poetry

This repository uses [poetry](https://python-poetry.org/docs/) for dependencies and virtual environment management.
Look at the [official documentation](https://python-poetry.org/docs/#installation) for installation instructions.

### Install dependencies

```bash
poetry install
```

## Usage

Look at the [`/experiments`](./experiments) folder for how to run different experiments using this codebase.
To run experiment 6, currently the [stream-multi-instrument branch](https://github.com/greenbech/multi-instrument-onsets-and-frames/tree/stream-multi-instrument) must be used.

## IDUN Setup (note to self)

1. Clone the repo to the home dir

2. Use Python 3.8.6 by running:

```bash
> module load Python/3.8.6-GCCcore-10.2.0
```

3. Install Poetry using pip:

```bash
> pip install poetry
```

4. Setup Poetry to create venvs inside project folders:

```bash
> poetry config virtualenvs.in-project true
```

4. Inside the project folder create venv and install all dependencies:

```bash
> cd multi-instrument-onsets-and-frames/
> poetry install
```


## Acknowledgement

This repository is based on [Henrik Grønbech's Multi-Instrument Onsets and Frames model](https://github.com/greenbech/multi-instrument-onsets-and-frames) and [Jong Wook Kim's Onsets and Frames implementation](https://github.com/jongwook/onsets-and-frames).
