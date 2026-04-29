# Movie Recommendation System with Matrix Factorization

This project implements a MovieLens recommender system using matrix factorization
with PyTorch. The code is organized as an importable package under `src/`.

## Project Structure

```bash
.
├── configs/
│   └── default.yaml
├── notebooks/
├── src/
│   └── recommender/
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── recommend.py
│       └── train.py
├── tests/
├── README.md
├── pyproject.toml
└── requirements.txt
```

The root-level `data.py`, `model.py`, `train.py`, `similar_movies.py`,
`get_movie_ids.py`, `paths.py`, and `utils.py` files are compatibility wrappers
for the package modules.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download and extract the MovieLens 25M dataset into `ml-25m/`.

## Usage

Train the model:

```bash
python -m recommender.train
```

Find similar movies:

```bash
python -m recommender.recommend --movie-id 2959 --top-n 5
```

The legacy wrappers still work:

```bash
python train.py
python similar_movies.py --movie-id 2959
python get_movie_ids.py --rows 10
```

Default paths and training parameters are documented in `configs/default.yaml`.
CLI arguments can override the dataset, movie metadata, and model paths.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Notes

The model predicts explicit ratings from the global rating mean, user bias,
movie bias, and the dot product of user/movie embeddings. It uses Mean Absolute
Error (L1 Loss), with an embedding size of 200 by default. Training the full
MovieLens 25M dataset can take significant time depending on hardware.
