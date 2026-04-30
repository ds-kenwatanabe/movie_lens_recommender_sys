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

Recommend movies for a user:

```bash
python -m recommender.recommend --user-id 123 --top-k 10
```

Similar movie search uses cosine similarity over learned movie embeddings. You
can filter by genre and use Annoy approximate nearest neighbors for faster
search:

```bash
python -m recommender.recommend --movie-id 2959 --genre Drama --use-annoy
```

Compare simple baselines:

```bash
python -m recommender.baselines
```

The legacy wrappers still work:

```bash
python train.py
python similar_movies.py --movie-id 2959
python get_movie_ids.py --rows 10
```

Default paths and training parameters are documented in `configs/default.yaml`.
Training uses a temporal validation split: earlier ratings train the model and
later ratings validate it. For top-N recommendation, high ratings are treated
as positives and sampled non-interacted movies are treated as negatives. CLI
arguments can override the dataset, movie metadata, model paths, and evaluation
settings.

Training saves resumable checkpoints at `--model-path`. A checkpoint contains
model weights, optimizer state, epoch, validation metrics, user/movie ID
mappings, and the training config. Resume with `--resume-from path/to/checkpoint`.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Notes

The model scores user/movie pairs with user bias, movie bias, and the dot
product of user/movie embeddings. Training uses binary cross entropy over
positive interactions and sampled negatives, optimized with Adam and weight
decay regularization. Validation reports MAE, RMSE, Precision@K, Recall@K,
NDCG@K, HitRate@K, and catalog coverage. Training and ranking evaluation on the
full MovieLens 25M dataset can take significant time depending on hardware.

Baseline comparison includes global mean, user mean, movie mean, popularity,
item-item cosine similarity, and truncated SVD.
