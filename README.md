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

Generate diagnostic plots:

```bash
python -m recommender.plots --output-dir plots
```

The legacy wrappers still work:

```bash
python train.py
python similar_movies.py --movie-id 2959
python get_movie_ids.py --rows 10
```

Default paths and training parameters are documented in `configs/default.yaml`.
Training uses a temporal validation split: earlier ratings train the model and
later ratings validate it. For sampled ranking evaluation, high ratings are
treated as positives and sampled non-interacted movies are treated as negatives.
CLI arguments can override the dataset, movie metadata, model paths, and
evaluation settings.

Training supports two modes:

```bash
python -m recommender.train --training-mode explicit --explicit-loss mse
python -m recommender.train --training-mode implicit --implicit-loss bce
python -m recommender.train --training-mode implicit --implicit-loss bpr
```

Training saves resumable checkpoints at `--model-path`. A checkpoint contains
model weights, optimizer state, epoch, validation metrics, user/movie ID
mappings, and the training config. Resume with `--resume-from path/to/checkpoint`.
Checkpoints also include training history for plotting training loss against
validation MAE.

Each training run appends final metrics and config metadata to
`outputs/metrics.json` by default. Use this file to compare runs by embedding
size, loss choice, negatives per positive, and learning rate.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Notes

The model scores user/movie pairs with user bias, movie bias, and the dot
product of user/movie embeddings. Explicit mode predicts ratings with MSE or
MAE. Implicit mode predicts relevance with BCE over sampled negatives or BPR
pairwise ranking loss. Validation reports MAE and RMSE for rating prediction,
plus Precision@K, Recall@K, NDCG@K, HitRate@K, and catalog coverage from sampled
ranking evaluation. Training and evaluation on the full MovieLens 25M dataset
can take significant time depending on hardware.

Baseline comparison includes global mean, user mean, movie mean, popularity,
item-item cosine similarity, and truncated SVD.

Unknown users and movies missing from the trained checkpoint receive cold-start
recommendations from `movies.csv` genres and a popularity-weighted movie mean
baseline, with optional genre filtering.

Diagnostic plotting includes training loss vs validation MAE, rating
distribution, ratings per user, ratings per movie, and movie embedding
visualization with t-SNE or UMAP.
