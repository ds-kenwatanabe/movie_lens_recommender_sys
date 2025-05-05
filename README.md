# 🎬 Movie Recommendation System with Matrix Factorization (PyTorch)
This project implements a movie recommendation system using Matrix Factorization with PyTorch, leveraging the MovieLens 25M dataset. It includes:

A PyTorch-based matrix factorization model

Training and evaluation pipelines

Functionality to find similar movies based on learned embeddings

## 📁 Project Structure
```bash
.
├── data.py                 # Data loading and preprocessing
├── model.py                # MatrixFactorization model definition
├── train.py                # Training loop
├── utils.py                # Utility functions (save/load model)
├── get_movie_ids.py        # Reads the .csv with movies    
├── similar_movies.py       # Script to find similar movies
├── requirements.txt
└── README.md              

```

## 🧠 Model Overview
The model employs Matrix Factorization to learn latent embeddings for users and movies. Each user and movie is represented by a dense vector in a shared embedding space. The predicted rating is computed as the dot product of the corresponding user and movie embeddings.

## 🚀 Getting Started
Prerequisites
Python 3.7+

PyTorch 1.7+

pandas

numpy

tqdm

## Dataset
Download and extract the MovieLens 25M dataset

## 📝 Notes
The model uses Mean Absolute Error (L1 Loss) as the loss function.

The embedding size is set to 200 by default.

Training may take a significant amount of time depending on your hardware.
