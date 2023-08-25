import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, random_split
from data import MovieLens
from model import MatrixFactorization
from utils import save_model

# Set device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# Hyperparameters
BATCH_SIZE = 4096
EMBEDDING_SIZE = 200
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
SEED = 42

# Training on the dataset
movielens = MovieLens()

# Setting up validation and test dataset ratios
val_ratio = 0.2
train_ratio = 1.0 - val_ratio

# Checking lengths
total_length = len(movielens)
train_length = int(train_ratio * total_length)
val_length = int(val_ratio * total_length)

# Split into train and validation datasets
# The generator is used to set a seed so when running multiple times,
# the model does not take different datasets
train_dataset, val_dataset = random_split(movielens,
                                          lengths=[train_length, val_length],
                                          generator=torch.Generator().manual_seed(SEED))

# Creating DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Creating MatrixFactorization model
num_users, num_movies = movielens.size
model = MatrixFactorization(num_users=num_users,
                            num_movies=num_movies,
                            embedding_size=EMBEDDING_SIZE).to(device)

loss_fn = torch.nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loop
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}\n')
    running_loss = 0.0
    min_val_loss = np.inf

    model.train()
    train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (user_id, movies_id, ratings) in train_loop:
        user_id = user_id.squeeze().to(device)
        movies_id = movies_id.squeeze().to(device)
        ratings = ratings.squeeze().to(device)

        # Forward pass
        preds = model(user_id, movies_id)
        # Loss and accuracy
        loss = loss_fn(preds, ratings)
        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Optimization
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        train_loop.set_postfix(loss=running_loss / (i + 1))

    running_loss = running_loss / len(train_dataloader)
    print(f"Train Loss: {running_loss:.2f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for user_id, movies_id, ratings in val_dataloader:
            user_id = user_id.squeeze().to(device)
            movies_id = movies_id.squeeze().to(device)
            ratings = ratings.squeeze().to(device)

            # Forward pass
            val_preds = model(user_id, movies_id)
            # Calculate accumulate the loss
            loss = loss_fn(val_preds, ratings)
            val_loss += loss.item()

    val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {val_loss:.2f}")

    if val_loss < min_val_loss:
        save_model(model, "recommender_model.pth")
        min_val_loss = val_loss

print('Training finished.')
