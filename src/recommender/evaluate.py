import torch


def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for user_id, movies_id, ratings in dataloader:
            user_id = user_id.squeeze().to(device)
            movies_id = movies_id.squeeze().to(device)
            ratings = ratings.squeeze().to(device)

            val_preds = model(user_id, movies_id)
            loss = loss_fn(val_preds, ratings)
            val_loss += loss.item()

    return val_loss / len(dataloader)

