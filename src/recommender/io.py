import torch


def save_model(model: torch.nn.Module, filepath: str):
    torch.save(model.state_dict(), filepath)
    return f"Model saved to: {filepath}"


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer,
    epoch: int,
    validation_metrics: dict,
    user_map: dict,
    movie_map: dict,
    config: dict,
    training_history=None,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "user_map": user_map,
        "movie_map": movie_map,
        "config": config,
        "training_history": training_history or [],
    }
    torch.save(checkpoint, filepath)
    return f"Checkpoint saved to: {filepath}"


def load_model(model, filepath, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    return f"Model loaded from: {filepath}"


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain resumable training state")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
