import torch


def save_model(model: torch.nn.Module, filepath: str):
    torch.save(model.state_dict(), filepath)
    return f"Model saved to: {filepath}"


def load_model(model, filepath, device="cpu"):
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    return f"Model loaded from: {filepath}"
