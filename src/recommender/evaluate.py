import torch


def _ndcg_at_k(recommended_movies, relevant_movies, k):
    dcg = 0.0
    for rank, movie_id in enumerate(recommended_movies[:k], start=1):
        if movie_id in relevant_movies:
            dcg += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()

    ideal_hits = min(len(relevant_movies), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / torch.log2(torch.tensor(rank + 1.0)).item()
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / idcg


def _rank_movies_for_user(model, user_id, candidate_movie_ids, k, device):
    movie_ids = torch.tensor(candidate_movie_ids, dtype=torch.long, device=device)
    user_ids = torch.full(
        (len(candidate_movie_ids),), user_id, dtype=torch.long, device=device
    )
    scores = model(user_ids, movie_ids)
    top_k = min(k, len(candidate_movie_ids))
    top_positions = torch.topk(scores, k=top_k).indices.cpu().tolist()
    return [candidate_movie_ids[position] for position in top_positions]


def evaluate_model(
    model, dataloader, device, k=10, relevance_threshold=4.0, implicit_feedback=False
):
    if k < 1:
        raise ValueError("k must be at least 1")

    model.eval()
    predictions = []
    targets = []
    candidates_by_user = {}
    relevant_by_user = {}

    with torch.no_grad():
        for user_id, movies_id, ratings in dataloader:
            user_id = user_id.view(-1).to(device)
            movies_id = movies_id.view(-1).to(device)
            ratings = ratings.view(-1).to(device)

            val_preds = model(user_id, movies_id)
            if implicit_feedback:
                predictions.append(torch.sigmoid(val_preds).detach().cpu())
            else:
                predictions.append(val_preds.detach().cpu())
            targets.append(ratings.detach().cpu())

            for batch_user_id, batch_movie_id, batch_rating in zip(
                user_id.detach().cpu().tolist(),
                movies_id.detach().cpu().tolist(),
                ratings.detach().cpu().tolist(),
            ):
                candidates_by_user.setdefault(batch_user_id, set()).add(batch_movie_id)
                if batch_rating >= relevance_threshold:
                    relevant_by_user.setdefault(batch_user_id, set()).add(
                        batch_movie_id
                    )

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        errors = predictions - targets

        mae = torch.mean(torch.abs(errors)).item()
        rmse = torch.sqrt(torch.mean(errors**2)).item()

        num_movies = model.movie_embedding.num_embeddings
        recommended_movies = set()
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        hit_scores = []

        for user_id, relevant_movies in relevant_by_user.items():
            candidate_movie_ids = sorted(candidates_by_user[user_id])
            top_movies = _rank_movies_for_user(
                model, user_id, candidate_movie_ids, k, device
            )
            recommended_movies.update(top_movies)
            hits = len(set(top_movies) & relevant_movies)

            precision_scores.append(hits / min(k, len(candidate_movie_ids)))
            recall_scores.append(hits / len(relevant_movies))
            ndcg_scores.append(_ndcg_at_k(top_movies, relevant_movies, k))
            hit_scores.append(1.0 if hits > 0 else 0.0)

    return {
        "mae": mae,
        "rmse": rmse,
        f"precision@{k}": (
            sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        ),
        f"recall@{k}": (
            sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        ),
        f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        f"hitrate@{k}": sum(hit_scores) / len(hit_scores) if hit_scores else 0.0,
        "coverage": len(recommended_movies) / num_movies if num_movies else 0.0,
    }
