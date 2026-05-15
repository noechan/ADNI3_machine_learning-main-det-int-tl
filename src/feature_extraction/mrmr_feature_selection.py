import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def mrmr_feature_selection_deterministic(X, y, k=15, random_state=0):
    """
    Minimum redundancy maximum relevance (mRMR) feature selection (greedy).
    """

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    if n_features <= k:
        return list(range(n_features))

    # Relevance: MI(feature, y)
    relevance = mutual_info_classif(X, y, random_state=random_state)

    selected = []
    not_selected = list(range(n_features))

    # Track cumulative redundancy sum for each feature (incremental update)
    redundancy_sum = np.zeros(n_features)

    first = int(np.argmax(relevance))
    selected.append(first)
    not_selected.remove(first)

    for _ in range(k - 1):
        if not not_selected:
            break

        not_selected_arr = np.array(not_selected)
        last_selected = selected[-1]

        # Batch MI: compute MI between ALL remaining candidates and the last
        # selected feature in a single call (instead of one call per candidate)
        mi_with_last = mutual_info_regression(
            X[:, not_selected_arr], X[:, last_selected].ravel(),
            random_state=random_state
        )

        # Incrementally update redundancy sums (avoid recomputing from scratch)
        redundancy_sum[not_selected_arr] += mi_with_last

        # Vectorised mRMR score computation
        num_selected = len(selected)
        avg_redundancy = redundancy_sum[not_selected_arr] / num_selected
        candidate_relevance = relevance[not_selected_arr]
        scores = candidate_relevance.copy()
        mask = avg_redundancy >= 1e-10
        scores[mask] = candidate_relevance[mask] / avg_redundancy[mask]

        # Minimal deterministic selection (np.argmax is deterministic given scores)
        next_feature = int(not_selected_arr[int(np.argmax(scores))])

        selected.append(next_feature)
        not_selected.remove(next_feature)

    return selected
