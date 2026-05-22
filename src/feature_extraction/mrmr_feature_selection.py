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

def constrained_mrmr_feature_selection(
    X,
    y,
    feature_columns,
    k,
    always_include=None,
    exclude=None,
    random_state=0,
):
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    always_include = always_include or []
    exclude = exclude or []

    feature_to_idx = {f: i for i, f in enumerate(feature_columns)}

    missing_always = [f for f in always_include if f not in feature_to_idx]
    missing_exclude = [f for f in exclude if f not in feature_to_idx]

    if missing_always:
        raise ValueError(f"Features in ALWAYS_INCLUDE_FEATURES not found: {missing_always}")
    if missing_exclude:
        raise ValueError(f"Features in EXCLUDED_FEATURES not found: {missing_exclude}")

    excluded_idx = {feature_to_idx[f] for f in exclude}
    fixed_idx = [
        feature_to_idx[f]
        for f in always_include
        if feature_to_idx[f] not in excluded_idx
    ]

    if k < len(fixed_idx):
        raise ValueError(
            f"NF={k} is smaller than the number of always-included features "
            f"({len(fixed_idx)})."
        )

    eligible_idx = [
        i for i, f in enumerate(feature_columns)
        if i not in excluded_idx and i not in fixed_idx
    ]

    n_to_select_with_mrmr = k - len(fixed_idx)

    if n_to_select_with_mrmr <= 0:
        return fixed_idx

    if n_to_select_with_mrmr >= len(eligible_idx):
        return fixed_idx + eligible_idx

    X_eligible = X[:, eligible_idx]

    selected_rel_idx = mrmr_feature_selection_deterministic(
        X_eligible,
        y,
        k=n_to_select_with_mrmr,
        random_state=random_state,
    )

    selected_abs_idx = [eligible_idx[i] for i in selected_rel_idx]

    return fixed_idx + selected_abs_idx
