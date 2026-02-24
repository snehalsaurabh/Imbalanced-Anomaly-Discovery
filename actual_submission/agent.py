import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _build_risk_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c1 = (
        df["institution_risk_score"].to_numpy(float)
        + df["company_risk_score"].to_numpy(float)
        + df["gpa_anomaly_score"].to_numpy(float)
        + 0.06 * df["tenure_gap_months"].to_numpy(float)
    )
    c2 = (
        0.09 * df["applications_7d"].to_numpy(float)
        + 0.03 * df["applications_30d"].to_numpy(float)
        + df["app_to_avg_ratio"].to_numpy(float)
        - 0.03 * df["time_since_last_app_hrs"].to_numpy(float)
    )
    c3 = (
        df["email_risk_score"].to_numpy(float)
        + 1.1 * df["ip_risk_score"].to_numpy(float)
        + 0.7 * df["is_new_device"].to_numpy(float)
        + 0.4 * df["failed_logins_24h"].to_numpy(float)
        + 0.05 * df["login_velocity_24h"].to_numpy(float)
    )
    c4 = (
        1.2 * df["copy_paste_ratio"].to_numpy(float)
        + 0.08 * df["skills_to_exp_ratio"].to_numpy(float)
        - 0.003 * df["profile_age_days"].to_numpy(float)
    )

    c1n, c2n, c3n, c4n = _minmax(c1), _minmax(c2), _minmax(c3), _minmax(c4)
    blended = 0.30 * c1n + 0.25 * c2n + 0.25 * c3n + 0.20 * c4n
    return {"c1": c1n, "c2": c2n, "c3": c3n, "c4": c4n, "blended": blended}


def _seed_indices(X_scaled: np.ndarray, risk: dict[str, np.ndarray], budget: int) -> list[int]:
    n = len(X_scaled)
    target = min(budget, max(28, int(0.35 * budget)))

    ordered: list[int] = []
    used: set[int] = set()

    def add_candidates(candidates: np.ndarray, k: int) -> None:
        for idx in candidates[:k]:
            i = int(idx)
            if i not in used:
                used.add(i)
                ordered.append(i)

    top_per_signal = max(4, target // 7)
    for key in ("c1", "c2", "c3", "c4", "blended"):
        add_candidates(np.argsort(-risk[key]), top_per_signal)

    remain = target - len(ordered)
    if remain > 0:
        n_clusters = int(np.clip(remain, 8, 24))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(X_scaled)
        dists = km.transform(X_scaled)
        for c in range(n_clusters):
            members = np.where(cluster_ids == c)[0]
            if len(members) == 0:
                continue
            local = members[np.argmin(dists[members, c])]
            if int(local) not in used:
                used.add(int(local))
                ordered.append(int(local))
            if len(ordered) >= target:
                break

    if len(ordered) < target:
        for i in np.argsort(-risk["blended"]):
            ii = int(i)
            if ii not in used:
                used.add(ii)
                ordered.append(ii)
            if len(ordered) >= target:
                break

    return ordered[:target]


def _fit_models(X: np.ndarray, y: np.ndarray):
    pos = int(np.sum(y))
    models = []

    lr = LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X, y)
    models.append((lr, 0.40))

    if pos >= 3 and pos < len(y) - 2:
        rf = RandomForestClassifier(
            n_estimators=320,
            max_depth=9,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X, y)
        models.append((rf, 0.35))

        sw = compute_sample_weight(class_weight="balanced", y=y)
        gb = GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=42,
        )
        gb.fit(X, y, sample_weight=sw)
        models.append((gb, 0.25))

    total_w = sum(w for _, w in models)
    return [(m, w / total_w) for m, w in models]


def _ensemble_proba(models, X: np.ndarray) -> np.ndarray:
    p = np.zeros(X.shape[0], dtype=float)
    for model, weight in models:
        p += weight * model.predict_proba(X)[:, 1]
    return p


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.30, 0.70, 33):
        pred = (p >= t).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_s:
            best_s = score
            best_t = float(t)
    return float(np.clip(best_t, 0.35, 0.65))


def run_agent(df: pd.DataFrame, oracle_fn, budget: int) -> np.ndarray:
    n = len(df)
    X = df.to_numpy(dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-8] = 1.0
    X_scaled = (X - mean) / std

    risk = _build_risk_scores(df)
    density = None

    try:
        km_density = KMeans(n_clusters=32, random_state=7, n_init=8)
        d = km_density.fit_transform(X_scaled)
        density = np.exp(-_minmax(np.min(d, axis=1)))
    except Exception:
        density = np.ones(n, dtype=float)

    labeled_idx: list[int] = []
    labeled_y: list[int] = []
    labeled_set: set[int] = set()

    seed = _seed_indices(X_scaled, risk, budget)
    if seed:
        y0 = oracle_fn(seed)
        labeled_idx.extend(seed)
        labeled_y.extend([int(v) for v in y0])
        labeled_set.update(seed)

    reserve_exploit = max(10, int(0.12 * budget))

    while len(labeled_idx) < budget - reserve_exploit:
        y_arr = np.asarray(labeled_y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            remaining = budget - reserve_exploit - len(labeled_idx)
            if remaining <= 0:
                break
            unl = np.array([i for i in np.argsort(-risk["blended"]) if i not in labeled_set], dtype=int)
            if len(unl) == 0:
                break
            take = unl[: min(remaining, 8)]
            ys = oracle_fn(take.tolist())
            labeled_idx.extend(take.tolist())
            labeled_y.extend([int(v) for v in ys])
            labeled_set.update(take.tolist())
            continue

        models = _fit_models(X_scaled[np.asarray(labeled_idx, dtype=int)], y_arr)
        p_all = _ensemble_proba(models, X_scaled)

        unlabeled = np.array([i for i in range(n) if i not in labeled_set], dtype=int)
        if len(unlabeled) == 0:
            break

        uncertainty = 1.0 - 2.0 * np.abs(p_all[unlabeled] - 0.5)
        risk_u = risk["blended"][unlabeled]
        dens_u = density[unlabeled]

        pos_count = int(np.sum(y_arr))
        if pos_count < 8:
            score = 0.45 * risk_u + 0.25 * uncertainty + 0.20 * dens_u + 0.10 * p_all[unlabeled]
        else:
            score = 0.40 * uncertainty + 0.35 * p_all[unlabeled] + 0.15 * dens_u + 0.10 * risk_u

        batch = min(8, budget - reserve_exploit - len(labeled_idx))
        if batch <= 0:
            break
        picks = unlabeled[np.argsort(-score)[:batch]]
        ys = oracle_fn(picks.tolist())

        labeled_idx.extend(picks.tolist())
        labeled_y.extend([int(v) for v in ys])
        labeled_set.update(picks.tolist())

    if len(labeled_idx) < budget and len(np.unique(np.asarray(labeled_y, dtype=int))) >= 2:
        models = _fit_models(X_scaled[np.asarray(labeled_idx, dtype=int)], np.asarray(labeled_y, dtype=int))
        p_all = _ensemble_proba(models, X_scaled)
        unlabeled = np.array([i for i in range(n) if i not in labeled_set], dtype=int)
        if len(unlabeled) > 0:
            remain = budget - len(labeled_idx)
            picks = unlabeled[np.argsort(-p_all[unlabeled])[:remain]]
            ys = oracle_fn(picks.tolist())
            labeled_idx.extend(picks.tolist())
            labeled_y.extend([int(v) for v in ys])
            labeled_set.update(picks.tolist())

    idx = np.asarray(labeled_idx, dtype=int)
    y = np.asarray(labeled_y, dtype=int)

    if len(idx) == 0 or len(np.unique(y)) < 2:
        return (risk["blended"] >= np.quantile(risk["blended"], 0.92)).astype(int)

    final_models = _fit_models(X_scaled[idx], y)
    p_final = _ensemble_proba(final_models, X_scaled)

    t = _best_threshold(y, p_final[idx])
    preds = (p_final >= t).astype(int)
    return preds