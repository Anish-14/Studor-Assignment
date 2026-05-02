import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def build_course_profiles(student_features, student_info):
    sf = student_features.copy()
    if "code_module" not in sf.columns:
        sf = sf.merge(
            student_info[["id_student", "code_module"]].drop_duplicates(),
            on="id_student", how="left"
        )

    profiles = sf.groupby("code_module").agg(
        avg_engagement = ("avg_engagement_score", "mean"),
        avg_score      = ("avg_score_w6",         "mean"),
        pass_rate      = ("target",               lambda x: 1 - x.mean()),
        avg_clicks     = ("total_clicks_w6",      "mean"),
        avg_active_wks = ("active_weeks",         "mean"),
        n_students     = ("id_student",           "count"),
    ).reset_index()

    scale_cols = ["avg_engagement", "avg_score", "avg_clicks", "avg_active_wks"]
    scale_cols = [c for c in scale_cols if c in profiles.columns]
    profiles[scale_cols] = MinMaxScaler().fit_transform(profiles[scale_cols])
    profiles["pass_rate"] = profiles["pass_rate"].clip(0, 1)

    return profiles


def _build_student_vector(student_row, feature_cols, course_matrix):
    source_map = {
        "avg_engagement": "avg_engagement_score",
        "avg_score":      "avg_score_w6",
        "avg_clicks":     "total_clicks_w6",
        "avg_active_wks": "active_weeks",
        "pass_rate":      None,
    }
    vec = []
    for col in feature_cols:
        src = source_map.get(col)
        if src and src in student_row.index:
            vec.append(float(student_row[src]))
        else:
            vec.append(0.5)

    vec       = np.array(vec, dtype=float)
    col_min   = course_matrix.min(axis=0)
    col_max   = course_matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    vec_norm  = np.clip((vec - col_min) / col_range, 0, 1)

    return vec_norm


def content_based_recommend(student_row, course_profiles, current_course=None, top_k=3):
    feature_cols = ["avg_engagement", "avg_score", "pass_rate",
                    "avg_clicks", "avg_active_wks"]
    feature_cols  = [c for c in feature_cols if c in course_profiles.columns]
    course_matrix = course_profiles[feature_cols].values

    student_vec   = _build_student_vector(student_row, feature_cols, course_matrix)

    col_min   = course_matrix.min(axis=0)
    col_max   = course_matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    course_matrix_norm = (course_matrix - col_min) / col_range

    sims = cosine_similarity(
        student_vec.reshape(1, -1), course_matrix_norm)[0]

    rec              = course_profiles[["code_module"]].copy()
    rec["similarity"] = sims
    rec["approach"]   = "Content-Based"

    if current_course:
        rec = rec[rec["code_module"] != current_course]

    return (rec
            .sort_values("similarity", ascending=False)
            .head(top_k)
            .reset_index(drop=True))


def build_interaction_matrix(student_info, student_features):
    sf = student_features.copy()
    if "code_module" not in sf.columns:
        sf = sf.merge(
            student_info[["id_student", "code_module"]].drop_duplicates(),
            on="id_student", how="left"
        )

    return sf.pivot_table(
        index="id_student",
        columns="code_module",
        values="avg_engagement_score",
        aggfunc="mean",
        fill_value=0.0,
    )


def collaborative_recommend(student_id, interaction_matrix, student_features,
                             top_k=3, n_neighbours=10):
    if student_id not in interaction_matrix.index:
        return pd.DataFrame(columns=["code_module", "similarity", "approach"])

    target_vec = interaction_matrix.loc[[student_id]].values
    all_vecs   = interaction_matrix.values
    sims       = cosine_similarity(target_vec, all_vecs)[0]

    sim_series = pd.Series(sims, index=interaction_matrix.index)
    sim_series = sim_series.drop(index=student_id, errors="ignore")
    neighbours = sim_series.nlargest(n_neighbours).index

    taken     = set(interaction_matrix.columns[
        interaction_matrix.loc[student_id] > 0])
    not_taken = [c for c in interaction_matrix.columns if c not in taken]

    if not not_taken:
        return pd.DataFrame(columns=["code_module", "similarity", "approach"])

    rec_scores = interaction_matrix.loc[neighbours, not_taken].mean(axis=0)

    return (
        pd.DataFrame({
            "code_module": rec_scores.index,
            "similarity":  rec_scores.values,
            "approach":    "Collaborative Filtering",
        })
        .sort_values("similarity", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def cold_start_recommend(course_profiles, top_k=3):
    profiles   = course_profiles.copy()
    popularity = profiles["n_students"] / profiles["n_students"].max()

    profiles["cold_start_score"] = (
        profiles["pass_rate"] * 0.7 + popularity * 0.3)

    rec              = (profiles
                        .sort_values("cold_start_score", ascending=False)
                        .head(top_k)[["code_module", "cold_start_score"]]
                        .rename(columns={"cold_start_score": "similarity"})
                        .copy())
    rec["approach"]  = "Cold-Start"
    return rec.reset_index(drop=True)


def evaluate_precision_at_k(student_info, student_features, course_profiles,
                              k=3, holdout_frac=0.20):
    np.random.seed(42)
    sf = student_features.copy()
    if "code_module" not in sf.columns:
        sf = sf.merge(
            student_info[["id_student", "code_module"]].drop_duplicates(),
            on="id_student", how="left"
        )

    all_ids            = sf["id_student"].unique()
    n_holdout          = max(50, int(len(all_ids) * holdout_frac))
    holdout            = np.random.choice(all_ids, size=n_holdout, replace=False)
    interaction_matrix = build_interaction_matrix(student_info, sf)

    hits_cb = 0
    hits_cf = 0

    for sid in holdout:
        row    = sf[sf["id_student"] == sid].iloc[0]
        actual = row.get("code_module", None)
        if actual is None:
            continue

        recs_cb = content_based_recommend(
            row, course_profiles, current_course=None, top_k=k)
        if actual in recs_cb["code_module"].values:
            hits_cb += 1

        recs_cf = collaborative_recommend(
            sid, interaction_matrix, sf, top_k=k)
        if actual in recs_cf["code_module"].values:
            hits_cf += 1

    n = len(holdout)
    return {
        "content_based_precision_at_k": round(hits_cb / n, 4),
        "collaborative_precision_at_k": round(hits_cf / n, 4),
        "k":         k,
        "n_holdout": n,
    }


def run_task3(student_features, student_info):
    sf = student_features.copy()
    if "code_module" not in sf.columns:
        sf = sf.merge(
            student_info[["id_student", "code_module"]].drop_duplicates(),
            on="id_student", how="left"
        )

    course_profiles    = build_course_profiles(sf, student_info)
    interaction_matrix = build_interaction_matrix(student_info, sf)

    sample_ids  = sf["id_student"].sample(3, random_state=42).values
    sample_recs = {}

    for sid in sample_ids:
        row         = sf[sf["id_student"] == sid].iloc[0]
        current_mod = row.get("code_module", None)

        sample_recs[int(sid)] = {
            "content_based": content_based_recommend(
                row, course_profiles, current_course=current_mod, top_k=3),
            "collaborative":  collaborative_recommend(
                sid, interaction_matrix, sf, top_k=3),
            "cold_start":     cold_start_recommend(course_profiles, top_k=3),
        }

    eval_metrics = evaluate_precision_at_k(
        student_info, sf, course_profiles, k=3)

    return {
        "course_profiles":              course_profiles,
        "interaction_matrix":           interaction_matrix,
        "sample_recs":                  sample_recs,
        "eval_metrics":                 eval_metrics,
        "student_features_with_module": sf,
    }
