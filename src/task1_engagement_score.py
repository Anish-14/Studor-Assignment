import numpy as np
import pandas as pd
from scipy.stats import linregress

WEIGHTS = {
    "frequency_score":   0.30,
    "recency_score":     0.25,
    "diversity_score":   0.20,
    "assessment_score":  0.15,
    "trend_score":       0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def compute_recency_score(df):
    week_end   = (df["week"] + 1) * 7
    days_since = (week_end - df["last_active_day"]).clip(lower=0)
    score      = 1 - (days_since.clip(upper=14) / 14)
    return score.rename("recency_score")


def compute_frequency_score(df, click_cap=300.0):
    score = df["total_clicks"].clip(upper=click_cap) / click_cap
    return score.rename("frequency_score")


def compute_diversity_score(df, max_types=8.0):
    score = df["unique_act_types"].clip(upper=max_types) / max_types
    return score.rename("diversity_score")


def compute_assessment_score(df):
    has_submission   = (df["n_submissions"] > 0).astype(float)
    delay_score      = 1 - (df["avg_delay"].clip(lower=0, upper=7) / 7)
    normalised_score = df["avg_score"].fillna(50) / 100.0

    score = np.where(
        has_submission > 0,
        0.4 * has_submission + 0.3 * delay_score + 0.3 * normalised_score,
        0.2,
    )
    return pd.Series(score, index=df.index, name="assessment_score")


def compute_trend_score(df):
    data   = df[["id_student", "week", "total_clicks"]].copy()
    slopes = {}

    for (sid,), grp in data.groupby(["id_student"]):
        grp    = grp.sort_values("week")
        weeks  = grp["week"].values
        clicks = grp["total_clicks"].values
        for i, w in enumerate(weeks):
            w_slice = weeks[max(0, i - 2): i + 1]
            c_slice = clicks[max(0, i - 2): i + 1]
            if len(w_slice) >= 2:
                slope, *_ = linregress(w_slice, c_slice)
            else:
                slope = 0.0
            slopes[(sid, w)] = slope

    raw_slope  = pd.Series(
        [slopes.get((r.id_student, r.week), 0.0) for r in df.itertuples()],
        index=df.index,
    )
    normalised = (raw_slope.clip(-30, 30) + 30) / 60
    return normalised.rename("trend_score")


def compute_engagement_scores(weekly_df):
    df = weekly_df.copy()

    df["recency_score"]    = compute_recency_score(df)
    df["frequency_score"]  = compute_frequency_score(df)
    df["diversity_score"]  = compute_diversity_score(df)
    df["assessment_score"] = compute_assessment_score(df)
    df["trend_score"]      = compute_trend_score(df)

    score                  = sum(WEIGHTS[feat] * df[feat] for feat in WEIGHTS)
    df["engagement_score"] = (score * 100).clip(0, 100).round(1)

    return df


def select_archetypes(scored_df):
    summary = (
        scored_df.groupby("id_student")["engagement_score"]
        .agg(
            mean   = "mean",
            std    = "std",
            first5 = lambda x: x.iloc[:5].mean() if len(x) >= 5 else x.mean(),
            last5  = lambda x: x.iloc[-5:].mean() if len(x) >= 5 else x.mean(),
        )
        .reset_index()
    )

    summary["steady_score"] = summary["mean"] - summary["std"]
    steady_id  = summary.sort_values("steady_score", ascending=False).iloc[0]["id_student"]

    summary["drop_score"] = summary["last5"] - summary["first5"]
    dropout_id = summary.sort_values("drop_score").iloc[0]["id_student"]

    late_df  = summary[summary["id_student"] != dropout_id]
    late_id  = late_df.sort_values("drop_score", ascending=False).iloc[0]["id_student"]

    return {
        "Steady Engager": int(steady_id),
        "Early Dropout":  int(dropout_id),
        "Late Recoverer": int(late_id),
    }
