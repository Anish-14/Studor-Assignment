import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

PREDICTION_WEEK  = 6
RECALL_THRESHOLD = 0.50

FEATURE_COLS = [
    "avg_engagement_score",
    "max_engagement_score",
    "engagement_trend",
    "total_clicks_w6",
    "avg_weekly_clicks",
    "active_weeks",
    "max_silence_gap",
    "avg_diversity",
    "n_submissions_w6",
    "avg_score_w6",
    "avg_delay_w6",
    "days_to_registration",
    "num_of_prev_attempts",
    "studied_credits",
]


def build_student_features(scored_df, student_reg=None):
    df = scored_df[scored_df["week"] <= PREDICTION_WEEK].copy()

    agg = df.groupby("id_student").agg(
        avg_engagement_score = ("engagement_score", "mean"),
        max_engagement_score = ("engagement_score", "max"),
        total_clicks_w6      = ("total_clicks",     "sum"),
        avg_weekly_clicks    = ("total_clicks",     "mean"),
        active_weeks         = ("total_clicks",     lambda x: (x > 0).sum()),
        avg_diversity        = ("diversity_score",  "mean"),
        n_submissions_w6     = ("n_submissions",    "sum"),
        avg_score_w6         = ("avg_score",        "mean"),
        avg_delay_w6         = ("avg_delay",        "mean"),
    ).reset_index()

    def _slope(grp):
        w = grp["week"].values
        s = grp["engagement_score"].values
        if len(w) < 2:
            return 0.0
        from scipy.stats import linregress
        slope, *_ = linregress(w, s)
        return float(slope)

    trend = (
        df.groupby("id_student")
        .apply(_slope)
        .reset_index()
        .rename(columns={0: "engagement_trend"})
    )
    agg = agg.merge(trend, on="id_student", how="left")

    def _max_silence(grp):
        active_weeks = set(grp[grp["total_clicks"] > 0]["week"].tolist())
        seq          = [1 if w in active_weeks else 0
                        for w in range(0, PREDICTION_WEEK + 1)]
        max_gap, cur = 0, 0
        for v in seq:
            if v == 0:
                cur    += 1
                max_gap = max(max_gap, cur)
            else:
                cur = 0
        return max_gap

    silence = (
        df.groupby("id_student")
        .apply(_max_silence)
        .reset_index()
        .rename(columns={0: "max_silence_gap"})
    )
    agg = agg.merge(silence, on="id_student", how="left")

    demo_cols = ["id_student", "num_of_prev_attempts", "studied_credits",
                 "final_result", "target"]
    demo_cols = [c for c in demo_cols if c in scored_df.columns]
    demo      = scored_df[demo_cols].drop_duplicates("id_student").copy()
    agg       = agg.merge(demo, on="id_student", how="left")

    if student_reg is not None:
        reg_timing = (
            student_reg[["id_student", "date_registration"]]
            .drop_duplicates("id_student")
            .rename(columns={"date_registration": "days_to_registration"})
        )
        agg = agg.merge(reg_timing, on="id_student", how="left")
    else:
        agg["days_to_registration"] = 0

    agg["avg_score_w6"]         = agg["avg_score_w6"].fillna(50.0)
    agg["avg_delay_w6"]         = agg["avg_delay_w6"].fillna(0.0)
    agg["days_to_registration"] = agg["days_to_registration"].fillna(0)
    agg["num_of_prev_attempts"] = agg.get(
        "num_of_prev_attempts", pd.Series(0, index=agg.index)).fillna(0)
    agg["studied_credits"]      = agg.get(
        "studied_credits", pd.Series(60, index=agg.index)).fillna(60)

    return agg


def get_features_and_target(student_features):
    available = [c for c in FEATURE_COLS if c in student_features.columns]
    X         = student_features[available].copy()
    y         = student_features["target"].astype(int)
    return X, y, available


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=0.5,
                random_state=42,
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ))
        ]),
    }


def evaluate_model(model, X, y, model_name, threshold=RECALL_THRESHOLD):
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred  = (proba >= threshold).astype(int)

    return {
        "model":     model_name,
        "threshold": threshold,
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall":    round(recall_score(y, pred, zero_division=0),    4),
        "f1":        round(f1_score(y, pred, zero_division=0),        4),
        "roc_auc":   round(roc_auc_score(y, proba),                   4),
        "brier":     round(brier_score_loss(y, proba),                 4),
        "confusion": confusion_matrix(y, pred).tolist(),
        "proba":     proba,
        "pred":      pred,
    }


def get_feature_importance(model, feature_names):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_task2(scored_df, student_reg=None):
    student_features     = build_student_features(scored_df, student_reg)
    X, y, feature_names  = get_features_and_target(student_features)

    models  = build_models()
    results = []

    for name, model in models.items():
        model.fit(X, y)
        res = evaluate_model(model, X, y, name)
        results.append(res)

    best_result   = max(results, key=lambda r: r["roc_auc"])
    best_model    = models[best_result["model"]]
    importance_df = get_feature_importance(best_model, feature_names)

    proba                = best_result["proba"]
    frac_pos, mean_pred  = calibration_curve(y, proba, n_bins=10,
                                              strategy="uniform")
    calibration = {
        "frac_pos":  frac_pos,
        "mean_pred": mean_pred,
        "brier":     best_result["brier"],
    }

    student_features["risk_score"] = best_model.predict_proba(X)[:, 1]
    student_features["alert"]      = (
        student_features["risk_score"] >= 0.70).astype(int)

    return {
        "student_features": student_features,
        "results":          results,
        "best_result":      best_result,
        "models":           models,
        "best_model":       best_model,
        "feature_names":    feature_names,
        "importance_df":    importance_df,
        "calibration":      calibration,
    }


def generate_advisor_alerts(student_features, importance_df, threshold=0.70):
    alerts = student_features[
        student_features["risk_score"] >= threshold].copy()

    alerts["risk_level"] = pd.cut(
        alerts["risk_score"],
        bins=[0, 0.70, 0.85, 1.0],
        labels=["Moderate (70-85%)", "High (85-95%)", "Critical (>95%)"],
    )

    top_feature              = (importance_df.iloc[0]["feature"]
                                if len(importance_df) > 0
                                else "avg_engagement_score")
    alerts["top_risk_factor"] = top_feature

    def _suggest(row):
        if row.get("active_weeks", 3) <= 2:
            return "Contact student immediately — minimal platform activity detected."
        elif row.get("avg_score_w6", 60) < 40:
            return "Schedule academic support session — low assessment scores."
        elif row.get("engagement_trend", 0) < -2:
            return "Reach out — engagement is declining week-on-week."
        else:
            return "Send check-in email; monitor for next 2 weeks."

    alerts["suggested_action"] = alerts.apply(_suggest, axis=1)

    output_cols = [
        "id_student", "risk_score", "risk_level",
        "avg_engagement_score", "active_weeks", "avg_score_w6",
        "top_risk_factor", "suggested_action",
    ]
    output_cols = [c for c in output_cols if c in alerts.columns]
    return (alerts[output_cols]
            .sort_values("risk_score", ascending=False)
            .reset_index(drop=True))
