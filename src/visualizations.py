import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "primary":   "#E63946",
    "secondary": "#457B9D",
    "accent":    "#2A9D8F",
    "warn":      "#E9C46A",
    "neutral":   "#6C757D",
    "bg":        "#F8F9FA",
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "figure.dpi":       150,
})


def plot_engagement_trajectories(scored_df, archetypes, save=True):
    fig, ax = plt.subplots(figsize=(13, 5))

    colors = {
        "Steady Engager": PALETTE["accent"],
        "Early Dropout":  PALETTE["primary"],
        "Late Recoverer": PALETTE["secondary"],
    }
    linestyles = {
        "Steady Engager": "-",
        "Early Dropout":  "--",
        "Late Recoverer": "-.",
    }

    for label, sid in archetypes.items():
        stu = scored_df[scored_df["id_student"] == sid].sort_values("week")
        if stu.empty:
            continue
        ax.plot(
            stu["week"], stu["engagement_score"],
            color=colors[label], linestyle=linestyles[label],
            linewidth=2.5, marker="o", markersize=4,
            label=f"{label}  (Student {sid})",
        )

    ax.axvspan(0, 6, alpha=0.09, color=PALETTE["warn"],
               label="Prediction window (Week <= 6)")
    ax.axhline(40, color=PALETTE["primary"], linewidth=1.2,
               linestyle=":", alpha=0.8, label="At-risk threshold (40)")

    ax.set_xlabel("Week Number", fontsize=12)
    ax.set_ylabel("Engagement Score  (0-100)", fontsize=12)
    ax.set_title("Fig 1 — Weekly Engagement Trajectories by Student Archetype",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig1_engagement_trajectories.png")
    if save:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_contributions(weights, save=True):
    labels = [k.replace("_score", "").replace("_", " ").title() for k in weights]
    values = list(weights.values())
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              PALETTE["warn"], PALETTE["neutral"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars    = ax.barh(labels, values, color=colors[:len(labels)],
                      edgecolor="white", height=0.55)

    for bar, val in zip(bars, values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Weight in Engagement Score", fontsize=11)
    ax.set_title("Fig 2 — Feature Contributions to Engagement Score",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, max(values) * 1.30)
    ax.invert_yaxis()
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig2_feature_contributions.png")
    if save:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(results, y_true, save=True):
    fig, ax       = plt.subplots(figsize=(7, 6))
    model_colors  = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    for res, color in zip(results, model_colors):
        fpr, tpr, _ = roc_curve(y_true, res["proba"])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{res['model']}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6,
            label="Random classifier  (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="grey")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("Fig 3 — ROC Curves: Disengagement Prediction Models",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig3_roc_curves.png")
    if save:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_calibration(calibration, save=True):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(calibration["mean_pred"], calibration["frac_pos"],
            "o-", color=PALETTE["primary"], lw=2.2, markersize=7,
            label=f"Model  (Brier = {calibration['brier']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Perfect calibration")
    ax.fill_between(
        calibration["mean_pred"], calibration["frac_pos"],
        calibration["mean_pred"],
        alpha=0.12, color=PALETTE["primary"], label="Calibration gap")

    ax.set_xlabel("Mean Predicted Risk", fontsize=12)
    ax.set_ylabel("Fraction of Actual Withdrawals / Fails", fontsize=12)
    ax.set_title("Fig 4 — Calibration Plot  (Reliability Diagram)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig4_calibration.png")
    if save:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(result, save=True):
    cm     = np.array(result["confusion"])
    labels = ["Pass / Distinction\n(Predicted Negative)",
              "Withdrawn / Fail\n(Predicted Positive)"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=labels,
        yticklabels=["Pass / Distinction\n(True Negative)",
                     "Withdrawn / Fail\n(True Positive)"],
        linewidths=0.5, linecolor="white",
        ax=ax, cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(
        f"Fig 5 — Confusion Matrix:  {result['model']}\n"
        f"Threshold = {result['threshold']}   |   "
        f"Recall = {result['recall']:.3f}   |   "
        f"Precision = {result['precision']:.3f}",
        fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig5_confusion_matrix.png")
    if save:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(importance_df, top_n=10, save=True):
    df     = importance_df.head(top_n).copy()
    df["label"] = df["feature"].str.replace("_", " ").str.title()
    n      = len(df)
    colors = ([PALETTE["primary"]] * min(3, n) +
              [PALETTE["secondary"]] * max(0, n - 3))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["label"][::-1], df["importance"][::-1],
            color=colors[::-1], edgecolor="white", height=0.65)

    max_val = df["importance"].max()
    for i, val in enumerate(df["importance"][::-1]):
        ax.text(val + max_val * 0.01, i, f"{val:.4f}", va="center", fontsize=9)

    top3_patch = mpatches.Patch(color=PALETTE["primary"],   label="Top 3 drivers")
    rest_patch = mpatches.Patch(color=PALETTE["secondary"], label="Other features")
    ax.legend(handles=[top3_patch, rest_patch], loc="lower right", fontsize=10)

    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Fig 6 — Feature Importances: Drivers of Disengagement Risk",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_facecolor(PALETTE["bg"])
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig6_feature_importance.png")
    if save:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_course_profiles(course_profiles, save=True):
    feat_cols = ["avg_engagement", "avg_score", "pass_rate",
                 "avg_clicks", "avg_active_wks"]
    feat_cols = [c for c in feat_cols if c in course_profiles.columns]

    heat_data       = course_profiles.set_index("code_module")[feat_cols].T
    heat_data.index = [c.replace("_", " ").title() for c in heat_data.index]

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(
        heat_data, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "Normalised value  (0-1)", "shrink": 0.8},
        annot_kws={"size": 9}, vmin=0, vmax=1,
    )
    ax.set_title("Fig 7 — Course Profile Heatmap  (Normalised Feature Vectors)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Course Code", fontsize=11)
    ax.set_ylabel("Feature",     fontsize=11)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig7_course_profiles.png")
    if save:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recommendation_comparison(eval_metrics, save=True):
    approaches = ["Content-Based\nFiltering", "Collaborative\nFiltering"]
    scores     = [
        eval_metrics["content_based_precision_at_k"],
        eval_metrics["collaborative_precision_at_k"],
    ]
    colors = [PALETTE["accent"], PALETTE["secondary"]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars    = ax.bar(approaches, scores, color=colors,
                     edgecolor="white", width=0.45)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=13, fontweight="bold")

    k = eval_metrics["k"]
    n = eval_metrics["n_holdout"]
    ax.set_ylabel(f"Precision @ {k}", fontsize=11)
    ax.set_title(
        f"Fig 8 — Recommendation Engine: Precision@{k} Comparison\n(holdout n={n})",
        fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(scores) * 1.40 if max(scores) > 0 else 0.5)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig8_recommendation_comparison.png")
    if save:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(scored_df, archetypes, weights, task2_output, task3_output):
    paths = []
    paths.append(plot_engagement_trajectories(scored_df, archetypes))
    paths.append(plot_feature_contributions(weights))

    y_true = task2_output["student_features"]["target"].values
    paths.append(plot_roc_curves(task2_output["results"], y_true))
    paths.append(plot_calibration(task2_output["calibration"]))
    paths.append(plot_confusion_matrix(task2_output["best_result"]))
    paths.append(plot_feature_importance(task2_output["importance_df"]))
    paths.append(plot_course_profiles(task3_output["course_profiles"]))
    paths.append(plot_recommendation_comparison(task3_output["eval_metrics"]))

    return paths
