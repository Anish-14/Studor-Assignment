import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader               import load_raw_data, build_unified_dataset
from src.task1_engagement_score    import compute_engagement_scores, select_archetypes, WEIGHTS
from src.task2_disengagement_model import run_task2, generate_advisor_alerts
from src.task3_recommendation      import run_task3
from src.visualizations            import generate_all_plots


def main():
    t_start = time.time()

    print("\nLoading data ...")
    raw     = load_raw_data()
    unified = build_unified_dataset(raw)
    print(f"  Students: {unified['id_student'].nunique()} | "
          f"Rows: {len(unified)}")
    print(f"  Outcome distribution:\n"
          + unified.drop_duplicates("id_student")["final_result"]
          .value_counts().to_string())

    print("\nTask 1 — Computing engagement scores ...")
    scored     = compute_engagement_scores(unified)
    archetypes = select_archetypes(scored)
    print(f"  Score range: {scored['engagement_score'].min():.1f} – "
          f"{scored['engagement_score'].max():.1f}  |  "
          f"Mean: {scored['engagement_score'].mean():.1f}")
    for label, sid in archetypes.items():
        mean_sc = scored[scored["id_student"] == sid]["engagement_score"].mean()
        print(f"  {label}: Student {sid}  (mean score {mean_sc:.1f})")

    print("\nTask 2 — Training disengagement models ...")
    task2 = run_task2(scored, raw.get("student_registration"))

    print(f"\n  {'Model':<26} {'AUC':>6}  {'Recall':>7}  {'Precision':>10}  {'F1':>6}")
    print("  " + "-" * 58)
    for res in task2["results"]:
        print(f"  {res['model']:<26} {res['roc_auc']:>6.3f}  "
              f"{res['recall']:>7.3f}  {res['precision']:>10.3f}  {res['f1']:>6.3f}")

    best = task2["best_result"]
    print(f"\n  Best model : {best['model']}  (AUC = {best['roc_auc']:.3f})")
    print(f"  Top 3 features driving disengagement:")
    for _, row in task2["importance_df"].head(3).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:.4f}")

    alerts = generate_advisor_alerts(
        task2["student_features"], task2["importance_df"], threshold=0.70)
    print(f"\n  Students flagged at >=70% risk: {len(alerts)}")

    print("\nTask 3 — Building recommendation engine ...")
    sf = task2["student_features"].merge(
        raw["student_info"][["id_student", "code_module"]].drop_duplicates(),
        on="id_student", how="left")
    task3 = run_task3(sf, raw["student_info"])

    em = task3["eval_metrics"]
    print(f"  Content-Based  P@3: {em['content_based_precision_at_k']:.4f}")
    print(f"  Collaborative  P@3: {em['collaborative_precision_at_k']:.4f}")

    print("\nGenerating plots ...")
    plot_paths = generate_all_plots(
        scored_df    = scored,
        archetypes   = archetypes,
        weights      = WEIGHTS,
        task2_output = task2,
        task3_output = task3,
    )
    for p in plot_paths:
        print(f"  Saved: {os.path.basename(p)}")

    print(f"\nDone in {time.time() - t_start:.1f}s — "
          f"{len(plot_paths)} plots saved to outputs/plots/")


if __name__ == "__main__":
    main()
