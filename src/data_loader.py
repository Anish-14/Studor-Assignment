import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

REQUIRED_FILES = [
    "studentInfo.csv",
    "studentVle.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "studentRegistration.csv",
    "vle.csv",
    "courses.csv",
]


def load_raw_data(data_dir=DATA_DIR):
    missing = [f for f in REQUIRED_FILES
               if not os.path.exists(os.path.join(data_dir, f))]

    if missing:
        from src.data_generator import generate_and_save_all
        return generate_and_save_all(data_dir)

    dfs = {}

    df = pd.read_csv(os.path.join(data_dir, "studentInfo.csv"))
    df["imd_band"]     = df["imd_band"].astype("category")
    df["final_result"] = df["final_result"].astype("category")
    dfs["student_info"] = df

    df = pd.read_csv(os.path.join(data_dir, "studentVle.csv"))
    df["sum_click"] = pd.to_numeric(df["sum_click"], errors="coerce").fillna(0)
    dfs["student_vle"] = df

    dfs["vle"] = pd.read_csv(os.path.join(data_dir, "vle.csv"))

    df = pd.read_csv(os.path.join(data_dir, "assessments.csv"))
    df["date"] = pd.to_numeric(df["date"], errors="coerce").fillna(999)
    dfs["assessments"] = df

    dfs["student_assessment"] = pd.read_csv(
        os.path.join(data_dir, "studentAssessment.csv"))

    dfs["student_registration"] = pd.read_csv(
        os.path.join(data_dir, "studentRegistration.csv"))

    dfs["courses"] = pd.read_csv(os.path.join(data_dir, "courses.csv"))

    return dfs


def build_week_column(student_vle):
    df         = student_vle.copy()
    df["week"] = (df["date"] // 7).astype(int)
    return df


def build_unified_dataset(raw):
    svle     = build_week_column(raw["student_vle"])
    vle_meta = raw["vle"][["id_site", "activity_type"]].drop_duplicates("id_site")
    svle     = svle.merge(vle_meta, on="id_site", how="left")
    svle["activity_type"] = svle["activity_type"].fillna("unknown")

    weekly = (
        svle.groupby(["id_student", "code_module", "week"])
        .agg(
            total_clicks      = ("sum_click",     "sum"),
            unique_activities = ("id_site",       "nunique"),
            unique_act_types  = ("activity_type", "nunique"),
            last_active_day   = ("date",          "max"),
        )
        .reset_index()
    )

    info_cols = [
        "id_student", "code_module", "final_result",
        "gender", "age_band", "imd_band", "highest_education",
        "num_of_prev_attempts", "studied_credits", "disability",
    ]
    existing = [c for c in info_cols if c in raw["student_info"].columns]
    weekly   = weekly.merge(raw["student_info"][existing],
                            on=["id_student", "code_module"], how="left")

    reg = raw["student_registration"][
        ["id_student", "code_module", "date_unregistration"]
    ].copy()
    reg["withdrawal_week"] = (reg["date_unregistration"] // 7).where(
        reg["date_unregistration"].notna(), other=np.nan)
    weekly = weekly.merge(reg, on=["id_student", "code_module"], how="left")

    asmnt = raw["assessments"].merge(
        raw["student_assessment"], on="id_assessment", how="inner")
    asmnt["submission_week"]  = (asmnt["date_submitted"] // 7).astype("Int64")
    asmnt["submission_delay"] = asmnt["date_submitted"] - asmnt["date"]

    weekly_asmnt = (
        asmnt.groupby(["id_student", "submission_week"])
        .agg(
            avg_score     = ("score",            "mean"),
            n_submissions = ("id_assessment",    "count"),
            avg_delay     = ("submission_delay", "mean"),
        )
        .reset_index()
        .rename(columns={"submission_week": "week"})
    )
    weekly = weekly.merge(weekly_asmnt, on=["id_student", "week"], how="left")
    weekly["avg_score"]     = weekly["avg_score"].fillna(np.nan)
    weekly["n_submissions"] = weekly["n_submissions"].fillna(0)
    weekly["avg_delay"]     = weekly["avg_delay"].fillna(0)

    weekly["target"] = weekly["final_result"].isin(
        ["Withdrawn", "Fail"]).astype(int)

    return weekly.sort_values(["id_student", "week"]).reset_index(drop=True)
