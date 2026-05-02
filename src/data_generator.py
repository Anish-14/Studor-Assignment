import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

N_STUDENTS     = 3000
N_WEEKS        = 26
COURSES        = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
ACTIVITY_TYPES = [
    "forumng", "oucollaborate", "content", "quiz",
    "resource", "url", "homepage", "subpage",
    "glossary", "dataplus"
]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _outcome_for_profile(engagement_level, rng):
    r = rng.random()
    if engagement_level > 0.65:
        return rng.choice(["Pass", "Distinction"], p=[0.55, 0.45])
    elif engagement_level > 0.35:
        if r < 0.45:
            return "Pass"
        elif r < 0.70:
            return "Fail"
        else:
            return "Withdrawn"
    else:
        if r < 0.15:
            return "Pass"
        elif r < 0.55:
            return "Fail"
        else:
            return "Withdrawn"


def generate_student_info(n=N_STUDENTS):
    rng        = RNG
    ids        = np.arange(1, n + 1)
    courses    = rng.choice(COURSES, size=n)
    latent_eng = rng.beta(2, 2, size=n)
    results    = [_outcome_for_profile(e, rng) for e in latent_eng]

    return pd.DataFrame({
        "id_student":           ids,
        "code_module":          courses,
        "code_presentation":    rng.choice(["2013B", "2014B", "2014J"], size=n),
        "gender":               rng.choice(["M", "F"], size=n),
        "region":               rng.choice(
            ["London", "South East", "North West", "Scotland",
             "East Anglian", "Yorkshire", "Wales"], size=n),
        "highest_education":    rng.choice(
            ["HE Qualification", "A Level or Equivalent",
             "Lower Than A Level", "Post Graduate Qualification",
             "No Formal quals"], size=n, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
        "imd_band":             rng.choice(
            ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
             "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"], size=n),
        "age_band":             rng.choice(
            ["0-35", "35-55", "55<="], size=n, p=[0.60, 0.30, 0.10]),
        "num_of_prev_attempts": rng.choice(
            [0, 1, 2, 3], size=n, p=[0.70, 0.18, 0.08, 0.04]),
        "studied_credits":      rng.choice(
            [30, 60, 90, 120], size=n, p=[0.20, 0.50, 0.20, 0.10]),
        "disability":           rng.choice(["Y", "N"], size=n, p=[0.10, 0.90]),
        "final_result":         results,
        "_latent_engagement":   latent_eng,
    })


def generate_student_registration(student_info):
    rng              = RNG
    n                = len(student_info)
    withdrawal_dates = []

    for result in student_info["final_result"]:
        if result == "Withdrawn":
            wd = int(rng.choice(range(1, 25),
                                p=np.array([0.15, 0.15, 0.12, 0.10, 0.08,
                                            0.07, 0.06, 0.05, 0.04, 0.04,
                                            0.03, 0.03, 0.02, 0.02, 0.01,
                                            0.01, 0.01, 0.00, 0.00, 0.00,
                                            0.00, 0.00, 0.00, 0.01])))
            withdrawal_dates.append(wd * 7)
        else:
            withdrawal_dates.append(np.nan)

    return pd.DataFrame({
        "id_student":          student_info["id_student"].values,
        "code_module":         student_info["code_module"].values,
        "code_presentation":   student_info["code_presentation"].values,
        "date_registration":   rng.integers(-30, 1, size=n),
        "date_unregistration": withdrawal_dates,
    })


def generate_vle(courses=COURSES):
    rows   = []
    act_id = 1
    for course in courses:
        for act_type in ACTIVITY_TYPES:
            for _ in range(3):
                rows.append({
                    "id_site":           act_id,
                    "code_module":       course,
                    "code_presentation": "2014B",
                    "activity_type":     act_type,
                    "week_from":         1,
                    "week_to":           N_WEEKS,
                })
                act_id += 1
    return pd.DataFrame(rows)


def _weekly_clicks_for_student(latent_eng, result, n_weeks, rng):
    base   = latent_eng * 80 + 5
    clicks = np.zeros(n_weeks)
    for w in range(n_weeks):
        noise = rng.normal(0, base * 0.3)
        trend = 0.0
        if result == "Withdrawn":
            drop_week = int(rng.integers(3, 8))
            if w < drop_week:
                trend = 0
            elif w < drop_week + 2:
                trend = -base * 0.5 * (w - drop_week)
            else:
                clicks[w] = 0
                continue
        elif result == "Fail":
            if w > 5:
                trend = -base * 0.04 * (w - 5)
        elif result in ("Pass", "Distinction"):
            if 8 < w < 15:
                trend = -base * 0.1
            elif w >= 20:
                trend = base * 0.05
        clicks[w] = max(0, int(base + trend + noise))
    return clicks


def generate_student_vle(student_info, vle):
    rng  = RNG
    rows = []
    for _, stu in student_info.iterrows():
        latent     = stu["_latent_engagement"]
        result     = stu["final_result"]
        course     = stu["code_module"]
        sid        = stu["id_student"]
        weekly_cl  = _weekly_clicks_for_student(latent, result, N_WEEKS, rng)
        course_vle = vle[vle["code_module"] == course]["id_site"].values
        if len(course_vle) == 0:
            continue
        for week_idx, total in enumerate(weekly_cl):
            if total <= 0:
                continue
            n_days = rng.integers(1, 8)
            days   = rng.choice(range(week_idx * 7, (week_idx + 1) * 7),
                                 size=n_days, replace=False)
            day_cl = rng.multinomial(int(total), np.ones(n_days) / n_days)
            for day, dc in zip(days, day_cl):
                if dc <= 0:
                    continue
                n_acts = rng.integers(1, min(4, len(course_vle)) + 1)
                acts   = rng.choice(course_vle, size=n_acts, replace=False)
                act_cl = rng.multinomial(dc, np.ones(n_acts) / n_acts)
                for aid, ac in zip(acts, act_cl):
                    if ac > 0:
                        rows.append({
                            "id_student":        sid,
                            "code_module":       course,
                            "code_presentation": stu["code_presentation"],
                            "id_site":           aid,
                            "date":              int(day),
                            "sum_click":         int(ac),
                        })
    return pd.DataFrame(rows)


def generate_assessments(courses=COURSES):
    rows = []
    aid  = 1
    for course in courses:
        for atype, due in [("TMA", 30), ("TMA", 60), ("TMA", 90), ("Exam", 240)]:
            rows.append({
                "id_assessment":     aid,
                "code_module":       course,
                "code_presentation": "2014B",
                "assessment_type":   atype,
                "date":              due,
                "weight":            25 if atype == "TMA" else 100,
            })
            aid += 1
    return pd.DataFrame(rows)


def generate_student_assessment(student_info, assessments):
    rng  = RNG
    rows = []
    for _, stu in student_info.iterrows():
        latent = stu["_latent_engagement"]
        result = stu["final_result"]
        sid    = stu["id_student"]
        course = stu["code_module"]
        for _, asmnt in assessments[assessments["code_module"] == course].iterrows():
            if result == "Withdrawn":
                due_week = asmnt["date"] // 7
                if due_week > 8 and rng.random() > 0.2:
                    continue
                if due_week > 4 and rng.random() > 0.5:
                    continue
            score = np.clip(rng.normal(latent * 60 + 30, 12), 0, 100)
            delay = rng.integers(-3, 2) if latent > 0.6 else rng.integers(-1, 8)
            rows.append({
                "id_assessment":  asmnt["id_assessment"],
                "id_student":     sid,
                "date_submitted": int(asmnt["date"] + delay),
                "is_banked":      0,
                "score":          round(float(score), 1),
            })
    return pd.DataFrame(rows)


def generate_courses(courses=COURSES):
    return pd.DataFrame({
        "code_module":                courses,
        "code_presentation":          "2014B",
        "module_presentation_length": [270] * len(courses),
    })


def generate_and_save_all(data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    student_info = generate_student_info()
    student_info.drop(columns=["_latent_engagement"]).to_csv(
        os.path.join(data_dir, "studentInfo.csv"), index=False)

    reg = generate_student_registration(student_info)
    reg.to_csv(os.path.join(data_dir, "studentRegistration.csv"), index=False)

    vle = generate_vle()
    vle.to_csv(os.path.join(data_dir, "vle.csv"), index=False)

    student_vle = generate_student_vle(student_info, vle)
    student_vle.to_csv(os.path.join(data_dir, "studentVle.csv"), index=False)

    assessments = generate_assessments()
    assessments.to_csv(os.path.join(data_dir, "assessments.csv"), index=False)

    student_assessment = generate_student_assessment(student_info, assessments)
    student_assessment.to_csv(
        os.path.join(data_dir, "studentAssessment.csv"), index=False)

    courses_df = generate_courses()
    courses_df.to_csv(os.path.join(data_dir, "courses.csv"), index=False)

    return {
        "student_info":         student_info,
        "student_registration": reg,
        "vle":                  vle,
        "student_vle":          student_vle,
        "assessments":          assessments,
        "student_assessment":   student_assessment,
        "courses":              courses_df,
    }


if __name__ == "__main__":
    generate_and_save_all()
