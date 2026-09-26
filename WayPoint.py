from pathlib import Path
from urllib.request import urlopen, Request
import json
import os
import shutil
import tempfile
import itertools

import joblib
import pandas as pd

# SHAP is used for model-based explainability.
try:
    import shap
except ImportError:
    shap = None


# ============================================================
# WAYPOINT - STUDENT APPLICATION
# Explainable Student Academic Risk + Dream Score Optimizer
# ============================================================

PROJECT_NAME = "WayPoint"

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models" / "current"

SCORE_MODEL_PATH = MODEL_DIR / "score_model.pkl"
RISK_MODEL_PATH = MODEL_DIR / "risk_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

REGISTRY_PATH = BASE_DIR / "project_registry.json"


# ============================================================
# OPTIONAL GITHUB FALLBACK
# ============================================================

GITHUB_OWNER = os.getenv(
    "WAYPOINT_GITHUB_OWNER",
    "YOUR_GITHUB_USERNAME"
)

GITHUB_REPO = os.getenv(
    "WAYPOINT_GITHUB_REPO",
    "WayPoint"
)

GITHUB_BRANCH = os.getenv(
    "WAYPOINT_GITHUB_BRANCH",
    "main"
)


def github_raw_url(relative_path):
    relative_path = relative_path.replace("\\", "/")

    return (
        f"https://raw.githubusercontent.com/"
        f"{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"
        f"{relative_path}"
    )


def download_from_github(relative_path, destination):
    if GITHUB_OWNER == "YOUR_GITHUB_USERNAME":
        return False

    url = github_raw_url(relative_path)

    try:
        request = Request(
            url,
            headers={"User-Agent": "WayPoint-Student-App"}
        )

        with urlopen(request, timeout=20) as response:
            data = response.read()

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=destination.parent,
            suffix=".tmp"
        ) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)

        shutil.move(
            str(temporary_path),
            str(destination)
        )

        return True

    except Exception as error:
        print(
            f"\nGitHub fallback could not retrieve "
            f"{relative_path}: {error}"
        )

        return False


def ensure_models_available():
    required_files = [
        (
            "models/current/score_model.pkl",
            SCORE_MODEL_PATH
        ),
        (
            "models/current/risk_model.pkl",
            RISK_MODEL_PATH
        ),
        (
            "models/current/metadata.json",
            METADATA_PATH
        )
    ]

    missing = [
        item
        for item in required_files
        if not item[1].exists()
    ]

    if not missing:
        return True

    print(
        "\nSome WayPoint model files are missing locally."
    )

    if GITHUB_OWNER == "YOUR_GITHUB_USERNAME":
        print(
            "GitHub fallback is not configured."
        )
        print(
            "Clone the complete WayPoint repository, or set:"
        )
        print(
            "WAYPOINT_GITHUB_OWNER"
        )
        print(
            "WAYPOINT_GITHUB_REPO"
        )
        print(
            "WAYPOINT_GITHUB_BRANCH"
        )
        return False

    print(
        "\nTrying to restore the missing files from GitHub..."
    )

    success = True

    for relative_path, destination in missing:
        if not download_from_github(
            relative_path,
            destination
        ):
            success = False

    return success


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "study_hours",
    "phone_hours",
    "days_before_exam",
    "assignment_percentage",
    "attendance_percentage",
    "previous_marks"
]


RISK_ORDER = [
    "Very High Risk",
    "High Risk",
    "Moderate Risk",
    "Low Risk",
    "Very Low Risk"
]


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not ensure_models_available():

        print(
            "\nERROR: Required model files could not be found."
        )

        print(
            "\nExpected location:"
        )

        print(MODEL_DIR)

        print(
            "\nRun training.py first or clone the complete "
            "WayPoint repository."
        )

        return None, None, None

    try:

        score_model = joblib.load(
            SCORE_MODEL_PATH
        )

        risk_model = joblib.load(
            RISK_MODEL_PATH
        )

        metadata = {}

        if METADATA_PATH.exists():

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(file)

        return score_model, risk_model, metadata

    except Exception as error:

        print(
            "\nERROR while loading WayPoint models."
        )

        print(error)

        return None, None, None


# ============================================================
# INPUT FUNCTION
# ============================================================

def get_float_input(
    prompt,
    minimum=None,
    maximum=None
):

    while True:

        try:

            value = float(input(prompt))

            if minimum is not None and value < minimum:

                print(
                    f"Please enter a value of at least {minimum}."
                )

                continue

            if maximum is not None and value > maximum:

                print(
                    f"Please enter a value of at most {maximum}."
                )

                continue

            return value

        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# GET STUDENT DATA
# ============================================================

def get_student_inputs():

    print("\n" + "=" * 70)

    print(
        "ENTER STUDENT INFORMATION"
    )

    print("=" * 70)

    print(
        "\nPlease enter the following six values.\n"
    )

    study_hours = get_float_input(
        "1. Study hours per day (0.25 - 8): ",
        0.25,
        8
    )

    phone_hours = get_float_input(
        "2. Phone usage hours per day (0 - 10): ",
        0,
        10
    )

    days_before_exam = get_float_input(
        "3. Days before exam preparation begins (1 - 30): ",
        1,
        30
    )

    assignment_percentage = get_float_input(
        "4. Assignment submission percentage (20 - 100): ",
        20,
        100
    )

    attendance_percentage = get_float_input(
        "5. Attendance percentage (40 - 100): ",
        40,
        100
    )

    previous_marks = get_float_input(
        "6. Previous marks percentage (20 - 100): ",
        20,
        100
    )

    return {
        "study_hours": study_hours,
        "phone_hours": phone_hours,
        "days_before_exam": days_before_exam,
        "assignment_percentage": assignment_percentage,
        "attendance_percentage": attendance_percentage,
        "previous_marks": previous_marks
    }


# ============================================================
# CREATE PREDICTION DATAFRAME
# ============================================================

def create_input_dataframe(student_data):

    data = {
        feature: [student_data[feature]]
        for feature in FEATURES
    }

    return pd.DataFrame(
        data,
        columns=FEATURES
    )


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def _normalise_shap_values(shap_values):

    values = shap_values

    if hasattr(values, "values"):
        values = values.values

    if isinstance(values, list):
        values = values[0]

    values = values[0] if getattr(values, "ndim", 0) > 1 else values

    return values


def _feature_display_name(feature):

    names = {
        "study_hours": "daily study time",
        "phone_hours": "phone usage",
        "days_before_exam": "exam preparation timing",
        "assignment_percentage": "assignment submission",
        "attendance_percentage": "attendance",
        "previous_marks": "previous academic performance"
    }

    return names.get(
        feature,
        feature.replace("_", " ")
    )


def _format_feature_value(feature, value):

    if feature in {
        "assignment_percentage",
        "attendance_percentage",
        "previous_marks"
    }:
        return f"{value:.1f}%"

    if feature == "days_before_exam":
        return f"{value:.0f} days"

    return f"{value:.1f} hours"


def _score_direction_text(shap_value):

    if shap_value > 0:
        return "contributed upward to the predicted score"

    if shap_value < 0:
        return "contributed downward to the predicted score"

    return "had very little influence on the predicted score"


def _risk_direction_text(shap_value, predicted_risk):

    if shap_value > 0:
        return f"supported the {predicted_risk} classification"

    if shap_value < 0:
        return f"worked against the {predicted_risk} classification"

    return f"had very little influence on the {predicted_risk} classification"


def _relative_strength(impact, strongest_impact):

    strongest = abs(float(strongest_impact))
    current = abs(float(impact))

    if strongest == 0:
        return "limited"

    ratio = current / strongest

    if ratio >= 0.70:
        return "strong"

    if ratio >= 0.35:
        return "noticeable"

    return "smaller"


def _build_score_summary(feature_impacts, student):

    meaningful = [
        (feature, float(impact))
        for feature, impact in feature_impacts
        if abs(float(impact)) > 1e-9
    ]

    if not meaningful:
        return (
            "The model did not find any strong factors to highlight "
            "for this score."
        )

    top = meaningful[:3]
    descriptions = []

    for feature, impact in top:
        name = _feature_display_name(feature)
        value = _format_feature_value(feature, float(student[feature]))

        if impact > 0:
            direction = "pushed the predicted score upward"
        elif impact < 0:
            direction = "pulled the predicted score downward"
        else:
            direction = "had very little influence on the predicted score"

        descriptions.append(
            f"{name} ({value}) {direction}"
        )

    if len(descriptions) == 1:
        joined = descriptions[0]
    elif len(descriptions) == 2:
        joined = descriptions[0] + " and " + descriptions[1]
    else:
        joined = (
            descriptions[0] + ", " + descriptions[1] + ", and " + descriptions[2]
        )

    return (
        f"The strongest influences on your predicted score were {joined}. "
        "These factors explain how the values you entered shaped this "
        "particular prediction."
    )


def _build_risk_summary(risk_impacts, student, predicted_risk):

    meaningful = [
        (feature, float(impact))
        for feature, impact in risk_impacts
        if abs(float(impact)) > 1e-9
    ]

    if not meaningful:
        return (
            f"The model did not find any strong factors to highlight "
            f"for your {predicted_risk} result."
        )

    top = meaningful[:3]
    descriptions = []

    for feature, impact in top:
        name = _feature_display_name(feature)
        value = _format_feature_value(feature, float(student[feature]))

        if impact > 0:
            direction = f"supported the {predicted_risk} result"
        elif impact < 0:
            direction = f"pulled the prediction away from {predicted_risk}"
        else:
            direction = f"had very little influence on the {predicted_risk} result"

        descriptions.append(
            f"{name} ({value}) {direction}"
        )

    if len(descriptions) == 1:
        joined = descriptions[0]
    elif len(descriptions) == 2:
        joined = descriptions[0] + " and " + descriptions[1]
    else:
        joined = (
            descriptions[0] + ", " + descriptions[1] + ", and " + descriptions[2]
        )

    return (
        f"The strongest factors behind your {predicted_risk} result were {joined}. "
        "These factors explain how the values you entered shaped this "
        "particular prediction."
    )


def generate_shap_explanation(
    score_model,
    X,
    student
):

    print("\n" + "=" * 70)

    print(
        "MODEL-BASED SHAP EXPLAINABILITY"
    )

    print("=" * 70)

    if shap is None:

        print(
            "\nSHAP is not installed."
        )

        print(
            "Install the packages from requirements.txt "
            "to enable model-based explanations."
        )

        return

    score_impacts = []

    # --------------------------------------------------------
    # SCORE MODEL SHAP EXPLANATION
    # --------------------------------------------------------

    try:

        explainer = shap.TreeExplainer(
            score_model
        )

        shap_values = explainer(
            X,
            check_additivity=False
        )

        values = _normalise_shap_values(
            shap_values
        )

        score_impacts = sorted(
            zip(
                FEATURES,
                values
            ),
            key=lambda item: abs(float(item[1])),
            reverse=True
        )

        print("\nWhy this score was predicted:\n")

        print(
            _build_score_summary(
                score_impacts,
                student
            )
        )

    except Exception as error:

        print(
            "\nSHAP score explanation could not be generated:"
        )

        print(error)


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    student,
    risk
):

    recommendations = []

    if student["study_hours"] < 2:

        target_study = min(
            student["study_hours"] + 1.5,
            4.5
        )

        recommendations.append(
            f"Increase focused study time gradually "
            f"to around {target_study:.1f} hours per day."
        )

    elif student["study_hours"] < 4:

        target_study = min(
            student["study_hours"] + 1.0,
            5.0
        )

        recommendations.append(
            f"Try increasing focused study time to "
            f"around {target_study:.1f} hours per day."
        )

    else:

        recommendations.append(
            "Maintain your current study duration "
            "while focusing on study quality."
        )

    if student["phone_hours"] > 6:

        target_phone = max(
            student["phone_hours"] - 2.0,
            2.0
        )

        recommendations.append(
            f"Reduce recreational phone usage gradually "
            f"toward about {target_phone:.1f} hours per day."
        )

    elif student["phone_hours"] > 4:

        target_phone = max(
            student["phone_hours"] - 1.0,
            2.0
        )

        recommendations.append(
            f"Try reducing recreational phone usage "
            f"toward about {target_phone:.1f} hours per day."
        )

    else:

        recommendations.append(
            "Keep phone usage controlled during "
            "dedicated study sessions."
        )

    if student["days_before_exam"] <= 3:

        recommendations.append(
            "Start exam preparation earlier instead "
            "of waiting until the final few days."
        )

    elif student["days_before_exam"] <= 7:

        recommendations.append(
            "Try beginning structured preparation "
            "at least one to two weeks before exams."
        )

    else:

        recommendations.append(
            "Continue using your early preparation "
            "advantage with regular revision."
        )

    if student["assignment_percentage"] < 70:

        recommendations.append(
            "Aim to submit at least 80–90% of "
            "assignments on time."
        )

    elif student["assignment_percentage"] < 85:

        recommendations.append(
            "Try to move assignment completion "
            "closer to 90% or above."
        )

    else:

        recommendations.append(
            "Maintain your strong assignment "
            "submission consistency."
        )

    if student["attendance_percentage"] < 70:

        recommendations.append(
            "Improve class attendance wherever possible "
            "and avoid unnecessary absences."
        )

    elif student["attendance_percentage"] < 85:

        recommendations.append(
            "Try to maintain attendance above 85%."
        )

    else:

        recommendations.append(
            "Maintain your current attendance level."
        )

    if student["previous_marks"] < 50:

        recommendations.append(
            "Spend additional time revising fundamentals "
            "and topics where previous marks were weak."
        )

    elif student["previous_marks"] < 70:

        recommendations.append(
            "Focus on strengthening weaker subjects "
            "and practicing more exam-style questions."
        )

    else:

        recommendations.append(
            "Use your previous academic performance "
            "as a foundation while targeting further improvement."
        )

    return recommendations


# ============================================================
# DREAM SCORE
# ============================================================

def get_dream_score():

    print(
        "\\nEnter the score you would like to target."
    )

    return get_float_input(
        "Dream Score (0 - 100): ",
        0,
        100
    )


def _dream_score_shap_features(
    score_model,
    X,
    student,
    max_features=3
):

    """Return the most influential features according to SHAP."""

    if shap is None:
        return FEATURES[:max_features]

    try:
        explainer = shap.TreeExplainer(score_model)
        shap_values = explainer(
            X,
            check_additivity=False
        )

        values = _normalise_shap_values(shap_values)

        ranked = sorted(
            zip(FEATURES, values),
            key=lambda item: abs(float(item[1])),
            reverse=True
        )

        return [
            feature
            for feature, impact in ranked[:max_features]
            if abs(float(impact)) > 1e-9
        ] or FEATURES[:max_features]

    except Exception:
        return FEATURES[:max_features]


def _dream_candidate_values(feature, current_value):

    """Create realistic candidate values around the student's current value."""

    limits = {
        "study_hours": (0.25, 8.0, 0.5),
        "phone_hours": (0.0, 10.0, 0.5),
        "days_before_exam": (1.0, 30.0, 2.0),
        "assignment_percentage": (20.0, 100.0, 5.0),
        "attendance_percentage": (40.0, 100.0, 5.0),
        "previous_marks": (20.0, 100.0, 5.0)
    }

    minimum, maximum, step = limits[feature]
    current_value = float(current_value)

    # Test both directions. The trained model decides which direction
    # actually helps; no hardcoded "good/bad" rule is used here.
    candidates = {
        current_value,
        max(minimum, min(maximum, current_value - step)),
        max(minimum, min(maximum, current_value + step)),
        max(minimum, min(maximum, current_value - 2 * step)),
        max(minimum, min(maximum, current_value + 2 * step))
    }

    return sorted(candidates)


def _dream_candidate_dataframe(student, changes):

    candidate = dict(student)
    candidate.update(changes)

    return create_input_dataframe(candidate), candidate


def _dream_change_amount(feature, current_value, target_value):

    limits = {
        "study_hours": 8.0,
        "phone_hours": 10.0,
        "days_before_exam": 30.0,
        "assignment_percentage": 100.0,
        "attendance_percentage": 100.0,
        "previous_marks": 100.0
    }

    return abs(float(target_value) - float(current_value)) / limits[feature]


def create_dream_score_plan(
    student,
    current_score,
    dream_score,
    score_model,
    risk_model
):

    print("\\n" + "=" * 70)

    print(
        "DREAM SCORE PATH"
    )

    print("=" * 70)

    print(
        f"\\nCurrent predicted score : {current_score:.2f}"
    )

    print(
        f"Dream Score             : {dream_score:.2f}"
    )

    gap = dream_score - current_score

    if gap <= 0:

        print(
            "\\nYour current predicted score already "
            "reaches your Dream Score."
        )

        print(
            "The trained model does not identify a higher target "
            "as necessary from the information provided."
        )

        return

    X = create_input_dataframe(student)

    # SHAP decides which features are most influential for this student.
    influential_features = _dream_score_shap_features(
        score_model,
        X,
        student,
        max_features=3
    )

    # The ML score model evaluates every realistic candidate combination.
    # No fixed threshold determines whether a feature should be changed.
    candidate_lists = [
        _dream_candidate_values(
            feature,
            student[feature]
        )
        for feature in influential_features
    ]

    candidates = []

    for values in itertools.product(*candidate_lists):

        changes = dict(
            zip(influential_features, values)
        )

        candidate_X, candidate_student = _dream_candidate_dataframe(
            student,
            changes
        )

        try:
            candidate_score = float(
                score_model.predict(candidate_X)[0]
            )

            candidate_score = max(
                0.0,
                min(100.0, candidate_score)
            )

            candidate_risk = str(
                risk_model.predict(candidate_X)[0]
            ).strip()

        except Exception:
            continue

        total_change = sum(
            _dream_change_amount(
                feature,
                student[feature],
                candidate_student[feature]
            )
            for feature in influential_features
        )

        reaches_target = candidate_score >= dream_score
        improvement = candidate_score - current_score

        candidates.append({
            "score": candidate_score,
            "risk": candidate_risk,
            "student": candidate_student,
            "change": total_change,
            "reaches_target": reaches_target,
            "improvement": improvement
        })

    if not candidates:

        print(
            "\\nThe trained model could not generate a Dream Score path "
            "from the available candidate scenarios."
        )

        return

    target_candidates = [
        item
        for item in candidates
        if item["reaches_target"]
    ]

    if target_candidates:

        # Among scenarios that reach the target, prefer the smallest
        # overall change and then the score closest to the target.
        best = min(
            target_candidates,
            key=lambda item: (
                item["change"],
                abs(item["score"] - dream_score)
            )
        )

        reached_target = True

    else:

        # If the requested score is not reached within the model's
        # realistic search range, show the highest model-predicted score.
        best = max(
            candidates,
            key=lambda item: (
                item["score"],
                -item["change"]
            )
        )

        reached_target = False

    print(
        f"\\nSHAP identified the main factors influencing the current "
        f"score as: {', '.join(_feature_display_name(f) for f in influential_features)}."
    )

    print(
        "\\nThe trained ML model was then used to test realistic "
        "changes to those factors."
    )

    if reached_target:

        print(
            f"\\nA model-tested scenario reaches approximately "
            f"{best['score']:.2f} / 100."
        )

    else:

        print(
            f"\\nThe requested Dream Score was not reached within "
            f"the tested realistic range. The highest model-predicted "
            f"scenario was {best['score']:.2f} / 100."
        )

    print(
        "\\nModel-tested changes:"
    )

    for feature in influential_features:

        current_value = float(student[feature])
        target_value = float(best["student"][feature])

        if abs(target_value - current_value) < 1e-9:
            continue

        name = _feature_display_name(feature)
        current_display = _format_feature_value(
            feature,
            current_value
        )
        target_display = _format_feature_value(
            feature,
            target_value
        )

        print(
            f"   {name}: {current_display} -> {target_display}"
        )

    unchanged = all(
        abs(float(best["student"][feature]) - float(student[feature])) < 1e-9
        for feature in influential_features
    )

    if unchanged:

        print(
            "   No change to the tested influential factors produced "
            "a better model-predicted result within the search range."
        )

    print(
        f"\\nModel-predicted score for this scenario : "
        f"{best['score']:.2f} / 100"
    )

    print(
        f"Predicted risk for this scenario       : "
        f"{best['risk']}"
    )

    print(
        "\\nThis is a model-based scenario, not a guarantee that "
        "changing these factors will produce the predicted score."
    )

# ============================================================
# MODEL INFORMATION
# ============================================================

def display_model_information(metadata):

    if not metadata:
        return

    print("\n" + "=" * 70)

    print(
        "MODEL INFORMATION"
    )

    print("=" * 70)

    print(
        f"\nProject                 : "
        f"{metadata.get('project', PROJECT_NAME)}"
    )

    print(
        f"Model version           : "
        f"{metadata.get('version', 'Unknown')}"
    )

    print(
        f"Score model             : "
        f"{metadata.get('best_score_model', 'Unknown')}"
    )

    print(
        f"Risk model              : "
        f"{metadata.get('best_risk_model', 'Unknown')}"
    )

    print(
        f"Training samples        : "
        f"{metadata.get('total_training_samples', 'Unknown')}"
    )

    print(
        f"Testing samples         : "
        f"{metadata.get('total_testing_samples', 'Unknown')}"
    )


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction(
    score_model,
    risk_model,
    metadata
):

    student = get_student_inputs()

    X = create_input_dataframe(
        student
    )

    try:

        predicted_score = float(
            score_model.predict(X)[0]
        )

        predicted_risk = str(
            risk_model.predict(X)[0]
        ).strip()

    except Exception as error:

        print(
            "\nERROR while making prediction."
        )

        print(error)

        return

    predicted_score = max(
        0,
        min(
            100,
            predicted_score
        )
    )

    print("\n" + "=" * 70)

    print(
        "WAYPOINT - PREDICTION RESULT"
    )

    print("=" * 70)

    print(
        f"\nPredicted Final Score : "
        f"{predicted_score:.2f} / 100"
    )

    print(
        f"Predicted Risk Level  : "
        f"{predicted_risk}"
    )

    generate_shap_explanation(
        score_model,
        X,
        student
    )

    print("\n" + "=" * 70)

    print(
        "PERSONALIZED RECOMMENDATIONS"
    )

    print("=" * 70)

    recommendations = generate_recommendations(
        student,
        predicted_risk
    )

    for recommendation in recommendations:

        print(
            f"\n• {recommendation}"
        )

    if predicted_risk in {
        "Very High Risk",
        "High Risk",
        "Moderate Risk"
    }:

        while True:

            answer = input(
                "\nWould you like to enter your Dream Score? "
                "(yes/no): "
            ).strip().lower()

            if answer in {"yes", "y"}:

                dream_score = get_dream_score()

                create_dream_score_plan(
                    student,
                    predicted_score,
                    dream_score,
                    score_model,
                    risk_model
                )

                break

            if answer in {"no", "n"}:

                print(
                    "\nDream Score planning skipped."
                )

                break

            print(
                "Please enter yes or no."
            )

    else:

        print(
            "\nDream Score planning is available for "
            "higher-risk predictions."
        )

    display_model_information(
        metadata
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "                         WAYPOINT"
    )

    print(
        "       Explainable Student Academic Risk System"
    )

    print("=" * 70)

    score_model, risk_model, metadata = load_models()

    if (
        score_model is None
        or risk_model is None
    ):

        return

    print(
        "\nModels loaded successfully."
    )

    print(
        f"Model version: {metadata.get('version', 'Unknown')}"
    )

    print(
        "Using the current trained model for predictions."
    )

    print(
        "The WayPoint application is ready to make predictions."
    )

    while True:

        run_prediction(
            score_model,
            risk_model,
            metadata
        )

        print("\n" + "=" * 70)

        choice = input(
            "\nWould you like to test another student? "
            "(yes/no): "
        ).strip().lower()

        if choice not in {"yes", "y"}:

            print(
                "\nThank you for using WayPoint."
            )

            print(
                "Good luck with your studies!\n"
            )

            break


if __name__ == "__main__":
    main()
