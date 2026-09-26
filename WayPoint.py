from pathlib import Path
from urllib.request import urlopen, Request
import json
import os
import shutil
import tempfile

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


def _score_direction_text(feature, shap_value):

    if shap_value > 0:
        return "pushed the predicted score upward"

    if shap_value < 0:
        return "pushed the predicted score downward"

    return "had almost no effect on the predicted score"


def _risk_direction_text(shap_value, predicted_risk):

    if shap_value > 0:
        return (
            f"pushed the risk model toward the predicted "
            f"{predicted_risk} class"
        )

    if shap_value < 0:
        return (
            f"pushed the risk model away from the predicted "
            f"{predicted_risk} class"
        )

    return (
        f"had almost no effect on the predicted "
        f"{predicted_risk} class"
    )


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
            "The score model found no meaningful feature-level "
            "SHAP influence for this prediction."
        )

    top = meaningful[:3]
    strongest = abs(top[0][1])

    descriptions = []

    for feature, impact in top:
        name = _feature_display_name(feature)
        value = _format_feature_value(
            feature,
            float(student[feature])
        )
        strength = _relative_strength(
            impact,
            strongest
        )
        direction = _score_direction_text(
            feature,
            impact
        )

        descriptions.append(
            f"{name} ({value}) had a {strength} influence and "
            f"{direction}"
        )

    if len(descriptions) == 1:
        return descriptions[0] + "."

    if len(descriptions) == 2:
        joined = descriptions[0] + " and " + descriptions[1]
    else:
        joined = (
            descriptions[0]
            + ", "
            + descriptions[1]
            + ", and "
            + descriptions[2]
        )

    return (
        f"The score prediction was driven mainly by {joined}. "
        "The direction describes how each feature affected this "
        "specific prediction relative to the model's baseline; "
        "it does not mean that increasing or decreasing the feature "
        "would necessarily improve the score."
    )


def _build_risk_summary(risk_impacts, student, predicted_risk):

    meaningful = [
        (feature, float(impact))
        for feature, impact in risk_impacts
        if abs(float(impact)) > 1e-9
    ]

    if not meaningful:
        return (
            "The risk model found no meaningful feature-level "
            "SHAP influence for this prediction."
        )

    top = meaningful[:3]
    strongest = abs(top[0][1])

    descriptions = []

    for feature, impact in top:
        name = _feature_display_name(feature)
        value = _format_feature_value(
            feature,
            float(student[feature])
        )
        strength = _relative_strength(
            impact,
            strongest
        )
        direction = _risk_direction_text(
            impact,
            predicted_risk
        )

        descriptions.append(
            f"{name} ({value}) had a {strength} influence and "
            f"{direction}"
        )

    if len(descriptions) == 1:
        joined = descriptions[0]
    elif len(descriptions) == 2:
        joined = descriptions[0] + " and " + descriptions[1]
    else:
        joined = (
            descriptions[0]
            + ", "
            + descriptions[1]
            + ", and "
            + descriptions[2]
        )

    return (
        f"For the predicted {predicted_risk} risk level, the strongest "
        f"model influences were {joined}. "
        "A positive SHAP direction means the feature pushed the model "
        "toward the selected risk class, while a negative direction "
        "means it pushed the model away from that class."
    )


def generate_shap_explanation(
    score_model,
    risk_model,
    X,
    student
):

    print("\n" + "=" * 70)

    print(
        "MODEL-BASED EXPLAINABILITY (SHAP)"
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
    # SCORE MODEL
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

    # --------------------------------------------------------
    # RISK MODEL
    # --------------------------------------------------------

    try:

        risk_explainer = shap.TreeExplainer(
            risk_model
        )

        risk_values = risk_explainer(
            X,
            check_additivity=False
        )

        raw_values = risk_values.values

        # Multiclass TreeExplainer can return:
        # (samples, features, classes)
        if getattr(raw_values, "ndim", 0) == 3:

            class_names = list(
                getattr(
                    risk_model,
                    "classes_",
                    []
                )
            )

            predicted_risk = str(
                risk_model.predict(X)[0]
            )

            if predicted_risk in class_names:

                class_index = class_names.index(
                    predicted_risk
                )

            else:

                class_index = 0

            selected_values = raw_values[
                0,
                :,
                class_index
            ]

        else:

            selected_values = _normalise_shap_values(
                risk_values
            )

            predicted_risk = str(
                risk_model.predict(X)[0]
            )

        risk_impacts = sorted(
            zip(
                FEATURES,
                selected_values
            ),
            key=lambda item: abs(float(item[1])),
            reverse=True
        )

        print("\nWhy this risk level was predicted:\n")

        print(
            _build_risk_summary(
                risk_impacts,
                student,
                predicted_risk
            )
        )

    except Exception as error:

        print(
            "\nSHAP risk explanation could not be generated:"
        )

        print(error)


# ============================================================
# DREAM SCORE INPUT
# ============================================================

def get_dream_score():

    print("\n" + "=" * 70)

    print(
        "DREAM SCORE"
    )

    print("=" * 70)

    print(
        "\nEnter the score you would like to target."
    )

    return get_float_input(
        "Dream Score (0 - 100): ",
        0,
        100
    )


# ============================================================
# DREAM SCORE PLAN
# ============================================================

def create_dream_score_plan(
    student,
    current_score,
    dream_score
):

    gap = dream_score - current_score

    print("\n" + "=" * 70)

    print(
        "DREAM SCORE PATH"
    )

    print("=" * 70)

    print(
        f"\nCurrent predicted score : "
        f"{current_score:.2f}"
    )

    print(
        f"Dream Score             : "
        f"{dream_score:.2f}"
    )

    if gap <= 0:

        print(
            "\nYour current predicted score already "
            "reaches your Dream Score."
        )

        print(
            "\nFocus on maintaining your current habits "
            "and avoiding a decline."
        )

        return

    print(
        f"\nScore improvement needed : "
        f"approximately {gap:.2f} points"
    )

    print(
        "\nRecommended areas to focus on:\n"
    )

    print(
        "1. Study time"
    )

    study_increase = min(
        max(0.5, gap / 12),
        2.0
    )

    target_study = min(
        student["study_hours"] + study_increase,
        6.0
    )

    print(
        f"   Current : "
        f"{student['study_hours']:.1f} hours/day"
    )

    print(
        f"   Target  : "
        f"approximately {target_study:.1f} hours/day"
    )

    print(
        f"   Increase: approximately "
        f"{max(0, target_study - student['study_hours']):.1f} "
        f"hours/day"
    )

    print(
        "\n2. Phone usage"
    )

    if student["phone_hours"] > 3:

        phone_reduction = min(
            max(0.5, gap / 15),
            2.0
        )

        target_phone = max(
            student["phone_hours"] - phone_reduction,
            2.0
        )

        print(
            f"   Current : "
            f"{student['phone_hours']:.1f} hours/day"
        )

        print(
            f"   Target  : "
            f"approximately {target_phone:.1f} hours/day"
        )

        print(
            f"   Reduction: approximately "
            f"{student['phone_hours'] - target_phone:.1f} "
            f"hours/day"
        )

    else:

        print(
            "   Phone usage is already relatively controlled."
        )

    print(
        "\n3. Exam preparation"
    )

    if student["days_before_exam"] < 14:

        target_days = min(
            max(
                student["days_before_exam"] + 4,
                10
            ),
            21
        )

        print(
            f"   Current preparation starts about "
            f"{student['days_before_exam']:.0f} "
            f"days before the exam."
        )

        print(
            f"   Target  : start structured preparation "
            f"around {target_days:.0f} days before the exam."
        )

    else:

        print(
            "   Your preparation timeline is already relatively strong."
        )

    print(
        "\n4. Assignments"
    )

    if student["assignment_percentage"] < 90:

        target_assignment = min(
            student["assignment_percentage"] + 10,
            95
        )

        print(
            f"   Current : "
            f"{student['assignment_percentage']:.1f}%"
        )

        print(
            f"   Target  : "
            f"approximately {target_assignment:.1f}%"
        )

    else:

        print(
            "   Assignment completion is already strong."
        )

    print(
        "\n5. Attendance"
    )

    if student["attendance_percentage"] < 90:

        target_attendance = min(
            student["attendance_percentage"] + 5,
            95
        )

        print(
            f"   Current : "
            f"{student['attendance_percentage']:.1f}%"
        )

        print(
            f"   Target  : "
            f"approximately {target_attendance:.1f}%"
        )

    else:

        print(
            "   Attendance is already strong."
        )

    print(
        "\n6. Previous academic performance"
    )

    if student["previous_marks"] < 70:

        print(
            "   Focus on strengthening the subjects "
            "and concepts where previous performance was weaker."
        )

    else:

        print(
            "   Previous marks are already relatively strong."
        )

    print("\n" + "-" * 70)

    print(
        "IMPORTANT:"
    )

    print(
        "The Dream Score path is a practical improvement plan, "
        "not a guarantee of achieving the target score."
    )

    print(
        "The recommendations are intentionally kept within "
        "realistic limits."
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

    print("\n" + "=" * 70)

    print(
        "MODEL-BASED EXPLANATION"
    )

    print("=" * 70)

    generate_shap_explanation(
        score_model,
        risk_model,
        X,
        student
    )

    if predicted_risk in {
        "Very High Risk",
        "High Risk",
        "Moderate Risk"
    }:

        print("\n" + "=" * 70)

        print(
            "DREAM SCORE OPTION"
        )

        print("=" * 70)

        print(
            "\nBecause your predicted risk is in a "
            "higher-risk category,"
        )

        print(
            "you can create a personalized Dream Score "
            "improvement path."
        )

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
                    dream_score
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

        print("\n" + "=" * 70)

        print(
            "DREAM SCORE"
        )

        print("=" * 70)

        print(
            "\nYour current predicted risk is not in "
            "the higher-risk group,"
        )

        print(
            "so the intervention path is not automatically "
            "shown."
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
