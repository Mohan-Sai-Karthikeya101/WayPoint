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


def _format_shap_explanation(
    feature,
    value,
    shap_value,
    score_model=True
):

    direction = (
        "increased"
        if shap_value > 0
        else "decreased"
    )

    amount = abs(float(shap_value))

    if score_model:
        return (
            f"{feature.replace('_', ' ').title()} "
            f"({value:.2f}) {direction} the model's "
            f"predicted score by approximately "
            f"{amount:.2f} points relative to the model's "
            f"baseline."
        )

    return (
        f"{feature.replace('_', ' ').title()} "
        f"({value:.2f}) contributed to the model's "
        f"risk prediction with a SHAP impact magnitude "
        f"of approximately {amount:.3f}."
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

        feature_impacts = sorted(
            zip(
                FEATURES,
                values
            ),
            key=lambda item: abs(float(item[1])),
            reverse=True
        )

        print(
            "\nTop factors influencing the predicted score:"
        )

        for feature, impact in feature_impacts[:6]:

            print(
                "\n• "
                + _format_shap_explanation(
                    feature,
                    student[feature],
                    float(impact),
                    score_model=True
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

        risk_impacts = sorted(
            zip(
                FEATURES,
                selected_values
            ),
            key=lambda item: abs(float(item[1])),
            reverse=True
        )

        print(
            "\nTop factors influencing the predicted risk:"
        )

        for feature, impact in risk_impacts[:6]:

            print(
                "\n• "
                + _format_shap_explanation(
                    feature,
                    student[feature],
                    float(impact),
                    score_model=False
                )
            )

    except Exception as error:

        print(
            "\nSHAP risk explanation could not be generated:"
        )

        print(error)


# ============================================================
# HUMAN-READABLE CONTEXT
# ============================================================

def generate_explanation(student):

    explanations = []

    if student["study_hours"] < 2:

        explanations.append(
            "Your daily study time is relatively low."
        )

    elif student["study_hours"] < 4:

        explanations.append(
            "Your daily study time is moderate and "
            "could be increased gradually."
        )

    else:

        explanations.append(
            "Your daily study time is relatively strong."
        )

    if student["phone_hours"] > 6:

        explanations.append(
            "Your phone usage is high and may be "
            "reducing time available for focused study."
        )

    elif student["phone_hours"] > 4:

        explanations.append(
            "Your phone usage is moderately high and "
            "reducing it could create more study time."
        )

    else:

        explanations.append(
            "Your phone usage is within a relatively "
            "manageable range."
        )

    if student["days_before_exam"] <= 3:

        explanations.append(
            "You begin exam preparation quite close "
            "to the examination."
        )

    elif student["days_before_exam"] <= 7:

        explanations.append(
            "Starting preparation earlier could improve "
            "your preparation time."
        )

    else:

        explanations.append(
            "You start preparing sufficiently early "
            "before the examination."
        )

    if student["assignment_percentage"] < 60:

        explanations.append(
            "Assignment completion is relatively low "
            "and should be improved."
        )

    elif student["assignment_percentage"] < 80:

        explanations.append(
            "Assignment completion is reasonable but "
            "has room for improvement."
        )

    else:

        explanations.append(
            "Assignment completion is strong."
        )

    if student["attendance_percentage"] < 60:

        explanations.append(
            "Attendance is relatively low."
        )

    elif student["attendance_percentage"] < 80:

        explanations.append(
            "Attendance is moderate and could be improved."
        )

    else:

        explanations.append(
            "Attendance is strong."
        )

    if student["previous_marks"] < 50:

        explanations.append(
            "Previous academic performance indicates "
            "that additional academic support may be useful."
        )

    elif student["previous_marks"] < 70:

        explanations.append(
            "Previous academic performance is moderate."
        )

    else:

        explanations.append(
            "Previous academic performance is relatively strong."
        )

    return explanations


# ============================================================
# RECOMMENDATIONS
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

    print("\n" + "=" * 70)

    print(
        "STUDENT CONTEXT"
    )

    print("=" * 70)

    explanations = generate_explanation(
        student
    )

    for explanation in explanations:

        print(
            f"\n• {explanation}"
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
