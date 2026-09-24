# ============================================================
# WAYPOINT
# Explainable Student Academic Risk + Dream Score ML Pipeline
#
# FINAL TRAINING PIPELINE
#
# Every execution:
#   1. Uses the local project state.
#   2. Generates 20,000 NEW training students.
#   3. Generates 10,000 NEW testing students.
#   4. Loads ALL historical training/testing batches.
#   5. Loads the previous .pkl models when available.
#   6. Builds fresh ML candidates on all accumulated data.
#   7. Builds persistent/warm-start candidates from previous models.
#   8. Evaluates all candidates on the accumulated test data.
#   9. Selects the best score and risk models.
#  10. Saves a new immutable model version.
#  11. Updates models/current/.
#  12. Writes project_registry.json for reproducible persistence.
#
# PROJECT:
#   WayPoint
#
# REQUIRED MODEL FEATURES:
#   study_hours
#   phone_hours
#   days_before_exam
#   assignment_percentage
#   attendance_percentage
#   previous_marks
#
# NOT USED:
#   recent assessment performance
#   study consistency
#
# IMPORTANT:
#   The synthetic-data generator uses a hidden nonlinear process only
#   to create training labels. The prediction application does NOT use
#   those formulas. The ML models learn the relationships from data.
#
#   "Persistent learning" here means:
#     - historical data is retained and reused;
#     - previous trained models are restored;
#     - compatible previous models are warm-started where sklearn supports
#       it;
#     - fresh models are also trained and compared;
#     - the best candidate becomes the next model version.
#
#   RandomForest warm-start keeps previous trees and adds new trees.
#   HistGradientBoosting warm-start keeps previous boosting iterations and
#   adds more iterations.
#   Decision trees do not support genuine incremental continuation, so a
#   fresh DecisionTree is trained on the complete accumulated dataset.
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "WayPoint"

TRAINING_BATCH_SIZE = 20_000
TESTING_BATCH_SIZE = 10_000

RANDOM_SEED = 42

FEATURES = [
    "study_hours",
    "phone_hours",
    "days_before_exam",
    "assignment_percentage",
    "attendance_percentage",
    "previous_marks",
]

SCORE_TARGET = "final_score"
RISK_TARGET = "risk_category"

RISK_ORDER = [
    "Very High Risk",
    "High Risk",
    "Moderate Risk",
    "Low Risk",
    "Very Low Risk",
]

# Number of additional trees/iterations used by a warm-start run.
WARM_START_ADDITIONAL_TREES = 100
WARM_START_ADDITIONAL_ITERATIONS = 100

REGISTRY_FILENAME = "project_registry.json"


# ============================================================
# 2. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAINING_DATA_DIR = BASE_DIR / "data" / "training"
TESTING_DATA_DIR = BASE_DIR / "data" / "testing"

CURRENT_MODEL_DIR = BASE_DIR / "models" / "current"
VERSIONS_DIR = BASE_DIR / "models" / "versions"

REGISTRY_PATH = BASE_DIR / REGISTRY_FILENAME

for directory in [
    TRAINING_DATA_DIR,
    TESTING_DATA_DIR,
    CURRENT_MODEL_DIR,
    VERSIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. SMALL UTILITY FUNCTIONS
# ============================================================

def utc_now():
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def print_section(title):
    print("\n")
    print("=" * 72)
    print(title)
    print("=" * 72)


def relative_project_path(path):
    """Return a repository-style path relative to the project root."""
    return str(Path(path).resolve().relative_to(BASE_DIR.resolve())).replace(
        "\\", "/"
    )


# ============================================================
# 4. LOCAL PROJECT REGISTRY
# ============================================================

def load_registry():
    """Load project_registry.json if it exists locally."""
    if not REGISTRY_PATH.exists():
        return None

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Warning: Could not read local registry: {error}")
        return None


def collect_tracked_files():
    """Collect persistent project artifacts for the local registry."""
    files = []

    for root in [
        TRAINING_DATA_DIR,
        TESTING_DATA_DIR,
        CURRENT_MODEL_DIR,
        VERSIONS_DIR,
    ]:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.is_file():
                files.append(relative_project_path(path))

    files.sort()
    return files


def write_project_registry(version_number, batch_number, metadata):
    """Write a local reproducibility registry."""
    registry = {
        "project": PROJECT_NAME,
        "registry_version": 1,
        "updated_at": utc_now(),
        "latest_version": f"v{version_number}",
        "latest_batch": batch_number,
        "latest_metadata": metadata,
        "tracked_files": collect_tracked_files(),
    }

    temporary_path = REGISTRY_PATH.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)

    temporary_path.replace(REGISTRY_PATH)
    print(f"\nRegistry updated: {REGISTRY_PATH}")
    return registry


# ============================================================
# 5. SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_data(n_samples, seed):
    """
    Generate synthetic student data.

    The generator creates a nonlinear hidden relationship to make the
    prediction task meaningful. These equations are NOT used by WayPoint
    when a student is predicted in the application.

    Only these six features are exposed to the ML models:
      - study_hours
      - phone_hours
      - days_before_exam
      - assignment_percentage
      - attendance_percentage
      - previous_marks
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Study hours
    # --------------------------------------------------------
    study_hours = rng.normal(
        loc=3.2,
        scale=1.4,
        size=n_samples,
    )
    study_hours = np.clip(
        study_hours,
        0.25,
        8.0,
    )

    # --------------------------------------------------------
    # Phone usage
    # --------------------------------------------------------
    phone_hours = rng.normal(
        loc=3.5,
        scale=1.8,
        size=n_samples,
    )
    phone_hours = np.clip(
        phone_hours,
        0.0,
        10.0,
    )

    # --------------------------------------------------------
    # Days before exam
    # --------------------------------------------------------
    days_before_exam = rng.normal(
        loc=12,
        scale=7,
        size=n_samples,
    )
    days_before_exam = np.clip(
        days_before_exam,
        1,
        30,
    )

    # --------------------------------------------------------
    # Assignment percentage
    # --------------------------------------------------------
    assignment_percentage = rng.normal(
        loc=75,
        scale=16,
        size=n_samples,
    )
    assignment_percentage = np.clip(
        assignment_percentage,
        20,
        100,
    )

    # --------------------------------------------------------
    # Attendance percentage
    # --------------------------------------------------------
    attendance_percentage = rng.normal(
        loc=78,
        scale=12,
        size=n_samples,
    )
    attendance_percentage = np.clip(
        attendance_percentage,
        40,
        100,
    )

    # --------------------------------------------------------
    # Previous marks
    # --------------------------------------------------------
    previous_marks = rng.normal(
        loc=65,
        scale=18,
        size=n_samples,
    )
    previous_marks = np.clip(
        previous_marks,
        20,
        100,
    )

    # ========================================================
    # Hidden nonlinear data-generating relationship
    # ========================================================

    study_effect = (
        18
        * (
            1
            - np.exp(-study_hours / 3.5)
        )
    )

    phone_effect = (
        1.2 * phone_hours
        + 0.32 * (phone_hours ** 2)
    )

    exam_prep_effect = (
        8
        * (
            1
            - np.exp(-days_before_exam / 10)
        )
    )

    assignment_effect = (
        0.10
        * assignment_percentage
    )

    attendance_effect = (
        0.12
        * attendance_percentage
    )

    previous_marks_effect = (
        0.38
        * previous_marks
    )

    study_phone_interaction = (
        0.45
        * study_hours
        * np.maximum(
            0,
            5 - phone_hours,
        )
    )

    noise = rng.normal(
        loc=0,
        scale=3.5,
        size=n_samples,
    )

    final_score = (
        8
        + study_effect
        - phone_effect
        + exam_prep_effect
        + assignment_effect
        + attendance_effect
        + previous_marks_effect
        + study_phone_interaction
        + noise
    )

    final_score = np.clip(
        final_score,
        0,
        100,
    )

    # Risk labels are generated from the synthetic ground-truth score.
    # The classifier itself learns the relationship from the six features.
    risk_category = np.select(
        [
            final_score < 40,
            final_score < 50,
            final_score < 65,
            final_score < 80,
        ],
        [
            "Very High Risk",
            "High Risk",
            "Moderate Risk",
            "Low Risk",
        ],
        default="Very Low Risk",
    )

    return pd.DataFrame(
        {
            "study_hours": study_hours,
            "phone_hours": phone_hours,
            "days_before_exam": days_before_exam,
            "assignment_percentage": assignment_percentage,
            "attendance_percentage": attendance_percentage,
            "previous_marks": previous_marks,
            "final_score": final_score,
            "risk_category": risk_category,
        }
    )


# ============================================================
# 6. DATA BATCH MANAGEMENT
# ============================================================

def get_existing_training_batches():
    return sorted(
        TRAINING_DATA_DIR.glob("train_batch_*.csv")
    )


def get_existing_testing_batches():
    return sorted(
        TESTING_DATA_DIR.glob("test_batch_*.csv")
    )


def load_historical_data(file_list):
    """
    Load all valid historical batches.

    Invalid/incomplete CSV files are skipped rather than silently becoming
    training data with missing target columns.
    """
    if not file_list:
        return pd.DataFrame()

    required_columns = FEATURES + [
        SCORE_TARGET,
        RISK_TARGET,
    ]

    frames = []

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)

            if all(
                column in df.columns
                for column in required_columns
            ):
                frames.append(df)
            else:
                print(
                    f"Warning: Skipping incomplete data file: "
                    f"{file_path.name}"
                )

        except Exception as error:
            print(
                f"Warning: Could not load "
                f"{file_path.name}: {error}"
            )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # Ensure numeric model inputs are actually numeric.
    for column in FEATURES + [SCORE_TARGET]:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

    combined = combined.dropna(
        subset=FEATURES + [SCORE_TARGET, RISK_TARGET]
    )

    return combined


def get_next_batch_number():
    files = get_existing_training_batches()

    if not files:
        return 1

    numbers = []

    for file_path in files:
        match = re.search(
            r"train_batch_(\d+)$",
            file_path.stem,
        )

        if match:
            numbers.append(
                int(match.group(1))
            )

    return max(numbers, default=0) + 1


def get_existing_version_numbers():
    numbers = []

    if not VERSIONS_DIR.exists():
        return numbers

    for path in VERSIONS_DIR.iterdir():
        if not path.is_dir():
            continue

        match = re.fullmatch(
            r"v(\d+)",
            path.name,
        )

        if match:
            numbers.append(
                int(match.group(1))
            )

    return sorted(numbers)


def get_next_version_number():
    return max(
        get_existing_version_numbers(),
        default=0,
    ) + 1


# ============================================================
# 7. PREVIOUS MODEL LOADING
# ============================================================

def load_previous_models():
    score_model_path = (
        CURRENT_MODEL_DIR / "score_model.pkl"
    )

    risk_model_path = (
        CURRENT_MODEL_DIR / "risk_model.pkl"
    )

    metadata_path = (
        CURRENT_MODEL_DIR / "metadata.json"
    )

    previous_score_model = None
    previous_risk_model = None
    previous_metadata = None

    if score_model_path.exists():
        try:
            previous_score_model = joblib.load(
                score_model_path
            )
        except Exception as error:
            print(
                "Warning: Could not load previous "
                f"score model: {error}"
            )

    if risk_model_path.exists():
        try:
            previous_risk_model = joblib.load(
                risk_model_path
            )
        except Exception as error:
            print(
                "Warning: Could not load previous "
                f"risk model: {error}"
            )

    if metadata_path.exists():
        try:
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                previous_metadata = json.load(file)

        except Exception as error:
            print(
                "Warning: Could not load previous "
                f"metadata: {error}"
            )

    return (
        previous_score_model,
        previous_risk_model,
        previous_metadata,
    )


# ============================================================
# 8. MODEL FACTORIES
# ============================================================

def create_fresh_score_models():
    """Fresh candidates trained from scratch on all accumulated data."""
    return {
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
        ),
    }


def create_fresh_risk_models():
    """Fresh candidates trained from scratch on all accumulated data."""
    return {
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
        ),
    }


# ============================================================
# 9. PERSISTENT / WARM-START MODEL CREATION
# ============================================================

def build_warm_start_score_candidates(previous_model):
    """
    Build continuation candidates from the previous selected score model.

    RandomForestRegressor:
      - previous trees are retained;
      - 100 additional trees are added.

    HistGradientBoostingRegressor:
      - previous boosting iterations are retained;
      - 100 additional iterations are added.

    DecisionTreeRegressor:
      - no genuine warm-start support, so no continuation candidate.
    """
    candidates = {}

    if isinstance(
        previous_model,
        RandomForestRegressor,
    ):
        try:
            model = previous_model
            model.warm_start = True
            model.n_estimators = (
                int(model.n_estimators)
                + WARM_START_ADDITIONAL_TREES
            )

            candidates[
                "RandomForestRegressor_WarmStart"
            ] = model

        except Exception as error:
            print(
                "Warning: Could not prepare "
                f"RandomForestRegressor continuation: {error}"
            )

    elif isinstance(
        previous_model,
        HistGradientBoostingRegressor,
    ):
        try:
            model = previous_model
            model.warm_start = True
            model.max_iter = (
                int(model.max_iter)
                + WARM_START_ADDITIONAL_ITERATIONS
            )

            candidates[
                "HistGradientBoostingRegressor_WarmStart"
            ] = model

        except Exception as error:
            print(
                "Warning: Could not prepare "
                f"HistGradientBoostingRegressor continuation: {error}"
            )

    return candidates


def build_warm_start_risk_candidates(previous_model):
    """
    Build continuation candidates from the previous selected risk model.
    """
    candidates = {}

    if isinstance(
        previous_model,
        RandomForestClassifier,
    ):
        try:
            model = previous_model
            model.warm_start = True
            model.n_estimators = (
                int(model.n_estimators)
                + WARM_START_ADDITIONAL_TREES
            )

            candidates[
                "RandomForestClassifier_WarmStart"
            ] = model

        except Exception as error:
            print(
                "Warning: Could not prepare "
                f"RandomForestClassifier continuation: {error}"
            )

    elif isinstance(
        previous_model,
        HistGradientBoostingClassifier,
    ):
        try:
            model = previous_model
            model.warm_start = True
            model.max_iter = (
                int(model.max_iter)
                + WARM_START_ADDITIONAL_ITERATIONS
            )

            candidates[
                "HistGradientBoostingClassifier_WarmStart"
            ] = model

        except Exception as error:
            print(
                "Warning: Could not prepare "
                f"HistGradientBoostingClassifier continuation: {error}"
            )

    return candidates


# ============================================================
# 10. MODEL EVALUATION HELPERS
# ============================================================

def evaluate_score_model(
    model,
    X_test,
    y_test,
):
    predictions = model.predict(X_test)

    r2 = r2_score(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def evaluate_risk_model(
    model,
    X_test,
    y_test,
):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
    }


# ============================================================
# 11. TRAIN ALL SCORE CANDIDATES
# ============================================================

def train_score_models(
    X_train,
    y_train,
    X_test,
    y_test,
    previous_score_model=None,
):
    """
    Train both:
      - fresh candidates
      - a compatible warm-start candidate from the previous model
    """
    print_section("TRAINING SCORE MODELS")

    candidates = create_fresh_score_models()

    warm_candidates = build_warm_start_score_candidates(
        previous_score_model
    )

    candidates.update(warm_candidates)

    trained_models = {}
    results = {}

    for model_name, model in candidates.items():
        print(f"\nTraining {model_name}...")

        try:
            model.fit(
                X_train,
                y_train,
            )

            metrics = evaluate_score_model(
                model,
                X_test,
                y_test,
            )

            trained_models[model_name] = model
            results[model_name] = metrics

            print(
                f"R²   : {metrics['r2']:.4f}"
            )
            print(
                f"RMSE : {metrics['rmse']:.4f}"
            )
            print(
                f"MAE  : {metrics['mae']:.4f}"
            )

        except Exception as error:
            print(
                f"Warning: {model_name} failed: {error}"
            )

    if not trained_models:
        raise RuntimeError(
            "No score model could be trained successfully."
        )

    return trained_models, results


# ============================================================
# 12. TRAIN ALL RISK CANDIDATES
# ============================================================

def train_risk_models(
    X_train,
    y_train,
    X_test,
    y_test,
    previous_risk_model=None,
):
    """
    Train both:
      - fresh candidates
      - a compatible warm-start candidate from the previous model
    """
    print_section("TRAINING RISK MODELS")

    candidates = create_fresh_risk_models()

    warm_candidates = build_warm_start_risk_candidates(
        previous_risk_model
    )

    candidates.update(warm_candidates)

    trained_models = {}
    results = {}

    for model_name, model in candidates.items():
        print(f"\nTraining {model_name}...")

        try:
            model.fit(
                X_train,
                y_train,
            )

            metrics = evaluate_risk_model(
                model,
                X_test,
                y_test,
            )

            trained_models[model_name] = model
            results[model_name] = metrics

            print(
                f"Accuracy    : {metrics['accuracy']:.4f}"
            )
            print(
                f"Weighted F1 : {metrics['weighted_f1']:.4f}"
            )

        except Exception as error:
            print(
                f"Warning: {model_name} failed: {error}"
            )

    if not trained_models:
        raise RuntimeError(
            "No risk model could be trained successfully."
        )

    return trained_models, results


# ============================================================
# 13. MODEL SELECTION
# ============================================================

def select_best_score_model(results):
    """
    Select by combined metric rank:
      - R² higher is better
      - RMSE lower is better
      - MAE lower is better
    """
    dataframe = pd.DataFrame(results).T.copy()

    dataframe["r2_rank"] = dataframe["r2"].rank(
        ascending=False,
        method="min",
    )

    dataframe["rmse_rank"] = dataframe["rmse"].rank(
        ascending=True,
        method="min",
    )

    dataframe["mae_rank"] = dataframe["mae"].rank(
        ascending=True,
        method="min",
    )

    dataframe["combined_rank"] = dataframe[
        [
            "r2_rank",
            "rmse_rank",
            "mae_rank",
        ]
    ].mean(axis=1)

    best_model_name = (
        dataframe["combined_rank"]
        .idxmin()
    )

    return best_model_name, dataframe


def select_best_risk_model(results):
    """
    Select by combined metric rank:
      - accuracy higher is better
      - weighted F1 higher is better
    """
    dataframe = pd.DataFrame(results).T.copy()

    dataframe["accuracy_rank"] = dataframe[
        "accuracy"
    ].rank(
        ascending=False,
        method="min",
    )

    dataframe["weighted_f1_rank"] = dataframe[
        "weighted_f1"
    ].rank(
        ascending=False,
        method="min",
    )

    dataframe["combined_rank"] = dataframe[
        [
            "accuracy_rank",
            "weighted_f1_rank",
        ]
    ].mean(axis=1)

    best_model_name = (
        dataframe["combined_rank"]
        .idxmin()
    )

    return best_model_name, dataframe


# ============================================================
# 14. DATA / MODEL DIAGNOSTICS
# ============================================================

def print_dataset_summary(
    training_data,
    testing_data,
):
    print_section("DATASET SUMMARY")

    print(
        f"Training samples : {len(training_data):,}"
    )
    print(
        f"Testing samples  : {len(testing_data):,}"
    )

    print("\nTraining risk distribution:")
    print(
        training_data[RISK_TARGET]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .to_string()
    )

    print("\nTesting risk distribution:")
    print(
        testing_data[RISK_TARGET]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .to_string()
    )


def print_model_predicted_score_ranges(
    score_model,
    risk_model,
    X_test,
):
    predicted_scores = score_model.predict(
        X_test
    )

    predicted_risks = risk_model.predict(
        X_test
    )

    prediction_dataframe = pd.DataFrame(
        {
            "predicted_score": predicted_scores,
            "predicted_risk": predicted_risks,
        }
    )

    print_section(
        "MODEL-PREDICTED SCORE RANGES BY RISK CATEGORY"
    )

    for risk in RISK_ORDER:
        subset = prediction_dataframe[
            prediction_dataframe["predicted_risk"] == risk
        ]

        print(f"\n{risk}")

        if subset.empty:
            print("  Number of test students : 0")
            print("  Predicted score range   : No students")
            print("  Average predicted score : No students")
            continue

        print(
            "  Number of test students : "
            f"{len(subset):,}"
        )
        print(
            "  Predicted score range   : "
            f"{subset['predicted_score'].min():.2f} - "
            f"{subset['predicted_score'].max():.2f}"
        )
        print(
            "  Average predicted score : "
            f"{subset['predicted_score'].mean():.2f}"
        )


# ============================================================
# 15. MODEL PERSISTENCE
# ============================================================

def save_model_version(
    version_number,
    score_model,
    risk_model,
    metadata,
):
    version_name = f"v{version_number}"

    version_directory = (
        VERSIONS_DIR / version_name
    )

    version_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        score_model,
        version_directory / "score_model.pkl",
    )

    joblib.dump(
        risk_model,
        version_directory / "risk_model.pkl",
    )

    with open(
        version_directory / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"\nSaved immutable model version: {version_name}"
    )

    return version_directory


def update_current_model(version_directory):
    """
    Copy the selected version into models/current/.

    The student application always reads from models/current/.
    """
    current_files = [
        "score_model.pkl",
        "risk_model.pkl",
        "metadata.json",
    ]

    for filename in current_files:
        source = (
            version_directory / filename
        )
        destination = (
            CURRENT_MODEL_DIR / filename
        )

        shutil.copy2(
            source,
            destination,
        )

    print(
        "\nCurrent model updated:"
        f"\n{CURRENT_MODEL_DIR}"
    )


# ============================================================
# 16. MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("\n")
    print("=" * 72)
    print(
        f"{PROJECT_NAME} — ML TRAINING PIPELINE"
    )
    print("=" * 72)

    batch_number = get_next_batch_number()
    version_number = get_next_version_number()

    print(
        f"\nNew batch number  : {batch_number}"
    )
    print(
        f"New model version : v{version_number}"
    )

    # --------------------------------------------------------
    # 16.1 Generate a NEW training/testing batch.
    # --------------------------------------------------------
    print_section("GENERATING NEW SYNTHETIC DATA")

    training_seed = (
        RANDOM_SEED
        + batch_number * 100
    )

    testing_seed = (
        RANDOM_SEED
        + batch_number * 100
        + 1
    )

    print(
        f"Generating {TRAINING_BATCH_SIZE:,} "
        "training students..."
    )

    new_training_data = generate_synthetic_data(
        TRAINING_BATCH_SIZE,
        training_seed,
    )

    print(
        f"Generating {TESTING_BATCH_SIZE:,} "
        "testing students..."
    )

    new_testing_data = generate_synthetic_data(
        TESTING_BATCH_SIZE,
        testing_seed,
    )

    # --------------------------------------------------------
    # 16.2 Save the new batches.
    # --------------------------------------------------------
    training_path = (
        TRAINING_DATA_DIR
        / f"train_batch_{batch_number:03d}.csv"
    )

    testing_path = (
        TESTING_DATA_DIR
        / f"test_batch_{batch_number:03d}.csv"
    )

    new_training_data.to_csv(
        training_path,
        index=False,
    )

    new_testing_data.to_csv(
        testing_path,
        index=False,
    )

    print(
        f"\nSaved training batch: {training_path}"
    )
    print(
        f"Saved testing batch : {testing_path}"
    )

    # --------------------------------------------------------
    # 16.3 Load ALL accumulated historical data.
    # --------------------------------------------------------
    print_section("LOADING ALL HISTORICAL DATA")

    training_files = get_existing_training_batches()
    testing_files = get_existing_testing_batches()

    all_training_data = load_historical_data(
        training_files
    )

    all_testing_data = load_historical_data(
        testing_files
    )

    if all_training_data.empty:
        raise RuntimeError(
            "No valid training data was found."
        )

    if all_testing_data.empty:
        raise RuntimeError(
            "No valid testing data was found."
        )

    print_dataset_summary(
        all_training_data,
        all_testing_data,
    )

    print(
        f"\nTraining batches : {len(training_files)}"
    )
    print(
        f"Testing batches  : {len(testing_files)}"
    )

    # --------------------------------------------------------
    # 16.4 Prepare ML matrices.
    # --------------------------------------------------------
    X_train = all_training_data[FEATURES]
    y_train_score = all_training_data[SCORE_TARGET]
    y_train_risk = all_training_data[RISK_TARGET]

    X_test = all_testing_data[FEATURES]
    y_test_score = all_testing_data[SCORE_TARGET]
    y_test_risk = all_testing_data[RISK_TARGET]

    # --------------------------------------------------------
    # 16.5 Load previous selected models.
    # --------------------------------------------------------
    (
        previous_score_model,
        previous_risk_model,
        previous_metadata,
    ) = load_previous_models()

    previous_model_loaded = (
        previous_score_model is not None
        or previous_risk_model is not None
    )

    previous_version = None

    if previous_metadata:
        previous_version = previous_metadata.get(
            "version"
        )

    print_section("PREVIOUS MODEL STATUS")

    if previous_model_loaded:
        print("Previous model found.")
        print(
            f"Previous version : {previous_version}"
        )
        print(
            "Persistent model candidates will be "
            "warm-started when supported."
        )
    else:
        print(
            "No previous model found."
        )
        print(
            "This is the first training run."
        )

    # --------------------------------------------------------
    # 16.6 Train score candidates.
    # --------------------------------------------------------
    (
        trained_score_models,
        score_results,
    ) = train_score_models(
        X_train,
        y_train_score,
        X_test,
        y_test_score,
        previous_score_model,
    )

    # --------------------------------------------------------
    # 16.7 Train risk candidates.
    # --------------------------------------------------------
    (
        trained_risk_models,
        risk_results,
    ) = train_risk_models(
        X_train,
        y_train_risk,
        X_test,
        y_test_risk,
        previous_risk_model,
    )

    # --------------------------------------------------------
    # 16.8 Select the best candidates.
    # --------------------------------------------------------
    (
        best_score_model_name,
        score_ranking,
    ) = select_best_score_model(
        score_results
    )

    (
        best_risk_model_name,
        risk_ranking,
    ) = select_best_risk_model(
        risk_results
    )

    best_score_model = trained_score_models[
        best_score_model_name
    ]

    best_risk_model = trained_risk_models[
        best_risk_model_name
    ]

    print_section("BEST MODELS SELECTED")

    print(
        f"Best score model : {best_score_model_name}"
    )
    print(
        f"Best risk model  : {best_risk_model_name}"
    )

    # --------------------------------------------------------
    # 16.9 Print candidate comparisons.
    # --------------------------------------------------------
    print_section("SCORE MODEL COMPARISON")

    print(
        score_ranking[
            [
                "r2",
                "rmse",
                "mae",
                "combined_rank",
            ]
        ].to_string()
    )

    print_section("RISK MODEL COMPARISON")

    print(
        risk_ranking[
            [
                "accuracy",
                "weighted_f1",
                "combined_rank",
            ]
        ].to_string()
    )

    # --------------------------------------------------------
    # 16.10 Diagnostic prediction ranges.
    # --------------------------------------------------------
    print_model_predicted_score_ranges(
        best_score_model,
        best_risk_model,
        X_test,
    )

    # --------------------------------------------------------
    # 16.11 Build complete metadata.
    # --------------------------------------------------------
    metadata = {
        "project": PROJECT_NAME,
        "version": f"v{version_number}",
        "created_at": utc_now(),
        "batch_number": batch_number,
        "new_training_samples": TRAINING_BATCH_SIZE,
        "new_testing_samples": TESTING_BATCH_SIZE,
        "total_training_samples": int(
            len(all_training_data)
        ),
        "total_testing_samples": int(
            len(all_testing_data)
        ),
        "training_batches": len(training_files),
        "testing_batches": len(testing_files),
        "features": FEATURES,
        "score_target": SCORE_TARGET,
        "risk_target": RISK_TARGET,
        "risk_order": RISK_ORDER,
        "candidate_score_models": list(
            trained_score_models.keys()
        ),
        "candidate_risk_models": list(
            trained_risk_models.keys()
        ),
        "best_score_model": best_score_model_name,
        "best_risk_model": best_risk_model_name,
        "score_metrics": score_results,
        "risk_metrics": risk_results,
        "previous_model_loaded": previous_model_loaded,
        "previous_version": previous_version,
        "persistent_learning": {
            "historical_data_reused": True,
            "previous_model_loaded": previous_model_loaded,
            "warm_start_candidates_enabled": True,
            "warm_start_additional_trees": (
                WARM_START_ADDITIONAL_TREES
            ),
            "warm_start_additional_iterations": (
                WARM_START_ADDITIONAL_ITERATIONS
            ),
        },
    }

    # --------------------------------------------------------
    # 16.12 Save immutable version.
    # --------------------------------------------------------
    version_directory = save_model_version(
        version_number,
        best_score_model,
        best_risk_model,
        metadata,
    )

    # --------------------------------------------------------
    # 16.13 Update the application's current models.
    # --------------------------------------------------------
    update_current_model(
        version_directory
    )

    # --------------------------------------------------------
    # 16.14 Update registry AFTER all artifacts exist.
    # --------------------------------------------------------
    registry = write_project_registry(
        version_number,
        batch_number,
        metadata,
    )

    # --------------------------------------------------------
    # 16.15 Final status.
    # --------------------------------------------------------
    print_section("WAYPOINT TRAINING RUN COMPLETE")

    print(
        f"Model version created : v{version_number}"
    )
    print(
        f"Best score model     : {best_score_model_name}"
    )
    print(
        f"Best risk model      : {best_risk_model_name}"
    )
    print(
        f"Total training data  : "
        f"{len(all_training_data):,}"
    )
    print(
        f"Total testing data   : "
        f"{len(all_testing_data):,}"
    )
    print(
        f"Version directory    : {version_directory}"
    )
    print(
        f"Current model dir    : {CURRENT_MODEL_DIR}"
    )
    print(
        f"Registry             : {REGISTRY_PATH}"
    )

    return registry


# ============================================================
# 17. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
