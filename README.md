
                                      # WayPoint
> **“Don’t just predict where you are. Understand where you can go.”**




WayPoint is an end-to-end machine learning system designed to analyse student academic performance and identify students who may require additional academic support.

The system takes six parameters from a student: study hours, phone usage, days before exam preparation, assignment percentage, attendance percentage, and previous marks. These parameters are analysed using trained machine learning models to predict expected performance and classify the student into an academic risk category.

WayPoint goes beyond simply predicting a result. It uses SHAP explainability to show which factors contributed most to a prediction, provides practical recommendations for students who need support, and introduces Dream Score, which allows a student to enter a desired target score and explore which areas could be improved to work towards it.

## Novelty Features

### 1. SHAP-Based Explainability

WayPoint does not only provide a prediction; it explains which input parameters contributed most to the predicted performance and risk classification. This makes the model's output more understandable and actionable.

### 2. Dream Score Analysis

Students can enter a desired target score, and WayPoint analyses what areas could be improved to work towards that target. The recommendations are constrained to practical and realistic changes rather than unrealistic study schedules or phone-usage reductions.

### 3. Persistent Learning & Model Versioning

WayPoint retains historical training data and previous model versions instead of replacing them after every training run. Compatible previous models can be used for continued training, allowing the system to build on earlier learning.

Each training run creates a new model version, while the selected model is stored separately as the **current model** used by the application. This provides model history, backup, and controlled updates without losing previous versions.

### 4. Training–Application Separation

The training system and student-facing application are separated. `training.py` handles data generation, historical data loading, model training, evaluation, model selection, and versioning, while `WayPoint.py` loads the saved current models for prediction.

This allows the application to be used without retraining the models every time and keeps the complete workflow organized from **data generation → training → evaluation → model selection → versioning → prediction**.


The project is divided into two main parts:

training.py — handles data generation, training, evaluation, model selection, persistence, and versioning.

WayPoint.py — the student-facing application that loads the saved current models and performs predictions.

IMPORTANT: The normal student application does not need to retrain the models. Training and model updating are handled separately.

 # **What Problem It Solves & Why It Is Unique**

A conventional student-performance prediction system generally answers “What is likely to happen?” but does not help the student understand “What can I work on to move towards my goal?”

WayPoint addresses this gap by combining prediction, risk classification, SHAP-based explanation, recommendations, and Dream Score in one workflow. The student can see their current position, understand the factors influencing the prediction, and then explore a desired target through a model-based improvement scenario.

What makes it distinctive is that the system connects prediction → explanation → actionable improvement → target-based planning, rather than treating prediction as the final output.

# **Risk Classification**

WayPoint classifies students into five categories:

Very High Risk, High Risk, Moderate Risk, Low Risk, and Very Low Risk.

These categories are not determined using manually fixed ranges such as "below 50 marks = High Risk."

The classification model learns the relationship between the six input parameters and the training outcomes. During training, it learns decision boundaries that separate the different risk classes. When a new student is analysed, their parameters are passed through the trained classifier, which determines which learned region they belong to.

This means the risk boundaries are learned from the training data and the model rather than being hard-coded into the application.


# **Persistent Learning & Model Versioning**

WayPoint is designed so that training and actual student usage are separate.

The training system is handled by training.py. It is responsible for generating data, loading previously stored training data, training multiple candidate models, evaluating them, selecting the best models, and saving the results.

The student-facing application, WayPoint.py, does not perform this training process. It loads the already-trained models and uses them to analyse the student's inputs. This keeps the application lightweight and gives the student a straightforward interface without unnecessary training operations happening in the background.

WayPoint also maintains previous training data and trained model versions. When the training process is run again, the existing project state can be loaded and used as the starting point for the next training cycle.

Compatible previous models can also be used for continued training where the algorithm supports it. For example, compatible Random Forest models can retain their existing trees and continue by adding new trees, while compatible HistGradientBoosting models can continue with additional boosting iterations.

Every trained model is saved as a separate version instead of simply overwriting the previous model. The selected model is also placed in the current-model directory, which is the version loaded by the student application.

This gives WayPoint a persistent training workflow where the project can continue improving while previous model versions remain available.


# *Training–Application Separation*
The WayPoint system is divided into two main parts:

training.py — Handles data generation, historical data loading, model training, evaluation, model selection, persistence, and model versioning.

WayPoint.py — Acts as the student-facing application. It loads the latest selected models from models/current and uses them to perform predictions, risk classification, explainability, recommendations, and Dream Score analysis.

The student application does not retrain the models every time it is opened. Training and model updates are handled separately through training.py, while WayPoint.py focuses on using the saved models.

This keeps the complete workflow organized as:

Data Generation → Training → Evaluation → Model Selection → Versioning → Current Model → Prediction

# Team

| Name                 | Department | Roll Number |
| -------------------- | ---------- | ----------- |
|Mohan Sai Karthikeya | CSE        | 130003184   |
| Vignesh              | Aerospace  | 130017057   |
| Boga Chenchu Gagana  | IT         | 130015014   |

