<h1 align="center">WayPoint</h1>

<blockquote align="center">
  <h2><i>“Don’t just predict where you are. Understand where you can go.”</i></h2>
</blockquote>



WayPoint is an end-to-end machine learning system designed to analyse student academic performance and identify students who may require additional academic support.

The system takes six parameters from a student: study hours, phone usage, days before exam preparation, assignment percentage, attendance percentage, and previous marks. These parameters are analysed using trained machine learning models to predict expected performance and classify the student into an academic risk category.

WayPoint goes beyond simply predicting a result. It uses SHAP explainability to show which factors contributed most to a prediction, provides practical recommendations for students who need support, and introduces Dream Score, which allows a student to enter a desired target score and explore which areas could be improved to work towards it.

## Novelty Features

### *1. SHAP-Based Explainability*

WayPoint does not only provide a prediction; it explains which input parameters contributed most to the predicted performance and risk classification. This makes the model's output more understandable and actionable.

### *2. Dream Score Analysis*

Students can enter a desired target score, and WayPoint analyses what areas could be improved to work towards that target. The recommendations are constrained to practical and realistic changes rather than unrealistic study schedules or phone-usage reductions.

### *3. Persistent Learning & Model Versioning*

WayPoint retains historical training data and previous model versions instead of replacing them after every training run. Compatible previous models can be used for continued training, allowing the system to build on earlier learning.

Each training run creates a new model version, while the selected model is stored separately as the **current model** used by the application. This provides model history, backup, and controlled updates without losing previous versions.

### *4. Training–Application Separation*
## Training–Application Separation
WayPoint has two separate parts that work together:

${\color{red}\text{### Training System — training.py}}$

${\color{red}\text{training.py is responsible for generating and retaining historical data, training multiple machine learning models,}}$
${\color{red}\text{evaluating them using metrics such as R², RMSE, MAE, Accuracy, and Weighted F1 Score, and selecting the}}$
${\color{red}\text{best-performing models. The selected models are saved as the current models, while previous model versions}}$
${\color{red}\text{are retained in models/versions.}}$

${\color{red}\text{### Student-Facing Application — WayPoint.py}}$

${\color{red}\text{WayPoint.py is the student-facing application. It loads the best-performing models selected by training.py from}}$
${\color{red}\text{models/current and uses them to provide performance prediction, risk classification, SHAP-based explanations,}}$
${\color{red}\text{recommendations, and Dream Score analysis.}}$

${\color{red}\text{The application does not retrain the models when it is opened. It uses the saved models produced by the training}}$
${\color{red}\text{process. When training.py is run again, the models can be updated and a new model version is saved without}}$
${\color{red}\text{losing previous versions.}}$



 # *What Problem It Solves & Why It Is Unique*

A conventional student-performance prediction system generally answers “What is likely to happen?” but does not help the student understand “What can I work on to move towards my goal?”

WayPoint addresses this gap by combining prediction, risk classification, SHAP-based explanation, recommendations, and Dream Score in one workflow. The student can see their current position, understand the factors influencing the prediction, and then explore a desired target through a model-based improvement scenario.

What makes it distinctive is that the system connects prediction → explanation → actionable improvement → target-based planning, rather than treating prediction as the final output.

# *Risk Classification*

WayPoint classifies students into five categories:

Very High Risk, High Risk, Moderate Risk, Low Risk, and Very Low Risk.

These categories are not determined using manually fixed ranges such as "below 50 marks = High Risk."

The classification model learns the relationship between the six input parameters and the training outcomes. During training, it learns decision boundaries that separate the different risk classes. When a new student is analysed, their parameters are passed through the trained classifier, which determines which learned region they belong to.

This means the risk boundaries are learned from the training data and the model rather than being hard-coded into the application.


# *Persistent Learning & Model Versioning*

WayPoint is designed so that training and actual student usage are separate.

The training system is handled by training.py. It is responsible for generating data, loading previously stored training data, training multiple candidate models, evaluating them, selecting the best models, and saving the results.

The student-facing application, WayPoint.py, does not perform this training process. It loads the already-trained models and uses them to analyse the student's inputs. This keeps the application lightweight and gives the student a straightforward interface without unnecessary training operations happening in the background.

WayPoint also maintains previous training data and trained model versions. When the training process is run again, the existing project state can be loaded and used as the starting point for the next training cycle.

Compatible previous models can also be used for continued training where the algorithm supports it. For example, compatible Random Forest models can retain their existing trees and continue by adding new trees, while compatible HistGradientBoosting models can continue with additional boosting iterations.

Every trained model is saved as a separate version instead of simply overwriting the previous model. The selected model is also placed in the current-model directory, which is the version loaded by the student application.

This gives WayPoint a persistent training workflow where the project can continue improving while previous model versions remain available.



# WayPoint — Installation & Execution Guide

## Step 1 — Install the Required Software

Install the following if they are not already installed:

- **Python** — required to run WayPoint.
- **Git** — required to clone the WayPoint repository.


Open **Windows PowerShell** from the Start Menu.

Verify Python:

```powershell
python --version
```

Verify Git:

```powershell
git --version
```

Both commands should display their installed versions.

---

## Step 2 — Clone the WayPoint Repository

In **Windows PowerShell**, run:

```powershell
git clone https://github.com/Mohan-Sai-Karthikeya101/WayPoint.git
```

Enter the project folder:

```powershell
cd WayPoint
```

All further commands can be executed directly in **Windows PowerShell**.

---

## Step 3 — Create and Activate the Python Environment

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Step 4 — Install the Required Libraries

Run:

```powershell
pip install -r requirements.txt
```

The `requirements.txt` file is already included in the WayPoint repository. `pip` reads this file and automatically installs the libraries required by the project.

---

## Step 5 — Run the Student Application

Run:

```powershell
python WayPoint.py
```

The WayPoint student application loads the saved models from:

```text
models/current
```

The repository already contains the current trained models, so no separate model download or manual model linking is required.

The application accepts the following six parameters:

1. **Study Hours**
2. **Phone Usage Hours**
3. **Days Before Exam Preparation**
4. **Assignment Percentage**
5. **Attendance Percentage**
6. **Previous Marks**

WayPoint then provides:

- **Predicted Performance Score**
- **Risk Classification**
- **SHAP-Based Explanation**
- **Personalized Recommendations**
- **Dream Score Analysis**

---

## Step 6 — Train and Update the Models

To generate new training data, train the models, evaluate them, and update the current models, run:

```powershell
python training.py
```

The training system:

- Generates new synthetic student data.
- Retains historical training data.
- Trains multiple candidate models.
- Evaluates their performance.
- Selects the best-performing models.
- Saves a new model version.
- Updates the models used by the application.

Performance prediction models are evaluated using:

**R², RMSE, and MAE**

Risk classification models are evaluated using:

**Accuracy and Weighted F1 Score**

The selected model versions are stored in:

```text
models/versions
```

The currently selected models are stored in:

```text
models/current
```

Previous model versions are retained rather than being overwritten, allowing the project to maintain a history of trained models.

---

## Step 7 — Run the Updated Student Application

After training is complete, run:

```powershell
python WayPoint.py
```

WayPoint automatically loads the newly selected models from:

```text
models/current
```

No changes to the application code are required after retraining.

---

## Complete Workflow

The complete WayPoint workflow is:

**Data Generation → Historical Data → Model Training → Model Evaluation → Best Model Selection → Model Versioning → Current Model → Student Application**

The training and application components are separated:

- **`training.py`** handles data generation, historical data, training, evaluation, model selection, persistence, and versioning.
- **`WayPoint.py`** loads the saved current models and provides the student-facing prediction and analysis interface.

This allows the student application to run using the saved models without retraining every time it is opened.

# Team

| Name                 | Department | Roll Number |
| -------------------- | ---------- | ----------- |
|Mohan Sai Karthikeya | CSE        | 130003184   |
| Vignesh              | Aerospace  | 130017057   |
| Boga Chenchu Gagana  | IT         | 130015014   |

