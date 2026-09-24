WayPoint is an end-to-end machine learning system designed to analyse student academic performance and identify students who may require additional academic support.

The system takes six parameters from a student: study hours, phone usage, days before exam preparation, assignment percentage, attendance percentage, and previous marks. These parameters are analysed using trained machine learning models to predict expected performance and classify the student into an academic risk category.

WayPoint goes beyond simply predicting a result. It uses SHAP explainability to show which factors contributed most to a prediction, provides practical recommendations for students who need support, and introduces Dream Score, which allows a student to enter a desired target score and explore which areas could be improved to work towards it.

What Problem It Solves & Why It Is Unique

A conventional student-performance prediction system generally answers “What is likely to happen?” but does not help the student understand “What can I work on to move towards my goal?”

WayPoint addresses this gap by combining prediction, risk classification, SHAP-based explanation, recommendations, and Dream Score in one workflow. The student can see their current position, understand the factors influencing the prediction, and then explore a desired target through a model-based improvement scenario.

What makes it distinctive is that the system connects prediction → explanation → actionable improvement → target-based planning, rather than treating prediction as the final output.

Risk Classification

WayPoint classifies students into five categories:

Very High Risk, High Risk, Moderate Risk, Low Risk, and Very Low Risk.

These categories are not determined using manually fixed ranges such as "below 50 marks = High Risk."

The classification model learns the relationship between the six input parameters and the training outcomes. During training, it learns decision boundaries that separate the different risk classes. When a new student is analysed, their parameters are passed through the trained classifier, which determines which learned region they belong to.

This means the risk boundaries are learned from the training data and the model rather than being hard-coded into the application.


Persistent Learning & Model Versioning

WayPoint is designed so that training and actual student usage are separate.

The training system is handled by training.py. It is responsible for generating data, loading previously stored training data, training multiple candidate models, evaluating them, selecting the best models, and saving the results.

The student-facing application, WayPoint.py, does not perform this training process. It loads the already-trained models and uses them to analyse the student's inputs. This keeps the application lightweight and gives the student a straightforward interface without unnecessary training operations happening in the background.

WayPoint also maintains previous training data and trained model versions. When the training process is run again, the existing project state can be loaded and used as the starting point for the next training cycle.

Compatible previous models can also be used for continued training where the algorithm supports it. For example, compatible Random Forest models can retain their existing trees and continue by adding new trees, while compatible HistGradientBoosting models can continue with additional boosting iterations.

Every trained model is saved as a separate version instead of simply overwriting the previous model. The selected model is also placed in the current-model directory, which is the version loaded by the student application.

This gives WayPoint a persistent training workflow where the project can continue improving while previous model versions remain available.

Data & Model Training

WayPoint uses synthetically generated student data so that the system can be trained and tested without using private student records. Each training run generates 20,000 training samples and 10,000 testing samples.
The model uses these six parameters:

#Team

| Name                 | Department | Roll Number |
| -------------------- | ---------- | ----------- |
| Vignesh              | Aerospace  | 130017057   |
| Boga Chenchu Gagana  | IT         | 130015014   |
| Mohan Sai Karthikeya | CSE        | 130003184   |
