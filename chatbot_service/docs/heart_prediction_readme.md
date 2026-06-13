
## 🌟 What does this file do?

The `heart_prediction.py` file is the core engine for diagnosing a user's risk of heart disease based on their health metrics (like age, blood pressure, cholesterol, etc.). 

It doesn't just give a simple "Yes/No" answer! It performs a sophisticated **two-step analysis**:

1. **Machine Learning Prediction:** First, it feeds the user's tabular data into a highly trained "Stacking Ensemble" of 8 different Machine Learning models to get a precise statistical probability of heart disease.
2. **MedGemma Clinical Interpretation:** Then, it passes that prediction—along with the user's specific test results and medical guidelines from our RAG database—to **MedGemma** (our specialized medical AI). MedGemma acts as a virtual cardiologist, explaining *why* the prediction was made in plain text, analyzing each abnormal test result, and offering actionable recommendations.

## 🤖 Models Used in This File

This file relies heavily on the pre-trained models located in the `chatbot_service/models/heart_disease/` folder.

### 1. The Stacking Ensemble Model
**File:** `stacking_heart_disease_model_v3.joblib`

Instead of relying on just one algorithm, we use a "Stacking Ensemble". This is a master model that looks at the predictions of 8 different base algorithms and combines them to make one final, highly accurate decision. 

The 8 base pipelines it uses are also loaded from the `models/heart_disease/` directory:
* `logistic_regression_pipeline.joblib`
* `svm_pipeline.joblib`
* `random_forest_pipeline.joblib`
* `knn_pipeline.joblib`
* `decision_tree_pipeline.joblib`
* `xgboost_pipeline.joblib`
* `lightgbm_pipeline.joblib`
* `mlp_pipeline.joblib`

*You can see this happening in the `ModelLoader.load_individual_pipelines()` function!*

### 2. MedGemma (via LLMGateway)
**Folder:** `chatbot_service/models/medgemma/`

While the ML models provide the *math*, MedGemma provides the *meaning*. 
In the `ClinicalInterpretationPipeline` class, the code connects to MedGemma. It uses MedGemma to generate a comprehensive, human-readable summary of the ML models' findings, tailored specifically to the patient's unique risk factors (e.g., "Your high cholesterol combined with chest pain increases your risk...").

## 🛣️ Key Endpoints Available

If you are a developer interacting with this service, here are the main API endpoints provided by this file:

* **`POST /heart/predict`**: The main endpoint. Submit patient data, get back the ML percentage risk AND the MedGemma written explanation.
* **`POST /heart/predict/ensemble`**: Geared for transparency. Get a detailed breakdown of exactly what each of the 8 individual ML models voted.
* **`POST /heart/predict/batch`**: Process multiple patients at once.
* **`POST /heart/insight`**: Ask a freeform cardiovascular health question to MedGemma.
* **`GET /heart/health`**: Check if all models and systems are loaded and running correctly.
