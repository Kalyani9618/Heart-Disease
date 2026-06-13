"""
Heart Disease Prediction Router - Integrated into main FastAPI app.

Routes the ML stacking ensemble heart disease prediction through MedGemma
for clinical interpretation. After the ML model predicts, MedGemma ALWAYS
explains WHY the disease is present or absent.

Endpoints:
    POST /heart/predict          - Predict + MedGemma explanation (default)
    POST /heart/predict/ensemble - Detailed per-model breakdown
    POST /heart/predict/batch    - Batch predictions
    POST /heart/insight          - Freeform MedGemma health insight
    GET  /heart/health           - Heart prediction service health
"""

import os
import sys
import time
import uuid
import logging
import warnings
from typing import Optional, Dict, Any, List
from enum import Enum

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

logger = logging.getLogger("heart-prediction")

# --- NUMPY 2.0+ BACKWARD COMPATIBILITY PATCH ---
import numpy.random

if "numpy.random._mt19937" not in sys.modules:
    import types as _types
    _mock_mt19937 = _types.ModuleType("numpy.random._mt19937")
    try:
        _mock_mt19937.MT19937 = numpy.random.MT19937
    except AttributeError:
        pass
    sys.modules["numpy.random._mt19937"] = _mock_mt19937


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class HeartDiseaseInput(BaseModel):
    """Validated input for heart disease prediction."""
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    chest_pain_type: int = Field(..., ge=1, le=4, description="1=Typical Angina, 2=Atypical, 3=Non-Anginal, 4=Asymptomatic")
    resting_bp_s: int = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    cholesterol: int = Field(..., ge=0, le=700, description="Cholesterol (mg/dl)")
    fasting_blood_sugar: int = Field(..., ge=0, le=1, description="0=FBS<120, 1=FBS>120")
    resting_ecg: int = Field(..., ge=0, le=2, description="0=Normal, 1=ST-T abnormality, 2=LV hypertrophy")
    max_heart_rate: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exercise_angina: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    oldpeak: float = Field(..., ge=-5.0, le=10.0, description="ST depression")
    st_slope: int = Field(..., ge=1, le=3, description="1=Up, 2=Flat, 3=Down")

    @validator("cholesterol")
    def warn_zero_cholesterol(cls, v):
        if v == 0:
            logger.warning("Cholesterol=0 — may be missing data, imputation applied")
        return v


class TestResultDetail(BaseModel):
    """Individual test result with normal range and risk assessment."""
    test_name: str
    value: str
    normal_range: str
    status: str  # "Normal", "Abnormal", "Borderline", "Critical"
    risk_contribution: str  # "Low", "Moderate", "High"
    explanation: str


class PredictionResponse(BaseModel):
    """Full prediction response with MedGemma explanation."""
    prediction: int
    probability: float
    risk_level: str
    confidence: float
    message: str
    test_results: Optional[List[TestResultDetail]] = None
    clinical_interpretation: Optional[str] = None
    triage_level: Optional[str] = None
    triage_actions: Optional[List[str]] = None
    guidelines_cited: Optional[List[str]] = None
    is_grounded: Optional[bool] = None
    quality_score: Optional[float] = None
    needs_medical_attention: bool = False
    processing_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Heart prediction health check."""
    status: str
    model_loaded: bool
    medgemma_available: bool
    rag_available: bool
    memory_available: bool
    numpy_version: str
    pipeline_version: str


# ---------------------------------------------------------------------------
# Risk Level Classification
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


def classify_risk(probability: float, input_data: HeartDiseaseInput) -> RiskLevel:
    """Classify risk using ML probability + clinical red flags."""
    red_flags = 0
    if input_data.chest_pain_type == 4:
        red_flags += 1
    if input_data.exercise_angina == 1:
        red_flags += 1
    if input_data.st_slope == 3:
        red_flags += 1
    if input_data.oldpeak > 2.0:
        red_flags += 1
    if input_data.resting_bp_s > 180:
        red_flags += 1
    if input_data.max_heart_rate < 100 and input_data.age < 60:
        red_flags += 1

    if probability > 0.85 or (probability > 0.7 and red_flags >= 3):
        return RiskLevel.CRITICAL
    elif probability > 0.6 or (probability > 0.45 and red_flags >= 2):
        return RiskLevel.HIGH
    elif probability > 0.35 or red_flags >= 2:
        return RiskLevel.MODERATE
    else:
        return RiskLevel.LOW


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "age", "resting bp s", "cholesterol", "max heart rate", "oldpeak",
    "sex_1",
    "chest pain type_2", "chest pain type_3", "chest pain type_4",
    "fasting blood sugar_1",
    "resting ecg_1", "resting ecg_2",
    "exercise angina_1",
    "ST slope_1", "ST slope_2", "ST slope_3",
]


def build_feature_dataframe(input_data: HeartDiseaseInput) -> pd.DataFrame:
    """Convert validated input into one-hot-encoded feature DataFrame."""
    data = {
        "age": [input_data.age],
        "resting bp s": [input_data.resting_bp_s],
        "cholesterol": [input_data.cholesterol],
        "max heart rate": [input_data.max_heart_rate],
        "oldpeak": [input_data.oldpeak],
        "sex_1": [1 if input_data.sex == 1 else 0],
        "chest pain type_2": [1 if input_data.chest_pain_type == 2 else 0],
        "chest pain type_3": [1 if input_data.chest_pain_type == 3 else 0],
        "chest pain type_4": [1 if input_data.chest_pain_type == 4 else 0],
        "fasting blood sugar_1": [1 if input_data.fasting_blood_sugar == 1 else 0],
        "resting ecg_1": [1 if input_data.resting_ecg == 1 else 0],
        "resting ecg_2": [1 if input_data.resting_ecg == 2 else 0],
        "exercise angina_1": [1 if input_data.exercise_angina == 1 else 0],
        "ST slope_1": [1 if input_data.st_slope == 1 else 0],
        "ST slope_2": [1 if input_data.st_slope == 2 else 0],
        "ST slope_3": [1 if input_data.st_slope == 3 else 0],
    }
    # Force object-typed column labels to avoid pandas Arrow string index crashes
    # observed on some Windows builds.
    safe_columns = np.asarray(FEATURE_COLUMNS, dtype=object)
    return pd.DataFrame(data, columns=safe_columns)


def format_patient_summary(input_data: HeartDiseaseInput) -> str:
    """Format patient data as clinical summary for MedGemma."""
    sex_str = "Male" if input_data.sex == 1 else "Female"
    cp_map = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}
    ecg_map = {0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"}
    slope_map = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}

    return f"""Patient Profile:
- Age: {input_data.age} years, Sex: {sex_str}
- Chest Pain Type: {cp_map.get(input_data.chest_pain_type, 'Unknown')}
- Resting Blood Pressure: {input_data.resting_bp_s} mm Hg
- Cholesterol: {input_data.cholesterol} mg/dl
- Fasting Blood Sugar > 120 mg/dl: {'Yes' if input_data.fasting_blood_sugar else 'No'}
- Resting ECG: {ecg_map.get(input_data.resting_ecg, 'Unknown')}
- Maximum Heart Rate: {input_data.max_heart_rate} bpm
- Exercise-Induced Angina: {'Yes' if input_data.exercise_angina else 'No'}
- ST Depression (Oldpeak): {input_data.oldpeak}
- ST Slope: {slope_map.get(input_data.st_slope, 'Unknown')}"""


def build_test_result_details(input_data: HeartDiseaseInput) -> List[TestResultDetail]:
    """Build structured test result analysis with normal ranges and risk flags."""
    cp_map = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}
    ecg_map = {0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"}
    slope_map = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}

    results = []

    # Age
    age_status = "Normal" if input_data.age < 45 else ("Borderline" if input_data.age < 55 else "Abnormal")
    age_risk = "Low" if input_data.age < 45 else ("Moderate" if input_data.age < 55 else "High")
    results.append(TestResultDetail(
        test_name="Age",
        value=f"{input_data.age} years",
        normal_range="Risk increases significantly after age 45 (men) / 55 (women)",
        status=age_status,
        risk_contribution=age_risk,
        explanation=f"Age {input_data.age} — {'increased cardiovascular risk due to aging' if input_data.age >= 45 else 'relatively lower age-related risk'}",
    ))

    # Resting Blood Pressure
    if input_data.resting_bp_s < 120:
        bp_status, bp_risk = "Normal", "Low"
    elif input_data.resting_bp_s < 130:
        bp_status, bp_risk = "Borderline", "Moderate"
    elif input_data.resting_bp_s < 140:
        bp_status, bp_risk = "Abnormal", "Moderate"
    elif input_data.resting_bp_s < 180:
        bp_status, bp_risk = "Abnormal", "High"
    else:
        bp_status, bp_risk = "Critical", "High"
    results.append(TestResultDetail(
        test_name="Resting Blood Pressure",
        value=f"{input_data.resting_bp_s} mm Hg",
        normal_range="Normal: <120 mm Hg | Elevated: 120-129 | High: 130-139 | Stage 2: 140-179 | Crisis: >=180",
        status=bp_status,
        risk_contribution=bp_risk,
        explanation=f"BP of {input_data.resting_bp_s} mm Hg is {bp_status.lower()} — {'hypertension increases strain on the heart and arteries' if input_data.resting_bp_s >= 130 else 'within acceptable range'}",
    ))

    # Cholesterol
    if input_data.cholesterol == 0:
        chol_status, chol_risk = "Abnormal", "Moderate"
        chol_explain = "Cholesterol=0 likely indicates missing data; imputed value may affect prediction"
    elif input_data.cholesterol < 200:
        chol_status, chol_risk = "Normal", "Low"
        chol_explain = f"Cholesterol {input_data.cholesterol} mg/dl is within desirable range"
    elif input_data.cholesterol < 240:
        chol_status, chol_risk = "Borderline", "Moderate"
        chol_explain = f"Cholesterol {input_data.cholesterol} mg/dl is borderline high — increased plaque buildup risk"
    else:
        chol_status, chol_risk = "Abnormal", "High"
        chol_explain = f"Cholesterol {input_data.cholesterol} mg/dl is HIGH — significantly increases atherosclerosis and coronary artery disease risk"
    results.append(TestResultDetail(
        test_name="Cholesterol",
        value=f"{input_data.cholesterol} mg/dl",
        normal_range="Desirable: <200 mg/dl | Borderline: 200-239 | High: >=240",
        status=chol_status,
        risk_contribution=chol_risk,
        explanation=chol_explain,
    ))

    # Max Heart Rate
    age_predicted_max = 220 - input_data.age
    hr_pct = (input_data.max_heart_rate / age_predicted_max * 100) if age_predicted_max > 0 else 0
    if hr_pct >= 85:
        hr_status, hr_risk = "Normal", "Low"
    elif hr_pct >= 70:
        hr_status, hr_risk = "Borderline", "Moderate"
    else:
        hr_status, hr_risk = "Abnormal", "High"
    results.append(TestResultDetail(
        test_name="Maximum Heart Rate",
        value=f"{input_data.max_heart_rate} bpm",
        normal_range=f"Age-predicted max: {age_predicted_max} bpm | Target during exercise: >={int(age_predicted_max * 0.85)} bpm (85%)",
        status=hr_status,
        risk_contribution=hr_risk,
        explanation=f"Max HR of {input_data.max_heart_rate} bpm is {hr_pct:.0f}% of age-predicted maximum — {'adequate cardiac response' if hr_pct >= 85 else 'chronotropic incompetence suggests reduced cardiac function'}",
    ))

    # Chest Pain Type
    cp_status = "Normal" if input_data.chest_pain_type == 1 else ("Abnormal" if input_data.chest_pain_type == 4 else "Borderline")
    cp_risk = "Low" if input_data.chest_pain_type == 1 else ("High" if input_data.chest_pain_type == 4 else "Moderate")
    results.append(TestResultDetail(
        test_name="Chest Pain Type",
        value=cp_map.get(input_data.chest_pain_type, "Unknown"),
        normal_range="1=Typical Angina (expected) | 2=Atypical | 3=Non-Anginal | 4=Asymptomatic (highest risk)",
        status=cp_status,
        risk_contribution=cp_risk,
        explanation=f"{cp_map.get(input_data.chest_pain_type)} — {'asymptomatic presentation is paradoxically the highest risk as it suggests silent ischemia' if input_data.chest_pain_type == 4 else 'typical angina is actually more expected and less concerning than asymptomatic' if input_data.chest_pain_type == 1 else 'atypical presentation warrants further investigation'}",
    ))

    # Fasting Blood Sugar
    fbs_status = "Abnormal" if input_data.fasting_blood_sugar == 1 else "Normal"
    fbs_risk = "Moderate" if input_data.fasting_blood_sugar == 1 else "Low"
    results.append(TestResultDetail(
        test_name="Fasting Blood Sugar",
        value=f"{'> 120 mg/dl' if input_data.fasting_blood_sugar else '<= 120 mg/dl'}",
        normal_range="Normal: <= 120 mg/dl | Elevated: > 120 mg/dl (suggests diabetes/pre-diabetes)",
        status=fbs_status,
        risk_contribution=fbs_risk,
        explanation=f"FBS {'> 120 mg/dl — elevated glucose is associated with diabetes, a major independent cardiovascular risk factor' if input_data.fasting_blood_sugar else '<= 120 mg/dl — within normal glycemic range'}",
    ))

    # Resting ECG
    ecg_status = "Normal" if input_data.resting_ecg == 0 else "Abnormal"
    ecg_risk = "Low" if input_data.resting_ecg == 0 else "High"
    results.append(TestResultDetail(
        test_name="Resting ECG",
        value=ecg_map.get(input_data.resting_ecg, "Unknown"),
        normal_range="0=Normal | 1=ST-T wave abnormality (indicates ischemia) | 2=LV hypertrophy (heart muscle thickening)",
        status=ecg_status,
        risk_contribution=ecg_risk,
        explanation=f"ECG shows {ecg_map.get(input_data.resting_ecg)} — {'normal electrical activity, no signs of ischemia or structural changes' if input_data.resting_ecg == 0 else 'ST-T abnormality suggests possible myocardial ischemia or injury' if input_data.resting_ecg == 1 else 'left ventricular hypertrophy indicates the heart muscle is thickened, often due to chronic hypertension'}",
    ))

    # Exercise-Induced Angina
    ea_status = "Abnormal" if input_data.exercise_angina == 1 else "Normal"
    ea_risk = "High" if input_data.exercise_angina == 1 else "Low"
    results.append(TestResultDetail(
        test_name="Exercise-Induced Angina",
        value=f"{'Yes' if input_data.exercise_angina else 'No'}",
        normal_range="Normal: No exercise-induced chest pain",
        status=ea_status,
        risk_contribution=ea_risk,
        explanation=f"{'Exercise-induced angina PRESENT — strong indicator of coronary artery disease; chest pain during exertion signals inadequate blood flow to the heart muscle' if input_data.exercise_angina else 'No exercise-induced angina — heart receives adequate blood flow during physical stress'}",
    ))

    # ST Depression (Oldpeak)
    if input_data.oldpeak <= 0:
        op_status, op_risk = "Normal", "Low"
    elif input_data.oldpeak <= 1.0:
        op_status, op_risk = "Borderline", "Moderate"
    elif input_data.oldpeak <= 2.0:
        op_status, op_risk = "Abnormal", "High"
    else:
        op_status, op_risk = "Critical", "High"
    results.append(TestResultDetail(
        test_name="ST Depression (Oldpeak)",
        value=f"{input_data.oldpeak}",
        normal_range="Normal: 0 | Mild: 0.1-1.0 | Significant: 1.1-2.0 | Severe: >2.0",
        status=op_status,
        risk_contribution=op_risk,
        explanation=f"Oldpeak of {input_data.oldpeak} — {'no significant ST depression, normal recovery' if input_data.oldpeak <= 0 else f'ST depression of {input_data.oldpeak}mm indicates myocardial ischemia during exercise; the heart muscle is not getting enough oxygen'}",
    ))

    # ST Slope
    slope_status = "Normal" if input_data.st_slope == 1 else ("Abnormal" if input_data.st_slope == 3 else "Borderline")
    slope_risk = "Low" if input_data.st_slope == 1 else ("High" if input_data.st_slope == 3 else "Moderate")
    results.append(TestResultDetail(
        test_name="ST Slope",
        value=slope_map.get(input_data.st_slope, "Unknown"),
        normal_range="1=Upsloping (normal) | 2=Flat (concerning) | 3=Downsloping (highly abnormal)",
        status=slope_status,
        risk_contribution=slope_risk,
        explanation=f"ST slope is {slope_map.get(input_data.st_slope)} — {'normal upsloping recovery, indicates healthy cardiac response' if input_data.st_slope == 1 else 'flat ST slope suggests borderline ischemic response during exercise' if input_data.st_slope == 2 else 'downsloping ST segment is a strong indicator of significant coronary artery disease'}",
    ))

    # Sex
    sex_risk = "Moderate" if input_data.sex == 1 else "Low"
    results.append(TestResultDetail(
        test_name="Sex",
        value="Male" if input_data.sex == 1 else "Female",
        normal_range="Males have statistically higher heart disease risk, especially before age 55",
        status="Borderline" if input_data.sex == 1 else "Normal",
        risk_contribution=sex_risk,
        explanation=f"{'Male — statistically higher risk of coronary artery disease; male sex is an independent risk factor' if input_data.sex == 1 else 'Female — relatively lower risk pre-menopause; estrogen provides some cardiovascular protection'}",
    ))

    return results


# ---------------------------------------------------------------------------
# ML Model Loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Handles stacking ensemble model loading with NumPy 2.0+ compat."""

    # Ordered list of directories to search for model files
    MODEL_SEARCH_PATHS = [
        # Primary: canonical models directory (chatbot_service/models/heart_disease)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "heart_disease"),
        # Fallback: legacy heart_disease_prediction/Models directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "heart_disease_prediction", "Models"),
    ]

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir:
            self.model_dir = model_dir
        else:
            # Auto-discover: use the first path that exists
            self.model_dir = next(
                (p for p in self.MODEL_SEARCH_PATHS if os.path.isdir(p)),
                self.MODEL_SEARCH_PATHS[0],  # default to primary even if missing
            )
        logger.info(f"ModelLoader using model directory: {self.model_dir}")
        self._model = None
        self._individual_pipelines: Dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_stacking_model(self, filename: str = "stacking_heart_disease_model_v3.joblib"):
        """Load the stacking ensemble model."""
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress loky CPU count warning (wmic deprecated on Windows 11)
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not find the number of physical cores.*")
            # Suppress LGBMClassifier feature names warning
            warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
            
            try:
                self._model = joblib.load(model_path, mmap_mode=None)
                logger.info(f"Stacking model loaded: {model_path}")
            except (ValueError, AttributeError) as e:
                if "BitGenerator" in str(e) or "MT19937" in str(e):
                    logger.warning("NumPy 2.0+ compat issue — fallback loader")
                    try:
                        from sklearn.utils._joblib import load as sklearn_load
                        self._model = sklearn_load(model_path, mmap_mode=None)
                    except ImportError:
                        import pickle
                        with open(model_path, "rb") as f:
                            self._model = pickle.load(f)
                    logger.info(f"Model loaded via fallback: {model_path}")
                else:
                    raise

    def load_individual_pipelines(self):
        """Load individual model pipelines for ensemble transparency."""
        pipeline_files = {
            "logistic_regression": "logistic_regression_pipeline.joblib",
            "svm": "svm_pipeline.joblib",
            "random_forest": "random_forest_pipeline.joblib",
            "knn": "knn_pipeline.joblib",
            "decision_tree": "decision_tree_pipeline.joblib",
            "xgboost": "xgboost_pipeline.joblib",
            "lightgbm": "lightgbm_pipeline.joblib",
            "mlp": "mlp_pipeline.joblib",
        }
        for name, filename in pipeline_files.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Suppress loky CPU count warning (wmic deprecated on Windows 11)
                        warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not find the number of physical cores.*")
                        # Suppress LGBMClassifier feature names warning
                        warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
                        self._individual_pipelines[name] = joblib.load(path)
                    logger.debug(f"Loaded pipeline: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")

    def predict(self, df: pd.DataFrame) -> tuple:
        """Run prediction → (prediction, probability)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        prediction = int(self._model.predict(df)[0])
        try:
            probability = float(self._model.predict_proba(df)[0][1])
        except Exception:
            probability = float(prediction)
        return prediction, probability

    def get_ensemble_votes(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get each individual model's prediction for transparency."""
        votes = {}
        for name, pipeline in self._individual_pipelines.items():
            try:
                pred = int(pipeline.predict(df)[0])
                try:
                    prob = float(pipeline.predict_proba(df)[0][1])
                except Exception:
                    prob = float(pred)
                votes[name] = {"prediction": pred, "probability": round(prob, 4)}
            except Exception as e:
                votes[name] = {"error": str(e)}
        return votes


# ---------------------------------------------------------------------------
# MedGemma Clinical Interpretation Pipeline
# ---------------------------------------------------------------------------

class ClinicalInterpretationPipeline:
    """
    After ML stacking model predicts, MedGemma explains WHY.

    Full 7-step pipeline:
    1. RETRIEVE — HeartDiseaseRAG for guidelines
    2. MEMORY  — MemoriRAGBridge for patient history
    3. TRIAGE  — TriageSystem for ESI urgency
    4. REASON  — MedGemma interprets ML prediction + context
    5. VALIDATE — HallucinationGrader for grounding
    6. EVALUATE — ResponseEvaluator for quality scoring
    7. GUARD   — SafetyGuardrail via LLMGateway
    """

    def __init__(self):
        self._llm = None
        self._rag = None
        self._memori = None
        self._grader = None
        self._triage = None
        self._diff_diagnosis = None
        self._evaluator = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy-load all subsystems on first use."""
        if self._initialized:
            return

        try:
            from core.llm.llm_gateway import get_llm_gateway
            self._llm = get_llm_gateway()
            logger.info("LLMGateway (MedGemma) connected for heart prediction")
        except Exception as e:
            logger.warning(f"LLMGateway not available: {e}")

        try:
            from rag.rag_engines import get_heart_disease_rag
            self._rag = get_heart_disease_rag()
            logger.info("HeartDiseaseRAG connected")
        except Exception as e:
            logger.warning(f"HeartDiseaseRAG not available: {e}")

        try:
            from core.dependencies import DIContainer
            container = DIContainer.get_instance()
            self._memori = container.get_service("memori_bridge")
        except Exception as e:
            logger.debug(f"MemoriRAGBridge not available: {e}")

        try:
            from core.safety.hallucination_grader import HallucinationGrader
            self._grader = HallucinationGrader()
        except Exception as e:
            logger.debug(f"HallucinationGrader not available: {e}")

        try:
            from agents.components.triage_system import TriageSystem
            self._triage = TriageSystem()
        except Exception as e:
            logger.debug(f"TriageSystem not available: {e}")

        try:
            from agents.components.differential_diagnosis import DifferentialDiagnosisEngine
            self._diff_diagnosis = DifferentialDiagnosisEngine(llm_gateway=self._llm)
        except Exception as e:
            logger.debug(f"DifferentialDiagnosisEngine not available: {e}")

        try:
            from agents.evaluation import ResponseEvaluator
            self._evaluator = ResponseEvaluator(llm_gateway=self._llm)
        except Exception as e:
            logger.debug(f"ResponseEvaluator not available: {e}")

        self._initialized = True

    async def interpret(
        self,
        input_data: HeartDiseaseInput,
        prediction: int,
        probability: float,
        risk_level: RiskLevel,
        ensemble_votes: Optional[Dict] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        MedGemma explains WHY the ML model predicted heart disease or not.

        Sends the ML prediction + patient features + guidelines to MedGemma
        and returns a clinical interpretation with reasoning.
        """
        self._lazy_init()

        result = {
            "clinical_interpretation": None,
            "triage_level": None,
            "triage_actions": [],
            "guidelines_cited": [],
            "is_grounded": None,
            "quality_score": None,
        }

        if not self._llm:
            result["clinical_interpretation"] = (
                f"MedGemma unavailable. ML-only result: "
                f"{'Positive' if prediction == 1 else 'Negative'} "
                f"({probability:.1%} probability, {risk_level.value} risk)."
            )
            return result

        patient_summary = format_patient_summary(input_data)

        # Step 1: RETRIEVE — Guidelines from RAG
        guidelines_context = ""
        citations = []
        if self._rag:
            try:
                retrieval = self._rag.retrieve_context(
                    query=f"heart disease risk assessment {patient_summary[:200]}",
                    top_k=5,
                )
                if isinstance(retrieval, dict):
                    guidelines_context = retrieval.get("context", "")
                    citations = retrieval.get("sources", [])
                elif hasattr(retrieval, "documents") and retrieval.documents:
                    guidelines_context = "\n".join(
                        f"- [{src}] {doc}" for doc, src in zip(retrieval.documents, retrieval.sources)
                    )
                    citations = list(set(retrieval.sources))
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        result["guidelines_cited"] = citations

        # Step 2: MEMORY — Patient history
        patient_history = ""
        if user_id and self._memori:
            try:
                memory_ctx = self._memori.get_context_for_query(
                    query=patient_summary, user_id=user_id, max_memories=3,
                )
                if memory_ctx:
                    patient_history = f"\n### Patient History:\n{memory_ctx}"
            except Exception as e:
                logger.debug(f"Memory retrieval failed: {e}")

        # Step 3: TRIAGE — ESI assessment
        if self._triage:
            try:
                triage_result = await self._triage.assess(
                    chief_complaint=(
                        f"Heart disease risk assessment. ML model predicts "
                        f"{'positive' if prediction == 1 else 'negative'} "
                        f"with {probability:.0%} probability."
                    ),
                    symptoms=patient_summary,
                )
                result["triage_level"] = f"ESI-{triage_result.esi_level.value}: {triage_result.category.value}"
                result["triage_actions"] = triage_result.recommended_actions
            except Exception as e:
                logger.debug(f"Triage assessment failed: {e}")

        # Step 4: REASON — MedGemma explains WHY
        ensemble_info = ""
        if ensemble_votes:
            ensemble_info = "\n### Individual Model Votes:\n"
            for model_name, vote in ensemble_votes.items():
                if "error" not in vote:
                    ensemble_info += (
                        f"- {model_name}: {'Positive' if vote['prediction'] == 1 else 'Negative'} "
                        f"(confidence: {vote['probability']:.1%})\n"
                    )

        # Build test result summary for MedGemma
        test_results = build_test_result_details(input_data)
        abnormal_results = [r for r in test_results if r.status in ("Abnormal", "Critical")]
        borderline_results = [r for r in test_results if r.status == "Borderline"]
        high_risk_results = [r for r in test_results if r.risk_contribution == "High"]

        test_results_text = "\n### Patient Test Results vs Normal Ranges:\n"
        for r in test_results:
            flag = "⚠️" if r.status in ("Abnormal", "Critical") else ("⚡" if r.status == "Borderline" else "✅")
            test_results_text += (
                f"{flag} **{r.test_name}**: {r.value} "
                f"(Normal: {r.normal_range}) — Status: {r.status}, Risk: {r.risk_contribution}\n"
            )

        prompt = f"""You are a medical AI assistant specializing in cardiovascular health.
A machine learning stacking ensemble (8 models) has analyzed patient data for heart disease risk.
Your job is to explain WHY the model predicted this result by analyzing EACH test result.

{patient_summary}
{test_results_text}
### ML Stacking Ensemble Prediction:
- **Prediction**: {'POSITIVE — Heart Disease Likely' if prediction == 1 else 'NEGATIVE — Heart Disease Unlikely'}
- **Probability**: {probability:.1%}
- **Risk Level**: {risk_level.value}
- **Abnormal Results**: {len(abnormal_results)} of {len(test_results)} tests
- **High-Risk Contributors**: {', '.join(r.test_name for r in high_risk_results) if high_risk_results else 'None'}
{ensemble_info}
### Medical Guidelines (Verified Sources):
{guidelines_context if guidelines_context else 'No specific guidelines retrieved. Use general cardiology knowledge.'}
{patient_history}

### Your Task:
Explain WHY the ML model predicted {'heart disease is likely' if prediction == 1 else 'heart disease is unlikely'} for this patient.
Go through EACH test result above and explain how it contributed to the prediction.

### Response Format:
**WHY This Prediction Was Made:**
[2-3 sentence summary explaining the core reason for the ML prediction based on the test results]

**Test-by-Test Analysis:**
For each abnormal or concerning result, explain:
- What the value means clinically
- How it compares to the normal range
- WHY it increases or decreases heart disease risk
- How it interacts with other test results

**Key Risk Factors That Drove the Prediction:**
[List the top 3-5 test results that most influenced the ML model's decision, ranked by importance]

**Protective Factors (if any):**
[List any test results that were normal and reduce risk]

**Clinical Recommendations:**
[Evidence-based next steps specific to THIS patient's abnormal results]

**Risk Summary:**
[One-line: risk level + the single most important reason]

IMPORTANT: This is an AI-assisted analysis for informational purposes only. Always consult a qualified healthcare provider for medical decisions."""

        try:
            interpretation = await self._llm.generate(
                prompt=prompt,
                content_type="medical_analysis",
                user_id=user_id,
            )
            result["clinical_interpretation"] = interpretation
        except Exception as e:
            logger.error(f"MedGemma interpretation failed: {e}")
            result["clinical_interpretation"] = (
                f"ML Prediction: {'Positive' if prediction == 1 else 'Negative'} "
                f"({probability:.1%} probability, {risk_level.value} risk). "
                "Detailed clinical interpretation unavailable — please consult a cardiologist."
            )
            return result

        # Step 5: VALIDATE — Hallucination grading
        if self._grader:
            try:
                context_for_grading = guidelines_context + patient_history + patient_summary
                is_grounded = await self._grader.grade(
                    answer=interpretation, context=context_for_grading,
                )
                result["is_grounded"] = is_grounded
                if not is_grounded:
                    logger.warning("MedGemma response may contain hallucinations")
            except Exception as e:
                logger.debug(f"Hallucination grading failed: {e}")

        # Step 6: EVALUATE — Quality scoring
        if self._evaluator:
            try:
                eval_result = await self._evaluator.evaluate_all(
                    query=patient_summary, response=interpretation, context=guidelines_context,
                )
                result["quality_score"] = eval_result.overall_score
            except Exception as e:
                logger.debug(f"Quality evaluation failed: {e}")

        return result


# ---------------------------------------------------------------------------
# Module-level State (initialized on router startup)
# ---------------------------------------------------------------------------

_model_loader: Optional[ModelLoader] = None
_interpretation_pipeline: Optional[ClinicalInterpretationPipeline] = None


def initialize_heart_prediction():
    """Initialize ML model and interpretation pipeline. Called during app startup."""
    global _model_loader, _interpretation_pipeline

    _model_loader = ModelLoader()
    _interpretation_pipeline = ClinicalInterpretationPipeline()

    try:
        _model_loader.load_stacking_model()
        logger.info("Heart disease stacking model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load stacking model: {e}")
        import traceback
        traceback.print_exc()

    try:
        _model_loader.load_individual_pipelines()
        count = len(_model_loader._individual_pipelines)
        logger.info(f"Loaded {count} individual pipelines for ensemble transparency")
    except Exception as e:
        logger.warning(f"Individual pipeline loading failed: {e}")


def shutdown_heart_prediction():
    """Clean up on app shutdown."""
    global _model_loader, _interpretation_pipeline
    _model_loader = None
    _interpretation_pipeline = None
    logger.info("Heart Disease Prediction Service shut down")


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_heart_disease(
    input_data: HeartDiseaseInput,
    user_id: Optional[str] = None,
):
    """
    Predict heart disease risk using ML stacking ensemble.
    MedGemma ALWAYS explains WHY the prediction was made.

    The pipeline:
    1. ML Stacking Ensemble (8 models) → binary prediction + probability
    2. Risk classification using probability + clinical red flags
    3. MedGemma interprets the result and explains WHY
    4. Hallucination grading + quality scoring
    """
    global _model_loader, _interpretation_pipeline

    if _model_loader is None or not _model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Heart disease model not loaded. Check server logs.")

    start_time = time.perf_counter()

    try:
        # Step 1: ML Prediction
        df = build_feature_dataframe(input_data)
        prediction, probability = _model_loader.predict(df)
        risk = classify_risk(probability, input_data)
        needs_attention = risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)

        # Build structured test result analysis
        test_result_details = build_test_result_details(input_data)
        abnormal_count = sum(1 for r in test_result_details if r.status in ("Abnormal", "Critical"))
        high_risk_factors = [r.test_name for r in test_result_details if r.risk_contribution == "High"]

        response_data = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk.value,
            "confidence": round(abs(probability - 0.5) * 2, 4),
            "message": (
                f"{'Heart Disease Likely' if prediction == 1 else 'Heart Disease Unlikely'} "
                f"(probability: {probability:.1%}, risk: {risk.value}). "
                f"{abnormal_count} of {len(test_result_details)} test results are abnormal"
                f"{' — ' + ', '.join(high_risk_factors) + ' are high-risk contributors' if high_risk_factors else ''}. "
                + ("Immediate medical consultation recommended." if needs_attention else
                   "Continue regular check-ups.")
            ),
            "test_results": [r.dict() for r in test_result_details],
            "needs_medical_attention": needs_attention,
        }

        # Step 2: MedGemma explains WHY (ALWAYS — not optional)
        if _interpretation_pipeline:
            try:
                ensemble_votes = _model_loader.get_ensemble_votes(df)
                interpretation = await _interpretation_pipeline.interpret(
                    input_data=input_data,
                    prediction=prediction,
                    probability=probability,
                    risk_level=risk,
                    ensemble_votes=ensemble_votes,
                    user_id=user_id,
                )
                response_data.update(interpretation)
            except Exception as e:
                logger.error(f"MedGemma interpretation failed: {e}")
                response_data["clinical_interpretation"] = (
                    f"ML model predicted {'heart disease likely' if prediction == 1 else 'heart disease unlikely'} "
                    f"with {probability:.1%} probability. "
                    "Detailed explanation temporarily unavailable."
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response_data["processing_time_ms"] = round(elapsed_ms, 2)
        response_data["metadata"] = {
            "model_type": "StackingClassifier (8 base models)",
            "models": list(_model_loader._individual_pipelines.keys()),
            "medgemma_interpreted": True,
            "pipeline_version": "2.1.0",
        }

        # Persist prediction to database
        if user_id:
            try:
                from core.database.postgres_db import get_database
                db = await get_database()
                await db.store_prediction(
                    user_id=user_id,
                    prediction_id=str(uuid.uuid4()),
                    input_data=input_data.dict(),
                    prediction=prediction,
                    probability=round(probability, 4),
                    risk_level=risk.value,
                    confidence=round(abs(probability - 0.5) * 2, 4),
                    clinical_interpretation=response_data.get("clinical_interpretation", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to store prediction history: {e}")

        return PredictionResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/ensemble")
async def predict_ensemble_detail(input_data: HeartDiseaseInput):
    """
    Detailed ensemble breakdown — shows each of the 8 models' predictions.
    Useful for transparency and debugging.
    """
    global _model_loader

    if _model_loader is None or not _model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = build_feature_dataframe(input_data)
    prediction, probability = _model_loader.predict(df)
    ensemble_votes = _model_loader.get_ensemble_votes(df)
    risk = classify_risk(probability, input_data)

    if ensemble_votes:
        predictions = [v["prediction"] for v in ensemble_votes.values() if "prediction" in v]
        agreement = sum(p == prediction for p in predictions) / max(len(predictions), 1)
    else:
        agreement = 1.0

    return {
        "stacking_prediction": prediction,
        "stacking_probability": round(probability, 4),
        "risk_level": risk.value,
        "ensemble_votes": ensemble_votes,
        "model_agreement": round(agreement, 4),
        "total_models": len(ensemble_votes),
        "agreeing_models": sum(1 for v in ensemble_votes.values() if v.get("prediction") == prediction),
    }


@router.post("/predict/batch")
async def predict_batch(
    patients: List[HeartDiseaseInput],
    explain: bool = True,
):
    """
    Batch prediction for multiple patients.
    MedGemma explains each prediction by default.
    """
    global _model_loader

    if _model_loader is None or not _model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 patients per batch.")

    results = []
    for i, patient in enumerate(patients):
        try:
            df = build_feature_dataframe(patient)
            prediction, probability = _model_loader.predict(df)
            risk = classify_risk(probability, patient)

            result = {
                "index": i,
                "prediction": prediction,
                "probability": round(probability, 4),
                "risk_level": risk.value,
                "needs_medical_attention": risk in (RiskLevel.HIGH, RiskLevel.CRITICAL),
            }

            if explain and _interpretation_pipeline:
                try:
                    interpretation = await _interpretation_pipeline.interpret(
                        input_data=patient,
                        prediction=prediction,
                        probability=probability,
                        risk_level=risk,
                    )
                    result["clinical_interpretation"] = interpretation.get("clinical_interpretation", "")
                except Exception:
                    result["clinical_interpretation"] = "Explanation unavailable."

            results.append(result)
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return {
        "total": len(patients),
        "results": results,
        "high_risk_count": sum(1 for r in results if r.get("risk_level") in ("High", "Critical")),
    }


@router.post("/insight")
async def generate_insight(request: dict):
    """Generate a cardiovascular health insight using MedGemma."""
    query = request.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    global _interpretation_pipeline
    if _interpretation_pipeline:
        _interpretation_pipeline._lazy_init()

    if _interpretation_pipeline and _interpretation_pipeline._llm:
        try:
            prompt = f"""You are a cardiovascular health expert.
Provide a brief, evidence-based health insight for the following question.
Keep the response concise (2-3 paragraphs).

Question: {query}

Provide actionable, evidence-based advice with appropriate medical disclaimers."""

            response = await _interpretation_pipeline._llm.generate(
                prompt=prompt, content_type="medical",
            )
            return {"insight": response, "source": "MedGemma", "grounded": True}
        except Exception as e:
            logger.error(f"MedGemma insight generation failed: {e}")

    return {
        "insight": (
            f"Based on your query about '{query}': "
            "Maintaining a heart-healthy lifestyle includes regular exercise, "
            "a balanced diet rich in fruits, vegetables, and whole grains, "
            "managing stress, and regular check-ups. "
            "Always consult a qualified healthcare professional for personalized advice."
        ),
        "source": "fallback",
        "grounded": False,
    }


@router.get("/health", response_model=HealthCheckResponse)
def heart_health_check():
    """Heart disease prediction service health check."""
    global _model_loader, _interpretation_pipeline

    medgemma_available = False
    rag_available = False
    memory_available = False

    if _interpretation_pipeline and _interpretation_pipeline._initialized:
        medgemma_available = _interpretation_pipeline._llm is not None
        rag_available = _interpretation_pipeline._rag is not None
        memory_available = _interpretation_pipeline._memori is not None

    return HealthCheckResponse(
        status="healthy" if (_model_loader and _model_loader.is_loaded) else "degraded",
        model_loaded=_model_loader.is_loaded if _model_loader else False,
        medgemma_available=medgemma_available,
        rag_available=rag_available,
        memory_available=memory_available,
        numpy_version=np.__version__,
        pipeline_version="2.1.0",
    )
