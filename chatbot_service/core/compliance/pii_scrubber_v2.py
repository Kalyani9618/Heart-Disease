"""
Enhanced PII Scrubber with NER and Domain-Aware Detection

Uses Microsoft Presidio + spaCy medical NER for comprehensive
de-identification of medical narratives.
"""

import re
import logging
import os
import platform
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Get models directory path
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
SPACY_MODELS_DIR = MODELS_DIR / "spacy_models"

_RUNNING_ON_WINDOWS = platform.system().lower() == "windows"
_DISABLE_PRESIDIO_ON_WINDOWS = os.getenv("PII_SCRUBBER_DISABLE_PRESIDIO_ON_WINDOWS", "true").lower() == "true"

if _RUNNING_ON_WINDOWS and _DISABLE_PRESIDIO_ON_WINDOWS:
    PRESIDIO_AVAILABLE = False
    logging.warning("Presidio disabled on Windows (PII_SCRUBBER_DISABLE_PRESIDIO_ON_WINDOWS=true) to avoid native CUDA/CuPy crashes.")
    class OperatorConfig:
        def __init__(self, operator_name, params):
            pass
else:
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
        PRESIDIO_AVAILABLE = True
    except Exception:
        PRESIDIO_AVAILABLE = False
        logging.warning("Presidio not installed. Using regex-only fallback.")
        # Define dummy OperatorConfig to avoid NameError
        class OperatorConfig:
            def __init__(self, operator_name, params):
                pass

try:
    import spacy
    SPACY_AVAILABLE = True
except (ImportError, Exception) as e:
    SPACY_AVAILABLE = False
    logging.warning(f"spaCy not available: {e}. Using regex-only PII detection.")

logger = logging.getLogger(__name__)





class EnhancedPIIScrubber:
    """
    Advanced PII scrubber using NER + Regex + Domain rules.
    
    Detection methods (in order):
    1. Microsoft Presidio (NER-based) → HIGH confidence
    2. spaCy medical NER → Domain-aware
    3. Enhanced regex patterns → Fallback for known formats
    4. Custom medical rules → Context-aware
    """
    
    # MEDICAL WHITELIST - Comprehensive drug/medication names that should NOT be redacted
    # Expanded from ~20 terms to 150+ terms covering all major drug classes
    MEDICAL_WHITELIST = {
        # ACE Inhibitors
        "lisinopril", "enalapril", "ramipril", "perindopril", "quinapril",
        "trandolapril", "captopril", "benazepril", "fosinopril", "moexipril",
        
        # Angiotensin II Receptor Blockers (ARBs)
        "losartan", "valsartan", "irbesartan", "olmesartan", "candesartan",
        "telmisartan", "eprosartan", "azilsartan", "fimasartan",
        
        # Beta Blockers
        "metoprolol", "atenolol", "propranolol", "labetalol", "carvedilol",
        "bisoprolol", "nebivolol", "timolol", "nadolol", "pindolol",
        "acebutolol", "betaxolol", "esmolol", "levobunolol",
        
        # Calcium Channel Blockers
        "amlodipine", "diltiazem", "verapamil", "nifedipine", "felodipine",
        "isradipine", "nicardipine", "clevidipine", "lercanidipine",
        
        # Diuretics
        "hydrochlorothiazide", "furosemide", "bumetanide", "torsemide",
        "spironolactone", "amiloride", "chlorothiazide", "indapamide",
        "chlorthalidone", "methazolamide", "acetazolamide",
        
        # Statins (HMG-CoA Reductase Inhibitors)
        "atorvastatin", "simvastatin", "pravastatin", "rosuvastatin",
        "lovastatin", "fluvastatin", "pitavastatin", "cerivastatin",
        
        # Fibrates
        "fenofibrate", "gemfibrozil", "bezafibrate", "ciprofibrate",
        
        # Antiplatelets
        "aspirin", "clopidogrel", "ticlopidine", "prasugrel", "ticagrelor",
        "dipyridamole",
        
        # Anticoagulants
        "warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
        "heparin", "enoxaparin", "dalteparin", "tinzaparin", "fondaparinux",
        
        # Diabetes Medications
        "metformin", "glipizide", "glyburide", "rosiglitazone", "pioglitazone",
        "linagliptin", "sitagliptin", "saxagliptin", "alogliptin", "vildagliptin",
        "insulin", "glucagon", "acarbose", "miglitol", "repaglinide",
        "nateglinide", "liraglutide", "semaglutide", "dulaglutide",
        "empagliflozin", "canagliflozin", "dapagliflozin",
        
        # Antibiotics
        "amoxicillin", "penicillin", "cephalexin", "azithromycin", "doxycycline",
        "ciprofloxacin", "levofloxacin", "moxifloxacin", "trimethoprim",
        "sulfamethoxazole", "cephalosporin", "erythromycin", "clarithromycin",
        "metronidazole", "clindamycin", "vancomycin", "flucloxacillin",
        "ceftriaxone", "cefotaxime", "amikacin", "gentamicin",
        
        # Antivirals
        "acyclovir", "valacyclovir", "famciclovir", "oseltamivir", "zanamivir",
        "remdesivir", "molnupiravir", "paxlovid", "ribavirin",
        
        # Antifungals
        "fluconazole", "itraconazole", "voriconazole", "posaconazole",
        "amphotericin", "terbinafine", "miconazole", "ketoconazole",
        
        # Proton Pump Inhibitors
        "omeprazole", "lansoprazole", "pantoprazole", "esomeprazole",
        "rabeprazole", "dexlansoprazole",
        
        # H2 Blockers
        "ranitidine", "famotidine", "cimetidine", "nizatidine",
        
        # Anti-nausea
        "ondansetron", "granisetron", "metoclopramide", "prochlorperazine",
        "promethazine", "aprepitant",
        
        # Painkillers/NSAIDs
        "ibuprofen", "acetaminophen", "naproxen", "indomethacin",
        "meloxicam", "piroxicam", "tenoxicam", "ketoprofen", "diclofenac",
        "celecoxib", "rofecoxib", "tramadol", "codeine",
        
        # Corticosteroids
        "prednisone", "prednisolone", "dexamethasone", "hydrocortisone",
        "methylprednisolone", "triamcinolone", "betamethasone", "fluticasone",
        "beclomethasone", "budesonide", "mometasone",
        
        # Thyroid medications
        "levothyroxine", "liothyronine", "propylthiouracil", "methimazole",
        
        # Cardiovascular
        "digoxin", "digitaloxin", "milrinone", "dobutamine", "dopamine",
        "isoproterenol", "epinephrine", "norepinephrine", "phenylephrine",
        "nitroprusside", "nitroglycerin", "isosorbide",
        
        # Respiratory
        "albuterol", "salbutamol", "terbutaline", "salmeterol", "formoterol",
        "theophylline", "aminophylline", "ipratropium", "tiotropium",
        
        # Minerals/Electrolytes
        "sodium", "potassium", "magnesium", "calcium", "phosphate",
        "chloride", "bicarbonate", "sulfate",
        
        # Vitamins
        "vitamin", "thiamine", "niacin", "riboflavin", "biotin", "folate",
        "folic acid", "cobalamin", "cyanocobalamin", "retinol", "carotenoid",
        "tocopherol", "ascorbic acid", "cholecalciferol",
        
        # Supplements commonly in medical context
        "omega", "omega-3", "fish oil", "flaxseed", "ginger", "turmeric",
        "curcumin", "resveratrol", "quercetin", "probiotics",
        
        # Common OTC brand names
        "tylenol", "advil", "motrin", "aleve", "bayer", "mucinex",
        "robitussin", "dayquil", "nyquil", "sudafed", "tums", "antacid",
        "pepto", "imodium",
        
        # Medical procedures/devices (often mistaken for names)
        "stent", "bypass", "pacemaker", "defibrillator", "catheter",
        "dialysis", "transplant", "prosthesis", "implant", "prosthetic",
        
        # Conditions/Diseases (should NOT be redacted as names)
        "hypertension", "hyperlipidemia", "diabetes", "heart failure",
        "coronary artery disease", "peripheral vascular disease", "pvd",
        "arrhythmia", "atrial fibrillation", "afib", "stroke", "tia",
        "myocardial infarction", "mi", "acute mi", "stemi", "nstemi",
        "unstable angina", "angina pectoris", "asthma", "copd",
        "pneumonia", "bronchitis", "pneumonitis", "pulmonary embolism",
        "deep vein thrombosis", "dvt", "thrombosis", "thromboembolism",
        "ischemia", "reperfusion", "infarction", "necrosis", "stenosis",
        "aneurysm", "atherosclerosis", "arteriosclerosis", "cardiomyopathy",
        "myocarditis", "pericarditis", "endocarditis", "vasculitis",
        "arthritis", "rheumatoid arthritis", "osteoarthritis", "gout",
        "lupus", "sclerosis", "multiple sclerosis", "alzheimer",
        "parkinson", "dementia", "encephalitis", "meningitis",
        "hepatitis", "cirrhosis", "fatty liver", "pancreatitis",
        "gastritis", "ulcer", "gerd", "crohn's disease", "ulcerative colitis",
        "ibs", "irritable bowel syndrome", "celiac disease", "renal failure",
        "kidney disease", "ckd", "esrd", "glomerulonephritis", "nephrotic",
        "prostate cancer", "breast cancer", "lung cancer", "colorectal cancer",
        "leukemia", "lymphoma", "melanoma", "carcinoma", "adenocarcinoma",
        "sarcoma", "myeloma", "squamous cell", "basal cell",
        
        # Symptoms/Signs (should NOT be redacted)
        "chest pain", "shortness of breath", "dyspnea", "sob", "fatigue",
        "palpitations", "dizziness", "vertigo", "syncope", "edema", "swelling",
        "tremor", "nausea", "vomiting", "diarrhea", "constipation",
        "heartburn", "dysphagia", "cough", "wheezing", "stridor",
        "fever", "chills", "sweats", "malaise", "weight loss",
        "weight gain", "hair loss", "alopecia", "rash", "pruritis",
        "urticaria", "petechiae", "ecchymosis", "bruising", "bleeding",
        "hematuria", "hemoptysis", "hematemesis", "melena",
        
        # Lab tests/values (should NOT be redacted)
        "ldl", "hdl", "triglyceride", "glucose", "creatinine", "gfr",
        "bun", "albumin", "protein", "ast", "alt", "alkaline phosphatase",
        "bilirubin", "troponin", "bnp", "pro-bnp", "crp", "esr",
        "hemoglobin", "hematocrit", "wbc", "rbc", "platelet", "inr", "pt",
        "ptt", "aptt", "d-dimer", "lactate", "ph", "po2", "pco2",
        "a1c", "fasting glucose", "glucose tolerance",
        
        # Imaging/Diagnostic (should NOT be redacted)
        "ct", "mri", "xray", "x-ray", "ultrasound", "echocardiogram", "echo",
        "ekg", "ecg", "eeg", "pet", "spect", "angiogram", "fluoroscopy",
        "mammogram", "colonoscopy", "endoscopy", "biopsy", "aspiration",
        
        # Physiological measurements
        "ejection fraction", "lvef", "bp", "blood pressure", "hr", "heart rate",
        "rr", "respiratory rate", "spo2", "oxygen saturation", "o2",
        "cardiac output", "stroke volume", "heart rate variability", "hrv",
        
        # Routes of administration (should NOT be redacted)
        "oral", "iv", "intravascular", "im", "intramuscular", "sc", "subcutaneous",
        "sublingual", "transdermal", "patch", "inhaler", "aerosol", "nebulized",
        "intraperitoneal", "intrathecal", "intraarterial", "intraocular",
        "intramuscular injection", "intravenous injection", "vaccine", "vaccination",
        
        # Dosage units (should NOT be redacted)
        "mg", "mcg", "microgram", "gram", "kg", "lb", "ml", "cc", "liter",
        "unit", "miu", "percent", "%", "ppm", "mmol", "meq", "iu",
        
        # Dosing frequency (should NOT be redacted as names)
        "qd", "od", "bid", "tid", "qid", "qh", "q2h", "q4h", "q6h",
        "q8h", "q12h", "qhs", "prn", "as needed", "daily", "twice daily",
        "three times daily", "four times daily", "every morning", "every night",
        "before bed", "with meals", "on an empty stomach",
        
        # Additional multi-word medical phrases that should NOT be redacted
        "type 1 diabetes", "type 2 diabetes", "type 1", "type 2",
        "chest pain", "acute pain", "chronic pain", "severe pain",
        "type 1 diabetes mellitus", "type 2 diabetes mellitus",
        "heart failure", "congestive heart failure", "acute heart failure",
        "renal failure", "acute renal failure", "chronic renal failure",
    }
    
    # Common English words that should never be redacted as names
    # Prevents false positives like "Heart Disease" or "Blood Pressure"
    # Expanded to cover general vocabulary, medical terms, and words
    # commonly found in LLM-generated health advisory output
    COMMON_WORDS = {
        # Articles / Prepositions / Conjunctions / Pronouns
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "must", "get", "got", "say",
        "said", "take", "takes", "took", "taken", "its", "it", "you", "your",
        "my", "our", "their", "his", "her", "we", "they", "them", "him",
        "about", "what", "when", "where", "why", "how", "which", "who", "whom",
        "there", "these", "those", "that", "this", "other", "such", "than",
        "into", "over", "under", "between", "through", "during", "after",
        "before", "above", "below", "each", "every", "both", "all", "any",
        "some", "most", "more", "less", "also", "very", "just", "only",
        "not", "nor", "yet", "so", "if", "then", "else", "while", "until",
        "because", "since", "although", "though", "whether", "either", "neither",

        # Common verbs / adjectives / adverbs
        "make", "made", "know", "known", "see", "seen", "need", "help",
        "work", "keep", "start", "show", "try", "call", "come", "give",
        "given", "find", "think", "tell", "become", "leave", "feel", "put",
        "seem", "include", "including", "following", "based", "related",
        "associated", "recommended", "required", "important", "significant",
        "possible", "likely", "unlikely", "normal", "abnormal", "elevated",
        "high", "low", "good", "bad", "new", "old", "long", "short",
        "large", "small", "great", "first", "last", "next", "early", "late",
        "right", "left", "different", "same", "specific", "general",
        "overall", "total", "major", "minor", "primary", "secondary",
        "current", "recent", "regular", "daily", "weekly", "monthly",
        "certain", "available", "positive", "negative", "active", "common",

        # Medical / Health domain common words
        "patient", "patients", "doctor", "doctors", "nurse", "nurses",
        "medicine", "medication", "medications", "drug", "drugs",
        "treatment", "treatments", "therapy", "disease", "diseases",
        "condition", "conditions", "symptom", "symptoms", "side", "effect",
        "effects", "use", "used", "uses", "dose", "dosage", "level", "levels",
        "risk", "risks", "factor", "factors", "rate", "rates", "test", "tests",
        "result", "results", "value", "values", "range", "ranges",
        "blood", "heart", "chest", "body", "brain", "lung", "liver", "kidney",
        "pressure", "sugar", "pain", "health", "healthy", "care",
        "clinical", "medical", "cardiac", "cardiovascular", "coronary",
        "arterial", "venous", "vascular", "pulmonary", "respiratory",
        "resting", "fasting", "exercise", "physical", "mental", "emotional",
        "maximum", "minimum", "average", "mean", "median",
        "prediction", "assessment", "evaluation", "analysis", "diagnosis",
        "prognosis", "interpretation", "recommendation", "recommendations",
        "management", "prevention", "monitoring", "screening", "detection",
        "lifestyle", "dietary", "nutritional", "behavioral", "genetic",
        "age", "sex", "male", "female", "weight", "height", "mass",
        "index", "score", "stage", "grade", "class", "category",
        "summary", "report", "note", "notes", "plan", "history",
        "information", "data", "evidence", "research", "study", "studies",
        "source", "sources", "reference", "references", "guideline", "guidelines",
        "standard", "protocol", "procedure", "intervention", "approach",
        "target", "goal", "aim", "outcome", "outcomes", "benefit", "benefits",
        "reduces", "reduce", "increase", "increases", "decrease", "decreases",
        "improve", "improves", "maintain", "prevent", "manage", "control",
        "lower", "raise", "check", "checks", "measure", "measures",
        "intake", "output", "consumption", "restriction", "modification",
        "changes", "change", "step", "steps", "action", "actions",
        "angina", "stroke", "attack", "failure", "arrest",
        "hypertrophy", "stenosis", "artery", "arteries", "vein", "veins",
        "muscle", "tissue", "cell", "cells", "organ", "system",
        "function", "structure", "volume", "flow", "rhythm", "beat",
        "reading", "readings", "measurement", "measurements",
        "ecg", "ekg", "echo", "stress", "slope", "segment", "wave",
        "depression", "elevation", "interval", "duration", "amplitude",
        "abnormality", "finding", "findings", "indicator", "indicators",
        "contributor", "contributors", "predictor", "predictors",
        "protective", "harmful", "beneficial", "adverse", "chronic",
        "acute", "severe", "mild", "moderate", "borderline",
        "diet", "nutrition", "sodium", "potassium", "fiber",
        "saturated", "trans", "unsaturated", "fatty", "lean",
        "whole", "grain", "grains", "fruit", "fruits", "vegetable", "vegetables",
        "smoking", "alcohol", "caffeine", "cessation", "abstinence",
        "walking", "running", "swimming", "cycling", "aerobic",
        "strength", "flexibility", "endurance", "recovery", "rest",
        "sleep", "relaxation", "meditation", "yoga",
        "professional", "specialist", "cardiologist", "physician",
        "consultant", "provider", "healthcare", "hospital", "clinic",
        "emergency", "urgent", "routine", "follow", "visit",
        "appointment", "consultation", "referral", "triage",
        "disclaimer", "warning", "caution", "notice", "advisory",
        "general", "educational", "informational", "personalized",
        "individual", "personal", "comprehensive", "detailed", "brief",
        "actionable", "practical", "effective", "safe", "unsafe",
        "approved", "certified", "verified", "validated", "trusted",
        "continue", "continued", "consider", "consult", "discuss",
        "seek", "obtain", "review", "evaluate", "assess",
        "probability", "confidence", "likelihood", "percentage",
        "unlikely", "likely", "possible", "probable", "definite",
        "Contributors", "section", "paragraph", "table", "list",
        "item", "point", "number", "figure", "image",
        "thinq", "cardio",
    }
    
    # Regex patterns enhanced with medical context
    ENHANCED_PATTERNS: List[Tuple[str, str]] = [
        # SSN - all variants
        (r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b", "[SSN_REDACTED]"),
        (r"\b\d{9}\b", "[SSN_REDACTED]"),
        
        # Phone numbers - comprehensive
        (r"\b\d{3}[\s\-.]?\d{3}[\s\-.]?\d{4}\b", "[PHONE_REDACTED]"),
        (r"\b\(\d{3}\)\s?\d{3}[\s\-.]?\d{4}\b", "[PHONE_REDACTED]"),
        (r"\b\+?1?\s?\d{10}\b", "[PHONE_REDACTED]"),
        
        # Email addresses
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
        
        # Credit card numbers
        (r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", "[CC_REDACTED]"),
        (r"\b\d{16}\b", "[CC_REDACTED]"),
        
        # Medical Record Numbers - enhanced
        (r"\b(?:MRN|MRNO|Medical Record)\s*:?\s*([A-Z]{1,3}\d{6,10})\b", "[MRN_REDACTED]"),
        (r"\bMR-\d{6,10}\b", "[MRN_REDACTED]"),
        
        # Insurance IDs - enhanced
        (r"\b(?:Insurance ID|Policy No|Member ID)\s*:?\s*([A-Z0-9]{8,12})\b", "[INSURANCE_ID_REDACTED]"),
        
        # Patient names with titles - IMPROVED
        # Handles: "Mr. John Doe", "Dr. Jane Smith", "Mrs. M. Johnson"
        # Using (?i) inline flag for title matching only, so case-insensitive
        # matching applies to the title prefix but the name parts still
        # require capitalized first letter via [A-Z].
        (r"(?i:\b(?:Mr|Mrs|Ms|Dr|Prof)\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?\b", "[NAME_REDACTED]"),
        # First-last name without title — CONSERVATIVE
        # Only matches exactly two capitalized words (3+ chars each).
        # Must NOT be used with re.IGNORECASE (see _regex_scrub).
        # The COMMON_WORDS / MEDICAL_WHITELIST checks in _regex_scrub
        # provide additional defence against false positives.
        (r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b", "[NAME_REDACTED]"),
        
        # Hospital/Clinic names (context: "admitted to <NAME>")
        (r"\b(?:admitted to|presented to|at|via)\s+(?:the\s+)?([A-Z][A-Za-z\s&]+(?:Hospital|Clinic|Center|Medical))\b", "[HOSPITAL_REDACTED]"),
        
        # Dates with patient context (Month/Day/Year patterns)
        (r"\b(?:DOB|Date of Birth|Birthday)\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4})\b", "[DOB_REDACTED]"),
        
        # Drug-specific dosages (avoid over-redaction)
        # This is actually NOT PII, but prevents mixing: "10mg QID of aspirin by John Doe"
        # We keep the drug info, redact the person
    ]
    
    def __init__(self, use_presidio: bool = True, use_scispacy: bool = True, auto_download: bool = False):
        """
        Initialize enhanced scrubber.
        
        Args:
            use_presidio: Whether to use Presidio NER (requires spaCy model)
            use_scispacy: Whether to use spaCy for additional NER
            auto_download: Whether to auto-download missing models (default: False)
        """
        running_on_windows = platform.system().lower() == "windows"
        disable_presidio_on_windows = os.getenv("PII_SCRUBBER_DISABLE_PRESIDIO_ON_WINDOWS", "true").lower() == "true"

        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE
        self.use_scispacy = use_scispacy and SPACY_AVAILABLE
        self.auto_download = auto_download
        # Keep PII scrubbing on CPU by default to avoid mixed cpu/cuda tensor failures.
        self.force_cpu = os.getenv("PII_SCRUBBER_FORCE_CPU", "true").lower() == "true"
        self._presidio_model_name = os.getenv("PII_SCRUBBER_SPACY_MODEL", "en_core_web_sm")

        if self.force_cpu:
            # Avoid CuPy/Thinc CUDA code paths that can trigger access violations.
            os.environ.setdefault("THINC_CPU_ONLY", "1")
            if SPACY_AVAILABLE:
                try:
                    spacy.require_cpu()
                except Exception:
                    pass

        if running_on_windows and disable_presidio_on_windows and self.use_presidio:
            logger.warning("Windows detected: disabling Presidio engine to avoid CUDA/CuPy access violations. Set PII_SCRUBBER_DISABLE_PRESIDIO_ON_WINDOWS=false to override.")
            self.use_presidio = False
        
        self.analyzer = None
        self.anonymizer = None
        self.nlp = None
        
        # Use a dedicated CPU spaCy model for PII by default.
        # Sharing the main transformer pipeline can trigger cpu/cuda mismatches.
        if not self.force_cpu:
            try:
                from core.services.spacy_service import get_spacy_service
                self.spacy_service = get_spacy_service()
                self.nlp = self.spacy_service.nlp
                logger.info("✅ SpaCyService loaded for PII scrubbing")
            except Exception as e:
                logger.warning(f"Failed to load SpaCyService: {e}")
                self.nlp = None

        if self.force_cpu and self.use_scispacy and self.nlp is None:
            try:
                for model_name in [self._presidio_model_name, "en_core_web_md", "en_core_web_sm"]:
                    try:
                        self.nlp = spacy.load(model_name)
                        logger.info(f"✅ CPU spaCy model loaded for PII scrubbing: {model_name}")
                        break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Failed to load CPU spaCy model for PII scrubber: {e}")
                self.nlp = None

        if self.use_scispacy and self.nlp is None:
            self.use_scispacy = False

        # Initialize Presidio NLP engine.
        # In CPU mode we should still attempt Presidio even if shared SpaCyService
        # is unavailable, because Presidio can create its own spaCy engine.
        if self.use_presidio:
            try:
                from presidio_analyzer.nlp_engine import NlpEngineProvider, NerModelConfiguration
                
                # Configure NER model to ignore non-PII entity types
                # This prevents warnings for unmapped entity types like PRODUCT, CARDINAL, etc.
                ner_model_config = NerModelConfiguration(
                    labels_to_ignore=[
                        "PRODUCT", "PERCENT", "CARDINAL", "QUANTITY", "ORDINAL", 
                        "LANGUAGE", "MONEY", "EVENT", "FAC", "GPE", "LAW", 
                        "NORP", "ORG", "WORK_OF_ART"
                    ]
                )
                
                # Use NlpEngineProvider with NerModelConfiguration embedded in the model config
                # NOTE: ner_model_configuration goes INSIDE the model dict, not at the top level
                presidio_model_name = self._presidio_model_name if self.force_cpu else "en_core_web_trf"
                nlp_configuration = {
                    "nlp_engine_name": "spacy",
                    "models": [{
                        "lang_code": "en", 
                        "model_name": presidio_model_name,
                        "ner_model_configuration": ner_model_config
                    }],
                }
                nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
                
                # Initialize AnalyzerEngine
                import logging as logging_module
                presidio_logger = logging_module.getLogger("presidio_analyzer")
                presidio_logger.setLevel(logging_module.WARNING)  # Only show warnings+, suppress debug messages
                
                # NOTE: NerModelConfiguration is passed via NlpEngineProvider, not AnalyzerEngine
                # The NLP engine already has the labels_to_ignore configured
                self.analyzer = AnalyzerEngine(
                    nlp_engine=nlp_engine,
                    supported_languages=["en"],
                    default_score_threshold=0.5
                )
                
                # Configure which entity types to ignore in medical documents
                # These are common entity types that spaCy detects but are NOT PII
                # Suppresses "not mapped to a Presidio entity" warnings
                labels_to_ignore = ["PRODUCT", "PERCENT", "CARDINAL", "QUANTITY", "ORDINAL", "LANGUAGE"]
                
                # Suppress warnings for these entity types
                import logging as logging_module
                presidio_analyzer_logger = logging_module.getLogger("presidio_analyzer")
                presidio_analyzer_logger.addFilter(
                    lambda record: not any(label in record.getMessage() for label in labels_to_ignore)
                )
                
                self.anonymizer = AnonymizerEngine()
                logger.info(f"✅ Presidio engine initialized with NerModelConfiguration (PRODUCT/CARDINAL/QUANTITY ignored)")
                logger.info(f"   Drug names and numbers will NOT be redacted")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio with NlpEngineProvider: {e}. Trying standard init...")
                try:
                    # Fallback to standard initialization
                    model_name = self._presidio_model_name if self.force_cpu else "en_core_web_trf"
                    if not spacy.util.is_package(model_name):
                        model_name = "en_core_web_md" if self.force_cpu else "en_core_web_lg"
                        if not spacy.util.is_package(model_name):
                             model_name = "en_core_web_sm"
                    
                    nlp_configuration = {
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": model_name}]
                    }
                    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
                    self.analyzer = AnalyzerEngine(
                        nlp_engine=nlp_engine,
                        supported_languages=["en"]
                    )
                    self.anonymizer = AnonymizerEngine()
                    logger.info(f"✅ Presidio engine initialized with model: {model_name} (fallback)")
                except Exception as e2:
                    logger.warning(f"Presidio fallback failed: {e2}. Using regex only.")
                    self.use_presidio = False
        if self.use_presidio and self.analyzer is None:
            logger.warning("Presidio enabled but no compatible spaCy model is available. Disabling Presidio.")
            self.use_presidio = False
    
    def scrub(self, text: str, language: str = "en") -> str:
        """
        Scrub PII from text using multi-pass approach.
        
        Args:
            text: Text to scrub
            language: Language code (default: "en")
        
        Returns:
            Scrubbed text with PII redacted
        """
        if not text:
            return text
        
        scrubbed = text
        
        # Pass 1: Presidio NER-based detection (highest confidence)
        if self.use_presidio:
            scrubbed = self._presidio_scrub(scrubbed, language)
        
        # Pass 2: spaCy medical NER
        if self.use_scispacy:
            scrubbed = self._spacy_scrub(scrubbed)
        
        # Pass 3: Enhanced regex patterns
        scrubbed = self._regex_scrub(scrubbed)
        
        # Pass 4: Custom medical rules
        scrubbed = self._custom_rules_scrub(scrubbed)
        
        if scrubbed != text:
            logger.debug(f"PII detected and scrubbed: {len(text) - len(scrubbed)} chars removed")
        
        return scrubbed
    
    def _presidio_scrub(self, text: str, language: str) -> str:
        """
        Use Presidio for NER-based PII detection.
        
        ENHANCED: 
        - Check medical whitelist BEFORE redacting
        - Verify detected text isn't a medical term
        - Raise confidence threshold for PERSON entity
        - Skip medical conditions and disease names
        """
        try:
            # Entity types to detect
            pii_entities = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME"]
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                presidio_logger = logging.getLogger("presidio_analyzer")
                old_level = presidio_logger.level
                presidio_logger.setLevel(logging.ERROR)
                
                try:
                    results = self.analyzer.analyze(
                        text=text,
                        language=language,
                        entities=pii_entities
                    )
                    
                    # Labels to ignore - these are not PII and should not be redacted
                    # PRODUCT is often triggered on drug names which we need to preserve
                    labels_to_ignore = {
                        "CARDINAL", "ID_NUMBER", "WORK_OF_ART", "QUANTITY", 
                        "ORDINAL", "LANGUAGE", "PRODUCT", "PERCENT", "MONEY",
                        "EVENT", "FAC", "GPE", "LAW", "NORP", "ORG"
                    }
                    filtered_results = []
                    
                    for r in results:
                        # Skip non-PII entity types
                        if r.entity_type in labels_to_ignore:
                            continue
                        
                        # ✅ NEW: Check medical whitelist BEFORE redacting
                        detected_text = text[r.start:r.end].lower()
                        
                        # Check if it's a drug, symptom, or medical term
                        if self._is_medical_term(detected_text):
                            logger.debug(f"Skipping medical term: '{detected_text}' (detected as {r.entity_type})")
                            continue
                        
                        # ✅ NEW: Check if it looks like a medical condition/diagnosis
                        if self._looks_like_medical_condition(detected_text):
                            logger.debug(f"Skipping medical condition: '{detected_text}'")
                            continue
                        
                        # ✅ NEW: Boost confidence threshold for PERSON entity
                        # Only redact person names if Presidio is VERY confident (>0.85)
                        # This prevents weak matches on drug names, medical terms, etc.
                        if r.entity_type == "PERSON" and r.score < 0.85:
                            logger.debug(f"Low confidence PERSON: '{detected_text}' (score: {r.score}), skipping")
                            continue
                        
                        # ✅ NEW: Additional check for common false positives
                        # Drug names often end in -opril, -sartan, -statin, etc.
                        if r.entity_type == "PERSON" and self._is_drug_name_pattern(detected_text):
                            logger.debug(f"Drug name pattern detected: '{detected_text}', skipping")
                            continue
                        
                        # Passed all checks, add to redaction list
                        filtered_results.append(r)
                    
                    if filtered_results:
                        scrubbed = self.anonymizer.anonymize(
                            text=text,
                            analyzer_results=filtered_results,
                            operators=self._get_anonymizers()
                        )
                        return scrubbed.text
                finally:
                    presidio_logger.setLevel(old_level)
                    
        except Exception as e:
            logger.warning(f"Presidio scrubbing failed: {e}")
        
        return text
    
    def _spacy_scrub(self, text: str) -> str:
        """Use spaCy medical NER for domain-aware scrubbing."""
        try:
            doc = self.nlp(text)
            
            # Identify protected entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:  # Person, Organization, Location
                    text = text.replace(ent.text, "[PII_REDACTED]")
            
            return text
        except Exception as e:
            logger.warning(f"spaCy scrubbing failed: {e}")
            # Prevent repeated failures in hot paths.
            self.use_scispacy = False
        
        return text
    
    def _regex_scrub(self, text: str) -> str:
        """
        Apply enhanced regex patterns with medical whitelist.
        
        Prevents over-redaction of drug names and medical terms that look like names.
        """
        scrubbed = text
        
        for pattern, replacement in self.ENHANCED_PATTERNS:
            try:
                # Special handling for name patterns (lines 95-97 in ENHANCED_PATTERNS)
                # These match "Firstname Lastname" but we need to exclude medical terms
                if replacement == "[NAME_REDACTED]":
                    # Find all matches first
                    # CRITICAL: Do NOT use re.IGNORECASE here!
                    # The name patterns rely on capitalization ([A-Z][a-z]) to
                    # distinguish "John Smith" from "heart disease". With
                    # IGNORECASE, [A-Z] matches lowercase too, turning the
                    # pattern into "any two 3+ letter words" — catastrophic.
                    import re as re_module
                    matches = list(re_module.finditer(pattern, scrubbed))
                    
                    # Process matches in reverse order to avoid position shifts
                    for match in reversed(matches):
                        matched_text = match.group(0)
                        matched_lower = matched_text.lower()
                        words = matched_lower.split()
                        
                        # Detect if this is a titled name (Mr./Mrs./Dr./etc.)
                        # Titles are a very strong signal of a real person name,
                        # so titled matches bypass the context/common-word checks.
                        _title_prefixes = ("mr.", "mr ", "mrs.", "mrs ", "ms.", "ms ",
                                           "dr.", "dr ", "prof.", "prof ")
                        is_titled = any(matched_lower.startswith(p) for p in _title_prefixes)
                        
                        # Skip if any word is a drug name (whitelisted medical term)
                        if any(word in self.MEDICAL_WHITELIST for word in words):
                            logger.debug(f"Skipping match with whitelisted drug: '{matched_text}'")
                            continue
                        
                        # ✅ NEW: Skip if matched text is a multi-word medical phrase
                        if matched_lower in self.MEDICAL_WHITELIST:
                            logger.debug(f"Skipping whitelisted medical phrase: '{matched_text}'")
                            continue
                        
                        # For titled names, we trust the title as strong PII signal
                        # and skip context/common-word heuristics that would wrongly
                        # protect "Mr. Robert Williams" near medical words.
                        if is_titled:
                            # Titled names → redact
                            scrubbed = scrubbed[:match.start()] + replacement + scrubbed[match.end():]
                            continue
                        
                        # ✅ NEW: Check if this is part of a medical phrase using context
                        # Get surrounding words to check for medical context
                        before_match = scrubbed[:match.start()].split()[-3:] if match.start() > 0 else []
                        after_match = scrubbed[match.end():].split()[:3] if match.end() < len(scrubbed) else []
                        context = before_match + words + after_match
                        context_str = " ".join(context).lower()
                        
                        # Check if context contains medical terms
                        medical_context_indicators = [
                            "type", "diabetes", "pain", "chest", "heart", "failure",
                            "renal", "acute", "chronic", "severe", "mild", "moderate",
                            "blood", "pressure", "risk", "factor", "rate", "level",
                            "disease", "health", "cardiac", "medical", "clinical",
                            "exercise", "resting", "fasting", "angina", "stroke",
                            "cholesterol", "ecg", "ekg", "prediction", "assessment",
                            "lifestyle", "dietary", "management", "treatment",
                            "recommendation", "monitoring", "screening", "test",
                            "result", "normal", "abnormal", "elevated", "high",
                            "low", "protective", "contributor", "indicator",
                            "slope", "segment", "depression", "hypertrophy",
                            "artery", "vein", "muscle", "tissue", "cell",
                        ]
                        if any(indicator in context_str for indicator in medical_context_indicators):
                            logger.debug(f"Skipping medical context match: '{matched_text}'")
                            continue
                        
                        # Skip if ANY word is a common English word
                        # In medical text, name-like patterns ("Heart Disease",
                        # "Blood Pressure") almost always contain common vocabulary.
                        # Real person names ("John Smith") rarely do.
                        if any(word in self.COMMON_WORDS for word in words):
                            logger.debug(f"Skipping match with common word(s): '{matched_text}'")
                            continue
                        
                        # Not whitelisted, apply redaction
                        scrubbed = scrubbed[:match.start()] + replacement + scrubbed[match.end():]
                else:
                    # Non-name patterns, apply normally
                    scrubbed = re.sub(pattern, replacement, scrubbed, flags=re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        return scrubbed
    
    def _custom_rules_scrub(self, text: str) -> str:
        """
        Apply custom medical domain rules with whitelist protection.
        
        Prevents redacting drug names in medical context.
        """
        
        # Rule 1: "Patient: <Name>" pattern - with whitelist check
        matches = list(re.finditer(r"Patient\s*:\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text))
        for match in reversed(matches):
            matched_text = match.group(0)
            # Extract the name part after "Patient: "
            name_part = re.search(r"Patient\s*:\s*(.+)$", matched_text).group(1)
            if name_part.lower() not in self.MEDICAL_WHITELIST:
                text = text[:match.start()] + "Patient: [NAME_REDACTED]" + text[match.end():]
        
        # Rule 2: "Dr. <Name>" pattern - with whitelist check
        matches = list(re.finditer(r"Dr\.\s+[A-Z][a-z]+", text))
        for match in reversed(matches):
            matched_text = match.group(0)
            # Extract the name part after "Dr. "
            name_part = matched_text.replace("Dr. ", "")
            if name_part.lower() not in self.MEDICAL_WHITELIST:
                text = text[:match.start()] + "Dr. [NAME_REDACTED]" + text[match.end():]
        
        # Rule 3: Age if combined with name "John (45 yo)"
        text = re.sub(r"\([0-9]{1,3}\s+(?:yo|year old|years old)\)", "[AGE_REDACTED]", text)
        
        # Rule 4: Admission numbers
        text = re.sub(r"\bAdmission#\s*:\s*(\d{6,})", "Admission#: [ADMISSION_REDACTED]", text)
        
        return text
    
    def _is_medical_term(self, term: str) -> bool:
        """
        Check if term is a known medical/pharmaceutical term.
        
        Includes: drugs, symptoms, conditions, medical abbreviations.
        
        Args:
            term: Term to check (e.g., "Lisinopril", "hypertension")
            
        Returns:
            True if it's a medical term, False otherwise
        """
        # Normalize
        normalized = term.lower().strip()
        
        # Check exact matches in whitelist (most common case)
        if normalized in self.MEDICAL_WHITELIST:
            return True
        
        # Check if it contains medical keywords
        medical_keywords = {
            # Common suffixes of drug names
            "-opril", "-sartan", "-statin", "-zole", "-mycin", "-cillin",
            "-dil", "-pine", "-ide", "-ine", "-ase", "-itis",
            
            # Common medical abbreviations
            "hypertension", "htn", "diabetes", "dm", "cad", "pvd", "copd",
            "gerd", "ckd", "esrd", "afib", "mi", "stemi", "nstemi",
            "dvt", "pe", "gfr", "bun", "ast", "alt", "inr", "pt",
            "ptt", "aptt", "crp", "esr", "wbc", "rbc", "ldl", "hdl",
        }
        
        # Check if normalized term is in keywords
        if normalized in medical_keywords:
            return True
        
        # Check if term ends with known drug suffixes
        drug_suffixes = [
            "opril", "sartan", "statin", "zole", "mycin", "cillin",
            "dil", "pine", "azole", "ide", "ine", "ase", "itis",
            "ose", "ol", "oxin", "one", "ate", "ium", "um",
        ]
        for suffix in drug_suffixes:
            if normalized.endswith(suffix):
                logger.debug(f"Recognized drug pattern: {normalized} (ends with {suffix})")
                return True
        
        return False
    
    def _looks_like_medical_condition(self, text: str) -> bool:
        """
        Detect if text is a medical condition/diagnosis name.
        
        Returns True if text contains medical keywords indicating diagnosis
        rather than a person's name.
        
        Examples:
        - "Heart Failure" → True (condition, not name)
        - "Type 2 Diabetes" → True (diagnosis, not name)
        - "Atrial Fibrillation" → True (condition, not name)
        - "John Smith" → False (looks like person name)
        
        Args:
            text: Text to check
            
        Returns:
            True if it looks like a medical condition, False otherwise
        """
        text_lower = text.lower()
        
        # Medical condition keywords
        medical_indicators = [
            "syndrome", "disease", "disorder", "failure", "attack",
            "infarction", "fibrillation", "itis", "osis", "pathy",
            "necrosis", "stenosis", "thrombosis", "hemorrhage",
            "edema", "hypertension", "hypotension", "tachycardia",
            "bradycardia", "arrhythmia", "arthritis", "dermatitis",
            "hepatitis", "nephritis", "pneumonia", "bronchitis",
            "gastritis", "colitis", "enteritis", "meningitis",
        ]
        
        for indicator in medical_indicators:
            if indicator in text_lower:
                return True
        
        # Check for medical adjective patterns: "Type 2 Diabetes", "Stage 4 Heart Failure"
        if re.search(r"(?:Type|Stage|Grade|Class|Level)\s+[\d\w]+", text, re.IGNORECASE):
            return True
        
        # Check for medical descriptors: "Acute X", "Chronic X", "Acute-on-Chronic X"
        if re.search(r"(?:Acute|Chronic|Subacute|Acute-on-Chronic|Severe|Mild|Moderate)\s+\w+", text, re.IGNORECASE):
            return True
        
        return False
    
    def _is_drug_name_pattern(self, text: str) -> bool:
        """
        Check if text matches known drug naming patterns.
        
        Uses common pharmaceutical naming conventions to identify drugs
        that Presidio might mistake for person names.
        
        Examples:
        - "Lisinopril" → True (ends in -opril, common ACE inhibitor suffix)
        - "Atorvastatin" → True (ends in -statin, common statin suffix)
        - "Amoxicillin" → True (ends in -cillin, common antibiotic suffix)
        - "John Smith" → False (doesn't match any drug pattern)
        
        Args:
            text: Text to check
            
        Returns:
            True if it matches a drug naming pattern, False otherwise
        """
        text_lower = text.lower()
        
        # Common pharmaceutical name endings (suffixes from INN naming conventions)
        pharma_suffixes = {
            # ACE Inhibitors
            "opril": 0.9,  # lisinopril, enalapril, etc.
            
            # ARBs (Angiotensin Receptor Blockers)
            "sartan": 0.9,  # losartan, valsartan, etc.
            
            # Statins
            "statin": 0.95,  # atorvastatin, simvastatin, etc.
            
            # Antifungals
            "azole": 0.85,  # fluconazole, itraconazole, etc.
            
            # Antibiotics
            "mycin": 0.85,  # azithromycin, erythromycin, etc.
            "cillin": 0.9,  # penicillin, amoxicillin, etc.
            
            # Calcium Channel Blockers
            "dipine": 0.9,  # amlodipine, nifedipine, etc.
            
            # Beta Blockers
            "olol": 0.85,  # propranolol, metoprolol, etc.
            
            # Other drug classes
            "oxin": 0.85,  # digoxin, digitoxin
            "one": 0.7,  # prednisolone, methylprednisolone (but also common names)
            "ine": 0.6,  # many drugs but also common names (caffeine, nicotine)
        }
        
        # Check against pharma suffixes
        for suffix, confidence in pharma_suffixes.items():
            if text_lower.endswith(suffix) and len(text) > len(suffix) + 2:
                # Only match if confidence is high and text is reasonable length
                logger.debug(f"Drug suffix match: {text_lower} → {suffix} (confidence: {confidence})")
                return True
        
        # Check for Roman numerals (often used in drug names: e.g., "Paxlovid", "Type II")
        if re.search(r"\b[IVX]{1,3}\b", text):
            # Likely a drug name or diagnosis with Roman numeral
            return True
        
        # Check for specific drug name patterns from the whitelist
        for drug in self.MEDICAL_WHITELIST:
            if drug.lower() == text_lower:
                return True
        
        return False
    
    def _get_anonymizers(self) -> Dict[str, OperatorConfig]:
        """Get Presidio anonymizer operators."""
        return {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CARD>"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "<SSN>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOC>"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE>"}),
            "NRP": OperatorConfig("replace", {"new_value": "<ID>"}),
            "MEDICAL_LICENSE": OperatorConfig("replace", {"new_value": "<LICENSE>"}),
            "URL": OperatorConfig("replace", {"new_value": "<URL>"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP>"}),
        }
    
    def scrub_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively scrub dictionary values."""
        scrubbed = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                scrubbed[key] = self.scrub(value)
            elif isinstance(value, dict):
                scrubbed[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                scrubbed[key] = self.scrub_list(value)
            else:
                scrubbed[key] = value
        
        return scrubbed
    
    def scrub_list(self, items: List[Any]) -> List[Any]:
        """Recursively scrub list items."""
        scrubbed = []
        
        for item in items:
            if isinstance(item, str):
                scrubbed.append(self.scrub(item))
            elif isinstance(item, dict):
                scrubbed.append(self.scrub_dict(item))
            elif isinstance(item, list):
                scrubbed.append(self.scrub_list(item))
            else:
                scrubbed.append(item)
        
        return scrubbed

# Singleton instance
_scrubber_instance = None

def get_enhanced_pii_scrubber() -> EnhancedPIIScrubber:
    """Get singleton PII scrubber instance."""
    global _scrubber_instance
    if _scrubber_instance is None:
        _scrubber_instance = EnhancedPIIScrubber()
    return _scrubber_instance
