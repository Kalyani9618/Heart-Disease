"""
Medical NLP Pipeline Factory
============================
Assembles a custom spaCy pipeline with:
1. Transformer model for high accuracy (en_core_web_trf)
2. EntityRuler for DETERMINISTIC drug detection (before NER)
3. Negation detector for medical safety
4. Custom medical annotator

Reference: spacy_guide2.md, spacy_guide3.md
"""


try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except Exception as e:
    spacy = None  # type: ignore
    Language = None  # type: ignore
    Doc = Span = Token = None  # type: ignore
    SPACY_AVAILABLE = False
    import logging as _logging
    _logging.warning(f"spaCy not available in NLP factory: {e}")
import json
import os
import logging

logger = logging.getLogger(__name__)

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "custom_medical_model")
SPACY_MODELS_DIR = os.path.join(MODELS_DIR, "spacy_models")


def _load_local_spacy_model(model_name: str) -> Language:
    """
    Try to load a spaCy model from the local spacy_models directory.
    
    Supports:
    - Extracted model directories (e.g., en_core_web_trf/)
    - Wheel files (e.g., en_core_web_trf-3.8.0-py3-none-any.whl)
    """
    # Check for extracted directory
    local_model_path = os.path.join(SPACY_MODELS_DIR, model_name)
    if os.path.isdir(local_model_path):
        nlp = spacy.load(local_model_path)
        logger.info(f"Loaded local model from: {local_model_path}")
        return nlp
    
    # Check for wheel file and install if found
    import glob
    wheel_pattern = os.path.join(SPACY_MODELS_DIR, f"{model_name}*.whl")
    wheel_files = glob.glob(wheel_pattern)
    
    if wheel_files:
        wheel_path = wheel_files[0]
        logger.info(f"Found local wheel: {wheel_path}. Installing...")
        import subprocess
        result = subprocess.run(
            ["pip", "install", wheel_path, "--quiet"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Reload spacy to pick up the new model
            import importlib
            importlib.invalidate_caches()
            nlp = spacy.load(model_name)
            logger.info(f"Installed and loaded model from wheel: {model_name}")
            return nlp
        else:
            logger.warning(f"Failed to install wheel: {result.stderr}")
    
    raise OSError(f"Local model {model_name} not found in {SPACY_MODELS_DIR}")


def build_medical_nlp(
    use_transformer: bool = True,
    load_from_disk: bool = True
) -> Language:
    """
    Build the complete medical NLP pipeline.
    
    Pipeline order:
    1. Tokenizer (built-in)
    2. EntityRuler (BEFORE ner) - deterministic drug/dosage detection
    3. NER (statistical) - context-aware entity recognition
    4. Negation Detector - marks negated entities
    5. Medical Annotator - categorizes and enriches entities
    
    Args:
        use_transformer: Use en_core_web_trf (GPU) vs en_core_web_md (CPU)
        load_from_disk: Try to load pre-built model from disk first
    
    Returns:
        Configured spaCy Language pipeline
    """
    
    # Try to load pre-built custom model (for fast startup)
    if load_from_disk and os.path.exists(CUSTOM_MODEL_PATH):
        try:
            nlp = spacy.load(CUSTOM_MODEL_PATH)
            logger.info(f"Loaded custom medical model from {CUSTOM_MODEL_PATH}")
            return nlp
        except Exception as e:
            logger.warning(f"Could not load custom model: {e}")
    
    # 1. Load base model (Transformer for high accuracy)
    model_name = "en_core_web_trf" if use_transformer else "en_core_web_md"
    
    # First, try to load from local spacy_models folder
    try:
        nlp = _load_local_spacy_model(model_name)
    except OSError:
        # Fall back to system-installed models
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded base model: {model_name}")
        except OSError:
            # Fallback chain - try available models in order of preference
            for fallback in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
                try:
                    nlp = spacy.load(fallback)
                    logger.warning(f"Using fallback model: {fallback}")
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("No spaCy models available!")
    
    # 1.5 Apply Custom Medical Tokenizer
    try:
        from rag.nlp.tokenizer import create_medical_tokenizer
        nlp.tokenizer = create_medical_tokenizer(nlp)
        logger.info("Applied custom medical tokenizer")
    except Exception as e:
        logger.warning(f"Failed to apply custom tokenizer: {e}")
    
    # 2. Add EntityRuler BEFORE NER (deterministic drug detection)
    _add_medical_entity_ruler(nlp)
    
    # 2.5 Add Medical Sentencizer
    # We use a standard sentencizer first to set initial boundaries,
    # then medical_sentencizer to correct them, then parser.
    if "sentencizer" not in nlp.pipe_names:
        try:
            nlp.add_pipe("sentencizer", before="parser")
        except Exception:
            pass

    if "medical_sentencizer" not in nlp.pipe_names:
        try:
            from rag.nlp.medical_sentencizer import medical_sentencizer
            if "sentencizer" in nlp.pipe_names:
                nlp.add_pipe("medical_sentencizer", after="sentencizer")
            elif "parser" in nlp.pipe_names:
                nlp.add_pipe("medical_sentencizer", before="parser")
            else:
                nlp.add_pipe("medical_sentencizer", first=True)
        except Exception as e:
            logger.warning(f"Failed to add medical_sentencizer: {e}")
    
    # 3. Register custom extensions
    _register_extensions()
    
    # 4. Add Negation Detector (after NER)
    if "negation_detector" not in nlp.pipe_names:
        # Import here to avoid circular imports if the component is in another file
        # But we will register it in the main service or here if it's a separate module
        # For now, assuming it's registered via entry points or imported
        try:
            nlp.add_pipe("negation_detector", after="ner")
        except ValueError:
            # If component not found, we might need to import it
            from rag.nlp.negation_detector import negation_detector
            nlp.add_pipe("negation_detector", after="ner")
    
    # 5. Add Medical Annotator (last)
    if "medical_annotator" not in nlp.pipe_names:
        try:
            from rag.nlp.medical_annotator import medical_annotator
            nlp.add_pipe("medical_annotator", last=True)
        except Exception as e:
            logger.warning(f"Failed to add medical_annotator: {e}")
    
    logger.info(f"Medical NLP pipeline: {nlp.pipe_names}")
    return nlp


def _add_medical_entity_ruler(nlp: Language) -> None:
    """
    Add EntityRuler with deterministic drug patterns.
    
    CRITICAL: This ensures we NEVER miss a known drug, 
    even if the statistical model is unsure.
    
    Reference: spcacy_guide2.md
    """
    # Create ruler BEFORE NER so statistical model can still override
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = []
    
    # Load drug dictionary
    drugs_path = os.path.join(DATA_DIR, "drugs.json")
    if os.path.exists(drugs_path):
        try:
            with open(drugs_path, "r") as f:
                drug_data = json.load(f)
            
            # Handle various JSON structures
            drug_names = _extract_drug_names(drug_data)
            
            for drug_name in drug_names:
                # Simple pattern for exact match (case-insensitive)
                patterns.append({
                    "label": "DRUG",
                    "pattern": [{"LOWER": drug_name.lower()}]
                })
                # Multi-word drugs
                words = drug_name.lower().split()
                if len(words) > 1:
                    patterns.append({
                        "label": "DRUG",
                        "pattern": [{"LOWER": w} for w in words]
                    })
            
            logger.info(f"Loaded {len(drug_names)} drugs into EntityRuler")
        except Exception as e:
            logger.error(f"Failed to load drugs.json: {e}")
    
    # Add dosage patterns (always include these)
    dosage_patterns = [
        # "500 mg", "250mg", "0.5 ml"
        {
            "label": "DOSAGE",
            "pattern": [
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["mg", "ml", "g", "mcg", "iu", "units", "tablets", "caps"]}}
            ]
        },
        # "500mg" (no space)
        {
            "label": "DOSAGE",
            "pattern": [{"TEXT": {"REGEX": r"^\d+\.?\d*(mg|ml|g|mcg)$"}}]
        },
        # Frequency patterns: "twice daily", "once a day", "q6h"
        {
            "label": "FREQUENCY",
            "pattern": [
                {"LOWER": {"IN": ["once", "twice", "three", "four"]}},
                {"LOWER": {"IN": ["daily", "weekly", "a"]}, "OP": "?"},
                {"LOWER": {"IN": ["day", "week"]}, "OP": "?"}
            ]
        },
        {
            "label": "FREQUENCY",
            "pattern": [{"TEXT": {"REGEX": r"^q\d+h$"}}]  # q6h, q8h, etc.
        },
    ]
    patterns.extend(dosage_patterns)
    
    # Add medication suffix patterns (catch new drugs)
    medication_suffixes = [
        "mab", "nib", "pril", "sartan", "statin", "olol", "pine",
        "zole", "pam", "lam", "mycin", "cillin", "floxacin", "vir"
    ]
    for suffix in medication_suffixes:
        patterns.append({
            "label": "DRUG",
            "pattern": [{"TEXT": {"REGEX": f"(?i).*{suffix}$"}}]
        })
    
    ruler.add_patterns(patterns)
    logger.info(f"EntityRuler configured with {len(patterns)} patterns")


def _extract_drug_names(data) -> list:
    """Extract drug names from drugs.json structure."""
    names = set()
    
    # Handle list of drug objects (standard drugs.json format)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Extract generic name
                if "generic_name" in item and isinstance(item["generic_name"], str):
                    names.add(item["generic_name"])
                
                # Extract brand names
                if "brand_names" in item and isinstance(item["brand_names"], list):
                    names.update(item["brand_names"])
                
                # Extract ID if it looks like a name
                if "id" in item and isinstance(item["id"], str):
                    names.add(item["id"])
                    
                # Recursive search for nested structures (just in case)
                # But avoid adding keys as names
                names.update(_extract_drug_names(item))
                
    elif isinstance(data, dict):
        # Handle dict values recursively
        for value in data.values():
            names.update(_extract_drug_names(value))
            
    return list(names)


def _register_extensions() -> None:
    """Register custom Doc/Span/Token extensions."""
    # Entity extensions
    if not Span.has_extension("is_negated"):
        Span.set_extension("is_negated", default=False)
    if not Span.has_extension("negation_cue"):
        Span.set_extension("negation_cue", default=None)
    if not Span.has_extension("confidence"):
        Span.set_extension("confidence", default=1.0)
    if not Span.has_extension("source"):
        Span.set_extension("source", default="ner")  # "ner", "ruler", "matcher"
    if not Span.has_extension("medical_category"):
        Span.set_extension("medical_category", default=None)
    
    # Token extensions
    if not Token.has_extension("is_negation"):
        Token.set_extension("is_negation", default=False)
    if not Token.has_extension("medical_category"):
        Token.set_extension("medical_category", default=None)
    
    # Doc extensions
    if not Doc.has_extension("medical_entities"):
        Doc.set_extension("medical_entities", default={})
    if not Doc.has_extension("has_negated_entities"):
        Doc.set_extension("has_negated_entities", getter=lambda doc: any(
            ent._.is_negated for ent in doc.ents
        ))
    if not Doc.has_extension("drug_interactions"):
        Doc.set_extension("drug_interactions", default=[])


def save_medical_model(nlp: Language, path: str = None) -> None:
    """
    Save the custom medical pipeline to disk for fast startup.
    
    Reference: spacy_guide5.md (Serialization)
    """
    path = path or CUSTOM_MODEL_PATH
    os.makedirs(path, exist_ok=True)
    nlp.to_disk(path)
    logger.info(f"Saved custom medical model to {path}")


def get_medical_nlp() -> Language:
    """
    Get the singleton medical NLP instance.
    Thread-safe lazy initialization.
    """
    global _nlp_instance
    if "_nlp_instance" not in globals() or _nlp_instance is None:
        _nlp_instance = build_medical_nlp()
    return _nlp_instance

_nlp_instance = None
