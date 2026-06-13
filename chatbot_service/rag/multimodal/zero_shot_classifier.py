"""
Zero-Shot Medical Image Classifier
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import base64
import io

logger = logging.getLogger(__name__)



@dataclass
class CategoryDefinition:
    name: str
    description: str
    visual_features: List[str]
    clinical_significance: str


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    reasoning: str
    alternative_categories: List[Dict[str, float]]
    visual_findings: List[str]


@dataclass
class MultiLabelResult:
    labels: List[ClassificationResult]
    primary_category: str
    summary: str


class ZeroShotMedicalClassifier:
    """
    Zero-Shot Classifier for Medical Images using Vision-Language Models.
    Allows classification into custom categories without training.
    """

    # Pre-defined category sets
    WOUND_CATEGORIES = [
        CategoryDefinition("Venous Ulcer", "Shallow wound on leg/ankle, irregular borders", ["shallow", "irregular border", "exudate"], "Requires compression"),
        CategoryDefinition("Arterial Ulcer", "Deep wound, punched-out appearance", ["deep", "punched-out", "pale base"], "Requires revascularization"),
        CategoryDefinition("Diabetic Foot Ulcer", "Ulcer on pressure point of foot", ["callus", "plantar surface", "deep"], "Off-loading needed"),
        CategoryDefinition("Pressure Injury", "Ulcer over bony prominence", ["bony prominence", "sacrum/heel", "necrosis"], "Pressure relief needed"),
        CategoryDefinition("Surgical Site Infection", "Redness/pus at incision site", ["erythema", "purulence", "dehiscence"], "Antibiotics needed"),
    ]

    SKIN_LESION_CATEGORIES = [
        CategoryDefinition("Melanoma", "Asymmetric, irregular border, multi-color", ["asymmetry", "irregular border", "color variation"], "Urgent biopsy"),
        CategoryDefinition("Basal Cell Carcinoma", "Pearly papule with telangiectasia", ["pearly", "rolled border", "telangiectasia"], "Dermatology referral"),
        CategoryDefinition("Squamous Cell Carcinoma", "Scaly red patch or nodule", ["scaly", "crusty", "ulcerated"], "Dermatology referral"),
        CategoryDefinition("Benign Nevus", "Symmetric, uniform color, smooth border", ["symmetry", "uniform color", "regular border"], "Routine monitoring"),
        CategoryDefinition("Seborrheic Keratosis", "Stuck-on waxy appearance", ["waxy", "stuck-on", "verrucous"], "Benign"),
    ]

    DIABETIC_RETINOPATHY_CATEGORIES = [
        CategoryDefinition("No DR", "Normal retina", ["clear disc", "normal vessels"], "Annual screening"),
        CategoryDefinition("Mild NPDR", "Microaneurysms only", ["microaneurysms"], "Monitor 6-12 months"),
        CategoryDefinition("Moderate NPDR", "More than just microaneurysms", ["hemorrhages", "hard exudates", "cotton wool spots"], "Monitor 3-6 months"),
        CategoryDefinition("Severe NPDR", "Severe bleeding/beading", ["4 quadrants hemorrhage", "venous beading"], "Urgent referral"),
        CategoryDefinition("PDR", "Neovascularization", ["new vessels", "vitreous hemorrhage"], "Immediate treatment"),
    ]

    def __init__(self, llm_gateway=None):
        self.llm_gateway = llm_gateway

    async def classify(
        self,
        image_data: str,
        categories: List[CategoryDefinition],
        context: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify an image into one of the provided categories.
        """
        if not self.llm_gateway:
            logger.warning("LLM Gateway not available for classification")
            return ClassificationResult("Unknown", 0.0, "LLM unavailable", [], [])

        prompt = self._construct_prompt(categories, context)
        
        try:
            # Assuming llm_gateway supports multimodal input
            # If not, this would need an adapter
            response = await self.llm_gateway.generate_multimodal(prompt, image_data)
            return self._parse_response(response, categories)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult("Error", 0.0, str(e), [], [])

    async def classify_multi_label(
        self,
        image_data: str,
        categories: List[CategoryDefinition],
        max_labels: int = 3,
        threshold: float = 0.3
    ) -> MultiLabelResult:
        """
        Classify image with multiple potential labels.
        """
        # Implementation similar to classify but parsing multiple labels
        # For brevity, reusing single classification logic structure
        result = await self.classify(image_data, categories)
        return MultiLabelResult([result], result.category, result.reasoning)

    def _construct_prompt(self, categories: List[CategoryDefinition], context: Optional[str] = None) -> str:
        cat_str = "\n".join([f"- {c.name}: {c.description} (Features: {', '.join(c.visual_features)})" for c in categories])
        
        return f"""
Analyze this medical image and classify it into EXACTLY ONE of the following categories:

{cat_str}

CONTEXT: {context or 'None provided'}

TASK:
1. Identify key visual features present in the image.
2. Compare features against category definitions.
3. Select the most likely category.
4. Provide a confidence score (0.0-1.0).
5. Explain your reasoning.

OUTPUT FORMAT:
Category: [Category Name]
Confidence: [0.0-1.0]
Reasoning: [Explanation]
Visual Findings: [List of features seen]
Alternatives: [Cat1: 0.X, Cat2: 0.Y]
"""

    def _parse_response(self, response: str, categories: List[CategoryDefinition]) -> ClassificationResult:
        # Simple parsing logic - in production use structured output
        lines = response.strip().split("\n")
        category = "Unknown"
        confidence = 0.0
        reasoning = ""
        findings = []
        alternatives = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Category:"):
                category = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError as e:
                    logger.debug(f"Could not parse confidence value: {e}")
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("Visual Findings:"):
                findings = [f.strip() for f in line.split(":", 1)[1].split(",")]
                
        return ClassificationResult(category, confidence, reasoning, alternatives, findings)


# Convenience function for tool integration
async def classify_medical_image(
    image_data: str,
    context: Optional[str] = None,
    category_set: str = "skin"
) -> str:
    """
    Classify a medical image using zero-shot vision model.
    
    Args:
        image_data: Base64 encoded image or URL
        context: Clinical context
        category_set: Pre-defined category set ("skin", "wound", "retina")
        
    Returns:
        Formatted classification result
    """
    from core.dependencies import DIContainer
    container = DIContainer.get_instance()
    
    classifier = ZeroShotMedicalClassifier(llm_gateway=container.llm_gateway)
    
    # Select categories
    if category_set == "wound":
        categories = ZeroShotMedicalClassifier.WOUND_CATEGORIES
    elif category_set == "retina":
        categories = ZeroShotMedicalClassifier.DIABETIC_RETINOPATHY_CATEGORIES
    else:
        categories = ZeroShotMedicalClassifier.SKIN_LESION_CATEGORIES
        
    result = await classifier.classify(image_data, categories, context)
    
    # Format output
    lines = [
        f"## Image Analysis: {result.category}",
        f"**Confidence**: {result.confidence*100:.0f}%",
        f"**Reasoning**: {result.reasoning}",
        f"**Visual Findings**: {', '.join(result.visual_findings)}",
    ]
    
    return "\n".join(lines)
