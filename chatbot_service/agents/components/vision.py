"""
Vision Agent - Agents with image understanding capabilities.

Provides:
- VisionCapableMixin: Add vision to any agent
- MedicalImageAnalyzer: Specialized for medical images

Based on smolagents vision_agents.ipynb patterns.
"""
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
import base64
import io
import logging


logger = logging.getLogger(__name__)

# Try to import PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


@dataclass
class ImageInput:
    """
    Wrapper for image inputs.
    
    Attributes:
        data: Image data (PIL, bytes, or path)
        caption: Optional description
        is_medical: Whether this is a medical image
    """
    data: Union[object, bytes, str]
    caption: Optional[str] = None
    is_medical: bool = False
    
    def to_base64(self) -> str:
        """Convert image to base64 string."""
        if isinstance(self.data, str):
            # File path
            with open(self.data, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif PIL_AVAILABLE and isinstance(self.data, Image.Image):
            # PIL Image
            buffer = io.BytesIO()
            self.data.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(self.data, bytes):
            return base64.b64encode(self.data).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(self.data)}")


class VisionCapableMixin:
    """
    Mixin that adds vision capabilities to agents.
    
    Use with any agent class to enable image processing.
    
    Usage:
        class MyAgent(VisionCapableMixin, BaseAgent):
            async def run(self, query, images=None):
                if images:
                    message = self.prepare_vision_message(query, images)
                    response = await self.vision_llm.ainvoke([message])
                ...
    """
    
    def __init__(self, vision_llm=None, **kwargs):
        """
        Initialize vision capability.
        
        Args:
            vision_llm: Vision-capable LLM (GPT-4V, Gemini Pro Vision, etc.)
        """
        self.vision_llm = vision_llm
        super().__init__(**kwargs) if hasattr(super(), '__init__') else None
    
    def encode_image(self, image: Union[object, str, bytes]) -> str:
        """
        Encode image to base64 for LLM.
        
        Args:
            image: PIL Image, file path, or bytes
            
        Returns:
            Base64 encoded string
        """
        if isinstance(image, ImageInput):
            return image.to_base64()
        
        if isinstance(image, str):
            # File path
            with open(image, 'rb') as f:
                image_bytes = f.read()
        elif PIL_AVAILABLE and isinstance(image, Image.Image):
            # PIL Image
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
        elif isinstance(image, bytes):
            image_bytes = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def prepare_vision_message(
        self,
        text: str,
        images: List[Union[object, str, ImageInput]]
    ) -> Dict[str, Any]:
        """
        Prepare message with images for vision LLM.
        
        Args:
            text: Text prompt
            images: List of images (PIL, path, or ImageInput)
            
        Returns:
            Message dict in OpenAI vision format
        """
        content = [{"type": "text", "text": text}]
        
        for img in images:
            is_medical = isinstance(img, ImageInput) and img.is_medical
            encoded = self.encode_image(img)
            
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"}
            }
            
            if is_medical:
                image_content["is_medical"] = True
            
            content.append(image_content)
        
        return {"role": "user", "content": content}
    
    async def analyze_image(
        self,
        image: Union[object, str, ImageInput],
        query: str = "Describe this image in detail"
    ) -> str:
        """
        Analyze a single image.
        
        Args:
            image: Image to analyze
            query: Question about the image
            
        Returns:
            LLM response
        """
        if not self.vision_llm:
            return "Vision LLM not configured."
        
        message = self.prepare_vision_message(query, [image])
        
        try:
            response = await self.vision_llm.ainvoke([message])
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return f"Error analyzing image: {e}"


class MedicalImageAnalyzer(VisionCapableMixin):
    """
    Specialized agent for medical image analysis.
    
    IMPORTANT: This is for informational purposes ONLY.
    Always includes safety disclaimers and recommends professional review.
    
    Supported image types:
    - X-rays
    - MRI scans
    - CT scans
    - Ultrasounds
    - Dermatological images
    - ECG/EKG traces
    
    Usage:
        analyzer = MedicalImageAnalyzer(vision_llm=my_llm)
        result = await analyzer.analyze(
            image="/path/to/xray.jpg",
            context="Patient reports chest pain"
        )
        print(result["analysis"])
        print(result["disclaimer"])
    """
    
    MEDICAL_IMAGE_PROMPT = """
Analyze this medical image and provide:
1. Type of image (X-ray, MRI, CT, ultrasound, etc.)
2. Anatomical region visible
3. Notable observations (describe what you see objectively)
4. Image quality assessment

CRITICAL SAFETY REQUIREMENTS:
- Do NOT provide any diagnosis or medical interpretation
- Do NOT suggest treatments or medications
- Use objective descriptions only ("appears to show", "visible", "can be observed")
- Always note image limitations that could affect interpretation
- Recommend professional medical review

Context from patient: {context}

Format your response as:
## Image Type
[type]

## Anatomical Region
[region]

## Observations
[objective observations]

## Image Quality
[quality assessment]

## Recommendation
[always recommend professional review]
"""

    DISCLAIMER = """
⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This analysis is generated by AI for INFORMATIONAL PURPOSES ONLY.

- This is NOT a medical diagnosis
- This does NOT replace professional medical evaluation
- Do NOT make treatment decisions based on this analysis
- ALWAYS consult a qualified healthcare provider

The AI may make errors. Image quality, positioning, and other factors
affect analysis accuracy. Only a licensed medical professional can
provide proper medical interpretation.
"""

    def __init__(self, vision_llm=None, strict_mode: bool = True):
        """
        Initialize the medical image analyzer.
        
        Args:
            vision_llm: Vision-capable LLM
            strict_mode: If True, always includes full disclaimers
        """
        super().__init__(vision_llm=vision_llm)
        self.strict_mode = strict_mode
    
    async def analyze(
        self,
        image: Union[object, str, ImageInput],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a medical image.
        
        Args:
            image: Medical image to analyze
            context: Additional patient context
            
        Returns:
            Dict with analysis, disclaimer, and metadata
        """
        if not self.vision_llm:
            return {
                "analysis": "Vision LLM not configured.",
                "disclaimer": self.DISCLAIMER,
                "requires_professional_review": True,
                "error": True
            }
        
        prompt = self.MEDICAL_IMAGE_PROMPT.format(
            context=context or "No additional context provided"
        )
        
        # Mark as medical image
        if isinstance(image, ImageInput):
            image.is_medical = True
        else:
            image = ImageInput(data=image, is_medical=True)
        
        message = self.prepare_vision_message(prompt, [image])
        
        try:
            response = await self.vision_llm.ainvoke([message])
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "analysis": analysis,
                "disclaimer": self.DISCLAIMER,
                "requires_professional_review": True,
                "context_provided": context,
                "strict_mode": self.strict_mode
            }
            
        except Exception as e:
            logger.error(f"Medical image analysis failed: {e}")
            return {
                "analysis": f"Error during analysis: {str(e)}",
                "disclaimer": self.DISCLAIMER,
                "requires_professional_review": True,
                "error": True
            }
    
    async def compare_images(
        self,
        image1: Union[object, str],
        image2: Union[object, str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two medical images (e.g., before/after, left/right).
        
        Args:
            image1: First image
            image2: Second image
            context: Comparison context
            
        Returns:
            Comparison analysis with disclaimers
        """
        prompt = f"""
Compare these two medical images.

Context: {context or "No context provided"}

Describe:
1. What type of images these are
2. Any observable differences between them
3. Any observable similarities
4. Image quality comparison

Do NOT diagnose or interpret medically. Only describe what is visible.
Always recommend professional medical review for any health decisions.
"""
        
        images = [
            ImageInput(data=image1, caption="Image 1", is_medical=True),
            ImageInput(data=image2, caption="Image 2", is_medical=True)
        ]
        
        message = self.prepare_vision_message(prompt, images)
        
        try:
            response = await self.vision_llm.ainvoke([message])
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "comparison": analysis,
                "disclaimer": self.DISCLAIMER,
                "requires_professional_review": True
            }
        except Exception as e:
            return {
                "comparison": f"Error: {e}",
                "disclaimer": self.DISCLAIMER,
                "requires_professional_review": True,
                "error": True
            }


# Factory function
def create_medical_image_analyzer(vision_llm) -> MedicalImageAnalyzer:
    """Create a configured medical image analyzer."""
    return MedicalImageAnalyzer(vision_llm=vision_llm, strict_mode=True)
