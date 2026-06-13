"""
OpenFDA Unified Safety Service

The 'Safety Officer' of HeartGuard AI.
Combines Drug Events, Drug Recalls, Food Recalls, and Food Events
into a single, easy-to-use interface for the chatbot router.

This service abstracts away the complexity of multiple API endpoints
and provides formatted, human-readable responses.


Usage:
    >>> from tools.openfda.openfda_safety_service import OpenFDASafetyService
    >>> safety = OpenFDASafetyService()
    >>> print(safety.get_drug_side_effects("Lipitor"))
    📊 Top 5 Reported Side Effects for LIPITOR:
    1. Myalgia: 12,500 reports
    ...
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List

from tools.openfda.drug_adverse_events import DrugAdverseEventService
from tools.openfda.drug_enforcement import DrugEnforcementQuerier
from tools.openfda.food_enforcement import FoodEnforcementQuerier
from tools.openfda.food_events import FoodEventService
from tools.openfda.api_client import OpenFDAClient

logger = logging.getLogger(__name__)


class OpenFDASafetyService:
    """
    Unified safety interface for HeartGuard AI.
    
    This class serves as the single point of access for all FDA safety data.
    It combines:
    - Drug side effects (FAERS - Adverse Event Reporting System)
    - Drug recalls (Enforcement Actions)
    - Drug severity statistics (Death/Hospitalization)
    - Food recalls (Food Enforcement)
    - Food adverse events (CAERS - Food/Supplement events)
    - Allergen contamination alerts
    
    The service returns formatted strings suitable for direct display
    to users or inclusion in LLM context.
    
    Example:
        >>> safety = OpenFDASafetyService()
        >>> 
        >>> # Drug queries
        >>> print(safety.get_drug_side_effects("Lipitor"))
        >>> print(safety.check_drug_severity("Warfarin"))
        >>> print(safety.verify_drug_reaction("Lisinopril", "Cough"))
        >>> 
        >>> # Food queries
        >>> print(safety.check_food_recalls("Spinach"))
        >>> print(safety.check_allergen_recalls("Peanut"))
    
    Attributes:
        api_key: Optional OpenFDA API key for higher rate limits
        drug_events: DrugAdverseEventService instance
        drug_enforcement: DrugEnforcementQuerier instance
        food_enforcement: FoodEnforcementQuerier instance
        food_events: FoodEventService instance
        client: Base OpenFDAClient for direct queries
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Unified Safety Service.
        
        Args:
            api_key: Optional OpenFDA API key. Without a key, you get
                     standard rate limits (240 requests/minute).
        """
        self.api_key = api_key
        
        # Initialize sub-services
        self.drug_events = DrugAdverseEventService(api_key)
        self.drug_enforcement = DrugEnforcementQuerier(api_key)
        self.food_enforcement = FoodEnforcementQuerier(api_key)
        self.food_events = FoodEventService(api_key)
        self.client = OpenFDAClient(api_key)
        
        logger.info("OpenFDASafetyService initialized with all sub-services")

    async def close(self):
        """Close all underlying clients."""
        await self.drug_events.close()
        await self.drug_enforcement.close()
        await self.food_enforcement.close()
        await self.food_events.close()
        await self.client.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DRUG SAFETY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_drug_side_effects(self, drug_name: str, limit: int = 5) -> str:
        """
        [Pharmacist Role]
        Returns the most common side effects reported for a medication.
        
        This queries the FAERS database and returns the top reported
        adverse reactions in a formatted, user-friendly string.
        
        Args:
            drug_name: Brand name (e.g., "Lipitor") or generic (e.g., "atorvastatin")
            limit: Number of side effects to return (default: 5)
            
        Returns:
            Formatted string with side effects list, or info message if none found.
            
        Example:
            >>> print(await safety.get_drug_side_effects("Lipitor"))
            📊 Top 5 Reported Side Effects for LIPITOR:
            1. Myalgia: 12,500 reports
            2. Pain In Extremity: 8,200 reports
            3. Diarrhea: 6,100 reports
            4. Nausea: 5,800 reports
            5. Fatigue: 4,200 reports
            
        Router Trigger Keywords:
            - "side effect"
            - "adverse reaction"
            - "what are the effects of"
            - "what happens if I take"
        """
        logger.info(f"[SafetyService] Getting side effects for: {drug_name}")
        result = await self.drug_events.get_top_side_effects(drug_name, limit)
        return result["formatted"]
    
    async def check_drug_severity(self, drug_name: str) -> str:
        """
        [Safety Check Role]
        Checks how many reports involved Death or Hospitalization.
        
        This provides a "red flag" check for potentially dangerous drugs,
        showing statistics on severe outcomes.
        
        Args:
            drug_name: The medication to check
            
        Returns:
            Formatted string with severity statistics, or all-clear message.
            
        Example (dangerous drug):
            >>> print(await safety.check_drug_severity("Warfarin"))
            ⚠️ **WARNING: Severe Outcomes Reported for WARFARIN**
            Deaths: 3,240 reports
            Hospitalizations: 8,900 reports
            Life-threatening: 1,200 reports
            *(Note: These are raw reports. Causation is not proven.)*
            
        Example (safer drug):
            >>> print(await safety.check_drug_severity("Vitamin D"))
            ✅ Vitamin D: No severe outcomes (Death/Hospitalization) found in FDA data.
            
        Router Trigger Keywords:
            - "dangerous"
            - "is [drug] safe"
            - "serious side effects"
            - "hospitalization"
        """
        logger.info(f"[SafetyService] Checking severity for: {drug_name}")
        result = await self.drug_events.check_severity(drug_name)
        return result["formatted"]
    
    async def verify_drug_reaction(self, drug_name: str, reaction: str) -> str:
        """
        [Verification Role]
        Checks if a specific drug is linked to a specific symptom.
        
        Answers the question "Does [drug] cause [symptom]?" by searching
        for FDA reports where both appear together.
        
        Args:
            drug_name: The medication name
            reaction: The symptom to verify (e.g., "Cough", "Myalgia")
            
        Returns:
            Formatted string confirming or denying the link
            
        Example:
            >>> print(await safety.verify_drug_reaction("Lisinopril", "Cough"))
            ✅ **YES.** There are 45,230 reports linking Lisinopril to Cough.
            
            *Note: These are raw reports. Causation is not proven.*
            
        Router Trigger Keywords:
            - "cause"
            - "linked to"
            - "does [drug] cause"
            - "is [symptom] a side effect"
        """
        logger.info(f"[SafetyService] Verifying: {drug_name} -> {reaction}")
        result = await self.drug_events.check_specific_reaction(drug_name, reaction)
        return result["formatted"]
    
    async def check_drug_recalls(self, drug_name: str) -> str:
        """
        [Recall Alert Role]
        Checks if a drug currently has active recalls.
        
        Args:
            drug_name: The medication to check
            
        Returns:
            Formatted string with recall information or all-clear
            
        Example (has recalls):
            >>> print(await safety.check_drug_recalls("Metformin"))
            🚨 **Active Recalls for METFORMIN**:
            1. [Class II] Impurity detected
               Company: Generic Pharma Inc.
               Distribution: Nationwide
            
        Example (no recalls):
            >>> print(await safety.check_drug_recalls("Aspirin"))
            ✅ No active recalls found for Aspirin.
            
        Router Trigger Keywords:
            - "recall"
            - "recalled"
            - "is [drug] recalled"
            - "safety alert"
        """
        logger.info(f"[SafetyService] Checking recalls for: {drug_name}")
        
        recalls = await self.drug_enforcement.find_recalls_for_drug(drug_name, limit=5)
        
        if not recalls:
            return f"✅ No recalls found for {drug_name}."
        
        output = f"🚨 **Active Recalls for {drug_name.upper()}** ({len(recalls)} found):\n\n"
        for i, recall in enumerate(recalls, 1):
            output += (
                f"{i}. [{recall.get('classification', 'N/A')}] "
                f"{recall.get('reason_for_recall', 'Recall')}\n"
                f"   Company: {recall.get('recalling_firm', 'Unknown')}\n"
                f"   Status: {recall.get('status', 'Unknown')}\n\n"
            )
        
        return output
    
    async def full_drug_safety_check(self, drug_name: str) -> str:
        """
        [Comprehensive Review Role]
        Performs a complete safety analysis for a drug.
        
        Combines side effects, severity check, and recall status.
        
        Args:
            drug_name: The medication to analyze
            
        Returns:
            Comprehensive safety report
            
        Example:
            >>> print(await safety.full_drug_safety_check("Warfarin"))
            ═════════════════════════════════════════
            COMPLETE SAFETY REPORT: WARFARIN
            ═════════════════════════════════════════
            
            📊 Top Side Effects:
            1. Bleeding: 45,000 reports
            2. Hemorrhage: 38,000 reports
            ...
            
            ⚠️ SEVERITY WARNING:
            Deaths: 3,240 reports
            Hospitalizations: 8,900 reports
            ...
        """
        logger.info(f"[SafetyService] Performing full safety check for: {drug_name}")
        
        output = "=" * 50 + "\n"
        output += f"COMPLETE SAFETY REPORT: {drug_name.upper()}\n"
        output += "=" * 50 + "\n\n"
        
        # Add side effects
        output += await self.get_drug_side_effects(drug_name, limit=5)
        output += "\n" + "-" * 50 + "\n\n"
        
        # Add severity
        output += await self.check_drug_severity(drug_name)
        output += "\n" + "-" * 50 + "\n\n"
        
        # Add recalls
        output += await self.check_drug_recalls(drug_name)
        output += "\n" + "=" * 50 + "\n"
        
        return output
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FOOD SAFETY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def check_food_recalls(self, food_item: str) -> str:
        """
        [Food Safety Role]
        Checks for recalls on a specific food product.
        
        Queries the Food Enforcement database for active and past recalls.
        
        Args:
            food_item: Food product name (e.g., "Spinach", "Peanut Butter")
            
        Returns:
            Formatted string with recall information
            
        Example:
            >>> print(await safety.check_food_recalls("Spinach"))
            🥬 **Spinach Recalls** (3 found):
            
            1. 🔴 [Class I] E. coli contamination
               Status: Ongoing
               Company: Fresh Farms LLC
               Distribution: Nationwide
               Date: 20260104
            
        Router Trigger Keywords:
            - "food recall"
            - "contaminated"
            - "is [food] recalled"
            - "safe to eat"
        """
        logger.info(f"[SafetyService] Checking food recalls for: {food_item}")
        result = await self.food_enforcement.check_food_recalls(food_item)
        return result["formatted"]
    
    async def check_food_adverse_events(self, product_name: str) -> str:
        """
        [CAERS Database Role]
        Check for reported adverse events from a food/supplement product.
        
        Queries the CAERS database for consumer-reported adverse reactions.
        
        Args:
            product_name: Product name (supplement or food)
            
        Returns:
            Formatted string with adverse event information
            
        Example:
            >>> print(await safety.check_food_adverse_events("5-Hour Energy"))
            ⚠️ **5-Hour Energy Adverse Events** (234 reported):
            
            Top Reported Reactions:
              1. Heart palpitations: 89 reports
              2. Chest pain: 67 reports
              3. Anxiety: 45 reports
            
            🚨 **Serious Outcomes:** 12 reports (hospitalization, death, etc.)
            
        Router Trigger Keywords:
            - "supplement"
            - "adverse events"
            - "health risks"
            - "bad reactions"
        """
        logger.info(f"[SafetyService] Checking adverse events for: {product_name}")
        result = await self.food_events.check_supplement_adverse_events(product_name)
        return result["formatted"]
    
    async def check_allergen_recalls(self, allergen: str) -> str:
        """
        [Allergen Alert Role]
        Finds all recalls related to undeclared allergen contamination.
        
        Helps people with allergies identify products to avoid.
        
        Args:
            allergen: Allergen name (e.g., "Peanut", "Milk", "Sesame")
            
        Returns:
            Formatted string with allergen recall alerts
            
        Example:
            >>> print(await safety.check_allergen_recalls("Peanut"))
            🥜 **Peanut Allergen Recalls** (5 found):
            
            1. 🔴 [Class I] Undeclared Peanuts
               Product: Cookie Mix, 8oz boxes
               Company: Best Bakery Inc.
               Distribution: Nationwide
               Status: Ongoing
            
        Router Trigger Keywords:
            - "allergy"
            - "allergen"
            - "peanut-free"
            - "contains peanuts"
        """
        logger.info(f"[SafetyService] Checking allergen recalls for: {allergen}")
        result = await self.food_enforcement.check_allergen_recalls(allergen)
        return result["formatted"]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def detect_intent_and_query(self, query: str) -> str:
        """
        Experimental: Automatically detect user intent and route to appropriate query.
        
        This is a placeholder for future NLP-based intent detection.
        
        Args:
            query: The user's natural language query
            
        Returns:
            The most relevant safety information
        """
        logger.info(f"[SafetyService] Processing intent query: {query}")
        
        query_lower = query.lower()
        
        # Side effects queries
        if any(keyword in query_lower for keyword in ["side effect", "adverse", "reaction"]):
            # Extract drug name (very basic extraction - would need NLP in production)
            drug_name = query.split()[-1]  # Get last word as drug name
            return await self.get_drug_side_effects(drug_name)
        
        # Safety queries
        if any(keyword in query_lower for keyword in ["safe", "dangerous", "risky"]):
            drug_name = query.split()[-1]
            return await self.check_drug_severity(drug_name)
        
        # Recall queries
        if any(keyword in query_lower for keyword in ["recall", "recalled"]):
            if "food" in query_lower or "eat" in query_lower:
                product_name = query.split()[-1]
                return await self.check_food_recalls(product_name)
            else:
                drug_name = query.split()[-1]
                return await self.check_drug_recalls(drug_name)
        
        # Allergen queries
        if any(keyword in query_lower for keyword in ["allerg", "peanut", "milk", "egg"]):
            allergen_name = query.split()[-1]
            return await self.check_allergen_recalls(allergen_name)
        
        # Default response
        return "Please ask about drug side effects, safety, recalls, or food allergens."


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

_safety_service_instance: Optional[OpenFDASafetyService] = None


def get_safety_service(api_key: Optional[str] = None) -> OpenFDASafetyService:
    """
    Get or create the singleton safety service instance.
    
    This ensures only one instance exists, preventing redundant
    initialization of sub-services.
    
    Args:
        api_key: Optional API key (only used on first call)
        
    Returns:
        The singleton OpenFDASafetyService instance
        
    Example:
        >>> safety = get_safety_service()
        >>> safety.get_drug_side_effects("Lipitor")
    """
    global _safety_service_instance
    if _safety_service_instance is None:
        logger.info("Creating new OpenFDASafetyService singleton")
        _safety_service_instance = OpenFDASafetyService(api_key)
    return _safety_service_instance


def reset_safety_service():
    """Reset the singleton (useful for testing)."""
    global _safety_service_instance
    _safety_service_instance = None
    logger.info("OpenFDASafetyService singleton reset")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def run_tests():
        print("=" * 60)
        print("OpenFDASafetyService - Interactive Test")
        print("=" * 60)
        
        safety = OpenFDASafetyService()
        
        try:
            # Drug Tests
            print("\n" + "=" * 60)
            print("DRUG SAFETY TESTS")
            print("=" * 60)
            
            print("\n📊 Test 1: Side Effects")
            print("-" * 40)
            print(await safety.get_drug_side_effects("Lipitor", limit=3))
            
            print("\n⚠️ Test 2: Severity Check")
            print("-" * 40)
            print(await safety.check_drug_severity("Warfarin"))
            
            print("\n🔍 Test 3: Verify Reaction")
            print("-" * 40)
            print(await safety.verify_drug_reaction("Lisinopril", "Cough"))
            
            print("\n📋 Test 4: Drug Recalls")
            print("-" * 40)
            print(await safety.check_drug_recalls("Metformin"))
            
            # Food Tests
            print("\n" + "=" * 60)
            print("FOOD SAFETY TESTS")
            print("=" * 60)
            
            print("\n🥗 Test 5: Food Recalls")
            print("-" * 40)
            print(await safety.check_food_recalls("Spinach"))
            
            print("\n🥜 Test 6: Allergen Recalls")
            print("-" * 40)
            print(await safety.check_allergen_recalls("Peanut"))
            
            print("\n⚠️ Test 7: Food Adverse Events")
            print("-" * 40)
            print(await safety.check_food_adverse_events("5-Hour Energy"))
        finally:
            await safety.close()
        
        print("\n" + "=" * 60)
        print("Tests Complete!")
        print("=" * 60)

    asyncio.run(run_tests())


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON PATTERN - Factory Function
# ═══════════════════════════════════════════════════════════════════════════════

_safety_service_instance: Optional['OpenFDASafetyService'] = None


def get_safety_service(api_key: Optional[str] = None) -> OpenFDASafetyService:
    """
    Get or create the singleton safety service instance.
    
    This ensures only one instance exists, preventing redundant
    initialization of sub-services and API clients.
    
    Args:
        api_key: Optional OpenFDA API key (only used on first call)
        
    Returns:
        The singleton OpenFDASafetyService instance
        
    Example:
        >>> safety = get_safety_service()
        >>> safety.get_drug_side_effects("Lipitor")
        >>> 
        >>> # All subsequent calls return the same instance
        >>> safety2 = get_safety_service()
        >>> assert safety is safety2  # True
    """
    global _safety_service_instance
    if _safety_service_instance is None:
        logger.info("Creating new OpenFDASafetyService singleton instance")
        _safety_service_instance = OpenFDASafetyService(api_key)
    return _safety_service_instance


def reset_safety_service() -> None:
    """
    Reset the singleton instance (useful for testing and reinitialization).
    
    After calling this, the next call to get_safety_service() will create
    a fresh instance.
    
    Example:
        >>> reset_safety_service()
        >>> safety = get_safety_service()  # Fresh instance
    """
    global _safety_service_instance
    logger.info("Resetting OpenFDASafetyService singleton")
    _safety_service_instance = None
