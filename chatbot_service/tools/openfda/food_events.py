"""
OpenFDA Food Safety - CAERS Adverse Events

This module handles food and supplement adverse event queries.
Queries the CAERS database (Center for Food Safety and Applied Nutrition Adverse Event Reporting System).

CAERS collects adverse event reports for:
- Dietary supplements
- Conventional foods
- Cosmetics (separate endpoint)

API Documentation: https://open.fda.gov/apis/food/event/
"""


import logging
import asyncio
from typing import List, Dict, Optional, Any

from tools.openfda.api_client import OpenFDAClient
from tools.openfda.models import FDAResult, FoodEvent

logger = logging.getLogger(__name__)


class FoodEventService:
    """Query food and supplement adverse events (CAERS)"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Food Event Service.
        
        Args:
            api_key: Optional OpenFDA API key for higher rate limits.
        """
        self.client = OpenFDAClient(api_key=api_key)
        logger.info("FoodEventService initialized")

    async def close(self):
        """Close the underlying client."""
        await self.client.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def check_supplement_adverse_events(
        self,
        supplement_name: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Check for reported adverse events from a dietary supplement.
        
        Args:
            supplement_name: Supplement product name (e.g., "5-Hour Energy", "Garcinia")
            limit: Maximum number of events to return
            
        Returns:
            Dictionary containing:
            - supplement_name (str): The supplement searched
            - event_count (int): Number of adverse events found
            - serious_outcomes (int): Count of serious outcomes
            - events (list): List of adverse event details
            - formatted (str): Human-readable formatted string
            - success (bool): Whether the query succeeded
            
        Example:
            >>> service = FoodEventService()
            >>> result = await service.check_supplement_adverse_events("5-Hour Energy")
            >>> print(result["formatted"])
            ⚠️ **5-Hour Energy Adverse Events** (234 reported):
            Top reactions:
            - Heart palpitations: 89 reports
            - Chest pain: 67 reports
            - Anxiety: 45 reports
            Serious outcomes: 12 (hospitalization, disability)
        """
        logger.info(f"Checking adverse events for supplement: {supplement_name}")
        
        # Search for supplement by name
        search_query = f'products.name_brand:"{supplement_name}"'
        params = {
            "search": search_query,
            "limit": 1,  # Just get total count first
        }
        
        result = await self.client._make_request("/food/event.json", params)
        
        if not result.success or result.meta.get("results", {}).get("total", 0) == 0:
            logger.info(f"No adverse events found for supplement: {supplement_name}")
            return {
                "supplement_name": supplement_name,
                "event_count": 0,
                "serious_outcomes": 0,
                "events": [],
                "formatted": f"✅ No adverse events reported for {supplement_name}.",
                "success": True
            }
        
        total_events = result.meta.get("results", {}).get("total", 0)
        
        # Get top reactions for this supplement
        params_reactions = {
            "search": search_query,
            "count": "reactions",
            "limit": min(limit, 100)
        }
        result_reactions = await self.client._make_request("/food/event.json", params_reactions)
        
        reactions = []
        if result_reactions.success:
            reactions = [
                {
                    "reaction": item.get("term", "Unknown"),
                    "count": item.get("count", 0)
                }
                for item in result_reactions.results
            ]
        
        # Check for serious outcomes
        search_serious = (
            f'products.name_brand:"{supplement_name}" AND '
            f'outcomes:("Hospitalization" OR "Death" OR "Life Threatening" OR "Disability")'
        )
        params_serious = {"search": search_serious, "limit": 1}
        result_serious = await self.client._make_request("/food/event.json", params_serious)
        serious_count = result_serious.meta.get("results", {}).get("total", 0) if result_serious.success else 0
        
        # Format output
        formatted = f"⚠️ **{supplement_name.upper()} Adverse Events** ({total_events} reported):\n\n"
        
        if reactions:
            formatted += "Top Reported Reactions:\n"
            for i, reaction in enumerate(reactions[:5], 1):
                formatted += f"  {i}. {reaction['reaction']}: {reaction['count']} reports\n"
            formatted += "\n"
        
        if serious_count > 0:
            formatted += f"🚨 **Serious Outcomes:** {serious_count} reports (hospitalization, death, life-threatening, or disability)\n"
        else:
            formatted += "✅ No serious outcomes (hospitalization/death/disability) reported.\n"
        
        logger.info(f"Found {total_events} adverse events for {supplement_name}, {serious_count} serious")
        
        return {
            "supplement_name": supplement_name,
            "event_count": total_events,
            "serious_outcomes": serious_count,
            "top_reactions": reactions[:5],
            "formatted": formatted,
            "success": True
        }
    
    async def check_food_adverse_events(
        self,
        food_product: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Check for reported adverse events from a food product.
        
        Args:
            food_product: Food product name
            limit: Maximum number of events to return
            
        Returns:
            Dictionary containing adverse events information
        """
        logger.info(f"Checking adverse events for food product: {food_product}")
        
        # Search for food product
        search_query = f'products.name_brand:"{food_product}"'
        params = {
            "search": search_query,
            "limit": 1,
        }
        
        result = await self.client._make_request("/food/event.json", params)
        
        if not result.success or result.meta.get("results", {}).get("total", 0) == 0:
            logger.info(f"No adverse events found for food product: {food_product}")
            return {
                "food_product": food_product,
                "event_count": 0,
                "serious_outcomes": 0,
                "formatted": f"✅ No adverse events reported for {food_product}.",
                "success": True
            }
        
        total_events = result.meta.get("results", {}).get("total", 0)
        
        # Get top reactions
        params_reactions = {
            "search": search_query,
            "count": "reactions",
            "limit": min(limit, 100)
        }
        result_reactions = await self.client._make_request("/food/event.json", params_reactions)
        
        reactions = []
        if result_reactions.success:
            reactions = [
                {
                    "reaction": item.get("term", "Unknown"),
                    "count": item.get("count", 0)
                }
                for item in result_reactions.results
            ]
        
        # Check for serious outcomes
        search_serious = (
            f'products.name_brand:"{food_product}" AND '
            f'outcomes:("Hospitalization" OR "Death" OR "Life Threatening" OR "Disability")'
        )
        params_serious = {"search": search_serious, "limit": 1}
        result_serious = await self.client._make_request("/food/event.json", params_serious)
        serious_count = result_serious.meta.get("results", {}).get("total", 0) if result_serious.success else 0
        
        # Format output
        formatted = f"🍽️ **{food_product.upper()} Adverse Events** ({total_events} reported):\n\n"
        
        if reactions:
            formatted += "Top Reported Reactions:\n"
            for i, reaction in enumerate(reactions[:5], 1):
                formatted += f"  {i}. {reaction['reaction']}: {reaction['count']} reports\n"
            formatted += "\n"
        
        if serious_count > 0:
            formatted += f"🚨 **Serious Outcomes:** {serious_count} reports\n"
        else:
            formatted += "✅ No serious outcomes reported.\n"
        
        return {
            "food_product": food_product,
            "event_count": total_events,
            "serious_outcomes": serious_count,
            "top_reactions": reactions[:5],
            "formatted": formatted,
            "success": True
        }
    
    async def get_serious_adverse_events(
        self,
        outcome_type: str = "Hospitalization",
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get food/supplement adverse events with serious outcomes.
        
        Args:
            outcome_type: Type of serious outcome ("Hospitalization", "Death", "Life Threatening", "Disability")
            limit: Maximum number of events to return
            
        Returns:
            Dictionary containing serious adverse events
        """
        logger.info(f"Searching for serious adverse events: {outcome_type}")
        
        search_query = f'outcomes:"{outcome_type}"'
        params = {
            "search": search_query,
            "limit": 1,
        }
        
        result = await self.client._make_request("/food/event.json", params)
        
        if not result.success or result.meta.get("results", {}).get("total", 0) == 0:
            logger.info(f"No {outcome_type} events found")
            return {
                "outcome_type": outcome_type,
                "event_count": 0,
                "formatted": f"No {outcome_type} adverse events found.",
                "success": True
            }
        
        total_events = result.meta.get("results", {}).get("total", 0)
        
        # Get top products associated with this outcome
        params_products = {
            "search": search_query,
            "count": "products.name_brand.exact",
            "limit": min(limit, 100)
        }
        result_products = await self.client._make_request("/food/event.json", params_products)
        
        products = []
        if result_products.success:
            products = [
                {
                    "product": item.get("term", "Unknown"),
                    "count": item.get("count", 0)
                }
                for item in result_products.results
            ]
        
        # Format output
        outcome_emoji = {
            "Hospitalization": "🏥",
            "Death": "☠️",
            "Life Threatening": "⚠️",
            "Disability": "♿"
        }.get(outcome_type, "🚨")
        
        formatted = f"{outcome_emoji} **{outcome_type.upper()} Adverse Events** ({total_events} total):\n\n"
        
        if products:
            formatted += f"Top Products Associated with {outcome_type}:\n"
            for i, product in enumerate(products[:10], 1):
                formatted += f"{i}. {product['product']}: {product['count']} reports\n"
        
        return {
            "outcome_type": outcome_type,
            "event_count": total_events,
            "top_products": products[:10],
            "formatted": formatted,
            "success": True
        }
    
    async def get_supplements_by_adverse_events(
        self,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get dietary supplements ranked by number of adverse events.
        
        Args:
            limit: Number of supplements to return
            
        Returns:
            Dictionary with top supplements by adverse event count
        """
        logger.info("Ranking supplements by adverse event count")
        
        # Count events by supplement
        params = {
            "search": 'products.industry_name:"Dietary Supplement"',
            "count": "products.name_brand.exact",
            "limit": min(limit, 100)
        }
        
        result = await self.client._make_request("/food/event.json", params)
        
        if not result.success or not result.results:
            logger.info("No dietary supplements found")
            return {
                "supplements": [],
                "formatted": "No dietary supplements with adverse events found.",
                "success": True
            }
        
        supplements = [
            {
                "name": item.get("term", "Unknown"),
                "event_count": item.get("count", 0)
            }
            for item in result.results
        ]
        
        # Format output
        formatted = "📊 **Top Dietary Supplements by Adverse Events**:\n\n"
        for i, supp in enumerate(supplements, 1):
            formatted += f"{i}. {supp['name']}: {supp['event_count']} adverse events\n"
        
        return {
            "supplements": supplements,
            "count": len(supplements),
            "formatted": formatted,
            "success": True
        }
    
    async def search_adverse_events_by_reaction(
        self,
        reaction: str,
        product_type: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search for products reported to cause a specific adverse reaction.
        
        Args:
            reaction: The adverse reaction/symptom (e.g., "Heart palpitations")
            product_type: Optional filter ("Dietary Supplement", "Food", etc.)
            limit: Maximum products to return
            
        Returns:
            Dictionary with products causing this reaction
        """
        logger.info(f"Searching products causing: {reaction}")
        
        search_query = f'reactions:"{reaction}"'
        if product_type:
            search_query += f' AND products.industry_name:"{product_type}"'
        
        params = {
            "search": search_query,
            "count": "products.name_brand.exact",
            "limit": min(limit, 100)
        }
        
        result = await self.client._make_request("/food/event.json", params)
        
        if not result.success or not result.results:
            logger.info(f"No products found causing {reaction}")
            return {
                "reaction": reaction,
                "products": [],
                "formatted": f"No products reported to cause {reaction}.",
                "success": True
            }
        
        products = [
            {
                "name": item.get("term", "Unknown"),
                "count": item.get("count", 0)
            }
            for item in result.results
        ]
        
        # Format output
        formatted = f"⚠️ **Products Reported to Cause {reaction.upper()}**:\n\n"
        for i, prod in enumerate(products, 1):
            formatted += f"{i}. {prod['name']}: {prod['count']} reports\n"
        
        return {
            "reaction": reaction,
            "products": products,
            "formatted": formatted,
            "success": True
        }
    
    async def get_food_adverse_events_by_demographic(
        self,
        age_group: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get adverse events filtered by demographic (age, gender).
        
        Args:
            age_group: Age range filter (e.g., "pediatric", "geriatric")
            limit: Maximum results to return
            
        Returns:
            Dictionary with demographic-filtered adverse events
        """
        logger.info(f"Searching adverse events by demographic: {age_group}")
        
        search_query = ""
        if age_group:
            search_query = f'consumer.age_group:"{age_group}"'
        
        params = {
            "search": search_query if search_query else "*",
            "count": "products.name_brand.exact",
            "limit": min(limit, 100)
        }
        
        result = await self.client._make_request("/food/event.json", params)
        
        if not result.success or not result.results:
            logger.info("No demographic-specific adverse events found")
            return {
                "age_group": age_group,
                "events": [],
                "formatted": f"No events found for age group: {age_group}.",
                "success": True
            }
        
        events = [
            {
                "product": item.get("term", "Unknown"),
                "count": item.get("count", 0)
            }
            for item in result.results
        ]
        
        # Format output
        formatted = f"📊 **Adverse Events - {age_group or 'All Ages'}**:\n\n"
        for i, event in enumerate(events[:15], 1):
            formatted += f"{i}. {event['product']}: {event['count']} reports\n"
        
        if len(events) > 15:
            formatted += f"\n... and {len(events) - 15} more"
        
        return {
            "age_group": age_group,
            "events": events,
            "formatted": formatted,
            "success": True
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def check_supplement_events(supplement_name: str) -> str:
    """Quick function to check supplement adverse events."""
    service = FoodEventService()
    try:
        result = await service.check_supplement_adverse_events(supplement_name)
        return result["formatted"]
    finally:
        await service.close()


async def check_food_events(food_product: str) -> str:
    """Quick function to check food adverse events."""
    service = FoodEventService()
    try:
        result = await service.check_food_adverse_events(food_product)
        return result["formatted"]
    finally:
        await service.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def run_tests():
        print("=" * 60)
        print("FoodEventService - Interactive Test")
        print("=" * 60)
        
        service = FoodEventService()
        
        try:
            # Test 1: Supplement adverse events
            print("\n⚠️ Test 1: 5-Hour Energy Adverse Events")
            print("-" * 40)
            result = await service.check_supplement_adverse_events("5-Hour Energy", limit=5)
            print(result["formatted"])
            
            # Test 2: Food adverse events
            print("\n🍽️ Test 2: Food Adverse Events")
            print("-" * 40)
            result = await service.check_food_adverse_events("Peanut Butter", limit=5)
            print(result["formatted"])
            
            # Test 3: Serious outcomes
            print("\n🏥 Test 3: Hospitalization Events")
            print("-" * 40)
            result = await service.get_serious_adverse_events("Hospitalization", limit=5)
            print(result["formatted"])
            
            # Test 4: Top supplements by events
            print("\n📊 Test 4: Top Supplements by Adverse Events")
            print("-" * 40)
            result = await service.get_supplements_by_adverse_events(limit=10)
            print(result["formatted"])
        finally:
            await service.close()
        
        print("\n" + "=" * 60)
        print("Tests Complete!")
        print("=" * 60)

    asyncio.run(run_tests())
