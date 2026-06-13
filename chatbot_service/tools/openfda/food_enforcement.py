"""
OpenFDA Food Safety - Enforcement & Recalls

This module handles food recall queries and allergen contamination detection.
Queries the FDA Food Enforcement Database for:
- Active/past recalls
- Allergen contamination alerts
- Recall reasons and status tracking
- Geographic distribution patterns

API Documentation: https://open.fda.gov/apis/food/enforcement/
"""


import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from tools.openfda.api_client import OpenFDAClient
from tools.openfda.models import FDAResult, FoodRecall

logger = logging.getLogger(__name__)


class FoodEnforcementQuerier:
    """Query food recalls, enforcement actions, and food safety alerts"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Food Enforcement Querier.
        
        Args:
            api_key: Optional OpenFDA API key for higher rate limits.
        """
        self.client = OpenFDAClient(api_key=api_key)
        logger.info("FoodEnforcementQuerier initialized")

    async def close(self):
        """Close the underlying client."""
        await self.client.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def check_food_recalls(
        self,
        food_item: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for recalls of a specific food product.
        
        Args:
            food_item: Food product name (e.g., "Spinach", "Romaine Lettuce")
            limit: Maximum number of recalls to return
            
        Returns:
            Dictionary containing:
            - recalls (list): List of matching recalls
            - count (int): Number of recalls found
            - formatted (str): Human-readable formatted string
            - success (bool): Whether the query succeeded
            
        Example:
            >>> querier = FoodEnforcementQuerier()
            >>> result = await querier.check_food_recalls("Spinach")
            >>> print(result["formatted"])
            🥬 **Spinach Recalls**:
            1. [Class I] E. coli contamination - Status: Ongoing
               Lot: 123456, Distributed: Nationwide
            2. [Class II] Listeria risk - Status: Completed
               Lot: 789012, Distributed: CA, OR, WA
        """
        logger.info(f"Searching food recalls for: {food_item}")
        
        # Search product description for the food item
        search_query = f'product_description:"{food_item}"'
        
        params = {
            "search": search_query,
            "limit": min(limit, 100),
            "sort": "-recall_initiation_date"  # Most recent first
        }
        
        result = await self.client._make_request("/food/enforcement.json", params)
        
        if not result.success or not result.results:
            logger.info(f"No recalls found for {food_item}")
            return {
                "food_item": food_item,
                "recalls": [],
                "count": 0,
                "formatted": f"✅ No recalls found for {food_item}.",
                "success": True
            }
        
        # Parse recalls
        recalls = []
        for item in result.results:
            recall = {
                "recall_number": item.get("recall_number", ""),
                "status": item.get("status", ""),
                "classification": item.get("classification", ""),
                "reason": item.get("reason_for_recall", ""),
                "product": item.get("product_description", ""),
                "company": item.get("recalling_firm", ""),
                "distribution": item.get("distribution_pattern", ""),
                "date": item.get("recall_initiation_date", ""),
                "state": item.get("state", ""),
            }
            recalls.append(recall)
        
        # Format output
        formatted = f"🥗 **{food_item.upper()} Recalls** ({len(recalls)} found):\n\n"
        for i, recall in enumerate(recalls, 1):
            classification_emoji = {
                "Class I": "🔴",
                "Class II": "🟡",
                "Class III": "🟢"
            }.get(recall["classification"], "")
            
            formatted += (
                f"{i}. {classification_emoji} [{recall['classification']}] {recall['reason']}\n"
                f"   Status: {recall['status']}\n"
                f"   Company: {recall['company']}\n"
                f"   Distribution: {recall['distribution']}\n"
                f"   Date: {recall['date']}\n\n"
            )
        
        logger.info(f"Found {len(recalls)} recalls for {food_item}")
        
        return {
            "food_item": food_item,
            "recalls": recalls,
            "count": len(recalls),
            "formatted": formatted,
            "success": True
        }
    
    async def check_allergen_recalls(
        self,
        allergen: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Find all food recalls related to undeclared allergen contamination.
        
        Queries for recalls where the reason mentions common allergens:
        - Peanut, Tree Nut, Milk, Egg, Soy, Wheat, Fish, Crustacean, Sesame
        
        Args:
            allergen: Allergen name (e.g., "Peanut", "Milk", "Sesame")
            limit: Maximum number of recalls to return
            
        Returns:
            Dictionary containing:
            - allergen (str): The allergen searched
            - recalls (list): List of allergen-related recalls
            - count (int): Number of recalls found
            - formatted (str): Human-readable formatted string
            - success (bool): Whether the query succeeded
            
        Example:
            >>> result = await querier.check_allergen_recalls("Peanut")
            >>> print(result["formatted"])
            🥜 **Peanut Allergen Recalls** (5 found):
            1. [Class I] Undeclared Peanuts
               Product: Cookie Mix, 8oz boxes
               Company: Best Bakery Inc.
               Distribution: Nationwide
        """
        logger.info(f"Searching allergen recalls for: {allergen}")
        
        # Build search query for undeclared allergen mentions
        search_query = (
            f'reason_for_recall:("undeclared {allergen}" OR "contains {allergen}" OR '
            f'"{allergen} contamination")'
        )
        
        params = {
            "search": search_query,
            "limit": min(limit, 100),
            "sort": "-recall_initiation_date"
        }
        
        result = await self.client._make_request("/food/enforcement.json", params)
        
        if not result.success or not result.results:
            logger.info(f"No allergen recalls found for {allergen}")
            return {
                "allergen": allergen,
                "recalls": [],
                "count": 0,
                "formatted": f"✅ No {allergen} allergen recalls found.",
                "success": True
            }
        
        # Parse recalls
        recalls = []
        for item in result.results:
            recall = {
                "recall_number": item.get("recall_number", ""),
                "status": item.get("status", ""),
                "classification": item.get("classification", ""),
                "reason": item.get("reason_for_recall", ""),
                "product": item.get("product_description", ""),
                "company": item.get("recalling_firm", ""),
                "distribution": item.get("distribution_pattern", ""),
                "date": item.get("recall_initiation_date", ""),
            }
            recalls.append(recall)
        
        # Format output with allergen emoji
        allergen_emojis = {
            "peanut": "🥜",
            "tree nut": "🌰",
            "milk": "🥛",
            "egg": "🥚",
            "soy": "🌾",
            "wheat": "🌾",
            "fish": "🐟",
            "crustacean": "🦐",
            "sesame": "🌻"
        }
        emoji = allergen_emojis.get(allergen.lower(), "⚠️")
        
        formatted = f"{emoji} **{allergen.upper()} Allergen Recalls** ({len(recalls)} found):\n\n"
        for i, recall in enumerate(recalls, 1):
            classification_emoji = {
                "Class I": "🔴",
                "Class II": "🟡",
                "Class III": "🟢"
            }.get(recall["classification"], "")
            
            formatted += (
                f"{i}. {classification_emoji} [{recall['classification']}] {recall['reason']}\n"
                f"   Product: {recall['product']}\n"
                f"   Company: {recall['company']}\n"
                f"   Distribution: {recall['distribution']}\n"
                f"   Status: {recall['status']}\n\n"
            )
        
        logger.info(f"Found {len(recalls)} allergen recalls for {allergen}")
        
        return {
            "allergen": allergen,
            "recalls": recalls,
            "count": len(recalls),
            "formatted": formatted,
            "success": True
        }
    
    async def get_active_recalls(
        self,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get all currently active food recalls (status = "Ongoing").
        
        Args:
            limit: Maximum number of recalls to return
            
        Returns:
            Dictionary containing active recalls
        """
        logger.info("Searching active food recalls...")
        
        search_query = 'status:"Ongoing"'
        params = {
            "search": search_query,
            "limit": min(limit, 100),
            "sort": "-recall_initiation_date"
        }
        
        result = await self.client._make_request("/food/enforcement.json", params)
        
        if not result.success or not result.results:
            logger.info("No active food recalls found")
            return {
                "recalls": [],
                "count": 0,
                "formatted": "✅ No active food recalls currently.",
                "success": True
            }
        
        recalls = []
        for item in result.results:
            recall = {
                "recall_number": item.get("recall_number", ""),
                "status": item.get("status", ""),
                "classification": item.get("classification", ""),
                "reason": item.get("reason_for_recall", ""),
                "product": item.get("product_description", ""),
                "company": item.get("recalling_firm", ""),
                "date": item.get("recall_initiation_date", ""),
            }
            recalls.append(recall)
        
        formatted = f"🚨 **Currently Active Food Recalls** ({len(recalls)} total):\n\n"
        for i, recall in enumerate(recalls[:10], 1):  # Show top 10
            formatted += (
                f"{i}. [{recall['classification']}] {recall['reason']}\n"
                f"   {recall['product']}\n"
                f"   Company: {recall['company']}\n\n"
            )
        
        if len(recalls) > 10:
            formatted += f"\n... and {len(recalls) - 10} more"
        
        return {
            "recalls": recalls,
            "count": len(recalls),
            "formatted": formatted,
            "success": True
        }
    
    async def get_class_i_recalls(
        self,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get Class I (most serious) food recalls.
        
        Class I indicates serious potential for adverse health consequences or death.
        
        Args:
            limit: Maximum number of recalls to return
            
        Returns:
            Dictionary containing Class I recalls
        """
        logger.info("Searching Class I food recalls...")
        
        search_query = 'classification:"Class I"'
        params = {
            "search": search_query,
            "limit": min(limit, 100),
            "sort": "-recall_initiation_date"
        }
        
        result = await self.client._make_request("/food/enforcement.json", params)
        
        if not result.success or not result.results:
            logger.info("No Class I food recalls found")
            return {
                "recalls": [],
                "count": 0,
                "formatted": "✅ No Class I food recalls found.",
                "success": True
            }
        
        recalls = []
        for item in result.results:
            recall = {
                "recall_number": item.get("recall_number", ""),
                "status": item.get("status", ""),
                "reason": item.get("reason_for_recall", ""),
                "product": item.get("product_description", ""),
                "company": item.get("recalling_firm", ""),
                "distribution": item.get("distribution_pattern", ""),
                "date": item.get("recall_initiation_date", ""),
            }
            recalls.append(recall)
        
        formatted = f"🔴 **CLASS I FOOD RECALLS** ({len(recalls)} total):\n\n"
        for i, recall in enumerate(recalls, 1):
            formatted += (
                f"{i}. {recall['reason']}\n"
                f"   Product: {recall['product']}\n"
                f"   Company: {recall['company']}\n"
                f"   Distribution: {recall['distribution']}\n"
                f"   Status: {recall['status']}\n\n"
            )
        
        return {
            "recalls": recalls,
            "count": len(recalls),
            "formatted": formatted,
            "success": True
        }
    
    async def get_recalls_by_company(
        self,
        company_name: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get all recalls from a specific food company.
        
        Args:
            company_name: Company name to search for
            limit: Maximum number of recalls to return
            
        Returns:
            Dictionary containing company recalls
        """
        logger.info(f"Searching recalls for company: {company_name}")
        
        search_query = f'recalling_firm:"{company_name}"'
        params = {
            "search": search_query,
            "limit": min(limit, 100),
            "sort": "-recall_initiation_date"
        }
        
        result = await self.client._make_request("/food/enforcement.json", params)
        
        if not result.success or not result.results:
            logger.info(f"No recalls found for company: {company_name}")
            return {
                "company": company_name,
                "recalls": [],
                "count": 0,
                "formatted": f"No recalls found for {company_name}.",
                "success": True
            }
        
        recalls = []
        for item in result.results:
            recall = {
                "recall_number": item.get("recall_number", ""),
                "status": item.get("status", ""),
                "classification": item.get("classification", ""),
                "reason": item.get("reason_for_recall", ""),
                "product": item.get("product_description", ""),
                "date": item.get("recall_initiation_date", ""),
            }
            recalls.append(recall)
        
        formatted = f"🏢 **{company_name} Recalls** ({len(recalls)} found):\n\n"
        for i, recall in enumerate(recalls, 1):
            formatted += (
                f"{i}. [{recall['classification']}] {recall['reason']}\n"
                f"   {recall['product']}\n"
                f"   Status: {recall['status']}\n\n"
            )
        
        return {
            "company": company_name,
            "recalls": recalls,
            "count": len(recalls),
            "formatted": formatted,
            "success": True
        }
    
    async def get_recalls_by_reason(
        self,
        reason: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get food recalls filtered by reason (e.g., contamination, mislabeling).
        
        Args:
            reason: Reason keyword (e.g., "Listeria", "E. coli", "Allergen")
            limit: Maximum number of recalls to return
            
        Returns:
            Dictionary containing recalls matching the reason
        """
        logger.info(f"Searching recalls with reason: {reason}")
        
        search_query = f'reason_for_recall:"{reason}"'
        params = {
            "search": search_query,
            "limit": min(limit, 100),
            "sort": "-recall_initiation_date"
        }
        
        result = await self.client._make_request("/food/enforcement.json", params)
        
        if not result.success or not result.results:
            logger.info(f"No recalls found with reason: {reason}")
            return {
                "reason": reason,
                "recalls": [],
                "count": 0,
                "formatted": f"No recalls found for reason: {reason}.",
                "success": True
            }
        
        recalls = []
        for item in result.results:
            recall = {
                "recall_number": item.get("recall_number", ""),
                "status": item.get("status", ""),
                "classification": item.get("classification", ""),
                "product": item.get("product_description", ""),
                "company": item.get("recalling_firm", ""),
                "date": item.get("recall_initiation_date", ""),
            }
            recalls.append(recall)
        
        formatted = f"⚠️ **Food Recalls: {reason.upper()}** ({len(recalls)} found):\n\n"
        for i, recall in enumerate(recalls[:15], 1):  # Show top 15
            formatted += (
                f"{i}. [{recall['classification']}] {recall['product']}\n"
                f"   Company: {recall['company']}\n"
                f"   Status: {recall['status']}\n\n"
            )
        
        if len(recalls) > 15:
            formatted += f"\n... and {len(recalls) - 15} more"
        
        return {
            "reason": reason,
            "recalls": recalls,
            "count": len(recalls),
            "formatted": formatted,
            "success": True
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def check_food_recalls(food_item: str) -> str:
    """Quick function to check food recalls."""
    querier = FoodEnforcementQuerier()
    try:
        result = await querier.check_food_recalls(food_item)
        return result["formatted"]
    finally:
        await querier.close()


async def check_allergen_recalls(allergen: str) -> str:
    """Quick function to check allergen recalls."""
    querier = FoodEnforcementQuerier()
    try:
        result = await querier.check_allergen_recalls(allergen)
        return result["formatted"]
    finally:
        await querier.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def run_tests():
        print("=" * 60)
        print("FoodEnforcementQuerier - Interactive Test")
        print("=" * 60)
        
        querier = FoodEnforcementQuerier()
        
        try:
            # Test 1: Food recalls
            print("\n🥬 Test 1: Spinach Recalls")
            print("-" * 40)
            result = await querier.check_food_recalls("Spinach", limit=5)
            print(result["formatted"])
            
            # Test 2: Allergen recalls
            print("\n🥜 Test 2: Peanut Allergen Recalls")
            print("-" * 40)
            result = await querier.check_allergen_recalls("Peanut", limit=5)
            print(result["formatted"])
            
            # Test 3: Active recalls
            print("\n🚨 Test 3: Currently Active Recalls")
            print("-" * 40)
            result = await querier.get_active_recalls(limit=5)
            print(result["formatted"])
            
            # Test 4: Class I recalls
            print("\n🔴 Test 4: Class I (Most Serious) Recalls")
            print("-" * 40)
            result = await querier.get_class_i_recalls(limit=5)
            print(result["formatted"])
        finally:
            await querier.close()
        
        print("\n" + "=" * 60)
        print("Tests Complete!")
        print("=" * 60)

    asyncio.run(run_tests())
