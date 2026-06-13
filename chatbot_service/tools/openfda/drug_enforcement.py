"""
OpenFDA Drug Enforcement & Recall Queries

This module handles:
- Drug recalls
- Safety alerts
- Market withdrawals
- Adverse event information
"""


import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime

from tools.openfda.api_client import OpenFDAClient
from tools.openfda.models import DrugRecall, RecallStatus

logger = logging.getLogger(__name__)


class DrugEnforcementQuerier:
    """Query drug recalls, enforcement actions, and safety alerts"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenFDAClient(api_key=api_key)

    async def close(self):
        """Close the underlying client."""
        await self.client.close()
    
    async def find_recalls_for_drug(
        self,
        drug_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find all recalls for a specific drug.
        
        Args:
            drug_name: Drug name to search for
            limit: Max results
        
        Returns:
            List of recall records
        """
        logger.info(f"Searching recalls for drug: {drug_name}")
        
        # Search for drug name in product description
        search_query = f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"'
        result = await self.client.search_drug_enforcement(search_query, limit=limit)
        
        if not result.success:
            logger.warning(f"No recalls found for {drug_name}")
            return []
        
        logger.info(f"Found {len(result.results)} recall entries")
        return result.results
    
    async def find_active_recalls(
        self,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Find all currently active recalls.
        
        Args:
            limit: Max results
        
        Returns:
            List of active recall records
        """
        logger.info("Searching active recalls...")
        
        # Search for ongoing recalls
        search_query = 'status:"Ongoing"'
        result = await self.client.search_drug_enforcement(
            search_query,
            limit=limit,
            sort="-recall_initiation_date"  # Most recent first
        )
        
        if not result.success:
            logger.warning("No active recalls found")
            return []
        
        logger.info(f"Found {len(result.results)} active recalls")
        return result.results
    
    async def find_recalls_by_classification(
        self,
        classification: str,  # "Class I", "Class II", or "Class III"
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find recalls by severity classification.
        
        Classification levels:
        - Class I: Serious, potentially fatal health consequences
        - Class II: Health consequences or violations unlikely to cause adverse health
        - Class III: Unlikely to cause adverse health consequences
        
        Args:
            classification: Recall classification
            limit: Max results
        
        Returns:
            List of matching recalls
        """
        logger.info(f"Searching {classification} recalls...")
        
        search_query = f'classification:"{classification}"'
        result = await self.client.search_drug_enforcement(
            search_query,
            limit=limit,
            sort="-recall_initiation_date"
        )
        
        if not result.success:
            logger.warning(f"No {classification} recalls found")
            return []
        
        logger.info(f"Found {len(result.results)} {classification} recalls")
        return result.results
    
    async def find_recalls_by_reason(
        self,
        reason: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find recalls by reason (e.g., contamination, impurity, mislabeling).
        
        Args:
            reason: Recall reason to search for
            limit: Max results
        
        Returns:
            List of matching recalls
        """
        logger.info(f"Searching recalls with reason: {reason}")
        
        search_query = f'recall_reason:"{reason}"'
        result = await self.client.search_drug_enforcement(search_query, limit=limit)
        
        if not result.success:
            logger.warning(f"No recalls found with reason {reason}")
            return []
        
        logger.info(f"Found {len(result.results)} recalls with reason {reason}")
        return result.results
    
    async def find_nationwide_recalls(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find nationwide recalls (vs. localized)"""
        logger.info("Searching nationwide recalls...")
        
        search_query = 'distribution_pattern:"Nationwide"'
        result = await self.client.search_drug_enforcement(
            search_query,
            limit=limit,
            sort="-recall_initiation_date"
        )
        
        if not result.success:
            logger.warning("No nationwide recalls found")
            return []
        
        logger.info(f"Found {len(result.results)} nationwide recalls")
        return result.results
    
    async def find_recent_recalls(
        self,
        days: int = 30,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find recalls initiated within last N days.
        
        Args:
            days: Number of days to look back
            limit: Max results
        
        Returns:
            List of recent recalls
        """
        logger.info(f"Searching recalls from last {days} days...")
        
        # Calculate date range (OpenFDA uses YYYYMMDD format)
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        search_query = f'recall_initiation_date:[{start_str} TO {end_str}]'
        result = await self.client.search_drug_enforcement(
            search_query,
            limit=limit,
            sort="-recall_initiation_date"
        )
        
        if not result.success:
            logger.warning(f"No recalls found in last {days} days")
            return []
        
        logger.info(f"Found {len(result.results)} recalls in last {days} days")
        return result.results
    
    async def find_recalls_by_manufacturer(
        self,
        manufacturer_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find all recalls issued by a specific manufacturer"""
        logger.info(f"Searching recalls by manufacturer: {manufacturer_name}")
        
        search_query = f'openfda.manufacturer_name:"{manufacturer_name}"'
        result = await self.client.search_drug_enforcement(search_query, limit=limit)
        
        if not result.success:
            logger.warning(f"No recalls found for manufacturer {manufacturer_name}")
            return []
        
        logger.info(f"Found {len(result.results)} recalls by {manufacturer_name}")
        return result.results
    
    async def check_drug_safety(
        self,
        drug_name: str,
    ) -> Dict[str, Any]:
        """
        Comprehensive safety check for a drug.
        
        Returns:
            Dictionary with:
            - has_recalls: bool
            - active_recalls: int
            - class_i_recalls: int
            - most_recent_recall: date
            - details: List of recall details
        """
        logger.info(f"Performing safety check for {drug_name}...")
        
        recalls = await self.find_recalls_for_drug(drug_name, limit=50)
        
        class_i_count = sum(1 for r in recalls if r.get('classification') == 'Class I')
        active_count = sum(1 for r in recalls if r.get('status') == 'Ongoing')
        
        most_recent = None
        if recalls:
            dates = [r.get('recall_initiation_date') for r in recalls if r.get('recall_initiation_date')]
            if dates:
                most_recent = max(dates)
        
        return {
            'drug_name': drug_name,
            'has_recalls': len(recalls) > 0,
            'total_recalls': len(recalls),
            'active_recalls': active_count,
            'class_i_recalls': class_i_count,
            'most_recent_recall': most_recent,
            'details': recalls[:5],  # Top 5 most relevant
        }


class FoodEnforcementQuerier:
    """Query food recalls and enforcement actions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenFDAClient(api_key=api_key)

    async def close(self):
        """Close the underlying client."""
        await self.client.close()
    
    async def find_food_recalls(
        self,
        product_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find food recalls for a product"""
        logger.info(f"Searching food recalls for: {product_name}")
        
        search_query = f'product_description:"{product_name}"'
        result = await self.client.search_food_enforcement(search_query, limit=limit)
        
        if not result.success:
            logger.warning(f"No food recalls found for {product_name}")
            return []
        
        logger.info(f"Found {len(result.results)} food recall entries")
        return result.results
    
    async def find_active_food_recalls(
        self,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find all active food recalls"""
        logger.info("Searching active food recalls...")
        
        search_query = 'status:"Ongoing"'
        result = await self.client.search_food_enforcement(
            search_query,
            limit=limit,
            sort="-recall_initiation_date"
        )
        
        if not result.success:
            logger.warning("No active food recalls found")
            return []
        
        logger.info(f"Found {len(result.results)} active food recalls")
        return result.results
    
    async def find_recalls_by_reason(
        self,
        reason: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find food recalls by reason (contamination, allergen, etc.)"""
        logger.info(f"Searching food recalls with reason: {reason}")
        
        search_query = f'recall_reason:"{reason}"'
        result = await self.client.search_food_enforcement(search_query, limit=limit)
        
        if not result.success:
            logger.warning(f"No food recalls found with reason {reason}")
            return []
        
        logger.info(f"Found {len(result.results)} food recalls with reason {reason}")
        return result.results
    
    async def find_nationwide_food_recalls(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find nationwide food recalls"""
        logger.info("Searching nationwide food recalls...")
        
        search_query = 'distribution_pattern:"Nationwide"'
        result = await self.client.search_food_enforcement(search_query, limit=limit)
        
        if not result.success:
            logger.warning("No nationwide food recalls found")
            return []
        
        logger.info(f"Found {len(result.results)} nationwide food recalls")
        return result.results
    
    async def find_recalls_by_allergen(
        self,
        allergen: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find food recalls due to allergen contamination"""
        logger.info(f"Searching food recalls due to {allergen}...")
        
        search_query = f'reason_for_recall:"{allergen}"'
        result = await self.client.search_food_enforcement(search_query, limit=limit)
        
        if not result.success:
            logger.warning(f"No food recalls found due to {allergen}")
            return []
        
        logger.info(f"Found {len(result.results)} food recalls due to {allergen}")
        return result.results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def run_tests():
        # Test drug enforcement
        print("Testing Drug Enforcement Queries")
        print("=" * 50)
        
        enforcer = DrugEnforcementQuerier()
        
        try:
            # Test: Find recalls for a drug
            print("\n1. Finding recalls for Metoprolol...")
            recalls = await enforcer.find_recalls_for_drug("Metoprolol", limit=5)
            print(f"Found {len(recalls)} recalls")
            
            # Test: Find active recalls
            print("\n2. Finding active recalls...")
            active = await enforcer.find_active_recalls(limit=5)
            print(f"Found {len(active)} active recalls")
            
            # Test: Comprehensive safety check
            print("\n3. Safety check for Aspirin...")
            safety = await enforcer.check_drug_safety("Aspirin")
            print(f"Safety check results: {safety}")
        finally:
            await enforcer.close()
        
        # Test food recalls
        print("\n\nTesting Food Enforcement Queries")
        print("=" * 50)
        
        food_enforcer = FoodEnforcementQuerier()
        
        try:
            print("\n1. Finding active food recalls...")
            active_food = await food_enforcer.find_active_food_recalls(limit=3)
            print(f"Found {len(active_food)} active food recalls")
        finally:
            await food_enforcer.close()

    asyncio.run(run_tests())

