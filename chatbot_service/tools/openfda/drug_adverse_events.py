"""
OpenFDA Drug Adverse Event Queries (FAERS Database)

This module handles real-time queries to the FDA Adverse Event Reporting System.
No local data storage required - all queries are live API calls.

API Documentation: https://open.fda.gov/apis/drug/event/

Key Concepts:
- FAERS: FDA Adverse Event Reporting System
- MedDRA: Medical Dictionary for Regulatory Activities (standardized terms)
- Seriousness flags: death, hospitalization, life-threatening, disability
"""


import logging
import asyncio
from typing import Dict, Any, List, Optional

from tools.openfda.api_client import OpenFDAClient
from tools.openfda.models import FDAResult, SideEffect, SeverityStats

logger = logging.getLogger(__name__)


class DrugAdverseEventService:
    """
    Real-time interface to the FDA Adverse Event Reporting System (FAERS).
    
    The FAERS database contains 14+ million reports of drug side effects,
    hospitalizations, and deaths submitted by healthcare providers,
    patients, and manufacturers.
    
    This service provides:
    1. Top side effects for any drug (count mode)
    2. Verification if a drug causes a specific reaction (search mode)
    3. Severity statistics (death/hospitalization counts)
    4. Demographic breakdowns (age/sex distribution)
    
    Example Usage:
        >>> service = DrugAdverseEventService()
        >>> result = await service.get_top_side_effects("Lipitor")
        >>> print(result["formatted"])
        📊 Top 10 Reported Side Effects for LIPITOR:
        1. Myalgia: 12,500 reports
        2. Pain In Extremity: 8,200 reports
        ...
    
    Note:
        - No API key required for basic usage (rate limited)
        - With API key: 1,000+ requests/minute
        - Without API key: 240 requests/minute
    """
    
    ENDPOINT = "/drug/event.json"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Drug Adverse Event Service.
        
        Args:
            api_key: Optional OpenFDA API key for higher rate limits.
        """
        self.client = OpenFDAClient(api_key=api_key)
        logger.info("DrugAdverseEventService initialized")

    async def close(self):
        """Close the underlying client."""
        await self.client.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_top_side_effects(
        self,
        drug_name: str,
        limit: int = 10,
        by_generic: bool = False
    ) -> Dict[str, Any]:
        """
        Get the most frequently reported side effects for a drug.
        
        Uses the COUNT query mode to aggregate all adverse event reports
        and return the most common reactions.
        
        Args:
            drug_name: Brand name (e.g., "Lipitor") or generic (e.g., "atorvastatin")
            limit: Number of side effects to return (default: 10, max: 1000)
            by_generic: If True, search by generic name; else by brand name
            
        Returns:
            Dictionary containing:
            - side_effects (list): List of {term, count} sorted by frequency
            - total_reports (int): Total number of adverse event reports
            - drug_name (str): The drug searched
            - formatted (str): Human-readable formatted string
            - success (bool): Whether the query succeeded
            
        Example:
            >>> result = await service.get_top_side_effects("Lipitor", limit=5)
            >>> print(result["formatted"])
            📊 Top 5 Reported Side Effects for LIPITOR:
            1. Myalgia: 12,500 reports
            2. Pain In Extremity: 8,200 reports
            3. Diarrhea: 6,100 reports
            4. Nausea: 5,800 reports
            5. Rhabdomyolysis: 5,100 reports
            
        API Query Generated:
            /drug/event.json?search=patient.drug.medicinalproduct:"LIPITOR"
            &count=patient.reaction.reactionmeddrapt.exact
            &limit=10
        """
        logger.info(f"Querying top side effects for: {drug_name}")
        
        # Choose field based on search type
        if by_generic:
            search_field = "patient.drug.activesubstance.activesubstancename"
            logger.debug(f"Searching by generic name: {search_field}")
        else:
            search_field = "patient.drug.medicinalproduct"
            logger.debug(f"Searching by brand name: {search_field}")
        
        # Build query parameters
        params = {
            "search": f'{search_field}:"{drug_name.upper()}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": min(limit, 1000)  # API max is 1000
        }
        
        result = await self.client._make_request(self.ENDPOINT, params)
        
        # Handle no results
        if not result.success:
            logger.warning(f"No adverse event data found for {drug_name}")
            return {
                "drug_name": drug_name,
                "side_effects": [],
                "total_reports": 0,
                "formatted": f"ℹ️ No adverse event data found for {drug_name}.",
                "success": False
            }
        
        # Process and format results
        side_effects = [
            {
                "term": item.get("term", "Unknown"),
                "count": item.get("count", 0)
            }
            for item in result.results
        ]
        
        # Calculate total (sum of all returned counts)
        total_reports = sum(se["count"] for se in side_effects)
        
        # Create human-readable format
        formatted = f"📊 **Top {len(side_effects)} Reported Side Effects for {drug_name.upper()}**:\n"
        for i, se in enumerate(side_effects, 1):
            formatted += f"{i}. {se['term']}: {se['count']:,} reports\n"
        
        logger.info(f"Found {len(side_effects)} side effects, {total_reports:,} total reports")
        
        return {
            "drug_name": drug_name,
            "side_effects": side_effects,
            "total_reports": total_reports,
            "formatted": formatted,
            "success": True
        }
    
    async def check_specific_reaction(
        self,
        drug_name: str,
        reaction: str
    ) -> Dict[str, Any]:
        """
        Check if a specific drug is linked to a specific reaction/symptom.
        
        Uses AND query to find reports where both the drug and reaction match.
        This is useful for answering questions like "Does Lisinopril cause cough?"
        
        Args:
            drug_name: The medication name (brand or generic)
            reaction: The symptom/reaction to check (e.g., "Cough", "Myalgia")
            
        Returns:
            Dictionary containing:
            - found (bool): True if drug-reaction link exists
            - count (int): Number of reports linking drug to reaction
            - formatted (str): Human-readable result string
            - success (bool): Whether the query succeeded
            
        Example:
            >>> result = await service.check_specific_reaction("Lisinopril", "Cough")
            >>> print(result["formatted"])
            ✅ YES. There are 45,230 reports linking Lisinopril to Cough.
            
        API Query Generated:
            /drug/event.json?search=patient.drug.medicinalproduct:"LISINOPRIL"
                             AND patient.reaction.reactionmeddrapt:"COUGH"
            &limit=1
            
        Note:
            A positive result does NOT prove causation. It only indicates
            that patients reported this reaction while taking the drug.
        """
        logger.info(f"Checking if {drug_name} causes {reaction}")
        
        # Construct AND query - both drug and reaction must match
        search_term = (
            f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND '
            f'patient.reaction.reactionmeddrapt:"{reaction.upper()}"'
        )
        
        params = {
            "search": search_term,
            "limit": 1
        }
        
        result = await self.client._make_request(self.ENDPOINT, params)
        
        if not result.success or result.meta.get("results", {}).get("total", 0) == 0:
            logger.info(f"No link found between {drug_name} and {reaction}")
            return {
                "drug_name": drug_name,
                "reaction": reaction,
                "found": False,
                "count": 0,
                "formatted": f"❌ No reports found linking {drug_name} to {reaction}.",
                "success": True
            }
        
        # Extract count from meta
        total_count = result.meta.get("results", {}).get("total", 0)
        
        logger.info(f"Found {total_count} reports linking {drug_name} to {reaction}")
        
        return {
            "drug_name": drug_name,
            "reaction": reaction,
            "found": total_count > 0,
            "count": total_count,
            "formatted": f"✅ **YES.** There are {total_count:,} reports linking {drug_name} to {reaction}.\n\n"
                         f"*Note: These are raw reports. Causation is not proven.*",
            "success": True
        }
    
    async def check_severity(
        self,
        drug_name: str
    ) -> Dict[str, Any]:
        """
        Check how many reports involved Death or Hospitalization.
        
        This provides a "red flag" check for potentially dangerous drugs,
        showing statistics on severe outcomes.
        
        Args:
            drug_name: The medication to check
            
        Returns:
            Dictionary containing:
            - has_severe_outcomes (bool): True if deaths/hospitalizations reported
            - death_count (int): Number of death reports
            - hospitalization_count (int): Number of hospitalization reports
            - severity_stats (SeverityStats): Full severity breakdown
            - formatted (str): Human-readable formatted string
            - success (bool): Whether the query succeeded
            
        Example (dangerous drug):
            >>> result = await service.check_severity("Warfarin")
            >>> print(result["formatted"])
            ⚠️ **WARNING: Severe Outcomes Reported for WARFARIN**
            Deaths: 3,240 reports
            Hospitalizations: 8,900 reports
            Life-threatening: 1,200 reports
            *(Note: These are raw reports. Causation is not proven.)*
            
        Example (safer drug):
            >>> result = await service.check_severity("Vitamin D")
            >>> print(result["formatted"])
            ✅ Vitamin D: No severe outcomes (Death/Hospitalization) found in FDA data.
            
        Router Trigger Keywords:
            - "dangerous"
            - "is [drug] safe"
            - "serious side effects"
            - "hospitalization"
        """
        logger.info(f"Checking severity for: {drug_name}")
        
        # Check for death reports
        search_death = f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND seriousnessdeath:1'
        params_death = {"search": search_death, "limit": 1}
        result_death = await self.client._make_request(self.ENDPOINT, params_death)
        death_count = result_death.meta.get("results", {}).get("total", 0) if result_death.success else 0
        
        # Check for hospitalization reports
        search_hosp = f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND seriousnesshospitalization:1'
        params_hosp = {"search": search_hosp, "limit": 1}
        result_hosp = await self.client._make_request(self.ENDPOINT, params_hosp)
        hosp_count = result_hosp.meta.get("results", {}).get("total", 0) if result_hosp.success else 0
        
        # Check for life-threatening
        search_lt = f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND seriousnesslifethreatening:1'
        params_lt = {"search": search_lt, "limit": 1}
        result_lt = await self.client._make_request(self.ENDPOINT, params_lt)
        lt_count = result_lt.meta.get("results", {}).get("total", 0) if result_lt.success else 0
        
        # Check for disability
        search_dis = f'patient.drug.medicinalproduct:"{drug_name.upper()}" AND seriousnessdisabling:1'
        params_dis = {"search": search_dis, "limit": 1}
        result_dis = await self.client._make_request(self.ENDPOINT, params_dis)
        dis_count = result_dis.meta.get("results", {}).get("total", 0) if result_dis.success else 0
        
        has_severe = (death_count > 0 or hosp_count > 0)
        
        if not has_severe:
            logger.info(f"No severe outcomes found for {drug_name}")
            formatted = f"✅ {drug_name.upper()}: No severe outcomes (Death/Hospitalization) found in FDA data."
        else:
            logger.warning(f"Severe outcomes found for {drug_name}: Deaths={death_count}, Hosp={hosp_count}")
            formatted = (
                f"⚠️ **WARNING: Severe Outcomes Reported for {drug_name.upper()}**\n"
                f"Deaths: {death_count:,} reports\n"
                f"Hospitalizations: {hosp_count:,} reports\n"
                f"Life-threatening: {lt_count:,} reports\n"
                f"Disability: {dis_count:,} reports\n\n"
                f"*Note: These are raw reports. Causation is not proven.*"
            )
        
        return {
            "drug_name": drug_name,
            "has_severe_outcomes": has_severe,
            "death_count": death_count,
            "hospitalization_count": hosp_count,
            "life_threatening_count": lt_count,
            "disability_count": dis_count,
            "formatted": formatted,
            "success": True
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_demographic_breakdown(
        self,
        drug_name: str
    ) -> Dict[str, Any]:
        """
        Get demographic breakdown of adverse event reports (age, sex distribution).
        
        Args:
            drug_name: The medication to check
            
        Returns:
            Dictionary containing demographic breakdowns by age and sex
        """
        logger.info(f"Getting demographics for: {drug_name}")
        
        # Count by sex
        search_term = f'patient.drug.medicinalproduct:"{drug_name.upper()}"'
        params = {
            "search": search_term,
            "count": "patient.patientsex",
            "limit": 10
        }
        
        result = await self.client._make_request(self.ENDPOINT, params)
        
        if not result.success:
            logger.warning(f"Could not get demographics for {drug_name}")
            return {
                "drug_name": drug_name,
                "success": False,
                "sex_breakdown": [],
                "formatted": f"No demographic data found for {drug_name}"
            }
        
        # Parse sex codes (1=Male, 2=Female, 0/other=Unknown)
        sex_mapping = {1: "Male", 2: "Female", 0: "Unknown"}
        sex_breakdown = []
        
        for item in result.results:
            sex_code = item.get("term")
            try:
                sex_code = int(sex_code)
                sex_name = sex_mapping.get(sex_code, "Unknown")
            except (ValueError, TypeError):
                sex_name = "Unknown"
            
            sex_breakdown.append({
                "sex": sex_name,
                "count": item.get("count", 0)
            })
        
        # Format output
        formatted = f"👥 **Demographics for {drug_name.upper()} Adverse Events**:\n"
        for sb in sex_breakdown:
            formatted += f"- {sb['sex']}: {sb['count']:,} reports\n"
        
        return {
            "drug_name": drug_name,
            "sex_breakdown": sex_breakdown,
            "formatted": formatted,
            "success": True
        }
    
    async def get_reports_by_outcome(
        self,
        drug_name: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get actual adverse event report details for a drug.
        
        Args:
            drug_name: The medication to check
            limit: Maximum number of reports to return
            
        Returns:
            List of adverse event reports with details
        """
        logger.info(f"Getting reports for: {drug_name}")
        
        search_term = f'patient.drug.medicinalproduct:"{drug_name.upper()}"'
        params = {
            "search": search_term,
            "limit": min(limit, 100)
        }
        
        result = await self.client._make_request(self.ENDPOINT, params)
        
        if not result.success:
            logger.warning(f"Could not get reports for {drug_name}")
            return {
                "drug_name": drug_name,
                "reports": [],
                "success": False
            }
        
        return {
            "drug_name": drug_name,
            "reports": result.results,
            "success": True
        }
    
    async def search_by_reaction(
        self,
        reaction: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search which drugs are most commonly linked to a specific reaction.
        
        Args:
            reaction: The symptom/reaction to search for
            limit: Number of drugs to return
            
        Returns:
            List of drugs ranked by frequency for this reaction
        """
        logger.info(f"Searching drugs causing: {reaction}")
        
        search_term = f'patient.reaction.reactionmeddrapt:"{reaction.upper()}"'
        params = {
            "search": search_term,
            "count": "patient.drug.medicinalproduct.exact",
            "limit": min(limit, 100)
        }
        
        result = await self.client._make_request(self.ENDPOINT, params)
        
        if not result.success:
            logger.warning(f"No drugs found for reaction: {reaction}")
            return {
                "reaction": reaction,
                "drugs": [],
                "success": False
            }
        
        drugs = [
            {
                "name": item.get("term", "Unknown"),
                "count": item.get("count", 0)
            }
            for item in result.results
        ]
        
        formatted = f"💊 **Drugs Most Commonly Linked to {reaction.upper()}**:\n"
        for i, drug in enumerate(drugs, 1):
            formatted += f"{i}. {drug['name']}: {drug['count']:,} reports\n"
        
        return {
            "reaction": reaction,
            "drugs": drugs,
            "formatted": formatted,
            "success": True
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def get_drug_side_effects(drug_name: str, limit: int = 10) -> str:
    """
    Quick function to get formatted side effects.
    
    Args:
        drug_name: The medication name
        limit: Number of side effects to return
        
    Returns:
        Formatted string with top side effects
    """
    service = DrugAdverseEventService()
    try:
        result = await service.get_top_side_effects(drug_name, limit)
        return result["formatted"]
    finally:
        await service.close()


async def check_drug_reaction(drug_name: str, reaction: str) -> str:
    """
    Quick function to check if a drug causes a specific reaction.
    
    Args:
        drug_name: The medication name
        reaction: The symptom to check
        
    Returns:
        Formatted string with verification result
    """
    service = DrugAdverseEventService()
    try:
        result = await service.check_specific_reaction(drug_name, reaction)
        return result["formatted"]
    finally:
        await service.close()


async def check_drug_severity(drug_name: str) -> str:
    """
    Quick function to get severity statistics.
    
    Args:
        drug_name: The medication name
        
    Returns:
        Formatted string with severity alert
    """
    service = DrugAdverseEventService()
    try:
        result = await service.check_severity(drug_name)
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
        print("DrugAdverseEventService - Interactive Test")
        print("=" * 60)
        
        service = DrugAdverseEventService()
        
        try:
            # Test 1: Top side effects
            print("\n📊 Test 1: Top Side Effects for Lipitor")
            print("-" * 40)
            result = await service.get_top_side_effects("Lipitor", limit=5)
            print(result["formatted"])
            
            # Test 2: Specific reaction check
            print("\n🔍 Test 2: Does Lisinopril cause Cough?")
            print("-" * 40)
            result = await service.check_specific_reaction("Lisinopril", "Cough")
            print(result["formatted"])
            
            # Test 3: Severity check
            print("\n⚠️ Test 3: Severity Check for Warfarin")
            print("-" * 40)
            result = await service.check_severity("Warfarin")
            print(result["formatted"])
            
            # Test 4: Demographics
            print("\n👥 Test 4: Demographics for Aspirin")
            print("-" * 40)
            result = await service.get_demographic_breakdown("Aspirin")
            print(result["formatted"])
            
        finally:
            await service.close()
        
        print("\n" + "=" * 60)
        print("Tests Complete!")
        print("=" * 60)

    asyncio.run(run_tests())
