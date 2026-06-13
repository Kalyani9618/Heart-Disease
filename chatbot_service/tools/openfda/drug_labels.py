import logging
import asyncio
from typing import List, Dict, Any, Optional
from tools.openfda.api_client import OpenFDAClient

logger = logging.getLogger(__name__)


class DrugLabelQuerier:
    """
    Query drug product labeling (package inserts) for clinical information.
    Handles interactions, warnings, side effects, and contraindications.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenFDAClient(api_key=api_key)

    async def close(self):
        """Close the underlying client."""
        await self.client.close()
        
    async def _search_section(self, drug_name: str, section_field: str, limit: int = 5) -> List[str]:
        """Helper to search a specific section of the drug label"""
        # Construct query: (Brand OR Generic) AND Section Exists
        query = f'(openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}") AND _exists_:{section_field}'
        
        result = await self.client.search_drug_label(query, limit=limit)
        
        if not result.success:
            logger.warning(f"No {section_field} found for {drug_name}")
            return []
            
        extracted_data = []
        for item in result.results:
            # Some fields are lists, some might be strings (though usually lists in OpenFDA)
            content = item.get(section_field)
            if isinstance(content, list):
                extracted_data.extend(content)
            elif isinstance(content, str):
                extracted_data.append(content)
                
        return extracted_data

    async def find_drug_interactions(self, drug_name: str, limit: int = 5) -> List[str]:
        """
        Find drug interaction warnings.
        Returns raw text from the 'drug_interactions' section of the label.
        """
        logger.info(f"Searching drug interactions for {drug_name}")
        return await self._search_section(drug_name, "drug_interactions", limit)

    async def find_warnings(self, drug_name: str, limit: int = 5) -> List[str]:
        """
        Find boxed warnings and general warnings.
        """
        logger.info(f"Searching warnings for {drug_name}")
        boxed = await self._search_section(drug_name, "boxed_warning", limit)
        general = await self._search_section(drug_name, "warnings", limit)
        return boxed + general

    async def find_adverse_reactions(self, drug_name: str, limit: int = 5) -> List[str]:
        """
        Find listed adverse reactions (side effects).
        """
        logger.info(f"Searching adverse reactions for {drug_name}")
        return await self._search_section(drug_name, "adverse_reactions", limit)
    
    async def find_contraindications(self, drug_name: str, limit: int = 5) -> List[str]:
        """
        Find contraindications (when NOT to use the drug).
        """
        logger.info(f"Searching contraindications for {drug_name}")
        return await self._search_section(drug_name, "contraindications", limit)
