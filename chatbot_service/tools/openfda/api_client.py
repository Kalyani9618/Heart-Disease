import logging
import httpx
import asyncio
from typing import Optional, Dict, Any

from tools.openfda.models import FDAResult

logger = logging.getLogger(__name__)


class OpenFDAClient:
    """
    Client for interacting with the OpenFDA API.
    Handles rate limiting, authentication, and error parsing.
    """
    BASE_URL = "https://api.fda.gov"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=10.0)
        
    async def close(self):
        """Close the async client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> FDAResult:
        """
        Execute API request with error handling and rate limit backoff.
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            # Log the query for debugging (masking API key if present)
            debug_params = params.copy()
            if "api_key" in debug_params:
                debug_params["api_key"] = "***"
            logger.debug(f"Requesting {url} with params {debug_params}")

            response = await self.client.get(url, params=params)
            
            # Handle Rate Limiting
            if response.status_code == 429:
                logger.warning("OpenFDA rate limit exceeded. Waiting 2 seconds...")
                await asyncio.sleep(2)
                return await self._make_request(endpoint, params)
                
            # Handle Not Found (common for specific queries)
            if response.status_code == 404:
                return FDAResult(success=False, error="No results found")
            
            # Handle other errors
            response.raise_for_status()
            
            data = response.json()
            return FDAResult(
                meta=data.get("meta", {}),
                results=data.get("results", []),
                success=True
            )
            
        except httpx.RequestError as e:
            logger.error(f"OpenFDA API connection error: {str(e)}")
            return FDAResult(success=False, error=f"Connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error querying OpenFDA: {str(e)}")
            return FDAResult(success=False, error=f"Unexpected error: {str(e)}")

    async def search_drug_enforcement(self, query: str, limit: int = 10, sort: str = None) -> FDAResult:
        """Search drug recall/enforcement endpoint"""
        params = {"search": query, "limit": limit}
        if sort:
            params["sort"] = sort
        return await self._make_request("/drug/enforcement.json", params)

    async def search_drug_label(self, query: str, limit: int = 10) -> FDAResult:
        """Search drug product labeling endpoint"""
        params = {"search": query, "limit": limit}
        return await self._make_request("/drug/label.json", params)
        
    async def search_food_enforcement(self, query: str, limit: int = 10, sort: str = None) -> FDAResult:
        """Search food recall/enforcement endpoint"""
        params = {"search": query, "limit": limit}
        if sort:
            params["sort"] = sort
        return await self._make_request("/food/enforcement.json", params)

    async def search_drug_events(self, query: str, limit: int = 5) -> FDAResult:
        """Search drug adverse events (FAERS)"""
        params = {"search": query, "limit": limit}
        return await self._make_request("/drug/event.json", params)

    async def search_food_events(self, query: str, limit: int = 5) -> FDAResult:
        """Search food/supplement adverse events (CAERS)"""
        params = {"search": query, "limit": limit}
        return await self._make_request("/food/event.json", params)
